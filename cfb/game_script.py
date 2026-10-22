"""CFB-only game-script Monte Carlo research challenger.

The existing CFB baseline remains the primary projection. This candidate
adds *historically informed dispersion*, sequential turnover/field-position
interactions, and an explicitly hypothetical clock/pace response to large leads.
Numbers governing the response are UNFITTED hypotheses; no ATS calibration.

Only previously completed-game records may be supplied. Caller is responsible
for the existing 24h embargo and pre-kickoff source-capture checks. Never pass
final-score, roster or postgame information from the target fixture.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from cfb.engine import CFB_RULES, _resolve_overtime, validate_projection

VERSION = 'cfb-game-script-research-v1'
# Hypotheses, explicitly not empirical fitted coefficients.
PACE_PRIOR_SD = 0.9
PACE_VARIANCE_PRIOR_GAMES = 8.0
LATENT_EFF_PRIOR_SD = 0.16
LATENT_EFF_MAX_SD = 0.55
EFF_VARIANCE_PRIOR_GAMES = 12.0
TURNOVER_CONCENTRATION_PRIOR = 65.0
MIN_TURNOVER_CONCENTRATION = 12.0
MAX_TURNOVER_CONCENTRATION = 200.0
TURNOVER_VARIANCE_PRIOR_GAMES = 12.0
LARGE_LEAD_HALFTIME_POINTS = 17
# Hypothesis: big halftime margin can burn about one possession for each side.
RUNAWAY_CLOCK_REDUCTION_PROB = 0.35
MAX_SHORTFIELD_BONUS = 1.10


def _positive_records(rows):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('Nonempty historical team game records required')
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid historical game record')
        for field in ('drives', 'faced', 'points', 'conceded'):
            v = row.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError('Missing or invalid '+field)
            if (field in ('drives', 'faced') and not 1 <= v <= 40) or (field in ('points', 'conceded') and not 0 <= v <= 150):
                raise ValueError('Out-of-bounds historical '+field)
        for field in ('interceptions', 'fumbles_lost'):
            v = row.get(field)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 15):
                raise ValueError('Invalid historical '+field)
    return rows


def _spread(values, prior_sd, prior_games, cap):
    vals = np.array(values, dtype=float)
    if vals.size == 0 or not np.isfinite(vals).all():
        raise ValueError('No finite game records')
    # Shrink noisy, small-sample observed dispersion towards league hypothesis.
    empirical = float(np.var(vals, ddof=1)) if len(vals) > 1 else prior_sd**2
    post = ((len(vals)-1) * empirical + prior_games * prior_sd**2) / (max(0, len(vals)-1) + prior_games)
    return float(np.clip(math.sqrt(post), 0.2, cap))


def _turnover_concentration(rows):
    known = [r for r in rows if r.get('interceptions') is not None and r.get('fumbles_lost') is not None]
    if len(known) < 3:
        return TURNOVER_CONCENTRATION_PRIOR, len(known)
    rates = np.array([(r['interceptions'] + r['fumbles_lost'])/r['drives'] for r in known])
    p = float(np.clip(rates.mean(), .02, .4))
    v = float(rates.var(ddof=1))
    # Total variation includes binomial play-count noise; don't mistake it for
    # truly persistent, latent turnover-rate variation.
    sampling = float(np.mean(p*(1-p)/np.array([r['drives'] for r in known])))
    latent = max(0., v-sampling)
    raw = float(np.clip(p*(1-p)/latent-1, MIN_TURNOVER_CONCENTRATION, MAX_TURNOVER_CONCENTRATION)) if latent else MAX_TURNOVER_CONCENTRATION
    weight = len(known)/(len(known)+TURNOVER_VARIANCE_PRIOR_GAMES)
    return float(np.clip((1-weight)*TURNOVER_CONCENTRATION_PRIOR+weight*raw,
                         MIN_TURNOVER_CONCENTRATION,MAX_TURNOVER_CONCENTRATION)),len(known)


def profile(home_history, away_history, home_projection, away_projection):
    """Pre-kickoff prior-games-only dispersion measurements, never fitted means."""
    h = _positive_records(home_history)
    a = _positive_records(away_history)
    hp, ap = validate_projection(home_projection, 'home'), validate_projection(away_projection,'away')
    pair_tempo = [float(x['drives']) for x in h] + [float(x['faced']) for x in a] + [float(x['drives']) for x in a] + [float(x['faced']) for x in h]
    sd = _spread(pair_tempo, PACE_PRIOR_SD, PACE_VARIANCE_PRIOR_GAMES, 2.5)
    def scoring(rows, defense):
        x=np.array([(r['conceded'] if defense else r['points'])/(r['faced'] if defense else r['drives']) for r in rows],float)
        return x
    scored_h=np.concatenate((scoring(h,False),scoring(a,True)))
    scored_a=np.concatenate((scoring(a,False),scoring(h,True)))
    def latent(x):
        # Game-to-game PPD variation includes different opponents and binomial
        # scoring. Conservative cap + shrink prevent overconfident high tails.
        spread=_spread(x,LATENT_EFF_PRIOR_SD,EFF_VARIANCE_PRIOR_GAMES,1.8)
        return float(np.clip(spread*len(x)/(len(x)+EFF_VARIANCE_PRIOR_GAMES), .05, LATENT_EFF_MAX_SD))
    ch,nh=_turnover_concentration(h)
    ca,na=_turnover_concentration(a)
    return {'version':VERSION,'games_home':len(h),'games_away':len(a),
            'shared_drives_sd':round(sd,5),
            'home_game_efficiency_ppd_sd':round(latent(scored_h),5),
            'away_game_efficiency_ppd_sd':round(latent(scored_a),5),
            'home_turnover_beta_concentration':round(ch,4),
            'away_turnover_beta_concentration':round(ca,4),
            'observed_turnover_games_home':nh,'observed_turnover_games_away':na,
            'expected_drives_home':hp['expected_drives'],
            'expected_drives_away':ap['expected_drives'],
            'distribution_coefficients_fitted':False,
            'assumptions':['Historical drive counts include imperfect PBP drive reconstruction',
                'Beta per-game turnover propensities are a sensitivity model, not measured persistence',
                'Game efficiency SD mixes opponent quality and within-game variation',
                '17-point halftime lead / 0.35 clock response is an explicit uncalibrated assumption']}


def _regulation_draw(p, rng, propensity, extra_ppd, short_field, active):
    """Turnover excludes scoring, then TD/FG rates match adjusted PPD where feasible."""
    if not np.any(active):
        return np.zeros(len(active),dtype=np.int16),np.zeros(len(active),dtype=bool)
    avg_scoring = p['_td']*6.965+(1-p['_td'])*3.0
    rate=np.clip(p['_ppd']+extra_ppd+short_field*MAX_SHORTFIELD_BONUS, .05,6.15)
    # A turnover and an offensive score cannot occur on the same possession.
    pscore=np.minimum(rate/avg_scoring, 1.0-propensity)
    draw=rng.random(len(active))
    to=active & (draw<propensity)
    score=active & (draw>=propensity) & (draw<propensity+pscore)
    td=score & (rng.random(len(active))<p['_td'])
    xp=(rng.random(len(active))<.965).astype(np.int16)
    points=np.where(td,6+xp,np.where(score,3,0)).astype(np.int16)
    return points,to


def _scenario_summary(mask,margin,home_score,away_score):
    count=int(mask.sum())
    return {'simulations':count,'share':round(float(mask.mean()),5),
            'mean_home_margin':round(float(margin[mask].mean()),3) if count else None,
            'mean_home_score':round(float(home_score[mask].mean()),3) if count else None,
            'mean_away_score':round(float(away_score[mask].mean()),3) if count else None}


def simulate_script(home_projection, away_projection, home_history, away_history,
                    *, runs=3000,seed=1,spread_home=None,total=None,allow_overtime=True,
                    clock_response=True, tempo_shift=0.0,
                    turnover_shift_home=0.0,turnover_shift_away=0.0,
                    efficiency_shift_home=0.0,efficiency_shift_away=0.0,
                    mechanisms=None):
    """Vectorized sequential possessions; outcomes remain a research hypothesis.

    Turnover-induced short field applies to the *next actual opposing series*;
    shared tempo/conditions and team-specific efficiency draws produce fat tails.
    The match's own final score never enters this function.
    """
    if type(runs) is not int or not 100 <= runs <= 300000:raise ValueError('Invalid run count')
    if type(seed) is not int or not 0 <= seed < 2**32:raise ValueError('Invalid 32-bit seed')
    if type(clock_response) is not bool:raise ValueError('Invalid clock response')
    for name,value,bound in (('tempo_shift',tempo_shift,3.),
        ('turnover_shift_home',turnover_shift_home,.15),('turnover_shift_away',turnover_shift_away,.15),
        ('efficiency_shift_home',efficiency_shift_home,.9),('efficiency_shift_away',efficiency_shift_away,.9)):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value)>bound:
            raise ValueError('Invalid scenario '+name)
    h,a=validate_projection(home_projection,'home'),validate_projection(away_projection,'away')
    if spread_home is not None:
        if isinstance(spread_home,bool) or not isinstance(spread_home,(int,float)) or not math.isfinite(spread_home) or abs(spread_home)>100:raise ValueError('Invalid spread')
    if total is not None:
        if isinstance(total,bool) or not isinstance(total,(int,float)) or not math.isfinite(total) or not 0<=total<=200:raise ValueError('Invalid total')
    if mechanisms is not None:
        if not isinstance(mechanisms,dict) or not all(k in mechanisms for k in ('home','away','turnover_field')):
            raise ValueError('Invalid Build16 mechanism evidence')
    evidence=profile(home_history,away_history,home_projection,away_projection)
    rng=np.random.default_rng(seed)
    n=int(runs)
    tempo=float(np.clip((h['expected_drives']+a['expected_drives'])/2+tempo_shift,6,19))
    draws=np.clip(np.rint(rng.normal(tempo,evidence['shared_drives_sd'],n)).astype(int),6,19)
    # Paired series; the extra-possession discrepancy from end-of-half/kicks
    # is not modelled; count claims are explicitly descriptive.
    latent_h=rng.normal(0,evidence['home_game_efficiency_ppd_sd'],n)
    latent_a=rng.normal(0,evidence['away_game_efficiency_ppd_sd'],n)
    global_env=rng.normal(0, .12, n)  # Unfitted shared weather / scoring environment.
    def beta_prop(p,c):
        mu=float(np.clip(p['_to']+ (turnover_shift_home if p is h else turnover_shift_away),.003,.47))
        return rng.beta(mu*c,(1-mu)*c,n)
    to_h=beta_prop(h,evidence['home_turnover_beta_concentration'])
    to_a=beta_prop(a,evidence['away_turnover_beta_concentration'])
    hs=np.zeros(n,dtype=np.int32); ass=np.zeros(n,dtype=np.int32)
    turnovers_h=np.zeros(n,dtype=np.int16);turnovers_a=np.zeros(n,dtype=np.int16)
    short_h=np.zeros(n,dtype=float);short_a=np.zeros(n,dtype=float)
    def state_propensity(side,original,delta):
        """Observed game-state passing shifts affect only interception exposure.

        No extra pressure impact, no invented penalty EPA, no double-counted
        turnover point deduction. The baseline beta propensity is retained.
        """
        if mechanisms is None:return original
        z=mechanisms[side]
        by=(z.get('pass_states') or {}).get('by_state')
        ip=z.get('int_rate');fl=z.get('fumble_rate');plays=z.get('plays_per_drive')
        neutral=z.get('neutral_pass_rate')
        if not isinstance(by,dict) or any(not isinstance(x,(float,int)) or not np.isfinite(x) for x in (ip,fl,plays,neutral)):
            return original
        if not 0<=ip<=.2 or not 0<=fl<=.2 or not 2<=plays<=15 or not .01<=neutral<=.99:
            return original
        desired=np.where(delta<=-9,by['trailing_9_plus'],
                          np.where(delta>=9,by['leading_9_plus'],by['within_8']))
        current=1-np.exp(-plays*(desired*ip+fl))
        reference=1-np.exp(-plays*(neutral*ip+fl))
        return np.clip(original + np.clip(current-reference,-.08,.08),.003,.47)
    def trailing_rush_resilience_ppd(side,score_diff):
        """Optional, pregame-frozen severe-mismatch response in trailing scripts.

        This shifts offensive scoring opportunity ONLY on possessions where
        that team is down >=9. Absence of research evidence means zero shift;
        original Build16 calls are numerically identical.
        """
        if mechanisms is None:return np.zeros(n,dtype=float)
        x=mechanisms[side].get('trailing_rush_resilience_ppd')
        if x is None:return np.zeros(n,dtype=float)
        if not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or abs(x)>.25:
            raise ValueError('Invalid unfitted trailing-rush sensitivity')
        return np.where(score_diff<=-9,float(x),0.)

    def next_field_bonus(turnovers):
        if mechanisms is None:return turnovers.astype(float)
        data=mechanisms['turnover_field']
        if data.get('status')!='OBSERVED':return turnovers.astype(float)
        starts=data.get('following_start_by_turnover_zone',{})
        zones=data.get('turnover_zone_counts',{})
        near=starts.get('opponent_40_or_closer',[])
        far=starts.get('outside_opponent_40',[])
        all_starts=near+far
        if len(all_starts)<4:return turnovers.astype(float)
        chance=(zones.get('opponent_40_or_closer',0)+2)/max(1,sum(zones.values())+4)
        isnear=rng.random(n)<chance
        result=np.zeros(n,dtype=float)
        # Sample observed follow-up possession positions, not turnover spots.
        for mask,choices in ((turnovers&isnear,near or all_starts),
                             (turnovers&~isnear,far or all_starts)):
            k=int(mask.sum())
            if k:
                selected=np.array(choices,float)[rng.integers(0,len(choices),size=k)]
                result[mask]=np.clip((selected-29.0)*.018,-.55,1.1)/MAX_SHORTFIELD_BONUS
        return result

    latent_h += float(efficiency_shift_home)
    latent_a += float(efficiency_shift_away)
    first_home=(rng.random(n)<.5)
    burn=np.zeros(n,dtype=bool)
    # Draw all planned regulation possessions with an optional late pace response.
    for i in range(19):
        if clock_response:
            halfway=(i==draws//2)
            burn=burn | (halfway&(np.abs(hs-ass)>=LARGE_LEAD_HALFTIME_POINTS)&(rng.random(n)<RUNAWAY_CLOCK_REDUCTION_PROB))
        active=(i<draws-burn.astype(int))
        # Within each series, possession order matters for turnovers and
        # field position; players are assigned per-game random opening side.
        for hfirst in (True,False):
            mask=active & (first_home==hfirst)
            if not np.any(mask):continue
            for side in (('home','away') if hfirst else ('away','home')):
                if side=='home':
                    prop=state_propensity('home',to_h,hs-ass)
                    pts,to=_regulation_draw(h,rng,prop,latent_h+global_env+
                        trailing_rush_resilience_ppd('home',hs-ass),short_h,mask)
                    hs+=pts;turnovers_h+=to.astype(np.int16)
                    short_h=np.where(mask,0.,short_h)
                    short_a=np.where(to,next_field_bonus(to),short_a)
                else:
                    prop=state_propensity('away',to_a,ass-hs)
                    pts,to=_regulation_draw(a,rng,prop,latent_a+global_env+
                        trailing_rush_resilience_ppd('away',ass-hs),short_a,mask)
                    ass+=pts;turnovers_a+=to.astype(np.int16)
                    short_a=np.where(mask,0.,short_a)
                    short_h=np.where(to,next_field_bonus(to),short_h)
    regulation_margin=hs.astype(float)-ass
    reg_ties=int(np.count_nonzero(hs==ass))
    if allow_overtime:
        ot=_resolve_overtime(h,a,dict(CFB_RULES),rng,hs,ass)
    else:
        ot={'regulation_ties':reg_ties,'ot1':0,'ot2':0,'ot3_or_later':0,'unresolved_after_limit':reg_ties}
    margin=hs.astype(float)-ass
    scores=hs+ass
    slow=draws<=max(6,round(tempo)-2)
    fast=draws>=min(19,round(tempo)+2)
    hto=turnovers_h-turnovers_a>=2
    ato=turnovers_a-turnovers_h>=2
    h_stall=(hs<=0.60*h['_ppd']*draws)
    a_stall=(ass<=0.60*a['_ppd']*draws)
    scripts={'slow_possessions':_scenario_summary(slow,margin,hs,ass),
       'fast_possessions':_scenario_summary(fast,margin,hs,ass),
       'home_two_more_turnovers':_scenario_summary(hto,margin,hs,ass),
       'away_two_more_turnovers':_scenario_summary(ato,margin,hs,ass),
       'home_scoring_stall':_scenario_summary(h_stall,margin,hs,ass),
       'away_scoring_stall':_scenario_summary(a_stall,margin,hs,ass),
       'runaway_clock_sensitivity_triggered':_scenario_summary(burn,margin,hs,ass)}
    return {'version':VERSION,'status':'UNFITTED_GAME_SCRIPT_CHALLENGER_NOT_PRIMARY',
       'mean_score_home':round(float(hs.mean()),3),
       'mean_score_away':round(float(ass.mean()),3),
       'mean_margin_home':round(float(margin.mean()),3),
       'median_margin_home':float(np.median(margin)),
       'margin_standard_deviation':round(float(np.std(margin)),3),
       'margin_p10_p50_p90':[float(v) for v in np.quantile(margin,[.1,.5,.9])],
       'home_win_probability_raw':round(float(np.mean(margin>0)),5),
       'mean_total':round(float(scores.mean()),3),
       'simulated_turnovers_home':round(float(turnovers_h.mean()),3),
       'simulated_turnovers_away':round(float(turnovers_a.mean()),3),
       'simulated_regulation_drives_per_side':round(float((draws-burn.astype(int)).mean()),3),
       'home_cover_probability_raw':round(float(np.mean(margin+spread_home>0)),5) if spread_home is not None else None,
       'away_cover_probability_raw':round(float(np.mean(margin+spread_home<0)),5) if spread_home is not None else None,
       'over_probability_raw':round(float(np.mean(scores>total)),5) if total is not None else None,
       'under_probability_raw':round(float(np.mean(scores<total)),5) if total is not None else None,
       'overtime_summary':ot,
       'scenario_distributions':scripts,
       'historical_variance_evidence':evidence,
       'simulations':runs,'seed':seed,'clock_response_enabled':clock_response,
       'build16_mechanisms_enabled':mechanisms is not None,
       'not_auto_deployed':True,'ATS_calibrated':False,
       'stress_input':{'tempo_shift':tempo_shift,'turnover_shift_home':turnover_shift_home,
            'turnover_shift_away':turnover_shift_away,'efficiency_shift_home':efficiency_shift_home,
            'efficiency_shift_away':efficiency_shift_away},
       'limitations':['Unfitted dispersion / turnover / clock-feedback assumptions; NOT a trained prediction',
            'Scores depend on opponent-agnostic historical volatility; only baseline means are matched',
            'PBP possession counts may omit special-teams events and partial drives',
            'Two teams receive paired regulation drive counts: incomplete clock/possession accounting',
            'Final team scores include possible non-offensive points; drive PPD reconstruction is imperfect',
            'Scoring overdispersion includes opponent-strength noise and may widen tails too much',
            'Observed final and final lineup from target game are never inputs',
            'Probabilities are raw and uncalibrated: no standalone ATS recommendation']}


def adverse_scripts(home_projection, away_projection, home_history, away_history, *, runs=600,seed=1):
    """Same-seed matched counterfactuals, not probabilities assigned to events.

    Each shock is specified in advance, not chosen after observing a result.
    These stress runs cannot be mistaken for a calibrated primary forecast.
    """
    if type(runs) is not int or not 100<=runs<=300000:
        raise ValueError('Invalid stress run count')
    configurations={
        'reference':{},
        'slower_game':{'tempo_shift':-1.5},
        'faster_game':{'tempo_shift':1.5},
        'home_turnover_adversity':{'turnover_shift_home':.07},
        'away_turnover_adversity':{'turnover_shift_away':.07},
        'home_scoring_adversity':{'efficiency_shift_home':-.35},
        'away_scoring_adversity':{'efficiency_shift_away':-.35},
        'turnover_probability_reversion':{'turnover_shift_home':max(-.15,min(.15,CFB_RULES['league_to_drive']-float(home_projection['_to']))),
                                          'turnover_shift_away':max(-.15,min(.15,CFB_RULES['league_to_drive']-float(away_projection['_to'])))},
    }
    results={}
    for label,kwargs in configurations.items():
        r=simulate_script(home_projection,away_projection,home_history,away_history,
                           runs=runs,seed=seed,**kwargs)
        results[label]={'mean_home_margin':r['mean_margin_home'],
                        'mean_home_score':r['mean_score_home'],
                        'mean_away_score':r['mean_score_away'],
                        'margin_p10_p90':[r['margin_p10_p50_p90'][0],r['margin_p10_p50_p90'][2]],
                        'counterfactual_changes':kwargs}
    return {'version':VERSION,'status':'UNFITTED_ADVERSE_SCRIPTS_RESEARCH_ONLY',
            'scenarios':results,'simulations_per_scenario':runs,
            'non_probabilistic_counterfactuals':True,'not_auto_deployed':True}
