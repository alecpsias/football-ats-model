"""Build 15: CFB-native integrated matchup-to-points scoring RESEARCH engine.

Distinct from the NFL model, and never the primary forecast. Constructs each
team's expected PPD from (1) league reference PPD, (2) leave-opponent-out,
shrunken offensive/defensive passing and rushing EPA, (3) observed situational
matchups with conservative residuals, (4) matched turnover probabilities, and
(5) CFB drives, home field and NCAA scoring rules.

EPA is a competing estimate, not an additional full correction on top of the
current baseline's already-realized scoring rates: blended *levels* avoid
counting existing offensive/defensive drive results twice. All constants below
are hypotheses, not fitted or validated coefficients. Full pressure and
special-teams/penalty EPA remain missing and NEVER replaced with zero-valued
'observations'. Target-game outcomes, odds, and guessed injuries are unused.
"""
from __future__ import annotations

import math
from types import MappingProxyType

from cfb.engine import CFB_RULES, simulate
from cfb.matchups import _adjusted_attack

VERSION='cfb-integrated-scoring-research-v1'
RULES=MappingProxyType({
    'epa_prior_plays':450.0,
    'epa_level_prior_plays':300.0,
    'league_plays_per_drive':6.1,
    'third_down_prior':.405,
    'red_zone_td_prior':.59,
    'interception_prior':.028,
    'lost_fumble_prior':.007,
    'event_prior_plays':120.0,
    'red_zone_prior_trips':35.0,
    'third_down_residual_ppd':.65,
    'red_zone_residual_ppd':.26,
    'field_position_ppd_per_yard':.018,
    'qb_rush_residual_points_per_rush':.20,
    'sack_residual_ppd':-.25,
    'success_pass_prior':.44,
    'success_rush_prior':.45,
    'explosive_pass_prior':.105,
    'explosive_rush_prior':.12,
    'success_interaction_ppd_scale':.35,
    'explosive_interaction_ppd_scale':.4,
})


def _finite(x):
    return type(x) in (int,float) and math.isfinite(x)


def _clip(x,a,b):return max(a,min(b,x))


def _count_rate(rows,num,den):
    total_n=total_d=0.
    for r in rows:
        n=r.get(num);d=r.get(den)
        if n is None or d is None:continue
        if not _finite(n) or not _finite(d) or d<0 or n<0 or n>d:
            raise ValueError('Malformed observed event count: '+num)
        total_n+=n;total_d+=d
    return (total_n,total_d) if total_d>0 else (None,0.)


def _matched_event(attack,defense,off_n,off_d,def_n,def_d,prior,pseudocount):
    o,on=_count_rate(attack,off_n,off_d)
    d,dn=_count_rate(defense,def_n,def_d)
    if o is None or d is None:return None
    return .5*((o+prior*pseudocount)/(on+pseudocount)+
               (d+prior*pseudocount)/(dn+pseudocount))


def _interaction_residual(off,defense,off_n,off_d,def_n,def_d,prior,scale):
    """Tiny nonlinear offense-defense interaction *beyond* EPA main effects.

    Matched posterior odds are compared to a shrunk arithmetic baseline. The
    residual is bounded: the bulk of observed success/explosive impact is
    already captured by pass/rush EPA, and MUST NOT be counted twice.
    """
    o,on=_count_rate(off,off_n,off_d)
    d,dn=_count_rate(defense,def_n,def_d)
    if o is None or d is None:return None
    k=RULES['event_prior_plays']
    a=_clip((o+k*prior)/(on+k),.01,.99)
    b=_clip((d+k*prior)/(dn+k),.01,.99)
    base=(a+b)/2
    logit=lambda x:math.log(x/(1-x))
    raw=logit(prior)+(logit(a)-logit(prior)+logit(b)-logit(prior))/2
    matched=1/(1+math.exp(-raw))
    return float(_clip((matched-base)*scale,-.02,.02)),matched


def _mean_optional(rows,key):
    z=[r[key] for r in rows if _finite(r.get(key))]
    return float(sum(z)/len(z)) if z else None


def _weight_and_match(team_id,opponent_id,histories,kind):
    attack=histories.get(team_id,());defense=histories.get(opponent_id,())
    matched=_adjusted_attack(team_id,opponent_id,histories,kind)
    if matched is None:return None
    # An SOS estimate supported by only one opponent must not inherit the
    # sample size of ALL team games. Count only plays with third-party
    # opponent-quality observations; the game being adjusted is excluded.
    def covered_play_samples(rows,observed,other,quality_key,quality_count):
        count=0.
        for r in rows:
            enemy=r.get('opponent_id')
            if enemy is None:continue
            peers=[q for q in histories.get(enemy,()) if q.get('opponent_id')!=other]
            if (any(_finite(q.get(quality_key)) and _finite(q.get(quality_count))
                    and q[quality_count]>0 for q in peers)):
                n=r.get(observed)
                if not _finite(n) or n<0:raise ValueError('Invalid observed EPA count')
                count+=n
        return count
    plays=covered_play_samples(attack,'off_'+kind+'_n',team_id,
                               'def_'+kind+'_sum','def_'+kind+'_n')
    faced=covered_play_samples(defense,'def_'+kind+'_n',opponent_id,
                               'off_'+kind+'_sum','off_'+kind+'_n')
    if min(plays,faced)<=0:return None
    reliable=min(plays,faced)
    # Two independent reliability controls. One limits the EPA estimate's
    # deviation from the league prior; the other controls the EPA-vs-drive blend.
    shrunk=matched*reliable/(reliable+RULES['epa_prior_plays'])
    level_weight=reliable/(reliable+RULES['epa_level_prior_plays'])
    return {'kind':kind,'matched_epa_raw':round(float(matched),6),
            'matched_epa_shrunk':round(float(shrunk),6),
            'plays_effective':round(float(reliable),2),
            'epa_level_weight':round(float(level_weight),6)}


def project_integrated(team_id,opponent_id,histories,neutral,side):
    """Return a transparent, separately simulated CFB scoring projection.

    No synthetic PBP fields or NFL imports. If verified leave-opponent-out pass
    *and* rush evidence is absent, declare UNAVAILABLE instead of scoring it.
    Context effects are optional and absent signals are logged in missing list.
    """
    if side not in ('home','away') or type(neutral) is not bool:
        raise ValueError('Invalid side or neutral-site value')
    if team_id==opponent_id or team_id not in histories or opponent_id not in histories:
        raise ValueError('Two distinct teams with histories are required')
    team=histories[team_id]; opponent=histories[opponent_id]
    if not team or not opponent:raise ValueError('Empty team history')
    # Old baseline used strictly as a reliability anchor, not modified in place.
    from cfb.research_backtest import project
    base=project(team,opponent,neutral,side,False)
    passing=_weight_and_match(team_id,opponent_id,histories,'pass')
    rushing=_weight_and_match(team_id,opponent_id,histories,'rush')
    if passing is None or rushing is None:
        return {'status':'UNAVAILABLE_MISSING_LEAVE_OPPONENT_OUT_EPA',
                'version':VERSION,'projection':None,'missing_features':['opponent_quality_pass_or_rush'],
                'primary_forecast_modified':False,'coefficients_empirically_fitted':False}
    pcount=sum(r['off_pass_n'] for r in team);rcount=sum(r['off_rush_n'] for r in team)
    share=float(pcount/(pcount+rcount)) if (pcount+rcount)>0 else None
    if share is None:raise ValueError('Missing observed offensive play mix')
    share=_clip(share,.1,.9)
    plays_per_drive=float(_clip((pcount+rcount)/sum(r['drives'] for r in team),3.5,10.5))
    epa=(share*passing['matched_epa_shrunk'] +
         (1-share)*rushing['matched_epa_shrunk'])
    # Integrated EPA model expresses a FULL PPD LEVEL, then blends with a
    # separate prior PPD LEVEL. This is NOT base_ppd + duplicated EPA correction.
    epa_level=float(_clip(CFB_RULES['league_ppd']+plays_per_drive*epa,.4,5.8))
    weight=share*passing['epa_level_weight']+(1-share)*rushing['epa_level_weight']
    # HFA enters both candidate and baseline on a consistent scale: remove
    # baseline HFA before blending two efficiency levels, then add it ONCE.
    hfa=(0. if neutral else (1. if side=='home' else -1.)*
         CFB_RULES['home_advantage_points']/2/base['expected_drives'])
    adjusted=float((1-weight)*(base['_ppd']-hfa)+weight*epa_level+hfa)
    missing=[]; components={'baseline_ppd_anchor':round(base['_ppd'],6),
        'home_field_ppd':round(hfa,6),
        'epa_ppd_level':round(epa_level,6),
        'epa_blend_weight':round(weight,6),
        'pass_share_observed':round(share,6),
        'plays_per_drive_observed':round(plays_per_drive,5),
        'matched_passing':passing,'matched_rushing':rushing}

    third=_matched_event(team,opponent,'third_conversions_sum','third_conversions_n',
                         'def_third_conversions_sum','def_third_conversions_n',
                         RULES['third_down_prior'],RULES['event_prior_plays']/4)
    if third is None:missing.append('third_down_matched')
    else:
        # SMALL bounded deviation from pre-existing drive-level efficiency.
        term=float(_clip((third-RULES['third_down_prior'])*RULES['third_down_residual_ppd'],-.08,.08))
        adjusted+=term;components['third_down_residual_ppd']=round(term,6)
        components['third_down_matched']=round(third,6)

    rz=_matched_event(team,opponent,'redzone_tds_sum','redzone_tds_n',
                      'def_redzone_tds_sum','def_redzone_tds_n',
                      RULES['red_zone_td_prior'],RULES['red_zone_prior_trips'])
    if rz is None:missing.append('red_zone_matched')
    else:
        term=float(_clip((rz-RULES['red_zone_td_prior'])*RULES['red_zone_residual_ppd'],-.055,.055))
        adjusted+=term;components['red_zone_residual_ppd']=round(term,6)
        components['red_zone_td_matched']=round(rz,6)

    # EPA already contains most success/explosive effects. Only the
    # nonlinear offense-versus-defense interaction, relative to the original
    # shrunken means, is added; each residual is bounded to 0.02 PPD.
    for kind,prior in (('pass',RULES['success_pass_prior']),
                       ('rush',RULES['success_rush_prior'])):
        v=_interaction_residual(team,opponent,kind+'_success_sum',kind+'_success_n',
            'def_'+kind+'_success_sum','def_'+kind+'_success_n',
            prior,RULES['success_interaction_ppd_scale'])
        if v is None:missing.append(kind+'_success_matchup')
        else:
            term,rate=v
            adjusted+=term*(share if kind=='pass' else 1-share)
            components[kind+'_success_residual_ppd']=round(term,6)
            components[kind+'_success_matched_rate']=round(rate,6)
    for kind,prior in (('pass',RULES['explosive_pass_prior']),
                       ('rush',RULES['explosive_rush_prior'])):
        v=_interaction_residual(team,opponent,kind+'_explosive_sum',kind+'_explosive_n',
            'def_'+kind+'_explosive_sum','def_'+kind+'_explosive_n',
            prior,RULES['explosive_interaction_ppd_scale'])
        if v is None:missing.append(kind+'_explosive_matchup')
        else:
            term,rate=v
            adjusted+=term*(share if kind=='pass' else 1-share)
            components[kind+'_explosive_residual_ppd']=round(term,6)
            components[kind+'_explosive_matched_rate']=round(rate,6)

    # Opponent matched turnover opportunities, with all missing fields treated
    # as unknown rather than zero. Their PPD impact is simulated by the CFB
    # engine; they are NOT also subtracted from PPD here.
    ip=_matched_event(team,opponent,'int_sum','int_n','def_int_sum','def_int_n',
                      RULES['interception_prior'],RULES['event_prior_plays'])
    fl=_matched_event(team,opponent,'fumble_sum','fumble_n','def_fumble_sum','def_fumble_n',
                      RULES['lost_fumble_prior'],RULES['event_prior_plays'])
    if ip is None:missing.append('matched_interception_rate')
    if fl is None:missing.append('matched_lost_fumble_rate')
    if ip is None or fl is None:
        turnover_drive=base['_to']; components['turnover_source']='BASELINE_FALLBACK_MISSING_OBSERVATIONS'
    else:
        turnover_drive=float(_clip(1-math.exp(-plays_per_drive*(share*ip+fl)),.01,.35))
        components.update({'matched_interception_rate':round(ip,6),
                          'matched_lost_fumble_rate':round(fl,6),
                          'turnover_source':'OFFENSE_VERSUS_OPPONENT_DEFENSE'})

    # Only observed sacks (not 'qb_hurry' or purported full pressure) supported.
    sack=_matched_event(team,opponent,'sack_sum','sack_n',
                        'def_sack_sum','def_sack_n',.07,RULES['event_prior_plays'])
    if sack is None:missing.append('sack_matchup_not_full_pressure')
    else:
        # A bounded residual to matched passing efficiency: the main sack
        # impacts are already in EPA. Full pressure is explicitly unavailable.
        t=float(_clip((sack-.07)*RULES['sack_residual_ppd'],-.025,.025))
        adjusted+=t;components['sack_residual_ppd']=round(t,6)
        components['sack_rate_matched']=round(sack,6)
    missing.append('full_pressure_rate_not_source_backed')

    qb_rushes=sum(r.get('qb_rushes') or 0 for r in team)
    qb_observed=[r for r in team if r.get('qb_rushes') is not None and
                 r.get('qb_rush_epa') is not None and _finite(r.get('qb_rush_epa'))]
    if qb_rushes>=10 and len(qb_observed)>=2:
        weighted=sum(r['qb_rushes']*r['qb_rush_epa'] for r in qb_observed)/max(
                sum(r['qb_rushes'] for r in qb_observed),1)
        nonqb=sum(r['off_rush_sum'] for r in team)/max(sum(r['off_rush_n'] for r in team),1)
        reliability=qb_rushes/(qb_rushes+60.)
        # Conditional residual on top of EPA (QB rushing already counted in
        # rushing EPA); ONLY a very small usage-interaction correction.
        term=float(_clip(reliability*(weighted-nonqb)*RULES['qb_rush_residual_points_per_rush'], -.04,.04))
        adjusted+=term;components['qb_rushing_residual_ppd']=round(term,6)
    else:missing.append('qb_rushing_samples')

    our_field=_mean_optional(team,'start_field_position')
    opp_field=_mean_optional(opponent,'start_field_position')
    if our_field is None or opp_field is None:missing.append('matched_field_position')
    else:
        # Delta relative to opponent, avoids inventing a global starting-yard
        # reference. Symmetric half-shift avoids doubling its effect in margin.
        term=float(_clip((our_field-opp_field)*RULES['field_position_ppd_per_yard']/2,-.06,.06))
        adjusted+=term;components['field_position_residual_ppd']=round(term,6)
    missing.extend(('special_teams_epa_unavailable','penalty_epa_unavailable',
                    'confirmed_next_game_lineup_unavailable'))
    ppd=float(_clip(adjusted,.35,5.5))
    td=base['_td'] if rz is None else float(_clip(
        base['_td']+.38*(rz-RULES['red_zone_td_prior']),.4,.83))
    projection={'_ppd':ppd,'_td':td,'_to':turnover_drive,
                'expected_drives':base['expected_drives']}
    return {'status':'AVAILABLE_UNFITTED_INTEGRATED_SCORING_RESEARCH_ONLY',
            'version':VERSION,'projection':projection,
            'components':components,'missing_features':missing,
            'primary_forecast_modified':False,'coefficients_empirically_fitted':False,
            'limitations':['Hypothetical parameter values; no prospective evidence for accuracy',
                'Matched EPA/drive levels are blended, not added to baseline EPA',
                'Only source-observed sacks included; no full pass-pressure dataset',
                'EPA and situational events overlap; residuals small and bounded',
                'No contemporaneous injury/source lineup verification']}


def simulate_integrated(home_id,away_id,histories,neutral,*,runs=3000,seed=1,
                         use_script=False,spread_home=None):
    """Pair two scoring profiles and run either original CFB or Build14 script."""
    home=project_integrated(home_id,away_id,histories,neutral,'home')
    away=project_integrated(away_id,home_id,histories,neutral,'away')
    if home['projection'] is None or away['projection'] is None:
        return {'status':'UNAVAILABLE_REQUIRED_OPPONENT_EPA','version':VERSION,
                'home':home,'away':away,'simulation':None,
                'simulation_version':'CFB_GAME_SCRIPT' if use_script else 'CFB_BASELINE_ENGINE',
                'not_primary':True}
    if use_script:
        from cfb.game_script import simulate_script
        result=simulate_script(home['projection'],away['projection'],histories[home_id],
                               histories[away_id],runs=runs,seed=seed,spread_home=spread_home)
    else:
        result=simulate(home['projection'],away['projection'],runs=runs,seed=seed,
                        spread_home=spread_home,allow_overtime=True)
    return {'status':'AVAILABLE_UNFITTED_INTEGRATED_RESEARCH_ONLY',
            'version':VERSION,'simulation_version':('CFB_GAME_SCRIPT' if use_script else 'CFB_BASELINE_ENGINE'),
            'home':home,'away':away,'simulation':result,
            'mean_score_home':result['mean_score_home'],
            'mean_score_away':result['mean_score_away'],
            'mean_margin_home':result['mean_margin_home'],
            'not_primary':True,'fitted':False}
