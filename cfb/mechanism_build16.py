"""Build 16: one CFB situational challenger, never a primary forecast.

Only earlier completed games enter histories. Mechanism data is descriptive,
missing fields remain missing, and experimental effects are bounded. There is
exactly ONE published candidate: integrated scorer + sequential script + these
observed interactions. Neither target outcome nor sportsbook line fits effects.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

from cfb.integrated_scoring import project_integrated
from cfb.game_script import simulate_script
from cfb.disruption import matchup_disruption
from cfb.situational_scoring import matchup_effects

VERSION='cfb-build16-unified-situational-v1'
BUCKETS=('short_1_2','medium_3_6','long_7_plus')
STATES=('trailing_9_plus','within_8','leading_9_plus')


def _counts(rows, key):
    agg={k:[0,0] for k in key}
    for r in rows:
        if not isinstance(r,dict):continue
        for k in key:
            z=r.get(k)
            if isinstance(z,(list,tuple)) and len(z)==2 and all(isinstance(n,int) and not isinstance(n,bool) and n>=0 for n in z) and z[0]<=z[1]:
                agg[k][0]+=z[0];agg[k][1]+=z[1]
    return agg


def observe_game_context(plays, team, opponent):
    """Create source-observed counts from one past game, not target fixtures.

    Context is optional and uses the ORIGINAL columns; omitted fields disable
    their mechanism. Drives are ordered by real game-play-number, not by IDs.
    Score state uses pos_score_diff_start (pre-play), never final score.
    """
    out={'third_off':None,'third_def':None,'pass_states_off':None,
         'scoring_off':None,'scoring_def':None,'fourth_off':None,'fourth_def':None,
         'penalty_first_down_off':None,'penalty_first_down_def':None,
         'turnover_zone_off':None,'turnover_next_field_by_zone':None}
    if not isinstance(plays,pd.DataFrame) or plays.empty:return out
    required={'pos_team_id','def_pos_team_id','game_play_number','down','distance',
              'firstD_by_yards','firstD_by_penalty','rush_td','pass_td','pass','rush'}
    if required.issubset(plays.columns):
        def buckets(team_id):
            part=plays.loc[(plays.pos_team_id==team_id)&(plays.down==3)].copy()
            used=part.loc[part.distance.notna()&part.distance.ge(1)]
            cols=('firstD_by_yards','firstD_by_penalty','rush_td','pass_td')
            if used.empty or used[list(cols)].isna().any().any():return None
            z={b:[0,0] for b in BUCKETS}
            for row in used.itertuples(index=False):
                d=int(row.distance);b=BUCKETS[0] if d<=2 else BUCKETS[1] if d<=6 else BUCKETS[2]
                good=any(bool(getattr(row,c)) for c in cols)
                z[b][0]+=int(good);z[b][1]+=1
            return z
        out['third_off']=buckets(team);out['third_def']=buckets(opponent)
    if {'pos_score_diff_start','pass','rush','pos_team_id'}.issubset(plays.columns):
        off=plays.loc[(plays.pos_team_id==team)&(plays['pos_score_diff_start'].notna())]
        off=off.loc[off['pass'].eq(True)^off['rush'].eq(True)]
        if not off.empty:
            z={s:[0,0] for s in STATES}
            for _,row in off.iterrows():
                diff=float(row['pos_score_diff_start'])
                state=STATES[0] if diff<=-9 else STATES[2] if diff>=9 else STATES[1]
                z[state][0]+=int(bool(row['pass']));z[state][1]+=1
            out['pass_states_off']=z
    def opportunities(tid):
        needed={'drive_id','pos_team_id','yards_to_goal','offense_score_play',
                'td_play','fg_made','downs_turnover','pos_fumble_lost','int'}
        if not needed.issubset(plays.columns):return None
        rows=plays.loc[plays.pos_team_id.eq(tid)].sort_values('game_play_number')
        if rows.empty or rows[['drive_id','yards_to_goal']].isna().any().any():return None
        counts={'td':0,'fg':0,'turnover':0,'turnover_on_downs':0,
                'other':0,'opportunities':0}
        for _,part in rows.groupby('drive_id',sort=False):
            if not part.yards_to_goal.le(40).any():continue
            cols=['offense_score_play','td_play','fg_made','downs_turnover','pos_fumble_lost','int']
            if part[cols].isna().any().any():return None
            counts['opportunities']+=1
            if (part.offense_score_play.eq(True)&part.td_play.eq(True)).any():counts['td']+=1
            elif part.fg_made.eq(True).any():counts['fg']+=1
            elif part.downs_turnover.eq(True).any():counts['turnover_on_downs']+=1
            elif (part['int'].eq(True)|part.pos_fumble_lost.eq(True)).any():counts['turnover']+=1
            else:counts['other']+=1
        return counts
    out['scoring_off']=opportunities(team)
    out['scoring_def']=opportunities(opponent)

    def fourth(tid):
        needed={'pos_team_id','down','rush','pass','downs_turnover',
                'firstD_by_yards','firstD_by_penalty','rush_td','pass_td'}
        if not needed.issubset(plays.columns):return None
        z=plays.loc[plays.pos_team_id.eq(tid)&plays.down.eq(4)&
                    (plays.rush.eq(True)|plays['pass'].eq(True))]
        fields=['downs_turnover','firstD_by_yards','firstD_by_penalty','rush_td','pass_td']
        if z.empty or z[fields].isna().any().any():return None
        return {'attempted':int(len(z)),
                'converted':int(z[fields[1:]].eq(True).any(axis=1).sum()),
                'failed_marked':int(z.downs_turnover.eq(True).sum())}
    out['fourth_off']=fourth(team)
    out['fourth_def']=fourth(opponent)

    def penalty_firsts(tid):
        needed={'pos_team_id','down','penalty_1st_conv'}
        if not needed.issubset(plays.columns):return None
        z=plays.loc[plays.pos_team_id.eq(tid)&plays.down.isin((1,2,4))]
        if z.empty or z.penalty_1st_conv.isna().any():return None
        count=int(z.penalty_1st_conv.eq(True).sum())
        return {'awarded':count,'allowed':count,'plays':int(len(z))}
    out['penalty_first_down_off']=penalty_firsts(team)
    out['penalty_first_down_def']=penalty_firsts(opponent)

    # Next actual opposition drive starting field position, conditional on the
    # field zone of the turnover. Do not equate turnover SPOT with next start.
    keys={'drive_id','game_play_number','pos_team_id','yards_to_goal','int','pos_fumble_lost'}
    if keys.issubset(plays.columns) and plays[list(keys)].notna().all().all():
        ordered=plays.sort_values('game_play_number')
        sequence=[p for _,p in ordered.groupby('drive_id',sort=False)]
        zone={'opponent_40_or_closer':[],'outside_opponent_40':[]}
        exposed={b:0 for b in zone}
        for i,part in enumerate(sequence):
            if int(part.pos_team_id.iloc[0])!=team:continue
            turnovers=part.loc[part['int'].eq(True)|part.pos_fumble_lost.eq(True)]
            if turnovers.empty:continue
            last=turnovers.iloc[-1]
            loc=float(last.yards_to_goal)
            if not 0<=loc<=100:continue
            b='opponent_40_or_closer' if loc<=40 else 'outside_opponent_40'
            exposed[b]+=1
            if i+1>=len(sequence):continue
            following=sequence[i+1]
            if int(following.pos_team_id.iloc[0])!=opponent:continue
            # Halftime transitions and pick-sixes can distort direct next-drive
            # joins; flag only same-half, opposing subsequent offensive drives.
            if 'half' in part and 'half' in following and part['half'].notna().all() and following['half'].notna().all():
                if int(part['half'].iloc[-1])!=int(following['half'].iloc[0]):continue
            nxt=float(following.yards_to_goal.iloc[0])
            if 1<=nxt<=99:zone[b].append(round(100.-nxt,2))
        if sum(exposed.values()):
            out['turnover_zone_off']=exposed
            out['turnover_next_field_by_zone']=zone
    return out


def _rate(row, prior, pseudo):
    yes,n=row
    return (yes+prior*pseudo)/(n+pseudo)


def mechanism_profile(home_id,away_id,histories, home_projection,away_projection):
    """Match possession mechanisms using pregame histories only."""
    h=histories.get(home_id,[]);a=histories.get(away_id,[])
    if not h or not a:raise ValueError('Missing team histories')
    def third(off,defense,base):
        oc=_counts(((r.get('mechanism') or {}).get('third_off') for r in off), BUCKETS)
        dc=_counts(((r.get('mechanism') or {}).get('third_def') for r in defense), BUCKETS)
        valid=sum(oc[b][1] for b in BUCKETS)>0 and sum(dc[b][1] for b in BUCKETS)>0
        if not valid:return {'status':'UNAVAILABLE','residual_ppd':0.,'distance_mix':None}
        matched={}
        for b in BUCKETS:
            p=_rate([oc[b][0]+dc[b][0],oc[b][1]+dc[b][1]],.405,60.)
            matched[b]=p
        obs=np.array([oc[b][1] for b in BUCKETS],float);weights=(obs+10)/(obs.sum()+30)
        weighted=sum(weights[i]*matched[b] for i,b in enumerate(BUCKETS))
        # Only differential from observed aggregate matchup, bounded to avoid
        # counting third-down conversions already represented in EPA and Build15.
        reference=(sum(oc[b][0]+dc[b][0] for b in BUCKETS)+60*.405)/(sum(oc[b][1]+dc[b][1] for b in BUCKETS)+60)
        resid=float(np.clip((weighted-reference)*.6,-.055,.055))
        return {'status':'OBSERVED_SHRUNK','matched_by_distance':{k:round(v,5) for k,v in matched.items()},
            'distance_mix':{b:round(float(weights[i]),5) for i,b in enumerate(BUCKETS)},
            'residual_ppd':round(resid,6),
            'off_attempts':int(obs.sum()),'def_attempts':sum(dc[b][1] for b in BUCKETS)}
    def pass_states(rows,neutral):
        agg=_counts(((r.get('mechanism') or {}).get('pass_states_off') for r in rows),STATES)
        normal=agg['within_8'];total=sum(agg[x][1] for x in STATES)
        if total<35 or normal[1]<12:return {'status':'INSUFFICIENT','by_state':None,'attempts':total}
        prior=normal[0]/normal[1]
        by={}
        for state in STATES:
            fraction=_rate(agg[state],prior,60.)
            by[state]=round(float(np.clip(neutral+fraction-_rate(normal,prior,60.),.12,.92)),6)
        return {'status':'OBSERVED_SHRUNK','by_state':by,'attempts':total}
    def fields(off,defense):
        # Draw from actual prior next-drive starts, not the turnover spot.
        ob={'opponent_40_or_closer':[],'outside_opponent_40':[]}
        exposed={k:0 for k in ob}
        for r in list(off)+list(defense):
            m=r.get('mechanism') or {}
            for k in ob:
                observed=(m.get('turnover_next_field_by_zone') or {}).get(k,[])
                ob[k].extend(float(x) for x in observed if isinstance(x,(int,float)) and math.isfinite(x) and 1<=x<=99)
                exposed[k]+=int((m.get('turnover_zone_off') or {}).get(k,0))
        return {'status':'OBSERVED' if sum(len(x) for x in ob.values())>=4 else 'INSUFFICIENT',
                'following_start_by_turnover_zone':ob,'turnover_zone_counts':exposed,
                'samples':sum(len(x) for x in ob.values())}
    hneutral=sum(r['off_pass_n'] for r in h)/max(1,sum(r['off_pass_n']+r['off_rush_n'] for r in h))
    aneutral=sum(r['off_pass_n'] for r in a)/max(1,sum(r['off_pass_n']+r['off_rush_n'] for r in a))
    # Situational outcome residuals are explicitly UNFITTED research only.
    return {'version':VERSION,
     'home':{'third':third(h,a,home_projection),'pass_states':pass_states(h,hneutral),
             'neutral_pass_rate':round(float(hneutral),6),
             'recorded_disruption':matchup_disruption(h,a),
             'situational_scoring':matchup_effects(h,a)},
     'away':{'third':third(a,h,away_projection),'pass_states':pass_states(a,aneutral),
             'neutral_pass_rate':round(float(aneutral),6),
             'recorded_disruption':matchup_disruption(a,h),
             'situational_scoring':matchup_effects(a,h)},
     'turnover_field':fields(h,a),
     'situational_scoring_policy':'BOUNDED_UNFITTED_RESEARCH_NOT_ATS_CALIBRATED',
     'assumptions':{'third_distance_residual':'bounded <=0.055 PPD per offense, not fitted',
                    'pass_state_response':'shrunken observed frequencies, not estimated causal elasticities',
                    'turnover_following_field_ppd_per_yard':0.018,
                    'recorded_disruption_adjustment_ppd':0.0,
                    'disruption_scoring_policy':'No incremental scoring coefficient until independent evidence'}}


def simulate_build16(home_id,away_id,histories,neutral,*,runs=3000,seed=1,spread_home=None,stress_runs=0,rush_research=None,chain_research=None,mismatch_research=None,failure1_fit_margin_delta=None,possession_response=None):
    if sum(v is not None for v in (rush_research,chain_research,mismatch_research,failure1_fit_margin_delta,possession_response))>1:
        raise ValueError('Only one within-challenger ablation at a time')
    if type(stress_runs) is not int or not (stress_runs==0 or 100<=stress_runs<=10000):
        raise ValueError('Stress runs must be 0 or 100..10000')
    home=project_integrated(home_id,away_id,histories,neutral,'home')
    away=project_integrated(away_id,home_id,histories,neutral,'away')
    if home['projection'] is None or away['projection'] is None:
        return {'status':'UNAVAILABLE_REQUIRED_OPPONENT_EPA','version':VERSION,
                'simulation':None,'not_primary':True,'home':home,'away':away}
    h=home['projection'].copy();a=away['projection'].copy()
    profile=mechanism_profile(home_id,away_id,histories,h,a)
    for side,proj in (('home',h),('away',a)):
        so=profile[side]['situational_scoring']
        # Apply finishing as an expected point shift, not just a changed
        # TD/FG conditional share that the PPD normalizer cancels.
        scoring_rate=float(np.clip(proj['_ppd']/(3.0+3.965*proj['_td']),.01,.8))
        finish_ppd=float(3.965*scoring_rate*so['delta_td_share'])
        so['applied_finishing_delta_ppd']=round(finish_ppd,6)
        proj['_ppd']=float(np.clip(proj['_ppd']+profile[side]['third']['residual_ppd']+
                                  so['delta_ppd']+finish_ppd,.35,5.5))
        proj['_td']=float(np.clip(proj['_td']+so['delta_td_share'],.35,.9))
    # Optional, explicit ABlation of one challenger. Never active on production
    # paths without caller selecting a research mode. No published second model.
    rush_evidence=None
    if rush_research is not None:
        from cfb.rushing_interaction_research import experiment
        rush_evidence=experiment(home_id,away_id,histories,
                                 home['components'],away['components'],rush_research)
        if rush_evidence['status']!='UNFITTED_SCENARIO_READY':
            return {'status':'UNAVAILABLE_RUSH_RESEARCH_EVIDENCE',
                    'version':VERSION,'simulation':None,
                    'rush_research':rush_evidence,'not_primary':True,
                    'ATS_calibrated':False}
        if rush_evidence['pass_mix_active']:
            for side,proj in (('home',h),('away',a)):
                z=rush_evidence[side]
                proj['_ppd']=float(np.clip(
                    proj['_ppd']+rush_evidence['ppd_'+side+'_delta'],.35,5.5))
                # Shift pregame pass tendency while preserving historical
                # within-game trailing/leading response; interception changes
                # are only the relative *exposure* already in the script.
                state=profile[side]['pass_states']
                prev=profile[side]['neutral_pass_rate']
                new=1.0-z['projected_rush_share']
                profile[side]['neutral_pass_rate']=float(np.clip(new,.05,.95))
                if isinstance(state.get('by_state'),dict):
                    state['by_state']={k:float(np.clip(v+new-prev,.12,.92))
                                       for k,v in state['by_state'].items()}
        if rush_evidence['tempo_active']:
            # Shared change to possession count, not an additional per-side
            # possession (both sides' game clocks are coupled).
            d=rush_evidence['shared_tempo_delta']
            for proj in (h,a):
                proj['expected_drives']=float(np.clip(proj['expected_drives']+d,6,19))
    # Experimental chain mechanism: opt-in only, no default impact. The
    # original third-down success and rushing EPA are already in Build16;
    # apply only a bounded *relative-to-own-history* interaction residual.
    chain_evidence=None
    if chain_research is not None:
        if rush_research is not None:
            raise ValueError('Only one optional CFB interaction experiment at a time')
        from cfb.chain_interaction_research import experiment as chain_experiment
        chain_evidence=chain_experiment(home_id,away_id,histories,chain_research)
        if chain_evidence['status']!='UNFITTED_SCENARIO_READY':
            return {'status':'UNAVAILABLE_CHAIN_RESEARCH_EVIDENCE',
                    'version':VERSION,'simulation':None,
                    'chain_research':chain_evidence,'not_primary':True,
                    'ATS_calibrated':False}
        h['_ppd']=float(np.clip(h['_ppd']+chain_evidence['ppd_home_delta'],.35,5.5))
        a['_ppd']=float(np.clip(a['_ppd']+chain_evidence['ppd_away_delta'],.35,5.5))
    # Failure-1 severe-rushing mismatch: pregame-only opt-in laboratory effect.
    # Preserve default Build16 and forbid simultaneous experimental toggles.
    mismatch_evidence=None
    if mismatch_research is not None:
        if chain_research is not None or rush_research is not None:
            raise ValueError('Only one within-challenger research ablation permitted')
        from cfb.rush_severity_research import experiment as severity_experiment
        mismatch_evidence=severity_experiment(home_id,away_id,histories,mismatch_research)
        if mismatch_evidence['status']!='UNFITTED_SCENARIO_READY':
            return {'status':'UNAVAILABLE_SEVERE_RUSH_EVIDENCE',
                    'version':VERSION,'simulation':None,'not_primary':True,
                    'mismatch_research':mismatch_evidence,'ATS_calibrated':False}
        h['_ppd']=float(np.clip(h['_ppd']+mismatch_evidence['ppd_home_delta'],.35,5.5))
        a['_ppd']=float(np.clip(a['_ppd']+mismatch_evidence['ppd_away_delta'],.35,5.5))
        if mismatch_research in ('resilience','combined'):
            profile['home']['trailing_rush_resilience_ppd']=mismatch_evidence['home_trailing_only_ppd']
            profile['away']['trailing_rush_resilience_ppd']=mismatch_evidence['away_trailing_only_ppd']
    # Lab-only, learned in an EARLIER year and bounded in home-margin points.
    # Translate one signed margin change into PPD using equal-and-opposite
    # shifts to the two teams' drive scoring. This changes actual simulated
    # margin distributions, NOT a posthoc edit to the mean prediction.
    fitted_delta=None
    if failure1_fit_margin_delta is not None:
        if type(failure1_fit_margin_delta) not in (float,int):
            raise ValueError('Fitted rushing margin adjustment must be numeric')
        fitted_delta=float(failure1_fit_margin_delta)
        if not np.isfinite(fitted_delta) or abs(fitted_delta)>3.0:
            raise ValueError('Fitted rushing correction must be capped at ±3.0 points')
        drives=(float(h['expected_drives'])+float(a['expected_drives']))/2
        if not 6<=drives<=19:raise ValueError('Invalid expected drives')
        h['_ppd']=float(np.clip(h['_ppd']+fitted_delta/(2*drives),.35,5.5))
        a['_ppd']=float(np.clip(a['_ppd']-fitted_delta/(2*drives),.35,5.5))
    # Opt-in empirical rushing-chain / real clock possession model.
    # Pre-game evidence and earlier-year fit are produced by the caller.
    # A scoring event still belongs to the SAME sequential drive simulation,
    # not an independent additive "rushing EPA" point bonus.
    possession_effect=None
    if possession_response is not None:
        if not isinstance(possession_response,dict) or possession_response.get('status')!='AVAILABLE_FROZEN_POSSESSION_RESPONSE':
            raise ValueError('Only verified, earlier-year-fitted possession response accepted')
        hh=possession_response.get('home_ppd_shift');aa=possession_response.get('away_ppd_shift')
        td=possession_response.get('shared_clock_drives_shift')
        for v in (hh,aa,td):
            if type(v) not in (int,float) or not np.isfinite(v):raise ValueError('Invalid possession numeric input')
        if abs(hh)>.22 or abs(aa)>.22 or abs(td)>.65:
            raise ValueError('Excessive possession/clock correction')
        if possession_response.get('real_clock_both_sides_valid') is False and abs(td)>1e-9:
            raise ValueError('Cannot modify tempo without verified possession seconds')
        h['_ppd']=float(np.clip(h['_ppd']+float(hh),.35,5.5))
        a['_ppd']=float(np.clip(a['_ppd']+float(aa),.35,5.5))
        # Two teams share the same sixty-minute regulation clock: never
        # change one side's opportunities without the other.
        h['expected_drives']=float(np.clip(h['expected_drives']+td,6,19))
        a['expected_drives']=float(np.clip(a['expected_drives']+td,6,19))
        possession_effect=possession_response
    # One, and only one, challenger: Build15 scoring + Build14 script + B16
    # state-dependent play selection and turnover-location field starts.
    out=simulate_script(h,a,histories[home_id],histories[away_id],runs=runs,seed=seed,
                        spread_home=spread_home,mechanisms={
                          'home':{**profile['home'],
                                  'int_rate':home['components'].get('matched_interception_rate'),
                                  'fumble_rate':home['components'].get('matched_lost_fumble_rate'),
                                  'plays_per_drive':home['components'].get('plays_per_drive_observed')},
                          'away':{**profile['away'],
                                  'int_rate':away['components'].get('matched_interception_rate'),
                                  'fumble_rate':away['components'].get('matched_lost_fumble_rate'),
                                  'plays_per_drive':away['components'].get('plays_per_drive_observed')},
                          'turnover_field':profile['turnover_field']})
    stress={}
    if stress_runs:
        shocks={'fewer_possessions':{'tempo_shift':-1.5},
                'more_possessions':{'tempo_shift':1.5},
                'home_turnover_swing':{'turnover_shift_home':.07},
                'away_turnover_swing':{'turnover_shift_away':.07},
                'home_scoring_stall':{'efficiency_shift_home':-.35},
                'away_scoring_stall':{'efficiency_shift_away':-.35}}
        mech={
            'home':{**profile['home'],'int_rate':home['components'].get('matched_interception_rate'),
                    'fumble_rate':home['components'].get('matched_lost_fumble_rate'),
                    'plays_per_drive':home['components'].get('plays_per_drive_observed')},
            'away':{**profile['away'],'int_rate':away['components'].get('matched_interception_rate'),
                    'fumble_rate':away['components'].get('matched_lost_fumble_rate'),
                    'plays_per_drive':away['components'].get('plays_per_drive_observed')},
            'turnover_field':profile['turnover_field']}
        for label,change in shocks.items():
            r=simulate_script(h,a,histories[home_id],histories[away_id],
                    runs=stress_runs,seed=seed,mechanisms=mech,**change)
            stress[label]={'mean_home_margin':r['mean_margin_home'],
                           'mean_home_score':r['mean_score_home'],
                           'mean_away_score':r['mean_score_away'],
                           'scenario_inputs':change}
    answer={'status':'AVAILABLE_UNFITTED_BUILD16_RESEARCH_ONLY','version':VERSION,
       'adverse_sensitivity':{'counterfactuals':stress,'not_fitted':True,
                              'simulations_per_scenario':stress_runs},
       'simulation':out,'mean_margin_home':out['mean_margin_home'],
       'mean_score_home':out['mean_score_home'],'mean_score_away':out['mean_score_away'],
       'margin_p10_p50_p90':out['margin_p10_p50_p90'],
       'home':home,'away':away,'mechanism_evidence':profile,
       'not_primary':True,'fitted':False,'ATS_calibrated':False}
    if rush_evidence is not None:
        answer['rush_research']=rush_evidence
        answer['not_deployed']=True
    if chain_evidence is not None:
        answer['chain_research']=chain_evidence
        answer['not_deployed']=True
    if mismatch_evidence is not None:
        answer['mismatch_research']=mismatch_evidence
        answer['not_deployed']=True
    if fitted_delta is not None:
        answer['failure1_fitted_rushing_margin_delta']=fitted_delta
        answer['failure1_scoring_status']='ACTIVE_IN_BUILD16_CHALLENGER_UNCALIBRATED'
        answer['not_deployed']=False
    if possession_effect is not None:
        answer['failure1_possession_response']=possession_effect
        answer['not_deployed']=True
    return answer
