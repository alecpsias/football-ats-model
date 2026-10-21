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
         'scoring_off':None,'fourth_off':None,'penalty_first_down_def':None,
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
    # Red-zone drive and fourth-down results are observed descriptive evidence;
    # do not assign causal scoring shifts without non-overlapping validation.
    if {'drive_id','pos_team_id','yards_to_goal','offense_score_play','td_play',
        'fg_made','downs_turnover','pos_fumble_lost','int'}.issubset(plays.columns):
        teamplays=plays.loc[plays.pos_team_id==team].sort_values('game_play_number')
        if not teamplays.empty:
            scoring={'td':0,'fg':0,'turnover':0,'turnover_on_downs':0,'other':0,'opportunities':0}
            for _,part in teamplays.groupby('drive_id',sort=False):
                part=part.sort_values('game_play_number')
                if not part['yards_to_goal'].le(40).any():continue
                scoring['opportunities']+=1
                if (part.offense_score_play.eq(True)&part.td_play.eq(True)).any():scoring['td']+=1
                elif part.fg_made.eq(True).any():scoring['fg']+=1
                elif part.downs_turnover.eq(True).any():scoring['turnover_on_downs']+=1
                elif (part['int'].eq(True)|part.pos_fumble_lost.eq(True)).any():scoring['turnover']+=1
                else:scoring['other']+=1
            out['scoring_off']=scoring
    if {'pos_team_id','down','distance','rush','pass','downs_turnover','firstD_by_yards',
        'firstD_by_penalty','rush_td','pass_td'}.issubset(plays.columns):
        q=plays.loc[(plays.pos_team_id==team)&plays.down.eq(4)&
                    (plays['rush'].eq(True)|plays['pass'].eq(True))]
        if not q.empty and not q[['downs_turnover','firstD_by_yards','firstD_by_penalty','rush_td','pass_td']].isna().any().any():
            out['fourth_off']={'attempted':int(len(q)),
              'converted':int(q[['firstD_by_yards','firstD_by_penalty','rush_td','pass_td']].eq(True).any(axis=1).sum()),
              'failed_marked':int(q['downs_turnover'].eq(True).sum())}
    if {'penalty_1st_conv','pos_team_id'}.issubset(plays.columns):
        against=plays.loc[plays.pos_team_id==opponent]
        if not against.empty and against.penalty_1st_conv.notna().all():
            out['penalty_first_down_def']={'allowed':int(against.penalty_1st_conv.eq(True).sum()),
                                         'plays':int(len(against))}
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
    # Other features (fourth-down decisions, scoring opportunities, penalty first
    # downs) remain descriptive until independently estimated as residuals.
    return {'version':VERSION,
     'home':{'third':third(h,a,home_projection),'pass_states':pass_states(h,hneutral),
             'neutral_pass_rate':round(float(hneutral),6),
             'recorded_disruption':matchup_disruption(h,a)},
     'away':{'third':third(a,h,away_projection),'pass_states':pass_states(a,aneutral),
             'neutral_pass_rate':round(float(aneutral),6),
             'recorded_disruption':matchup_disruption(a,h)},
     'turnover_field':fields(h,a),
     'unscored_descriptive':{'scoring_opportunity_outcomes':'OBSERVED_WHEN_AVAILABLE_NO_FITTED_EFFECT',
          'fourth_down':'OBSERVED_WHEN_AVAILABLE_NO_FITTED_EFFECT',
          'penalty_automatic_first_down':'OBSERVED_WHEN_AVAILABLE_NO_FITTED_EFFECT'},
     'assumptions':{'third_distance_residual':'bounded <=0.055 PPD per offense, not fitted',
                    'pass_state_response':'shrunken observed frequencies, not estimated causal elasticities',
                    'turnover_following_field_ppd_per_yard':0.018,
                    'recorded_disruption_adjustment_ppd':0.0,
                    'disruption_scoring_policy':'No incremental scoring coefficient until independent evidence'}}


def simulate_build16(home_id,away_id,histories,neutral,*,runs=3000,seed=1,spread_home=None,stress_runs=0):
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
        proj['_ppd']=float(np.clip(proj['_ppd']+profile[side]['third']['residual_ppd'],.35,5.5))
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
    return {'status':'AVAILABLE_UNFITTED_BUILD16_RESEARCH_ONLY','version':VERSION,
       'adverse_sensitivity':{'counterfactuals':stress,'not_fitted':True},
       'simulation':out,'mean_margin_home':out['mean_margin_home'],
       'mean_score_home':out['mean_score_home'],'mean_score_away':out['mean_score_away'],
       'margin_p10_p50_p90':out['margin_p10_p50_p90'],
       'home':home,'away':away,'mechanism_evidence':profile,
       'not_primary':True,'fitted':False,'ATS_calibrated':False}
