"""Source-audited CFB possession duration and rush-share validation.

All observed drive time comes from ESPN drive elapsed clocks cross-checked
against period start/end. Uses prior completed games with a 24h embargo for
opponent-conditioned historical predictions. Retrospective published source
does NOT establish original historical row availability. This study does not
fit/activate a live model, use final scores, or change any baseline.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from cfb.pipeline import verify,iso_utc
from cfb.clock_possession_validation import audit_game,clock_reference_difference

# Independently published USC Athletics box for Aug 29, 2026.
# https://usctrojans.com/sports/football/stats/2026/san-jos-state/boxscore/35264
USC_SJSU_OFFICIAL={
    'game_id':401864494,'usc_id':30,'sjsu_id':23,
    'usc_possession_seconds':2235,'sjsu_possession_seconds':1365,
    'usc_rushes':40,'sjsu_rushes':24,
    'usc_pass_attempts':36,'sjsu_pass_attempts':32}

# Independently published UNC and TCU Athletic game center, Aug 29, 2026.
# https://goheels.com/game-center/26876
UNC_TCU_OFFICIAL={
    'game_id':401856766,'tcu_id':2628,'unc_id':153,
    'tcu_possession_seconds':1981,'unc_possession_seconds':1618}

def _weighted(rows,key,weight):
    included=[r for r in rows if r.get(key) is not None and r.get(weight,0)>0]
    if not included:return None
    den=sum(r[weight] for r in included)
    return sum(r[key]*r[weight] for r in included)/den if den else None

def _metric(actual,pred):
    if not actual:return None
    a=np.array(actual,dtype=float);p=np.array(pred,dtype=float)
    return {'n':len(a),'mae':round(float(np.mean(np.abs(a-p))),5),
            'bias_pred_minus_actual':round(float(np.mean(p-a)),5)}

def study(snapshot,out,min_games=3,max_games=None):
    verify(snapshot)
    root=Path(snapshot)
    stamp=iso_utc(json.loads((root/'manifest.json').read_text())['captured_at_utc'])
    schedule=pd.read_parquet(root/'schedule.parquet')
    schedule['_kickoff']=pd.to_datetime(schedule.start_date,utc=True,errors='coerce')
    valid=(schedule.fbs_participant.fillna(False).eq(True)&
        schedule.completed.fillna(False).eq(True)&
        (schedule['_kickoff']+pd.Timedelta(hours=24)<=stamp)&
        ~((schedule.home_division=='fcs')&(schedule.away_division=='fcs'))&
        schedule.home_points.notna()&schedule.away_points.notna())
    games=schedule.loc[valid].sort_values(['_kickoff','game_id']).copy()
    if max_games is not None:games=games.head(max_games).copy()
    # Never read 500+ wide columns if only source clock and class are required.
    cols=['game_id','game_play_number','drive.id','drive_id','pos_team_id',
          'pass','rush','drive.timeElapsed.displayValue',
          'drive.start.period.number','drive.start.clock.displayValue',
          'drive.end.period.number','drive.end.clock.displayValue']
    import pyarrow.parquet as pq
    present=set(pq.read_schema(root/'pbp.parquet').names)
    data=pd.read_parquet(root/'pbp.parquet',columns=[c for c in cols if c in present])
    data=data.loc[data.game_id.isin(games.game_id)]
    groups={gid:g for gid,g in data.groupby('game_id',sort=False)}
    measured={};statuses=defaultdict(int)
    for _,g in games.iterrows():
        gid=g.game_id
        result=audit_game(groups.get(gid),gid,int(g.home_id),int(g.away_id))
        statuses[result['status']]+=1
        measured[str(gid)]=result
    # Independent official cross-check on first observed 2026 opener.
    off=USC_SJSU_OFFICIAL
    ref=measured.get(str(off['game_id']))
    if ref is None:
        observed_official={'status':'GAME_NOT_IN_CAPTURE'}
    else:
        observed_official=clock_reference_difference(ref,
          {off['usc_id']:off['usc_possession_seconds'],
           off['sjsu_id']:off['sjsu_possession_seconds']})
        observed_official.update({'source_drive_count':ref['source_drive_count'],
          'valid_drive_count':ref['valid_drive_count'],
          'provider_elapsed_only_drives':ref['source_only_elapsed_drive_count'],
          'excluded_drives':ref['excluded'],
          'excluded_drive_examples':ref['excluded_drive_examples'],
          'excluded_seconds_from_clock':3600-ref['observed_seconds_total'],
          'game_id':str(off['game_id']),
          'source':'USC Athletics 2026 box score Aug 29 2026',
          'model_time_home_seconds':ref['teams'][str(off['usc_id'])]['observed_time_seconds'],
          'model_time_away_seconds':ref['teams'][str(off['sjsu_id'])]['observed_time_seconds'],
          'official_seconds_home':off['usc_possession_seconds'],
          'official_seconds_away':off['sjsu_possession_seconds'],
          'rush_plays_home':ref['teams'][str(off['usc_id'])]['rush_plays'],
          'rush_plays_away':ref['teams'][str(off['sjsu_id'])]['rush_plays'],
          'official_rushes_home':off['usc_rushes'],
          'official_rushes_away':off['sjsu_rushes'],
          'pass_plays_home':ref['teams'][str(off['usc_id'])]['pass_plays'],
          'pass_plays_away':ref['teams'][str(off['sjsu_id'])]['pass_plays'],
          'official_pass_attempts_home':off['usc_pass_attempts'],
          'official_pass_attempts_away':off['sjsu_pass_attempts']})
    # Second independent published game, contrasting playing styles and a
    # neutral site. One second can be omitted in official period bookkeeping.
    n=UNC_TCU_OFFICIAL
    other=measured.get(str(n['game_id']))
    if other is None:
        second_official={'status':'GAME_NOT_IN_CAPTURE'}
    else:
        second_official=clock_reference_difference(other,{
          n['tcu_id']:n['tcu_possession_seconds'],
          n['unc_id']:n['unc_possession_seconds']})
        second_official.update({'game_id':str(n['game_id']),
          'source':'University of North Carolina Athletics Aug 29 game center',
          'official_seconds_tcu':n['tcu_possession_seconds'],
          'official_seconds_unc':n['unc_possession_seconds'],
          'observed_seconds_tcu':other['teams'][str(n['tcu_id'])]['observed_time_seconds'],
          'observed_seconds_unc':other['teams'][str(n['unc_id'])]['observed_time_seconds'],
          'provider_elapsed_only_drives':other['source_only_elapsed_drive_count'],
          'excluded_drives':other['excluded']})
    # Exclude incomplete possession sources from statistical claims.
    eligible_games=[g for _,g in games.iterrows() if measured[str(g.game_id)]['status']=='COMPLETE_REGULATION_CLOCK_AUDIT']
    history=defaultdict(list);pending=[];rows=[]
    for g in eligible_games:
        kickoff=g['_kickoff']
        current=[x for x in pending if x[0]<=kickoff]
        pending=[x for x in pending if x[0]>kickoff]
        for _,tid,rec in current:history[tid].append(rec)
        game=measured[str(g.game_id)]
        home,away=int(g.home_id),int(g.away_id)
        prior={home:history[home],away:history[away]}
        if min(len(prior[home]),len(prior[away]))>=min_games:
            for tid,opponent in ((home,away),(away,home)):
                actual=game['teams'][str(tid)]
                own=prior[tid]
                other=prior[opponent]
                if actual['run_share'] is None or actual['seconds_per_drive'] is None:
                    continue
                own_mix=_weighted(own,'run_share','total_scrimmage')
                def_mix=_weighted(other,'run_share_allowed','opponent_total_scrimmage')
                own_duration=_weighted(own,'seconds_per_drive','valid_drives')
                def_duration=_weighted(other,'seconds_per_drive_allowed','opponent_drives')
                if None in (own_mix,def_mix,own_duration,def_duration):continue
                # Explicit, UNFITTED hypotheses; no outcome-derived coefficients.
                reliability=min(sum(x['total_scrimmage'] for x in own),
                                sum(x['opponent_total_scrimmage'] for x in other))
                w=reliability/(reliability+350.)
                proposed_mix=float(np.clip(own_mix+.45*w*(def_mix-own_mix),.10,.90))
                proposed_time=own_duration+.45*w*(def_duration-own_duration)
                rows.append({'game_id':str(g.game_id),'kickoff':kickoff.isoformat(),
                    'team_id':tid,'opponent_id':opponent,'home':tid==home,
                    'observed_rush_share':actual['run_share'],
                    'observed_seconds_per_drive':actual['seconds_per_drive'],
                    'observed_plays_per_drive':actual['scrimmage_plays_per_drive'],
                    'observed_drives':actual['valid_drives'],
                    'own_prior_rush_share':round(own_mix,6),
                    'opponent_allowed_rush_share':round(def_mix,6),
                    'own_prior_seconds_per_drive':round(own_duration,4),
                    'opponent_allowed_seconds_per_drive':round(def_duration,4),
                    'predicted_run_share_reference':round(own_mix,6),
                    'predicted_run_share_opponent':round(proposed_mix,6),
                    'predicted_drive_seconds_reference':round(own_duration,4),
                    'predicted_drive_seconds_opponent':round(proposed_time,4),
                    'reliability':round(w,6)})
        # Include complete game's actual observed features ONLY after this
        # match's predictions; prohibit same-kickoff and next-day leakage.
        for tid,opponent in ((home,away),(away,home)):
            t=game['teams'][str(tid)]
            o=game['teams'][str(opponent)]
            rec={'run_share':t['run_share'],'total_scrimmage':t['rush_plays']+t['pass_plays'],
                 'run_share_allowed':o['run_share'],
                 'opponent_total_scrimmage':o['rush_plays']+o['pass_plays'],
                 'seconds_per_drive':t['seconds_per_drive'],
                 'valid_drives':t['valid_drives'],
                 'seconds_per_drive_allowed':o['seconds_per_drive'],
                 'opponent_drives':o['valid_drives']}
            pending.append((kickoff+pd.Timedelta(hours=24),tid,rec))
    # Correlation and within-game cluster bootstrap, not causal inference.
    observed=[]
    for g in eligible_games:
        game=measured[str(g.game_id)]
        for t in (int(g.home_id),int(g.away_id)):
            z=game['teams'][str(t)]
            if z['run_share'] is not None:
                observed.append({'game_id':str(g.game_id),
                    'seconds_per_drive':z['seconds_per_drive'],
                    'rush_share':z['run_share'],
                    'scrimmage_plays_per_drive':z['scrimmage_plays_per_drive'],
                    'drives':z['valid_drives']})
    def linear_fit(records):
        x=np.array([[1.,v['scrimmage_plays_per_drive'],v['rush_share']] for v in records])
        y=np.array([v['seconds_per_drive'] for v in records])
        coeff=np.linalg.lstsq(x,y,rcond=None)[0]
        return coeff.tolist()
    regression=None
    if len(observed)>=30:
        coeff=linear_fit(observed)
        ids=sorted(set(r['game_id'] for r in observed))
        byid={gid:[r for r in observed if r['game_id']==gid] for gid in ids}
        rng=np.random.default_rng(20261022)
        estimates=[]
        for _ in range(500):
            sample=rng.choice(ids,size=len(ids),replace=True)
            group=[x for gid in sample for x in byid[gid]]
            estimates.append(linear_fit(group)[2])
        xs=np.array(estimates,float)
        regression={'n_team_games':len(observed),'n_games':len(ids),
           'seconds_per_extra_rush_share_unit_controlling_plays_per_drive':round(coeff[2],4),
           'seconds_per_plus_10pct_rush_share':round(coeff[2]/10,4),
           'cluster_game_bootstrap95_seconds_per_plus_10pct_run_share':[
                round(float(v),4) for v in np.quantile(xs/10,[.025,.975])],
           'seconds_per_additional_scrimmage_play':round(coeff[1],4),
           'interpretation':'DESCRIPTIVE_CONDITIONAL_ASSOCIATION_NOT_CAUSAL'}
    evaluations={}
    if rows:
        evaluations['rush_mix_reference']=_metric([r['observed_rush_share'] for r in rows],
            [r['predicted_run_share_reference'] for r in rows])
        evaluations['rush_mix_opponent']=_metric([r['observed_rush_share'] for r in rows],
            [r['predicted_run_share_opponent'] for r in rows])
        evaluations['clock_reference']=_metric([r['observed_seconds_per_drive'] for r in rows],
            [r['predicted_drive_seconds_reference'] for r in rows])
        evaluations['clock_opponent']=_metric([r['observed_seconds_per_drive'] for r in rows],
            [r['predicted_drive_seconds_opponent'] for r in rows])
    report={'schema':'cfb_clock_rush_relationship_v1',
        'source_capture':str(stamp),'games_from_schedule':len(games),
        'statuses':dict(statuses),
        'source_clock_completeness_gate':'14 drives, at least 6 per team, 3150-3650 sec; maximum two flagged provider-elapsed-only drives',
        'independent_official_validation':observed_official,
        'second_independent_official_validation':second_official,
        'observational_regression':regression,
        'historical_prediction_pairs':len(rows),
        'distinct_prediction_games':len(set(r['game_id'] for r in rows)),
        'prediction_metrics':evaluations,
        'prediction_records':rows,
        'deployment':False,'ATS_calibrated':False,
        'limitations':['Recorded clock durations are provider drive elapsed times, checked against start/end clocks',
           'Source full data captured retrospectively, historical per-row publication times not proven',
           'Training uses only games with kickoff at least 24h before target kickoff',
           'Correlation is observational, confounded by game script and down-distance',
           'No spread betting data, no ATS validation or full margin challenger promotion']}
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError('Results are append-only')
    p.write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n')
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True)
    p.add_argument('--max-games',type=int,default=None)
    args=p.parse_args()
    r=study(args.snapshot,args.out,max_games=args.max_games)
    print(json.dumps({k:v for k,v in r.items() if k!='prediction_records'},indent=2,default=str))

if __name__=='__main__':main()
