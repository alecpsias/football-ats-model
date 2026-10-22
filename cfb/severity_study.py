"""Chronological, paired CFB nonlinear rushing mismatch/severity pilot.

One unchanged Build16 reference plus three opt-in scenarios within that
challenger. Every *actual* model invocation uses 5,000 draws and the same
fixture seed. Never fit on NM State v Sam Houston, never touch NFL/baseline.
The retrospective release is NOT a timestamp-authenticated as-of archive.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from cfb.pipeline import verify,iso_utc,normalize_pbp_teams
from cfb.research_backtest import observed_game
from cfb.mechanism_build16 import simulate_build16
from cfb.rush_severity_source import extract,append_training_record

MODES=('reference','severity','negative','resilience','combined')
EXCLUDE_GAME_ID=401871070

def mini(value):
    z=value['simulation']
    return {'margin':z['mean_margin_home'],'total':z['mean_total'],
            'win_probability_raw':z['home_win_probability_raw'],
            'margin_p10_p50_p90':z['margin_p10_p50_p90'],
            'simulation_count':z['simulations'],'seed':z['seed']}

def paired_score(rows,modes=('reference','severity','negative')):
    rows=[r for r in rows if all(m in r['simulated'] for m in modes)]
    ret={'paired_games':len(rows),'tested_modes':list(modes)}
    if not rows:return ret
    actual=np.array([r['actual_home_margin'] for r in rows],float)
    total=np.array([r['actual_total'] for r in rows],float)
    win=(actual>0).astype(float)
    for mode in modes:
        x=np.array([r['simulated'][mode]['margin'] for r in rows])
        t=np.array([r['simulated'][mode]['total'] for r in rows])
        p=np.array([r['simulated'][mode]['win_probability_raw'] for r in rows])
        cov=np.array([r['simulated'][mode]['margin_p10_p50_p90'] for r in rows])
        err=x-actual
        ret[mode]={'margin_mae':round(float(np.mean(abs(err))),5),
            'margin_bias':round(float(np.mean(err)),5),
            'margin_rmse':round(float(np.sqrt(np.mean(err*err))),5),
            'total_mae':round(float(np.mean(abs(t-total))),5),
            'outright_brier_raw':round(float(np.mean((p-win)**2)),6),
            'margin_p10_p90_coverage':round(float(np.mean((actual>=cov[:,0])&(actual<=cov[:,2]))),5),
            'winner_direction_accuracy':round(float(np.mean(np.sign(actual)==np.sign(x))),5)}
    reference=np.array([abs(r['simulated']['reference']['margin']-r['actual_home_margin'])
                         for r in rows])
    for mode in modes:
        if mode=='reference':continue
        alt=np.array([abs(r['simulated'][mode]['margin']-r['actual_home_margin'])
                      for r in rows])
        diff=reference-alt
        rng=np.random.default_rng(20261022)
        selections=rng.integers(0,len(rows),size=(3000,len(rows)))
        bootstrap=diff[selections].mean(axis=1)
        ret[mode]['mae_improvement_vs_reference']=round(float(diff.mean()),5)
        ret[mode]['paired_bootstrap95_improvement']=[
            round(float(x),5) for x in np.quantile(bootstrap,[.025,.975])]
    return ret

def study(snapshot,out,*,runs=5000,max_games=180,min_prior_games=3):
    if type(runs) is not int or runs!=5000:
        raise ValueError('Protocol requires exactly 5,000 simulations each')
    if type(max_games) is not int or not 1<=max_games<=300:
        raise ValueError('Invalid game cap')
    root=Path(snapshot);manifest=verify(root);snapshot_at=iso_utc(manifest['captured_at_utc'])
    sched=pd.read_parquet(root/'schedule.parquet')
    pbp=pd.read_parquet(root/'pbp.parquet')
    roster=pd.read_parquet(root/'roster.parquet')
    sched['_kickoff']=pd.to_datetime(sched.start_date,utc=True,errors='coerce')
    valid=(sched.fbs_participant.fillna(False).eq(True)&
        sched.completed.fillna(False).eq(True)&
        (sched['_kickoff']+pd.Timedelta(hours=24)<=snapshot_at)&
        ~((sched.home_division=='fcs')&(sched.away_division=='fcs'))&
        sched.home_points.notna()&sched.away_points.notna()&
        sched.game_id.ne(EXCLUDE_GAME_ID)&
        sched['_kickoff'].lt(pd.Timestamp('2026-10-21',tz='UTC')))
    sched=sched.loc[valid].sort_values(['_kickoff','game_id'])
    mapped=normalize_pbp_teams(pbp.loc[pbp.game_id.isin(sched.game_id)],sched)
    plays_by_id={i:g for i,g in mapped.groupby('game_id',sort=False)}
    raw_by_id={i:g for i,g in pbp.loc[pbp.game_id.isin(sched.game_id)].groupby('game_id',sort=False)}
    qb={int(t):set(g.athlete_id.dropna()) for t,g in
        roster.loc[roster.position_abbreviation.eq('QB')].groupby('team_id')}
    history=defaultdict(list);pending=[];chosen=[];excluded=defaultdict(int)
    third_box=[]
    for kickoff,block in sched.groupby('_kickoff',sort=True):
        ready=[x for x in pending if x[0]<=kickoff]
        pending=[x for x in pending if x[0]>kickoff]
        for _,team,record in ready:history[team].append(record)
        for _,g in block.iterrows():
            hid,aid=int(g.home_id),int(g.away_id)
            if len(chosen)<max_games and min(len(history[hid]),len(history[aid]))>=min_prior_games:
                seed=int(g.game_id)%2**32
                simulations={}
                ref=simulate_build16(hid,aid,history,bool(g.neutral_site),runs=runs,seed=seed)
                if ref.get('simulation') is None:
                    excluded['reference_not_available']+=1
                else:
                    simulations['reference']=mini(ref)
                    for mode in MODES[1:]:
                        result=simulate_build16(hid,aid,history,bool(g.neutral_site),
                               runs=runs,seed=seed,mismatch_research=mode)
                        if result.get('simulation') is None:
                            excluded[mode+'_missing_history']+=1
                            detail=result.get('mismatch_research') or {}
                            for side in ('home','away'):
                                reason=(detail.get(side) or {}).get('status')
                                if reason and reason!='AVAILABLE_UNFITTED':
                                    excluded['script_data_'+str(reason)] += 1
                            continue
                        simulations[mode]=mini(result)
                        simulations[mode]['evidence']=result['mismatch_research']
                    if all(m in simulations for m in ('reference','severity','negative')):
                        chosen.append({'game_id':str(g.game_id),'kickoff':kickoff.isoformat(),
                            'severity_max_pregame':simulations['severity']['evidence']['max_joint_severity'],
                            'severity_asymmetry_pregame':simulations['severity']['evidence']['difference_joint_severity'],
                            'home':str(g.home_team),'away':str(g.away_team),
                            'prior_home_games':len(history[hid]),
                            'prior_away_games':len(history[aid]),
                            'simulated':simulations,
                            'actual_home_margin':float(g.home_points-g.away_points),
                            'actual_total':float(g.home_points+g.away_points)})
            elif len(chosen)<max_games:excluded['insufficient_prior']+=1
            part=plays_by_id.get(g.game_id)
            if part is None:
                excluded['no_pbp']+=1;continue
            c_home=extract(part,hid,aid)
            c_away=extract(part,aid,hid)
            if c_home.get('status')!='OBSERVED_VALID' or c_away.get('status')!='OBSERVED_VALID':
                excluded['rush_severity_input_incomplete']+=1
            h=observed_game(part,hid,aid,g.home_points,g.away_points,qb.get(hid,set()),
                            special_plays=raw_by_id.get(g.game_id))
            a=observed_game(part,aid,hid,g.away_points,g.home_points,qb.get(aid,set()),
                            special_plays=raw_by_id.get(g.game_id))
            if h is None or a is None:continue
            append_training_record(h,c_home)
            append_training_record(a,c_away)
            # Any target game's history is inaccessible before 24h after
            # kickoff. For official third-box tests game outcome is never fit.
            pending.extend(((kickoff+pd.Timedelta(hours=24),hid,h),
                            (kickoff+pd.Timedelta(hours=24),aid,a)))
        if len(chosen)>=max_games:break
    # These subgroup choices are functions of PRE-KICKOFF features only.
    # The quartile cutpoint is reported, not claimed as an untouched holdout.
    max_signal=[r['severity_max_pregame'] for r in chosen]
    cutoff_q75=float(np.quantile(max_signal,.75)) if chosen else None
    strong=[r for r in chosen if r['severity_max_pregame']>=.08]
    top_quartile=[r for r in chosen if r['severity_max_pregame']>=cutoff_q75] if chosen else []
    others=[r for r in chosen if r['severity_max_pregame']<cutoff_q75] if chosen else []
    report={'schema':'cfb_failure1_severity_negative_script_experiment_v1',
      'snapshot_captured_at_utc':str(snapshot_at),
      'source':'Free SportsDataverse 2026 ESPN PBP',
      'modes':MODES,'runs_per_game_per_mode':runs,
      'studied_games':len(chosen),'exclusions':dict(excluded),
      'feature_policy':'pre-play-score state only, missing never tied; designed rush excluding sacks',
      'past_games_after_2026_10_20_filtered':True,
      'motivation_game_included':False,
      'predeclared_strong_joint_severity_cutoff':.08,
      'sample_quartile_cutoff':round(cutoff_q75,6) if cutoff_q75 is not None else None,
      'cohorts':{'all':paired_score(chosen),
                 'strict_08_severity':paired_score(strong),
                 'top_quartile_exante_signal':paired_score(top_quartile),
                 'other_three_quartiles':paired_score(others),
                 'with_resilience_verified':paired_score(chosen,('reference','resilience','combined')),
                 'severe_with_resilience_verified':paired_score(strong,('reference','resilience','combined'))},
      'records':chosen,'deploy':False,'ATS_calibrated':False,
      'coefficients_fitted':False,
      'limitations':['Retrospective dataset release, historical row publication times not independently authenticated',
                     'Mode uses unfitted hypotheses; no score-line timing or true pregame market odds',
                     'Conditional rushing trailing-state samples may be sparse and selected',
                     'Severe matchup quantile is picked from ex-ante features, not results',
                     'Existing Build16 includes prior rush EPA/third down adjustments; avoid double-counting',
                     'FBS vs FCS permitted but FCS vs FCS never considered',
                     'No automatic promotion or additional competing active challenger']}
    dst=Path(out);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():raise FileExistsError('Research output immutable')
    dst.write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n')
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True)
    p.add_argument('--max-games',type=int,default=180)
    p.add_argument('--runs',type=int,default=5000)
    args=p.parse_args()
    r=study(args.snapshot,args.out,runs=args.runs,max_games=args.max_games)
    print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2,default=str))

if __name__=='__main__':main()
