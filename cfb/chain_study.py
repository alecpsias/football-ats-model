"""Chronological, paired CFB rush -> third-down -> sustained-drives pilot.

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
from cfb.chain_source import observed_chain,append_historical_chain

MODES=('reference','distance','sustain','combined')
EXCLUDE_GAME_ID=401871070

def mini(value):
    z=value['simulation']
    return {'margin':z['mean_margin_home'],'total':z['mean_total'],
            'win_probability_raw':z['home_win_probability_raw'],
            'margin_p10_p50_p90':z['margin_p10_p50_p90'],
            'simulation_count':z['simulations'],'seed':z['seed']}

def paired_score(rows):
    ret={'paired_games':len(rows)}
    if not rows:return ret
    actual=np.array([r['actual_home_margin'] for r in rows],float)
    total=np.array([r['actual_total'] for r in rows],float)
    win=(actual>0).astype(float)
    for mode in MODES:
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
    for mode in MODES[1:]:
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

def study(snapshot,out,*,runs=5000,max_games=160,min_prior_games=3):
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
        sched.game_id.ne(EXCLUDE_GAME_ID))
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
                               runs=runs,seed=seed,chain_research=mode)
                        if result.get('simulation') is None:
                            excluded[mode+'_missing_history']+=1;break
                        simulations[mode]=mini(result)
                        simulations[mode]['evidence']=result['chain_research']
                    if len(simulations)==len(MODES):
                        chosen.append({'game_id':str(g.game_id),'kickoff':kickoff.isoformat(),
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
            c_home=observed_chain(part,hid,aid)
            c_away=observed_chain(part,aid,hid)
            if c_home.get('status')=='OBSERVED_QUALIFIED' and c_away.get('status')=='OBSERVED_QUALIFIED':
                for tid,own in ((hid,c_home),(aid,c_away)):
                    # Capture third-attempts/known-conversions separately for
                    # official audit. NO target outcome is used as predictor.
                    v=own['offense']
                    if int(g.game_id) in (401864494,401871070):
                        third_box.append({'game_id':str(g.game_id),'team_id':tid,
                             'third_attempts_pbp':v['third_attempts'],
                             'third_resolved_n':v['third_resolved_n'],
                             'third_conversion_yes_pbp':v['third_conversion_yes'],
                             'terminal_unresolved':v['third_unresolved_n']})
            else:excluded['chain_unavailable']+=1
            h=observed_game(part,hid,aid,g.home_points,g.away_points,qb.get(hid,set()),
                            special_plays=raw_by_id.get(g.game_id))
            a=observed_game(part,aid,hid,g.away_points,g.home_points,qb.get(aid,set()),
                            special_plays=raw_by_id.get(g.game_id))
            if h is None or a is None:continue
            if c_home.get('status')=='OBSERVED_QUALIFIED':
                append_historical_chain(h,c_home,'offense')
                append_historical_chain(h,c_home,'opponent')
            if c_away.get('status')=='OBSERVED_QUALIFIED':
                append_historical_chain(a,c_away,'offense')
                append_historical_chain(a,c_away,'opponent')
            # Any target game's history is inaccessible before 24h after
            # kickoff. For official third-box tests game outcome is never fit.
            pending.extend(((kickoff+pd.Timedelta(hours=24),hid,h),
                            (kickoff+pd.Timedelta(hours=24),aid,a)))
        if len(chosen)>=max_games:break
    report={'schema':'cfb_third_drive_chain_matched_ablation_v1',
       'snapshot_captured_at_utc':str(snapshot_at),
       'source':'Free SportsDataverse 2026 ESPN PBP',
       'modes':MODES,'runs_per_game_per_mode':runs,
       'excluded_causal_motivation_game_id':str(EXCLUDE_GAME_ID),
       'games_eligible':len(chosen),'exclusions':dict(excluded),
       'third_down_source_checks':third_box,
       'paired_metrics':paired_score(chosen),'records':chosen,
       'deploy':False,'ATS_calibrated':False,'coefficients_empirically_fit':False,
       'limitations':['Retrospective snapshot; historical row publication unverified',
           'Third-down terminal plays may be unresolved; source counts are audited',
           'Drive chains are operationally >=3 first-down series, not possession seconds',
           'Existing challenger already includes rush EPA and third-down conversion residuals',
           'No timestamped odds / no ATS validation; no forward untouched confirmation']}
    dst=Path(out);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():raise FileExistsError('Research output immutable')
    dst.write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n')
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True)
    p.add_argument('--max-games',type=int,default=160)
    p.add_argument('--runs',type=int,default=5000)
    args=p.parse_args()
    r=study(args.snapshot,args.out,runs=args.runs,max_games=args.max_games)
    print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2,default=str))

if __name__=='__main__':main()
