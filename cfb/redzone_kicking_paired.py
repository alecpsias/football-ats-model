"""Chronological 5,000-draw same-seed Build16 FG/red-zone paired audit.

The historical reference is today's Build16 *with* the default third-down and
rushing interactions, but redzone_kicking_enabled=False. Every other flag is
identical. The study is diagnostic, not a deployment gate or an ATS claim.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from cfb.pipeline import verify, iso_utc, normalize_pbp_teams
from cfb.research_backtest import observed_game
from cfb.mechanism_build16 import simulate_build16
from cfb.chain_source import observed_chain, append_historical_chain

def _data(sim):
    r=sim['simulation']
    return {'mean_margin':r['mean_margin_home'],'total':r['mean_total'],
            'mean_home':r['mean_score_home'],'mean_away':r['mean_score_away'],
            'win_probability':r['home_win_probability_raw'],
            'p10p90':r['margin_p10_p50_p90'],'simulations':r['simulations'],
            'field_goal_evidence':sim.get('redzone_kicking_evidence')}

def study(snapshot,out,*,runs=5000,max_games=160):
    if runs != 5000:
        raise ValueError('Exactly 5,000 paired simulations required')
    root=Path(snapshot)
    manifest=verify(root)
    capture=iso_utc(manifest['captured_at_utc'])
    sched=pd.read_parquet(root/'schedule.parquet')
    pbp=pd.read_parquet(root/'pbp.parquet')
    roster=pd.read_parquet(root/'roster.parquet')
    sched['_kickoff']=pd.to_datetime(sched.start_date,utc=True,errors='coerce')
    # No retrospective look at the motivating Oct 22 JMU/App State outcome.
    eligible=(sched.fbs_participant.fillna(False).eq(True)&
      sched.completed.fillna(False).eq(True)&
      (sched['_kickoff']+pd.Timedelta(hours=24)<=capture)&
      sched['_kickoff'].lt(pd.Timestamp('2026-10-22',tz='UTC'))&
      ~((sched.home_division=='fcs')&(sched.away_division=='fcs'))&
      sched.home_points.notna()&sched.away_points.notna())
    sched=sched.loc[eligible].sort_values(['_kickoff','game_id'])
    mapped=normalize_pbp_teams(pbp.loc[pbp.game_id.isin(sched.game_id)],sched)
    by_game={i:g for i,g in mapped.groupby('game_id',sort=False)}
    raw_game={i:g for i,g in pbp.loc[pbp.game_id.isin(sched.game_id)].groupby('game_id',sort=False)}
    qb={int(t):set(g.athlete_id.dropna()) for t,g in
        roster.loc[roster.position_abbreviation.eq('QB')].groupby('team_id')}
    histories=defaultdict(list);pending=[];paired=[];excluded=defaultdict(int)
    for kickoff,block in sched.groupby('_kickoff',sort=True):
        ready=[x for x in pending if x[0]<=kickoff]
        pending=[x for x in pending if x[0]>kickoff]
        for _,team,rec in ready: histories[team].append(rec)
        for _,game in block.iterrows():
            hid,aid=int(game.home_id),int(game.away_id)
            if len(paired)<max_games and min(len(histories[hid]),len(histories[aid]))>=3:
                seed=int(game.game_id)%(2**32)
                args=(hid,aid,histories,bool(game.neutral_site))
                previous=simulate_build16(*args,runs=runs,seed=seed,
                                          redzone_kicking_enabled=False)
                current=simulate_build16(*args,runs=runs,seed=seed,
                                         redzone_kicking_enabled=True)
                if previous.get('simulation') is None or current.get('simulation') is None:
                    excluded['unavailable_projection']+=1
                else:
                    paired.append({'game_id':str(game.game_id),
                        'kickoff_utc':kickoff.isoformat(),
                        'actual_margin':float(game.home_points-game.away_points),
                        'actual_total':float(game.home_points+game.away_points),
                        'reference':_data(previous),'updated':_data(current)})
            else: excluded['insufficient_or_sample_full']+=1
            play=by_game.get(game.game_id)
            if play is None:
                excluded['missing_pbp']+=1;continue
            raw=raw_game.get(game.game_id)
            h=observed_game(play,hid,aid,game.home_points,game.away_points,
                            qb.get(hid,set()),special_plays=raw,home_id=hid)
            a=observed_game(play,aid,hid,game.away_points,game.home_points,
                            qb.get(aid,set()),special_plays=raw,home_id=hid)
            if h is None or a is None:
                excluded['invalid_observed_game']+=1;continue
            for row,team,opp in ((h,hid,aid),(a,aid,hid)):
                chain=observed_chain(play,team,opp)
                if chain.get('status')=='OBSERVED_QUALIFIED':
                    append_historical_chain(row,chain,'offense')
                    append_historical_chain(row,chain,'opponent')
            # 24h history embargo; no simultaneous-game or target observation.
            pending.extend(((kickoff+pd.Timedelta(hours=24),hid,h),
                            (kickoff+pd.Timedelta(hours=24),aid,a)))
        if len(paired)>=max_games: break
    actual=np.array([r['actual_margin'] for r in paired],dtype=float)
    total=np.array([r['actual_total'] for r in paired],dtype=float)
    metrics={}
    for mode in ('reference','updated'):
        m=np.array([r[mode]['mean_margin'] for r in paired])
        t=np.array([r[mode]['total'] for r in paired])
        metrics[mode]={'margin_mae':round(float(np.abs(m-actual).mean()),5) if len(m) else None,
            'margin_rmse':round(float(np.sqrt(np.mean((m-actual)**2))),5) if len(m) else None,
            'total_mae':round(float(np.abs(t-total).mean()),5) if len(t) else None,
            'margin_bias':round(float(np.mean(m-actual)),5) if len(m) else None}
    if paired:
        diff=np.array([abs(x['reference']['mean_margin']-x['actual_margin'])-
                       abs(x['updated']['mean_margin']-x['actual_margin']) for x in paired])
        rng=np.random.default_rng(20261023)
        sel=rng.integers(0,len(diff),size=(2500,len(diff)))
        metrics['paired_margin_mae_improvement']=round(float(diff.mean()),5)
        metrics['paired_bootstrap95']=list(map(lambda x:round(float(x),5),
                                               np.quantile(diff[sel].mean(axis=1),[.025,.975])))
    report={'schema':'cfb_build16_redzone_kicking_paired_v1','games':len(paired),
      'runs_per_game_per_model':runs,'snapshot_captured_at':str(capture),
      'reference':'Current Build16 with redzone_kicking_enabled=False',
      'updated':'Same Build16 with redzone_kicking_enabled=True',
      'both_modes_keep_active_third_down_and_rush_corrections':True,
      'future_games_excluded_from_features':True,'motivation_fixture_excluded':True,
      'betting_lines_verified':False,'ATS_accuracy_calibrated':False,
      'metrics':metrics,'excluded':dict(excluded),'games_data':paired}
    dest=Path(out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): raise FileExistsError('Immutable output already exists')
    dest.write_text(json.dumps(report,indent=2,allow_nan=False,default=str)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--runs',type=int,default=5000)
    parser.add_argument('--max-games',type=int,default=160)
    args=parser.parse_args()
    result=study(args.snapshot,args.out,runs=args.runs,max_games=args.max_games)
    print(json.dumps({k:v for k,v in result.items() if k!='games_data'},indent=2))
