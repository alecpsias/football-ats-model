"""Chronological CFB rush possession evidence -> exact 5000-run holdout study.

Train from 2023 only; subsequent seasons may NEVER update coefficient. Valid
past games must predate target by 24h; same-kickoff games quarantined.
Cross-validated possession durations come only from source-regulation clocks.
No FCS-v-FCS, no named motivating NM State-Sam Houston game leakage.
"""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
from cfb.pipeline import verify,normalize_pbp_teams,iso_utc
from cfb.research_backtest import observed_game
from cfb.chain_source import observed_chain,append_historical_chain
from cfb.clock_possession_validation import audit_game
from cfb.rush_possession_response import signal,fit,response
from cfb.mechanism_build16 import simulate_build16
EXCLUDE=401871070

def metric(rows):
    r={'games':len(rows)}
    if not rows:return r
    y=np.asarray([a['actual_margin'] for a in rows],float)
    base=np.asarray([a['reference']['margin'] for a in rows],float)
    pp=np.asarray([a['candidate']['margin'] for a in rows],float)
    brier=lambda k:float(np.mean([(a[k]['p_home']-float(a['actual_margin']>0))**2 for a in rows]))
    err=np.abs(base-y)-np.abs(pp-y)
    rng=np.random.default_rng(20261022)
    ix=rng.integers(0,len(rows),size=(2000,len(rows)))
    refq=np.array([a['reference']['quantiles'] for a in rows])
    newq=np.array([a['candidate']['quantiles'] for a in rows])
    r.update({'reference_margin_mae':round(float(abs(base-y).mean()),5),
      'candidate_margin_mae':round(float(abs(pp-y).mean()),5),
      'margin_MAE_gain':round(float(err.mean()),5),
      'paired_bootstrap95':np.quantile(err[ix].mean(axis=1),[.025,.975]).round(5).tolist(),
      'reference_winner_brier':round(brier('reference'),6),
      'candidate_winner_brier':round(brier('candidate'),6),
      'reference_margin80_coverage':round(float(np.mean((y>=refq[:,0])&(y<=refq[:,2]))),5),
      'candidate_margin80_coverage':round(float(np.mean((y>=newq[:,0])&(y<=newq[:,2]))),5),
      'reference_winner_accuracy':round(float(np.mean(np.sign(base)==np.sign(y))),5),
      'candidate_winner_accuracy':round(float(np.mean(np.sign(pp)==np.sign(y))),5)})
    return r

def study(snapshot,out,*,runs=5000,max_games=180,fitted=None):
    if runs!=5000:raise ValueError('All matchup simulations must be 5000 draws')
    if not isinstance(max_games,int) or not 1<=max_games<=300:raise ValueError('Invalid requested sample')
    root=Path(snapshot);manifest=verify(root)
    cutoff=iso_utc(manifest['captured_at_utc'])
    sched=pd.read_parquet(root/'schedule.parquet')
    pbp=pd.read_parquet(root/'pbp.parquet')
    roster=pd.read_parquet(root/'roster.parquet')
    sched['_kickoff']=pd.to_datetime(sched.start_date,utc=True,errors='coerce')
    valid=(sched.fbs_participant.fillna(False).eq(True)&
          sched.completed.fillna(False).eq(True)&
          (sched['_kickoff']+pd.Timedelta(hours=24)<=cutoff)&
          ~((sched.home_division=='fcs')&(sched.away_division=='fcs'))&
          sched.home_points.notna()&sched.away_points.notna()&
          sched.game_id.ne(EXCLUDE))
    if int(sched['_kickoff'].dt.year.mode()[0])==2026:
        valid &= sched['_kickoff'].lt(pd.Timestamp('2026-10-21',tz='UTC'))
    games=sched.loc[valid].sort_values(['_kickoff','game_id'])
    if games.empty:raise ValueError('No eligible true final-score games')
    mapped=normalize_pbp_teams(pbp.loc[pbp.game_id.isin(games.game_id)],games)
    norm={i:g for i,g in mapped.groupby('game_id',sort=False)}
    raw={i:g for i,g in pbp.loc[pbp.game_id.isin(games.game_id)].groupby('game_id',sort=False)}
    qbs={int(t):set(g.athlete_id.dropna()) for t,g in
         roster.loc[roster.position_abbreviation.eq('QB')].groupby('team_id')}
    hist=defaultdict(list);pending=[];rows=[];fail=defaultdict(int);source_status=defaultdict(int)
    for kickoff,block in games.groupby('_kickoff',sort=True):
        eligible=[z for z in pending if z[0]<=kickoff]
        pending=[z for z in pending if z[0]>kickoff]
        for _,tid,rec in eligible:hist[tid].append(rec)
        for _,game in block.iterrows():
            hid,aid=int(game.home_id),int(game.away_id)
            if min(len(hist[hid]),len(hist[aid]))>=3 and len(rows)<max_games:
                seed=int(game.game_id)%(2**32)
                snap=signal(hid,aid,hist)
                if snap['status']=='AVAILABLE_PREGAME_DRIVE_CONTINUATION':
                    default=simulate_build16(hid,aid,hist,bool(game.neutral_site),
                                             seed=seed,runs=runs)
                    if default.get('simulation') is None:fail['build16_missing']+=1
                    else:
                        def record(x):
                            z=x['simulation']
                            return {'margin':z['mean_margin_home'],
                                    'total':z['mean_total'],
                                    'p_home':z['home_win_probability_raw'],
                                    'quantiles':z['margin_p10_p50_p90'],
                                    'simulations':z['simulations'],'seed':z['seed']}
                        ref=record(default)
                        one={'game_id':str(game.game_id),'year':kickoff.year,
                           'kickoff':kickoff.isoformat(),'home':str(game.home_team),
                           'away':str(game.away_team),'prior_home_games':len(hist[hid]),
                           'prior_away_games':len(hist[aid]),
                           'actual_margin':float(game.home_points-game.away_points),
                           'actual_total':float(game.home_points+game.away_points),
                           'reference_margin':ref['margin'],
                           'reference_expected_drives':.5*(default['home']['projection']['expected_drives']+
                                                         default['away']['projection']['expected_drives']),
                           'signal':snap,'reference':ref,'candidate':None}
                        if fitted is not None:
                            y=response(fitted,snap)
                            cand=simulate_build16(hid,aid,hist,bool(game.neutral_site),
                                  seed=seed,runs=runs,possession_response=y)
                            if cand.get('simulation') is None:fail['candidate_missing']+=1
                            else:
                                one['candidate']=record(cand)
                                one['possession_adjustment']=y
                        if fitted is None or one['candidate'] is not None:
                            rows.append(one)
                else:fail['missing_verified_chains']+=1
            elif len(rows)<max_games:fail['too_few_prior_games']+=1
            part=norm.get(game.game_id);original=raw.get(game.game_id)
            if part is None or original is None:
                fail['pbp_missing']+=1;continue
            chain_h=observed_chain(part,hid,aid)
            chain_a=observed_chain(part,aid,hid)
            clock=audit_game(original,game.game_id,hid,aid)
            source_status[clock['status']]+=1
            ho=observed_game(part,hid,aid,game.home_points,game.away_points,qbs.get(hid,set()),special_plays=original)
            aw=observed_game(part,aid,hid,game.away_points,game.home_points,qbs.get(aid,set()),special_plays=original)
            if ho is None or aw is None:fail['observed_game_missing']+=1;continue
            for own,other,observation,tid,op in ((ho,aw,chain_h,hid,aid),(aw,ho,chain_a,aid,hid)):
                if observation['status']=='OBSERVED_QUALIFIED':
                    append_historical_chain(own,observation,'offense')
                    append_historical_chain(own,observation,'opponent')
                else:fail['chain_source_'+observation['status']]+=1
                if clock['status']=='COMPLETE_REGULATION_CLOCK_AUDIT':
                    ot=clock['teams'].get(str(tid));dt=clock['teams'].get(str(op))
                    if ot and dt:
                        own['possession_observed_time_sec']=ot['observed_time_seconds']
                        own['possession_verified_drive_n']=ot['valid_drives']
                        own['possession_def_observed_time_sec']=dt['observed_time_seconds']
                        own['possession_def_verified_drive_n']=dt['valid_drives']
            ho['opponent_id']=aid;aw['opponent_id']=hid
            pending.extend(((kickoff+pd.Timedelta(hours=24),hid,ho),
                            (kickoff+pd.Timedelta(hours=24),aid,aw)))
        if len(rows)>=max_games:break
    year=int(games['_kickoff'].dt.year.mode()[0])
    scored=[r for r in rows if r['candidate'] is not None]
    summary={'schema':'cfb_possession_rush_full_matchup_validation_v1',
      'season':year,'source_captured_at_utc':str(cutoff),
      'pairs_simulated_5000_per_variant':len(rows),
      'training_only':fitted is None,
      'observed_clock_audit_status_counts':dict(source_status),
      'exclusion_counts':dict(fail),
      'matched_verified_clock_pairs':sum(r['signal']['clock_both_sides_qualified'] for r in rows),
      'scored_metrics':{'all':metric(scored),
         'clock_verified':metric([r for r in scored if r['signal']['clock_both_sides_qualified']]),
         'severe_rush_advantage':metric([r for r in scored if
             max(abs(r['signal']['home_continuation_delta']),abs(r['signal']['away_continuation_delta']))>=.02])},
      'study_rows':rows,'baseline_changed':False,'automatic_promotion':False,
      'asof_statement':'24h historical embargo, retrospective source timestamp not proof each past source row was live at past kickoff',
      'no_bet_spreads_unless_captured_before_kickoff':True}
    dst=Path(out);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():raise FileExistsError('Immutable study output')
    dst.write_text(json.dumps(summary,indent=2,default=str,allow_nan=False)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True)
    p.add_argument('--out',required=True);p.add_argument('--frozen-fit')
    p.add_argument('--runs',type=int,default=5000);p.add_argument('--max-games',type=int,default=180)
    a=p.parse_args()
    fit_data=json.loads(Path(a.frozen_fit).read_text()) if a.frozen_fit else None
    data=study(a.snapshot,a.out,runs=a.runs,max_games=a.max_games,fitted=fit_data)
    print(json.dumps({k:v for k,v in data.items() if k!='study_rows'},indent=2))
