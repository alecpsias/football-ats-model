"""Chronological paired CFB rushing-ablation RESEARCH study.

No production forecasting paths are touched. The same historical game, seed,
5,000 Monte Carlo draws and pre-kickoff/24-hour history are used for the
unchanged Build16 challenger versus its three OPTIONAL within-model ablations.
This is retrospective from a later captured source, not proven as-of or ATS.
An outcome cannot affect which experiment is run or its coefficients.
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
from cfb.clock_possession_validation import audit_game

EXPERIMENTS=('reference','mix','pace','combined')


def paired_metrics(rows):
    """Paired metrics for IDENTICAL games, not unequal available cohorts."""
    out={'games':len(rows)}
    if not rows:return out
    y=np.array([r['actual_home_margin'] for r in rows],float)
    totals=np.array([r['actual_total'] for r in rows],float)
    homewins=(y>0).astype(float)
    for mode in EXPERIMENTS:
        predicted=np.array([r['experiments'][mode]['margin'] for r in rows],float)
        ptot=np.array([r['experiments'][mode]['total'] for r in rows],float)
        win=np.array([r['experiments'][mode]['win_probability_raw'] for r in rows],float)
        lower=np.array([r['experiments'][mode]['p10_p50_p90'][0] for r in rows],float)
        upper=np.array([r['experiments'][mode]['p10_p50_p90'][2] for r in rows],float)
        out[mode]={'margin_mae':round(float(np.mean(np.abs(predicted-y))),4),
                   'margin_bias':round(float(np.mean(predicted-y)),4),
                   'total_mae':round(float(np.mean(np.abs(ptot-totals))),4),
                   'outright_brier':round(float(np.mean((win-homewins)**2)),5),
                   'margin_p10_p90_coverage':round(float(np.mean((y>=lower)&(y<=upper))),4),
                   'margin_direction_accuracy':round(float(np.mean(
                      np.sign(predicted[y!=0])==np.sign(y[y!=0]))),4) if (y!=0).any() else None,
                   'mean_regulation_drives_per_side':round(float(np.mean(
                      [r['experiments'][mode]['drives'] for r in rows])),4)}
    baseline=np.array([abs(r['experiments']['reference']['margin']-r['actual_home_margin'])
                       for r in rows],float)
    for mode in EXPERIMENTS[1:]:
        errors=np.array([abs(r['experiments'][mode]['margin']-r['actual_home_margin'])
                         for r in rows],float)
        diffs=baseline-errors # Positive = experimental improvement
        rng=np.random.default_rng(20261022)
        samples=rng.integers(0,len(rows),size=(2000,len(rows)))
        bootstrap=diffs[samples].mean(axis=1)
        out[mode]['paired_mae_improvement_vs_reference']=round(float(diffs.mean()),4)
        out[mode]['bootstrap95_improvement_interval']=[
            round(float(x),4) for x in np.quantile(bootstrap,[.025,.975])]
    return out


def _mini(result):
    sim=result['simulation']
    return {'margin':sim['mean_margin_home'],
            'total':sim['mean_total'],
            'p10_p50_p90':sim['margin_p10_p50_p90'],
            'win_probability_raw':sim['home_win_probability_raw'],
            'drives':sim['simulated_regulation_drives_per_side'],
            'seed':sim['seed'],'simulations':sim['simulations']}


def study(snapshot, out, *, runs=5000,max_games=24,min_games=3):
    if type(runs) is not int or runs!=5000:
        raise ValueError('CFB rushing study requires exactly 5000 draws per comparison')
    if type(max_games) is not int or not 1<=max_games<=200:
        raise ValueError('Invalid chronological study size')
    root=Path(snapshot);manifest=verify(root);cutoff=iso_utc(manifest['captured_at_utc'])
    schedule=pd.read_parquet(root/'schedule.parquet')
    pbp=pd.read_parquet(root/'pbp.parquet')
    roster=pd.read_parquet(root/'roster.parquet')
    schedule['_kickoff']=pd.to_datetime(schedule.start_date,utc=True,errors='coerce')
    eligible=(schedule.fbs_participant.fillna(False).eq(True)&
              schedule.completed.fillna(False).eq(True)&
              (schedule['_kickoff']+pd.Timedelta(hours=24)<=cutoff)&
              ~((schedule.home_division=='fcs')&(schedule.away_division=='fcs'))&
              schedule.home_points.notna()&schedule.away_points.notna())
    games=schedule.loc[eligible].sort_values(['_kickoff','game_id'])
    mapped=normalize_pbp_teams(pbp.loc[pbp.game_id.isin(games.game_id)],games)
    grouped={k:g for k,g in mapped.groupby('game_id',sort=False)}
    raw={k:g for k,g in pbp.loc[pbp.game_id.isin(games.game_id)].groupby('game_id',sort=False)}
    qb={int(t):set(g.athlete_id.dropna()) for t,g in
        roster.loc[roster.position_abbreviation.eq('QB')].groupby('team_id')}
    hist=defaultdict(list);pending=[];results=[];excluded=defaultdict(int)
    clock_statuses=defaultdict(int)
    # Historical game outcome is used ONLY as a post-prediction label.
    # A result cannot enter training histories until 24h after kickoff.
    for kickoff,block in games.groupby('_kickoff',sort=True):
        ready=[x for x in pending if x[0]<=kickoff]
        pending=[x for x in pending if x[0]>kickoff]
        for _,team,row in ready:hist[team].append(row)
        for _,g in block.iterrows():
            home,away=int(g.home_id),int(g.away_id)
            if min(len(hist[home]),len(hist[away]))>=min_games and len(results)<max_games:
                seed=int(g.game_id)%(2**32);chosen={}
                ref=simulate_build16(home,away,hist,bool(g.neutral_site),runs=runs,seed=seed)
                if ref.get('simulation') is None:
                    excluded['reference_evidence_missing']+=1
                else:
                    chosen['reference']=_mini(ref)
                    for mode in EXPERIMENTS[1:]:
                        val=simulate_build16(home,away,hist,bool(g.neutral_site),
                            runs=runs,seed=seed,rush_research=mode)
                        if val.get('simulation') is None:
                            excluded[mode+'_unavailable']+=1;break
                        chosen[mode]=_mini(val)
                        chosen[mode]['rush_research']=val['rush_research']
                    if len(chosen)==len(EXPERIMENTS):
                        results.append({'game_id':str(g.game_id),
                            'kickoff':kickoff.isoformat(),
                            'home':str(g.home_team),'away':str(g.away_team),
                            'prior_games_home':len(hist[home]),
                            'prior_games_away':len(hist[away]),
                            'experiments':chosen,
                            # target labels deliberately assigned AFTER predictions
                            'actual_home_margin':float(g.home_points-g.away_points),
                            'actual_total':float(g.home_points+g.away_points)})
            elif len(results)<max_games:
                excluded['insufficient_prior_games']+=1
            clock=audit_game(raw.get(g.game_id),g.game_id,home,away)
            clock_statuses[clock['status']]+=1
            part=grouped.get(g.game_id)
            if part is None:continue
            h=observed_game(part,home,away,g.home_points,g.away_points,qb.get(home,set()),
                            special_plays=raw.get(g.game_id))
            a=observed_game(part,away,home,g.away_points,g.home_points,qb.get(away,set()),
                            special_plays=raw.get(g.game_id))
            if h is not None and a is not None:
                if clock['status']=='COMPLETE_REGULATION_CLOCK_AUDIT' and clock['official_stat_classification_available']:
                    for tid,other,rec in ((home,away,h),(away,home,a)):
                        t=clock['teams'][str(tid)]
                        enemy=clock['teams'][str(other)]
                        rec.update({'clock_run_n':t['official_rush_attempts'],
                                    'clock_pass_n':t['official_pass_attempts'],
                                    'clock_def_run_n':enemy['official_rush_attempts'],
                                    'clock_def_pass_n':enemy['official_pass_attempts'],
                                    'clock_time_sum':t['observed_time_seconds'],
                                    'clock_drive_n':t['valid_drives'],
                                    'clock_def_time_sum':enemy['observed_time_seconds'],
                                    'clock_def_drive_n':enemy['valid_drives']})
                pending.extend(((kickoff+pd.Timedelta(hours=24),home,h),
                                (kickoff+pd.Timedelta(hours=24),away,a)))
        if len(results)>=max_games:break
    obj={'schema':'cfb_rush_interaction_chronological_research_v1',
         'status':'EXPLORATORY_NOT_PROSPECTIVE_NOT_ATS_VALIDATED',
         'snapshot_utc':str(cutoff),'simulations_per_model_per_game':runs,
         'sample_policy':'First eligible fixtures in chronological order, before inspecting actual margins',
         'NMSU_SHSU_GAME_USED_FOR_COEFFICIENT_FIT':False,
         'experiments':list(EXPERIMENTS),
         'evaluated_games':len(results),
         'excluded':dict(excluded),'clock_statuses':dict(clock_statuses),
         'paired':paired_metrics(results),
         'records':results,
         'deploy_automatically':False,
         'limitations':['2026 archived snapshot captured after historical games; row publication times not independently verified',
                        'No contemporaneous market quote means margin/winner assessment, NOT ATS',
                        'The rush ablation was proposed after NMSU-SHSU; retrospective cohort is exploratory, not untouched confirmation',
                        'Clock modes use source drive elapsed seconds, verified against regulation clock and capped for uncertain endpoints',
                        'Exactly one published CFB challenger; these are laboratory ablations only']}
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError('Refusing to overwrite rushing study audit')
    p.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
    return obj


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--runs',type=int,default=5000)
    p.add_argument('--max-games',type=int,default=24)
    args=p.parse_args()
    report=study(args.snapshot,args.out,runs=args.runs,max_games=args.max_games)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
