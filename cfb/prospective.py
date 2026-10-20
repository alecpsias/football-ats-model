"""Freeze independent CFB pre-kickoff margin research forecasts from a verified snapshot.

NO NFL dependency. NO odds/ATS pick without a separately archived pregame quote.
Predictions are timestamped, deterministic, tied to source SHA256, and immutable
within the output folder. GitHub Actions artifacts themselves eventually expire;
download/export to durable storage independently of this code.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from cfb.engine import simulate, CFB_RULES
from cfb.matchups import matchup_signals
from cfb.advanced_scoring import simulate_challenger
from cfb.pipeline import verify, split_games, normalize_pbp_teams, iso_utc
from cfb.research_backtest import observed_game, project


def _json_bytes(obj):
    return (json.dumps(obj,sort_keys=True,separators=(',',':'),default=str)+'\n').encode()


def freeze(snapshot, out, runs=3000, days=7, min_games=3, now=None,
           git_sha=None, workflow_run_id=None):
    """Generate a frozen RESEARCH forecast artifact for each eligible future game.

    For a game with insufficient past-game records, create a frozen PASS record
    rather than extrapolating invented statistics.
    """
    if type(runs) is not int or not 100 <= runs <= 300000:raise ValueError('Invalid CFB run count')
    if type(days) is not int or not 1 <= days <= 31:raise ValueError('Invalid CFB window')
    if type(min_games) is not int or not 1 <= min_games <= 15:raise ValueError('Invalid minimum games')
    folder=Path(snapshot)
    manifest=verify(folder)
    cutoff=iso_utc(manifest['captured_at_utc'])
    stamp=iso_utc(now if now is not None else datetime.now(timezone.utc))
    if stamp < cutoff:raise ValueError('Prediction generation predates actual snapshot capture')
    schedule=pd.read_parquet(folder/'schedule.parquet')
    pbp=pd.read_parquet(folder/'pbp.parquet')
    roster=pd.read_parquet(folder/'roster.parquet')
    history, future, past_plays=split_games(schedule,pbp,cutoff)
    # A previous kickoff alone does NOT prove final data existed by forecast time.
    embargo=timedelta(hours=24)
    history=history.loc[history['_kickoff']+embargo <= cutoff].copy()
    mapped=normalize_pbp_teams(past_plays.loc[past_plays.game_id.isin(history.game_id)],history)
    by_game={gid:part for gid,part in mapped.groupby('game_id',sort=False)}
    qb_ids={int(team):set(group.athlete_id.dropna()) for team,group in
            roster.loc[roster.position_abbreviation.eq('QB')].groupby('team_id')}
    histories=defaultdict(list)
    for _,g in history.iterrows():
        part=by_game.get(g.game_id)
        if part is None or pd.isna(g.home_points) or pd.isna(g.away_points):
            continue
        h,a=int(g.home_id),int(g.away_id)
        ho=observed_game(part,h,a,g.home_points,g.away_points,qb_ids.get(h,set()))
        aw=observed_game(part,a,h,g.away_points,g.home_points,qb_ids.get(a,set()))
        if ho is None or aw is None:continue
        ho['opponent_id']=a
        aw['opponent_id']=h
        histories[h].append(ho);histories[a].append(aw)
    destination=Path(out)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Refusing to overwrite frozen CFB predictions')
    destination.mkdir(parents=True,exist_ok=True)
    manifest_sha=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()
    rows=[]
    deadline=min(cutoff+timedelta(days=days),stamp+timedelta(days=days))
    candidates=future.loc[(future['_kickoff']>stamp)&(future['_kickoff']<=deadline)].sort_values(['_kickoff','game_id'])
    for _,g in candidates.iterrows():
        hid,aid=int(g.home_id),int(g.away_id)
        base={'schema':'cfb_prospective_research_forecast_v1',
              'game_id':str(g.game_id),'home':str(g.home_team),'away':str(g.away_team),
              'home_id':str(hid),'away_id':str(aid),
              'kickoff_utc':g['_kickoff'].isoformat(),
              'prediction_generated_at_utc':stamp.isoformat(),
              'source_snapshot_at_utc':cutoff.isoformat(),
              'source_snapshot_manifest_sha256':manifest_sha,
              'source_sha256':{k:v['sha256'] for k,v in manifest['datasets'].items()},
              'git_commit_sha':git_sha,'github_run_id':str(workflow_run_id) if workflow_run_id else None,
              'engine':'cfb.engine.v1','model_variant':'UNMODIFIED_UNCALIBRATED_BASELINE',
              'market_quote':None,'ats_prediction':None,'wager_grade':None,
              'trained_with_games_at_or_after_target_kickoff':False,
              'historical_source_publication_times_verified':False,
              'status':'INSUFFICIENT_OBSERVED_GAMES_RESEARCH_PASS',
              'limitations':['Research-only; no empirically validated ATS cover model',
                 'Earlier-game source rows lack point-in-time publication timestamps',
                 'No independently archived signed pregame spread',
                 'Later outcomes must never alter this frozen pregame file']}
        if len(histories[hid])>=min_games and len(histories[aid])>=min_games:
            neutral=(bool(g.neutral_site) if pd.notna(g.neutral_site) else False)
            hp=project(histories[hid],histories[aid],neutral,'home',False)
            ap=project(histories[aid],histories[hid],neutral,'away',False)
            sim=simulate(hp,ap,runs=runs,seed=int(g.game_id)%(2**32))
            signals=matchup_signals(hid,aid,histories)
            advanced=simulate_challenger(hp,ap,signals,len(histories[hid]),
                        len(histories[aid]),runs=runs,seed=int(g.game_id)%(2**32))
            base.update({'advanced_research_challenger':advanced,
                         'advanced_challenger_not_primary':True})
            base.update({'status':'FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE',
                'completed_prior_games_home':len(histories[hid]),
                'completed_prior_games_away':len(histories[aid]),
                'home_projection':hp,'away_projection':ap,
                'matchup_evidence':signals,
                'simulated_margin_home':sim['mean_margin_home'],
                'simulated_score_home':sim['mean_score_home'],
                'simulated_score_away':sim['mean_score_away'],
                'simulation':sim})
        else:
            base.update({'completed_prior_games_home':len(histories[hid]),
                         'completed_prior_games_away':len(histories[aid])})
        if now is None and datetime.now(timezone.utc) >= g['_kickoff'].to_pydatetime():
            # Fail closed rather than publish a retrospective 'pregame' forecast.
            raise ValueError('Fixture kicked off during forecast generation; refusing freeze')
        raw=_json_bytes(base)
        record_sha=hashlib.sha256(raw).hexdigest()
        filename=str(g.game_id)+'.json'
        (destination/filename).write_bytes(raw)
        rows.append({'game_id':base['game_id'],'file':filename,'sha256':record_sha,
                     'home':base['home'],'away':base['away'],
                     'kickoff_utc':base['kickoff_utc'],'status':base['status']})
    summary={'status':'CFB_PROSPECTIVE_RESEARCH_RECORDS_FROZEN',
             'generated_at_utc':stamp.isoformat(),'snapshot_at_utc':cutoff.isoformat(),
             'model_variant':'UNMODIFIED_UNCALIBRATED_BASELINE',
             'game_records':len(rows),
             'margin_projections':sum(r['status']=='FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE' for r in rows),
             'insufficient_data':sum(r['status']=='INSUFFICIENT_OBSERVED_GAMES_RESEARCH_PASS' for r in rows),
             'predictions':rows,
             'source_manifest_sha256':manifest_sha,
             'github_run_id':str(workflow_run_id) if workflow_run_id else None,
             'git_commit_sha':git_sha,
             'retention_warning':'GitHub Actions uploaded artifacts expire. Download/store immutable files or hash-identical archive elsewhere before expiry.',
             'ats_validated':False}
    (destination/'index.json').write_bytes(_json_bytes(summary))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True)
    p.add_argument('--runs',type=int,default=3000);p.add_argument('--days',type=int,default=7)
    p.add_argument('--git-sha');p.add_argument('--github-run-id')
    args=p.parse_args()
    result=freeze(args.snapshot,args.out,args.runs,args.days,
        git_sha=args.git_sha,workflow_run_id=args.github_run_id)
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2))
if __name__=='__main__':main()
