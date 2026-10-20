#!/usr/bin/env python3
"""Independent CFB chronological margin backtest, research only.

Uses CFB engine and CFB-only experimental rules. No imports from NFL model.
Historical replay limits *game inclusion* to games before each kickoff.
But a source snapshot captured AFTER the game cannot prove historical as-of
availability. Do NOT claim live/pregame out-of-sample ATS validation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.engine import CFB_RULES, simulate
from cfb.pipeline import verify, normalize_pbp_teams, iso_utc


def total(rows, key):
    return sum(float(row[key]) for row in rows)


def mean_epa(records, kind):
    count = total(records, kind + '_n')
    return total(records, kind + '_sum') / count if count else None


def observed_game(plays, team, opponent, points, conceded, qb_ids):
    """Read observed PBP per game without accessing future fixtures."""
    off = plays.loc[plays.pos_team_id.eq(team)]
    opp = plays.loc[plays.pos_team_id.eq(opponent)]
    if off.empty or opp.empty:
        return None

    def epa(rows, kind):
        x = pd.to_numeric(rows.loc[rows[kind].fillna(False).eq(True), 'EPA'],
                          errors='coerce').dropna()
        return float(x.sum()), len(x)

    op,pn = epa(off,'pass')
    or_,rn = epa(off,'rush')
    dp,dpn = epa(opp,'pass')
    dr,drn = epa(opp,'rush')
    drives = off.drive_id.nunique()
    faced = opp.drive_id.nunique()
    if not drives or not faced or pn + rn < 2 or dpn + drn < 2:
        return None

    ints = off['int']
    interceptions = int(ints.fillna(False).eq(True).sum()) if ints.notna().all() else None
    lost = (int(off.pos_fumble_lost.eq(True).sum())
            if 'pos_fumble_lost' in off and off.pos_fumble_lost.notna().all() else None)
    qb_rushes, qb_rush_epa = None, None
    if 'rusher_player_id' in off and qb_ids:
        rush = off.loc[off.rush.fillna(False).eq(True)]
        matched = rush.loc[rush.rusher_player_id.isin(qb_ids)]
        if rush.rusher_player_id.notna().all():
            qb_rushes = len(matched)
            qb_rush_epa = (float(matched.EPA.mean())
                           if len(matched) and matched.EPA.notna().all() else None)

    return {'points':float(points),'conceded':float(conceded),
            'drives':int(drives),'faced':int(faced),
            'off_pass_sum':op,'off_pass_n':pn,'off_rush_sum':or_,'off_rush_n':rn,
            'def_pass_sum':dp,'def_pass_n':dpn,'def_rush_sum':dr,'def_rush_n':drn,
            'interceptions':interceptions,'fumbles_lost':lost,
            'qb_rushes':qb_rushes,'qb_rush_epa':qb_rush_epa}


def project(team, opponent, neutral, side, use_epa):
    """CFB-only experimental matched possession model; uncalibrated priors."""
    prior = CFB_RULES['league_ppd']
    shrink_drives = 25.0
    off = (total(team,'points') + prior*shrink_drives) / (total(team,'drives')+shrink_drives)
    defense = (total(opponent,'conceded') + prior*shrink_drives) / (total(opponent,'faced')+shrink_drives)
    ppd = prior + .5*((off-prior) + (defense-prior))
    if use_epa:
        po = mean_epa(team,'off_pass'); pd_ = mean_epa(opponent,'def_pass')
        ro = mean_epa(team,'off_rush'); rd = mean_epa(opponent,'def_rush')
        if all(v is not None for v in (po,pd_,ro,rd)):
            pcount=total(team,'off_pass_n');rcount=total(team,'off_rush_n')
            share=pcount/(pcount+rcount)
            epa = share*(po+pd_)/2+(1-share)*(ro+rd)/2
            ppd += float(np.clip(1.5*epa, -.45, .45))
    tempo_off = total(team,'drives')/len(team)
    tempo_def = total(opponent,'faced')/len(opponent)
    expected_drives = float(np.clip((tempo_off+tempo_def)/2,7,17))
    adv = 0 if neutral else CFB_RULES['home_advantage_points']
    ppd += (adv/2 if side=='home' else -adv/2)/expected_drives

    # Estimate turnover opportunity from only games with nonmissing INT/fumble.
    known = [r for r in team if r['interceptions'] is not None and
             r['fumbles_lost'] is not None]
    turnovers = sum(r['interceptions']+r['fumbles_lost'] for r in known)
    opportunities = sum(r['drives'] for r in known)
    to_rate = ((CFB_RULES['league_to_drive']*20+turnovers)/(20+opportunities)
               if opportunities else CFB_RULES['league_to_drive'])
    return {'_ppd':float(np.clip(ppd,.35,5.0)),
            '_to':float(np.clip(to_rate,.01,.35)),
            '_td':CFB_RULES['td_share_scoring_drives'],
            'expected_drives':expected_drives}


def metrics(rows, field):
    if not rows:
        return {'games':0, 'mae':None, 'rmse':None, 'bias':None,
                'margin_direction_accuracy_excluding_actual_ties':None}
    actual=np.asarray([r['actual_margin'] for r in rows],float)
    predicted=np.asarray([r[field] for r in rows],float)
    error=predicted-actual
    directional=(actual!=0)
    return {'games':len(rows),
            'mae':round(float(np.abs(error).mean()),4),
            'rmse':round(float(np.sqrt(np.mean(error*error))),4),
            'bias':round(float(error.mean()),4),
            'margin_direction_accuracy_excluding_actual_ties':
                round(float(np.mean((predicted[directional]>0)==(actual[directional]>0))),4)
                if directional.any() else None}


def replay(snapshot, out, runs=1500, min_games=3):
    """Chronological 70/30 holdout. Not verified contemporaneous as-of data."""
    manifest=verify(snapshot)
    cutoff=iso_utc(manifest['captured_at_utc'])
    folder=Path(snapshot)
    schedule=pd.read_parquet(folder/'schedule.parquet')
    pbp=pd.read_parquet(folder/'pbp.parquet')
    roster=pd.read_parquet(folder/'roster.parquet')
    schedule['_kickoff']=pd.to_datetime(schedule.start_date,utc=True,errors='coerce')
    eligible=(schedule.fbs_participant.fillna(False).eq(True) &
              schedule.completed.fillna(False).eq(True) &
              schedule['_kickoff'].lt(cutoff) &
              ~((schedule.home_division=='fcs') & (schedule.away_division=='fcs')) &
              schedule.home_points.notna() & schedule.away_points.notna())
    games=schedule.loc[eligible].sort_values(['_kickoff','game_id']).copy()
    mapped=normalize_pbp_teams(pbp.loc[pbp.game_id.isin(games.game_id)],games)
    by_game={gid:part for gid,part in mapped.groupby('game_id',sort=False)}
    qb_ids={int(t):set(g.athlete_id.dropna()) for t,g in
            roster.loc[roster.position_abbreviation.eq('QB')].groupby('team_id')}
    history=defaultdict(list)
    preds=[]
    for kickoff, group in games.groupby('_kickoff',sort=True):
        # Entire kickoff cluster is predicted before adding any same-time result.
        additions=[]
        for _,g in group.iterrows():
            home,away=int(g.home_id),int(g.away_id)
            plays=by_game.get(g.game_id)
            if plays is None:
                continue
            h=observed_game(plays,home,away,g.home_points,g.away_points,qb_ids.get(home,set()))
            a=observed_game(plays,away,home,g.away_points,g.home_points,qb_ids.get(away,set()))
            if h is None or a is None:
                continue
            if len(history[home])>=min_games and len(history[away])>=min_games:
                item={'game_id':str(g.game_id),'kickoff':str(kickoff),
                      'home':str(g.home_team),'away':str(g.away_team),
                      'actual_margin':float(g.home_points-g.away_points)}
                for candidate,field in ((False,'baseline_margin'),(True,'epa_candidate_margin')):
                    hproj=project(history[home],history[away],bool(g.neutral_site),'home',candidate)
                    aproj=project(history[away],history[home],bool(g.neutral_site),'away',candidate)
                    result=simulate(hproj,aproj,runs=runs,
                                    seed=int(g.game_id)%(2**32),allow_overtime=True)
                    item[field]=result['mean_margin_home']
                preds.append(item)
            additions.extend(((home,h),(away,a)))
        for t,r in additions:
            history[t].append(r)
    preds.sort(key=lambda x:(x['kickoff'],x['game_id']))
    split=int(len(preds)*.7)
    while 1<split<len(preds) and preds[split-1]['kickoff']==preds[split]['kickoff']:
        split-=1
    holdout=preds[split:]
    baseline=metrics(holdout,'baseline_margin')
    challenger=metrics(holdout,'epa_candidate_margin')
    benchmark=[dict(r,constant_home_margin=CFB_RULES['home_advantage_points']) for r in holdout]
    summary={
        'engine':'cfb.engine.v1', 'nfl_code_imported':False,
        'status':'RETROSPECTIVE_CFB_RESEARCH_NOT_ATS_VALIDATED',
        'source_snapshot_utc':str(cutoff),
        'chronologically_evaluated_games':len(preds),
        'holdout_games':len(holdout),
        'holdout_baseline':baseline,'holdout_epa_challenger':challenger,
        'holdout_constant_home_advantage':metrics(benchmark,'constant_home_margin'),
        'candidate_improves_holdout_mae':
            challenger['mae']<baseline['mae'] if holdout else None,
        'disclosures':[
            '2026 source snapshot captured after the historical games; not true historical as-of.',
            'No market spreads or wager grades used; winner direction is NOT ATS accuracy.',
            'No future-game scores used in previous-matchup feature averages.',
            '2026 season results do not validate next-season predictions.',
            'CFB scoring coefficients and overtime outcomes are uncalibrated.',
            'QB roster evidence may have later updates; used only for descriptive tracking.'
        ]}
    target=Path(out);target.mkdir(parents=True,exist_ok=True)
    (target/'margin_replay.json').write_text(json.dumps(preds,indent=2)+'\n')
    (target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--runs',type=int,default=1500)
    parser.add_argument('--min-games',type=int,default=3)
    args=parser.parse_args()
    replay(args.snapshot,args.out,args.runs,args.min_games)

if __name__=='__main__':
    main()
