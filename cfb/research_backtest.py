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
from cfb.advanced_scoring import simulate_challenger
from cfb.game_script import simulate_script
from cfb.mechanism_build16 import simulate_build16, observe_game_context
from cfb.disruption import observed_disruption
from cfb.matchups import matchup_signals
from cfb.calibration import calibrate
from cfb.pipeline import verify, normalize_pbp_teams, iso_utc
from cfb.field_special import observed_special_teams


def total(rows, key):
    return sum(float(row[key]) for row in rows)


def mean_epa(records, kind):
    count = total(records, kind + '_n')
    return total(records, kind + '_sum') / count if count else None


def observed_game(plays, team, opponent, points, conceded, qb_ids, special_plays=None):
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

    def event_count(df, indicator, subset=None):
        if indicator not in df: return (None, None)
        sample=df if subset is None else df.loc[df[subset].fillna(False).eq(True)]
        if sample.empty or sample[indicator].isna().any(): return (None, None)
        return (int(sample[indicator].eq(True).sum()), int(len(sample)))

    def explosive(df, kind):
        subset=df.loc[df[kind].fillna(False).eq(True)]
        if subset.empty:return None,None
        if 'yards_gained' not in subset or subset.yards_gained.isna().any():return None,None
        y=pd.to_numeric(subset.yards_gained,errors='coerce')
        if y.isna().any():return None,None
        return int((y >= (20 if kind=='pass' else 10)).sum()),int(len(y))

    pe,pn2=explosive(off,'pass'); re,rn2=explosive(off,'rush')
    dp_e,dp_n=explosive(opp,'pass');dr_e,dr_n=explosive(opp,'rush')
    def success(df, kind):
        z=df.loc[df[kind].fillna(False).eq(True)]
        if z.empty or not {'down','distance','yards_gained'}.issubset(z.columns):
            return None,None
        down=pd.to_numeric(z.down,errors='coerce')
        dist=pd.to_numeric(z.distance,errors='coerce')
        gain=pd.to_numeric(z.yards_gained,errors='coerce')
        if down.isna().any() or dist.isna().any() or gain.isna().any():
            return None,None
        threshold=dist.where(down>=3,dist*.7).where(down!=1,dist*.5)
        return int((gain>=threshold).sum()),int(len(z))
    ps,pns=success(off,'pass'); rs,rns=success(off,'rush')
    dps,dpns=success(opp,'pass'); drs,drns=success(opp,'rush')
    # Explicit source-backed *counts*: interception and lost fumble opportunities,
    # offensive sacks allowed and defensive sacks made; no fillna=0 on nulls.
    int_s,int_n=event_count(off,'int','pass')
    dint_s,dint_n=event_count(opp,'int','pass')
    fum_s,fum_n=event_count(off,'pos_fumble_lost')
    dfum_s,dfum_n=event_count(opp,'pos_fumble_lost')
    sack_s,sack_n=event_count(off,'sack','pass')
    dsack_s,dsack_n=event_count(opp,'sack','pass')
    hc,hn=event_count(off,'qb_hurry','pass')
    dhc,dhn=event_count(opp,'qb_hurry','pass')
    third=off.loc[off.down.eq(3)]; dthird=opp.loc[opp.down.eq(3)]
    def third_rate(df):
        required=('firstD_by_yards','firstD_by_penalty','rush_td','pass_td')
        if df.empty or any(c not in df or df[c].isna().any() for c in required):return None,None
        x=df[list(required)].eq(True).any(axis=1)
        return int(x.sum()),int(len(x))
    tc,tn=third_rate(third);dtc,dtn=third_rate(dthird)
    # Recorded starting position and QB rushing remain optional, descriptive.
    starts=off.sort_values('game_play_number').drop_duplicates('drive_id')
    field_position=None
    start_field_n=None
    if 'yards_to_goal' in starts and len(starts):
        vals=pd.to_numeric(starts.yards_to_goal,errors='coerce')
        if vals.notna().all() and vals.between(1,99).all():
            field_position=float((100-vals).mean())
            start_field_n=int(len(vals))
    # Opponent's TRUE starting field position while this school's defense
    # was on the field. Not an inferred punt location or turnover spot.
    opponent_starts=opp.sort_values('game_play_number').drop_duplicates('drive_id')
    field_allowed=None
    field_allowed_n=None
    if 'yards_to_goal' in opponent_starts and len(opponent_starts):
        vals=pd.to_numeric(opponent_starts.yards_to_goal,errors='coerce')
        if vals.notna().all() and vals.between(1,99).all():
            field_allowed=float((100-vals).mean())
            field_allowed_n=int(len(vals))
    rz=off.loc[pd.to_numeric(off.yards_to_goal,errors='coerce').le(20)] if 'yards_to_goal' in off else off.iloc[:0]
    redzone_drives=int(rz.drive_id.nunique()) if not rz.empty else 0
    rz_td=int(rz.loc[rz.offense_score_play.eq(True)&rz.td_play.eq(True),'drive_id'].nunique()) if len(rz) else 0
    drzone=opp.loc[pd.to_numeric(opp.yards_to_goal,errors='coerce').le(20)] if 'yards_to_goal' in opp else opp.iloc[:0]
    faced_rz=int(drzone.drive_id.nunique()) if len(drzone) else 0
    conceded_rz_tds=int(drzone.loc[drzone.offense_score_play.eq(True)&drzone.td_play.eq(True),'drive_id'].nunique()) if len(drzone) else 0

    return {'opponent_id':int(opponent),
            'mechanism':observe_game_context(plays,team,opponent),
            'disruption_off':observed_disruption(plays,team),
            'disruption_def':observed_disruption(plays,opponent),
            'points':float(points),'conceded':float(conceded),
            'drives':int(drives),'faced':int(faced),
            'off_pass_sum':op,'off_pass_n':pn,'off_rush_sum':or_,'off_rush_n':rn,
            'def_pass_sum':dp,'def_pass_n':dpn,'def_rush_sum':dr,'def_rush_n':drn,
            'interceptions':interceptions,'fumbles_lost':lost,
            'int_sum':int_s,'int_n':int_n,'def_int_sum':dint_s,'def_int_n':dint_n,
            'fumble_sum':fum_s,'fumble_n':fum_n,
            'def_fumble_sum':dfum_s,'def_fumble_n':dfum_n,
            'sack_sum':sack_s,'sack_n':sack_n,
            'def_sack_sum':dsack_s,'def_sack_n':dsack_n,
            'qb_rushes':qb_rushes,'qb_rush_epa':qb_rush_epa,
            'pass_explosive_sum':pe,'pass_explosive_n':pn2,
            'rush_explosive_sum':re,'rush_explosive_n':rn2,
            'def_pass_explosive_sum':dp_e,'def_pass_explosive_n':dp_n,
            'def_rush_explosive_sum':dr_e,'def_rush_explosive_n':dr_n,
            'pass_success_sum':ps,'pass_success_n':pns,
            'rush_success_sum':rs,'rush_success_n':rns,
            'def_pass_success_sum':dps,'def_pass_success_n':dpns,
            'def_rush_success_sum':drs,'def_rush_success_n':drns,
            'qb_hurries_sum':hc,'qb_hurries_n':hn,
            'def_qb_hurries_sum':dhc,'def_qb_hurries_n':dhn,
            'third_conversions_sum':tc,'third_conversions_n':tn,
            'def_third_conversions_sum':dtc,'def_third_conversions_n':dtn,
            'start_field_position':field_position,
            'start_field_n':start_field_n,
            'start_field_position_allowed':field_allowed,
            'start_field_allowed_n':field_allowed_n,
            'special_teams':observed_special_teams(special_plays,team,opponent),
            'redzone_trips':redzone_drives,'redzone_tds':rz_td,
            'def_redzone_trips':faced_rz,'def_redzone_tds':conceded_rz_tds,
            'redzone_tds_sum':rz_td,'redzone_tds_n':redzone_drives,
            'def_redzone_tds_sum':conceded_rz_tds,'def_redzone_tds_n':faced_rz}



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
    raw_by_game={gid:part for gid,part in
                 pbp.loc[pbp.game_id.isin(games.game_id)].groupby('game_id',sort=False)}
    qb_ids={int(t):set(g.athlete_id.dropna()) for t,g in
            roster.loc[roster.position_abbreviation.eq('QB')].groupby('team_id')}
    history=defaultdict(list)
    pending=[]
    preds=[]
    for kickoff, group in games.groupby('_kickoff',sort=True):
        # Conservatively avoid using an earlier *kickoff* as if that game's
        # final were already known. Its outcome becomes eligible a full 24h
        # after kickoff; actual publish/final timestamps are not archived.
        mature=[(ready,team,result) for ready,team,result in pending if ready<=kickoff]
        pending=[item for item in pending if item[0]>kickoff]
        for _,team,result in mature:
            history[team].append(result)
        # Entire kickoff cluster is predicted before adding any same-time result.
        additions=[]
        for _,g in group.iterrows():
            home,away=int(g.home_id),int(g.away_id)
            plays=by_game.get(g.game_id)
            if plays is None:
                continue
            h=observed_game(plays,home,away,g.home_points,g.away_points,qb_ids.get(home,set()),
                            special_plays=raw_by_game.get(g.game_id))
            a=observed_game(plays,away,home,g.away_points,g.home_points,qb_ids.get(away,set()),
                            special_plays=raw_by_game.get(g.game_id))
            if h is None or a is None:
                continue
            if len(history[home])>=min_games and len(history[away])>=min_games:
                item={'game_id':str(g.game_id),'kickoff':str(kickoff),
                      'home':str(g.home_team),'away':str(g.away_team),
                      'actual_margin':float(g.home_points-g.away_points)}
                hp=project(history[home],history[away],bool(g.neutral_site),'home',False)
                ap=project(history[away],history[home],bool(g.neutral_site),'away',False)
                baseline=simulate(hp,ap,runs=runs,seed=int(g.game_id)%(2**32),allow_overtime=True)
                item['baseline_margin']=baseline['mean_margin_home']
                challenger=simulate_build16(home,away,history,bool(g.neutral_site),
                    runs=runs,seed=int(g.game_id)%(2**32))
                item['build16_candidate_status']=challenger['status']
                item['build16_candidate_margin']=(challenger.get('mean_margin_home')
                    if challenger.get('simulation') is not None else None)
                item['build16_candidate_interval']=challenger.get('margin_p10_p50_p90')
                item['completed_prior_games_home']=len(history[home])
                item['completed_prior_games_away']=len(history[away])
                # Stored diagnostic inputs are not additional competing forecasts.
                item['matchup_evidence']=matchup_signals(home,away,history)
                preds.append(item)
            additions.extend(((home,h),(away,a)))
        for t,r in additions:
            pending.append((kickoff+pd.Timedelta(hours=24),t,r))
    preds.sort(key=lambda x:(x['kickoff'],x['game_id']))
    split=int(len(preds)*.7)
    while 1<split<len(preds) and preds[split-1]['kickoff']==preds[split]['kickoff']:
        split-=1
    holdout=preds[split:]
    baseline=metrics(holdout,'baseline_margin')
    paired=[r for r in holdout if r['build16_candidate_margin'] is not None]
    summary={
        'engine':'cfb.engine.v1', 'nfl_code_imported':False,
        'status':'RETROSPECTIVE_CFB_RESEARCH_NOT_ATS_VALIDATED',
        'source_snapshot_utc':str(cutoff),
        'chronologically_evaluated_games':len(preds),
        'holdout_games':len(holdout),
        'holdout_baseline':baseline,
        'build16_challenger':{
            'status':'RESEARCH_ONLY_NOT_VALIDATED_OR_DEPLOYED',
            'eligible_holdout_games':len(paired),
            'unavailable_holdout_games':len(holdout)-len(paired),
            'paired_baseline':metrics(paired,'baseline_margin'),
            'paired_build16':metrics(paired,'build16_candidate_margin'),
            'p10_p90_coverage':round(sum(
                r['build16_candidate_interval'][0]<=r['actual_margin']<=r['build16_candidate_interval'][2]
                for r in paired)/len(paired),4) if paired else None,
            'not_primary':True},
        'disclosures':[
            'Two published models only: original baseline vs Build 16 unified challenger.',
            '2026 source snapshot captured after historical games; not true as-of evidence.',
            'No historical sportsbook spreads here; margin winner accuracy is NOT ATS accuracy.',
            'Earlier-game results embargoed for 24h after kickoff.',
            'Build16 parameters uncalibrated; do not tune to previously examined holdout.',
            'Missing complete pressure/penalty/special teams data are never synthesized.'
        ]}
    # Calibrate on the SAME eligible baseline/Build16 pairs; never impute a
    # missing challenger as a zero-valued effect or revive retired variants.
    calibrated=calibrate([r for r in preds if r['build16_candidate_margin'] is not None])
    summary['margin_calibration']=calibrated
    summary['feature_definitions']={
        'sos_margin_feature':'Leave-opponent-out pass/rush EPA differential using only preceding games',
        'context_margin_feature':'Observed explosive, third-down and QB-hurry paired rates; not a fitted effect',
        'history_embargo':'Previous games are excluded until 24h after kickoff',
        'qb_hurry':'QB hurry is not total defensive pressure',
        'field_position':'Scrimmage drive-start average; omits some non-scrimmage drives'
    }
    target=Path(out);target.mkdir(parents=True,exist_ok=True)
    (target/'calibration.json').write_text(json.dumps(calibrated,indent=2)+'\n')
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
