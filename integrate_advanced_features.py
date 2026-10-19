#!/usr/bin/env python3
"""Audited nflverse/PFR supplementary-feature bridge for the ATS matchup engine.

Pressure: PFR times_pressured / same-game nflverse PBP QB dropbacks, not a
play-level pressure flag. Missing game coverage cannot silently become zero.
Penalties: observed yards, NOT isolated EPA. Special teams: FG makes/attempts
and descriptive punting; do not manufacture special-teams EPA.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

from collect_extra_metrics import open_csv, parse_stamp, games_before_cutoff, derive_pbp, num, flag
from model import validate_game, InputError


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):h.update(chunk)
    return h.hexdigest()


def verified_snapshot(snapshot:Path, metrics_dir:Path, cutoff:str):
    snapshot,metrics_dir=Path(snapshot),Path(metrics_dir)
    m=json.loads((snapshot/'manifest.json').read_text())
    a=json.loads((metrics_dir/'metrics_manifest.json').read_text())
    if m.get('status')!='COLLECTED' or a.get('status') not in ('COMPLETE','PARTIAL_PFR'):
        raise InputError('Source collection has no valid completed manifest')
    deadline=parse_stamp(cutoff)
    checked={}
    if int(a['season'])!=int(m['season']):raise InputError('Season mismatch in supplementary dataset')
    if a.get('base_snapshot_captured_at_utc')!=m.get('created_at_utc'):
        raise InputError('Supplementary metrics computed from a different base snapshot')
    for k in ('pbp','schedule'):
        meta=m['datasets'][k];p=snapshot/meta['local_file']
        if digest(p)!=meta['sha256']:raise InputError(f'{k} SHA-256 verification failed')
        times=[meta['captured_at_utc']]
        checked[k]=p
        for t in times:
            if parse_stamp(t)>deadline:raise InputError(f'{k} captured after game data cutoff')
    if parse_stamp(a['base_snapshot_captured_at_utc'])>deadline:
        raise InputError('Snapshot captured after game data cutoff')
    if parse_stamp(a['finalized_at_utc'])>deadline:
        raise InputError('Supplementary metrics finalized after game data cutoff')
    extra=metrics_dir/a['derived_file']
    if digest(extra)!=a['derived_sha256']:raise InputError('Derived-metric SHA-256 mismatch')
    checked['extra']=extra
    if a.get('pfr',{}).get('pass',{}).get('status')=='COLLECTED':
        meta=a['pfr']['pass'];p=metrics_dir/meta['path']
        if digest(p)!=meta['sha256']:raise InputError('PFR passing SHA-256 mismatch')
        if parse_stamp(meta['capture_time_utc'])>deadline:raise InputError('PFR passing published after game cutoff')
        checked['pfr_pass']=p
    return m,a,checked


def valid_pressure_rates(pbp:Path,pfr:Path,games:dict,min_games:int=3,min_fraction:float=0.6):
    """Aligned pressure per dropback; excludes nullified and non-dropback plays.

    Games with partial QB-pressure rows are excluded. Each denominator uses the
    same game and offense as the numerator. Pressure is PFR-defined, not pbp
    participation-defined; output documents the distinct measurement.
    """
    pfr_counts=defaultdict(float); pfr_players=defaultdict(set); invalid=set()
    with open_csv(pfr) as f:
        for r in csv.DictReader(f):
            gid=str(r.get('game_id',''))
            if gid not in games:continue
            team=str(r.get('team','')); opp=str(r.get('opponent',''))
            if {team,opp}!={games[gid]['home'],games[gid]['away']} or team==opp:continue
            key=(team,gid); value=num(r.get('times_pressured'))
            if value is None or value<0 or value!=int(value):invalid.add(key);continue
            player=str(r.get('pfr_player_id',''))
            if not player or player in pfr_players[key]:invalid.add(key);continue
            pfr_players[key].add(player)
            pfr_counts[key]+=value
    # Qualified QB dropbacks, same pre-snapshot eligible games.
    dropbacks=defaultdict(int)
    with open_csv(pbp) as f:
        for r in csv.DictReader(f):
            gid=str(r.get('game_id',''))
            if gid not in games:continue
            t=str(r.get('posteam',''))
            if t not in (games[gid]['home'], games[gid]['away']):continue
            if str(r.get('play_type','')).lower() in ('no_play','qb_kneel','qb_spike'):continue
            if flag(r.get('qb_kneel')) or flag(r.get('qb_spike')):continue
            if flag(r.get('qb_dropback')) or flag(r.get('pass')):dropbacks[(t,gid)]+=1
    totals=defaultdict(lambda:{'pressures':0.0,'dropbacks':0,'games':set()})
    possible=defaultdict(int)
    for gid,v in games.items():
        possible[v['home']]+=1;possible[v['away']]+=1
    for (team,gid),n in dropbacks.items():
        key=(team,gid)
        if key in invalid or key not in pfr_counts:continue
        count=pfr_counts[key]
        if n<=0 or count>n:continue  # incompatible definitions/denominator: never clip fabricated rates
        defense=games[gid]['away'] if games[gid]['home']==team else games[gid]['home']
        for role,actor in (('offense',team),('defense',defense)):
            sample=totals[(role,actor)]
            sample['pressures']+=count
            sample['dropbacks']+=n
            sample['games'].add(gid)
    result={}
    for team in possible:
        val={}
        for role in ('offense','defense'):
            v=totals[(role,team)];played=len(v['games'])
            coverage=played/max(1,possible[team]);rate=v['pressures']/v['dropbacks'] if v['dropbacks'] else None
            if played<min_games or coverage<min_fraction or rate is None or not 0<=rate<=1:
                val[role]={'status':'INSUFFICIENT_COVERAGE','games':played,'eligible_games':possible[team], 'coverage':round(coverage,4)}
            else:
                val[role]={'status':'AVAILABLE','rate':round(rate,7),'pressures':int(v['pressures']),
                     'dropbacks':v['dropbacks'],'games':played,'eligible_games':possible[team],
                     'coverage':round(coverage,4)}
        result[team]=val
    return result


def build_overlay(snapshot:Path,metrics_dir:Path, cutoff:str, target_teams:tuple[str,str]):
    """Build source-checked input overlay; no fabricated EPA or market information."""
    m,a,src=verified_snapshot(snapshot,metrics_dir,cutoff)
    cutoff_time=parse_stamp(cutoff)
    # A conservative six-hour buffer matches the underlying strict ingestor.
    games=games_before_cutoff(src['schedule'],cutoff_time,hours=6)
    covered=valid_pressure_rates(src['pbp'],src['pfr_pass'],games) if 'pfr_pass' in src else {}
    league_press=sum(v['offense']['pressures'] for v in covered.values() if v['offense']['status']=='AVAILABLE')
    league_drop=sum(v['offense']['dropbacks'] for v in covered.values() if v['offense']['status']=='AVAILABLE')
    league_rate=round(league_press/league_drop,7) if league_drop else None
    # Critical as-of consistency: BOTH pressure and penalty/kicking measurements
    # use the SAME six-hour eligible-game window as ingest._prepare_pbp.
    # The earlier v0.5 sidecar used a four-hour window. We verify its SHA-256
    # but reconstruct the team penalty/FG aggregates from the immutable PBP
    # using the matching six-hour window rather than mixing two populations.
    event_data,observed,event_meta=derive_pbp(src['pbp'],games)
    overlay={'schema':'ats_advanced_overlay_v0.6','captured_at_utc':a['finalized_at_utc'],
             'definitions':{'pressure':'PFR QB times_pressured / same-game PBP QB dropbacks; not participation was_pressure',
               'penalties':'Observed nonnegative penalty yards per game, NOT penalty-only EPA',
               'field_goals':'Recorded makes and attempts, NOT isolated special teams EPA'},
             'source_sha256':{'pbp':m['datasets']['pbp']['sha256'],'derived':a['derived_sha256'],
                'pfr_pass':a.get('pfr',{}).get('pass',{}).get('sha256')},
             'teams':{},'missing':[], 'eligible_games':len(games), 'league_pfr_pressure_rate':league_rate, 'league_pressure_sample_dropbacks':league_drop,'consistent_completion_buffer_hours':6,'eligible_pbp_event_count':event_meta['eligible_pbp_rows']}
    for team in target_teams:
        p=covered.get(team,{'offense':{'status':'UNAVAILABLE'},'defense':{'status':'UNAVAILABLE'}})
        r=event_data.get(team)
        entry={'pressure':p,'advanced':None}
        if r is not None:
            g=len(observed[team])
            py=r['penalty_yards_observed']/g if g else None
            made=r['field_goals_made'];at=r['field_goals_attempted']
            fg_known=r['field_goals_unknown_result']
            gross=r['punt_distance_sum']/r['punt_distance_known'] if r['punt_distance_known'] else None
            if g>=3 and py is not None and 0<=py<=300 and 0<=made<=at and fg_known==0 and event_meta['missing_penalty_yards_count']==0:
                entry['advanced']={
                    'penalty_yards_per_game':round(py,5), 'penalty_games':int(g),
                    'field_goals_made':int(made),'field_goals_attempted':int(at),
                    'punt_gross_yards':round(gross,5) if gross is not None else None,
                    'source':'Verified nflverse/PFR v0.5 collector and SHA-256 manifests',
                    'as_of_utc':a['finalized_at_utc'],
                    'status':'DESCRIPTIVE_UNCALIBRATED',
                }
        if p['offense'].get('status')!='AVAILABLE' or p['defense'].get('status')!='AVAILABLE':overlay['missing'].append(f'{team}.PFR_pressure: offense={p["offense"]["status"]}; defense={p["defense"]["status"]}')
        if entry['advanced'] is None:overlay['missing'].append(f'{team}.penalty_FG_descriptive: unavailable')
        overlay['teams'][team]=entry
    return overlay


def attach_overlay(game:dict,audit:dict,overlay:dict, codes:dict):
    """Add only independently measured fields, never fabricate penalty or ST EPA."""
    if parse_stamp(overlay['captured_at_utc'])>parse_stamp(game['data_as_of_utc']):
        raise InputError('Supplementary overlay is post-cutoff')
    game['pressure_reference_rate']=overlay.get('league_pfr_pressure_rate')
    for side in ('home','away'):
        item=overlay['teams'][codes[side]]
        rate=item['pressure']
        # A verified play-level pressure join is higher fidelity than PFR's
        # weekly proxy; never silently override it.
        if audit.get('coverage',{}).get('pressure_join')!='OK':
            if rate['offense'].get('status')=='AVAILABLE':
                game[side]['offense']['pressure_allowed']=rate['offense']['rate']
                game[side]['pressure_source']='PFR_QB_TIMES_PRESSURED_OVER_SAME_GAME_PBP_DROPBACKS'
            if rate['defense'].get('status')=='AVAILABLE':
                game[side]['defense']['pressure_generated']=rate['defense']['rate']
        if item['advanced'] is not None:game[side]['advanced']=item['advanced']
    audit['supplementary_feature_overlay']={'schema':overlay['schema'],
        'by_team':{k:{'pressure':v['pressure'],'advanced':v['advanced']} for k,v in overlay['teams'].items()},
        'omissions':overlay['missing'],'source_sha256':overlay['source_sha256'],
        'notes':'Penalty yards and kicking not relabeled as EPA. Source-capture cutoff checked.'}
    # Restore validation status after supplemental additions.
    from model import OFF_KEYS,DEF_KEYS,SOS_KEYS,CONTEXT_KEYS,PERSONNEL_KEYS
    missing=[]
    for side in ('home','away'):
        t=game[side]
        for section,keys in (('offense',OFF_KEYS),('defense',DEF_KEYS),('schedule',SOS_KEYS),('context',('avg_start_yardline',)),('personnel',PERSONNEL_KEYS)):
            missing.extend(f'{side}.{section}.{k}' for k in keys if t[section].get(k) is None)
        if t['stat_as_of_utc'] is None:missing.append(f'{side}.stat_as_of_utc')
    missing.extend(f'market.{k}' for k in ('spread_home','home_odds_american','away_odds_american','total','as_of_utc','source') if game['market'].get(k) is None)
    audit['missing_fields']=sorted(set(missing));audit['missing_count']=len(audit['missing_fields'])
    if not missing and not audit['asof_issues']:
        try:validate_game(game);audit['status']='READY_FOR_MODEL_UNCALIBRATED'
        except (ValueError,TypeError) as e:audit['status']='BLOCKED_SCHEMA_VALIDATION';audit['validation_error']=str(e)
    else:audit['status']='BLOCKED_MISSING_INPUTS_OR_ASOF'
    return game,audit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--metrics-dir',type=Path,required=True)
    ap.add_argument('--cutoff',required=True);ap.add_argument('--home',required=True);ap.add_argument('--away',required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();overlay=build_overlay(a.snapshot,a.metrics_dir,a.cutoff,(a.home,a.away))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(overlay,indent=2)+'\n')
    print(json.dumps({'overlay':str(a.output),'missing':overlay['missing'], 'teams':overlay['teams']},indent=2))

if __name__=='__main__':main()
