#!/usr/bin/env python3
"""Free nflverse/PFR supplementary metrics. NEVER changes the original PBP manifest.

Produces per-team descriptive features and an audit manifest. Not a calibrated
ATS model and not a replacement for play-level pressure/participation tracking.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SOURCE = 'nflreadpy.load_pfr_advstats(seasons=YEAR, stat_type=TYPE, summary_level="week")'
PFR_TYPES = ('pass', 'def')


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def parse_stamp(s: str) -> datetime:
    if not s: raise ValueError('Missing timestamp')
    d = datetime.fromisoformat(str(s).replace('Z','+00:00'))
    if d.tzinfo is None: raise ValueError('Timezone missing in timestamp: '+str(s))
    return d.astimezone(timezone.utc)


def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def open_csv(p: Path):
    return gzip.open(p,'rt',newline='',encoding='utf-8-sig') if str(p).endswith('.gz') else p.open('r',newline='',encoding='utf-8-sig')


def num(x):
    try:
        if x is None or str(x).strip()=='' :return None
        n=float(x)
        return n if math.isfinite(n) else None
    except (ValueError,TypeError):return None


def flag(x):
    return num(x)==1


def sample_mean(total, count):
    return round(total/count,5) if count else None


def games_before_cutoff(schedule_file: Path, cutoff: datetime, hours: int=4):
    """Only games with scheduled kickoff >=4h before source capture; not proof of final whistle.

    Cross-check PBP game presence below. Scheduled timestamps are nflverse's ET.
    """
    games={}
    with open_csv(schedule_file) as f:
        for row in csv.DictReader(f):
            gid=row.get('game_id')
            if not gid:continue
            if row.get('kickoff_utc'):
                kickoff=parse_stamp(row['kickoff_utc'])
            else:
                day=row.get('gameday'); hour=row.get('gametime')
                if not day or not hour:continue
                kickoff=datetime.fromisoformat(day+'T'+hour).replace(tzinfo=ZoneInfo('America/New_York')).astimezone(timezone.utc)
            if kickoff+timedelta(hours=hours)<=cutoff:
                games[gid]={'home':row['home_team'],'away':row['away_team'],'kickoff_utc':kickoff.isoformat()}
    return games


def empty():
    return defaultdict(float)


def derive_pbp(pbp_file: Path, games:dict):
    """Count observed penalty events, kicking/punting events, and team-perspective event EPA.

    Avoids inaccurate claim that full penalty-play EPA == isolated penalty cost.
    """
    data=defaultdict(empty)
    observed=defaultdict(set)
    missing_penalty_yards=0
    examined=0
    with open_csv(pbp_file) as f:
        reader=csv.DictReader(f)
        need={'game_id','posteam','defteam','play_type','penalty','penalty_team','penalty_yards',
              'field_goal_attempt','field_goal_result','punt_attempt','punt_blocked',
              'punt_inside_twenty','kickoff_attempt','return_yards','kick_distance','epa'}
        if need-set(reader.fieldnames or []):
            raise ValueError('PBP missing necessary descriptive columns: '+str(sorted(need-set(reader.fieldnames or []))))
        for r in reader:
            gid=r['game_id']
            if gid not in games:continue
            home,away=games[gid]['home'],games[gid]['away']
            a,b=r['posteam'],r['defteam']; typ=r['play_type']
            examined+=1
            if flag(r['penalty']) and r['penalty_team'] in (home,away):
                t=r['penalty_team'];data[t]['penalty_flagged_plays']+=1
                yards=num(r['penalty_yards'])
                if yards is None or yards < 0:missing_penalty_yards+=1
                else:
                    data[t]['penalty_yards_observed']+=yards
                    data[t]['penalty_plays_with_yards']+=1
                observed[t].add(gid)
            # These events have distinct team attribution in nflverse PBP.
            if typ=='field_goal' and flag(r['field_goal_attempt']) and a in (home,away):
                t=a;data[t]['field_goals_attempted']+=1
                result=r['field_goal_result'].lower().strip()
                if result=='made':data[t]['field_goals_made']+=1
                if result not in ('made','missed','blocked'):
                    data[t]['field_goals_unknown_result']+=1
                observed[t].add(gid)
                e=num(r['epa'])
                if e is not None:
                    data[t]['fg_event_epa_for_possession_team']+=e
                    data[t]['fg_epa_events']+=1
            elif typ=='punt' and flag(r['punt_attempt']) and a in (home,away):
                t=a;data[t]['punts']+=1
                data[t]['punts_inside20']+=int(flag(r['punt_inside_twenty']))
                data[t]['punts_blocked']+=int(flag(r['punt_blocked']))
                distance=num(r['kick_distance'])
                if distance is not None and distance>=0:
                    data[t]['punt_distance_sum']+=distance;data[t]['punt_distance_known']+=1
                ret=num(r['return_yards'])
                if ret is not None and ret>=0:
                    data[t]['punt_return_yards_allowed']+=ret
                    data[t]['punt_return_yards_known']+=1
                observed[t].add(gid)
                e=num(r['epa'])
                if e is not None:
                    data[t]['punt_event_epa_for_possession_team']+=e
                    data[t]['punt_epa_events']+=1
            elif typ=='kickoff' and flag(r['kickoff_attempt']) and a in (home,away) and b in (home,away):
                # nflverse POSTEAM is receiving team on kickoffs; DEFTEAM kicked off.
                t=b;data[t]['kickoffs']+=1
                data[t]['kickoff_touchbacks']+=int(flag(r.get('touchback')))
                ret=num(r['return_yards'])
                if ret is not None and ret>=0:
                    data[t]['kickoff_return_yards_allowed']+=ret
                    data[t]['kickoff_return_yards_known']+=1
                observed[t].add(gid)
                e=num(r['epa'])
                if e is not None:
                    data[t]['kickoff_event_epa_for_kicking_team']-=e
                    data[t]['kickoff_epa_events']+=1
            # No all-play 'special-teams EPA': non-special plays and nullified
            # plays require separate attribution; the above event splits are safer.
    # Denominators from PBP game IDs, not from only team penalty/event games.
    with open_csv(pbp_file) as f:
        for r in csv.DictReader(f):
            gid=r['game_id']
            if gid in games:
                observed[games[gid]['home']].add(gid)
                observed[games[gid]['away']].add(gid)
    return data,observed,{'eligible_pbp_rows':examined,'missing_penalty_yards_count':missing_penalty_yards}


def pfr_team_pressure(rows:list[dict],games:dict):
    """From PFR QB-level passing pressure to offense and the opponent's defense.

    Definition: credited pressure occurrences per PFR-observed completed game;
    NOT a calibrated pass-rush win rate and NOT the play-level was_pressure flag.
    """
    offensive=defaultdict(float);defensive=defaultdict(float)
    coverage_off=defaultdict(set);coverage_def=defaultdict(set)
    skipped=0; missing=0
    for r in rows:
        gid=str(r.get('game_id') or '')
        if gid not in games:skipped+=1;continue
        a=str(r.get('team') or '').upper();b=str(r.get('opponent') or '').upper()
        if {a,b}!={games[gid]['home'],games[gid]['away']}:
            skipped+=1;continue
        val=num(r.get('times_pressured'))
        if val is None or val<0:
            missing+=1;continue
        offensive[a]+=val;defensive[b]+=val
        coverage_off[a].add(gid);coverage_def[b].add(gid)
    teams=set(offensive)|set(defensive)
    stats={}
    for t in sorted(teams):
        off_games=len(coverage_off[t]);def_games=len(coverage_def[t]);
        stats[t]={
            'pfr_qb_pressures_allowed':int(offensive[t]) if off_games else None,
            'pfr_pressure_allowed_per_observed_game':sample_mean(offensive[t],off_games),
            'pfr_pass_games_observed':off_games,
            'pfr_opponent_qb_pressures_generated':int(defensive[t]) if def_games else None,
            'pfr_pressure_generated_per_observed_game':sample_mean(defensive[t],def_games),
            'pfr_opponent_pass_games_observed':def_games,
        }
    return stats,{'pfr_rows_outside_eligible_games_or_invalid_team':skipped,
                  'pfr_rows_missing_times_pressured':missing,
                  'pfr_eligible_games_covered':len(set().union(*coverage_off.values())) if coverage_off else 0}


def _write_dict_rows(p:Path, rows:list[dict]):
    if not rows:return
    keys=list(rows[0]);extra=set().union(*(set(r) for r in rows))-set(keys)
    keys+=sorted(extra)
    with (gzip.open(p,'wt',newline='',encoding='utf8') if str(p).endswith('.gz') else p.open('w',newline='',encoding='utf8')) as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)


def collect(snapshot:Path, output:Path, *, nfl_module=None, pfr_files=None,
            minimum_hours:int=4, now=None):
    """Run after successful base nflverse collection. PFR failures are disclosed, not hidden."""
    snapshot=Path(snapshot);output=Path(output)
    base=json.loads((snapshot/'manifest.json').read_text())
    if base.get('status')!='COLLECTED':raise ValueError('Base snapshot not collected')
    for kind in ('pbp','schedule'):
        d=base['datasets'][kind]
        f=snapshot/d['local_file']
        if not f.exists() or sha256(f)!=d['sha256']:
            raise ValueError('Base '+kind+' file is missing or failed SHA-256 validation')
    season=int(base['season'])
    cutoff=parse_stamp(base['created_at_utc'])
    schedule=snapshot/base['datasets']['schedule']['local_file']
    pbp=snapshot/base['datasets']['pbp']['local_file']
    games=games_before_cutoff(schedule,cutoff,hours=minimum_hours)
    data,observed,meta=derive_pbp(pbp,games)
    # Do not mix with base manifest, which records the provenance of original files.
    output.mkdir(parents=True,exist_ok=True)
    result={'schema_version':'1.0','season':season,'base_manifest':str(snapshot/'manifest.json'),
            'base_snapshot_captured_at_utc':base['created_at_utc'],
            'minimum_game_completion_buffer_hours':minimum_hours,
            'eligible_schedule_games':len(games), 'eligible_games_with_pbp':len(set.union(*observed.values())) if observed else 0,
            'pbp':meta,'pfr':{},'warnings':[],
            'feature_definitions':{
                'penalty_flagged_plays':'PBP rows with penalty==1 and a matching penalty_team; not guaranteed to equal official assessed penalties',
                'penalty_yards_observed':'Sum of valid nonnegative PBP penalty_yards for the assessed team; declines/offsets can differ from official totals',
                'field_goals_attempted':'field_goal_attempt==1, play_type=field_goal, kicking team=posteam',
                'punt_distance_mean':'mean kick_distance on punt plays where it is valid; gross, not net',
                'punt_return_yards_allowed':'PBP return_yards on punt plays where posteam is the punting team',
                'kickoff_return_yards_allowed':'PBP return_yards on kickoff plays where defteam is the kicking team',
                'event_epa':'Raw nflverse EPA of FG/punt/kickoff events with correct team-perspective signs; NOT causal isolated special-teams impact',
                'pfr_qb_pressures_allowed':'Sum of times_pressured across QBs against eligible opponents, NOT play-level charting',
                'pfr_opponent_qb_pressures_generated':'Those same opponent-QB PFR pressure counts credited as a defensive descriptive proxy',
            },'status':'IN_PROGRESS'}
    press={}
    for kind in PFR_TYPES:
        try:
            if pfr_files and kind in pfr_files:
                with open_csv(Path(pfr_files[kind])) as f: rows=list(csv.DictReader(f))
                loader='user-supplied pfr CSV fixture (offline/test mode)'
            else:
                if nfl_module is None:
                    nfl_module=importlib.import_module('nflreadpy')
                frame=nfl_module.load_pfr_advstats(seasons=season,stat_type=kind,summary_level='week')
                if hasattr(frame,'to_dicts'):rows=frame.to_dicts()
                elif hasattr(frame,'to_dict'):rows=frame.to_dict(orient='records')
                else:raise ValueError('PFR source not a recognized DataFrame')
                loader=SOURCE.replace('TYPE',kind).replace('YEAR',str(season))
            if not rows:raise ValueError('PFR weekly '+kind+' dataset empty')
            if kind=='pass' and not {'game_id','team','opponent','times_pressured'}<=set(rows[0]):
                raise ValueError('Missing essential PFR pressure fields: game_id/team/opponent/times_pressured')
            if 'game_id' not in rows[0] or 'team' not in rows[0]:
                raise ValueError('PFR '+kind+' missing game_id/team')
            p=output/f'pfr_advstats_week_{kind}_{season}.csv.gz'
            _write_dict_rows(p,rows)
            result['pfr'][kind]={'status':'COLLECTED','rows':len(rows),'sha256':sha256(p),
                                 'path':p.name,'columns':list(rows[0]), 'capture_time_utc':stamp(),
                                 'method':loader, 'point_in_time_note':'Retrieved now; rows must be filtered to eligible completed games before use.'}
            if kind=='pass':
                press,cov=pfr_team_pressure(rows,games)
                result['pfr'][kind]['coverage']=cov
        except Exception as e:
            result['pfr'][kind]={'status':'UNAVAILABLE','reason':f'{type(e).__name__}: {e}',
                                  'capture_time_utc':stamp()}
            result['warnings'].append(f'PFR {kind} not available. Existing PBP statistics remain usable; pressure metrics remain missing.')
    all_teams=sorted(observed)
    features=[]
    for t in all_teams:
        d=data[t];g=len(observed[t]);pressure=press.get(t,{})
        row={'team':t,'season':season,'pbp_games_observed':g,
             'penalty_flagged_plays':int(d['penalty_flagged_plays']),
             'penalty_flagged_plays_per_game':sample_mean(d['penalty_flagged_plays'],g),
             'penalty_yards_observed':round(d['penalty_yards_observed'],2),
             'penalty_yards_per_game':sample_mean(d['penalty_yards_observed'],g),
             'field_goals_attempted':int(d['field_goals_attempted']),
             'field_goals_made':int(d['field_goals_made']),
             'field_goals_unknown_result':int(d['field_goals_unknown_result']),
             'field_goal_pct':sample_mean(100*d['field_goals_made'],d['field_goals_attempted']) if not d['field_goals_unknown_result'] else None,
             'punts':int(d['punts']), 'punts_inside20':int(d['punts_inside20']),
             'punts_blocked':int(d['punts_blocked']),
             'punt_distance_mean':sample_mean(d['punt_distance_sum'],d['punt_distance_known']),
             'punt_return_yards_allowed':round(d['punt_return_yards_allowed'],2),
             'kickoffs':int(d['kickoffs']), 'kickoff_touchbacks':int(d['kickoff_touchbacks']),
             'kickoff_return_yards_allowed':round(d['kickoff_return_yards_allowed'],2),
             'fg_event_epa_for_possession_team':round(d['fg_event_epa_for_possession_team'],4) if d['fg_epa_events'] else None,
             'punt_event_epa_for_possession_team':round(d['punt_event_epa_for_possession_team'],4) if d['punt_epa_events'] else None,
             'kickoff_event_epa_for_kicking_team':round(d['kickoff_event_epa_for_kicking_team'],4) if d['kickoff_epa_events'] else None,
             'pfr_qb_pressures_allowed':pressure.get('pfr_qb_pressures_allowed'),
             'pfr_pressure_allowed_per_observed_game':pressure.get('pfr_pressure_allowed_per_observed_game'),
             'pfr_pass_games_observed':pressure.get('pfr_pass_games_observed',0),
             'pfr_opponent_qb_pressures_generated':pressure.get('pfr_opponent_qb_pressures_generated'),
             'pfr_pressure_generated_per_observed_game':pressure.get('pfr_pressure_generated_per_observed_game'),
             'pfr_opponent_pass_games_observed':pressure.get('pfr_opponent_pass_games_observed',0),
             }
        features.append(row)
    features_file=output/'team_extra_metrics.csv'
    _write_dict_rows(features_file,features)
    result['derived_file']=features_file.name
    result['derived_rows']=len(features)
    result['derived_sha256']=sha256(features_file)
    result['finalized_at_utc']=stamp()
    result['asof_policy']='Combined features are NOT proven pregame for any target unless PFR capture time and base collection time are both BEFORE its kickoff. No historic bookmaker/injury data are inferred.'
    result['status']='COMPLETE' if all(result['pfr'][k]['status']=='COLLECTED' for k in PFR_TYPES) else 'PARTIAL_PFR'
    (output/'metrics_manifest.json').write_text(json.dumps(result,indent=2,default=str)+'\n',encoding='utf8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--minimum-hours',type=int,default=4)
    opts=p.parse_args()
    summary=collect(opts.snapshot,opts.out,minimum_hours=opts.minimum_hours)
    print(json.dumps({'status':summary['status'],'teams':summary['derived_rows'],
                      'eligible_schedule_games':summary['eligible_schedule_games'],
                      'pfr':{k:{'status':v['status'],'rows':v.get('rows')} for k,v in summary['pfr'].items()},
                      'features':str(opts.out/'team_extra_metrics.csv')},indent=2))

if __name__=='__main__':main()
