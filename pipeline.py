#!/usr/bin/env python3
"""Free CFB data ingestion, strict capture provenance, and pregame feature audits.

Data source: SportsDataverse's *released* ESPN CFB PBP plus the unified
cfb_schedules and ESPN roster files. Never fill unavailable mandatory model
fields with conjecture. Never reconstruct an earlier information state by
filtering a later-captured file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

DATASET_ROOT = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download'
ASSETS = {
    'pbp': ('espn_cfb_pbp', 'play_by_play_{season}.parquet'),
    'schedule': ('cfb_schedules', 'cfb_schedules_{season}.parquet'),
    'roster': ('espn_cfb_rosters', 'cfb_rosters_{season}.parquet'),
}
# Field lists grounded in sportsdataverse/cfbfastR-cfb-data producer source.
# IMPORTANT: release PBP has team *names*, 'drive_id', 'yards_gained', not
# the raw source's pos_team_id, drive.id, statYardage.
REQUIRED_SCHEDULE = ('game_id', 'start_date', 'completed', 'home_id', 'away_id',
                     'home_team', 'away_team', 'fbs_participant', 'home_division',
                     'away_division', 'neutral_site', 'start_time_tbd')
REQUIRED_PBP = ('game_id', 'game_play_number', 'pos_team', 'def_pos_team',
                'home', 'away', 'drive_id', 'pass', 'rush', 'EPA', 'down',
                'distance', 'yards_gained', 'yards_to_goal', 'sack', 'int',
                'firstD_by_yards', 'firstD_by_penalty', 'rush_td', 'pass_td',
                'offense_score_play', 'td_play', 'turnover')

class CFBError(ValueError):
    pass

def iso_utc(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise CFBError('Timezone-aware timestamp required')
    return t.tz_convert('UTC')

def source_urls(season):
    if not 2002 <= int(season) <= 2100:
        raise CFBError('Invalid college football season')
    return {key: f'{DATASET_ROOT}/{tag}/{filename.format(season=season)}'
            for key, (tag, filename) in ASSETS.items()}

def capture(season, destination):
    """Hash real bytes; publish the manifest only after all files are acquired."""
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise CFBError('Immutable snapshot directory is nonempty; refusing overwrite')
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for kind, url in source_urls(season).items():
        p = destination / (kind + '.parquet')
        part = destination / (kind + '.part')
        digest = hashlib.sha256()
        size = 0
        try:
            with urlopen(Request(url, headers={'User-Agent': 'free-cfb-ats-research/0.2'}), timeout=120) as response, part.open('wb') as target:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
                    size += len(chunk)
            if size < 100 or not part.read_bytes()[-4:] == b'PAR1':
                raise CFBError('Invalid or incomplete parquet download: ' + kind)
            part.replace(p)
            files[kind] = {'file': p.name, 'sha256': digest.hexdigest(),
                           'size_bytes': size, 'url': url,
                           'captured_at_utc': datetime.now(timezone.utc).isoformat()}
        finally:
            part.unlink(missing_ok=True)
    manifest = {'schema': 'cfb_source_snapshot_v2', 'season': int(season),
                'captured_at_utc': datetime.now(timezone.utc).isoformat(),
                'source': 'SportsDataverse public GitHub release assets',
                'datasets': files,
                'asof_warning': 'Capture times are not proof of row-level historical publication times. No retrospective as-of backdating.'}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest

def verify(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest.get('schema') != 'cfb_source_snapshot_v2' or set(manifest.get('datasets', {})) != set(ASSETS):
        raise CFBError('Unexpected snapshot schema or datasets')
    snapshot_at = iso_utc(manifest['captured_at_utc'])
    for kind, record in manifest['datasets'].items():
        p = folder / record['file']
        if p.name != kind + '.parquet' or not p.is_file():
            raise CFBError('Missing or unsafe path: ' + kind)
        if p.stat().st_size != record['size_bytes']:
            raise CFBError('Snapshot size mismatch: ' + kind)
        with p.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        if sha != record['sha256']:
            raise CFBError('Snapshot SHA256 mismatch: ' + kind)
        if iso_utc(record['captured_at_utc']) > snapshot_at:
            raise CFBError('Snapshot capture time earlier than dataset capture')
    return manifest

def require(df, columns, where):
    absent = sorted(set(columns) - set(df.columns))
    if absent:
        raise CFBError(f'{where}: missing release columns {absent}')

def flag(s):
    """Return False only for observed False; caller handles missing separately."""
    return s.fillna(False).eq(True)

def mean(s):
    v = pd.to_numeric(s, errors='coerce')
    if not len(v) or v.isna().any():
        return None
    x = float(v.mean())
    return x if math.isfinite(x) else None

def boolean_rate(df, col):
    if df.empty or col not in df or df[col].isna().any():
        return None
    return float(flag(df[col]).mean())

def split_games(schedule, pbp, cutoff):
    """Exclude FCS-FCS and unplayed/in-progress game PBP at a *captured* cutoff.

    The caller must verify the immutable capture manifest first.
    """
    require(schedule, REQUIRED_SCHEDULE, 'schedule')
    require(pbp, REQUIRED_PBP, 'pbp')
    s = schedule.copy()
    if s['game_id'].isna().any() or s['game_id'].duplicated().any():
        raise CFBError('Invalid or duplicate schedule game_id')
    s['_kickoff'] = pd.to_datetime(s['start_date'], utc=True, errors='coerce')
    if s['_kickoff'].isna().any():
        raise CFBError('Unparseable scheduled kickoff')
    if pbp['game_id'].isna().any():
        raise CFBError('PBP contains null game_id')
    cutoff = iso_utc(cutoff)
    done = flag(s['completed'])
    eligible = flag(s['fbs_participant']) & ~((s['home_division'] == 'fcs') & (s['away_division'] == 'fcs'))
    history = s.loc[eligible & done & (s['_kickoff'] < cutoff)].copy()
    future = s.loc[eligible & ~done & (s['_kickoff'] > cutoff) & s['start_time_tbd'].fillna(True).eq(False)].copy()
    past_plays = pbp.loc[pbp['game_id'].isin(history['game_id'])].copy()
    return history, future, past_plays

def normalize_pbp_teams(pbp, history):
    """Join actual ESPN home/away labels to stable schedule IDs by game_id.

    No fuzzy matches or team-name guessing. A vendor mismatch is an error, not
    grounds to silently assign all yards/turnovers to the wrong school.
    """
    require(pbp, REQUIRED_PBP, 'pbp')
    if pbp.empty:
        return pbp.assign(pos_team_id=pd.Series(dtype='Int64'), def_pos_team_id=pd.Series(dtype='Int64'))
    # Kickoffs, punts and dead-ball plays may have no unambiguous possession team.
    # They are not included in pass/rush efficiency, so restrict validation to
    # play rows that actually contribute to the model.
    plays = pbp.loc[flag(pbp['pass']) ^ flag(pbp['rush'])].copy()
    ids = history[['game_id', 'home_id', 'away_id']].copy()
    data = plays.merge(ids, on='game_id', how='left', validate='many_to_one', indicator=True)
    if (data['_merge'] != 'both').any() or data[['home_id', 'away_id']].isna().any().any():
        raise CFBError('PBP game without matching schedule game ID or team IDs')
    if data[['home', 'away', 'pos_team', 'def_pos_team']].isna().any().any():
        raise CFBError('Missing ESPN team identity')
    # Only the SAME provider's 'home' and 'away' labels are compared with pos_team.
    ph = data['pos_team'].eq(data['home'])
    pa = data['pos_team'].eq(data['away'])
    dh = data['def_pos_team'].eq(data['home'])
    da = data['def_pos_team'].eq(data['away'])
    if not ((ph ^ pa) & (dh ^ da) & ph.eq(da) & pa.eq(dh)).all():
        raise CFBError('PBP team cannot be unambiguously mapped using ESPN home/away labels')
    data['pos_team_id'] = data['home_id'].where(ph, data['away_id'])
    data['def_pos_team_id'] = data['home_id'].where(dh, data['away_id'])
    return data.drop(columns=['_merge'])

def play_stats(rows, kind):
    if rows.empty:
        return {'epa': None, 'success': None, 'explosive': None}
    down = pd.to_numeric(rows['down'], errors='coerce')
    dist = pd.to_numeric(rows['distance'], errors='coerce')
    yards = pd.to_numeric(rows['yards_gained'], errors='coerce')
    if down.isna().any() or dist.isna().any() or yards.isna().any():
        success = explosive = None
    else:
        # 50% first down, 70% second down, 100% third/fourth. Explicit
        # definition different from EPA>0 provider 'success' statistic.
        target = dist.where(down >= 3, dist * .7).where(down != 1, dist * .5)
        success = float((yards >= target).mean())
        explosive = float((yards >= (20 if kind == 'pass' else 10)).mean())
    return {'epa': mean(rows['EPA']), 'success': success, 'explosive': explosive}

def drive_stats(rows, games):
    default = {'drives_per_game': None, 'plays_per_drive': None,
               'avg_start_yardline': None, 'redzone_td_rate': None, 'redzone_trips': None}
    if rows.empty or games < 1:
        return default
    drives = rows.dropna(subset=['drive_id']).copy()
    if drives.empty:
        return default
    drives['_key'] = drives['game_id'].astype(str) + '/' + drives['drive_id'].astype(str)
    # PBP has chronological game_play_number; scrimmage-only drive count is
    # descriptive and omits kickoff/punt/penalty-only possessions.
    starts = drives.sort_values(['game_id', 'game_play_number']).groupby('_key', sort=False).head(1)
    start_yards = pd.to_numeric(starts['yards_to_goal'], errors='coerce')
    start = mean(100-start_yards) if start_yards.notna().all() else None
    positions = pd.to_numeric(drives['yards_to_goal'], errors='coerce')
    if positions.isna().any() or drives[['offense_score_play', 'td_play']].isna().any().any():
        n = rz = None
    else:
        drives['_rz'] = positions <= 20
        drives['_td'] = flag(drives['offense_score_play']) & flag(drives['td_play'])
        groups = drives.groupby('_key', sort=False)[['_rz', '_td']].any()
        opportunities = groups.loc[groups['_rz']]
        n = int(len(opportunities))
        rz = float(opportunities['_td'].mean()) if n else None
    return {'drives_per_game': float(drives['_key'].nunique()/games),
            'plays_per_drive': float(len(drives)/drives['_key'].nunique()),
            'avg_start_yardline': start, 'redzone_td_rate': rz, 'redzone_trips': n}

def opponent_averages(team, off, defense, all_plays):
    """Leave-opponent-game-out SOS: historical opponents' rates versus OTHER teams.

    This is deliberately unestimated when an opponent has no other games.
    """
    result = {}
    for prefix, cohort, oppcol, vscol in (
        ('def', off, 'def_pos_team_id', 'pos_team_id'),
        ('off', defense, 'pos_team_id', 'def_pos_team_id')):
        for kind in ('pass', 'rush'):
            w = []
            for opponent, sample in cohort.loc[flag(cohort[kind])].groupby(oppcol):
                others = all_plays.loc[(all_plays[oppcol] == opponent) &
                                        (all_plays[vscol] != team) & flag(all_plays[kind])]
                epa = mean(others['EPA'])
                if epa is None:
                    w = []
                    break
                w.append((epa,len(sample)))
            result[f'{prefix}_{kind}'] = (sum(a*n for a,n in w)/sum(n for _,n in w)) if w else None
    return {'opp_def_pass_epa_allowed_avg': result['def_pass'],
            'opp_def_rush_epa_allowed_avg': result['def_rush'],
            'opp_off_pass_epa_avg': result['off_pass'],
            'opp_off_rush_epa_avg': result['off_rush']}

def team_features(team_id, pbp):
    """Observed stats only. Missing pressure, personnel and QB rushing stay null."""
    require(pbp, ('pos_team_id','def_pos_team_id'), 'mapped pbp')
    clean = pbp.loc[flag(pbp['pass']) ^ flag(pbp['rush'])].copy()
    offense = clean.loc[clean['pos_team_id'] == team_id]
    defense = clean.loc[clean['def_pos_team_id'] == team_id]
    games = int(offense['game_id'].nunique())
    def_games = int(defense['game_id'].nunique())
    op,orr = offense.loc[flag(offense['pass'])],offense.loc[flag(offense['rush'])]
    dp,dr = defense.loc[flag(defense['pass'])],defense.loc[flag(defense['rush'])]
    p, r, pd_, rd_ = play_stats(op,'pass'),play_stats(orr,'rush'),play_stats(dp,'pass'),play_stats(dr,'rush')
    od,dd = drive_stats(offense,games),drive_stats(defense,def_games)
    def third(d):
        part = d.loc[d['down'] == 3]
        if part.empty or part[['firstD_by_yards','firstD_by_penalty','rush_td','pass_td']].isna().any().any():
            return None
        return float((flag(part['firstD_by_yards']) | flag(part['firstD_by_penalty']) |
                      flag(part['rush_td']) | flag(part['pass_td'])).mean())
    # Don't interpret 'turnover' as lost fumble: it also includes interceptions.
    # ESPN released schema has 'fumble_vec' (not equivalent to LOST fumbles).
    features = {
      'offense': {'games': games or None,'plays': len(offense) or None,
        'epa_pass':p['epa'],'epa_rush':r['epa'],
        'pass_rate': len(op)/len(offense) if len(offense) else None,
        'success_pass':p['success'],'success_rush':r['success'],
        'explosive_pass_rate':p['explosive'],'explosive_rush_rate':r['explosive'],
        'pressure_allowed':None,'sack_allowed':boolean_rate(op,'sack'),
        'interception_rate':boolean_rate(op,'int'),'fumble_lost_rate':None,
        'redzone_td_rate':od['redzone_td_rate'],'redzone_trips':od['redzone_trips'],
        'third_down_rate':third(offense),'drives_per_game':od['drives_per_game'],
        'plays_per_drive':od['plays_per_drive'],
        'qb_rushes_per_game':None,'qb_rush_epa':None},
      'defense': {'plays':len(defense) or None,
        'epa_pass_allowed':pd_['epa'],'epa_rush_allowed':rd_['epa'],
        'success_pass_allowed':pd_['success'],'success_rush_allowed':rd_['success'],
        'explosive_pass_allowed':pd_['explosive'],'explosive_rush_allowed':rd_['explosive'],
        'pressure_generated':None,'sack_generated':boolean_rate(dp,'sack'),
        'interception_rate_forced':boolean_rate(dp,'int'),
        'fumble_recovery_rate':None,'redzone_td_allowed':dd['redzone_td_rate'],
        'redzone_trips_faced':dd['redzone_trips'],'third_down_allowed':third(defense),
        'drives_faced_per_game':dd['drives_per_game'],
        'plays_per_drive_allowed':dd['plays_per_drive']},
      'schedule':opponent_averages(team_id,offense,defense,clean),
      'context':{'avg_start_yardline':od['avg_start_yardline'],
        'penalty_epa_per_game':None,'special_teams_epa_per_game':None}}
    return features

def missing_features(obj, prefix=''):
    errors = []
    for key,value in obj.items():
        name = prefix+'.'+key if prefix else key
        if isinstance(value,dict):
            errors += missing_features(value,name)
        elif value is None:
            errors.append(name)
    return errors

def report(folder, output, window_days=7):
    manifest = verify(folder)
    cutoff = iso_utc(manifest['captured_at_utc'])
    root = Path(folder)
    schedule = pd.read_parquet(root/'schedule.parquet')
    pbp = pd.read_parquet(root/'pbp.parquet')
    require(schedule, REQUIRED_SCHEDULE, 'actual schedule file')
    require(pbp, REQUIRED_PBP, 'actual PBP file')
    history, future, eligible = split_games(schedule,pbp,cutoff)
    mapped = normalize_pbp_teams(eligible,history)
    out = Path(output)
    out.mkdir(parents=True,exist_ok=True)
    results = []
    teams = {}
    for _,game in future.iterrows():
        if game['_kickoff'] > cutoff+timedelta(days=window_days):
            continue
        for team in (game['home_id'],game['away_id']):
            if team not in teams:
                teams[team] = team_features(team,mapped)
        home,away = teams[game['home_id']],teams[game['away_id']]
        obj = {'schema':'cfb_pregame_evidence_v2','status':'INCOMPLETE_CORE_INPUTS_NO_ATS_PICK',
          'game_id':str(game['game_id']),'kickoff_utc':game['_kickoff'].isoformat(),
          'snapshot_captured_at_utc':cutoff.isoformat(),
          'home':{'id':str(game['home_id']),'name':game['home_team'],
                  'division':game['home_division'],'features':home},
          'away':{'id':str(game['away_id']),'name':game['away_team'],
                  'division':game['away_division'],'features':away},
          'neutral_site': bool(game['neutral_site']) if pd.notna(game['neutral_site']) else None,
          'missing':{'home':missing_features(home),'away':missing_features(away)},
          'additional_required':['Free consistently-defined full pressure charting',
             'Verified QB rushing identities and per-play EPA',
             'Game-day injuries and confirmed lineups',
             'Dated signed betting spread and American odds'],
          'market_status':'NO_TIMESTAMPED_VERIFIED_QUOTE',
          'source_sha256':{k:manifest['datasets'][k]['sha256'] for k in ASSETS},
          'historical_games_used':int(len(history)),
          'limitations':['Model grade and ATS probability not computed',
                         'Scrimmage-only drive count may omit non-scrimmage drives',
                         '2026 snapshot cannot be retrospectively used as-of a prior game',
                         'Unverified missing stats are null, never substituted']}
        (out/(str(game['game_id'])+'.json')).write_text(json.dumps(obj,indent=2,default=str)+'\n')
        results.append({'game_id':obj['game_id'],'home':game['home_team'],
            'away':game['away_team'],'missing_count':len(obj['missing']['home'])+len(obj['missing']['away'])})
    index = {'schema':'cfb_snapshot_report_v2','captured_at_utc':cutoff.isoformat(),
             'completed_games_used':int(len(history)),'upcoming_games':results,
             'status':'UNRATED_DATA_AUDIT_ONLY',
             'validation':{'download_schema_confirmed_from_actual_files':True,
                 'raw_file_rows':int(len(pbp)),'eligible_play_rows':int(len(mapped))}}
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    return index

def inspect(folder, output):
    """Write actual parquet field/dtype audit before any normalization."""
    import pyarrow.parquet as pq
    manifest=verify(folder)
    result={'schema':'cfb_actual_parquet_schema_audit_v1',
            'snapshot_at':manifest['captured_at_utc'],'datasets':{},
            'status':'ACTUAL_FILES_CHECKED'}
    for kind in ASSETS:
        file=Path(folder)/(kind+'.parquet')
        pqfile=pq.ParquetFile(file)
        fields={f.name:str(f.type) for f in pqfile.schema_arrow}
        required=REQUIRED_PBP if kind=='pbp' else REQUIRED_SCHEDULE if kind=='schedule' else ()
        result['datasets'][kind]={'rows':int(pqfile.metadata.num_rows),
            'columns':fields, 'missing_required':sorted(set(required)-set(fields)),
            'sha256':manifest['datasets'][kind]['sha256'],
            'captured_at_utc':manifest['datasets'][kind]['captured_at_utc']}
    result['status']=('MISSING_REQUIRED_COLUMNS' if any(x['missing_required'] for x in result['datasets'].values())
                      else 'SCHEMA_REQUIRED_FIELDS_PRESENT')
    out=Path(output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd', required=True)
    a=sub.add_parser('collect');a.add_argument('--season',required=True,type=int);a.add_argument('--out',required=True)
    v=sub.add_parser('verify');v.add_argument('--snapshot',required=True)
    i=sub.add_parser('inspect');i.add_argument('--snapshot',required=True);i.add_argument('--out',required=True)
    r=sub.add_parser('report');r.add_argument('--snapshot',required=True);r.add_argument('--out',required=True)
    r.add_argument('--days',type=int,default=7)
    args=p.parse_args()
    if args.cmd=='collect':result=capture(args.season,args.out)
    elif args.cmd=='verify':result=verify(args.snapshot)
    elif args.cmd=='inspect':result=inspect(args.snapshot,args.out)
    else:result=report(args.snapshot,args.out,args.days)
    print(json.dumps(result,indent=2))
    if isinstance(result,dict) and result.get('status')=='MISSING_REQUIRED_COLUMNS':
        raise SystemExit(2)
if __name__=='__main__':main()
