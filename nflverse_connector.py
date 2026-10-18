#!/usr/bin/env python3
"""Permitted, FREE nflverse -> immutable local snapshots for football_ats_model.

Live mode uses the official `nflreadpy` loader (no HTML scraping, accounts,
paid API keys, or commercial data).  Produces ingest.py-compatible files.
This connector intentionally DOES NOT label a backfilled snapshot 'pregame'.
An as-of test compares actual file-capture time with the game's cutoff.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

LICENSE_INFO = {
    "project": "nflverse",
    "project_url": "https://github.com/nflverse/nflverse-data",
    "loader": "https://github.com/nflverse/nflreadpy",
    "notice": "nflverse datasets are separately licensed; check each dataset's current license and attribution obligations before redistributing.",
}

DATASETS = {
    "pbp": {"function": "load_pbp", "origin": "nflverse-data / pbp", "required": True},
    "schedule": {"function": "load_schedules", "origin": "nflverse-data / schedules", "required": True},
    "roster": {"function": "load_rosters_weekly", "origin": "nflverse-data / weekly_rosters", "required": True},
    "player_stats": {"function": "load_player_stats", "origin": "nflverse-data / player_stats", "required": False},
}

class CollectionError(Exception):
    pass


def iso_utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise CollectionError("Capture timestamp requires an explicit timezone")
    return moment.astimezone(timezone.utc).isoformat().replace('+00:00','Z')


def to_pandas(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        try:
            result = value.to_pandas()
            if isinstance(result, pd.DataFrame):
                return result
        except ImportError:
            pass  # Polars -> pandas sometimes requires optional pyarrow
    if hasattr(value, "to_dicts"):
        return pd.DataFrame(value.to_dicts())
    raise CollectionError("nflreadpy did not return a Polars/Pandas DataFrame")


def standardize(kind: str, df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Normalize structural fields only; do NOT invent absent football metrics."""
    if df.empty:
        raise CollectionError(f"{kind}: empty dataset")
    df = df.copy()
    if kind == "schedule":
        required = {"game_id", "home_team", "away_team", "gameday", "gametime"}
        absent = required - set(df.columns)
        if absent and not ({"game_id", "home_team", "away_team", "kickoff_utc"} <= set(df.columns)):
            raise CollectionError(f"schedule missing kickoff fields: {sorted(absent)}")
    elif kind == "pbp":
        essential = {"game_id", "play_id", "posteam", "defteam", "epa", "qb_dropback", "pass", "rush", "success"}
        absent = essential - set(df.columns)
        if absent:
            raise CollectionError(f"pbp missing essential columns: {sorted(absent)}")
    elif kind == "roster":
        if "player_id" not in df and "gsis_id" in df:
            df = df.rename(columns={"gsis_id": "player_id"})
        req = {"player_id", "position"}
        if req - set(df.columns):
            raise CollectionError(f"roster missing QB-position join columns: {sorted(req-set(df.columns))}")
        # Weekly rosters can repeat player IDs: preserve all source rows and
        # deduplicate only on read for QB-position membership in ingest.py.
    if "season" in df.columns:
        df = df.loc[pd.to_numeric(df.season, errors="coerce").eq(season)].copy()
        if df.empty:
            raise CollectionError(f"{kind}: no rows for season {season}")
    if kind == "pbp":
        if df["game_id"].isna().any() or df["play_id"].isna().any():
            raise CollectionError("pbp contains null play identifiers")
        if df.duplicated(["game_id", "play_id"]).any():
            raise CollectionError("pbp has duplicate game_id/play_id")
    return df


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collect(
    season: int,
    root: Path,
    *,
    nfl_module: Any = None,
    capture_now=None,
    include_player_stats: bool = False,
) -> dict:
    """Collect an atomic, timestamped free snapshot. No synthetic fallbacks.

    root is the chosen UNIQUE final snapshot directory. It must not exist.
    For tests, nfl_module may be a deterministic stub with documented APIs.
    """
    if not (1999 <= season <= 2100):
        raise CollectionError('season must be in [1999,2100]')
    if nfl_module is None:
        try:
            nfl_module = importlib.import_module('nflreadpy')
        except ImportError as exc:
            raise CollectionError('Install the free connector: pip install -r requirements-nflverse.txt') from exc
    now = capture_now or (lambda: datetime.now(timezone.utc))
    root = Path(root)
    if root.exists():
        raise CollectionError(f'Refusing to overwrite any snapshot: {root}')
    root.mkdir(parents=True, exist_ok=False)
    selected = ('pbp', 'schedule', 'roster', 'player_stats') if include_player_stats else ('pbp', 'schedule', 'roster')
    output = {
        'schema_version': '1.0', 'connector_version': '0.3.0', 'season': season,
        'created_at_utc': iso_utc(now()), 'sources': LICENSE_INFO,
        'datasets': {}, 'status': 'IN_PROGRESS',
        'asof_policy': 'Capture time proves local possession only. A snapshot captured after a historical cutoff is NOT strict pregame evidence.',
        'may_require_manual': ['contemporaneous pressure/participation charting','official verified injuries and game status','market spread, odds, price capture','penalty and special teams EPA'],
    }
    for key in selected:
        spec = DATASETS[key]
        try:
            start = iso_utc(now())
            function = getattr(nfl_module, spec['function'])
            frame = standardize(key, to_pandas(function([season])), season)
            file_path = root / f'{key}_{season}.csv.gz'
            partial = root / f'{key}_{season}.partial.csv.gz'
            frame.to_csv(partial, index=False, compression='gzip')
            os.replace(partial, file_path)
            finish = iso_utc(now())
            output['datasets'][key] = {
                'status': 'SUCCESS', 'origin':spec['origin'],
                'retrieval_method':f'nflreadpy.{spec["function"]}([season])',
                'local_file':file_path.name, 'retrieval_started_at_utc':start,
                'captured_at_utc':finish, 'row_count':int(len(frame)),
                'column_count':int(len(frame.columns)),
                'sha256':_sha256(file_path), 'bytes':file_path.stat().st_size,
                'games_count':int(frame.game_id.nunique()) if 'game_id' in frame.columns else None,
            }
        except Exception as e:
            output['datasets'][key] = {'status':'FAILED', 'origin':spec['origin'],
                                       'error':f'{type(e).__name__}: {e}'}
    required_ok = all(output['datasets'].get(key,{}).get('status')=='SUCCESS' for key in DATASETS if DATASETS[key]['required'])
    output['status'] = 'COLLECTED' if required_ok else 'INCOMPLETE'
    output['finalized_at_utc'] = iso_utc(now())
    (root/'manifest.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    return output


def verify(root: Path, cutoff: str | None = None) -> dict:
    """Check immutable file hashes, timestamps and historical as-of eligibility."""
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    problems=[]
    verified=[]
    for name, d in manifest['datasets'].items():
        if d['status']!='SUCCESS':
            problems.append(f'{name}: download failed: {d.get("error", "unknown")}')
            continue
        path=root/d['local_file']
        if not path.is_file():
            problems.append(f'{name}: missing local file')
            continue
        if _sha256(path)!=d['sha256'] or path.stat().st_size!=d['bytes']:
            problems.append(f'{name}: checksum or file size mismatch')
            continue
        verified.append(name)
    asof='NOT_EVALUATED'
    if cutoff is not None:
        end = datetime.fromisoformat(cutoff.replace('Z','+00:00'))
        if end.tzinfo is None:
            raise CollectionError('cutoff must be timezone aware')
        times=[datetime.fromisoformat(d['captured_at_utc'].replace('Z','+00:00'))
               for d in manifest['datasets'].values() if d['status']=='SUCCESS']
        if times and all(t <= end for t in times) and not problems and manifest['status']=='COLLECTED':
            asof='CAPTURED_BEFORE_CUTOFF'
        else:
            asof='NOT_VERIFIED_PREGAME'
    required = all(k in verified for k in ('pbp','schedule','roster'))
    return {'status':'VERIFIED' if not problems and required else 'BLOCKED',
            'snapshot_directory':str(root),'season':manifest['season'],
            'verified_datasets':verified,'problems':problems,
            'pregame_eligibility':asof,
            'note':'A locally verified checksum does not establish that a source was historically published before an earlier cutoff.'}


def main():
    parser=argparse.ArgumentParser(description='Free nflverse data collection and verification')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('collect',help='Download free datasets into a NEW snapshot directory')
    p.add_argument('--season',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--include-player-stats',action='store_true')
    p=sub.add_parser('verify',help='Check hashes and pregame eligibility')
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--cutoff',help='UTC ISO8601 prediction cutoff')
    args=parser.parse_args()
    if args.command=='collect':
        result=collect(args.season,args.out,include_player_stats=args.include_player_stats)
        print(json.dumps(result,indent=2))
        if result['status']!='COLLECTED': sys.exit(2)
    else:
        result=verify(args.snapshot,args.cutoff)
        print(json.dumps(result,indent=2))
        if result['status']!='VERIFIED': sys.exit(2)

if __name__=='__main__':
    try: main()
    except (CollectionError,OSError,ValueError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(2)
