#!/usr/bin/env python3
"""Ingest independently obtained free nflverse CSV files when live GitHub access is blocked.

No network, nflreadpy, fee, or fictitious data. Input files MUST be actual
nflverse releases. This code can validate format and hash locally; it cannot
retroactively prove pre-kickoff public availability.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
from nflverse_connector import CollectionError, LICENSE_INFO, _sha256, iso_utc, standardize
import json
import os

EXPECTED = {'pbp': 'play_by_play_2026.csv.gz', 'schedule': 'games.csv', 'roster': 'roster_weekly_2026.csv.gz'}


def _load(path: Path) -> pd.DataFrame:
    # Explicitly accepts the project's free CSV family without pyarrow or Polars.
    if not path.is_file():
        raise CollectionError('Missing input file: ' + str(path))
    lower = path.name.lower()
    if lower.endswith(('.csv', '.csv.gz')):
        return pd.read_csv(path, low_memory=False)
    raise CollectionError('Unsupported free import type: use official CSV / CSV.GZ release, not a webpage or HTML')


def import_files(season: int, root: Path, sources: dict[str, Path], *, recorded_source_urls: dict[str, str] | None = None, now=None) -> dict:
    root = Path(root)
    if root.exists():
        raise CollectionError('Refusing to overwrite existing snapshot: ' + str(root))
    if set(sources) != {'pbp', 'schedule', 'roster'}:
        raise CollectionError('pbp, schedule, roster are all required')
    # Pre-validate completely before making a snapshot directory.
    loaded = {key: standardize(key, _load(Path(path)), season) for key, path in sources.items()}
    now = now or (lambda: datetime.now(timezone.utc))
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        'schema_version': '1.0', 'connector_version': '0.3.1-local', 'season': season,
        'created_at_utc': iso_utc(now()), 'sources': LICENSE_INFO,
        'asof_policy': 'Imported-file collection proves local copy time; source publication before a historical cutoff NOT established.',
        'status': 'IN_PROGRESS', 'datasets': {},
        'may_require_manual': ['independently archived pregame releases', 'pressure charting', 'injury timestamps', 'market timestamps and prices', 'penalty/special teams EPA'],
    }
    for key, frame in loaded.items():
        start = iso_utc(now())
        dest = root / f'{key}_{season}.csv.gz'
        tmp = root / f'{key}_{season}.partial.csv.gz'
        frame.to_csv(tmp, compression='gzip', index=False)
        os.replace(tmp, dest)
        manifest['datasets'][key] = {
            'status': 'SUCCESS', 'origin': 'nflverse provided free file',
            'retrieval_method': 'local CSV import (nflreadpy not required)',
            'upstream_url_user_supplied': (recorded_source_urls or {}).get(key),
            'original_local_filename': str(sources[key]),
            'original_sha256': _sha256(Path(sources[key])),
            'local_file': dest.name, 'retrieval_started_at_utc': start,
            'captured_at_utc': iso_utc(now()), 'row_count': int(len(frame)),
            'column_count': int(len(frame.columns)), 'sha256': _sha256(dest),
            'bytes': dest.stat().st_size,
            'games_count': int(frame.game_id.nunique()) if 'game_id' in frame else None,
        }
    manifest['status'] = 'COLLECTED'
    manifest['finalized_at_utc'] = iso_utc(now())
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--season', type=int, default=2026)
    p.add_argument('--pbp', type=Path, required=True)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--roster', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    m = import_files(args.season, args.out, {'pbp': args.pbp, 'schedule': args.schedule, 'roster': args.roster})
    print(json.dumps({'status': m['status'], 'files': {k: v['row_count'] for k,v in m['datasets'].items()}, 'manifest': str(args.out / 'manifest.json')}, indent=2))

if __name__ == '__main__':
    main()
