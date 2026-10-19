#!/usr/bin/env python3
"""Archive free nflverse injury-report observations for upcoming NFL games.

A row's date_modified is optional in newer feeds. If no trustworthy per-row
publication timestamp exists, preserve the downloaded report with its actual
capture time and hash, but mark the rows SNAPSHOT-ONLY rather than fabricating
an original publication time or claiming confirmed game-day inactives.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from nflverse_connector import to_pandas
from collect_extra_metrics import parse_stamp

SOURCE = 'nflreadpy.load_injuries([season])'
REQUIRED = {'season', 'week', 'team', 'report_status'}
# Use an alternate field only when it actually exists AND is timezone-aware.
# Injury date (a day without a clock/zone) is not a row's publication time.
ROW_TIMESTAMP_COLUMNS = (
    'date_modified', 'last_updated_utc', 'updated_at_utc',
    'report_updated_at_utc', 'last_updated', 'updated_at', 'report_updated_at',
)


def iso(now=None):
    d = now if now is not None else datetime.now(timezone.utc)
    if d.tzinfo is None:
        raise ValueError('Naive capture time')
    return d.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def _explicit_row_time(v):
    """Accept only explicit timezone/offset timestamps, never guess UTC."""
    if v is None or pd.isna(v):
        return pd.NaT
    try:
        return pd.Timestamp(parse_stamp(str(v)))
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def collect(season: int, out: Path, nfl_module=None, now=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema': 'injury_feed_v0.7.1', 'season': season,
        'retrieved_at_utc': iso(now), 'source': SOURCE, 'status': 'UNAVAILABLE',
        'file': None, 'sha256': None, 'rows': 0, 'observations': [],
        'row_timestamp_column': None, 'timestamp_provenance': 'UNAVAILABLE',
        'note': 'Report observations are NOT confirmed game-day inactives; no injury point adjustment is applied.',
    }
    try:
        nfl = nfl_module if nfl_module is not None else importlib.import_module('nflreadpy')
        df = to_pandas(nfl.load_injuries([season]))
        absent = REQUIRED - set(df.columns)
        if absent:
            raise ValueError('Missing required injury columns: ' + str(sorted(absent)))
        if df.empty:
            raise ValueError('No injury records received')
        if pd.to_numeric(df['season'], errors='coerce').ne(season).any():
            raise ValueError('Injury feed season mismatch')
        weeks = pd.to_numeric(df['week'], errors='coerce')
        if weeks.isna().any() or (~weeks.between(1, 22)).any() or (weeks != weeks.astype(int)).any():
            raise ValueError('Injury feed contains invalid weeks')
        if df['team'].isna().any() or df['team'].astype('string').str.strip().eq('').any():
            raise ValueError('Injury feed contains missing team identifiers')

        # Capture after the source download finishes. The saved-file time is
        # corrected again after writing, never backdated.
        captured = pd.Timestamp(iso(now))
        timestamp_field = next((k for k in ROW_TIMESTAMP_COLUMNS if k in df.columns), None)
        manifest['source_columns'] = list(df.columns)
        manifest['row_timestamp_column'] = timestamp_field
        if timestamp_field:
            updates = pd.to_datetime(df[timestamp_field].map(_explicit_row_time), utc=True, errors='coerce')
            safe = updates.notna() & (updates <= captured)
            eligible = df.loc[safe].copy()
            excluded = int((~safe).sum())
            status = 'COLLECTED' if len(eligible) else 'CAPTURED_ROWS_UNVERIFIED_ASOF'
            provenance = 'SOURCE_ROW_TIME_EXPLICIT_TZ_VERIFIED' if len(eligible) else 'NO_VALID_ROW_TIMESTAMPS'
        else:
            # The current injury feed can omit date_modified. We really did
            # observe these rows at collection time, but do NOT know when the
            # publisher last changed each row. Preserve the full snapshot.
            eligible = df.copy()
            excluded = 0
            status = 'CAPTURED_SNAPSHOT_ONLY'
            provenance = 'SNAPSHOT_OBSERVED_NO_ROW_PUBLICATION_TIME'

        outfile = out / f'injuries_{season}.csv.gz'
        eligible.to_csv(outfile, index=False, compression='gzip')
        summary = []
        for (team, week), group in eligible.groupby(['team', 'week'], dropna=True):
            valid = group[group['report_status'].notna() & group['report_status'].astype('string').str.strip().ne('')]
            summary.append({'team': str(team), 'week': int(week), 'game_status_rows': int(len(valid))})

        manifest.update({
            'status': status, 'timestamp_provenance': provenance,
            'file': outfile.name, 'sha256': hashlib.sha256(outfile.read_bytes()).hexdigest(),
            'rows': int(len(eligible)), 'excluded_undated_or_future_rows': excluded,
            'observations': summary,
        })
        if timestamp_field is None:
            manifest['warning'] = 'No per-row update timestamp in source. Capture time verifies possession of this snapshot only, not original report publication.'
        elif excluded:
            manifest['warning'] = f'{excluded} rows excluded because their update time was missing, timezone-ambiguous, or after capture.'
    except Exception as exc:
        manifest['warning'] = f'{type(exc).__name__}: {exc}'
    manifest['retrieved_at_utc'] = iso(now)
    (out / 'injury_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--season', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(collect(args.season, args.out), indent=2))
