#!/usr/bin/env python3
"""Free, optional NFL weekly injury-report snapshot; not confirmed inactives.

The source's row-level date_modified is checked before producing summaries.
Retrieval time establishes observation, not when the NFL originally published data.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from nflverse_connector import to_pandas
from collect_extra_metrics import parse_stamp

SOURCE='nflreadpy.load_injuries([season])'
REQUIRED={'season','week','team','report_status','date_modified'}

def iso(now=None):
    d=now or datetime.now(timezone.utc)
    if d.tzinfo is None:raise ValueError('Naive capture time')
    return d.astimezone(timezone.utc).isoformat().replace('+00:00','Z')

def collect(season:int, out:Path, nfl_module=None, now=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    timestamp=iso(now)
    manifest={'schema':'injury_feed_v0.7','season':season,'retrieved_at_utc':timestamp,'source':SOURCE,'status':'UNAVAILABLE','file':None,'sha256':None,'rows':0,'observations':[], 'note':'Practice/game status is NOT a confirmed game-day inactive list and has NO automatically assigned point impact.'}
    try:
        nfl=nfl_module if nfl_module is not None else importlib.import_module('nflreadpy')
        df=to_pandas(nfl.load_injuries([season]))
        absent=REQUIRED-set(df.columns)
        if absent:raise ValueError('Missing injury columns: '+str(sorted(absent)))
        if df.empty:raise ValueError('No injury records received')
        if pd.to_numeric(df.season,errors='coerce').ne(season).any():raise ValueError('Injury feed season mismatch')
        # An unlabeled timezone is NOT a trustworthy as-of timestamp.
        def explicit_utc(v):
            try:return pd.Timestamp(parse_stamp(str(v)))
            except (ValueError,TypeError):return pd.NaT
        # Only after the remote download completes can we claim possession.
        timestamp=iso(now)
        updates=pd.to_datetime(df.date_modified.map(explicit_utc),utc=True,errors='coerce')
        # Drop rows whose per-record modification lies after collection time,
        # or whose as-of time is UNKNOWN. Never call these verified.
        safe=updates.notna() & (updates <= pd.Timestamp(timestamp))
        eligible=df.loc[safe].copy();excluded=int((~safe).sum())
        outfile=out/f'injuries_{season}.csv.gz'
        eligible.to_csv(outfile,index=False,compression='gzip')
        digest=hashlib.sha256(outfile.read_bytes()).hexdigest()
        summary=[]
        for (team,week),g in eligible.groupby(['team','week'],dropna=True):
            valid=g[g.report_status.notna() & g.report_status.astype('string').str.strip().ne('')]
            summary.append({'team':str(team),'week':int(week),'game_status_rows':int(len(valid))})
        manifest.update({'status':'COLLECTED' if len(eligible) else 'CAPTURED_ROWS_UNVERIFIED_ASOF','file':outfile.name,'sha256':digest,'rows':int(len(eligible)),'excluded_undated_or_future_rows':excluded,'observations':summary})
    except Exception as exc:
        manifest['warning']=f'{type(exc).__name__}: {exc}'
    # Finalization can occur after download / serialization. Never backdate it.
    manifest['retrieved_at_utc']=iso(now)
    (out/'injury_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--season',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(collect(a.season,a.out),indent=2))
