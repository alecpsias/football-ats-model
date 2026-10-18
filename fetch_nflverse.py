#!/usr/bin/env python3
"""Download nflverse inputs and freeze timestamp/hash for reproducible snapshots.

This is an OPTIONAL online collection script.  For backtests, a snapshot
captured AFTER the game's cutoff is NOT proof of pregame data availability.
"""
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://github.com/nflverse/nflverse-data/releases/download"

def fetch(season: int, folder: Path) -> dict:
    if season < 1999 or season > datetime.now(timezone.utc).year + 1:
        raise ValueError("Season outside supported download range")
    files = {
        "pbp":f"{BASE}/pbp/play_by_play_{season}.parquet",
        "schedule":f"{BASE}/schedules/games.parquet",
        "weekly_rosters":f"{BASE}/weekly_rosters/roster_weekly_{season}.parquet",
    }
    folder.mkdir(parents=True, exist_ok=True)
    summary = {"season":season,"assets":{},
               "warning":"Downloaded-at timestamps are not historical publication proofs. Late source snapshots must not be reused as strict earlier pregame inputs."}
    for key,url in files.items():
        target = folder/Path(url).name
        digest = hashlib.sha256()
        byte_count = 0
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"football-ats-ingest/0.2"})
            with urllib.request.urlopen(req, timeout=90) as src, target.with_suffix(target.suffix+".partial").open("wb") as out:
                while True:
                    block=src.read(2**20)
                    if not block: break
                    out.write(block)
                    digest.update(block)
                    byte_count += len(block)
            target.with_suffix(target.suffix+".partial").replace(target)
            summary["assets"][key] = {"path":str(target),"source_url":url,
                                      "sha256":digest.hexdigest(),"bytes":byte_count,
                                      "captured_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
                                      "download_status":"SUCCESS"}
        except Exception as exc:
            target.with_suffix(target.suffix+".partial").unlink(missing_ok=True)
            summary["assets"][key] = {"source_url":url,"download_status":"FAILED","error":str(exc)}
    (folder/"download_manifest.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return summary

if __name__=="__main__":
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--season",type=int,required=True)
    pa.add_argument("--out",type=Path,required=True)
    a=pa.parse_args()
    s=fetch(a.season,a.out)
    for k,v in s["assets"].items():
        print(k,v["download_status"],v.get("path",v.get("error","")))
    print("Manifest:", a.out/"download_manifest.json")
