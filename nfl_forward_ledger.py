#!/usr/bin/env python3
"""Prospective NFL scoring of truly frozen forecasts from GitHub Actions artifacts.

Download-only data path: existing pregame forecast bytes plus later official
nflverse schedule finals. No retroactive rerunning on target-game box scores.
No signed bookmaker quotes => no ATS betting record.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import zipfile

import pandas as pd
import requests

from evaluate_locked_forecasts import evaluate
from model import InputError

# Read immutable archived v0.9/v0.10 and current v0.11 artifacts. Only one candidate from
# each frozen game is evaluated; historical version is explicitly labeled.
ARTIFACT_PREFIX = ("football-ats-v09-", "football-ats-v10-", "football-ats-v11-")
FORECAST_ROOT = "future-predictions/"


def extract_forecasts(archive_bytes: bytes, destination: Path, archive_id: int) -> list[Path]:
    """Ingest valid pregame JSON, not index files or schedule/results in artifact."""
    output = []
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        for n, member in enumerate(z.infolist()):
            path = member.filename.replace("\\", "/")
            if member.is_dir() or FORECAST_ROOT not in path:
                continue
            if not path.lower().endswith(".json") or path.endswith("/index.json"):
                continue
            # Fail closed on malformed/non-pregame records, with no path reuse.
            try:
                raw = z.read(member)
                rec = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            if rec.get("status") != "EXPLORATORY_CORE_PROJECTION_UNCALIBRATED":
                continue
            if not all(k in rec for k in
                       ("game_id", "kickoff_utc", "data_as_of_utc", "home", "away")):
                continue
            out = destination / f"artifact-{archive_id}-forecast-{n}.json"
            out.write_bytes(raw)  # Exact frozen bytes retained for SHA-256 ledger.
            output.append(out)
    return output


def list_free_pregame_artifacts(repo: str, token: str, limit: int = 25) -> list[dict]:
    """Only artifacts produced by our NFL pregame workflow. Never download CFB."""
    if not token:
        raise InputError("GITHUB_TOKEN required to read Actions artifacts")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}",
                            "Accept": "application/vnd.github+json"})
    selected = []
    for page in range(1, 5):
        url = f"https://api.github.com/repos/{repo}/actions/artifacts"
        resp = session.get(url, params={"per_page": 100, "page": page}, timeout=45)
        resp.raise_for_status()
        entries = resp.json().get("artifacts", [])
        selected += [x for x in entries
                     if str(x.get("name", "")).startswith(ARTIFACT_PREFIX)
                     and not x.get("expired", False)
                     and x.get("size_in_bytes", 0) <= 80_000_000]
        if len(selected) >= limit or len(entries) < 100:
            break
    selected.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return selected[:limit]


def free_nfl_finals(season: int) -> pd.DataFrame:
    """nflverse/nflreadpy schedule supplies realized scores ONLY."""
    import nflreadpy
    # Polars to_dicts avoids requiring paid services or optional pyarrow.
    data = pd.DataFrame(nflreadpy.load_schedules([int(season)]).to_dicts())
    fields = ("game_id", "home_team", "away_team", "home_score", "away_score",
              "gameday", "gametime")
    missing = set(fields).difference(data.columns)
    if missing:
        raise InputError(f"nflverse final schedule missing {sorted(missing)}")
    data = data.copy()
    data["home_score"] = pd.to_numeric(data["home_score"], errors="coerce")
    data["away_score"] = pd.to_numeric(data["away_score"], errors="coerce")
    # 'result' is a final-score difference; never a market line.
    data["result"] = data["home_score"] - data["away_score"]
    return data.dropna(subset=["home_score", "away_score"])


def score_archive(repo: str, token: str, season: int, output: Path,
                  folder: Path, limit: int = 25) -> dict:
    artifacts = list_free_pregame_artifacts(repo, token, limit)
    folder.mkdir(parents=True, exist_ok=True)
    downloaded = []
    errors = []
    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {token}",
                                "Accept": "application/vnd.github+json"})
        for a in artifacts:
            try:
                response = session.get(a["archive_download_url"], timeout=90)
                response.raise_for_status()
                downloaded.extend(extract_forecasts(response.content, folder, int(a["id"])))
            except (requests.RequestException, zipfile.BadZipFile, ValueError) as exc:
                errors.append({"artifact_id": a.get("id"), "error": type(exc).__name__})
    finals = free_nfl_finals(season)
    # evaluate() itself enforces >=120 minutes before kickoff, deduplicates
    # distinct games, checks opposing teams, and validates final result.
    results = evaluate(downloaded, finals)
    report = {"schema": "nfl_prospective_artifact_ledger_v1",
              "generated_utc": datetime.now(timezone.utc).isoformat(),
              "season": season, "artifact_count": len(artifacts),
              "forecast_file_count": len(downloaded),
              "artifact_errors": errors,
              "source": "Frozen Actions forecast bytes + later free nflverse schedule finals",
              "not_a_backfit": True, "bets_scored": False,
              "evaluation": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    p.add_argument("--token", default=os.environ.get("GH_TOKEN", ""))
    p.add_argument("--season", type=int,
                   default=(datetime.now(timezone.utc).year if
                            datetime.now(timezone.utc).month >= 7 else
                            datetime.now(timezone.utc).year-1))
    p.add_argument("--out", type=Path, default=Path("evaluations/nfl-prospective.json"))
    p.add_argument("--workdir", type=Path, default=Path("evaluations/frozen"))
    p.add_argument("--max-artifacts", type=int, default=25)
    args = p.parse_args()
    if not args.repo or not 1 <= args.max_artifacts <= 50:
        p.error("Require repository and 1-50 eligible artifact cap")
    r = score_archive(args.repo, args.token, args.season, args.out,
                      args.workdir, args.max_artifacts)
    print(json.dumps({k: v for k, v in r.items() if k != "evaluation"}, indent=2))
    print(json.dumps(r["evaluation"], indent=2))
