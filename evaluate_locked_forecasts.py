#!/usr/bin/env python3
"""Score immutable NFL baseline against ONE versioned challenger per game.

Legacy forecast artifacts may contain both historical v0.9 and v0.10 outputs.
They are immutable archives, NOT two active challengers. For legacy artifacts,
prefer v0.10 if present, otherwise v0.9, and report version-specific results.
Never pool errors from different challenger versions when evaluating promotion.
No verified frozen signed line => no ATS bet claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from collect_extra_metrics import parse_stamp
from model import InputError
from nfl_challenger_policy import ACTIVE_CHALLENGER_VERSION, require_single_active_challenger


def _pick_one_challenger(data: dict):
    """Return (version, candidate); never return two competing forecasts."""
    if "challenger" in data:
        require_single_active_challenger(data)
        return data["challenger"]["version"], data["challenger"].get("candidate")
    # Read-only migration: never edit or rerun historical frozen predictions.
    stateful = data.get("stateful_research") or {}
    earlier = data.get("mechanism_research") or {}
    if stateful.get("candidate") is not None:
        return "0.10.0", stateful["candidate"]
    if earlier.get("candidate") is not None:
        return "0.9.0", earlier["candidate"]
    return None, None


def _average(rows, key):
    nums = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


def _aggregate(rows):
    bs = [r for r in rows if r.get("challenger_abs_margin_error") is not None]
    return {
        "distinct_games": len(rows),
        "challenger_games": len(bs),
        "baseline_mae_margin": _average(rows, "baseline_abs_margin_error"),
        "baseline_mae_total": _average(rows, "baseline_total_abs_error"),
        "baseline_mae_home_score": _average(rows, "baseline_home_score_abs_error"),
        "baseline_mae_away_score": _average(rows, "baseline_away_score_abs_error"),
        # Same-game comparison prevents selective missing candidates bias.
        "paired_baseline_mae_margin": _average(bs, "baseline_abs_margin_error"),
        "paired_baseline_mae_total": _average(bs, "baseline_total_abs_error"),
        "challenger_mae_margin": _average(bs, "challenger_abs_margin_error"),
        "challenger_mae_total": _average(bs, "challenger_total_abs_error"),
        "challenger_mae_home_score": _average(bs, "challenger_home_score_abs_error"),
        "challenger_mae_away_score": _average(bs, "challenger_away_score_abs_error"),
        "challenger_better_margin_games": sum(
            r["challenger_abs_margin_error"] < r["baseline_abs_margin_error"]
            for r in bs
        ),
    }


def evaluate(forecast_files, completed_schedule: pd.DataFrame, out=None):
    needed = {"game_id", "home_score", "away_score", "result",
              "home_team", "away_team", "gameday", "gametime"}
    missing = needed - set(completed_schedule)
    if missing:
        raise InputError("Missing final-score schedule columns: " + str(sorted(missing)))
    if completed_schedule.game_id.duplicated().any():
        raise InputError("Duplicate completed schedule games")
    outcomes = completed_schedule.set_index("game_id")
    chosen = {}
    for path in sorted(map(Path, forecast_files)):
        raw = path.read_bytes()
        data = json.loads(raw)
        if data.get("status") != "EXPLORATORY_CORE_PROJECTION_UNCALIBRATED":
            continue
        gid = str(data["game_id"])
        cutoff = parse_stamp(data["data_as_of_utc"])
        kickoff = parse_stamp(data["kickoff_utc"])
        if (kickoff - cutoff).total_seconds() < 120 * 60:
            raise InputError(f"After-lock forecast excluded: {gid}")
        if gid not in outcomes.index:
            continue
        if gid in chosen and chosen[gid][0] == cutoff and chosen[gid][1] != raw:
            raise InputError(f"Conflicting forecasts recorded at identical cutoff: {gid}")
        if gid not in chosen or cutoff > chosen[gid][0]:
            chosen[gid] = (cutoff, raw, data)
    rows = []
    for gid, (_, raw, data) in sorted(chosen.items()):
        row = outcomes.loc[gid]
        if str(row["home_team"]) != str(data["home"]) or str(row["away_team"]) != str(data["away"]):
            raise InputError(f"Team mismatch: {gid}")
        try:
            h, a, result = (float(row.home_score), float(row.away_score),
                            float(row.result))
        except (ValueError, TypeError):
            continue
        if not all(math.isfinite(x) and x >= 0 for x in (h, a)) or not math.isfinite(result):
            continue
        if abs((h - a) - result) > 1e-6:
            raise InputError(f"Conflicting final score vs official result: {gid}")
        # Signed margin is home-minus-away; baseline never changed.
        baseline_margin = float(data["projected_home_margin"])
        base_home = float(data["score_home"])
        base_away = float(data["score_away"])
        version, candidate = _pick_one_challenger(data)
        r = {
            "game_id": gid,
            "frozen_file_sha256": hashlib.sha256(raw).hexdigest(),
            "prediction_as_of_utc": data["data_as_of_utc"],
            "actual_home_margin": h - a,
            "baseline_margin_home": baseline_margin,
            "baseline_abs_margin_error": round(abs(baseline_margin - (h - a)), 4),
            "baseline_total_abs_error": round(abs(base_home + base_away - h - a), 4),
            "baseline_home_score_abs_error": round(abs(base_home - h), 4),
            "baseline_away_score_abs_error": round(abs(base_away - a), 4),
            "challenger_version": version,
            "challenger_margin_home": None,
            "challenger_abs_margin_error": None,
            "challenger_total_abs_error": None,
            "challenger_home_score_abs_error": None,
            "challenger_away_score_abs_error": None,
            "ATS_side_evaluated": False,
            "market_quote_status": "NO_VERIFIED_FROZEN_SIGNED_LINE",
        }
        if candidate is not None:
            if candidate.get("promoted_to_primary") is not False:
                raise InputError(f"Candidate incorrectly promoted for {gid}")
            margin = float(candidate["mean_margin_home"])
            ch = float(candidate["score_home"])
            ca = float(candidate["score_away"])
            if not all(math.isfinite(x) for x in (margin, ch, ca)):
                raise InputError(f"Invalid candidate scores for {gid}")
            r.update({
                "challenger_margin_home": margin,
                "challenger_abs_margin_error": round(abs(margin - (h - a)), 4),
                "challenger_total_abs_error": round(abs(ch + ca - h - a), 4),
                "challenger_home_score_abs_error": round(abs(ch - h), 4),
                "challenger_away_score_abs_error": round(abs(ca - a), 4),
            })
        rows.append(r)
    current = [r for r in rows if r["challenger_version"] == ACTIVE_CHALLENGER_VERSION]
    legacy_versions = sorted({r["challenger_version"] for r in rows
                              if r["challenger_version"] not in (None, ACTIVE_CHALLENGER_VERSION)})
    result = {
        "schema": "nfl_one_evolving_challenger_evaluation_v1",
        "selection_policy": "LATEST_VALID_CAPTURE_120_PLUS_MINUTES_PRE_KICKOFF",
        "active_challenger_version": ACTIVE_CHALLENGER_VERSION,
        "distinct_eligible_game_ids": len(chosen),
        "completed_games": len(rows),
        "all_baseline": _aggregate(rows),
        "current_challenger": _aggregate(current),
        "archived_challenger_versions": {
            v: _aggregate([r for r in rows if r["challenger_version"] == v])
            for v in legacy_versions
        },
        "ats_record": "NOT_MEASURED_NO_VALIDATED_PRICED_SIGNED_LINES",
        "model_promoted": False,
        "conclusion": ("Frozen versioned evaluation only; no inference from one game. "
                       "Improve the SAME challenger in future revisions, not a second branch."),
        "games": rows,
    }
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--forecasts-dir", required=True, type=Path)
    p.add_argument("--final-schedule", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    files = [x for x in a.forecasts_dir.rglob("*.json") if x.name != "index.json"]
    print(json.dumps(evaluate(files, pd.read_csv(a.final_schedule), a.output), indent=2))
