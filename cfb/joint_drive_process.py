"""Source-observed joint rush/pass/elapsed-time drive sampler for Build16.

An incremental, single-challenger mechanism: the *existing* opponent-adjusted
drive model decides whether a possession scores, stalls, or turns the ball over.
Then a REAL, source-clock-verified previous possession supplies a jointly
observed rush count, pass count, and clock duration, conditional on that outcome
and pre-drive score state. This preserves correlated drive characteristics;
it is NOT a play-by-play causal simulator and adds NO scoring bonus.

Missing observations disable this mechanism and preserve legacy Build16.
The target fixture is never supplied to the history collection.
"""
from __future__ import annotations

import math
import numpy as np

VERSION = "cfb-joint-drive-observed-v1"
OUTCOMES = ("explosive_td", "chain_score", "turnover", "stalled")
MIN_GAMES_PER_SIDE = 2
MIN_MATCHED_DRIVES = 24
MIN_CATEGORY_ROWS = 3


def _extract(rows, side):
    drives = []
    games = 0
    for record in rows:
        clock = record.get("drive_clock")
        if not isinstance(clock, dict):
            continue
        if clock.get("status") != "SOURCE_CLOCK_VERIFIED":
            continue
        if clock.get("clock_verification") != "ESPN_ELAPSED_AND_ENDPOINTS":
            continue
        items = clock.get("joint_drives", {}).get(side)
        if not isinstance(items, list):
            continue
        valid = []
        for item in items:
            if not isinstance(item, dict) or item.get("outcome") not in OUTCOMES:
                continue
            t, r, p = (item.get(k) for k in ("seconds", "rush_plays", "pass_plays"))
            if not all(type(v) is int for v in (t, r, p)):
                continue
            if not (1 <= t <= 900 and 0 <= r <= 30 and 0 <= p <= 30 and 1 <= r + p <= 35):
                continue
            valid.append({"seconds": t, "rush_plays": r, "pass_plays": p,
                          "rush_share": r / (r + p), "outcome": item["outcome"]})
        if valid:
            games += 1
            drives.extend(valid)
    return games, drives


def matchup_joint_drives(home_id, away_id, histories):
    """Pool prior offense + future-opponent prior defense, never current game."""
    result = {"version": VERSION, "status": "INSUFFICIENT_VERIFIED_JOINT_DRIVES",
              "matched_observed_only": True, "ATS_calibrated": False}
    for side, team, opponent in (("home", home_id, away_id), ("away", away_id, home_id)):
        off_games, offense = _extract(histories.get(team, ()), "off")
        def_games, defense = _extract(histories.get(opponent, ()), "def")
        merged = offense + defense
        if min(off_games, def_games) < MIN_GAMES_PER_SIDE or len(merged) < MIN_MATCHED_DRIVES:
            return {**result, "missing_for": side, "offense_games": off_games,
                    "opponent_defense_games": def_games, "matched_drives": len(merged)}
        pools = {kind: [r for r in merged if r["outcome"] == kind] for kind in OUTCOMES}
        result[side] = {"pools": pools, "counts": {k: len(v) for k, v in pools.items()},
                        "offense_games": off_games, "defense_games": def_games,
                        "total_observed_drives": len(merged)}
    result["status"] = "ACTIVE_SOURCE_JOINT_DRIVE_BOOTSTRAP"
    result["score_mean_preserved"] = True
    result["clock_seconds_are_observed"] = True
    result["outcome_probabilities_from_existing_build16"] = True
    result["score_state_rush_matching_is_unfitted"] = True
    return result


def sample_joint(rng, pool, outcome, mask, desired_rush_share):
    """Return sampled (clock, runs, passes, unavailable mask).

    Draw THREE real candidate drives from the matching outcome stratum.
    With probability 0.6 choose the closest observed rush mix to the
    score-dependent target; otherwise retain an ordinary empirical draw.
    This is a bounded UNFITTED dependence assumption, not a fitted effect.
    No score or scoring probability is assigned by this routine.
    """
    n = len(mask)
    sec = np.zeros(n, dtype=float)
    rush = np.zeros(n, dtype=np.int16)
    pas = np.zeros(n, dtype=np.int16)
    missing = np.zeros(n, dtype=bool)
    selected = np.flatnonzero(mask)
    if not selected.size:
        return sec, rush, pas, missing
    rows = pool.get("pools", {}).get(outcome, ())
    if len(rows) < MIN_CATEGORY_ROWS:
        missing[selected] = True
        return sec, rush, pas, missing
    if len(desired_rush_share) != n or not np.all(np.isfinite(desired_rush_share[selected])):
        raise ValueError("Invalid pre-drive rushing state")
    durations = np.asarray([r["seconds"] for r in rows], dtype=float)
    run = np.asarray([r["rush_plays"] for r in rows], dtype=np.int16)
    throws = np.asarray([r["pass_plays"] for r in rows], dtype=np.int16)
    shares = np.asarray([r["rush_share"] for r in rows], dtype=float)
    candidates = rng.integers(0, len(rows), size=(len(selected), 3))
    d = np.abs(shares[candidates] - np.clip(desired_rush_share[selected, None], 0, 1))
    nearest = candidates[np.arange(len(selected)), np.argmin(d, axis=1)]
    ordinary = candidates[:, 0]
    use_nearest = rng.random(len(selected)) < 0.6
    pick = np.where(use_nearest, nearest, ordinary)
    sec[selected] = durations[pick]
    rush[selected] = run[pick]
    pas[selected] = throws[pick]
    return sec, rush, pas, missing
