#!/usr/bin/env python3
"""Starter-specific NFL QB performance residual for the SOLE evolving challenger.

Source: completed, PRE-CUTOFF nflverse PBP only. Fitted/shrunken coefficients
(half-life=1 week, 30 skill prior dropbacks, 200 form prior dropbacks)
were selected in the prospective Week 1-6 research QB-EPA walk-forward check.
That three-week check was NOT independent ATS validation.

This module never predicts a confirmed starting lineup or uses game-day
inactive lists. It selects a probable incumbent from the latest eligible
COMPLETED game; ambiguous, low-sample or missing cases get a documented
zero residual, not an invented QB evaluation. The baseline is untouched.
"""
from __future__ import annotations

from datetime import timedelta
import math
import pandas as pd
import numpy as np

from collect_extra_metrics import parse_stamp
from ingest import normalize_schedule

VERSION = "qb-form-v0.11-regularized-v1"
BUFFER_HOURS = 6
DEFENSE_PRIOR_DROPBACKS = 150.0
SKILL_PRIOR_DROPBACKS = 30.0
FORM_PRIOR_DROPBACKS = 200.0
HALF_LIFE_WEEKS = 1.0
MIN_QB_DROPBACKS = 40
MIN_LATEST_DROPBACKS = 10
MIN_LAST_GAME_SHARE = 0.65
MAX_ABS_QB_EPA_RESIDUAL = 0.35
MAX_ABS_PPD_RESIDUAL = 0.40
REQUIRED_PBP = frozenset({
    "game_id", "posteam", "defteam", "qb_dropback", "epa",
    "passer_player_name", "rusher_player_name", "qb_scramble",
})


def _neutral(team, reason, **extra):
    return {
        "team": team, "version": VERSION, "status": "NOT_APPLIED",
        "reason": reason, "projected_starter_confirmed": False,
        "adjusted_ppd_residual": 0.0,
        "fitted_params": {
            "defense_prior_dropbacks": DEFENSE_PRIOR_DROPBACKS,
            "skill_prior_dropbacks": SKILL_PRIOR_DROPBACKS,
            "form_prior_dropbacks": FORM_PRIOR_DROPBACKS,
            "recent_half_life_weeks": HALF_LIFE_WEEKS,
        }, **extra,
    }


def pre_cutoff_qb_plays(pbp: pd.DataFrame, schedule: pd.DataFrame, cutoff):
    """Completed-game QB dropbacks, scrambles and sacks; no game after cutoff.

    Require verified final scores AND kickoff <= cutoff minus six hours.
    Reject duplicate IDs and missing required data, rather than expanding
    to uncertain or uncompleted games. Does not use active game observations.
    """
    missing = REQUIRED_PBP.difference(pbp.columns)
    if missing:
        raise ValueError("QB_PBP_MISSING_COLUMNS: " + ", ".join(sorted(missing)))
    need = {"game_id", "home_score", "away_score"}
    if need.difference(schedule.columns):
        raise ValueError("QB_SCHEDULE_MISSING_FINALS")
    games = normalize_schedule(schedule)
    if games.game_id.duplicated().any():
        raise ValueError("QB_DUPLICATE_GAME_IDS")
    eligible = games[
        (games.kickoff_utc <= pd.Timestamp(parse_stamp(str(cutoff)) - timedelta(hours=BUFFER_HOURS)))
        & pd.to_numeric(games.home_score, errors="coerce").notna()
        & pd.to_numeric(games.away_score, errors="coerce").notna()
    ]
    ids = set(eligible.game_id.astype(str))
    p = pbp[pbp.game_id.astype(str).isin(ids) &
            pd.to_numeric(pbp.qb_dropback, errors="coerce").eq(1) &
            pd.to_numeric(pbp.epa, errors="coerce").notna()].copy()
    if p.empty:
        return pd.DataFrame(columns=["game_id", "posteam", "defteam", "qb",
                                      "epa", "week", "kickoff_utc"])
    p["qb"] = p.passer_player_name.where(p.passer_player_name.notna(),
        p.rusher_player_name.where(pd.to_numeric(p.qb_scramble, errors="coerce").eq(1)))
    p = p[p.qb.notna() & p.posteam.notna() & p.defteam.notna()].copy()
    # Use verified schedule metadata for ordering rather than untrusted PBP week.
    safe = eligible[["game_id", "kickoff_utc"]].copy()
    if "week" in eligible:
        safe["week"] = pd.to_numeric(eligible.week, errors="coerce")
    else:
        safe["week"] = pd.to_numeric(eligible.kickoff_utc.dt.isocalendar().week, errors="coerce")
    p = p.drop(columns=[c for c in ("week", "kickoff_utc") if c in p], errors="ignore")
    p = p.merge(safe, on="game_id", how="inner", validate="many_to_one")
    p = p[p.week.notna()].copy()
    p["week"] = p.week.astype(int)
    return p


def adjusted_dropback_epa(plays: pd.DataFrame):
    """Defense adjustment using that defense vs other offenses only.

    A ridge-style 150-dropback prior mitigates small-sample and particularly
    early-season swings. No individual opponent's own offense is used in the
    adjustment applied to itself. Other confounders remain.
    """
    if plays.empty:
        return plays.assign(adj_epa=pd.Series(dtype=float))
    t = plays.copy()
    t["epa"] = pd.to_numeric(t.epa, errors="raise")
    league = float(t.epa.mean())
    totals = t.groupby("defteam").epa.agg(["count", "sum"])
    pair = t.groupby(["defteam", "posteam"]).epa.agg(["count", "sum"])
    pairs = t[["defteam", "posteam"]].drop_duplicates().itertuples(index=False, name=None)
    correction = {}
    for deff, off in pairs:
        all_def = totals.loc[deff]
        duo = pair.loc[(deff, off)]
        n = float(all_def["count"] - duo["count"])
        epa_sum = float(all_def["sum"] - duo["sum"])
        if n < 0:
            raise ValueError("INVALID_QB_DEFENSE_SAMPLE")
        mean_other = epa_sum / n if n > 0 else league
        correction[(deff, off)] = n / (n + DEFENSE_PRIOR_DROPBACKS) * (mean_other - league)
    t["adj_epa"] = t.epa.to_numpy(dtype=float) - np.array(
        [correction[(d, o)] for d, o in zip(t.defteam, t.posteam)])
    return t


def qb_effect_for_team(adjusted: pd.DataFrame, team: str, projection: dict):
    """Fitted EPA residual, opponent-shrunk and applied only to challenger PPD."""
    team_plays = adjusted[adjusted.posteam.eq(team)].copy()
    if team_plays.empty:
        return _neutral(team, "NO_ELIGIBLE_COMPLETED_QB_GAMES")
    latest = team_plays.loc[team_plays.kickoff_utc.eq(team_plays.kickoff_utc.max())]
    counts = latest.qb.value_counts()
    if len(counts) == 0:
        return _neutral(team, "NO_LATEST_QB")
    chosen = str(counts.index[0])
    share = float(counts.iloc[0] / len(latest))
    if len(latest) < MIN_LATEST_DROPBACKS or share < MIN_LAST_GAME_SHARE:
        return _neutral(team, "AMBIGUOUS_LAST_GAME_QB_USAGE",
                        last_game_qb=chosen, last_game_dropback_share=round(share, 4))
    qb = team_plays[team_plays.qb.eq(chosen)]
    if len(qb) < MIN_QB_DROPBACKS:
        return _neutral(team, "QB_HISTORY_TOO_SMALL",
                        last_game_qb=chosen, observed_qb_dropbacks=len(qb))
    q_mu = float(qb.adj_epa.mean())
    team_mu = float(team_plays.adj_epa.mean())
    max_week = int(team_plays.week.max())
    weights = np.exp2((qb.week.to_numpy(dtype=float) - max_week) / HALF_LIFE_WEEKS)
    recency = float(np.average(qb.adj_epa, weights=weights))
    recent_n = float(weights.sum())
    skill = (q_mu - team_mu) * len(qb) / (len(qb) + SKILL_PRIOR_DROPBACKS)
    trend = (recency - q_mu) * recent_n / (recent_n + FORM_PRIOR_DROPBACKS)
    total = float(np.clip(skill + trend, -MAX_ABS_QB_EPA_RESIDUAL,
                          MAX_ABS_QB_EPA_RESIDUAL))
    # QB dropbacks are counted as a fraction of team pass/rush scrimmage plays.
    # The team pass rate is the matched expected dropback share.
    # 0.5 scales to the base model's 50/50 offense/defense EPA combination.
    # Scramble EPA enters here, not as a second team rushing adjustment.
    dropback_share = float(projection["pass_rate"])
    drive_plays = float(projection["plays_per_drive"])
    if not (.2 <= dropback_share <= .9 and 3 <= drive_plays <= 10):
        return _neutral(team, "INVALID_MATCHED_PROJECTION")
    ppd = float(np.clip(0.5 * total * dropback_share * drive_plays,
                        -MAX_ABS_PPD_RESIDUAL, MAX_ABS_PPD_RESIDUAL))
    if not math.isfinite(ppd):
        return _neutral(team, "NONFINITE_QB_ESTIMATE")
    return {
        "team": team, "version": VERSION, "status": "APPLIED",
        "reason": "LATEST_COMPLETED_GAME_QB_PROJECTION_NOT_CONFIRMED_STARTER",
        "projected_starter_confirmed": False, "projected_qb_name": chosen,
        "latest_game_dropback_share": round(share, 6),
        "qb_games": int(qb.game_id.nunique()), "qb_dropbacks": len(qb),
        "team_dropbacks": len(team_plays),
        "qb_opponent_adjusted_epa": round(q_mu, 6),
        "team_opponent_adjusted_epa": round(team_mu, 6),
        "recency_adjusted_epa": round(recency, 6),
        "recent_effective_dropbacks": round(recent_n, 3),
        "regularized_skill_delta_epa": round(skill, 6),
        "regularized_form_delta_epa": round(trend, 6),
        "fitted_delta_epa_per_dropback": round(total, 6),
        "projection_dropback_share": round(dropback_share, 6),
        "adjusted_ppd_residual": round(ppd, 6),
        "fitted_params": {
            "defense_prior_dropbacks": DEFENSE_PRIOR_DROPBACKS,
            "skill_prior_dropbacks": SKILL_PRIOR_DROPBACKS,
            "form_prior_dropbacks": FORM_PRIOR_DROPBACKS,
            "recent_half_life_weeks": HALF_LIFE_WEEKS,
        },
        "limits": [
            "Probable incumbent inferred from last completed game, NOT confirmed starter",
            "QB performance also reflects opponent, receivers, protection and coaching",
            "Shrunk forward hypothesis; not ATS-calibrated or independent long-run validated",
            "Pass/rush team totals already contain some QB impact; bounded residual only",
        ],
    }


def estimate_qb_adjustments(pbp, schedule, cutoff, home, away, projections):
    """One evidence object for each side, with no data from the target game."""
    history = pre_cutoff_qb_plays(pbp, schedule, cutoff)
    adjusted = adjusted_dropback_epa(history)
    return {team: qb_effect_for_team(adjusted, team, projections[team])
            for team in (home, away)}
