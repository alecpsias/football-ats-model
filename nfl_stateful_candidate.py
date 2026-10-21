#!/usr/bin/env python3
"""NFL v0.10 STATEFUL RESEARCH CHALLENGER; never a production ATS forecast.

Uses only pre-cutoff drive outcomes extracted by mechanism_research.py. The
v0.8.1 production remains fixed; v0.9 mechanisms have been incorporated\ninto this SOLE evolving challenger rather than retained as a second model.
This is a *hypothesis* for forward testing, not a coefficient calibration.
"""
from __future__ import annotations

import math
import numpy as np

from model import InputError, clip
from mechanism_research import (
    _passrate_for_state, _state, matchup_third_down,
    turnover_following_field_bonus,
)
from nfl_situational_scoring import scored_situations

STATES = ("trailing_9_plus", "within_8", "leading_9_plus")


def state_scoring_residuals(profile: dict, offense: str, defense: str) -> dict:
    """Smoothed state-by-state PPD, *centered* to avoid adding main effects twice.

    Drive points are estimated as 7 for TD and 3 for FG, not isolated EPA.
    Offense and defense samples are shrunk to historical league STATE means.
    Zero-size target-state samples pull to league state results; they are not
    interpreted as zero scoring. No target game appears in the input.
    """
    try:
        lg = profile["league"]["drive_states"]
        off = profile["offense"][offense]["drive_states"]
        deff = profile["defense"][defense]["drive_states"]
    except (KeyError, TypeError) as exc:
        raise InputError("Missing pre-cutoff NFL drive-state observations") from exc
    n_total = sum(int(lg[s]["drives"]) for s in STATES)
    if n_total < 50:
        raise InputError("Insufficient historical drives for stateful candidate")
    league_all_ppd = sum(float(lg[s]["points_proxy"]) for s in STATES) / n_total
    expected = {}
    for s in STATES:
        n = int(lg[s]["drives"])
        league_prior = (float(lg[s]["points_proxy"]) + 24.0 * league_all_ppd) / (n + 24.0)
        on = int(off[s]["drives"])
        dn = int(deff[s]["drives"])
        attack = (float(off[s]["points_proxy"]) + 18.0 * league_prior) / (on + 18.0)
        allowed = (float(deff[s]["points_proxy"]) + 18.0 * league_prior) / (dn + 18.0)
        expected[s] = .5 * (attack + allowed)
    # Center at league's PRE-CUTOFF state mix, not at observed TARGET outcome.
    center = sum(expected[s] * lg[s]["drives"] for s in STATES) / n_total
    return {
        "ppd_delta": {s: round(clip(expected[s] - center, -.40, .40), 6) for s in STATES},
        "state_drives_pre_cutoff": {s: int(lg[s]["drives"]) for s in STATES},
        "reference_ppd": round(center, 6),
        "status": "OBSERVED_SHRUNK_UNFITTED",
    }


def _fourth_failure_rate(profile, offense, defense):
    """Diagnostic label on empty drives; NOT extra lost points or causal EPA."""
    league = profile["league"]
    off = profile["offense"][offense]
    df = profile["defense"][defense]
    drives_lg = sum(league["drive_states"][s]["drives"] for s in STATES)
    drives_off = sum(off["drive_states"][s]["drives"] for s in STATES)
    drives_df = sum(df["drive_states"][s]["drives"] for s in STATES)
    if not min(drives_lg, drives_off, drives_df):
        raise InputError("No pre-cutoff drives for fourth-down diagnostic")
    prior = (league["fourth_failed"] + 5.) / (drives_lg + 100.)
    off_rate = (off["fourth_failed"] + 25. * prior) / (drives_off + 25.)
    df_rate = (df["fourth_failed"] + 25. * prior) / (drives_df + 25.)
    return clip(.5 * (off_rate + df_rate), .001, .20)


def stateful_candidate_simulation(
    home: dict, away: dict, profile: dict, rules: dict, runs: int, seed: int
) -> dict:
    """Joint possession generator with persistent game scoring shocks.

    Correlated in-game efficiency is an UNFITTED variance hypothesis, estimated
    using pre-cutoff between-game PPD variation, with conservative shrinkage.
    Total drive count is shared by teams; legacy v0.9 features are integrated.
    Score state changes BOTH pass tendency/turnover risk AND drive scoring.
    In-game injuries are NOT simulated without verified hazard/replacement data.
    """
    if not 1000 <= runs <= 100000:
        raise ValueError("Stateful candidate requires 1,000-100,000 runs")
    teams = (str(home["team"]), str(away["team"]))
    if len(set(teams)) != 2:
        raise InputError("Home and away NFL teams must differ")
    for team in teams:
        if team not in profile["offense"] or team not in profile["defense"]:
            raise InputError("Incomplete pregame NFL mechanism profile")
    hstates = state_scoring_residuals(profile, teams[0], teams[1])
    astates = state_scoring_residuals(profile, teams[1], teams[0])
    h_situ = scored_situations(profile, teams[0], teams[1])
    a_situ = scored_situations(profile, teams[1], teams[0])
    # Third-down correction is explicitly a residual to the main projection.
    hthird = matchup_third_down(profile, teams[0], teams[1])
    athird = matchup_third_down(profile, teams[1], teams[0])
    rng = np.random.default_rng(seed)
    game_ppd = [float(x) for x in profile["league"].get("completed_team_game_ppd", [])]
    if len(game_ppd) < 16:
        raise InputError("Insufficient historical team-game PPD variance")
    # 0.45 shrinkage factor is NOT fitted; candidate must win a forward test.
    shock_sigma = clip(.45 * float(np.std(game_ppd, ddof=1)), .12, .50)

    def prepare(p, states, situ, third, opp):
        for k in ("_ppd", "_td", "_to", "expected_drives", "pass_rate",
                  "plays_per_drive", "interception_rate_per_dropback",
                  "fumble_lost_rate_per_play", "third_down_conversion"):
            if k not in p or not math.isfinite(float(p[k])):
                raise InputError("Missing numeric projection." + k)
        return {
            "base_ppd": float(p["_ppd"]) + situ["total_delta_ppd"],
            "td_share": clip(float(p["_td"]) + situ["total_delta_td_share"], .10, .95),
            "base_turnover": clip(float(p["_to"]), .005, .42),
            "drives": float(p["expected_drives"]),
            "pass_ref": float(p["pass_rate"]),
            "plays": float(p["plays_per_drive"]),
            "int_rate": float(p["interception_rate_per_dropback"]),
            "fumble_rate": float(p["fumble_lost_rate_per_play"]),
            "third_delta": clip(1.25 * (third["weighted_conversion_rate"] -
                                       float(p["third_down_conversion"])), -.12, .12),
            "state_delta": states["ppd_delta"],
            "pass_by_state": {
                s: _passrate_for_state(profile, p["team"], s, p["pass_rate"])
                for s in STATES
            },
            "fourth_fail": _fourth_failure_rate(profile, p["team"], opp),
        }
    hp = prepare(home, hstates, h_situ, hthird, teams[1])
    ap = prepare(away, astates, a_situ, athird, teams[0])
    # nflverse-derived empirical next-start data; an opponent-17 turnover
    # must not be blindly awarded the same short-field bonus as a pick-six.
    starts = profile["next_start_own_yardline_after_takeaway_by_prior_offense_zone"]
    pooled = [float(v) for values in starts.values() for v in values]
    if not pooled:
        raise InputError("No historical next-possession starts")
    avg_field_bonus = float(np.mean([
        clip((v - rules["ref_start_yardline"]) *
             rules["field_position_points_per_yard"], -1.25, 2.2)
        for v in pooled
    ]))
    ndrive = np.clip(np.rint(rng.normal(
        .5 * (hp["drives"] + ap["drives"]), 1.0, runs)), 7, 17).astype(int)
    offense_shock = rng.normal(0.0, shock_sigma, size=(runs, 2))
    home_scores = np.zeros(runs)
    away_scores = np.zeros(runs)
    turnovers = np.zeros((runs, 2), dtype=int)
    fourth_failures = np.zeros((runs, 2), dtype=int)
    leading_drives = np.zeros((runs, 2), dtype=int)
    trailing_drives = np.zeros((runs, 2), dtype=int)
    next_field_bonus = np.zeros((runs, 2))
    opening = rng.integers(0, 2, size=runs)

    def one(p, side, i):
        score_diff = (home_scores[i] - away_scores[i]) * (1 if side == 0 else -1)
        state = _state(score_diff)
        if state == "leading_9_plus": leading_drives[i, side] += 1
        if state == "trailing_9_plus": trailing_drives[i, side] += 1
        pass_share = p["pass_by_state"][state]
        # Center turnover-risk change on each team's projected neutral mix.
        def torate(pass_mix):
            return 1 - math.exp(-p["plays"] *
                                (pass_mix * p["int_rate"] + p["fumble_rate"]))
        turnover = clip(p["base_turnover"] + .6 * (
            torate(pass_share) - torate(p["pass_ref"])), .005, .42)
        ppd = clip(p["base_ppd"] + p["third_delta"] + p["state_delta"][state] +
                   offense_shock[i, side] + next_field_bonus[i, side] -
                   rules["league_to_drive"] * avg_field_bonus, .10, 5.5)
        next_field_bonus[i, side] = 0.0
        if rng.random() < turnover:
            turnovers[i, side] += 1
            # Historical share of turnovers inside the opponent 40,
            # smoothed to prior NFL league frequency.
            obs = profile["offense"][teams[side]]
            lg = profile["league"]
            frac = (obs["turnovers_near_scoring"] + 12. *
                    (lg["turnovers_near_scoring"] + 1.) /
                    (lg["turnovers"] + 2.)) / (obs["turnovers"] + 12.)
            zone = 25 if rng.random() < frac else 72
            bonus, _ = turnover_following_field_bonus(profile, zone, rules, rng)
            next_field_bonus[i, 1-side] += bonus
            return
        score_value = 3. + 3.95 * p["td_share"]
        # Neutral mean already includes TOs. Do not subtract turnover twice.
        score_chance = clip(ppd / (score_value * (1 - p["base_turnover"])), .01, .97)
        if rng.random() < score_chance:
            pts = (7 if rng.random() < .96 else 6) if rng.random() < p["td_share"] else 3
            if side == 0: home_scores[i] += pts
            else: away_scores[i] += pts
        else:
            # A zero-point drive may be a fourth-down failure. Label only:
            # the underlying scoring probability already priced empty drives.
            empty_chance = max(.03, 1 - score_chance)
            if rng.random() < min(.65, p["fourth_fail"] / empty_chance):
                fourth_failures[i, side] += 1

    for i in range(runs):
        for drive in range(ndrive[i]):
            if opening[i] == 0:
                one(hp, 0, i)
                one(ap, 1, i)
            else:
                one(ap, 1, i)
                one(hp, 0, i)
    margins = home_scores - away_scores
    return {
        "schema": "nfl_stateful_challenger_v0.10",
        "research_only": True, "promoted_to_primary": False,
        "score_home": round(float(np.mean(home_scores)), 3),
        "score_away": round(float(np.mean(away_scores)), 3),
        "mean_margin_home": round(float(np.mean(margins)), 3),
        "margin_p10_p50_p90": [round(float(x), 3)
                              for x in np.quantile(margins, [.1, .5, .9])],
        "total_p10_p50_p90": [round(float(x), 3)
                             for x in np.quantile(home_scores+away_scores, [.1, .5, .9])],
        "home_win_probability_raw_uncalibrated": round(float(np.mean(margins > 0)), 5),
        "expected_turnovers": {
            "home": round(float(np.mean(turnovers[:, 0])), 3),
            "away": round(float(np.mean(turnovers[:, 1])), 3),
        },
        "empty_possession_fourth_down_labels": {
            "home": round(float(np.mean(fourth_failures[:, 0])), 3),
            "away": round(float(np.mean(fourth_failures[:, 1])), 3),
        },
        "game_script_9plus_drives": {
            "home_leading": round(float(np.mean(leading_drives[:, 0])), 3),
            "away_leading": round(float(np.mean(leading_drives[:, 1])), 3),
            "home_trailing": round(float(np.mean(trailing_drives[:, 0])), 3),
            "away_trailing": round(float(np.mean(trailing_drives[:, 1])), 3),
        },
        "pre_cutoff_state_scoring": {"home": hstates, "away": astates},
        "pre_cutoff_situational_scoring": {"home": h_situ, "away": a_situ},
        "shared_game_scoring_shock_sd_ppd_unfitted": round(shock_sigma, 6),
        "n": runs, "seed": seed,
        "limitations": [
            "Research only; no ATS calibration, calibrated coverage, or EV.",
            "7/3 drive points are TD/FG proxies, not play-level EPA.",
            "State PPD and game shocks are shrunken observational proxies; coefficients not fitted.",
            "Fourth-down labels reclassify empty drives; no extra independent penalty applied.",
            "In-game injuries, changes to roster depth and precise running clock not inferred.",
            "All features use pre-cutoff snapshots; never use this game's later PBP.",
        ],
    }
