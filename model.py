#!/usr/bin/env python3
"""Football ATS Model v0.1 — auditable research engine, NOT empirically calibrated.

Dependencies: numpy. Inputs must be fully populated pregame snapshots.
No fabricated/missing statistics and no pregame claim of validated wagering value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "0.7.0"
MODEL_RULES = {
    "NFL": {
        "league_ppd": 2.08, "plays_per_drive": 5.8, "drives_per_team": 10.8,
        "pass_rate": .57, "success": .44, "pass_explosive": .10,
        "rush_explosive": .10, "pressure": .32, "sack": .067,
        "interception": .025, "fumble": .0065, "third_down": .395,
        "rz_td": .56, "td_share_scoring_drives": .58, "league_to_drive": .115,
        "home_advantage_points": 1.5, "play_shrink": 450,
        "rz_shrink": 35, "drive_noise_sd": .15, "shortfield_bonus_ppd": .9,
        "ref_start_yardline": 29.0, "field_position_points_per_yard": .044,
        "penalty_yards_points_per_yard": .005, "fg_proxy_points_scale": 2.25,
    },
    "CFB": {
        "league_ppd": 2.35, "plays_per_drive": 6.1, "drives_per_team": 11.5,
        "pass_rate": .51, "success": .45, "pass_explosive": .105,
        "rush_explosive": .12, "pressure": .30, "sack": .068,
        "interception": .028, "fumble": .007, "third_down": .405,
        "rz_td": .59, "td_share_scoring_drives": .64, "league_to_drive": .13,
        "home_advantage_points": 2.4, "play_shrink": 600,
        "rz_shrink": 45, "drive_noise_sd": .20, "shortfield_bonus_ppd": 1.05,
        "ref_start_yardline": 29.0, "field_position_points_per_yard": .046,
    },
}
# All constants are starter/illustrative hypotheses, NOT coefficients fitted to
# 2026 football or historical ATS outcomes. Users must backtest and recalibrate.

OFF_KEYS = (
    "games", "plays", "epa_pass", "epa_rush", "pass_rate", "success_pass",
    "success_rush", "explosive_pass_rate", "explosive_rush_rate",
    "pressure_allowed", "sack_allowed", "interception_rate", "fumble_lost_rate",
    "redzone_td_rate", "redzone_trips", "third_down_rate",
    "drives_per_game", "plays_per_drive", "qb_rushes_per_game", "qb_rush_epa",
)
DEF_KEYS = (
    "plays", "epa_pass_allowed", "epa_rush_allowed", "success_pass_allowed",
    "success_rush_allowed", "explosive_pass_allowed", "explosive_rush_allowed",
    "pressure_generated", "sack_generated", "interception_rate_forced",
    "fumble_recovery_rate", "redzone_td_allowed", "redzone_trips_faced",
    "third_down_allowed", "drives_faced_per_game", "plays_per_drive_allowed",
)
SOS_KEYS = (
    "opp_def_pass_epa_allowed_avg", "opp_def_rush_epa_allowed_avg",
    "opp_off_pass_epa_avg", "opp_off_rush_epa_avg",
)
CONTEXT_KEYS = ("avg_start_yardline", "penalty_epa_per_game", "special_teams_epa_per_game")
PERSONNEL_KEYS = ("pass_epa_shift", "rush_epa_shift", "pressure_allowed_shift", "qb_rush_multiplier", "pace_multiplier", "notes", "source", "checked_at_utc")

class InputError(ValueError):
    pass


def need(d: dict, keys: tuple[str, ...], where: str) -> None:
    for k in keys:
        if k not in d or d[k] is None:
            raise InputError(f"Missing REQUIRED {where}.{k}; refusing to predict")


def number(v: Any, field: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise InputError(f"{field} must be numeric (not null or a string)")
    x = float(v)
    if not math.isfinite(x) or x < lo or x > hi:
        raise InputError(f"{field} must be finite and in [{lo}, {hi}]")
    return x


def utc(v: str, name: str) -> datetime:
    if not isinstance(v, str):
        raise InputError(f"{name} must be ISO 8601 with an explicit UTC offset")
    try:
        parsed = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InputError(f"Invalid datetime {name}") from exc
    if parsed.tzinfo is None:
        raise InputError(f"{name} must specify timezone")
    return parsed.astimezone(timezone.utc)


def validate_team(t: dict, label: str) -> None:
    need(t, ("name", "stat_source", "stat_as_of_utc", "offense", "defense", "schedule", "context", "personnel"), label)
    if not isinstance(t["name"], str) or not t["name"].strip():
        raise InputError(f"{label}.name required")
    if not isinstance(t["stat_source"], str) or not t["stat_source"].strip():
        raise InputError(f"{label}.stat_source required")
    for section, keys in (("offense", OFF_KEYS), ("defense", DEF_KEYS), ("schedule", SOS_KEYS), ("context", ("avg_start_yardline",)), ("personnel", PERSONNEL_KEYS)):
        if not isinstance(t[section], dict):
            raise InputError(f"{label}.{section} must be an object")
        need(t[section], keys, f"{label}.{section}")
    o, d, s, c, p = (t[k] for k in ("offense", "defense", "schedule", "context", "personnel"))
    for k in ("games", "plays", "redzone_trips"):
        number(o[k], f"{label}.offense.{k}", 1 if k != "redzone_trips" else 0, 100000)
    for k in ("plays", "redzone_trips_faced"):
        number(d[k], f"{label}.defense.{k}", 1 if k == "plays" else 0, 100000)
    for k in ("epa_pass", "epa_rush"):
        number(o[k], f"{label}.offense.{k}", -1.5, 1.5)
    # Tiny QB rushing samples can have valid extreme per-play EPA. Validate
    # finite, then shrink the DIFFERENCE toward team rushing EPA below.
    number(o["qb_rush_epa"], f"{label}.offense.qb_rush_epa", -7, 7)
    for k in ("epa_pass_allowed", "epa_rush_allowed"):
        number(d[k], f"{label}.defense.{k}", -1.5, 1.5)
    for k in SOS_KEYS:
        number(s[k], f"{label}.schedule.{k}", -1.5, 1.5)
    for k in OFF_KEYS:
        if k in ("games", "plays", "redzone_trips", "epa_pass", "epa_rush", "qb_rush_epa", "drives_per_game", "plays_per_drive", "qb_rushes_per_game"):
            continue
        number(o[k], f"{label}.offense.{k}", 0, 1)
    for k in DEF_KEYS:
        if k in ("plays", "redzone_trips_faced", "epa_pass_allowed", "epa_rush_allowed", "drives_faced_per_game", "plays_per_drive_allowed"):
            continue
        number(d[k], f"{label}.defense.{k}", 0, 1)
    for k in ("drives_per_game", "plays_per_drive"):
        number(o[k], f"{label}.offense.{k}", 2.5 if k == "plays_per_drive" else 6, 20)
    number(o["qb_rushes_per_game"], f"{label}.offense.qb_rushes_per_game", 0, 25)
    for k in ("drives_faced_per_game", "plays_per_drive_allowed"):
        number(d[k], f"{label}.defense.{k}", 2.5 if k == "plays_per_drive_allowed" else 6, 20)
    number(c["avg_start_yardline"], f"{label}.context.avg_start_yardline", 15, 45)
    # True penalty-isolated EPA / causal special-teams EPA are OPTIONAL and may
    # remain null. Null disables that contribution; it is never imputed as a
    # measured zero or replaced by unrelated penalty yards / kicking event EPA.
    for k in ("penalty_epa_per_game", "special_teams_epa_per_game"):
        if c.get(k) is not None: number(c[k], f"{label}.context.{k}", -15, 15)
    if t.get("advanced") is not None:
        v=t["advanced"]
        if not isinstance(v,dict):raise InputError(f"{label}.advanced must be an object")
        need(v,("penalty_yards_per_game","penalty_games","field_goals_made","field_goals_attempted", "source","as_of_utc","status"),f"{label}.advanced")
        number(v["penalty_yards_per_game"],f"{label}.advanced.penalty_yards_per_game",0,300)
        number(v["penalty_games"],f"{label}.advanced.penalty_games",3,30)
        made=number(v["field_goals_made"],f"{label}.advanced.field_goals_made",0,100)
        tried=number(v["field_goals_attempted"],f"{label}.advanced.field_goals_attempted",0,100)
        if made>tried:raise InputError(f"{label}.advanced.field_goals_made exceeds attempts")
        if v.get("punt_gross_yards") is not None:
            number(v["punt_gross_yards"],f"{label}.advanced.punt_gross_yards",20,80)
        if v["status"]!="DESCRIPTIVE_UNCALIBRATED" or not isinstance(v["source"],str) or not v["source"].strip():
            raise InputError(f"{label}.advanced source/status not verified")
        utc(v["as_of_utc"],f"{label}.advanced.as_of_utc")
    for k in ("pass_epa_shift", "rush_epa_shift"):
        number(p[k], f"{label}.personnel.{k}", -.4, .4)
    number(p["pressure_allowed_shift"], f"{label}.personnel.pressure_allowed_shift", -.25, .25)
    for k in ("qb_rush_multiplier", "pace_multiplier"):
        number(p[k], f"{label}.personnel.{k}", .5, 1.5)
    changed = any(abs(float(p[k])) > 1e-10 for k in ("pass_epa_shift", "rush_epa_shift", "pressure_allowed_shift")) or any(abs(float(p[k])-1) > 1e-10 for k in ("qb_rush_multiplier", "pace_multiplier"))
    utc(p["checked_at_utc"], f"{label}.personnel.checked_at_utc")
    if not isinstance(p["source"], str) or not p["source"].strip():
        raise InputError(f"{label}.personnel.source cannot be blank; news must be checked")
    if changed and (not isinstance(p["notes"], str) or not p["notes"].strip() or not isinstance(p["source"], str) or not p["source"].strip()):
        raise InputError(f"{label}.personnel non-neutral shifts require notes AND source")
    if o["qb_rushes_per_game"] > o["plays_per_drive"]*o["drives_per_game"]*(1-o["pass_rate"])+1:
        raise InputError(f"{label}.offense QB rushes exceed total team rushes")
    if o["sack_allowed"] > o["pressure_allowed"] + .015 or d["sack_generated"] > d["pressure_generated"] + .015:
        raise InputError(f"{label}: sack rate cannot materially exceed pressure rate")


def validate_game(g: dict) -> dict:
    need(g, ("id", "league", "kickoff_utc", "data_as_of_utc", "neutral_site", "home", "away", "market"), "game")
    if g["league"] not in MODEL_RULES:
        raise InputError("game.league must be NFL or CFB")
    if not isinstance(g["neutral_site"], bool):
        raise InputError("game.neutral_site must be boolean")
    kickoff, cutoff = utc(g["kickoff_utc"], "kickoff_utc"), utc(g["data_as_of_utc"], "data_as_of_utc")
    if cutoff > kickoff:
        raise InputError("Post-kickoff data detected: rejecting look-ahead leakage")
    validate_team(g["home"], "home")
    validate_team(g["away"], "away")
    if g["home"]["name"] == g["away"]["name"]:
        raise InputError("Home and away must differ")
    for k in ("home", "away"):
        a = utc(g[k]["stat_as_of_utc"], f"{k}.stat_as_of_utc")
        if a > cutoff:
            raise InputError(f"{k} stats dated later than game.data_as_of_utc")
        if g[k].get("advanced") is not None and utc(g[k]["advanced"]["as_of_utc"],f"{k}.advanced.as_of_utc")>cutoff:
            raise InputError(f"{k} advanced features dated later than data cutoff")
        person_checked = utc(g[k]["personnel"]["checked_at_utc"], f"{k}.personnel.checked_at_utc")
        if person_checked > cutoff or (cutoff-person_checked).total_seconds() > 24*3600:
            raise InputError(f"{k} personnel check must occur within 24 hours before data_as_of_utc")
    m = g["market"]
    need(m, ("spread_home", "home_odds_american", "away_odds_american", "total", "as_of_utc", "source"), "game.market")
    number(m["spread_home"], "market.spread_home", -70, 70)
    number(m["total"], "market.total", 10, 120)
    for k in ("home_odds_american", "away_odds_american"):
        x = number(m[k], "market." + k, -20000, 20000)
        if -100 < x < 100:
            raise InputError("American odds must be <= -100 or >= +100")
    if not isinstance(m["source"], str) or not m["source"].strip():
        raise InputError("Market source required")
    if utc(m["as_of_utc"], "market.as_of_utc") > cutoff:
        raise InputError("Market price is later than the allowed snapshot")
    if "first_half" in m and m["first_half"] is not None:
        fh = m["first_half"]
        if not isinstance(fh, dict):
            raise InputError("market.first_half must be a market object")
        need(fh, ("spread_home", "home_odds_american", "away_odds_american", "total", "source", "as_of_utc"), "market.first_half")
        number(fh["spread_home"], "market.first_half.spread_home", -40, 40)
        number(fh["total"], "market.first_half.total", 5, 75)
        for k in ("home_odds_american", "away_odds_american"):
            x=number(fh[k], "market.first_half."+k, -20000, 20000)
            if -100<x<100:
                raise InputError("First-half American odds must be <= -100 or >= +100")
        if not fh["source"] or utc(fh["as_of_utc"], "market.first_half.as_of_utc")>cutoff:
            raise InputError("First-half market requires source and valid pregame snapshot")
    return MODEL_RULES[g["league"]]


def clip(x: float, a: float, b: float) -> float:
    return float(max(a, min(b, x)))


def shrink(v: float, base: float, count: float, k: float) -> float:
    return base + (v-base) * (count / (count+k))


def logit(p: float) -> float:
    p = clip(p, 1e-5, 1-1e-5)
    return math.log(p/(1-p))


def sigmoid(x: float) -> float:
    return 1/(1+math.exp(-clip(x, -30, 30)))


def rate_match(attack: float, defend: float, ref: float, oa: float, dd: float, k: float) -> float:
    attack = shrink(attack, ref, oa, k)
    defend = shrink(defend, ref, dd, k)
    return clip(sigmoid(logit(ref) + .65*(logit(attack)-logit(ref)) + .65*(logit(defend)-logit(ref))), .0001, .95)


def projection(attack: dict, defend: dict, r: dict, side: str, home_adv_ppg: float) -> dict:
    a, d, sos, dsos = attack["offense"], defend["defense"], attack["schedule"], defend["schedule"]
    p, c, cp, cd = attack["personnel"], attack["context"], attack["context"], defend["context"]
    a_n, d_n, k = a["plays"], d["plays"], r["play_shrink"]
    # SOS-aware EPA: remove the average difficulty of opponents faced before shrinkage.
    pass_o = shrink(a["epa_pass"]-sos["opp_def_pass_epa_allowed_avg"], 0, a_n*a["pass_rate"], k)
    rush_o = shrink(a["epa_rush"]-sos["opp_def_rush_epa_allowed_avg"], 0, a_n*(1-a["pass_rate"]), k)
    pass_d = shrink(d["epa_pass_allowed"]-dsos["opp_off_pass_epa_avg"], 0, d_n*.57, k)
    rush_d = shrink(d["epa_rush_allowed"]-dsos["opp_off_rush_epa_avg"], 0, d_n*.43, k)
    pass_epa = .5*(pass_o+pass_d)+p["pass_epa_shift"]
    rush_epa = .5*(rush_o+rush_d)+p["rush_epa_shift"]
    pressure_o = clip(a["pressure_allowed"]+p["pressure_allowed_shift"], .005, .9)
    pressure = rate_match(pressure_o, d["pressure_generated"], r["pressure"], a_n*a["pass_rate"], d_n*.57, k)
    base_press = .5*(shrink(pressure_o, r["pressure"], a_n*a["pass_rate"], k) + shrink(d["pressure_generated"], r["pressure"], d_n*.57, k))
    pressure_residual_epa = clip(-.26*(pressure-base_press), -.035, .035)
    sack_conditional = .5*(clip(a["sack_allowed"]/max(a["pressure_allowed"], .01), .02, .65) + clip(d["sack_generated"]/max(d["pressure_generated"], .01), .02, .65))
    expected_sack_rate = clip(pressure*sack_conditional, .005, .22)
    # Success/explosive/third-down are RESIDUAL interaction corrections only;
    # their main effects are already in EPA and must not be fully counted twice.
    success_pass = rate_match(a["success_pass"], d["success_pass_allowed"], r["success"], a_n*a["pass_rate"], d_n*.57, k)
    success_rush = rate_match(a["success_rush"], d["success_rush_allowed"], r["success"], a_n*(1-a["pass_rate"]), d_n*.43, k)
    expl_pass = rate_match(a["explosive_pass_rate"], d["explosive_pass_allowed"], r["pass_explosive"], a_n*a["pass_rate"], d_n*.57, k)
    expl_rush = rate_match(a["explosive_rush_rate"], d["explosive_rush_allowed"], r["rush_explosive"], a_n*(1-a["pass_rate"]), d_n*.43, k)
    base_expl_pass = .5*(a["explosive_pass_rate"]+d["explosive_pass_allowed"])
    base_expl_rush = .5*(a["explosive_rush_rate"]+d["explosive_rush_allowed"])
    base_success_pass = .5*(shrink(a["success_pass"],r["success"],a_n*a["pass_rate"],k)+shrink(d["success_pass_allowed"],r["success"],d_n*.57,k))
    base_success_rush = .5*(shrink(a["success_rush"],r["success"],a_n*(1-a["pass_rate"]),k)+shrink(d["success_rush_allowed"],r["success"],d_n*.43,k))
    success_pass_residual = clip(.3*(success_pass-base_success_pass), -.018, .018)
    success_rush_residual = clip(.25*(success_rush-base_success_rush), -.018, .018)
    pass_epa += pressure_residual_epa + success_pass_residual + clip(.3*(expl_pass-base_expl_pass), -.022, .022)
    rush_epa += success_rush_residual + clip(.22*(expl_rush-base_expl_rush), -.02, .02)
    pass_rate = float(a["pass_rate"])
    base_team_rush_attempts = max(1, a["drives_per_game"]*a["plays_per_drive"]*(1-pass_rate))
    pressure_ratio = clip(1+.35*(pressure-pressure_o)/max(r["pressure"], .15), .8, 1.2)
    qb_rush_projection = clip(a["qb_rushes_per_game"]*p["qb_rush_multiplier"]*pressure_ratio, 0, base_team_rush_attempts)
    qb_rush_opportunities=a["qb_rushes_per_game"]*a["games"]
    qb_rush_difference=(a["qb_rush_epa"]-a["epa_rush"])*qb_rush_opportunities/(qb_rush_opportunities+25)
    qb_rush_shift = clip((qb_rush_projection-a["qb_rushes_per_game"])/base_team_rush_attempts * qb_rush_difference, -.065, .065)
    rush_epa += qb_rush_shift
    plays_per_drive = clip(.50*a["plays_per_drive"]+.50*d["plays_per_drive_allowed"], 4.2, 8.3)
    third_down = rate_match(a["third_down_rate"], d["third_down_allowed"], r["third_down"], a_n*.2, d_n*.2, k*.28)
    third_baseline = .5*(shrink(a["third_down_rate"], r["third_down"], a_n*.2, k*.28)+shrink(d["third_down_allowed"], r["third_down"], d_n*.2, k*.28))
    third_delta_ppd = clip((third_down-third_baseline)*1.25, -.12, .12)
    # Red-zone rates shrink by the NUMBER OF OPPORTUNITIES, not total plays.
    rz_off = shrink(a["redzone_td_rate"], r["rz_td"], a["redzone_trips"], r["rz_shrink"])
    rz_def = shrink(d["redzone_td_allowed"], r["rz_td"], d["redzone_trips_faced"], r["rz_shrink"])
    rz_prob = clip(.5*(rz_off+rz_def), .38, .79)
    rz_delta_ppd = clip(.55*(rz_prob-rz_off), -.10, .10)
    td_share = clip(r["td_share_scoring_drives"]+.6*(rz_prob-r["rz_td"]), .38, .82)
    # Turnover production: separate interceptions and fumbles; compare both units consistently.
    interceptions = rate_match(a["interception_rate"], d["interception_rate_forced"], r["interception"], a_n*pass_rate, d_n*.57, k)
    # Pressure-to-turnover dependency: only apply a small INTERACTION residual;
    # main INT/sack effects are already embedded in the teams' EPA rates.
    interception_pressure_adjustment = clip(.12*(pressure-base_press), -.006, .006)
    interceptions = clip(interceptions + interception_pressure_adjustment, .002, .10)
    fumbles = rate_match(a["fumble_lost_rate"], d["fumble_recovery_rate"], r["fumble"], a_n, d_n, k)
    fumbles = clip(fumbles + .06*(expected_sack_rate-r["sack"]), .0008, .04)
    to_drive = clip(1-math.exp(-plays_per_drive*(pass_rate*interceptions+fumbles)), .015, .38)
    # Special teams & penalties as a matchup NET differential, preventing a full
    # double count of context already reflected in offensive and defensive EPA.
    net_special_per_game = (.5*(cp["special_teams_epa_per_game"]-cd["special_teams_epa_per_game"])
                            if cp.get("special_teams_epa_per_game") is not None and cd.get("special_teams_epa_per_game") is not None else 0.0)
    net_penalty_per_game = (.35*(cp["penalty_epa_per_game"]-cd["penalty_epa_per_game"])
                            if cp.get("penalty_epa_per_game") is not None and cd.get("penalty_epa_per_game") is not None else 0.0)
    # Optional experimental residuals: these are NOT substitutes for the true
    # EPA fields above. Coefficients are deliberately tiny, bounded, and NOT
    # historically calibrated. Do not describe as empirically predictive.
    ac, ad = attack.get("advanced"), defend.get("advanced")
    advanced_penalty = 0.0
    if (ac is not None and ad is not None and cp.get("penalty_epa_per_game") is None
            and cd.get("penalty_epa_per_game") is None):
        advanced_penalty=clip(r.get("penalty_yards_points_per_yard",0)*
            (ad["penalty_yards_per_game"]-ac["penalty_yards_per_game"]),-.25,.25)
    advanced_fg=0.0
    if (ac is not None and ac["field_goals_attempted"]>0 and
            cp.get("special_teams_epa_per_game") is None and cd.get("special_teams_epa_per_game") is None):
        # Beta-binomial shrinkage toward prior 84% based on ~20 FG attempts.
        fg_posterior=(ac["field_goals_made"]+20*.84)/(ac["field_goals_attempted"]+20)
        advanced_fg=clip(r.get("fg_proxy_points_scale",0)*(fg_posterior-.84),-.20,.20)
    drives_est = clip(.5*(a["drives_per_game"]+d["drives_faced_per_game"]), 8, 16)
    field_ppd = (c["avg_start_yardline"]-r["ref_start_yardline"])*r["field_position_points_per_yard"]
    efficiency_ppd = plays_per_drive * (pass_rate*pass_epa+(1-pass_rate)*rush_epa)
    adjusted_ppd = clip(r["league_ppd"]+efficiency_ppd+rz_delta_ppd+third_delta_ppd+field_ppd+(net_special_per_game+net_penalty_per_game+advanced_penalty+advanced_fg+home_adv_ppg)/drives_est, .5, 4.8)
    return {
        "team": attack["name"], "side": side, "pass_rate": pass_rate, "plays_per_drive": round(plays_per_drive, 3),
        "expected_drives": round(drives_est*p["pace_multiplier"], 3),
        "sos_adjusted_pass_epa_offense": round(pass_o, 4), "sos_adjusted_pass_epa_defense_allowed": round(pass_d, 4),
        "sos_adjusted_rush_epa_offense": round(rush_o, 4), "sos_adjusted_rush_epa_defense_allowed": round(rush_d, 4),
        "matched_pass_epa": round(pass_epa, 4), "matched_rush_epa": round(rush_epa, 4),
        "pressure_rate": round(pressure, 4), "pressure_residual_epa": round(pressure_residual_epa, 4),
        "pass_success_residual_epa": round(success_pass_residual, 4), "rush_success_residual_epa": round(success_rush_residual, 4),
        "expected_sack_rate_per_dropback": round(expected_sack_rate, 4),
        "expected_sacks_per_game": round(expected_sack_rate*pass_rate*plays_per_drive*drives_est, 3),
        "qb_rushes_per_game": round(qb_rush_projection, 3), "qb_rushing_epa_adjustment": round(qb_rush_shift, 4),
        "pass_success_rate": round(success_pass, 4), "rush_success_rate": round(success_rush, 4),
        "explosive_pass_rate": round(expl_pass, 4), "explosive_rush_rate": round(expl_rush, 4),
        "third_down_conversion": round(third_down, 4), "redzone_td_rate": round(rz_prob, 4),
        "touchdown_share_of_scoring_drives": round(td_share, 4), "interception_rate_per_dropback": round(interceptions, 4),
        "fumble_lost_rate_per_play": round(fumbles, 5), "turnover_drive_probability": round(to_drive, 4),
        "pressure_related_interception_rate_adjustment": round(interception_pressure_adjustment, 5),
        "epa_points_per_drive_adjustment": round(efficiency_ppd, 4), "redzone_residual_ppd": round(rz_delta_ppd, 4),
        "third_down_residual_ppd": round(third_delta_ppd, 4), "field_position_ppd": round(field_ppd, 4),
        "net_penalties_ppg_component": round(net_penalty_per_game, 4), "net_special_teams_ppg_component": round(net_special_per_game, 4),
        "advanced_penalty_yards_proxy_ppg":round(advanced_penalty,4),
        "advanced_fg_accuracy_proxy_ppg":round(advanced_fg,4),
        "advanced_punt_gross_descriptive_not_scored":ac.get("punt_gross_yards") if ac else None,
        "advanced_feature_note":"Experimental bounded effects, NOT backtested; punt gross not used as net punting",
        "home_adv_ppg_component": round(home_adv_ppg, 4),
        "raw_points_per_drive": round(adjusted_ppd, 4),
        "redzone_sample_off_def": [a["redzone_trips"], d["redzone_trips_faced"]],
        "personnel_adjustment_source": p["source"], "personnel_news_checked_at_utc": p["checked_at_utc"], "personnel_notes": p["notes"],
        "_ppd": adjusted_ppd, "_td": td_share, "_to": to_drive,
    }


def money_profit(american: float) -> float:
    return american/100 if american >= 100 else 100/abs(american)


def draw_possession(proj: dict, r: dict, rng: np.random.Generator, active: np.ndarray,
                    bonus: np.ndarray, game_noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(active)
    # Adjust for turnover compared with a centered league turnover baseline.
    # Short-field benefits are applied on subsequent possession, and the baseline
    # benefit is subtracted so league-average opponents remain centered.
    nominal_ppd = (proj["_ppd"] - r["shortfield_bonus_ppd"]*r["league_to_drive"] +
                   bonus.astype(float)*r["shortfield_bonus_ppd"] + game_noise)
    nominal_ppd = np.clip(nominal_ppd, .15, 5.5)
    p_to = proj["_to"]
    avg_score = proj["_td"]*6.95+(1-proj["_td"])*3.0
    scoring_prob = nominal_ppd/avg_score * (1-p_to)/(1-r["league_to_drive"])
    scoring_prob = np.clip(scoring_prob, .02, 1-p_to-.025)
    u, v = rng.random(n), rng.random(n)
    turnover = active & (u < p_to)
    score_event = active & (u >= p_to) & (u < p_to+scoring_prob)
    td = score_event & (v < proj["_td"])
    fg = score_event & ~td
    # Conversions: 6 or 7 for TD, rare missed PAT handled probabilistically.
    td_points = np.where(rng.random(n) < .96, 7, 6)
    points = np.where(td, td_points, np.where(fg, 3, 0))
    return points, turnover


def simulate(home: dict, away: dict, r: dict, spread_home: float | None, total: float | None, runs: int, seed: int,
             drives_fraction: float=1.0, allow_overtime: bool=True) -> dict:
    rng = np.random.default_rng(seed)
    n = runs
    tempo = .5*(home["expected_drives"]+away["expected_drives"])
    drive_counts = np.clip(np.rint(rng.normal(tempo*drives_fraction, max(.6, drives_fraction), n)),
                           max(2, int(7*drives_fraction)), int(17*drives_fraction)+2).astype(int)
    home_scores = np.zeros(n, dtype=np.int32)
    away_scores = np.zeros(n, dtype=np.int32)
    bonus_h = np.zeros(n, dtype=bool)
    bonus_a = np.zeros(n, dtype=bool)
    home_first = rng.random(n) < .5
    # A shared game efficiency factor induces game-level correlation; adds
    # uncertainty to totals beyond the binomial drive events.
    shared_noise = rng.normal(0, r["drive_noise_sd"], n)

    def step(p: dict, active: np.ndarray, bonus: np.ndarray, opp_bonus: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        pts, to = draw_possession(p, r, rng, active, bonus, noise)
        next_opp_bonus = np.where(to, True, opp_bonus)
        return pts, np.where(active, False, bonus), next_opp_bonus

    # Separate groups to respect who takes the opening possession.
    for i in range(int(17*drives_fraction)+2):
        active = i < drive_counts
        h_first = active & home_first
        a_first = active & ~home_first
        if np.any(h_first):
            pts, bonus_h, bonus_a = step(home, h_first, bonus_h, bonus_a, shared_noise)
            home_scores += pts
            pts, bonus_a, bonus_h = step(away, h_first, bonus_a, bonus_h, shared_noise)
            away_scores += pts
        if np.any(a_first):
            pts, bonus_a, bonus_h = step(away, a_first, bonus_a, bonus_h, shared_noise)
            away_scores += pts
            pts, bonus_h, bonus_a = step(home, a_first, bonus_h, bonus_a, shared_noise)
            home_scores += pts

    # Approximate overtime, NOT a simulation of the complete NFL/CFB OT rules.
    # Only impacts the small subset tied after regulation; avoids pretending a tie
    # in regulation is always the final result.
    tied = home_scores == away_scores
    for _ in range(2 if allow_overtime else 0):
        if not np.any(tied):
            break
        h, _ = draw_possession(home, r, rng, tied, np.zeros(n, dtype=bool), shared_noise)
        a, _ = draw_possession(away, r, rng, tied, np.zeros(n, dtype=bool), shared_noise)
        home_scores += h
        away_scores += a
        tied = home_scores == away_scores
    margins = home_scores.astype(float)-away_scores
    settle = margins+spread_home if spread_home is not None else None
    home_cover = float(np.mean(settle > 1e-9)) if settle is not None else None
    away_cover = float(np.mean(settle < -1e-9)) if settle is not None else None
    push = 1-home_cover-away_cover if settle is not None else None
    return {
        "mean_score_home": round(float(home_scores.mean()), 2),
        "mean_score_away": round(float(away_scores.mean()), 2),
        "median_score_home": float(np.median(home_scores)),
        "median_score_away": float(np.median(away_scores)),
        "mean_margin_home": round(float(margins.mean()), 2),
        "median_margin_home": round(float(np.median(margins)), 2),
        "margin_p10_p50_p90": [float(x) for x in np.quantile(margins, [.1,.5,.9])],
        "margin_standard_deviation": round(float(margins.std()), 2),
        "home_cover_probability_raw": round(home_cover, 5) if home_cover is not None else None,
        "away_cover_probability_raw": round(away_cover, 5) if away_cover is not None else None,
        "push_probability": round(push, 5) if push is not None else None,
        "home_win_probability_raw": round(float(np.mean(margins>0) + .5*np.mean(margins==0)), 5),
        "over_probability_raw": round(float(np.mean(home_scores+away_scores>total)), 5) if total is not None else None,
        "under_probability_raw": round(float(np.mean(home_scores+away_scores<total)), 5) if total is not None else None,
        "total_push_probability": round(float(np.mean(home_scores+away_scores==total)), 5) if total is not None else None,
        "mean_total": round(float(np.mean(home_scores+away_scores)), 2),
        "home_cover_monte_carlo_se": round(float(math.sqrt(home_cover*(1-home_cover)/n)), 5) if home_cover is not None else None,
        "simulations": runs, "seed": seed,
        "simulation_note": ("Approximate possession model with provisional priors; " +
                            ("simplified overtime. " if allow_overtime else "no overtime (first half). ")+
                            "Probabilities NOT empirically calibrated. MC uncertainty omits model error."),
    }


def stress_test(home: dict, away: dict, r: dict, m: dict, n: int, seed: int) -> dict:
    """Re-run explicit adverse cases with identical random seeds (common RNG)."""
    common_n = max(3000, min(15000, n//5))
    cases = {}

    def record(name: str, h: dict, a: dict) -> None:
        s = simulate(h, a, r, m["spread_home"], m["total"], common_n, seed)
        cases[name] = {"home_cover_raw": s["home_cover_probability_raw"],
                       "away_cover_raw": s["away_cover_probability_raw"],
                       "mean_margin_home": s["mean_margin_home"], "mean_total": s["mean_total"]}

    h, a = dict(home), dict(away)
    record("baseline_resampled", h, a)
    h, a = dict(home), dict(away)
    h["_to"] = a["_to"] = r["league_to_drive"]
    record("turnover_rates_regress_to_league", h, a)
    h, a = dict(home), dict(away)
    for p in (h, a):
        p["_td"] = r["td_share_scoring_drives"]
        p["_ppd"] -= p["redzone_residual_ppd"]
    record("red_zone_reversion", h, a)
    h, a = dict(home), dict(away)
    h["_ppd"] -= .40
    a["_ppd"] += .15
    record("home_adverse_offensive_script", h, a)
    h, a = dict(home), dict(away)
    a["_ppd"] -= .40
    h["_ppd"] += .15
    record("away_adverse_offensive_script", h, a)
    h, a = dict(home), dict(away)
    h["_ppd"] -= h["plays_per_drive"]*h["pass_rate"]*.075
    a["_ppd"] += a["plays_per_drive"]*a["pass_rate"]*.075
    record("pass_matchup_swings_toward_away", h, a)
    h, a = dict(home), dict(away)
    a["_ppd"] -= a["plays_per_drive"]*a["pass_rate"]*.075
    h["_ppd"] += h["plays_per_drive"]*h["pass_rate"]*.075
    record("pass_matchup_swings_toward_home", h, a)
    by_line = {}
    for shift in (-1.5, -.5, 0, .5, 1.5):
        s = simulate(home, away, r, m["spread_home"]+shift, m["total"], common_n, seed)
        by_line[f"{m['spread_home']+shift:+g}"] = {
            "home_cover_raw": s["home_cover_probability_raw"],
            "away_cover_raw": s["away_cover_probability_raw"],
            "push": s["push_probability"],
        }
    return {
        "simulations_per_case": common_n,
        "scenario_results": cases,
        "spread_home_sensitivity": by_line,
        "limitations": "Hypothetical perturbations, not measured scenario frequencies; they cannot establish calibrated probabilities.",
    }


def run(g: dict, runs: int=50000, seed: int=20261018, perform_stress: bool=True, calibration: dict | None=None) -> dict:
    if not (1000 <= runs <= 300000):
        raise InputError("runs must be between 1,000 and 300,000")
    if not (0 <= seed < 2**32):
        raise InputError("seed must be a nonnegative 32-bit integer")
    r = dict(validate_game(g))  # NEVER mutate global reference constants
    sources=[g[t].get('pressure_source') for t in ('home','away')]
    if any(x=='PFR_QB_TIMES_PRESSURED_OVER_SAME_GAME_PBP_DROPBACKS' for x in sources):
        if not all(x=='PFR_QB_TIMES_PRESSURED_OVER_SAME_GAME_PBP_DROPBACKS' for x in sources):
            raise InputError('Both teams must use the same pressure definition')
        r['pressure']=number(g.get('pressure_reference_rate'),'PFR pressure reference rate',.10,.65)
    hfa = 0 if g["neutral_site"] else r["home_advantage_points"]
    home = projection(g["home"], g["away"], r, "home", .5*hfa)
    away = projection(g["away"], g["home"], r, "away", -.5*hfa)
    m = g["market"]
    sim = simulate(home, away, r, float(m["spread_home"]), float(m["total"]), runs, seed)
    stress = stress_test(home, away, r, m, runs, seed+7) if perform_stress else None
    first_half = None
    if "first_half" in m and m["first_half"] is not None:
        fh = m["first_half"]
        fh_sim = simulate(home, away, r, fh["spread_home"], fh["total"], runs, seed+23,
                          drives_fraction=.5, allow_overtime=False)
        hp, ap = money_profit(fh["home_odds_american"]), money_profit(fh["away_odds_american"])
        first_half = {
            "market": fh,
            "simulation": fh_sim,
            "home_ev_RAW_UNCALIBRATED": round(fh_sim["home_cover_probability_raw"]*hp-fh_sim["away_cover_probability_raw"],4),
            "away_ev_RAW_UNCALIBRATED": round(fh_sim["away_cover_probability_raw"]*ap-fh_sim["home_cover_probability_raw"],4),
            "note": "Do not call first-half stronger than full-game from raw uncalibrated probabilities. Separate empirical first-half calibration is required."
        }
    home_profit = money_profit(m["home_odds_american"])
    away_profit = money_profit(m["away_odds_american"])
    home_ev = sim["home_cover_probability_raw"]*home_profit-sim["away_cover_probability_raw"]
    away_ev = sim["away_cover_probability_raw"]*away_profit-sim["home_cover_probability_raw"]
    audit = {
        "schema_complete": True, "strict_source_dates_checked": True,
        "opponent_strength_adjusted": True, "small_sample_shrinkage": True,
        "pass_and_rush_marginal_models": True, "qb_rush_volume_adjusted": True,
        "pass_pressure_sack_interaction": True, "success_and_explosives_checked": True,
        "turnover_interaction": True, "redzone_regression": True,
        "third_down_situations": True, "special_teams_and_penalties": True,
        "field_position_and_pace": True, "personnel_adjustments_traced": True,
        "drive_level_score_distribution": True, "spread_and_vig_checked": True,
        "first_half_market_evaluated": first_half is not None,
        "supplemental_pfr_pressure_rates_used":any(g[t].get('pressure_source')=='PFR_QB_TIMES_PRESSURED_OVER_SAME_GAME_PBP_DROPBACKS' for t in ('home','away')),
        "supplemental_proxy_coefficients_empirically_calibrated":False,
        "optional_causal_penalty_or_specialteams_epa_omitted":any(
            g[t]['context'].get(field) is None for t in ('home','away')
            for field in ('penalty_epa_per_game','special_teams_epa_per_game')),
        "stress_tests": perform_stress, "historical_out_of_sample_validation": False,
        "probabilities_empirically_calibrated": False,
    }
    is_synthetic = any('SYNTHETIC' in g[t]['stat_source'].upper() for t in ('home','away'))
    calibration_result = None
    if calibration is not None and is_synthetic:
        raise InputError('Do not apply real calibration to a synthetic example game')
    if calibration is not None:
        if not isinstance(calibration, dict) or calibration.get("model_version")!=VERSION or calibration.get("league")!=g["league"]:
            raise InputError("Calibration file incompatible with model version or league")
        if calibration.get("validation_status")!="HOLDOUT_CHECK_PASSED":
            raise InputError("Calibration did not pass chronological holdout; cannot be applied")
        if utc(calibration["holdout_last_kickoff_utc"], "calibration.holdout_last_kickoff_utc") >= utc(g["kickoff_utc"], "kickoff_utc"):
            raise InputError("Calibration includes games at/after target kickoff (lookahead)")
        a_cal, b_cal = float(calibration["calibration_alpha"]), float(calibration["calibration_beta"])
        if not math.isfinite(a_cal+b_cal) or b_cal<=0:
            raise InputError("Invalid calibration coefficients")
        live=1-sim["push_probability"]
        raw_conditional=sim["home_cover_probability_raw"]/max(live,1e-8)
        p_cal=sigmoid(a_cal+b_cal*logit(raw_conditional))
        home_c=live*p_cal
        away_c=live*(1-p_cal)
        home_ev_c=home_c*home_profit-away_c
        away_ev_c=away_c*away_profit-home_c
        calibration_result = {
            "historical_train_games":calibration["train_games"],
            "historical_holdout_games":calibration["holdout_games"],
            "calibration_holdout_brier":calibration["holdout_calibrated"]["brier"],
            "calibrated_home_cover_probability":round(home_c,5),
            "calibrated_away_cover_probability":round(away_c,5),
            "home_EV_per_unit":round(home_ev_c,4),
            "away_EV_per_unit":round(away_ev_c,4),
            "warning":"A good holdout calibration score does not establish positive betting ROI or guarantee model accuracy."
        }
    result = {
        "version": VERSION, "game_id": g["id"], "league": g["league"],
        "teams": {"home": g["home"]["name"], "away": g["away"]["name"]},
        "snapshot_utc": g["data_as_of_utc"], "kickoff_utc": g["kickoff_utc"],
        "input_sha256": hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest(),
        "model_status": "SYNTHETIC_EXAMPLE_UNCALIBRATED" if is_synthetic else "EXPLORATORY_UNCALIBRATED",
        "grade": "UNRATED", "wager_decision": "PASS_UNTIL_BACKTESTED",
        "market": m, "projection_home": {k:v for k,v in home.items() if not k.startswith("_")},
        "projection_away": {k:v for k,v in away.items() if not k.startswith("_")},
        "simulation": sim, "first_half": first_half, "stress_tests": stress,
        "calibration": calibration_result,
        "market_comparison_exploratory": {
            "market_implied_margin_home": -m["spread_home"],
            "model_minus_market_margin": round(sim["mean_margin_home"]+m["spread_home"], 2),
            "home_cover_break_even_excluding_push": round(1/(home_profit+1), 5),
            "away_cover_break_even_excluding_push": round(1/(away_profit+1), 5),
            "home_ev_per_unit_staked_RAW_UNCALIBRATED": round(home_ev, 4),
            "away_ev_per_unit_staked_RAW_UNCALIBRATED": round(away_ev, 4),
        },
        "audit": audit,
        "required_followup": "Out-of-sample backtest, probability calibration and sensitivity runs are prerequisites to interpreting raw cover probability as a betting edge.",
    }
    if calibration_result is not None:
        result["model_status"]="CHRONOLOGICAL_HOLDOUT_CALIBRATOR_APPLIED"
        audit["historical_out_of_sample_validation"]=True
        audit["probabilities_empirically_calibrated"]=True
        # Do not auto-promote to A/B based on backtest alone: realized returns
        # and robustness must independently validate the bet/grade thresholds.
        result["wager_decision"]="PASS_PENDING_INDEPENDENT_ROI_VALIDATION"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Fully populated pregame JSON")
    parser.add_argument("--runs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None, help="Human-readable report Markdown")
    parser.add_argument("--calibration", type=Path, default=None, help="Validated historical calibration JSON")
    args = parser.parse_args()
    try:
        game = json.loads(args.input.read_text(encoding="utf-8"))
        cal = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
        result = run(game, args.runs, args.seed, calibration=cal)
    except (InputError, ValueError, KeyError, TypeError) as e:
        parser.error(str(e))
    result_json = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result_json+"\n", encoding="utf-8")
        print(f"Saved audit: {args.output}")
    else:
        print(result_json)
    if args.markdown:
        from report import render
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render(result)+"\n", encoding="utf-8")
        print(f"Saved human-readable report: {args.markdown}")

if __name__ == "__main__":
    main()
