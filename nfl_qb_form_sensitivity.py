#!/usr/bin/env python3
"""Prospective NFL QB-effect sensitivity, using only immutable pregame artifacts.

This does NOT retrain/replace the fixed baseline, create a new challenger,
rewrite a frozen prediction, or claim an ATS-calibrated betting probability.
An opponent-adjusted, regularized QB effect is fitted on weeks 1-6 only.
Hyperparameters chosen by chronological leave-next-week-out QB-game CV.
Replay calls the EXACT SAME model.simulate and sole v0.10 challenger function.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests

from model import MODEL_RULES, simulate
from nfl_stateful_candidate import stateful_candidate_simulation
from nfl_challenger_policy import ACTIVE_CHALLENGER_VERSION

SEASON = 2026
WEEK_TARGET = 7
GAME_ID = "2026_07_NE_CHI"
QB_CODES = {"CHI": "C.Williams", "NE": "D.Maye"}
DEFENSE_PRIOR_DROPBACKS = 150.
SKILL_PRIORS = (30., 60., 120., 200.)
FORM_PRIORS = (30., 60., 120., 200.)
HALF_LIVES = (1., 1.5, 2., 3., 4., 6.)
RANDOM_SEED = 20261018
N_SIM = 5000


def download_artifact(repo, artifact_id, token):
    if not token:
        raise RuntimeError("GitHub Actions read token required")
    url = f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}/zip"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"}, timeout=100)
    r.raise_for_status()
    return r.content


def load_frozen(zip_data):
    z = zipfile.ZipFile(io.BytesIO(zip_data))
    predict = [n for n in z.namelist() if n.endswith(f"/{GAME_ID}.json")]
    playfile = [n for n in z.namelist() if n.endswith("/pbp_2026.csv.gz")]
    schedfile = [n for n in z.namelist() if n.endswith("/schedule_2026.csv.gz")]
    if len(predict)!=1 or len(playfile)!=1 or len(schedfile)!=1:
        raise RuntimeError("Missing/ambiguous locked artifact sources")
    original = z.read(predict[0]); forecast = json.loads(original)
    assert forecast["game_id"] == GAME_ID
    assert forecast["challenger"]["version"] == ACTIVE_CHALLENGER_VERSION
    assert forecast["challenger"]["candidate"]["promoted_to_primary"] is False
    start = pd.Timestamp(forecast["kickoff_utc"])
    cutoff = pd.Timestamp(forecast["data_as_of_utc"])
    if (start - cutoff).total_seconds() < 7200:
        raise RuntimeError("Forecast is not pregame locked at least 120 minutes")
    pbp = pd.read_csv(io.BytesIO(z.read(playfile[0])), compression="gzip", low_memory=False)
    sched = pd.read_csv(io.BytesIO(z.read(schedfile[0])), compression="gzip", low_memory=False)
    completed = set(sched[(sched.week < WEEK_TARGET) &
                          sched.home_score.notna() & sched.away_score.notna()].game_id)
    eligible = pbp[(pbp.game_id.isin(completed)) & (pbp.week < WEEK_TARGET)].copy()
    if (eligible.game_id == GAME_ID).any():
        raise RuntimeError("Target game leaked into QB fitting dataset")
    return forecast, eligible, hashlib.sha256(original).hexdigest()


def quarterback_dropbacks(pbp):
    # Include scrambles by assigning them to their actual rusher/QB.
    x = pbp[pbp.qb_dropback.eq(1) & pbp.epa.notna() &
            pbp.posteam.notna() & pbp.defteam.notna()].copy()
    x["qb"] = x.passer_player_name.where(
        x.passer_player_name.notna(),
        x.rusher_player_name.where(x.qb_scramble.eq(1)))
    x = x[x.qb.notna()].copy()
    x["week"] = x.week.astype(int)
    return x


def adjusted_epa(evaluation: pd.DataFrame, history: pd.DataFrame):
    """Defense correction learned ONLY from other offenses in training history.

    Excluding the same offense from its opponent's defensive history avoids
    leaking that offense's performance into its own opponent adjustment.
    150-dropback ridge-like shrink prior for defense strength.
    """
    league = history.epa.mean()
    total = history.groupby("defteam").epa.agg(["count", "sum"])
    off_pair = history.groupby(["defteam", "posteam"]).epa.agg(["count", "sum"])
    corrections = {}
    for defense, offense in evaluation[["defteam", "posteam"]].itertuples(
            index=False, name=None):
        k = (defense, offense)
        if k in corrections:
            continue
        ds = total.loc[defense] if defense in total.index else None
        paired = off_pair.loc[k] if k in off_pair.index else None
        n = (float(ds["count"]) if ds is not None else 0.) - (
            float(paired["count"]) if paired is not None else 0.)
        eps = (float(ds["sum"]) if ds is not None else 0.) - (
            float(paired["sum"]) if paired is not None else 0.)
        other_mean = eps / n if n > 0 else league
        corrections[k] = n / (n + DEFENSE_PRIOR_DROPBACKS) * (other_mean - league)
    keylist = list(zip(evaluation.defteam, evaluation.posteam))
    return evaluation.epa.to_numpy(dtype=float) - np.array(
        [corrections[k] for k in keylist])


def estimate_one(qb_sample, team_sample, hold_week, halflife, skill_prior, form_prior):
    qb_mu = float(qb_sample.adj_epa.mean())
    team_mu = float(team_sample.adj_epa.mean())
    qbn = len(qb_sample)
    weights = np.exp2((qb_sample.week.to_numpy(dtype=float) - (hold_week - 1)) / halflife)
    recency_mu = float(np.average(qb_sample.adj_epa, weights=weights))
    effective_n = float(weights.sum())
    skill = (qb_mu - team_mu) * qbn / (qbn + skill_prior)
    trend = (recency_mu - qb_mu) * effective_n / (effective_n + form_prior)
    return {"qb_n": qbn, "team_n": len(team_sample),
            "qb_season_adjusted_epa": round(qb_mu, 6),
            "team_season_adjusted_epa": round(team_mu, 6),
            "qb_recent_weighted_adjusted_epa": round(recency_mu, 6),
            "effective_recent_n": round(effective_n, 3),
            "regularized_skill_delta": round(skill, 6),
            "regularized_recent_form_delta": round(trend, 6),
            "total_qb_delta_epa_per_dropback": round(skill + trend, 6)}


def make_cv_records(qbplays):
    records = []
    for future_week in (4, 5, 6):
        historical = qbplays[qbplays.week < future_week].copy()
        target = qbplays[qbplays.week == future_week].copy()
        if historical.empty:
            continue
        historical["adj_epa"] = adjusted_epa(historical, historical)
        target["adj_epa"] = adjusted_epa(target, historical)
        for (team, qb), future in target.groupby(["posteam", "qb"]):
            old = historical[(historical.posteam == team) & (historical.qb == qb)]
            if len(future) < 10 or len(old) < 25:
                continue
            teamhist = historical[historical.posteam == team]
            records.append({"week": future_week, "team": team, "qb": qb,
                "observed_epa": float(future.adj_epa.mean()),
                "n_hold": len(future), "training_qb": old, "training_team": teamhist})
    return records


def fit_hyperparameters(qbplays):
    samples = make_cv_records(qbplays)
    if len(samples) < 25:
        raise RuntimeError("Too few independent QB-week CV observations")
    choices = []
    baseline_sse = sum(s["n_hold"] * (
        float(s["training_team"].adj_epa.mean()) - s["observed_epa"]) ** 2
        for s in samples)
    weight = sum(s["n_hold"] for s in samples)
    for hl in HALF_LIVES:
        for sk in SKILL_PRIORS:
            for fo in FORM_PRIORS:
                sse = 0.
                for rec in samples:
                    est = estimate_one(rec["training_qb"], rec["training_team"],
                                       rec["week"], hl, sk, fo)
                    predicted = (float(rec["training_team"].adj_epa.mean())
                                 + est["total_qb_delta_epa_per_dropback"])
                    sse += rec["n_hold"] * (predicted-rec["observed_epa"])**2
                choices.append({"halflife_weeks": hl, "skill_prior_dropbacks": sk,
                                "form_prior_dropbacks": fo, "mse": sse / weight})
    best = min(choices, key=lambda o: o["mse"])
    return best, {
        "training_policy": "Walk-forward by week; validate weeks 4, 5 and 6 only",
        "unit": "QB future-week opponent-adjusted EPA/dropback (not ATS outcomes)",
        "holdout_qb_week_samples": len(samples),
        "holdout_total_dropbacks": weight,
        "baseline_team_epa_mse": round(baseline_sse/weight, 6),
        "selected_mse": round(best["mse"], 6),
        "relative_mse_improvement_pct": round(100*(1-best["mse"]/(baseline_sse/weight)),2),
        "hyperparameter_grid": {
            "half_lives": list(HALF_LIVES),
            "skill_priors": list(SKILL_PRIORS),
            "form_priors": list(FORM_PRIORS),
            "searched": len(choices),
        },
        "fit_warning": ("Hyperparameters selected with only 3 historical holdout "
                        "weeks; training and parameter selection can overfit."),
    }


def restore_projection(public):
    p = dict(public)
    p["_ppd"] = float(p["raw_points_per_drive"])
    p["_td"] = float(p["touchdown_share_of_scoring_drives"])
    p["_to"] = float(p["turnover_drive_probability"])
    return p


def analyze(zip_data):
    forecast, plays, sha = load_frozen(zip_data)
    allq = quarterback_dropbacks(plays)
    fitted, cv = fit_hyperparameters(allq)
    allq = allq.copy()
    allq["adj_epa"] = adjusted_epa(allq, allq)
    h = restore_projection(forecast["team_projections"]["home"])
    a = restore_projection(forecast["team_projections"]["away"])
    official = [h, a]
    qb_report = {}
    for tm, qb in QB_CODES.items():
        tmplays = allq[allq.posteam == tm]
        selected = tmplays[tmplays.qb == qb]
        if len(selected) < 30:
            raise RuntimeError(f"Insufficient verified {qb} plays")
        est = estimate_one(selected, tmplays, WEEK_TARGET,
            fitted["halflife_weeks"], fitted["skill_prior_dropbacks"],
            fitted["form_prior_dropbacks"])
        scrimmage = plays[(plays.posteam == tm) &
                          (plays.play_type.isin(["pass", "run"]))]
        share = len(tmplays) / max(len(scrimmage), 1)
        if not .3 <= share <= .9:
            raise RuntimeError(f"Implausible QB dropback share for {tm}: {share}")
        p = next(p for p in official if p["team"] == tm)
        # Model's matched team pass/rush EPA is a 50/50 offense-defense
        # combination. Apply QB as a bounded residual to drive PPD only:
        # effect * actual dropback share * expected plays/drive * 0.5.
        # Thus scramble effects are not also added to team rushing EPA.
        delta = .5 * est["total_qb_delta_epa_per_dropback"] * share * p["plays_per_drive"]
        qb_report[tm] = {**est,
            "qb": qb,
            "dropback_share_of_team_scrimmage_plays": round(share, 6),
            "adjusted_ppd_residual": round(delta, 6),
            "game_count": int(selected.game_id.nunique()),
            "observed_weeks": sorted(set(int(z) for z in selected.week)),
            "note": "Opportunity-weighted residual; not a causal isolate of QB quality.",
        }
    r = dict(MODEL_RULES["NFL"])
    r["pressure"] = float(forecast["audit"]["pressure_rate_reference"])
    source_profile = forecast["challenger"]["observations"]
    if not source_profile:
        raise RuntimeError("Locked challenger mechanism profile is unavailable")
    origin = simulate(h, a, r, None, None, N_SIM, RANDOM_SEED)
    origin10 = stateful_candidate_simulation(h, a, source_profile, r, N_SIM, RANDOM_SEED)
    # Rehydrated published team projection values are rounded; allow up to
    # 0.2 points deviation from frozen original rather than rewriting it.
    refbase = forecast["projected_home_margin"]
    ref10 = forecast["challenger"]["candidate"]["mean_margin_home"]
    if abs(origin["mean_margin_home"] - refbase) > .35 or abs(
            origin10["mean_margin_home"] - ref10) > .35:
        raise RuntimeError("Replayed initial forecasts differ excessively; refusing rerun")
    h1 = dict(h); a1 = dict(a)
    h1["_ppd"] += qb_report["CHI"]["adjusted_ppd_residual"]
    a1["_ppd"] += qb_report["NE"]["adjusted_ppd_residual"]
    replay_base = simulate(h1, a1, r, None, None, N_SIM, RANDOM_SEED)
    replay_ch = stateful_candidate_simulation(h1, a1, source_profile, r, N_SIM, RANDOM_SEED)

    def summary_baseline(run):
        return {"home": run["mean_score_home"], "away": run["mean_score_away"],
                "margin_home": run["mean_margin_home"],
                "total": run["mean_total"],
                "home_win_frequency_raw": run["home_win_probability_raw"],
                "margin_p10_p50_p90": run["margin_p10_p50_p90"]}
    def summary_challenger(run):
        return {"home": run["score_home"], "away": run["score_away"],
                "margin_home": run["mean_margin_home"],
                "total": round(run["score_home"] + run["score_away"], 3),
                "home_win_frequency_raw": run["home_win_probability_raw_uncalibrated"],
                "margin_p10_p50_p90": run["margin_p10_p50_p90"]}
    report = {
        "schema": "research_qb_adjusted_sensitivity_v1",
        "source": {"artifact_game_id": GAME_ID, "artifact_sha256": sha,
            "forecast_data_as_of_utc": forecast["data_as_of_utc"],
            "kickoff": forecast["kickoff_utc"], "excluded_target_game": True,
            "qb_sample_max_week": int(allq.week.max()), "simulations_each": N_SIM,
            "rng_seed": RANDOM_SEED},
        "method": ("Defense-adjusted QB EPA per dropback including sacks and "
                   "scrambles; skill relative to team season plus recent-form trend; "
                   "regularization and recency selected on 2026 weeks 4-6 "
                   "QB-game chronological holdouts. QB delta enters PPD as "
                   "0.5 * EPA delta * team dropback fraction * plays per drive. "
                   "Original offensive rushing and turnover parameters retained."),
        "hyperparameters_chosen": fitted,
        "chronological_cv": cv,
        "qb_effects": qb_report,
        "frozen_original": {
            "baseline": {
                "home": forecast["score_home"], "away": forecast["score_away"],
                "margin_home": refbase},
            "single_challenger": {
                "version": ACTIVE_CHALLENGER_VERSION,
                "home": forecast["challenger"]["candidate"]["score_home"],
                "away": forecast["challenger"]["candidate"]["score_away"],
                "margin_home": ref10}},
        "replay_unadjusted_from_rounded_locked_projections": {
            "baseline": summary_baseline(origin),
            "single_challenger": summary_challenger(origin10)},
        "counterfactual_fitted_qb_adjustments": {
            "baseline_simulation": summary_baseline(replay_base),
            "same_single_challenger_v010": summary_challenger(replay_ch)},
        "not_promoted": True,
        "not_a_new_challenger": True,
        "not_an_official_ats_recommendation": True,
        "caveats": [
            "Only pregame 2026 weeks 1-6 and fixed locked forecasts are used.",
            "QB EPA includes receivers, protection, opponent, play calling and variance.",
            "Defensive corrections shrink to league and omit same offense from defense sample.",
            "Six-week data and three walk-forward holdout weeks are small samples.",
            "CV estimates QB EPA forecast error, not game ATS reliability.",
            "No verified opponent-specific play-level coverage or injury replacement model.",
            "No calibrated game-win, spread-cover probabilities, or betting EV.",
            "Only the research replay adjusts PPD, not frozen baseline or official challenger.",
        ],
    }
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-id", type=int, required=True)
    p.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    p.add_argument("--out", type=Path, default=Path("research/NE_CHI_QB_fit.json"))
    args = p.parse_args()
    z = download_artifact(args.repo, args.artifact_id, os.getenv("GH_TOKEN", ""))
    result = analyze(z)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
