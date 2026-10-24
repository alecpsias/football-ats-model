"""Reproducible same-seed 5000-draw ablation of Build16 joint-drive coupling.

Independent historical holdout fixtures. The Duke-Virginia motivating game is
excluded. Outcome of a fixture is never used to construct its prior-game input.
No historic sportsbook lines are invented and no ATS accuracy is inferred.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cfb.failure2_paired import cohort
from cfb.failure1_learned_shadow import correction as fitted_rush
from cfb.mechanism_build16 import simulate_build16
from cfb.pipeline import verify

DUKE_VIRGINIA_ID = "401858271"


def evaluate(snapshot, max_games=120):
    manifest = verify(Path(snapshot))
    year = int(manifest["season"])
    if year not in (2025, 2026):
        raise ValueError("Only 2025 and 2026 chronological replay are authorized")
    fixtures, screening = cohort(snapshot, max_games, enrich_rush=True)
    matched = []
    missing = Counter()
    for index, fixture in enumerate(fixtures):
        if fixture["game_id"] == DUKE_VIRGINIA_ID:
            missing["DUKE_VIRGINIA_MOTIVATING_GAME_EXCLUDED"] += 1
            continue
        home, away = fixture["home_id"], fixture["away_id"]
        history = fixture["historical"]
        seed = int(fixture["game_id"]) % (2**32)
        common = dict(
            runs=5000, seed=seed, coupled_rush_possessions=True,
            failure1_fit_margin_delta=fitted_rush(home, away, history)["margin_delta_points"],
            continuous_rush_sos_enabled=True,
            failure2_attribution=True, failure3_sustainability=True)
        a = simulate_build16(home, away, history, fixture["neutral"],
                             joint_drive_enabled=False, **common)
        b = simulate_build16(home, away, history, fixture["neutral"],
                             joint_drive_enabled=True, **common)
        before, after = a.get("simulation"), b.get("simulation")
        if before is None or after is None:
            missing["MISSING_REQUIRED_PROJECTION"] += 1
            continue
        if before["simulations"] != 5000 or after["simulations"] != 5000:
            raise AssertionError("Exactly 5000 same-seed simulations required")
        if not after.get("joint_drive_active"):
            missing[b["joint_drive_process"]["status"]] += 1
            continue
        if before.get("joint_drive_active"):
            raise AssertionError("Reference arm contains candidate mechanism")
        actual_margin = float(fixture["actual_margin"])
        actual_total = float(fixture["actual_total"])
        matched.append({
            "game_id": fixture["game_id"], "kickoff": fixture["kickoff"],
            "seed": seed, "actual_margin": actual_margin,
            "actual_total": actual_total,
            "reference_margin": before["mean_margin_home"],
            "candidate_margin": after["mean_margin_home"],
            "reference_total": before["mean_total"],
            "candidate_total": after["mean_total"],
            "reference_p10_p90": [before["margin_p10_p50_p90"][0],
                                   before["margin_p10_p50_p90"][2]],
            "candidate_p10_p90": [after["margin_p10_p50_p90"][0],
                                   after["margin_p10_p50_p90"][2]],
            "reference_win_probability": before["home_win_probability_raw"],
            "candidate_win_probability": after["home_win_probability_raw"],
            "sampled_rush_plays_home": after["joint_drive_avg_rush_plays_home"],
            "sampled_rush_plays_away": after["joint_drive_avg_rush_plays_away"],
            "sampled_possession_seconds_home": after["joint_drive_avg_possession_seconds_home"],
            "sampled_possession_seconds_away": after["joint_drive_avg_possession_seconds_away"]})
        if (index+1)%20==0:
            print("JOINT_DRIVE_PROGRESS",year,index+1,len(matched),flush=True)

    def metrics(rows):
        if not rows:
            return {"games": 0}
        actual=np.array([r["actual_margin"] for r in rows])
        old=np.array([r["reference_margin"] for r in rows])
        new=np.array([r["candidate_margin"] for r in rows])
        total=np.array([r["actual_total"] for r in rows])
        old_total=np.array([r["reference_total"] for r in rows])
        new_total=np.array([r["candidate_total"] for r in rows])
        covered_old=np.mean([r["reference_p10_p90"][0]<=r["actual_margin"]<=r["reference_p10_p90"][1] for r in rows])
        covered_new=np.mean([r["candidate_p10_p90"][0]<=r["actual_margin"]<=r["candidate_p10_p90"][1] for r in rows])
        probs0=np.array([r["reference_win_probability"] for r in rows])
        probs1=np.array([r["candidate_win_probability"] for r in rows])
        winners=(actual>0).astype(float)
        delta=np.abs(actual-old)-np.abs(actual-new)
        boot=np.random.default_rng(20261023+year).choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
        return {"games":len(rows),
           "margin_MAE_reference":round(float(np.mean(np.abs(actual-old))),4),
           "margin_MAE_candidate":round(float(np.mean(np.abs(actual-new))),4),
           "margin_MAE_improvement_positive_is_better":round(float(delta.mean()),4),
           "bootstrap_95_MAE_improvement":np.percentile(boot,[2.5,97.5]).round(4).tolist(),
           "total_MAE_reference":round(float(np.mean(np.abs(total-old_total))),4),
           "total_MAE_candidate":round(float(np.mean(np.abs(total-new_total))),4),
           "win_Brier_reference":round(float(np.mean((probs0-winners)**2)),5),
           "win_Brier_candidate":round(float(np.mean((probs1-winners)**2)),5),
           "margin_p10p90_coverage_reference":round(float(covered_old),4),
           "margin_p10p90_coverage_candidate":round(float(covered_new),4)}
    return {
        "schema":"cfb_build16_joint_drive_paired_v1", "season":year,
        "source_capture_at":manifest.get("captured_at_utc"),
        "retrospective_source_snapshot_not_independent_point_in_time_verified":True,
        "duke_virginia_excluded_from_fitting_and_scoring":True,
        "historical_ATS_accuracy":None,"verified_spreads":0,
        "baseline_unchanged":True,"one_challenger":True,
        "sample_size_requested":max_games,
        "screen":screen,"insufficient":dict(missing),
        "metrics":metrics(matched),"games":matched}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--snapshot",required=True)
    parser.add_argument("--season",type=int,required=True)
    parser.add_argument("--max-games",type=int,default=120)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    result=evaluate(args.snapshot,args.max_games)
    if result["season"] != args.season:
        raise ValueError("Incorrect season in snapshot")
    dest=Path(args.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,sort_keys=True,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="games"},indent=2))


if __name__=="__main__":
    main()
