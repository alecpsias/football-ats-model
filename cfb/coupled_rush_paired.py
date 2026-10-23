"""2025/2026 5k paired check: default Build16 coupled 2-way drive scripts.

The 2024 rush/SOS model is held constant in both arms. 5,000 runs each,
same seed, same earlier-game source histories; no result from target game used
as a pregame feature. No verified historic betting spreads are synthesized.
The initiating Liberty-Kennesaw game is excluded from scoring/promotion.
"""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from cfb.failure2_paired import cohort
from cfb.failure1_learned_shadow import correction as f1_rush

MOTIVATING_ID="401871056"

def evaluate(snapshot,max_games=120):
    from cfb.pipeline import verify
    from cfb.mechanism_build16 import simulate_build16
    manifest=verify(Path(snapshot))
    year=int(manifest['season'])
    if year not in (2025,2026):raise ValueError("Strict later-than-2024 validation only")
    games,screen=cohort(snapshot,max_games,enrich_rush=True)
    records=[];unavailable=Counter()
    for index,g in enumerate(games):
        if g['game_id']==MOTIVATING_ID:
            unavailable["LIBERTY_KENNESAW_MOTIVATING_CASE_HELD_OUT"]+=1
            continue
        h,a=g['home_id'],g['away_id']
        common=dict(runs=5000,seed=int(g['game_id'])%(2**32),
          failure1_fit_margin_delta=f1_rush(h,a,g['historical'])['margin_delta_points'],
          continuous_rush_sos_enabled=True,
          failure2_attribution=True,failure3_sustainability=True)
        reference=simulate_build16(h,a,g['historical'],g['neutral'],
            coupled_rush_possessions=False,**common)
        candidate=simulate_build16(h,a,g['historical'],g['neutral'],
            coupled_rush_possessions=True,**common)
        r0,r1=reference.get('simulation'),candidate.get('simulation')
        if r0 is None or r1 is None:
            unavailable["UNAVAILABLE_REQUIRED_EPA"]+=1;continue
        if r0['simulations']!=5000 or r1['simulations']!=5000:
            raise AssertionError("5000 actual paired draws required")
        cfg=candidate["coupled_rush_possession_interaction"]
        if cfg["status"]!="ACTIVE_OBSERVED_COUPLED_RUSH_CHAIN_SCRIPTS":
            unavailable[cfg["status"]]+=1
            continue
        actual=float(g['actual_margin'])
        if not np.isfinite(actual):raise ValueError('Actual margin source invalid')
        if r0.get("coupled_rush_chain_active") or not r1.get("coupled_rush_chain_active"):
            raise AssertionError("Paired mechanism activation is inconsistent")
        records.append({
          "game_id":g["game_id"],"kickoff":g["kickoff"],
          "home":g["home"],"away":g["away"],"actual_home_margin":actual,
          "margin_control":r0["mean_margin_home"],
          "margin_coupled":r1["mean_margin_home"],
          "absolute_error_control":round(abs(actual-r0["mean_margin_home"]),5),
          "absolute_error_coupled":round(abs(actual-r1["mean_margin_home"]),5),
          "total_control":r0["mean_total"],"total_coupled":r1["mean_total"],
          "matched_run_advantage":cfg["relative_matchup_rush_advantage"],
          "strength_home":cfg["home_two_way_strength"],
          "strength_away":cfg["away_two_way_strength"],
          "clock_verified_affected":bool(r1["coupled_rush_chain_empirical_clock_compounding"]),
          "clock_not_verified":bool(r1["coupled_rush_chain_clock_unverified_fallback"]),
          "rush_control_conditional_tail_home":r1["scenario_distributions"].get("home_rushing_control_tail"),
          "rush_control_conditional_tail_away":r1["scenario_distributions"].get("away_rushing_control_tail"),
          "mean_abs_conditional_ppd_shift":r1["coupled_rush_chain_mean_abs_ppd_shift_home"],
          "seed":int(g['game_id'])%(2**32),
          "paired_runs_per_side":5000})
        if (index+1)%30==0:
            print("PAIR_PROGRESS",year,index+1,len(records),flush=True)
    def summary(items):
        if not items:return {"games":0}
        errs0=np.array([x["absolute_error_control"] for x in items])
        errs1=np.array([x["absolute_error_coupled"] for x in items])
        actual=np.array([x["actual_home_margin"] for x in items])
        old=np.array([x["margin_control"] for x in items])
        new=np.array([x["margin_coupled"] for x in items])
        diff=errs0-errs1
        return {"games":len(items),
          "margin_MAE_reference":round(float(errs0.mean()),5),
          "margin_MAE_coupled":round(float(errs1.mean()),5),
          "MAE_improvement_positive_is_better":round(float(diff.mean()),5),
          "RMSE_reference":round(float(np.sqrt(np.mean((actual-old)**2))),5),
          "RMSE_coupled":round(float(np.sqrt(np.mean((actual-new)**2))),5),
          "mean_abs_margin_change":round(float(np.mean(np.abs(new-old))),5),
          "paired_margin_change_gt_0_001":int((np.abs(new-old)>.001).sum()),
          "empirically_verified_clock_games":sum(int(x["clock_verified_affected"]) for x in items)}
    gains=np.array([x['absolute_error_control']-x['absolute_error_coupled'] for x in records])
    if len(gains):
        rng=np.random.default_rng(801123+year)
        draws=rng.choice(gains,(2500,len(gains)),replace=True).mean(axis=1)
        ci=np.percentile(draws,[2.5,97.5]).round(5).tolist()
    else:ci=None
    ranked=sorted(records,key=lambda r:abs(r["matched_run_advantage"]),reverse=True)
    n=max(1,len(ranked)//4) if ranked else 0
    out={"schema":"cfb_coupled_rush_possession_pair_5000_v1",
         "season":year,"snapshot_capture_at":manifest.get("captured_at_utc"),
         "snapshot_row_publication_asof_verified":False,
         "training_rush_SOS_2024_frozen_in_both_arms":True,
         "historical_ATS_accuracy":None,"verified_historical_spreads":0,
         "target_game_outcomes_used_in_pregame_features":False,
         "motivating_game_excluded_from_comparison":True,
         "candidate_status":"ACTIVE_EXPERIMENTAL_BUILD16_NOT_ATS_VALIDATED",
         "screen":screen,"insufficient_or_excluded":dict(unavailable),
         "all":summary(records),
         "strongest_rushing_mismatch_quartile":summary(ranked[:n]),
         "MAE_delta_bootstrap_95":ci,"game_records":records}
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",required=True);p.add_argument("--year",type=int,required=True)
    p.add_argument("--max-games",type=int,default=120);p.add_argument("--out",required=True)
    args=p.parse_args()
    x=evaluate(args.snapshot,args.max_games)
    if args.year!=x["season"]:raise ValueError("Requested different source season")
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(x,sort_keys=True,indent=2))
    print("COUPLED_RUSH_PAIRED_5000_ACTUALS")
    print(json.dumps({k:v for k,v in x.items() if k!='game_records'},sort_keys=True,indent=2))
if __name__=="__main__":main()
