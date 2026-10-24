"""Paired, chronological, same-seed 5000-draw Build16 Improvement 1 replay.

The original unified Build16 with the new mechanism OFF is the reference;
exactly one new candidate with it ON. No baseline modification, no synthetic
spreads, no same-game or future-game statistics in the inputs. Retrospective
provider capture dates are not original immutable pre-kickoff archives.
"""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from cfb.failure2_paired import cohort
from cfb.failure1_learned_shadow import correction as fitted_rush
from cfb.mechanism_build16 import simulate_build16
from cfb.pipeline import verify

# The motivating Oct 23 games must not be credited as independent validation.
MOTIVATING_IDS={"401858271","401862808","401864525","401858272"}

def evaluate(snapshot,max_games=80):
    m=verify(Path(snapshot))
    year=int(m["season"])
    if year not in (2025,2026):raise ValueError("Unsupported season")
    fixtures,screening=cohort(snapshot,max_games,enrich_rush=True)
    matched=[];missing=Counter()
    for idx,g in enumerate(fixtures):
        if g["game_id"] in MOTIVATING_IDS:
            missing["MOTIVATING_GAME_EXCLUDED"]+=1
            continue
        h,a=g["home_id"],g["away_id"]
        history=g["historical"]
        seed=int(g["game_id"])%(2**32)
        common=dict(runs=5000,seed=seed,
            failure1_fit_margin_delta=fitted_rush(h,a,history)["margin_delta_points"],
            continuous_rush_sos_enabled=True)
        old=simulate_build16(h,a,history,g["neutral"],two_way_venue_enabled=False,**common)
        new=simulate_build16(h,a,history,g["neutral"],two_way_venue_enabled=True,**common)
        prior=old.get("simulation");next_=new.get("simulation")
        if prior is None or next_ is None:
            missing["UNAVAILABLE_PROJECTED_MATCHUP"]+=1
            continue
        if prior["simulations"]!=5000 or next_["simulations"]!=5000:
            raise RuntimeError("Expected same-seed, 5000-draw comparisons")
        evidence=new["two_way_venue_improvement1"]
        if old["two_way_venue_improvement1"]["status"]!="DISABLED_FOR_PAIRED_REFERENCE":
            raise RuntimeError("Improvement 1 applied in reference")
        matched.append({
            "game_id":g["game_id"],"kickoff":g["kickoff"],"seed":seed,
            "actual_margin":float(g["actual_margin"]),
            "actual_total":float(g["actual_total"]),
            "margin_before":prior["mean_margin_home"],
            "margin_after":next_["mean_margin_home"],
            "total_before":prior["mean_total"],
            "total_after":next_["mean_total"],
            "prob_before":prior["home_win_probability_raw"],
            "prob_after":next_["home_win_probability_raw"],
            "venue_active":evidence["venue_component_active"],
            "two_way_route_active":evidence["opponent_counterattack_active"],
            "home_venue_ppd":evidence["venue_delta_ppd_home"],
            "away_venue_ppd":evidence["venue_delta_ppd_away"],
            "two_way_margin_route_ppd":evidence["two_way_margin_route_ppd"]})
        if (idx+1)%20==0:print("TWO_WAY_VENUE_PROGRESS",year,idx+1,len(matched),flush=True)
    def metrics(rows):
        if not rows:return {"games":0}
        actual=np.array([z["actual_margin"] for z in rows],float)
        b=np.array([z["margin_before"] for z in rows],float)
        c=np.array([z["margin_after"] for z in rows],float)
        totals=np.array([z["actual_total"] for z in rows],float)
        tb=np.array([z["total_before"] for z in rows],float)
        tc=np.array([z["total_after"] for z in rows],float)
        winners=(actual>0).astype(float)
        pb=np.array([z["prob_before"] for z in rows],float)
        pc=np.array([z["prob_after"] for z in rows],float)
        delta=np.abs(actual-b)-np.abs(actual-c)
        boot=np.random.default_rng(20261024+year).choice(delta,(1500,len(delta)),replace=True).mean(axis=1)
        return {"games":len(rows),
          "margin_MAE_previous":round(float(np.mean(np.abs(actual-b))),4),
          "margin_MAE_improvement1":round(float(np.mean(np.abs(actual-c))),4),
          "margin_MAE_gain_positive_improved":round(float(delta.mean()),4),
          "bootstrap_95_gain":np.percentile(boot,[2.5,97.5]).round(4).tolist(),
          "total_MAE_previous":round(float(np.mean(np.abs(totals-tb))),4),
          "total_MAE_improvement1":round(float(np.mean(np.abs(totals-tc))),4),
          "winner_Brier_previous":round(float(np.mean((pb-winners)**2)),5),
          "winner_Brier_improvement1":round(float(np.mean((pc-winners)**2)),5)}
    active_venue=[z for z in matched if z["venue_active"]]
    changed=[z for z in matched if abs(z["margin_after"]-z["margin_before"])>.01]
    return {"schema":"cfb_build16_two_way_venue_paired_v1",
        "year":year,"snapshot_capture":m["captured_at_utc"],
        "retrospective_snapshot_asof_publication_not_verified":True,
        "motivating_games_excluded":sorted(MOTIVATING_IDS),
        "baseline_unmodified":True,"one_challenger":True,
        "same_seed_simulations_each":5000,"verified_historical_spreads":0,
        "historical_ATS_accuracy":None,"requested_games":max_games,
        "screen":screening,"ineligible":dict(missing),
        "overall_metrics":metrics(matched),
        "venue_active_metrics":metrics(active_venue),
        "venue_active_games":len(active_venue),
        "two_way_route_active_games":sum(z["two_way_route_active"] for z in matched),
        "changed_margin_games":len(changed),
        "games":matched}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",required=True)
    p.add_argument("--season",type=int,required=True)
    p.add_argument("--max-games",type=int,default=80)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    r=evaluate(args.snapshot,args.max_games)
    if r["year"]!=args.season:raise ValueError("Wrong source season")
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(r,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k!="games"},indent=2))

if __name__=="__main__":main()
