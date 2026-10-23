"""Build16 improvements 2 + 4: source-observed coupled rush and chain scripts.

Conditional game-to-game rush-control states are bootstrapped from ACTUAL
prior-game offense and opponent-defense early-down rush success rates.
The changes are centered on the already-fitted EPA/points-per-drive baseline:
NO unconditional bonus for rushing twice. Opponent-specific vulnerability to
short, stalled drives modulates the shared rush-control shock, affecting both
teams' scoring opportunities together.

When Failure3 verified source clocks are active, changed mutually exclusive
drive outcomes lead to empirical duration draws and actual end-of-regulation
clock exhaustion. No undocumented durations are inserted if clocks are absent.
This dependency is a hypothesis needing paired multi-year validation, not an
ATS-calibrated causal coefficient.
"""
from __future__ import annotations
import math
import numpy as np
from cfb.sustainability import correction as sustainability_correction

VERSION="cfb-coupled-rush-drive-scripts-v1"
MIN_GAMES=3
MIN_EARLY_RUSH_ATTEMPTS=35
MIN_MATCHED_DRIVES=24
MAX_SCORING_FRACTION=.24
# Fixed experimental sensitivity; evaluate on historical margins, do not
# call it a learned coefficient or promote ATS confidence on its own.
HYPOTHETICAL_RESPONSE=1.0

def _history_rates(rows,prefix):
    num="mismatch_"+prefix+"early_rush_success_yes"
    den="mismatch_"+prefix+"early_rush_n"
    records=[]
    for row in rows:
        n,k=row.get(num),row.get(den)
        if type(n) not in (int,float) or type(k) not in (int,float):
            continue
        if not math.isfinite(n) or not math.isfinite(k) or not 0<=n<=k:
            return None
        if k>=5:records.append((float(n),float(k)))
    if len(records)<MIN_GAMES or sum(n for _,n in records)<MIN_EARLY_RUSH_ATTEMPTS:
        return None
    total=sum(n for _,n in records)
    mean=sum(x for x,_ in records)/total
    # Recenter empirical game-to-game outcomes using the weighted observed
    # historical rate; preserve heterogeneity without random fabricated time.
    return {"rates":[x/n for x,n in records],
            "counts":[int(n) for _,n in records],
            "rate":mean,"games":len(records),"trials":int(total)}

def _match(home_id,away_id,histories):
    match={}
    for side,team,opp in (("home",home_id,away_id),("away",away_id,home_id)):
        own=_history_rates(histories.get(team,()),"off_")
        allow=_history_rates(histories.get(opp,()),"def_")
        if own is None or allow is None:return None,{"status":"INSUFFICIENT_PRIOR_RUSH_CONTROL",
          "missing_side":side,"home_id":home_id,"away_id":away_id}
        match[side]={"offense":own,"opponent_defense":allow,
                     "matched_success":.5*(own["rate"]+allow["rate"])}
    return match,None

def _fragility(evidence,side):
    item=evidence.get(side,{})
    if item.get("status")!="AVAILABLE_BOUNDED_UNFITTED":return None
    drives=item.get("matched_drives")
    low=item.get("low_chain_non_td_matched")
    chain=item.get("chain_share_matched")
    if not all(type(x) in (float,int) and math.isfinite(x) for x in (drives,low,chain)):
        return None
    if drives<MIN_MATCHED_DRIVES or not 0<=low<=1 or not 0<=chain<=1:return None
    # Both are already opponent-matched observed rates, not a claim about the
    # upcoming game's offense. High value: frequent no-chain stalled drives.
    return .5*low+.5*(1.-chain)

def parameters(home_id,away_id,histories):
    """Use only genuinely prior histories (chronology enforced upstream)."""
    if home_id==away_id:raise ValueError("Rush control requires distinct teams")
    matches,error=_match(home_id,away_id,histories)
    if error:return error
    s=sustainability_correction(home_id,away_id,histories)
    weak_h=_fragility(s,"home")
    weak_a=_fragility(s,"away")
    if weak_h is None or weak_a is None:
        return {"status":"INSUFFICIENT_MATCHED_CHAIN_HISTORY",
                "rush_control_matched":True}
    mh=matches["home"]["matched_success"]
    ma=matches["away"]["matched_success"]
    # Relative advantage is continuously valued, never a binary severe gate.
    # A strong rushing advantage is leveraged when the other side is likely
    # to struggle to answer with repeatable offensive chains.
    h_interaction=max(0.,mh-ma)*weak_a
    a_interaction=max(0.,ma-mh)*weak_h
    def payload(side):
        o=matches[side]
        return {"matched_rush_early_success":round(o["matched_success"],7),
                "offense_history":o["offense"],
                "opponent_defense_history":o["opponent_defense"],
                "chain_fragility_opponent":round(weak_a if side=="home" else weak_h,7),
                "qualified_chain_drives":s[side]["matched_drives"]}
    return {"status":"ACTIVE_OBSERVED_COUPLED_RUSH_CHAIN_SCRIPTS",
            "version":VERSION,"home":payload("home"),"away":payload("away"),
            "home_two_way_strength":round(h_interaction,8),
            "away_two_way_strength":round(a_interaction,8),
            "relative_matchup_rush_advantage":round(mh-ma,8),
            "scoring_response_cap_fraction":MAX_SCORING_FRACTION,
            "hypothetical_response":HYPOTHETICAL_RESPONSE,
            "mean_scoring_baseline_preserved_in_expected_game_state":True,
            "scoring_is_joint_not_independent_bonus":True,
            "uses_unverified_clock_durations":False,
            "ATS_calibrated":False,"fitted":False}

def draw_coupled_state(config,rng,runs):
    """One correlated game-level state per run, centered around prior rates.

    The outcomes of both offenses move on the SAME latent matchup state,
    unlike independent score multipliers. No change to baseline home/away
    projected PPD averages unless nonlinear constraints actually bind.
    """
    if config.get("status")!="ACTIVE_OBSERVED_COUPLED_RUSH_CHAIN_SCRIPTS":
        raise ValueError("Missing verified matchup")
    def empirical(x):
        o=x["offense_history"];d=x["opponent_defense_history"]
        if min(len(o["rates"]),len(d["rates"]))<MIN_GAMES:
            raise ValueError("Insufficient empirical game source")
        # Independent choice from prior real-game rates, mixed equally from
        # offense success and opposing defense success allowed.
        off=rng.choice(np.asarray(o["rates"],float),size=runs)
        defense=rng.choice(np.asarray(d["rates"],float),size=runs)
        mean=.5*(np.mean(o["rates"])+np.mean(d["rates"]))
        return .5*(off+defense)-mean
    home=empirical(config["home"])
    away=empirical(config["away"])
    shared=home-away
    dominant_strength=(config["home_two_way_strength"]+
                       config["away_two_way_strength"])
    if not 0<=dominant_strength<=1:raise ValueError("Invalid control strength")
    # Equal/opposite shifts: one offense's chain control suppresses the
    # opponent's scoring opportunity in a coupled possession script.
    weight=min(MAX_SCORING_FRACTION,
        dominant_strength*HYPOTHETICAL_RESPONSE)
    # Widen/tilt GAME-TO-GAME OUTCOMES, not unconditional scoring means.
    delta=np.clip(shared*weight,-MAX_SCORING_FRACTION,MAX_SCORING_FRACTION)
    return {"home_ppd_fraction":delta,"away_ppd_fraction":-delta,
            "shared_rush_control_sample":shared,
            "absolute_matchup_response_weight":float(weight),
            "conditional_source_status":"EMPIRICAL_PREVIOUS_GAME_BOOTSTRAP",
            "fitted_coefficient":False}
