"""CFB ATS spread-concordance check — an explicit risk/decision rule, not a score fit.

2026-10-23 NC State @ Stanford revealed that a one-sided challenger edge
can arise when the frozen baseline has essentially zero margin versus the
observed line. Treat that disagreement as a reason to HOLD, not a calibrated
cover probability. No invented sportsbook quotes, and no result inputs.
"""
from __future__ import annotations

import math

VERSION = "cfb-ats-baseline-market-concordance-v1"
MIN_ACTIONABLE_MARGIN_BUFFER = 2.0
BASELINE_ZERO_EDGE_BAND = 1.0
MEAN_MODEL_DISCORDANCE = 2.0


def assess(baseline_margin_home, challenger_margin_home, home_spread=None):
    """Compare two independently frozen model margins to a VERIFIED market line.

    With no verified as-of quote the output is a diagnostic ONLY. Signs:
      projected margin home + home spread > 0 => the home side covers.
    A line-matched baseline does NOT establish an ATS advantage for the
    challenger just because the challenger predicts a larger margin.
    """
    for name, value in (("baseline", baseline_margin_home),
                        ("challenger", challenger_margin_home)):
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("Nonfinite or unavailable "+name+" projection")
    b = float(baseline_margin_home)
    c = float(challenger_margin_home)
    gap = c-b
    result = {"version": VERSION, "status": None, "calibrated_ats_probability": False,
              "baseline_mean_home_margin": round(b, 5),
              "challenger_mean_home_margin": round(c, 5),
              "challenger_minus_baseline_margin": round(gap, 5),
              "material_model_disagreement": abs(gap) >= MEAN_MODEL_DISCORDANCE,
              "baseline_edge_vs_spread_points": None,
              "challenger_edge_vs_spread_points": None,
              "native_pre_kickoff_quote_required": True,
              "automatic_bet_recommendation": False}
    if home_spread is None:
        result["status"] = ("HOLD_NO_VERIFIED_QUOTE_AND_MATERIAL_MODEL_GAP"
                            if result["material_model_disagreement"]
                            else "HOLD_NO_VERIFIED_QUOTE")
        return result
    if type(home_spread) not in (int, float) or not math.isfinite(home_spread) or abs(home_spread)>75:
        raise ValueError("Invalid verified home spread")
    edge_b, edge_c = b+float(home_spread), c+float(home_spread)
    result.update({"baseline_edge_vs_spread_points": round(edge_b, 5),
                   "challenger_edge_vs_spread_points": round(edge_c, 5),
                   "baseline_model_has_near_zero_market_edge": abs(edge_b)<=BASELINE_ZERO_EDGE_BAND,
                   "model_spread_side_conflict": edge_b*edge_c<0})
    if edge_b*edge_c<0 and abs(edge_c)>=1.0:
        result["status"] = "HOLD_CONFLICTING_BASELINE_CHALLENGER_SPREAD_SIDES"
    elif abs(edge_b)<=BASELINE_ZERO_EDGE_BAND and abs(edge_c)>=1.0:
        result["status"] = "HOLD_CHALLENGER_ONLY_EDGE_BASELINE_AT_MARKET"
    elif abs(edge_c)<MIN_ACTIONABLE_MARGIN_BUFFER:
        result["status"] = "HOLD_SMALL_CHALLENGER_MARGIN_BUFFER"
    else:
        result["status"] = "MODELS_NOT_DISCORDANT_NO_CALIBRATED_BET"
    return result
