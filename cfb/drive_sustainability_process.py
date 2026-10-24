"""Failure 3 alternative scoring routes: sustained chains vs isolated explosives.

No extra points are credited for a long play. Historical scoring PPD and TD
share are decomposed into DISJOINT scoring pathways. A common game-level
chain-effectiveness draw makes repeated productive possessions correlated,
while isolated explosive touchdowns remain possible even in weak-chain games.
The per-possession duration response is a bounded UNFITTED sensitivity,
not a claim of an independently calibrated clock model.
"""
from __future__ import annotations
import math
import numpy as np

VERSION="cfb-failure3-possession-mixture-v1"
MAX_EXTRA_CYCLES=2

def parameters(evidence):
    """Fail closed without observed, opponent-adjusted prior-game evidence."""
    if not isinstance(evidence,dict) or evidence.get("status")!="ACTIVE_BUILD16_UNCALIBRATED_FAILURE3":
        return None
    out={}
    for side in ("home","away"):
        e=evidence.get(side,{})
        if e.get("status")!="AVAILABLE_BOUNDED_UNFITTED":
            return None
        def checked(key,lower,upper):
            x=e.get(key)
            if type(x) not in (int,float) or not math.isfinite(x) or not lower<=x<=upper:
                raise ValueError("Unverified Failure 3 historical input: "+key)
            return float(x)
        chain=checked("chain_share_matched",0.,1.)
        explosive=checked("short_explosive_td_share_shrunk" if
                          "short_explosive_td_share_shrunk" in e
                          else "short_explosive_pass_td_share_shrunk",0.,1.)
        drives=checked("matched_drives",1.,10000.)
        out[side]={"chain_probability":chain,
                   "explosive_td_share":explosive,
                   "matched_drives":drives,
                   # Wider game-to-game sustain volatility for limited chain
                   # success, but shrink correlated effect with more samples.
                   "chain_concentration":float(np.clip(
                       9.+.16*drives,12.,45.))}
    return out


def chain_draws(rate,rng,n):
    """One correlated chain-capability state PER SIMULATED GAME.

    Centered around observed matched historical success; no fixed deduction
    from opponent-adjusted PPD. Independent from explosive scoring route.
    """
    chain=rate["chain_probability"]
    k=rate["chain_concentration"]
    # Keep cases with observed 0/1 rates finite; tiny support is explicit.
    expected=np.clip(chain,.01,.99)
    actual=rng.beta(expected*k,(1.-expected)*k,size=n)
    multiplier=1.+1.15*(actual-expected)/max(float(expected),.18)
    return np.clip(multiplier,.40,1.60)


def draw_possession(projection, rng, propensity, extra_ppd, short_field, active,
                    process, chain_multiplier, rush_chain_extra_ppd=0.0):
    """Generate mutually exclusive turnover, explosive TD, chain score, stall.

    Base PPD is preserved in expectation before bounds and game-level variance:
    explosive + chain scoring probabilities sum to the original scoring
    probability when chain_multiplier=1. Never add a separate explosive TD.
    """
    n=len(active)
    zeros=np.zeros(n,dtype=bool)
    if not np.any(active):
        return np.zeros(n,dtype=np.int16),zeros,zeros,zeros
    p=projection
    avg=p["_td"]*6.965+(1.-p["_td"])*3.
    ppd=np.clip(p["_ppd"]+extra_ppd+short_field*1.1,.05,6.15)
    pscore=np.minimum(ppd/avg,1.-propensity)
    share=process["explosive_td_share"]
    base_explosive=pscore*p["_td"]*share
    base_chain=pscore-base_explosive
    # Improvement 2: a source-observed rushing-chain state changes ONLY the
    # sustainable possession-scoring route. Explosive-only TD probability is
    # unchanged. This avoids rewarding an isolated long pass for more rush
    # control and preserves turnover/return short-field exclusivity.
    nonexp_td_share=(p["_td"]*(1.-share))/(1.-p["_td"]*share)
    chain_points_per_score=3.+3.965*nonexp_td_share
    chain_shift=np.asarray(rush_chain_extra_ppd,dtype=float)/chain_points_per_score
    p_chain=np.clip(base_chain*chain_multiplier+chain_shift,
                    0.,np.maximum(0.,1.-propensity-base_explosive))
    draw=rng.random(n)
    turnover=active & (draw<propensity)
    explosive=active & (draw>=propensity) & (draw<propensity+base_explosive)
    chained=active & (draw>=propensity+base_explosive) & (
        draw<propensity+base_explosive+p_chain)
    # Remaining nonexplosive scoring still has the correct TD/FG mix.
    chain_td=chained & (rng.random(n)<nonexp_td_share)
    td=explosive | chain_td
    xp=(rng.random(n)<.965).astype(np.int16)
    points=np.where(td,6+xp,np.where(chained,3,0)).astype(np.int16)
    return points,turnover,explosive,chained


def duration_multiplier(points,turnover,explosive):
    """Transparent provisional play-clock response, not verified drive seconds.

    Rapid explosive touchdowns use fewer clock seconds; sustained chain
    scoring consumes more. The legacy scorer remains unchanged when off.
    """
    duration=np.full(len(points),.99,dtype=float)
    duration=np.where(turnover,.74,duration)
    duration=np.where((points>0)&~explosive,1.24,duration)
    duration=np.where(explosive,.57,duration)
    return duration
