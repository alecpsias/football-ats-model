"""Source-backed Build16 red-zone finishing and distance-regressed FG outcomes.

Uses only previously completed PBP stored in historical records. This module
does not infer attempt distance from a scoring play, use target-game outcomes,
or pretend a 4/4 kicker has 100% future accuracy. Prior coefficients remain
explicit hypotheses, not a claim of ATS calibration.
"""
from __future__ import annotations

import math
from collections import defaultdict

VERSION = "cfb-build16-rz-distance-fg-v1"
RZ_PRIOR = .56
RZ_OFF_PRIOR_N = 20.
RZ_DEF_PRIOR_N = 24.
FG_TEAM_PRIOR_N = 16.
FG_LEAGUE_PRIOR_N = 40.
BINS = ((15, 29, 24.), (30, 39, 35.), (40, 49, 45.),
        (50, 59, 55.), (60, 75, 65.))
DEFAULT_MIX = (.18, .38, .27, .14, .03)


def _clip(x, low, high):
    return max(low, min(high, float(x)))


def _prior_make(distance):
    # Conservative, monotone 2026 research prior; replaced progressively by
    # verified earlier-game league observations when available.
    return 1. / (1. + math.exp(-(2.1 - .067*(distance-30.))))


def _records(histories):
    return [r for rows in histories.values() for r in rows if isinstance(r, dict)]


def _kicks(rows):
    result = []
    for r in rows:
        special = r.get("special_teams") or {}
        events = special.get("fg_distance_events")
        if not isinstance(events, list):
            continue
        for item in events:
            if (not isinstance(item, (list, tuple)) or len(item)!=2
                    or type(item[0]) not in (int, float)
                    or type(item[1]) is not int
                    or not math.isfinite(item[0])
                    or not 15 <= item[0] <= 75
                    or item[1] not in (0, 1)):
                raise ValueError("Invalid verified field goal distance/outcome")
            result.append((float(item[0]), int(item[1])))
    return result


def _bucket(distance):
    for idx, (low, high, _) in enumerate(BINS):
        if low <= distance <= high:
            return idx
    raise ValueError("Unrecognized field goal distance")


def _redzone(rows, yes, trials):
    made = opportunities = 0
    for r in rows:
        a,b=r.get(yes),r.get(trials)
        if type(a) is not int or type(b) is not int or a<0 or b<0 or a>b:
            continue
        made+=a;opportunities+=b
    return made,opportunities


def matchup(team_id, opponent_id, histories):
    """Return pregame posterior probabilities and bounded PPD residuals.

    Red-zone touchdown probability is conditional on a *red-zone trip*. A
    bounded residual maps it to the existing drive-level TD/FG pathways;
    Build16 does not pretend to reconstruct exact in-drive field movement.
    """
    offense=histories.get(team_id,[])
    defense=histories.get(opponent_id,[])
    off_yes,off_n=_redzone(offense,"redzone_tds","redzone_trips")
    def_yes,def_n=_redzone(defense,"def_redzone_tds","def_redzone_trips")
    rz=None
    rz_ppd_delta=0.
    rz_td_share_delta=0.
    if len(offense)>=3 and len(defense)>=3 and off_n>=6 and def_n>=6:
        own=(off_yes+RZ_OFF_PRIOR_N*RZ_PRIOR)/(off_n+RZ_OFF_PRIOR_N)
        allowed=(def_yes+RZ_DEF_PRIOR_N*RZ_PRIOR)/(def_n+RZ_DEF_PRIOR_N)
        matched=_clip(own+.5*(allowed-RZ_PRIOR),.2,.85)
        improvement=matched-own
        drives=sum(r.get("drives",0) for r in offense)
        trips_per_drive=_clip(off_n/max(1.,drives),0.,.85)
        rz_ppd_delta=_clip(improvement*trips_per_drive*3.965,-.18,.18)
        rz_td_share_delta=_clip(improvement*.28,-.045,.045)
        rz={"status":"REGRESSED_OPPONENT_MATCHED",
            "historical_offense_trips":off_n,"historical_defense_trips":def_n,
            "offense_td_probability_shrunk":round(own,6),
            "opponent_td_allowed_probability_shrunk":round(allowed,6),
            "matched_redzone_td_probability":round(matched,6),
            "prior_probability":RZ_PRIOR,
            "expected_trips_per_drive":round(trips_per_drive,6)}
    team_kicks=_kicks(offense)
    league_kicks=_kicks(_records(histories))
    fg=None
    fg_ppd_delta=0.
    if len(offense)>=3 and len(team_kicks)>=2:
        league_bin=defaultdict(list); team_bin=defaultdict(list)
        for d,made in league_kicks:
            league_bin[_bucket(d)].append((d,made))
        for d,made in team_kicks:
            team_bin[_bucket(d)].append((d,made))
        total_league=len(league_kicks)
        bins=[]
        for idx,(_,_,center) in enumerate(BINS):
            pool=league_bin[idx]
            own=team_bin[idx]
            league_p=(sum(m for _,m in pool)+FG_LEAGUE_PRIOR_N*_prior_make(center))/(
                len(pool)+FG_LEAGUE_PRIOR_N)
            team_p=(sum(m for _,m in own)+FG_TEAM_PRIOR_N*league_p)/(
                len(own)+FG_TEAM_PRIOR_N)
            distance=(sum(d for d,_ in own)/len(own) if own
                      else sum(d for d,_ in pool)/len(pool) if pool else center)
            # Move within bucket in the direction of the actual kick distance.
            odds=math.log(team_p/(1.-team_p))-.067*(distance-center)
            p=_clip(1./(1.+math.exp(-odds)),.05,.99)
            league_mix=(len(pool)/total_league if total_league>=20 else DEFAULT_MIX[idx])
            weight=(len(own)+10.*league_mix)/(len(team_kicks)+10.)
            bins.append({"distance":round(distance,3),"probability":round(p,8),
                         "weight":round(weight,8),
                         "observed_attempts":len(own),
                         "league_prior_attempts":len(pool)})
        # Normalize rounded weights to avoid numerical drift.
        total_weight=sum(b["weight"] for b in bins)
        for b in bins:
            b["weight"]=b["weight"]/total_weight
        predicted=sum(b["probability"]*b["weight"] for b in bins)
        raw_made=sum(m for _,m in team_kicks)
        raw_pct=raw_made/len(team_kicks)
        drives=sum(r.get("drives",0) for r in offense)
        attempts_per_drive=_clip(len(team_kicks)/max(1.,drives),0.,.65)
        # Historical PPD already embeds those raw FG makes/misses. Correct
        # only the *regression departure*, never add expected FG points again.
        fg_ppd_delta=_clip(3.*(predicted-raw_pct)*attempts_per_drive,-.18,.18)
        fg={"status":"REGRESSED_DISTANCE_CONDITIONED",
            "team_attempts":len(team_kicks),"team_makes":raw_made,
            "raw_make_rate":round(raw_pct,6),
            "predicted_make_rate":round(predicted,6),
            "attempts_per_drive":round(attempts_per_drive,6),
            "distance_bins":bins,
            "league_prior_attempts":len(league_kicks),
            "fallback_prior_is_hypothesis":True}
    return {"version":VERSION,
        "status":("ACTIVE_VERIFIED_EVIDENCE" if rz is not None or fg is not None
                  else "UNAVAILABLE_LEAVE_EXISTING_MODEL_UNCHANGED"),
        "redzone":rz,"field_goals":fg,
        "redzone_ppd_delta":round(rz_ppd_delta,6),
        "fg_regression_ppd_delta":round(fg_ppd_delta,6),
        "delta_ppd":round(rz_ppd_delta+fg_ppd_delta,6),
        "td_share_delta":round(rz_td_share_delta,6),
        "nonoffensive_scoring_touched":False,
        "coefficients_empirically_fitted":False}


def draw_missed_field_goals(points, fg_evidence, rng):
    """Adjudicate ONLY 3-point offensive FG attempt pathways.

    Sampling an attempt distance from previously observed distance mix, then
    drawing its outcome, is disjoint from turnover, TD, and return pathways.
    """
    import numpy as np
    if fg_evidence is None:
        return points, np.zeros(len(points),dtype=bool)
    bins=fg_evidence["distance_bins"]
    weights=np.array([b["weight"] for b in bins],dtype=float)
    probs=np.array([b["probability"] for b in bins],dtype=float)
    index=np.flatnonzero(points==3)
    missed=np.zeros(len(points),dtype=bool)
    if len(index):
        chosen=rng.choice(len(bins),size=len(index),p=weights)
        misses=rng.random(len(index)) >= probs[chosen]
        missed[index[misses]]=True
    return np.where(missed,0,points).astype(points.dtype),missed


def gross_up_for_misses(projection,fg_evidence):
    """Normalize gross FG-attempt pathways to net intended offensive PPD."""
    if fg_evidence is None:
        return dict(projection)
    p=dict(projection)
    td=float(p["_td"])
    made=fg_evidence["predicted_make_rate"]
    gross=td*6.965+(1.-td)*3.
    net=td*6.965+(1.-td)*3.*made
    p["_ppd"]=min(6.5,p["_ppd"]*gross/net)
    return p
