"""Build16 improvement 1: two-way pass/rush/QB counterattack + venue residual.

All observations come from prior completed games, as filtered by the caller.
Current model ALREADY accounts for opponent-adjusted pass/rush EPA and the
league-wide 2.4 point HFA. This module only models bounded INCREMENTAL:
  * adaptation of play selection to relative matchup strengths, considering
    historical QB rushing as an alternative to ordinary rushing;
  * team-specific home/away efficiency differences, after subtracting each
    historical opponent's performance against OTHER teams and the league HFA.
Missing data fails closed; no fabricated opponent quality, personnel status,
game outcomes, odds, or extra home bonus. Constants are UNFITTED sensitivity
parameters, not ATS-calibrated coefficients or an established accuracy gain.
"""
from __future__ import annotations
import math
from cfb.engine import CFB_RULES
from cfb.integrated_scoring import _weight_and_match

VERSION="cfb-build16-two-way-venue-v1"
MAX_PASS_MIX_SHIFT=.06
MAX_ROUTING_PPD=.12
MAX_VENUE_PPD=.20
MIN_VENUE_GAMES=2
MIN_VENUE_KIND_PLAYS=12
MIN_VENUE_TOTAL_PLAYS=45
VENUE_PRIOR_PLAYS=180.
# Existing +2.4 margin HFA is split +/-1.2 scoring points. Dividing by
# 11 drives * 6.1 plays estimates the EPA/attempt main effect to REMOVE.
LEAGUE_VENUE_EPA=CFB_RULES["home_advantage_points"]/(2.*11.*6.1)

def _finite(x):
    return type(x) in (int,float) and math.isfinite(x)

def _clamp(x,lo,hi):
    return max(lo,min(hi,x))

def _rate(rows,prefix,kind):
    n=sum(r[prefix+"_"+kind+"_n"] for r in rows
          if _finite(r.get(prefix+"_"+kind+"_n")) and
          _finite(r.get(prefix+"_"+kind+"_sum")) and r[prefix+"_"+kind+"_n"]>0)
    s=sum(r[prefix+"_"+kind+"_sum"] for r in rows
          if _finite(r.get(prefix+"_"+kind+"_n")) and
          _finite(r.get(prefix+"_"+kind+"_sum")) and r[prefix+"_"+kind+"_n"]>0)
    return s/n if n else None

def _venue_feature(team_id,histories,role):
    """Role=off uses own EPA; role=def uses EPA ALLOWED.

    Subtract opponent defense/offense against third parties. For each school,
    compare only source-complete home and away games; a league-expected
    location main effect is removed before computing home-minus-away.
    Returns a signed EPA residual centered across this school's venues.
    """
    if role not in ("off","def"):raise ValueError("Invalid venue role")
    rows=histories.get(team_id,())
    buckets={}
    for venue in ("home","away"):
        valid=[]
        for r in rows:
            if r.get("venue")!=venue:continue
            opp=r.get("opponent_id")
            if type(opp) is not int or opp==team_id:continue
            # Never use an opponent's game AGAINST this school to determine
            # that opponent's independent strength.
            peers=[x for x in histories.get(opp,()) if x.get("opponent_id")!=team_id]
            other="def" if role=="off" else "off"
            quality={k:_rate(peers,other,k) for k in ("pass","rush")}
            if any(v is None for v in quality.values()):continue
            item={}
            for kind in ("pass","rush"):
                n=r.get(role+"_"+kind+"_n");s=r.get(role+"_"+kind+"_sum")
                if not (_finite(n) and _finite(s) and n>0):
                    item=None;break
                item[kind]=(float(s)-float(n)*quality[kind],float(n))
            if item is not None:valid.append((opp,item))
        if len(valid)<MIN_VENUE_GAMES or len({opp for opp,_ in valid})<2:
            return {"status":"MISSING_INDEPENDENT_OPPONENTS_OR_VENUES",
                    "role":role,"missing_venue":venue}
        sums={k:sum(x[k][0] for _,x in valid) for k in ("pass","rush")}
        counts={k:sum(x[k][1] for _,x in valid) for k in ("pass","rush")}
        if (min(counts.values())<MIN_VENUE_KIND_PLAYS or
            sum(counts.values())<MIN_VENUE_TOTAL_PLAYS):
            return {"status":"INSUFFICIENT_VENUE_PLAY_SAMPLE",
                    "role":role,"missing_venue":venue}
        # Stable play weights across venues prevent pass-heavy HOME games
        # from masquerading as an improvement in home EPA.
        buckets[venue]={"per_kind":{k:sums[k]/counts[k] for k in sums},
                        "counts":counts,"games":len(valid)}
    totals={k:buckets["home"]["counts"][k]+buckets["away"]["counts"][k]
            for k in ("pass","rush")}
    share=totals["pass"]/(totals["pass"]+totals["rush"])
    for venue in ("home","away"):
        per=buckets[venue]["per_kind"]
        value=share*per["pass"]+(1.-share)*per["rush"]
        # Offense scores more at home; defense allows fewer points at home.
        league=LEAGUE_VENUE_EPA*(1 if venue=="home" else -1)
        if role=="def":league=-league
        buckets[venue]["adjusted"]=value-league
    effective=min(sum(buckets[v]["counts"].values()) for v in ("home","away"))
    shrink=effective/(effective+VENUE_PRIOR_PLAYS)
    half_diff=.5*(buckets["home"]["adjusted"]-buckets["away"]["adjusted"])*shrink
    if not math.isfinite(half_diff):raise ValueError("Invalid venue residual")
    return {"status":"SOURCE_ADJUSTED_VENUE_CONTRAST","role":role,
            "half_home_minus_away_epa":round(half_diff,8),
            "effective_plays":round(effective,1),
            "shrinkage":round(shrink,6),
            "home_games":buckets["home"]["games"],
            "away_games":buckets["away"]["games"],
            "league_home_field_removed":True,
            "opponent_quality_from_other_teams":True}

def _counterattack(team_id,opp_id,histories):
    """Use both opponent-adjusted route strengths, not rushing alone."""
    p=_weight_and_match(team_id,opp_id,histories,"pass")
    r=_weight_and_match(team_id,opp_id,histories,"rush")
    if p is None or r is None:
        return {"status":"MISSING_OPPONENT_ADJUSTED_ROUTES"}
    rows=histories.get(team_id,())
    passed=sum(x.get("off_pass_n") or 0 for x in rows)
    rushed=sum(x.get("off_rush_n") or 0 for x in rows)
    drives=sum(x.get("drives") or 0 for x in rows)
    if min(passed,rushed,drives)<=0:
        return {"status":"INSUFFICIENT_OFFENSIVE_USAGE"}
    pass_mix=passed/(passed+rushed)
    plays=_clamp((passed+rushed)/drives,3.5,10.5)
    pass_epa=p["matched_epa_shrunk"]
    rush_epa=r["matched_epa_shrunk"]
    # Historical QB rushing changes the possible play-selection response,
    # NOT the already-counted base rushing EPA / player availability.
    eligible=[x for x in rows if _finite(x.get("qb_rushes"))
              and _finite(x.get("qb_rush_epa")) and x["qb_rushes"]>0]
    qb_n=sum(x["qb_rushes"] for x in eligible)
    qb_supported=(len(eligible)>=2 and qb_n>=12)
    qb_shift=0.
    if qb_supported:
        qb_epa=sum(x["qb_rushes"]*x["qb_rush_epa"] for x in eligible)/qb_n
        own_rush=_rate(rows,"off","rush")
        if own_rush is not None:
            qb_excess=max(0.,qb_epa-own_rush)
            qb_fraction=_clamp(qb_n/rushed,0.,1.)
            # A bounded alternative route, only if source-rushing QB has a
            # positive excess over ordinary team rushing AND the opponent's
            # matched rushing matchup is competitive.
            qb_shift=min(.025,.035*qb_fraction*math.tanh(qb_excess/.18))
    diff=pass_epa-(rush_epa+qb_shift)
    shift=_clamp(MAX_PASS_MIX_SHIFT*math.tanh(diff/.20),
                 -MAX_PASS_MIX_SHIFT,MAX_PASS_MIX_SHIFT)
    shift=_clamp(shift,-pass_mix,1.-pass_mix)
    # An adaptive route selection advantage is incremental, and further
    # shrunk to avoid repeating the main pass/rush EPA valuation.
    reliable=min(p["epa_level_weight"],r["epa_level_weight"])
    gain=_clamp(plays*shift*diff*.45*reliable,0.,MAX_ROUTING_PPD)
    return {"status":"SOURCE_OPPONENT_MATCHED_TWO_WAY_ROUTES",
            "matched_pass_epa":round(pass_epa,7),
            "matched_rush_epa":round(rush_epa,7),
            "qb_route_increment_epa":round(qb_shift,7),
            "qb_usage_games_qualified":len(eligible) if qb_supported else 0,
            "qb_rushing_used_as_historical_route_not_personnel":qb_supported,
            "historical_pass_share":round(pass_mix,6),
            "adaptive_pass_mix_change":round(shift,6),
            "gain_ppd_before_two_team_centering":round(gain,8),
            "main_epa_counted_only_by_existing_integrated_scorer":True}

def matchup(home_id,away_id,histories,neutral):
    """Return incremental PPD for each side, no total rerating.

    * Both independent pass/rush counterattacks compete for a ZERO-SUM
      residual margin effect; it cannot simply boost both offenses.
    * Venue contrasts are conditional on enough home/away third-party data,
      and cannot duplicate existing league-average HFA.
    """
    if home_id==away_id or type(neutral) is not bool:
        raise ValueError("Two teams and explicit neutral flag required")
    home=_counterattack(home_id,away_id,histories)
    away=_counterattack(away_id,home_id,histories)
    route=0.
    if (home["status"]=="SOURCE_OPPONENT_MATCHED_TWO_WAY_ROUTES" and
        away["status"]=="SOURCE_OPPONENT_MATCHED_TWO_WAY_ROUTES"):
        route=.5*(home["gain_ppd_before_two_team_centering"]-
                  away["gain_ppd_before_two_team_centering"])
    venue_evidence={}
    venue_home=venue_away=0.
    if not neutral:
        for label,team,role in (("home_off",home_id,"off"),
                                ("home_def",home_id,"def"),
                                ("away_off",away_id,"off"),
                                ("away_def",away_id,"def")):
            venue_evidence[label]=_venue_feature(team,histories,role)
        if all(x["status"]=="SOURCE_ADJUSTED_VENUE_CONTRAST"
               for x in venue_evidence.values()):
            # A defense residual measures EPA ALLOWED; stronger HOME defense
            # suppresses the AWAY offense (negative sign).
            hv=venue_evidence["home_off"]["half_home_minus_away_epa"]
            av=venue_evidence["away_off"]["half_home_minus_away_epa"]
            hd=venue_evidence["home_def"]["half_home_minus_away_epa"]
            ad=venue_evidence["away_def"]["half_home_minus_away_epa"]
            # Defense away residual equals -its home half difference.
            # Half weighting between own offense / opposing defense prevents
            # adding two full copies of the same observed EPA change.
            venue_home=_clamp(6.1*.5*(hv-ad),-MAX_VENUE_PPD,MAX_VENUE_PPD)
            venue_away=_clamp(6.1*.5*(-av+hd),-MAX_VENUE_PPD,MAX_VENUE_PPD)
    else:
        venue_evidence={"status":"NEUTRAL_SITE_NO_TEAM_VENUE_ADJUSTMENT"}
    total_home=round(route+venue_home,8)
    total_away=round(-route+venue_away,8)
    return {"status":"ACTIVE_SOURCE_INCREMENTAL_TWO_WAY_VENUE",
            "version":VERSION,"home_route":home,"away_route":away,
            "opponent_counterattack_active":bool(route),
            "two_way_margin_route_ppd":round(route,8),
            "venue":venue_evidence,
            "venue_component_active":bool(venue_home or venue_away),
            "venue_delta_ppd_home":round(venue_home,8),
            "venue_delta_ppd_away":round(venue_away,8),
            "delta_ppd_home":total_home,"delta_ppd_away":total_away,
            "uncalibrated_sensitivity":True,
            "numerical_personnel_model_reactivated":False,
            "league_hfa_not_double_counted":True,
            "baseline_untouched":True}
