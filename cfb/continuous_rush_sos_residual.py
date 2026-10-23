"""ONE experimental, earlier-season-fitted Build16 margin residual.

Improvement 1: Continuous opponent-specific designed-rush advantages, positive
AND negative. Improvement 3: opponent-quality EPA margin residual after Build16
already modeled EPA, rushing, F1, F2 and F3. Coefficients learn the REMAINING
margin errors; never add the raw SOS feature as "points".

Keep default Build16 unmodified until chronologically later paired validation
warrants promotion. This module cannot use target game outcomes at prediction
time and returns UNAVAILABLE instead of zero-imputing missing features.
"""
from __future__ import annotations
import hashlib,json,math
import numpy as np
from cfb.mandatory_rushing_audit import audit
from cfb.matchups import matchup_signals

VERSION="cfb-continuous-rush-sos-residual-v1"
NAMES=("matched_early_rush_advantage","matched_negative_run_avoidance",
       "opponent_adjusted_rush_epa","opponent_adjusted_total_margin_feature",
       "matched_rush_advantage_x_expected_run_share")
SCALES=np.array((.16,.11,.20,12.,.12),dtype=float)
REGULARIZATION=150.0
MIN_TRAINING_GAMES=80
MAX_ABS_DELTA=6.0
TRAIN_YEAR=2024

def _rate(side,key):
    r=side.get(key)
    if not isinstance(r,dict) or type(r.get("rate")) not in (int,float):
        return None
    x=float(r["rate"])
    return x if math.isfinite(x) and 0<=x<=1 else None

def _mix(rows):
    rush=sum(r.get("off_rush_n") or 0 for r in rows)
    passed=sum(r.get("off_pass_n") or 0 for r in rows)
    if rush+passed<90:return None
    return rush/(rush+passed)

def feature_vector(home_id,away_id,histories):
    """All evidence is computed from previous games; no target/market info."""
    if home_id==away_id:raise ValueError("Matchup needs distinct teams")
    signals=matchup_signals(home_id,away_id,histories)
    rushing=audit(home_id,away_id,histories,signals)
    h,a=rushing["home"],rushing["away"]
    required=(
        _rate(h,"early_rush_success"),
        _rate(h,"opponent_early_rush_success_allowed"),
        _rate(a,"early_rush_success"),
        _rate(a,"opponent_early_rush_success_allowed"),
        _rate(h,"offense_negative_run_rate"),
        _rate(h,"opponent_negative_run_rate_allowed"),
        _rate(a,"offense_negative_run_rate"),
        _rate(a,"opponent_negative_run_rate_allowed"),
        signals.get("sos_rush_epa_differential"),
        signals.get("sos_margin_feature"),
        _mix(histories.get(home_id,())),
        _mix(histories.get(away_id,()))
    )
    if (any(type(v) not in (float,int) or not math.isfinite(v) for v in required)
        or rushing["status"]=="REQUIRES_HUMAN_REVIEW_INCOMPLETE_RUSH_EVIDENCE"):
        return {"status":"UNAVAILABLE_PREGAME_RUSH_OR_OPPONENT_QUALITY",
                "features":None,"missing_instead_of_zero":True}
    oh,dh,oa,da,nh,ndh,na,nda,sos_rush,sos_margin,mixh,mixa=required
    # Opponent's allowed success/negative-run rates are DEFENSE observations
    # of opponent-designed runs. Both sides enter symmetrically; no threshold.
    home_success=.5*(oh+dh)
    away_success=.5*(oa+da)
    home_negative=.5*(nh+ndh)
    away_negative=.5*(na+nda)
    continuous_adv=home_success-away_success
    avoidance=away_negative-home_negative
    expected_share=.5*(mixh+mixa)
    raw=np.array((continuous_adv,avoidance,sos_rush,sos_margin,
                  continuous_adv*expected_share),dtype=float)
    v=np.clip(raw/SCALES,-3.,3.)
    if not np.isfinite(v).all():raise ValueError("Invalid continuous feature")
    return {
       "status":"AVAILABLE_PREGAME_CONTINUOUS_RUSH_SOS",
       "version":VERSION,
       "vector":v.tolist(),
       "features":dict(zip(NAMES,v.round(8).tolist())),
       "unscaled":dict(zip(NAMES,raw.round(8).tolist())),
       "own_prior_games_home":len(histories[home_id]),
       "own_prior_games_away":len(histories[away_id]),
       "continuous_rush_no_trigger_required":True,
       "sos_feature_is_NOT_calibrated_points":True,
       "single_residual_after_preexisting_build16_EPA_and_F1":True,
       "no_market_or_current_game_outcome_used":True
    }

def fit(records):
    """Fit one ridge residual ONLY from complete earlier-year rows.

    Records must be generated with 5,000 real Build16 draws and chronological
    previous-game histories. A separate later-year run evaluates the model.
    """
    if not isinstance(records,list) or len(records)<MIN_TRAINING_GAMES:
        raise ValueError("Need >=80 real earlier-season games")
    if any(r.get("year")!=TRAIN_YEAR or r.get("simulations")!=5000
           for r in records):
        raise ValueError("Training year/draws not source verified")
    ids=[str(r.get("game_id")) for r in records]
    if len(ids)!=len(set(ids)):raise ValueError("Duplicate training game")
    x=np.array([r["feature_vector"] for r in records],dtype=float)
    y=np.array([r["actual_margin"]-r["reference_margin"] for r in records],dtype=float)
    if x.shape!=(len(records),len(NAMES)) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid training observations")
    coef=np.linalg.solve(x.T@x+REGULARIZATION*np.eye(len(NAMES)),x.T@y)
    pred=x@coef
    return {
       "version":VERSION,"training_season":TRAIN_YEAR,
       "training_games":len(records),"training_game_ids_sha256":
         hashlib.sha256((",".join(ids)).encode()).hexdigest(),
       "fitted_after_existing_build16_reference":True,
       "feature_names":list(NAMES),
       "scale":SCALES.tolist(),
       "ridge_regularization":REGULARIZATION,
       "coefficients":coef.round(10).tolist(),
       "margin_adjustment_cap_points":MAX_ABS_DELTA,
       "training_residual_MAE_before":round(float(np.mean(np.abs(y))),5),
       "training_residual_MAE_after":round(float(np.mean(np.abs(y-pred))),5),
       "training_retrospective_source_timestamps_unverified":True,
       "cross_season_ATS_validated":False,
       "promotion_authorized":False,
       "no_2025_or_2026_outcomes_in_fit":True
    }

def score(model,features):
    if model.get("version")!=VERSION or model.get("training_season")!=TRAIN_YEAR:
        raise ValueError("Wrong frozen model type or training year")
    if model.get("feature_names")!=list(NAMES):
        raise ValueError("Frozen feature mismatch")
    if model.get("margin_adjustment_cap_points")!=MAX_ABS_DELTA:
        raise ValueError("Unexpected cap changed since frozen calibration")
    if features.get("status")!="AVAILABLE_PREGAME_CONTINUOUS_RUSH_SOS":
        return {"status":"UNAVAILABLE_NO_PREKICKOFF_FEATURES",
                "margin_delta_points":None,"not_deployed":True}
    vector=np.asarray(features["vector"],dtype=float)
    beta=np.asarray(model["coefficients"],dtype=float)
    if vector.shape!=(len(NAMES),) or not np.isfinite(vector).all() or not np.isfinite(beta).all():
        raise ValueError("Invalid model coefficients/features")
    correction=float(vector@beta)
    bounded=float(np.clip(correction,-MAX_ABS_DELTA,MAX_ABS_DELTA))
    return {"status":"FROZEN_EARLIER_SEASON_RESIDUAL_AVAILABLE",
            "margin_delta_points":round(bounded,6),
            "raw_margin_delta_points":round(correction,6),
            "cap_binding":abs(correction)>MAX_ABS_DELTA,
            "training_season":model["training_season"],
            "not_deployed":True,"ATS_calibrated":False,
            "residual_not_raw_sos_points":True}
