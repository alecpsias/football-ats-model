"""Enable the single 2024-fitted rush/SOS residual in standard CFB Build16.

Never self-fit from future games during a pregame prediction. Read only a
committed, immutable earlier-year fit. Missing/corrupt artifact or unavailable
pregame features triggers transparent legacy Build16 fallback; never assume
zero rushing evidence and never invent coefficients. Historical ablations
can disable this auto feature to preserve their frozen comparison.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from cfb.continuous_rush_sos_residual import (
    VERSION,TRAIN_YEAR,MIN_TRAINING_GAMES,feature_vector,score)
FROZEN_MODEL=Path(__file__).parent/"frozen_models"/"continuous_rush_sos_2024.json"

def read_model(path=None):
    file=Path(path) if path is not None else FROZEN_MODEL
    if not file.exists():
        return None,{"status":"FROZEN_2024_RESIDUAL_MODEL_NOT_INSTALLED","path":str(file)}
    try:
        data=json.loads(file.read_text(encoding="utf-8"))
        if data.get("version")!=VERSION or data.get("training_season")!=TRAIN_YEAR:
            raise ValueError("Wrong frozen fit year/model version")
        if data.get("training_games",0)<MIN_TRAINING_GAMES:
            raise ValueError("Insufficient original training games")
        if data.get("promoted_into_build16") is not True:
            raise ValueError("User-requested activation flag missing")
        if data.get("training_retrospective_source_timestamps_unverified") is not True:
            raise ValueError("Missing honest source-time caveat")
        if data.get("cross_season_ATS_validated") is not False:
            raise ValueError("An unsupported ATS calibration claim is present")
        v=[float(x) for x in data.get("coefficients",())]
        if len(v)!=5 or not all(math.isfinite(x) for x in v):
            raise ValueError("Invalid coefficients")
        return data,{"status":"FROZEN_2024_RESIDUAL_MODEL_LOADED",
                     "version":VERSION,"training_games":data["training_games"],
                     "training_year":data["training_season"],
                     "ATS_calibrated":False,"retrospective_training_capture":True}
    except (ValueError,TypeError,KeyError,OverflowError,UnicodeError,json.JSONDecodeError) as ex:
        return None,{"status":"FROZEN_2024_RESIDUAL_INVALID",
                     "reason":str(ex),"path":str(file)}

def resolve(home_id,away_id,histories):
    model,audit=read_model()
    if model is None:return None,audit
    try:
        vector=feature_vector(home_id,away_id,histories)
        scored=score(model,vector)
        if scored["status"]!="FROZEN_EARLIER_SEASON_RESIDUAL_AVAILABLE":
            return None,{**audit,"status":scored["status"],"feature_status":vector["status"],
                         "applied":False}
        return scored["margin_delta_points"],{
            **audit,"status":"ACTIVE_2024_FITTED_RUSH_SOS_RESIDUAL",
            "applied":True,"delta_margin_points_home":scored["margin_delta_points"],
            "feature_status":vector["status"],
            "training_only_previous_season":True,
            "margin_residual_after_existing_EPA_and_F1":True,
            "mode":"default_Build16_live",
            "holdout_ATS_accuracy_not_established":True}
    except (ValueError,KeyError,TypeError,ArithmeticError) as ex:
        return None,{**audit,"status":"FROZEN_2024_RESIDUAL_FEATURES_UNAVAILABLE",
                     "applied":False,"reason":str(ex)}
