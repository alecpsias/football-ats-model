"""Forward-only ONE research shadow for the 2023-fitted running mismatch.

Immutable frozen 2023 regression coefficients are imported exactly from repo,
new PBP/snapshot data must predate kickoff, 5,000 real draws, same seed as
Build16 reference, never a separate published challenger or ATS recommendation.
The older fixed 'negative' shadow is retired for new pregame reports.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from cfb.failure1_ridge_correction import feature_vector,score
from cfb.mechanism_build16 import simulate_build16

MODEL=Path(__file__).parent/'FAILURE1_RIDGE_2023_FROZEN.json'
PROTOCOL='cfb-failure1-learned-rush-shadow-v1'

def shadow(home_id,away_id,histories,neutral,*,runs,seed,source_manifest_sha256):
    if type(runs) is not int or runs!=5000:
        raise ValueError('Frozen learned rushing shadow requires EXACT 5000 simulations')
    if not isinstance(source_manifest_sha256,str) or len(source_manifest_sha256)!=64:
        raise ValueError('Require exact original source-manifest SHA256')
    raw=MODEL.read_bytes()
    mdl=json.loads(raw)
    if mdl['fit_year']!=2023 or mdl['n_training_games']!=180:
        raise ValueError('2023-only model unexpectedly mutated')
    snap=feature_vector(home_id,away_id,histories)
    corr=score(mdl,snap)
    base={'status':'UNAVAILABLE_PREKICKOFF_RUSH_EVIDENCE',
      'protocol':PROTOCOL,'mode':'previously_fitted_rushing_interaction',
      'fitted_year':2023,'training_games':180,
      'source_manifest_sha256':source_manifest_sha256,
      'frozen_fitted_model_sha256':hashlib.sha256(raw).hexdigest(),
      'runs':5000,'seed':seed,'market_quote':None,'bet_recommendation':None,
      'not_deployed':True,'ATS_calibrated':False}
    if corr['status']!='VALID_FROZEN_MODEL_PREGAME_RUSH_CORRECTION':
        return {**base,'reason':corr['status'],'margin_delta_points':None}
    result=simulate_build16(home_id,away_id,histories,neutral,runs=runs,
       seed=seed,failure1_fit_margin_delta=corr['margin_points'])
    if result.get('simulation') is None:
        return {**base,'reason':result.get('status'),'margin_delta_points':None}
    sim=result['simulation']
    return {**base,'status':'FROZEN_2023_FITTED_RUSH_SHADOW_5000',
       'margin_delta_points':corr['margin_points'],
       'projected_margin_home':sim['mean_margin_home'],
       'margin_p10_p50_p90':sim['margin_p10_p50_p90'],
       'win_probability_home_raw':sim['home_win_probability_raw'],
       'score_home':sim['mean_score_home'],'score_away':sim['mean_score_away'],
       'genuine_5000_run_distribution':True,
       'not_resimulated_after_kickoff':True}
