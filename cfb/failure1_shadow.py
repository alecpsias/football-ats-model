"""Immutable, pre-kickoff-only, opt-in Failure-1 negative-run shadow.

This is ONE internal ablation of the existing Build16 challenger, never a
separate published challenger. The default Build16 + separate baseline are
unmodified. Null inputs remain null; no posterior refit or ATS recommendation.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import math
from cfb.rush_severity_source import extract,append_training_record
from cfb.mechanism_build16 import simulate_build16

PROTOCOL="cfb-failure1-frozen-negative-designed-runs-v1"
CONFIG_FILE=Path(__file__).parent/'rush_severity_research.py'

def include_prior_game(h, a, plays, home_id, away_id):
    """Record source-derived negative rushing features for prior games only."""
    x=extract(plays,home_id,away_id)
    y=extract(plays,away_id,home_id)
    append_training_record(h,x)
    append_training_record(a,y)
    return {'home':x['status'],'away':y['status']}

def shadow(home_id,away_id,histories,neutral,*,runs=5000,seed,source_manifest_sha256):
    if runs!=5000 or type(runs) is not int:raise ValueError('Shadow requires 5000 simulations')
    if not isinstance(source_manifest_sha256,str) or len(source_manifest_sha256)!=64:
        raise ValueError('Frozen source manifest digest required')
    sha=sha256(CONFIG_FILE.read_bytes()).hexdigest()
    base={'protocol':PROTOCOL,'mode':'negative','source_manifest_sha256':source_manifest_sha256,
          'candidate_source_sha256':sha,'runs':5000,'seed':seed,
          'training_games_from_target_fixture':False,'empirically_calibrated_ats':False,
          'separate_published_challenger':False,'is_pre_game_research_only':True,
          'market_quote':None,'bet_recommendation':None,
          'status':'INSUFFICIENT_PREKICKOFF_RUSH_EVIDENCE'}
    r=simulate_build16(home_id,away_id,histories,neutral,
          runs=runs,seed=seed,mismatch_research='negative')
    if r.get('simulation') is None:
        return {**base,'failure_reason':r.get('status'),
                'source_failure':(r.get('mismatch_research') or {}).get('status')}
    result=r['simulation']
    return {**base,'status':'FROZEN_NEGATIVE_RUN_SHADOW_5000',
         'version':r['version'],
         'projected_margin_home':result['mean_margin_home'],
         'p10_p50_p90':result['margin_p10_p50_p90'],
         'home_win_probability_raw':result['home_win_probability_raw'],
         'margin_sd':result['margin_standard_deviation'],
         'frozen_effect_home_ppd':r['mismatch_research']['ppd_home_delta'],
         'frozen_effect_away_ppd':r['mismatch_research']['ppd_away_delta'],
         'uncertainty':'Unfitted; 2025+2026 retrospective gains not ATS validated'}
