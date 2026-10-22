"""Opponent-specific CFB rushing interaction ABlation RESEARCH; not deployed.

Use only a verified, pre-kickoff per-team game history. Existing independent
integrated scoring already uses matched rush/pass EPA. The *only* efficiency
change below is the residual caused by changing the projected run/pass mix;
do not add the rush EPA advantage a second time.

Play counts per drive are NOT wall-clock possession duration. Clock-consumption
effects remain a hypothesis pending PBP clock validation and calibration.
"""
from __future__ import annotations

import math
from types import MappingProxyType

VERSION='cfb-rush-interaction-research-v1'
HYPOTHESES=MappingProxyType({
    'prior_plays':350.0,
    'defense_mix_response':.45,
    'epa_mix_response':.18,
    'drivelength_per_rush_share_point':1.0,
    'tempo_drives_per_added_play':.40,
    'max_run_share_shift':.10,
    'max_ppd_residual':.12,
    'max_tempo_shift':.75,
})
ABLATIONS=('mix','pace','combined')


def _finite(x):
    return type(x) in (int,float) and math.isfinite(x)


def _clip(x,a,b):
    return max(a,min(b,x))


def _counts(rows,pass_field,rush_field,drive_field):
    p=r=d=0.
    if not rows:return None
    for row in rows:
        values=[row.get(k) for k in (pass_field,rush_field,drive_field)]
        if any(not _finite(v) or v<0 for v in values):
            return None
        if values[2]==0:return None
        p+=values[0];r+=values[1];d+=values[2]
    if p+r<40 or d<3:return None
    return {'plays':p+r,'rush_share':r/(p+r),
            'plays_per_drive':(p+r)/d,'drives':d}


def _side(team_id,opponent_id,histories,scoring_components):
    own=_counts(histories.get(team_id), 'off_pass_n','off_rush_n','drives')
    allowed=_counts(histories.get(opponent_id),'def_pass_n','def_rush_n','faced')
    matched_pass=scoring_components.get('matched_passing') or {}
    matched_rush=scoring_components.get('matched_rushing') or {}
    mpass=matched_pass.get('matched_epa_shrunk')
    mrush=matched_rush.get('matched_epa_shrunk')
    if own is None or allowed is None or not _finite(mpass) or not _finite(mrush):
        return {'status':'UNAVAILABLE_RUSH_HISTORY_OR_MATCHED_EPA'}
    effective=min(own['plays'],allowed['plays'],
                  float(matched_pass.get('plays_effective',0)),
                  float(matched_rush.get('plays_effective',0)))
    if effective<=0:return {'status':'UNAVAILABLE_EFFECTIVE_SAMPLE'}
    reliable=effective/(effective+HYPOTHESES['prior_plays'])
    # Matchup-specific evidence: opponent's allowed run tendency (after
    # shrinkage) and relative adjusted rush-vs-pass efficiency. Neither is a
    # post-kickoff feature. All coefficients here are predeclared hypotheses.
    defense_delta=(allowed['rush_share']-own['rush_share'])*reliable
    epa_delta=_clip(mrush-mpass,-.4,.4)*reliable
    desired_delta=_clip(HYPOTHESES['defense_mix_response']*defense_delta+
                        HYPOTHESES['epa_mix_response']*epa_delta,
                        -HYPOTHESES['max_run_share_shift'],
                         HYPOTHESES['max_run_share_shift'])
    projected_share=_clip(own['rush_share']+desired_delta,.15,.85)
    delta_share=projected_share-own['rush_share']
    # Counterfactual efficiency ONLY of moving shares; the existing EPA
    # component for the original share remains unchanged.
    ppd_delta=_clip(own['plays_per_drive']*delta_share*(mrush-mpass),
                    -HYPOTHESES['max_ppd_residual'],HYPOTHESES['max_ppd_residual'])
    # PBP-scrimmage plays per drive; a drive-length proxy, not measured time.
    length_delta=.5*reliable*(allowed['plays_per_drive']-own['plays_per_drive'])+(
        HYPOTHESES['drivelength_per_rush_share_point']*delta_share)
    return {'status':'AVAILABLE_UNFITTED',
            'old_rush_share':round(own['rush_share'],6),
            'projected_rush_share':round(projected_share,6),
            'rush_share_delta':round(delta_share,6),
            'matched_shrunk_rush_epa':round(mrush,6),
            'matched_shrunk_pass_epa':round(mpass,6),
            'opponent_allowed_rush_share':round(allowed['rush_share'],6),
            'old_plays_per_drive':round(own['plays_per_drive'],5),
            'opponent_allowed_plays_per_drive':round(allowed['plays_per_drive'],5),
            'projected_plays_per_drive':round(own['plays_per_drive']+length_delta,5),
            'length_delta':round(length_delta,6),
            'ppd_delta':round(ppd_delta,6),
            'sample_reliability':round(reliable,6),
            'effective_plays':round(effective,1)}


def experiment(home_id,away_id,histories,home_scoring,away_scoring,mode):
    """Return observable scenario changes for a SINGLE existing challenger.

    This function neither alters a baseline nor runs a second published model.
    A return value with status unavailable MUST NOT silently become a fitted
    zero adjustment. Caller must skip the ablation and preserve existing model.
    """
    if mode not in ABLATIONS:raise ValueError('Unknown rushing ablation mode')
    if home_id==away_id:raise ValueError('Opponent must be distinct')
    h=_side(home_id,away_id,histories,home_scoring)
    a=_side(away_id,home_id,histories,away_scoring)
    if h['status']!='AVAILABLE_UNFITTED' or a['status']!='AVAILABLE_UNFITTED':
        return {'status':'UNAVAILABLE_REQUIRED_EVIDENCE','mode':mode,'home':h,
                'away':a,'not_deployed':True}
    tempo_delta=0.
    if mode in ('pace','combined'):
        tempo_delta=_clip(-HYPOTHESES['tempo_drives_per_added_play']*
                          (h['length_delta']+a['length_delta'])/2,
                          -HYPOTHESES['max_tempo_shift'],
                           HYPOTHESES['max_tempo_shift'])
    return {'status':'UNFITTED_SCENARIO_READY','version':VERSION,'mode':mode,
            'home':h,'away':a,
            'ppd_home_delta':h['ppd_delta'] if mode in ('mix','combined') else 0.,
            'ppd_away_delta':a['ppd_delta'] if mode in ('mix','combined') else 0.,
            'shared_tempo_delta':round(tempo_delta,6),
            'pass_mix_active':mode in ('mix','combined'),
            'tempo_active':mode in ('pace','combined'),
            'hypotheses':dict(HYPOTHESES),
            'measured_clock_duration':False,
            'coefficients_fitted':False,'ATS_calibrated':False,
            'not_deployed':True,
            'limitations':['Drive length is observed scrimmage plays/drive, not elapsed time',
                 'Observed opponent run shares reflect prior opponents and scheme',
                 'Scoring contains only a counterfactual mix residual to avoid EPA double counting',
                 'Hypotheses require chronological train/validation/holdout before use']}
