"""Opponent-specific CFB rushing interaction ABlation RESEARCH; not deployed.

Use only a verified, pre-kickoff per-team game history. Existing independent
integrated scoring already uses matched rush/pass EPA. The *only* efficiency
change below is the residual caused by changing the projected run/pass mix;
do not add the rush EPA advantage a second time.

Experimental clock responses now require source-verified, regulation-only elapsed
drive times. Historic EPA labels and official NCAA box play-mix labels remain separate.
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
    'tempo_drives_per_added_play':.40, # RETIRED PROXY: not used by clock mode
    'seconds_per_full_rush_share':75.0, # unfitted sensitivity, not causal
    'clock_response_prior_drives':60.0,
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


def _clock_avg(records,seconds,drives):
    sample=[(r.get(seconds),r.get(drives)) for r in records
            if _finite(r.get(seconds)) and _finite(r.get(drives)) and r[drives]>0]
    if len(sample)<3:return None
    total=sum(x for x,_ in sample)
    count=sum(y for _,y in sample)
    if count<16 or not 100*count<=total<=420*count:return None
    return {'seconds_per_drive':total/count,'drives':count,'games':len(sample)}

def _box_share(records,rush_col,pass_col):
    sample=[(r.get(rush_col),r.get(pass_col)) for r in records
            if _finite(r.get(rush_col)) and _finite(r.get(pass_col))
            and r[rush_col]+r[pass_col]>0]
    if len(sample)<3:return None
    rush=sum(x for x,_ in sample);pas=sum(y for _,y in sample)
    if rush+pas<40:return None
    return {'share':rush/(rush+pas),'plays':rush+pas,'games':len(sample)}

def _side(team_id,opponent_id,histories,scoring_components):
    own=_counts(histories.get(team_id), 'off_pass_n','off_rush_n','drives')
    allowed=_counts(histories.get(opponent_id),'def_pass_n','def_rush_n','faced')
    matched_pass=scoring_components.get('matched_passing') or {}
    matched_rush=scoring_components.get('matched_rushing') or {}
    mpass=matched_pass.get('matched_epa_shrunk')
    mrush=matched_rush.get('matched_epa_shrunk')
    if own is None or allowed is None or not _finite(mpass) or not _finite(mrush):
        return {'status':'UNAVAILABLE_RUSH_HISTORY_OR_MATCHED_EPA'}
    source_own=_box_share(histories.get(team_id,[]),'clock_run_n','clock_pass_n')
    source_allowed=_box_share(histories.get(opponent_id,[]),'clock_def_run_n','clock_def_pass_n')
    if source_own is not None and source_allowed is not None:
        own['rush_share']=source_own['share']
        allowed['rush_share']=source_allowed['share']
        source_type='NCAA_BOX_SACKS_AS_RUSH'
    else:
        source_type='EPA_ORIGINAL_RUSH_PASS'
    own_clock=_clock_avg(histories.get(team_id,[]),'clock_time_sum','clock_drive_n')
    allowed_clock=_clock_avg(histories.get(opponent_id,[]),'clock_def_time_sum','clock_def_drive_n')
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
    # Explicitly forbid plays-per-drive as a clock proxy. Clock-only modes
    # fail closed when current historical provider observations are missing.
    clock=None
    if own_clock is not None and allowed_clock is not None:
        drive_n=min(own_clock['drives'],allowed_clock['drives'])
        clock_reliability=drive_n/(drive_n+HYPOTHESES['clock_response_prior_drives'])
        clock={'own_seconds_per_drive':own_clock['seconds_per_drive'],
               'opponent_allowed_seconds_per_drive':allowed_clock['seconds_per_drive'],
               'clock_drive_samples':drive_n,
               'reliability':clock_reliability}
    length_delta=0. # no arbitrary plays-per-drive adjustment
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
            'clock':clock,
            'rush_share_source':source_type,
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
        if h['clock'] is None or a['clock'] is None:
            return {'status':'UNAVAILABLE_SOURCE_VERIFIED_DRIVE_CLOCK',
                    'mode':mode,'home':h,'away':a,'not_deployed':True}
        own_seconds=h['clock']['own_seconds_per_drive']+a['clock']['own_seconds_per_drive']
        if own_seconds<200:return {'status':'INVALID_DRIVE_TIME','mode':mode,'not_deployed':True}
        increments=[]
        for side in (h,a):
            clk=side['clock']
            delta=(.45*clk['reliability']*
                   (clk['opponent_allowed_seconds_per_drive']-clk['own_seconds_per_drive']))
            if mode=='combined':
                delta+=HYPOTHESES['seconds_per_full_rush_share']*side['rush_share_delta']
            side['projected_seconds_per_drive']=round(
                clk['own_seconds_per_drive']+delta,5)
            side['seconds_per_drive_delta']=round(delta,5)
            increments.append(delta)
        new_total=own_seconds+sum(increments)
        if new_total<=200:return {'status':'INVALID_PROJECTED_DRIVE_TIME','mode':mode,'not_deployed':True}
        tempo_delta=_clip(3600/new_total-3600/own_seconds,
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
            'measured_clock_duration':mode in ('pace','combined'),
            'coefficients_fitted':False,'ATS_calibrated':False,
            'not_deployed':True,
            'limitations':['Clock modes require provider elapsed drive seconds, no plays-per-drive clock proxy',
                 'Observed opponent run shares reflect prior opponents and scheme',
                 'Scoring contains only a counterfactual mix residual to avoid EPA double counting',
                 'Hypotheses require chronological train/validation/holdout before use']}
