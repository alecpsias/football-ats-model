"""Opponent-specific early-down rushing -> third down -> sustained drives.

This module is a *research toggle inside the ONE CFB challenger*, not another
published model. Each observation must precede kickoff (>=24h replay embargo).
The existing Build16 already uses rushing EPA and third-down conversion rates:
only a small *incremental matchup departure* from own history enters PPD.
All sensitivity coefficients are unfit prior hypotheses. Missing fails closed.
"""
from __future__ import annotations
import math

VERSION='cfb-rush-third-chain-ablation-v1'
MODES=('distance','sustain','combined')
RUSH_PRIOR=150.
THIRD_PRIOR=120.
CHAIN_PRIOR=45.
CAP_PPD=.10

def _clamp(x,lo,hi):return max(lo,min(hi,x))

def _rate(rows,prefix,num,den,minimum):
    try:
        vals=[(r[prefix+num],r[prefix+den]) for r in rows]
        if len(vals)<3 or any(type(a) not in (float,int) or type(b) not in (float,int)
          or not math.isfinite(a) or not math.isfinite(b)
          or a<0 or b<0 or a>b for a,b in vals):return None
        a=sum(v for v,_ in vals);b=sum(v for _,v in vals)
        if b<minimum:return None
        return (a/b,b)
    except (KeyError,TypeError,ValueError):return None

def _team(team_rows,opp_rows,mode):
    own_rush=_rate(team_rows,'chain_off_','early_rush_success_yes','early_rush_success_n',30)
    def_rush=_rate(opp_rows,'chain_def_','early_rush_success_yes','early_rush_success_n',30)
    own_mix=_rate(team_rows,'chain_off_','early_rush_plays','early_offense_plays',30)
    own_manage=_rate(team_rows,'chain_off_','third_manageable_yes','third_attempts',15)
    def_manage=_rate(opp_rows,'chain_def_','third_manageable_yes','third_attempts',15)
    own_chain=_rate(team_rows,'chain_off_','sustained_drives','observed_drives',20)
    def_chain=_rate(opp_rows,'chain_def_','sustained_drives','observed_drives',20)
    values=(own_rush,def_rush,own_mix,own_manage,def_manage,own_chain,def_chain)
    if any(v is None for v in values):
        return {'status':'UNAVAILABLE_VERIFIED_SOURCE_COUNTS'}
    r,d,m,t,dt,c,dc=values
    shrink_r=min(r[1],d[1])/(min(r[1],d[1])+RUSH_PRIOR)
    shrink_t=min(t[1],dt[1])/(min(t[1],dt[1])+THIRD_PRIOR)
    shrink_c=min(c[1],dc[1])/(min(c[1],dc[1])+CHAIN_PRIOR)
    # A matched advantage must change expected *next down* or *chains*,
    # not add the same raw rushing EPA once more. Group interactions relative
    # to each offense's own historical usage and third-down mix.
    matched_rush=.5*(r[0]+d[0])
    run_interaction=(matched_rush-r[0])*m[0]*shrink_r
    manageable_delta=_clamp(.55*(dt[0]-t[0])*shrink_t+
                             .25*run_interaction,-.09,.09)
    chain_delta=_clamp(.48*(dc[0]-c[0])*shrink_c+
                       .35*manageable_delta,-.08,.08)
    dist_ppd=_clamp(.35*manageable_delta,-.035,.035)
    sustain_ppd=_clamp(.65*chain_delta,-.065,.065)
    applied=dist_ppd if mode=='distance' else sustain_ppd if mode=='sustain' else dist_ppd+sustain_ppd
    return {'status':'AVAILABLE_UNFITTED','prior_off_early_rush_success':round(r[0],5),
        'opponent_allowed_early_rush_success':round(d[0],5),
        'own_prior_early_rush_share':round(m[0],5),
        'prior_third_manageable_share':round(t[0],5),
        'opponent_allowed_third_manageable_share':round(dt[0],5),
        'prior_sustained_drive_share':round(c[0],5),
        'opponent_allowed_sustained_drive_share':round(dc[0],5),
        'matched_rush_success_difference':round(matched_rush-r[0],6),
        'rush_success_sample_reliability':round(shrink_r,6),
        'manageable_third_share_delta':round(manageable_delta,6),
        'sustained_drive_share_delta':round(chain_delta,6),
        'distance_ppd_delta':round(dist_ppd,6),
        'chain_ppd_delta':round(sustain_ppd,6),
        'total_ppd_delta':round(_clamp(applied,-CAP_PPD,CAP_PPD),6),
        'effective_drive_samples':min(c[1],dc[1]),
        'source_type':'THIRD_DISTANCE_NEXT_DOWN_PLUS_CHAINED_FIRST_DOWN_SERIES'}

def experiment(home_id,away_id,histories,mode):
    if mode not in MODES:raise ValueError('Unknown rush-chain ablation')
    if home_id==away_id:raise ValueError('Teams must be distinct')
    h=_team(histories.get(home_id,[]),histories.get(away_id,[]),mode)
    a=_team(histories.get(away_id,[]),histories.get(home_id,[]),mode)
    if h['status']!='AVAILABLE_UNFITTED' or a['status']!='AVAILABLE_UNFITTED':
        return {'status':'UNAVAILABLE_REQUIRED_HISTORICAL_CHAIN_EVIDENCE',
                'home':h,'away':a,'mode':mode,'not_deployed':True}
    return {'status':'UNFITTED_SCENARIO_READY','version':VERSION,'mode':mode,
            'home':h,'away':a,'ppd_home_delta':h['total_ppd_delta'],
            'ppd_away_delta':a['total_ppd_delta'],
            'does_not_double_count_rush_epa_or_third_conversions':True,
            'assumptions_unfitted':{'early_rush_success_prior':RUSH_PRIOR,
               'third_mix_prior':THIRD_PRIOR,'drive_chain_prior':CHAIN_PRIOR,
               'maximum_absolute_ppd_increment':CAP_PPD},
            'not_deployed':True,'fitted':False,'ATS_calibrated':False,
            'limitations':['Existing Build16 already includes rush EPA and third conversions',
              'Drive sustenance is 3+ first-down series; not a direct verified first down total',
              'Counterfactual rate coefficients are unfitted hypotheses',
              'Source third-down flags alone are not used as failure evidence']}
