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
MODES=('distance','sustain','combined','situational')
RUSH_PRIOR=150.
THIRD_PRIOR=120.
CHAIN_PRIOR=45.
CAP_PPD=.10
EARLY_PRIOR=160.
THIRD_BUCKET_PRIOR=90.

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
    # New contextual mechanism: only resolved historical third-down outcomes
    # count, stratified by distance. Adjust toward the defense allowed rate,
    # with the offense's own *observed* distance mix and heavy shrinkage.
    # Early success supplies the causal path into manageable third downs.
    situational_ppd=0.
    situational={}
    if mode=='situational':
        early=_rate(team_rows,'chain_off_','early_success_yes','early_success_n',55)
        allowed_early=_rate(opp_rows,'chain_def_','early_success_yes','early_success_n',55)
        buckets={}
        for bucket in ('short','medium','long'):
            numerator='third_'+bucket+'_yes'
            denominator='third_'+bucket+'_n'
            own=_rate(team_rows,'chain_off_',numerator,denominator,5)
            allowed=_rate(opp_rows,'chain_def_',numerator,denominator,5)
            if own is None or allowed is None:
                return {'status':'UNAVAILABLE_VERIFIED_SITUATIONAL_COUNTS'}
            buckets[bucket]=(own,allowed)
        if early is None or allowed_early is None:
            return {'status':'UNAVAILABLE_VERIFIED_SITUATIONAL_COUNTS'}
        reliability=min(early[1],allowed_early[1])/(min(early[1],allowed_early[1])+EARLY_PRIOR)
        early_diff=(allowed_early[0]-early[0])*reliability
        # Shift third-down distance mix, not a second standalone rush-EPA bonus.
        short_weight=(buckets['short'][0][1]+10.) / (
            sum(v[0][1] for v in buckets.values())+30.)
        medium_weight=(buckets['medium'][0][1]+10.) / (
            sum(v[0][1] for v in buckets.values())+30.)
        long_weight=1.-short_weight-medium_weight
        new_short=_clamp(short_weight+.30*early_diff,.02,.90)
        new_long=_clamp(long_weight-.30*early_diff,.02,.90)
        new_medium=1.-new_short-new_long
        if new_medium<0:
            return {'status':'UNAVAILABLE_INVALID_DISTANCE_MIX'}
        matched=baseline=0.
        for bucket,w0,w1 in (
            ('short',short_weight,new_short),('medium',medium_weight,new_medium),
            ('long',long_weight,new_long)):
            own,allowed=buckets[bucket]
            shrink=min(own[1],allowed[1])/(min(own[1],allowed[1])+THIRD_BUCKET_PRIOR)
            p=_clamp(own[0]+.5*(allowed[0]-own[0])*shrink,.01,.99)
            matched+=w1*p;baseline+=w0*own[0]
        # Increment only the matchup departure from the offense's existing
        # third-down expectation (already represented by Build16).
        situational_ppd=_clamp(.32*(matched-baseline),-.055,.055)
        situational={'early_success_matchup_delta':round(early_diff,6),
          'third_conversion_baseline':round(baseline,6),
          'third_conversion_distance_adjusted':round(matched,6),
          'third_distance_mix':{'short':round(new_short,6),
             'medium':round(new_medium,6),'long':round(new_long,6)},
          'method':'Resolved earlier-game conversions only; prior opponent defensive rates'}
    applied=(dist_ppd if mode=='distance' else sustain_ppd if mode=='sustain'
             else situational_ppd if mode=='situational' else dist_ppd+sustain_ppd)
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
        'situational':situational,'situational_ppd_delta':round(situational_ppd,6),
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
               'early_success_prior':EARLY_PRIOR,'third_bucket_prior':THIRD_BUCKET_PRIOR,
               'maximum_absolute_ppd_increment':CAP_PPD},
            'not_deployed':True,'fitted':False,'ATS_calibrated':False,
            'limitations':['Existing Build16 already includes rush EPA and third conversions',
              'Drive sustenance is 3+ first-down series; not a direct verified first down total',
              'Counterfactual rate coefficients are unfitted hypotheses',
              'Source third-down flags alone are not used as failure evidence']}
