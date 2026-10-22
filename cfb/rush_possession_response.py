"""CFB opponent-specific rushing -> sustained possessions -> real clock research.

NOT raw rushing EPA again. This measures the pregame shift in (a) 3+
first-down-series drive probability (source PBP) conditional on opponent,
and (b) shared drive opportunities from strictly verified duration. Training
is earlier-season-only, learned *incrementally* to default Build16 residuals.
No synthetic possession clock, and no new active challenger by default.
"""
from __future__ import annotations
import math
import numpy as np

VERSION='cfb-possession-rushing-opponent-residual-v1'
CLOCK_PRIOR_DRIVES=60.
EARLY_PRIOR_RUNS=150.
CHAIN_PRIOR_DRIVES=55.
MAX_SUSTAIN_PPD=.22
MAX_SHARED_TEMPO=.65

def _rate(records,prefix,num,den,minimum=20,minimum_games=3):
    pairs=[]
    for x in records:
        if num+prefix not in x or den+prefix not in x:
            continue
        a,b=x[num+prefix],x[den+prefix]
        if type(a) not in (int,float) or type(b) not in (int,float) or not math.isfinite(a) or not math.isfinite(b) or a<0 or b<0 or a>b:return None
        if b>0:pairs.append((float(a),float(b)))
    if len(pairs)<minimum_games:return None
    n=sum(a for a,_ in pairs);d=sum(b for _,b in pairs)
    if d<minimum:return None
    return n/d,d

def _clock(records,off):
    p='possession_' if off else 'possession_def_'
    pairs=[]
    for row in records:
        s=row.get(p+'observed_time_sec')
        d=row.get(p+'verified_drive_n')
        if s is None or d is None:continue
        if type(s) not in (int,float) or type(d) not in (int,float) or not math.isfinite(s) or not math.isfinite(d) or d<=0 or not 85*d<=s<=460*d:return None
        pairs.append((s,d))
    if len(pairs)<3:return None
    total=sum(x for x,_ in pairs);drives=sum(x for _,x in pairs)
    if drives<24:return None
    return {'seconds_per_drive':total/drives,'drives':drives,'qualified_games':len(pairs)}

def _side(team,opp,histories):
    a=histories.get(team,[]);b=histories.get(opp,[])
    if len(a)<3 or len(b)<3:return {'status':'MISSING_PRIOR_GAMES'}
    o=_rate(a,'chain_off_','early_rush_success_yes','early_rush_success_n',30)
    d=_rate(b,'chain_def_','early_rush_success_yes','early_rush_success_n',30)
    mix=_rate(a,'chain_off_','early_rush_plays','early_offense_plays',50)
    long_o=_rate(a,'chain_off_','sustained_drives','observed_drives',20)
    long_d=_rate(b,'chain_def_','sustained_drives','observed_drives',20)
    manage_o=_rate(a,'chain_off_','third_manageable_yes','third_attempts',15)
    manage_d=_rate(b,'chain_def_','third_manageable_yes','third_attempts',15)
    if any(v is None for v in (o,d,mix,long_o,long_d,manage_o,manage_d)):
        return {'status':'MISSING_MATCHED_DRIVE_CHAIN_RUSH_EVIDENCE'}
    rush_reliability=min(o[1],d[1])/(min(o[1],d[1])+EARLY_PRIOR_RUNS)
    chain_reliability=min(long_o[1],long_d[1])/(min(long_o[1],long_d[1])+CHAIN_PRIOR_DRIVES)
    # Matched offense-vs-defense departures from offense's own past usage.
    # These probability shifts refer to actual observed ability to chain series,
    # not scoring output and not the original rush EPA already in Build16.
    matched_rush=0.5*(o[0]+d[0])
    rush_edge=(matched_rush-o[0])*mix[0]*rush_reliability
    sustain_edge=(long_d[0]-long_o[0])*.50*chain_reliability
    third_edge=(manage_d[0]-manage_o[0])*.15*chain_reliability
    net=float(np.clip(rush_edge*.55+sustain_edge*.75+third_edge*.20,-.075,.075))
    clock_o=_clock(a,True)
    clock_d=_clock(b,False)
    timediff=None
    if clock_o is not None and clock_d is not None:
        w=min(clock_o['drives'],clock_d['drives'])
        reliability=w/(w+CLOCK_PRIOR_DRIVES)
        # Source-observed opponent-duration interaction (not plays/drive proxy).
        timediff=float(np.clip(.45*reliability*
            (clock_d['seconds_per_drive']-clock_o['seconds_per_drive']),-14.,14.))
    return {'status':'AVAILABLE_SOURCE_DRIVE_CHAIN',
       'own_early_rush_success':round(o[0],6),
       'opponent_early_rush_success_allowed':round(d[0],6),
       'own_early_rush_share':round(mix[0],6),
       'own_sustained_drive_rate':round(long_o[0],6),
       'opponent_allowed_sustained_drive_rate':round(long_d[0],6),
       'own_third_manageable_share':round(manage_o[0],6),
       'opponent_allowed_third_manageable_share':round(manage_d[0],6),
       'matched_drive_continuation_shift':round(net,8),
       'observed_clock_own':clock_o,'observed_clock_opponent_allowed':clock_d,
       'opponent_clock_delta_seconds_per_drive':round(timediff,5) if timediff is not None else None,
       'clock_data_status':'SOURCE_VERIFIED' if timediff is not None else 'NOT_VERIFIED_NO_TEMPO_CORRECTION',
       'drive_continuation_source':'CHAIN_OF_AT_LEAST_3_FIRST_DOWN_SERIES',
       'rush_and_chain_samples':{'own_early_rushes':o[1],'opp_rushes':d[1],
                                  'own_drives':long_o[1],'opp_drives':long_d[1]}}

def signal(home_id,away_id,histories):
    if home_id==away_id:raise ValueError('Distinct CFB opponents required')
    home=_side(home_id,away_id,histories)
    away=_side(away_id,home_id,histories)
    if home['status']!='AVAILABLE_SOURCE_DRIVE_CHAIN' or away['status']!='AVAILABLE_SOURCE_DRIVE_CHAIN':
        return {'status':'UNAVAILABLE_VERIFIED_DRIVE_CHAIN','home':home,'away':away,
                'not_deployed':True,'no_imputed_history':True}
    tempo=0.
    if home['opponent_clock_delta_seconds_per_drive'] is not None and away['opponent_clock_delta_seconds_per_drive'] is not None:
        h=home['observed_clock_own']['seconds_per_drive']
        a=away['observed_clock_own']['seconds_per_drive']
        d=home['opponent_clock_delta_seconds_per_drive']+away['opponent_clock_delta_seconds_per_drive']
        if h+a>200 and h+a+d>200:
            tempo=float(np.clip(3600/(h+a+d)-3600/(h+a),-MAX_SHARED_TEMPO,MAX_SHARED_TEMPO))
    return {'status':'AVAILABLE_PREGAME_DRIVE_CONTINUATION',
        'version':VERSION,'home':home,'away':away,
        'home_continuation_delta':home['matched_drive_continuation_shift'],
        'away_continuation_delta':away['matched_drive_continuation_shift'],
        'shared_verified_clock_drives_delta':round(tempo,6),
        'clock_both_sides_qualified':home['clock_data_status']=='SOURCE_VERIFIED' and away['clock_data_status']=='SOURCE_VERIFIED',
        'source':'previously completed same-game PBP only, 24h embargo required in caller',
        'does_not_duplicate_rushing_EPA':True,
        'not_deployed':True}

def fit(rows):
    """Single parameter learned in training season from Build16 margin residual."""
    z=[r for r in rows if r['signal']['status']=='AVAILABLE_PREGAME_DRIVE_CONTINUATION']
    if len(z)<75:raise ValueError('Too few genuine completed-game historical rush-chain observations')
    year={r['year'] for r in z}
    if len(year)!=1:raise ValueError('Only a single earlier training year')
    x=np.asarray([(r['signal']['home_continuation_delta']-r['signal']['away_continuation_delta'])*
      r['reference_expected_drives'] for r in z],float)
    y=np.asarray([r['actual_margin']-r['reference_margin'] for r in z],float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite training information')
    # Freeze regularization, no tuning on a subsequent season. Coefficient
    # measures marginal points per extra expected sustained possession.
    ridge=25.
    alpha=float(np.clip(x.dot(y)/(x.dot(x)+ridge),-4.0,4.0))
    return {'version':VERSION,'trained_year':list(year)[0],
        'train_games':len(z),'ridge_prior_strength':ridge,
        'continuation_points_per_extra_drive':round(alpha,10),
        'absolute_ppd_cap':MAX_SUSTAIN_PPD,
        'max_verified_clock_shared_drives_shift':MAX_SHARED_TEMPO,
        'clock_reliability_prior_drives':CLOCK_PRIOR_DRIVES,
        'not_ATS_calibrated':True,
        'trained_on_reference_margin_residual_only':True}

def response(fitted,observation):
    if observation['status']!='AVAILABLE_PREGAME_DRIVE_CONTINUATION':
        return {'status':'NO_PREGAME_CHAIN_OBSERVATIONS','not_deployed':True}
    if fitted.get('version')!=VERSION or not 75<=fitted.get('train_games',0) or fitted.get('trained_year')!=2023:
        raise ValueError('Incorrect frozen earlier-year possession fit')
    alpha=fitted.get('continuation_points_per_extra_drive')
    if type(alpha) not in (int,float) or not math.isfinite(alpha) or not -4<=alpha<=4:
        raise ValueError('Bad fitted points-per-sustained-drive coefficient')
    h=float(np.clip(alpha*observation['home_continuation_delta'],-MAX_SUSTAIN_PPD,MAX_SUSTAIN_PPD))
    a=float(np.clip(alpha*observation['away_continuation_delta'],-MAX_SUSTAIN_PPD,MAX_SUSTAIN_PPD))
    return {'status':'AVAILABLE_FROZEN_POSSESSION_RESPONSE',
            'home_ppd_shift':round(h,6),'away_ppd_shift':round(a,6),
            'shared_clock_drives_shift':observation['shared_verified_clock_drives_delta'],
            'real_clock_both_sides_valid':observation['clock_both_sides_qualified'],
            'fitted_train_year':fitted['trained_year'],
            'simulations_required':5000,
            'not_deployed':True,'ATS_calibrated':False}
