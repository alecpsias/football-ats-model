"""One prespecified opponent rushing matchup residual to Build16 margin.

Pre-game features ONLY, from verified >=24h-old histories. Fits one
no-intercept ridge correction to *score margin error* on a training season.
Coefficients are not fitted to season or game being tested. The adjustment
changes two teams' PPD symmetrically so ALL 5000 future sim draws respond.
No ATS calibration and no extrapolation for missing play-level rates.
"""
from __future__ import annotations
import math
import numpy as np
from cfb.mandatory_rushing_audit import audit
from cfb.matchups import matchup_signals

VERSION='cfb-rushing-residual-ridge-v1'
RIDGE_LAMBDA=250.0 # fixed before seeing any training-model results
MAX_DELTA_POINTS=3.0
SCALES=np.array([.20,.20,.10,.10,.20,.20,.25],float)
NAMES=('home_rush_success_adv','away_rush_success_adv',
       'home_negative_avoidance','away_negative_avoidance',
       'sos_rush_epa','relative_run_share','joint_severity_difference')

def feature_vector(home_id,away_id,histories):
    signals=matchup_signals(home_id,away_id,histories)
    a=audit(home_id,away_id,histories,signals)
    h,away=a['home'],a['away']
    def val(side,key):
        z=side.get(key)
        return z.get('rate') if isinstance(z,dict) else None
    ownh=val(h,'early_rush_success')
    ownaway=val(away,'early_rush_success')
    allowh=val(h,'opponent_early_rush_success_allowed')
    allowaway=val(away,'opponent_early_rush_success_allowed')
    nh=h['matched_negative_run_difference']
    na=away['matched_negative_run_difference']
    rush_epa=a['sos_rush_epa_differential']
    def run_share(records):
        ru=sum(r.get('off_rush_n',0) for r in records)
        pa=sum(r.get('off_pass_n',0) for r in records)
        return ru/(ru+pa) if ru+pa>=100 else None
    hs=run_share(histories.get(home_id,[]))
    ads=run_share(histories.get(away_id,[]))
    fields=(ownh,ownaway,allowh,allowaway,nh,na,rush_epa,hs,ads,
        h['joint_severity_excess'],away['joint_severity_excess'])
    if any(type(x) not in (float,int) or not math.isfinite(x) for x in fields):
        return {'status':'UNAVAILABLE_REQUIRED_ASOF_RUSH_INPUTS',
                'missing':'opponent-success, negative-run, SOS rush EPA, run share, or severity',
                'source':'historical only'}
    raw=np.array([(.5*(ownh+allowh))-.45,
                  (.5*(ownaway+allowaway))-.45,
                  -nh,-na,rush_epa,hs-ads,
                  h['joint_severity_excess']-away['joint_severity_excess']],float)
    # Odd indices are AWAY-positive features: convert into home-score sign.
    raw[[1,3]]*=-1.
    scaled=np.clip(raw/SCALES,-3.,3.)
    return {'status':'AVAILABLE_PREGAME_RUSH_VECTOR',
       'version':VERSION,'features':{n:round(float(x),7) for n,x in zip(NAMES,scaled)},
       'vector':scaled.tolist(),'audit_severe':a['requires_full_rush_stress'],
       'source_status':a['status'],'no_outcomes_used':True}

def fit(train):
    if len(train)<120:raise ValueError('Require >=120 earlier-year qualified observations')
    x=np.array([r['features'] for r in train],float)
    error=np.array([r['actual_margin']-r['reference_margin'] for r in train],float)
    if x.shape!=(len(train),len(NAMES)):raise ValueError('Bad training design matrix')
    if not np.isfinite(x).all() or not np.isfinite(error).all():raise ValueError('Non-finite labels/features')
    beta=np.linalg.solve(x.T@x+RIDGE_LAMBDA*np.eye(len(NAMES)),x.T@error)
    return {'version':VERSION,'fit_year':int(train[0]['year']),
        'n_training_games':len(train),'ridge_lambda':RIDGE_LAMBDA,
        'feature_names':list(NAMES),'features_scale':SCALES.tolist(),
        'coefficients':beta.round(10).tolist(),'max_delta_points':MAX_DELTA_POINTS,
        'training_captures_retrospective':True,'ATS_calibrated':False,
        'frozen_after_earlier_year':True}

def score(model,vector):
    if model['version']!=VERSION or model['feature_names']!=list(NAMES):
        raise ValueError('Wrong frozen fitted model')
    if vector['status']!='AVAILABLE_PREGAME_RUSH_VECTOR':
        return {'status':'NO_RUSH_CORRECTION_MISSING_FEATURES',
                 'margin_points':None,'not_deployed':True}
    coefficient=np.array(model['coefficients'],float)
    raw=float(np.dot(coefficient,vector['vector']))
    if not math.isfinite(raw):raise ValueError('Nonfinite correction')
    return {'status':'VALID_FROZEN_MODEL_PREGAME_RUSH_CORRECTION',
         'margin_points':round(float(np.clip(raw,-MAX_DELTA_POINTS,MAX_DELTA_POINTS)),6),
         'fitted_year':model['fit_year'],'training_game_count':model['n_training_games'],
         'not_deployed':True,'ATS_calibrated':False}
