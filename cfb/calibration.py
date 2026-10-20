"""CFB-only chronological margin calibration: train/validation/untouched holdout.

Not ATS calibration. Uses previously saved replay predictions; never fits on
holdout results. The 2026 released PBP snapshot is retrospectively captured,
so historical as-of status cannot be certified by chronological row filtering.
"""
from __future__ import annotations
import math
import numpy as np

FIELDS=('baseline_margin','build16_delta','sos_margin_feature','context_margin_feature')
PRIOR=np.array([0.,1.,0.,0.,0.],dtype=float)
STRENGTHS=(10.,30.,100.,300.,1000.)


def _build16_margin(r):
    """Use the single live challenger, with legacy fixture compatibility only."""
    if 'build16_candidate_margin' in r:
        value=r['build16_candidate_margin']
    elif 'epa_candidate_margin' in r:
        value=r['epa_candidate_margin']  # Legacy unit-test fixture only.
    else:
        raise ValueError('Missing paired Build 16 margin for calibration')
    if value is None or not math.isfinite(float(value)):
        raise ValueError('Invalid or unavailable paired challenger margin')
    return float(value)


def _design(rows):
    arr=[]
    for r in rows:
        base=float(r['baseline_margin']);epa=_build16_margin(r)-base
        vals=[base,epa,r.get('sos_margin_feature'),r.get('context_margin_feature')]
        if not all(math.isfinite(float(v)) for v in vals if v is not None):
            raise ValueError('Nonfinite CFB margin feature')
        arr.append([1.,base,epa,0. if vals[2] is None else float(vals[2]),
                    0. if vals[3] is None else float(vals[3])])
    return np.asarray(arr,dtype=float).reshape(-1,5)


def _cut(rows,fraction):
    ix=max(1,min(len(rows)-1,int(len(rows)*fraction)))
    while ix>1 and rows[ix-1]['kickoff']==rows[ix]['kickoff']:
        ix-=1
    return ix


def _metrics(y,yp):
    err=yp-y
    non_tie=y!=0
    return {'games':len(y),
            'mae':round(float(np.mean(np.abs(err))),4),
            'rmse':round(float(np.sqrt(np.mean(err**2))),4),
            'bias':round(float(np.mean(err)),4),
            'direction_accuracy_excluding_ties':round(float(np.mean(np.sign(yp[non_tie])==np.sign(y[non_tie]))),4)
                   if np.any(non_tie) else None}


def _ridge(X,y,strength):
    # Ridge around identity baseline; stronger penalties on exploratory extras.
    penalty=np.diag([1.,1.,3.,6.,6.])*strength
    return np.linalg.solve(X.T@X+penalty,X.T@y+penalty@PRIOR)


def calibrate(rows,min_games=100):
    rows=sorted(rows,key=lambda r:(r['kickoff'],str(r['game_id'])))
    ids=[str(r['game_id']) for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate CFB game in margin replay')
    if len(rows)<min_games:
        return {'status':'INSUFFICIENT_GAMES_FOR_CFB_TRAIN_VALIDATION_TEST',
                'games':len(rows),'minimum':min_games,
                'ats_validated':False}
    first=_cut(rows,.6);second=_cut(rows,.8)
    if first>=second or second>=len(rows) or min(first,second-first,len(rows)-second)<15:
        return {'status':'INSUFFICIENT_DISTINCT_KICKOFF_GROUPS',
                'games':len(rows),'ats_validated':False}
    X=_design(rows);y=np.asarray([float(r['actual_margin']) for r in rows]);
    if not np.isfinite(y).all():raise ValueError('Nonfinite CFB actual margin')
    Xtr,Xv,Xt=X[:first],X[first:second],X[second:]
    ytr,yv,yt=y[:first],y[first:second],y[second:]
    fits=[]
    for strength in STRENGTHS:
        w=_ridge(Xtr,ytr,strength)
        fits.append((float(np.mean(abs(Xv@w-yv))),strength,w))
    fits.sort(key=lambda x:(x[0],x[1]))
    val_baseline=float(np.mean(abs(Xv@PRIOR-yv)))
    # No post-hoc promotion if validation doesn't improve the baseline.
    promote=fits[0][0]+0.05<val_baseline
    weights=fits[0][2] if promote else PRIOR.copy()
    pred=Xt@weights
    result={'status':'CFB_MARGIN_RESEARCH_VALIDATION_ONLY_NOT_ATS_VALIDATED',
            'engine':'cfb.engine.v1','ats_validated':False,
            'train_games':first,'validation_games':second-first,
            'untouched_holdout_games':len(rows)-second,
            'train_last_kickoff':rows[first-1]['kickoff'],
            'validation_first_kickoff':rows[first]['kickoff'],
            'validation_last_kickoff':rows[second-1]['kickoff'],
            'holdout_first_kickoff':rows[second]['kickoff'],
            'holdout_last_kickoff':rows[-1]['kickoff'],
            'selected_ridge_strength':fits[0][1],
            'validation_selected_mae':round(fits[0][0],4),
            'validation_baseline_mae':round(val_baseline,4),
            'chosen_on_validation':bool(promote),
            'deployment_status':'NOT_APPROVED_UNTIL_INDEPENDENT_ASOF_AND_ATS_TESTS',
            'coefficients':dict(zip(('intercept',)+FIELDS,(round(float(v),8) for v in weights))),
            'holdout_selected':_metrics(yt,pred),
            'holdout_baseline':_metrics(yt,Xt@PRIOR),
            'holdout_build16_challenger':_metrics(yt,
                    np.asarray([_build16_margin(r) for r in rows[second:]])),
            'warnings':['RESEARCH ONLY: later-captured source is not historical as-of proof',
                        'Margin accuracy is not ATS accuracy',
                        'Do not tune coefficients using untouched holdout games',
                        'This file is not a calibrated ATS probability model']}
    return result


def predict_margin(sample,calibration):
    """Apply fitted CFB margin coefficients to a pregame feature row."""
    if calibration.get('status')!='CFB_MARGIN_RESEARCH_VALIDATION_ONLY_NOT_ATS_VALIDATED':
        raise ValueError('CFB margin calibration missing or incompatible')
    cs=calibration['coefficients']
    x=_design([sample])[0]
    return float(sum(xi*cs[k] for xi,k in zip(x,('intercept',)+FIELDS)))
