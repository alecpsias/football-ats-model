"""CFB-only, chronological error forensics and feature ablation (research).

This does not use actual sportsbook spreads, does not prove historical source
publication times, and never chooses predictors using held-out results.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from cfb.calibration import _cut

FEATURES = ('build16_delta', 'sos_margin_feature', 'context_margin_feature',
            'sos_pass_epa_differential', 'sos_rush_epa_differential')


def _metric(y, p):
    e=np.asarray(p,float)-np.asarray(y,float)
    return {'games':len(e),'mae':round(float(np.mean(np.abs(e))),4),
            'rmse':round(float(np.sqrt(np.mean(e*e))),4),
            'bias':round(float(np.mean(e)),4)}


def _feature(r, name):
    try:
        value = ((float(r['build16_candidate_margin'] if 'build16_candidate_margin' in r
                          else r['epa_candidate_margin'])-float(r['baseline_margin']))
                 if name == 'build16_delta' else r.get(name))
        if value is None or not math.isfinite(float(value)):
            return None
        return float(value)
    except (ValueError,TypeError,KeyError):
        return None


def _coefficients(rows, feature, penalty=1000.0):
    """Fit base correction and a single centered feature, on TRAIN ONLY.

    No imputation of missing feature values as evidence of observed zero.
    When applying, missing feature -> retain baseline prediction.
    """
    train=[r for r in rows if _feature(r,feature) is not None]
    if len(train)<30 or len(train)/len(rows)<.60:
        return None
    x=np.asarray([_feature(r,feature) for r in train],float)
    base=np.asarray([float(r['baseline_margin']) for r in train])
    actual=np.asarray([float(r['actual_margin']) for r in train])
    mu=float(np.mean(x));sd=float(np.std(x))
    if sd<1e-9:return None
    z=np.clip((x-mu)/sd,-5,5)
    # Bounded, ridge-shrunk incremental margin correction only; no intercept.
    coefficient=float((z@(actual-base))/(z@z+penalty))
    return {'feature':feature,'center':mu,'scale':sd,
            'coefficient':coefficient,'train_coverage':len(train)/len(rows)}


def _predict(rows, fit):
    baseline=np.asarray([float(r['baseline_margin']) for r in rows])
    if fit is None:return baseline
    adj=np.asarray([0. if _feature(r,fit['feature']) is None else
        np.clip((_feature(r,fit['feature'])-fit['center'])/fit['scale'],-5,5)
        for r in rows])
    return baseline+np.clip(fit['coefficient']*adj,-6.,6.)


def _profile(rows):
    if not rows:return {'games':0}
    pred=np.asarray([float(r['baseline_margin']) for r in rows])
    actual=np.asarray([float(r['actual_margin']) for r in rows])
    e=pred-actual
    bins={'close_predicted_abs_margin_lt_7':np.abs(pred)<7,
          'projected_favorite_7_to_14':(np.abs(pred)>=7)&(np.abs(pred)<14),
          'projected_favorite_14_plus':np.abs(pred)>=14,
          'actual_blowout_abs_margin_21_plus':np.abs(actual)>=21,
          'missed_winner':(pred*actual<0),
          'home_undervalued_by_14_plus':e<=-14,
          'home_overvalued_by_14_plus':e>=14}
    out={}
    for name,mask in bins.items():
        if np.any(mask):out[name]=_metric(actual[mask],pred[mask])
        else:out[name]={'games':0}
    return out


def investigate(rows, min_games=100):
    rows=sorted(rows,key=lambda r:(r['kickoff'],str(r['game_id'])))
    ids=[str(r['game_id']) for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate CFB replay game IDs')
    if len(rows)<min_games:
        return {'status':'INSUFFICIENT_GAMES','games':len(rows),
                'minimum':min_games,'ats_validated':False}
    first,second=_cut(rows,.6),_cut(rows,.8)
    if min(first,second-first,len(rows)-second)<15:
        return {'status':'INSUFFICIENT_DISTINCT_KICKOFF_GROUPS','ats_validated':False}
    train,validation,test=rows[:first],rows[first:second],rows[second:]
    yval=np.asarray([float(r['actual_margin']) for r in validation])
    ytest=np.asarray([float(r['actual_margin']) for r in test])
    valbase=_metric(yval,_predict(validation,None))
    ranking=[]
    for feature in FEATURES:
        fit=_coefficients(train,feature)
        if fit is None:
            ranking.append({'feature':feature,'status':'SKIPPED_INSUFFICIENT_TRAIN_COVERAGE'})
            continue
        pred=_predict(validation,fit)
        metric=_metric(yval,pred)
        ranking.append({'feature':feature,'status':'EVALUATED_ON_VALIDATION',
                        'train_coverage':round(fit['train_coverage'],4),
                        'validation_coverage':round(sum(_feature(r,feature) is not None
                            for r in validation)/len(validation),4),
                        'validation_mae':metric['mae'],
                        'validation_improvement_vs_baseline':round(valbase['mae']-metric['mae'],4),
                        'fit_coefficient':round(fit['coefficient'],5)})
    valid=[r for r in ranking if r['status']=='EVALUATED_ON_VALIDATION']
    valid.sort(key=lambda r:(r['validation_mae'],r['feature']))
    # One model choice. Require 0.25 pt absolute validation improvement to
    # mitigate multiple comparisons / noise. Freeze before reading test outcomes.
    champion=valid[0]['feature'] if valid and valid[0]['validation_mae']<=valbase['mae']-.25 else None
    model=_coefficients(train,champion) if champion else None
    ptest=_predict(test,model)
    top=sorted(test,key=lambda r:abs(float(r['baseline_margin'])-float(r['actual_margin'])),reverse=True)[:15]
    worst=[{'game_id':str(r['game_id']),'kickoff':r['kickoff'],'home':r['home'],'away':r['away'],
            'actual_margin':r['actual_margin'],'predicted_baseline_margin':r['baseline_margin'],
            'absolute_error':round(abs(r['actual_margin']-r['baseline_margin']),4),
            'predicted_winner_correct':(r['baseline_margin']*r['actual_margin']>0),
            'sos_feature_available':_feature(r,'sos_margin_feature') is not None}
           for r in top]
    coverage={f:{'train':round(sum(_feature(r,f) is not None for r in train)/len(train),4),
                 'validation':round(sum(_feature(r,f) is not None for r in validation)/len(validation),4),
                 'test':round(sum(_feature(r,f) is not None for r in test)/len(test),4)} for f in FEATURES}
    return {'status':'CFB_RETROSPECTIVE_DIAGNOSTICS_NO_ATS_VALIDATION',
            'train_games':len(train),'validation_games':len(validation),'test_games':len(test),
            'train_ends':train[-1]['kickoff'],'validation_ends':validation[-1]['kickoff'],
            'test_starts':test[0]['kickoff'],'baseline_validation':valbase,
            'candidate_validation_only':ranking,'feature_coverage':coverage,
            'champion_selected_using_validation_only':champion,
            'selection_rule':'Univariate ridge; require >=0.25 points validation MAE reduction',
            'untouched_test_baseline':_metric(ytest,_predict(test,None)),
            'untouched_test_selected':_metric(ytest,ptest),
            'test_error_buckets_after_model_selection':_profile(test),
            'worst_15_test_misses_post_selection_diagnostics_only':worst,
            'ats_validated':False,
            'limitations':['Holdout outcomes used only for post-selection scoring/diagnostics',
                           'Do not repeatedly tune on this holdout; freeze new candidates for future data',
                           'Snapshot was collected after these games; no historical publication-time proof',
                           'Margin-direction accuracy is not ATS cover accuracy',
                           'Game-level largest misses are descriptive, not proof of root cause']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();data=json.loads(Path(a.replay).read_text())
    result=investigate(data)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('worst_15_test_misses_post_selection_diagnostics_only','test_error_buckets_after_model_selection')},indent=2))
if __name__=='__main__':main()
