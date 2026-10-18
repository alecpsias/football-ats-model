#!/usr/bin/env python3
"""Chronological, non-leaking replay of pregame snapshots and ATS results.

Input JSONL rows: {"game": <complete pregame input>,
                   "actual_home_score": <integer>, "actual_away_score": <integer>}
Train first 70% on chronological order; report later 30% as never-seen holdout.
The calibration is a probability mapping, not evidence of profitability by itself.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from model import InputError, VERSION, logit, run, sigmoid, utc


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    if len(p) == 0:
        return {"games":0}
    pc = np.clip(p, 1e-8, 1-1e-8)
    return {
        "games":len(p),
        "brier":round(float(np.mean((pc-y)**2)),5),
        "log_loss":round(float(np.mean(-y*np.log(pc)-(1-y)*np.log(1-pc))),5),
        "ats_directional_accuracy":round(float(np.mean((pc >= .5)==y)),5),
        "mean_predicted_home_cover":round(float(np.mean(pc)),5),
        "actual_home_cover_rate":round(float(np.mean(y)),5),
    }


def fit_sigmoid(x: np.ndarray, y: np.ndarray) -> tuple[float,float]:
    # Regularized 2-parameter logistic calibration; x is log-odds of RAW
    # non-push home cover probability; weak prior toward identity mapping.
    X = np.column_stack((np.ones(len(x)),x))
    w=np.array([0.0,1.0], dtype=float)
    ref=np.array([0.0,1.0], dtype=float)
    ridge=np.diag([2.0,5.0])
    for _ in range(100):
        z=np.clip(X@w,-20,20)
        p=1/(1+np.exp(-z))
        grad=X.T@(p-y)+ridge@(w-ref)
        hess=X.T@((p*(1-p))[:,None]*X)+ridge
        update=np.linalg.solve(hess,grad)
        w-=update
        if np.max(np.abs(update))<1e-7:
            break
    return float(w[0]),float(w[1])


def replay(path: Path, sims: int, seed: int, allow_synthetic_for_software_tests: bool=False) -> dict:
    rows=[]
    for rowno, line in enumerate(path.read_text(encoding='utf8').splitlines(),1):
        if not line.strip():
            continue
        try:
            row=json.loads(line)
            g=row['game']
            if not allow_synthetic_for_software_tests and any('SYNTHETIC' in g[t]['stat_source'].upper() for t in ('home','away')):
                raise InputError('Synthetic inputs cannot be used as historical calibration evidence')
            home_pts=row['actual_home_score']
            away_pts=row['actual_away_score']
            if isinstance(home_pts,bool) or not isinstance(home_pts,int) or home_pts<0 or home_pts>150:
                raise InputError('actual_home_score must be integer 0..150')
            if isinstance(away_pts,bool) or not isinstance(away_pts,int) or away_pts<0 or away_pts>150:
                raise InputError('actual_away_score must be integer 0..150')
            output=run(g,runs=sims,seed=seed+rowno,perform_stress=False)
            margin=home_pts-away_pts+g['market']['spread_home']
            if abs(margin)<1e-9:
                continue
            result=output['simulation']
            h=result['home_cover_probability_raw']
            a=result['away_cover_probability_raw']
            raw_conditional=h/(h+a)
            rows.append({'id':g['id'],'league':g['league'],'kickoff':utc(g['kickoff_utc'],'kickoff_utc'),
                         'raw_conditional_p':raw_conditional,'result':int(margin>0),
                         'year':utc(g['kickoff_utc'],'kickoff_utc').year})
        except (KeyError,ValueError,TypeError) as exc:
            raise InputError(f'Line {rowno}: {exc}') from exc
    rows.sort(key=lambda r:r['kickoff'])
    if len({r['id'] for r in rows})!=len(rows):
        raise InputError('Duplicate game IDs in backtest — cannot calibrate')
    if len({r['league'] for r in rows})!=1:
        raise InputError('Do not mix NFL and CFB in a single backtest/calibration')
    if len(rows)<2:
        raise InputError('At least two graded historical games required')
    split=max(1,int(len(rows)*.7))
    if split>=len(rows): split=len(rows)-1
    # Never divide games with the exact same kickoff across the train/test boundary.
    while split>1 and rows[split-1]['kickoff']==rows[split]['kickoff']:
        split-=1
    train,test=rows[:split],rows[split:]
    to_p=lambda z:np.array([r['raw_conditional_p'] for r in z])
    to_y=lambda z:np.array([r['result'] for r in z])
    ptr,ytr=to_p(train),to_y(train)
    pte,yte=to_p(test),to_y(test)
    alpha,beta=fit_sigmoid(np.array([logit(x) for x in ptr]),ytr)
    calibrated=np.array([sigmoid(alpha+beta*logit(x)) for x in pte])
    base=metrics(pte,yte)
    calibrated_metrics=metrics(calibrated,yte)
    enough=len(train)>=100 and len(test)>=40
    better=calibrated_metrics['brier']<=base['brier'] and calibrated_metrics['brier']<.25 and beta>0
    status=('HOLDOUT_CHECK_PASSED' if enough and better else 'RESEARCH_ONLY_INSUFFICIENT_OR_UNSUCCESSFUL_VALIDATION')
    return {
        'model_version':VERSION, 'league':rows[0]['league'], 'method':'chronological 70/30 train-holdout; ridge logistic probability mapping',
        'games_excluding_pushes':len(rows), 'train_games':len(train),'holdout_games':len(test),
        'train_last_kickoff_utc':train[-1]['kickoff'].isoformat(),
        'holdout_first_kickoff_utc':test[0]['kickoff'].isoformat(),
        'holdout_last_kickoff_utc':test[-1]['kickoff'].isoformat(),
        'calibration_alpha':round(alpha,7), 'calibration_beta':round(beta,7),
        'holdout_raw':base,'holdout_calibrated':calibrated_metrics,'holdout_baseline_p50':metrics(np.full(len(test),.5),yte),
        'validation_status':status,
        'cautions':[
            'No pregame data later than kickoff is allowed; still independently audit stats collection for leakage.',
            'Do NOT claim profitability from a small holdout or from a Brier score alone.',
            'Recalibrate separately for NFL and CFB, and ideally test on a different season.',
            'Output only certifies these specific historical replay results, not future accuracy.'
        ],
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history_jsonl',type=Path)
    parser.add_argument('--runs',type=int,default=4000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path,default=Path('backtest_report.json'))
    args=parser.parse_args()
    result=replay(args.history_jsonl,args.runs,args.seed)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(f"Backtest: {result['games_excluding_pushes']} games; {result['validation_status']}")
    print('Holdout raw:',result['holdout_raw'])
    print('Holdout mapped:',result['holdout_calibrated'])
    print('Saved:',args.output)

if __name__=='__main__': main()
