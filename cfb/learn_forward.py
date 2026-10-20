"""CFB genuine-forward calibration research, triggered after graded games.

Only archived-before-kickoff predictions enter; model is NEVER promoted here.
Fit on earliest truly prospective finals, select on later validation, score an
untouched final time block once. Do not repeatedly tune to the same test set.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from cfb.market import _date


def observations(root):
    earliest={}
    for p in sorted(Path(root).glob('*.json')):
        data=json.loads(p.read_text())
        if data.get('schema')!='cfb_forward_research_audit_v1':continue
        stamp=_date(data['free_schedule_observed_at_utc'],'score source observed')
        for row in data.get('scored_games',[]):
            gid=str(row['game_id']); kickoff=_date(row['kickoff_utc'],'game kickoff')
            frozen=_date(row['forecast_generated_at_utc'],'frozen')
            if not frozen<kickoff<stamp:continue
            try: pred=float(row['predicted_home_margin']);actual=float(row['actual_home_margin'])
            except (ValueError,TypeError,KeyError):continue
            if not math.isfinite(pred) or not math.isfinite(actual):continue
            candidate={'game_id':gid,'kickoff_utc':kickoff.isoformat(),
                       'final_source_captured_at_utc':stamp.isoformat(),
                       'projected_margin':pred,'actual_margin':actual}
            if gid not in earliest or stamp<_date(earliest[gid]['final_source_captured_at_utc'],'first final source'):
                earliest[gid]=candidate
    return sorted(earliest.values(),key=lambda z:(z['kickoff_utc'],z['game_id']))


def _splits(rows):
    """No same-kickoff weekend straddles partitions; use UTC date as cluster."""
    days=[]
    for r in rows:
        key=r['kickoff_utc'][:10]
        if not days or days[-1][0]!=key:days.append((key,[]))
        days[-1][1].append(r)
    if len(days)<9:return None
    n=len(rows);train=[];val=[];test=[]
    for day,group in days:
        if len(train)<.6*n:train.extend(group)
        elif len(val)<.2*n:val.extend(group)
        else:test.extend(group)
    if min(len(train),len(val),len(test))<20:return None
    return train,val,test


def _err(rows, fn):
    y=np.array([r['actual_margin'] for r in rows],dtype=float)
    yp=np.array([fn(r['projected_margin']) for r in rows],dtype=float)
    return {'n':len(rows),'mae':round(float(np.abs(yp-y).mean()),4),
            'bias':round(float((yp-y).mean()),4)}


def investigate(rows,min_games=120,gate_points=.5):
    """Train-only fits, validation-only selection; holdout never tunes model."""
    dedup={r['game_id']:r for r in rows}
    rows=sorted(dedup.values(),key=lambda r:(r['kickoff_utc'],r['game_id']))
    out={'status':'INSUFFICIENT_GENUINE_FORWARD_GAMES','completed_forward_games':len(rows),
         'minimum_forward_games':min_games,'deploy_automatically':False,
         'selection':'requires >=0.5-point validation margin MAE gain; ATS still separately unvalidated'}
    if len(rows)<min_games:return out
    parts=_splits(rows)
    if parts is None:return {**out,'reason':'Insufficient separated chronological game-date groups'}
    train,val,test=parts
    pred=np.array([r['projected_margin'] for r in train]);actual=np.array([r['actual_margin'] for r in train]);err=actual-pred
    bias=float(np.clip(err.mean(),-7,7))
    ridge=1000.;mu=float(pred.mean())
    slope=float(np.clip(1+((pred-mu)*err).sum()/(((pred-mu)**2).sum()+ridge),.65,1.35))
    funcs={'baseline':lambda z:z,'bias_adjusted':lambda z:z+bias,
           'shrink_expand':lambda z:mu+(z-mu)*slope+bias}
    val_scores={k:_err(val,f) for k,f in funcs.items()}
    base=val_scores['baseline']['mae']
    best=min((k for k in funcs if k!='baseline'),key=lambda k:val_scores[k]['mae'])
    improvement=round(base-val_scores[best]['mae'],4)
    eligible=improvement>=gate_points
    chosen=best if eligible else 'baseline'
    out.update({'status':'CANDIDATE_REQUIRES_MANUAL_REVIEW' if eligible else 'NO_VALIDATED_IMPROVEMENT',
                'split_games':{'train':len(train),'validation':len(val),'untouched_test':len(test)},
                'train_coefficients_research_only':{'bias':round(bias,6),'slope':round(slope,6),'train_margin_center':round(mu,6)},
                'validation':val_scores,'best_candidate':best,'validation_mae_improvement':improvement,
                'passes_validation_gate':eligible,
                'untouched_test_baseline':_err(test,funcs['baseline']),
                'untouched_test_selected':_err(test,funcs[chosen]),
                'selection':f'Validation only; >= {gate_points:.2f} points MAE improvement',
                'holdout_policy':'Never use this test split for later tuning; wait for new frozen games',
                'ats_proven_edge':False,'deployment_status':'REQUIRES_EXPLICIT_APPROVAL_AND_FUTURE_ATS_VALIDATION'})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grades',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();report=investigate(observations(a.grades))
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise ValueError('Refusing to overwrite forward learning report')
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
