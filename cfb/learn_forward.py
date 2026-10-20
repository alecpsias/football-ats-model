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
            z=row.get('build16_challenger')
            if isinstance(z,dict) and z.get('status')=='ORIGINAL_FROZEN_BUILD16_CHALLENGER':
                v=z.get('projected_home_margin')
                if isinstance(v,(float,int)) and math.isfinite(v):
                    candidate['build16_projected_margin']=float(v)
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
    out['build16_paired_games']=sum('build16_projected_margin' in r for r in rows)
    out['build16_status']='AWAITING_GENUINE_FORWARD_COMPARISON'
    if len(rows)<min_games:return out
    parts=_splits(rows)
    if parts is None:return {**out,'reason':'Insufficient separated chronological game-date groups'}
    train,val,test=parts
    # Only the original-frozen Build16 candidate may compete with baseline.
    valpaired=[r for r in val if 'build16_projected_margin' in r]
    testpaired=[r for r in test if 'build16_projected_margin' in r]
    if len(valpaired)>=20 and len(testpaired)>=20:
        def _mae(group,k):
            return round(sum(abs(r[k]-r['actual_margin']) for r in group)/len(group),4)
        b=_mae(valpaired,'projected_margin');c=_mae(valpaired,'build16_projected_margin')
        out['build16_validation']={
            'paired_validation_games':len(valpaired),
            'paired_untouched_test_games':len(testpaired),
            'baseline_validation_mae':b,'build16_validation_mae':c,
            'validation_gain':round(b-c,4),'passes_gate':(b-c)>=gate_points,
            'untouched_test_baseline_mae':_mae(testpaired,'projected_margin'),
            'untouched_test_build16_mae':_mae(testpaired,'build16_projected_margin'),
            'auto_promoted':False,'status':'REQUIRES_MANUAL_REVIEW_NO_ATS_VALIDATION'}
        out['build16_status']=('VALIDATION_GATE_PASSED_AWAITING_MANUAL_REVIEW'
            if b-c>=gate_points else 'NOT_BETTER_ON_VALIDATION')
    else:
        out['build16_status']='INSUFFICIENT_PAIRED_VALIDATION_AND_TEST_GAMES'
    out['split_games']={'train':len(train),'validation':len(val),'untouched_test':len(test)}
    out['selection']='Original baseline versus frozen Build 16 only; validation precedes untouched test'
    out['status']=('BUILD16_REQUIRES_MANUAL_REVIEW' if out['build16_status']=='VALIDATION_GATE_PASSED_AWAITING_MANUAL_REVIEW'
                   else 'NO_VALIDATED_BUILD16_IMPROVEMENT')
    out['ats_proven_edge']=False
    out['deployment_status']='REQUIRES_EXPLICIT_APPROVAL_AND_FUTURE_ATS_VALIDATION'
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
