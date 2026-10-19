#!/usr/bin/env python3
"""Score frozen, pre-lock primary vs v0.9 candidate against LATER results.

A later data snapshot is used ONLY for realized scores; never to refit or
recompute the frozen prediction. Outputs no ATS record without verified line.
A single observation can start a ledger but cannot prove calibration.
"""
from __future__ import annotations
import argparse,json,hashlib,math
from pathlib import Path
import pandas as pd
from collect_extra_metrics import parse_stamp
from model import InputError


def evaluate(forecast_files,completed_schedule:pd.DataFrame,out=None):
    needed={'game_id','home_score','away_score','result','home_team','away_team','gameday','gametime'}
    missing=needed-set(completed_schedule)
    if missing:raise InputError('Missing final-score schedule columns: '+str(sorted(missing)))
    if completed_schedule.game_id.duplicated().any():raise InputError('Duplicate completed schedule games')
    outcomes=completed_schedule.set_index('game_id')
    # A game may have MANY scheduled snapshots. Predeclare the evaluator's
    # selection policy: use only the LATEST VALID PRE-LOCK forecast per game.
    # Do not inflate the sample by treating refreshed forecasts as new games.
    chosen={}
    for path in sorted(map(Path,forecast_files)):
        raw=path.read_bytes();data=json.loads(raw)
        if data.get('status')!='EXPLORATORY_CORE_PROJECTION_UNCALIBRATED':continue
        gid=str(data['game_id'])
        cutoff=parse_stamp(data['data_as_of_utc']);kickoff=parse_stamp(data['kickoff_utc'])
        if (kickoff-cutoff).total_seconds()<120*60:raise InputError(f'After-lock forecast excluded: {gid}')
        if gid not in outcomes.index:continue
        if gid in chosen and chosen[gid][0]==cutoff and chosen[gid][1]!=raw:
            raise InputError(f'Conflicting forecasts recorded at identical cutoff: {gid}')
        if gid not in chosen or cutoff>chosen[gid][0]:chosen[gid]=(cutoff,raw,data)
    rows=[]
    for gid,(cutoff,raw,data) in sorted(chosen.items()):
        if_candidate=data.get('mechanism_research',{}).get('candidate')
        if if_candidate is None:continue
        if if_candidate.get('promoted_to_primary') is not False:
            raise InputError(f'Candidate incorrectly promoted for {gid}')
        row=outcomes.loc[gid]
        if str(row['home_team'])!=str(data['home']) or str(row['away_team'])!=str(data['away']):
            raise InputError(f'Team mismatch: {gid}')
        # Verify forecast capture was >=2 hours before game and has no game outcome.
        try:h,a,result=float(row.home_score),float(row.away_score),float(row.result)
        except (ValueError,TypeError):continue
        if not all(math.isfinite(x) and x>=0 for x in (h,a)) or not math.isfinite(result):continue
        if abs((h-a)-result)>1e-6:
            raise InputError(f'Conflicting final score vs official result in schedule: {gid}')
        bmargin=float(data['projected_home_margin']);cmargin=float(if_candidate['mean_margin_home'])
        bm=abs(bmargin-(h-a));cm=abs(cmargin-(h-a))
        rows.append({'game_id':gid,'frozen_file_sha256':hashlib.sha256(raw).hexdigest(),
            'prediction_as_of_utc':data['data_as_of_utc'], 'actual_home_margin':h-a,
            'baseline_margin_home':bmargin,'candidate_margin_home':cmargin,
            'baseline_abs_margin_error':round(bm,4),'candidate_abs_margin_error':round(cm,4),
            'baseline_total_abs_error':round(abs(float(data['score_home'])+float(data['score_away'])-h-a),4),
            'candidate_total_abs_error':round(abs(float(if_candidate['score_home'])+float(if_candidate['score_away'])-h-a),4),
            'ATS_side_evaluated':False,'market_quote_status':'NO_VERIFIED_FROZEN_SIGNED_LINE'})
    def avg(key):return round(sum(r[key] for r in rows)/len(rows),4) if rows else None
    result={'schema':'locked_future_model_comparison_v0.9','evaluation_selection_policy':'LATEST_CAPTURE_AT_LEAST_120_MINUTES_PRE_KICKOFF',
        'distinct_eligible_game_ids':len(chosen),'completed_games':len(rows),
        'base_mae_margin':avg('baseline_abs_margin_error'),
        'candidate_mae_margin':avg('candidate_abs_margin_error'),
        'base_mae_total':avg('baseline_total_abs_error'),
        'candidate_mae_total':avg('candidate_total_abs_error'),
        'candidate_better_margin_games':sum(r['candidate_abs_margin_error']<r['baseline_abs_margin_error'] for r in rows),
        'ats_record':'NOT_MEASURED_NO_VALIDATED_PRICED_SIGNED_LINES',
        'model_promoted':False,'conclusion':'No reliability inference from a small, selected sample. Use chronological out-of-sample validation.',
        'games':rows}
    if out:
        path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--forecasts-dir',required=True,type=Path)
    p.add_argument('--final-schedule',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    fs=[x for x in a.forecasts_dir.rglob('*.json') if x.name!='index.json']
    print(json.dumps(evaluate(fs,pd.read_csv(a.final_schedule),a.output),indent=2))
