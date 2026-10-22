"""Prespecified CFB rush->drive scoring promotion gate (no automatic merge).

Do not promote a coefficient merely because it has same-seed Monte Carlo
output. Require broad margin gain, severe-run gain, honest clock coverage,
consistent mechanistic sign, and *later* real observed ATS lines.
"""
import argparse,json
from pathlib import Path
from cfb.rush_possession_study import metric

def judge(model,files):
    if model['trained_year']!=2023:raise ValueError('Training source must be 2023')
    rows=[];years={}
    for f in files:
        d=json.loads(Path(f).read_text())
        year=d['season']
        if year not in (2024,2025,2026) or d['training_only']:
            raise ValueError('Only heldout later seasons allowed')
        records=[z for z in d['study_rows'] if z['candidate'] is not None]
        if any(z['year']!=year for z in records):raise ValueError('Year leakage')
        years[str(year)]={'paired_games':len(records),
          'all':metric(records),
          'clock_verified':metric([z for z in records if z['signal']['clock_both_sides_qualified']]),
          'severe_matchup':metric([z for z in records if max(
                  abs(z['signal']['home_continuation_delta']),
                  abs(z['signal']['away_continuation_delta']))>=.02])}
        rows.extend(records)
    if len(years)!=3:raise ValueError('Require all three frozen holdouts')
    severe=[z for z in rows if max(
                  abs(z['signal']['home_continuation_delta']),
                  abs(z['signal']['away_continuation_delta']))>=.02]
    pooled=metric(rows);cohort=metric(severe)
    gates={'three_later_seasons':len(years)==3,
           'minimum_300_paired_games':len(rows)>=300,
           'positive_physical_drive_continuation_scoring_sign':model['continuation_points_per_extra_drive']>0,
           'all_slate_margin_gain_at_least_0_15':pooled.get('margin_MAE_gain',-999)>.15,
           'paired_95pct_gain_lower_above_zero':(pooled.get('paired_bootstrap95') or [-1])[0]>0,
           'severe_group_at_least_75':len(severe)>=75,
           'severe_group_margin_gain_positive':cohort.get('margin_MAE_gain',-999)>0,
           'no_more_than_0_2_mae_worse_any_year':all(
               x['all'].get('margin_MAE_gain',-999)>=-.2 for x in years.values()),
           'winner_brier_not_worse':pooled.get('candidate_winner_brier',999)<pooled.get('reference_winner_brier',-1),
           '80pct_margin_coverage_not_down_two_points':pooled.get('candidate_margin80_coverage',0)>=pooled.get('reference_margin80_coverage',1)-.02,
           'genuine_forward_quoted_ATS_outcomes_verified':False}
    return {'schema':'cfb_failure1_rush_possession_mechanism_three_holdout_decision_v1',
       'fitted_parameter':model,
       'years':years,'all_holdouts':pooled,'severe_rush_group':cohort,
       'clock_verified_group':metric([z for z in rows if z['signal']['clock_both_sides_qualified']]),
       'promotion_criteria':gates,'eligible_to_promote':all(gates.values()),
       'deploy':False,'current_primary_Build16_unchanged':True,
       'real_pregame_spread_graded_games':0,
       'warning':'Retrospective data release; no certifiable sportsbook ATS calibration; no result-derived re-tuning allowed',
       'research_nonpromoted':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frozen-fit',required=True)
    p.add_argument('--holdouts',nargs=3,required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    result=judge(json.loads(Path(a.frozen_fit).read_text()),a.holdouts)
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Never overwrite decision receipt')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
