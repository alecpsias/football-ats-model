"""Evaluate a single pre-frozen 2023-trained CFB rushing correction, no refit.

Reports transparent paired margin error/uncertainty, severe-matchup subsets,
probability calibration and empirical status. Does NOT authorize live ATS picks.
"""
import argparse,json
from pathlib import Path
import numpy as np

def group(rows):
    n=len(rows)
    result={'paired_games':n}
    if not n:return result
    actual=np.asarray([r['actual_home_margin'] for r in rows],float)
    abs_base=np.array([abs(r['reference']['margin']-r['actual_home_margin']) for r in rows])
    abs_new=np.array([abs(r['learned_candidate']['margin']-r['actual_home_margin']) for r in rows])
    delta=abs_base-abs_new
    rng=np.random.default_rng(20261022)
    ix=rng.integers(0,n,size=(5000,n))
    bs=delta[ix].mean(axis=1)
    for label in ('reference','learned_candidate'):
        data=[r[label] for r in rows]
        pred=np.array([x['margin'] for x in data],float)
        tot=np.array([x['total'] for x in data],float)
        probs=np.array([x['home_win_probability_raw'] for x in data],float)
        q=np.array([x['margin_p10_p50_p90'] for x in data],float)
        result[label]={'margin_mae':round(float(np.mean(abs(pred-actual))),5),
          'margin_bias':round(float(np.mean(pred-actual)),5),
          'total_mae':round(float(np.mean(abs(tot-np.array([r['actual_total'] for r in rows])))),5),
          'winner_brier':round(float(np.mean((probs-(actual>0))**2)),6),
          'margin80_coverage':round(float(np.mean((actual>=q[:,0])&(actual<=q[:,2]))),5),
          'winner_direction':round(float(np.mean(np.sign(pred)==np.sign(actual))),5)}
    result['mae_gain_reference_minus_learned']=round(float(delta.mean()),5)
    result['paired_bootstrap95_gain']=[round(float(x),5) for x in np.quantile(bs,[.025,.975])]
    result['mean_abs_margin_shift']=round(float(np.mean([
         abs(r['learned_candidate']['margin']-r['reference']['margin']) for r in rows])),5)
    return result

def summarize(files):
    seasons={}
    for file in files:
        d=json.loads(Path(file).read_text())
        year=d['source_season']
        frozen=d['fitted_model_year']
        if frozen!=2023 or year<=2023:raise ValueError('2023 model must only test later seasons')
        rows=[r for r in d['replays'] if 'learned_candidate' in r]
        if any(r['year']!=year for r in rows):raise ValueError('Wrong season')
        seasons[str(year)]={'all':group(rows),
          'severe_ex_ante':group([r for r in rows if r['severe_screen_ex_ante']]),
          'ordinary':group([r for r in rows if not r['severe_screen_ex_ante']]),
          'home_neg_run_advantage':group([r for r in rows if r['home_negative_rate_excess'] is not None
                  and r['home_negative_rate_excess']<-.03]),
          'away_neg_run_advantage':group([r for r in rows if r['away_negative_rate_excess'] is not None
                  and r['away_negative_rate_excess']<-.03])}
    if len(seasons)<2:raise ValueError('Require multiple separate validation seasons')
    test=[json.loads(Path(f).read_text()) for f in files]
    pooled=[r for d in test for r in d['replays'] if 'learned_candidate' in r]
    allstats=group(pooled)
    if len(pooled)<300:raise ValueError('Minimum pooled holdout 300')
    severe=group([r for r in pooled if r['severe_screen_ex_ante']])
    # Conditions chosen before reviewing combined heldout results.
    evidence={
      '>=300_paired_games':len(pooled)>=300,
      'mean_margin_MAE_gain_at_least_0.2':allstats['mae_gain_reference_minus_learned']>=.2,
      'paired_95pct_CI_lower_above_zero':allstats['paired_bootstrap95_gain'][0]>0,
      'severe_group_at_least_100_games':severe['paired_games']>=100,
      'severe_group_mean_gain_positive':severe['mae_gain_reference_minus_learned']>0,
      'Brier_not_worse':allstats['learned_candidate']['winner_brier']<=allstats['reference']['winner_brier'],
      'margin_coverage_not_down_over_two_pct_points':allstats['learned_candidate']['margin80_coverage']>=allstats['reference']['margin80_coverage']-.02,
      'verified_future_sportsbook_quotes_and_ATS_calibration':False}
    return {'schema':'cfb_failure1_fitted_rushing_holdout_decision_v1',
        'trained_only_on_2023':True,
        'validation_seasons':seasons,'pooled':allstats,'severe_pooled':severe,
        'evidence_gates':evidence,
        'all_validation_gates_pass':all(evidence.values()),
        'promote_to_standard_build16':False,
        'not_automatically_deployed':True,
        'limitations':['2024+ sources are retrospectively captured; not original point-in-time publication proof',
          'No verified pregame historical sportsbook markets; margin error is not ATS cover probability',
          'V2 high severity definitions were explored on 2026 feature distributions',
          'Multiple research ideas examined; holdout exploratory, prospective ATS still required']}
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--results',nargs='+',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    r=summarize(a.results)
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Cannot overwrite validation result')
    target.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
