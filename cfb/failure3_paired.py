"""Failure 3 paired 5,000-run real-data evaluation of ONE evolving Build16.

Original Build16+Failure1+Failure2 vs the SAME challenger with Failure3
switched on; neither baseline nor other mechanisms change. Historical
PBP captured retrospectively is not certified point-in-time ATS data.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from cfb.failure2_paired import cohort
from cfb.mechanism_build16 import simulate_build16
from cfb.failure1_learned_shadow import correction as frozen_failure1_correction

MOTIVATION_GAME='401871070'

def evaluate(snapshot,*,max_games=160,runs=5000):
    if type(runs) is not int or runs!=5000:
        raise ValueError('Both candidates must use exactly 5000 simulations')
    games,screen=cohort(snapshot,max_games=max_games+1)
    chosen=[g for g in games if g['game_id']!=MOTIVATION_GAME][:max_games]
    paired=[]
    for game in chosen:
        home,away=game['home_id'],game['away_id']
        h=game['historical']
        seed=int(game['game_id'])%(2**32)
        rush=frozen_failure1_correction(home,away,h)['margin_delta_points']
        shared=dict(runs=5000,seed=seed,failure1_fit_margin_delta=rush,
                    failure2_attribution=True)
        before=simulate_build16(home,away,h,game['neutral'],failure3_sustainability=False,**shared)
        after=simulate_build16(home,away,h,game['neutral'],failure3_sustainability=True,**shared)
        a,b=before.get('simulation'),after.get('simulation')
        if a is None or b is None:continue
        if a['simulations']!=5000 or b['simulations']!=5000 or a['seed']!=b['seed']:
            raise AssertionError('Paired simulation protocol violated')
        observed=game['actual_margin']
        evidence=after['failure3_sustainability']
        row={'game_id':game['game_id'],'home':game['home'],'away':game['away'],
             'kickoff':game['kickoff'],'actual_margin':observed,
             'seed':seed,'simulations_each':5000,
             'failure1_delta_held_constant':rush,
             'failure2_status_held_constant':after['failure2_scoring_attribution']['status'],
             'failure3_status':evidence['status'],
             'margin_before':a['mean_margin_home'],'margin_after':b['mean_margin_home'],
             'total_before':a['mean_total'],'total_after':b['mean_total'],
             'margin_interval_before':a['margin_p10_p50_p90'],
             'margin_interval_after':b['margin_p10_p50_p90'],
             'ppd_global_subtraction_applied':evidence.get('legacy_ppd_subtraction_applied'),
             'possession_process_status':evidence.get('possession_process_status'),
             'explosive_td_home_per_game':b.get('failure3_explosive_touchdowns_home'),
             'explosive_td_away_per_game':b.get('failure3_explosive_touchdowns_away'),
             'chain_scores_home_per_game':b.get('failure3_chain_scoring_drives_home'),
             'chain_scores_away_per_game':b.get('failure3_chain_scoring_drives_away'),
             'simulated_possessions_home':b.get('failure3_possessions_home'),
             'simulated_possessions_away':b.get('failure3_possessions_away'),
             'clock_method':b.get('failure3_clock_model'),
             'source_clock_verified':b.get('failure3_source_clock_verified'),
             'clock_audit':evidence.get('clock_audit'),
             'mean_elapsed_source_seconds':b.get('failure3_mean_elapsed_seconds'),
             'source_clock_status':evidence.get('possession_process_status'),
             'mae_before':round(abs(a['mean_margin_home']-observed),5),
             'mae_after':round(abs(b['mean_margin_home']-observed),5),
             'paired_gain':round(abs(a['mean_margin_home']-observed)-
                                 abs(b['mean_margin_home']-observed),5),
             'margin_shift':round(b['mean_margin_home']-a['mean_margin_home'],5)}
        paired.append(row)
    if not paired:raise RuntimeError('No paired eligible completed matchups')
    active=[r for r in paired if r['source_clock_verified'] is True]
    changed=[r for r in active if abs(r['margin_shift'])>.001]
    def summary(rows):
        if not rows:return {'games':0,'mae_before':None,'mae_after':None}
        gains=np.array([r['paired_gain'] for r in rows],float)
        rng=np.random.default_rng(20261022)
        sample=rng.integers(len(rows),size=(3000,len(rows)))
        boot=gains[sample].mean(axis=1)
        return {'games':len(rows),
          'mae_before':round(float(np.mean([r['mae_before'] for r in rows])),5),
          'mae_after':round(float(np.mean([r['mae_after'] for r in rows])),5),
          'mean_mae_gain':round(float(gains.mean()),5),
          'gain_bootstrap95':[round(float(x),5) for x in np.quantile(boot,[.025,.975])],
          'mean_absolute_margin_change':round(float(np.mean(
             [abs(r['margin_shift']) for r in rows])),5)}
    return {'schema':'cfb_failure3_empirical_verified_possession_clock_paired_v3',
      'snapshot':str(snapshot),'precutoff_eligible_screen':screen,
      'motivation_fixture_excluded':MOTIVATION_GAME,
      'paired_protocol':'Same game, seed, 5000 simulations; Failure1+Failure2 constant',
      'sample_games':len(chosen),
      'all':summary(paired),'active':summary(active),
      'changed_games':len(changed),
      'verified_source_clock_games':len(active),
      'fallback_clock_games':len(paired)-len(active),
      'fallback_reasons':{reason:sum(x['clock_audit'].get('status')==reason for x in paired
                           if x['source_clock_verified'] is not True) for reason in
                           sorted({x['clock_audit'].get('status') for x in paired
                                   if x['source_clock_verified'] is not True})},
      'mean_total_before':round(float(np.mean([r['total_before'] for r in paired])),4),
      'mean_total_after':round(float(np.mean([r['total_after'] for r in paired])),4),
      'mean_absolute_total_shift':round(float(np.mean([abs(r['total_after']-r['total_before']) for r in paired])),4),
      'clock_evidence_class':'ESPN_SOURCE_ELAPSED_CROSSCHECKED_START_END_GAME_COVERAGE_NOT_INDEPENDENT_OFFICIAL_TOP',
      'fitted':False,'ATS_calibrated':False,
      'limitations':['Retrospective source capture, not contemporaneous as-of data',
       'No verified sportsbook spreads for ATS testing',
       'Chain persistence remains unfitted; drive durations are empirical when audit passes and fall back to unchanged Build16 otherwise',
       'Game scores in validation not used to choose coefficient'],
      'games':paired}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--max-games',type=int,default=160)
    parser.add_argument('--runs',type=int,default=5000)
    args=parser.parse_args()
    result=evaluate(args.snapshot,max_games=args.max_games,runs=args.runs)
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps({k:v for k,v in result.items() if k!='games'},indent=2))
    print('ACTIVE_CASES',json.dumps([g for g in result['games']
          if g['failure3_status']=='ACTIVE_BUILD16_UNCALIBRATED_FAILURE3'][:8]))
if __name__=='__main__':main()
