"""Single CFB Build16 research candidate: 2024 fit, 2025 holdout, 2026 check.

Every comparison is 5000 real same-seed simulations, before/after the sole
optional Build16 residual; F1-3 held constant. No quotes are synthesized and
no historical back-filled spread is called a verified ATS result. Historical
ESPN snapshots are retrospective and do not prove row-publication timestamps.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math
from pathlib import Path
import numpy as np
from cfb.failure2_paired import cohort
from cfb.failure1_learned_shadow import correction as earlier_f1
from cfb.mechanism_build16 import simulate_build16
from cfb.continuous_rush_sos_residual import (
    feature_vector,fit,score,VERSION,MIN_TRAINING_GAMES)

def check_year(path,year):
    from cfb.pipeline import verify
    import pandas as pd
    root=Path(path)
    manifest=verify(root)
    sched=pd.read_parquet(root/'schedule.parquet')
    years=set(pd.to_datetime(sched.start_date,utc=True,errors='coerce').dt.year.dropna().astype(int).unique())
    if year not in years:raise ValueError('Source season does not match requested '+str(year))
    return manifest

def _sim(g,extra=None):
    h,a=g['home_id'],g['away_id']
    kwargs=dict(runs=5000,seed=int(g['game_id'])%(2**32),
         failure1_fit_margin_delta=earlier_f1(h,a,g['historical'])['margin_delta_points'])
    if extra is not None:
        kwargs['continuous_rush_sos_residual_delta']=extra
    s=simulate_build16(h,a,g['historical'],g['neutral'],**kwargs)
    return s.get('simulation')

def training(snapshot,max_games=240):
    manifest=check_year(snapshot,2024)
    games,counts=cohort(snapshot,max_games,enrich_rush=True)
    rows=[];exclusions=collections.Counter()
    for i,g in enumerate(games):
        if not g['kickoff'].startswith('2024'):continue
        f=feature_vector(g['home_id'],g['away_id'],g['historical'])
        if f['status']!='AVAILABLE_PREGAME_CONTINUOUS_RUSH_SOS':
            exclusions[f['status']]+=1;continue
        r=_sim(g)
        if r is None:
            exclusions['UNAVAILABLE_EXISTING_BUILD16']+=1;continue
        if r['simulations']!=5000:raise AssertionError('Training did not run 5000 simulations')
        rows.append({'year':2024,'game_id':g['game_id'],'kickoff':g['kickoff'],
                     'simulations':5000,'reference_margin':r['mean_margin_home'],
                     'actual_margin':g['actual_margin'],'feature_vector':f['vector'],
                     'feature_sources':f['unscaled']})
        if (i+1)%40==0:print('TRAIN_PROGRESS',i+1,len(rows),flush=True)
    if len(rows)<MIN_TRAINING_GAMES:
        raise ValueError(f'Insufficient complete 2024 training games {len(rows)} of {len(games)}. Causes: {dict(exclusions)}')
    model=fit(rows)
    return {'schema':'cfb_rush_sos_2024_frozen_fit_v1',
      'captured_snapshot':manifest.get('captured_at_utc'),
      'sample_games_before_feature_exclusion':len(games),
      'qualified_5000_draw_training_games':len(rows),
      'exclusion_reasons':dict(exclusions),
      'cohort_screen':counts,
      'frozen_model':model,'training_records':rows,
      'data_provenance':'2024 observed PBP from retrospective snapshot, with 24h prior-game embargo',
      'forward_frozen_asof_2024_pubtimestamps_verified':False,
      'no_future_game_label_in_each_matchup_feature':True}

def _summarize(rows):
    if not rows:return {'n':0}
    y=np.array([r['actual_margin'] for r in rows],float)
    m=np.array([r['reference_margin'] for r in rows],float)
    a=np.array([r['candidate_margin'] for r in rows],float)
    before=np.abs(y-m);after=np.abs(y-a)
    def brier(which):
        errors=np.array([float(r['actual_margin']>0)-r[which] for r in rows],float)
        return float(np.mean(np.square(errors)))
    # Cast boolean to float explicitly; no tying home-win counts to an ATS bet.
    return {'n':len(rows),
      'margin_mae_reference':round(float(before.mean()),5),
      'margin_mae_candidate':round(float(after.mean()),5),
      'margin_mae_improvement':round(float((before-after).mean()),5),
      'rmse_reference':round(float(np.sqrt(np.mean((y-m)**2))),5),
      'rmse_candidate':round(float(np.sqrt(np.mean((y-a)**2))),5),
      'mean_margin_shift_abs':round(float(np.mean(np.abs(a-m))),5),
      'bias_reference':round(float(np.mean(y-m)),5),
      'bias_candidate':round(float(np.mean(y-a)),5),
      'brier_home_win_reference':round(brier('reference_win_prob'),6),
      'brier_home_win_candidate':round(brier('candidate_win_prob'),6),
      'margin_p10_p90_coverage_reference':round(float(np.mean([
          r['reference_p10']<=r['actual_margin']<=r['reference_p90'] for r in rows])),5),
      'margin_p10_p90_coverage_candidate':round(float(np.mean([
          r['candidate_p10']<=r['actual_margin']<=r['candidate_p90'] for r in rows])),5)}

def evaluate(snapshot,model,year,max_games=120):
    if year not in (2025,2026):raise ValueError('Require later-year holdout')
    manifest=check_year(snapshot,year)
    games,counts=cohort(snapshot,max_games,enrich_rush=True)
    rows=[];missing=collections.Counter()
    for i,g in enumerate(games):
        if not g['kickoff'].startswith(str(year)):continue
        # The 2026 Liberty-Kennesaw result motivated this research design.
        # Exclude it even from later-year promotion/model-choice summaries.
        if g['game_id']=='401871056':
            missing['EXCLUDED_MOTIVATING_LIBERTY_KENNESAW_GAME']+=1
            continue
        feat=feature_vector(g['home_id'],g['away_id'],g['historical'])
        scored=score(model,feat)
        if scored['status']!='FROZEN_EARLIER_SEASON_RESIDUAL_AVAILABLE':
            missing[scored['status']]+=1;continue
        ref=_sim(g)
        cand=_sim(g,extra=scored['margin_delta_points'])
        if ref is None or cand is None:
            missing['BUILD16_INCOMPLETE_EPA']+=1;continue
        if ref['simulations']!=5000 or cand['simulations']!=5000:
            raise AssertionError('Paired 5000 draws required')
        rows.append({'game_id':g['game_id'],'kickoff':g['kickoff'],
           'home':g['home'],'away':g['away'],'actual_margin':g['actual_margin'],
           'reference_margin':ref['mean_margin_home'],
           'candidate_margin':cand['mean_margin_home'],
           'reference_total':ref['mean_total'],
           'candidate_total':cand['mean_total'],
           'actual_total':g['actual_total'],
           'reference_p10':ref['margin_p10_p50_p90'][0],
           'reference_p90':ref['margin_p10_p50_p90'][2],
           'candidate_p10':cand['margin_p10_p50_p90'][0],
           'candidate_p90':cand['margin_p10_p50_p90'][2],
           'reference_win_prob':ref['home_win_probability_raw'],
           'candidate_win_prob':cand['home_win_probability_raw'],
           'seed':int(g['game_id'])%(2**32),'paired_runs':5000,
           'sos_margin_feature_unfitted':feat['unscaled']['opponent_adjusted_total_margin_feature'],
           'rush_advantage_continuous':feat['unscaled']['matched_early_rush_advantage'],
           'proposed_fitted_residual_home':scored['margin_delta_points'],
           'actual_margin_error_reference':round(abs(g['actual_margin']-ref['mean_margin_home']),5),
           'actual_margin_error_candidate':round(abs(g['actual_margin']-cand['mean_margin_home']),5)})
        if (i+1)%30==0:print('HOLDOUT_PROGRESS',year,i+1,len(rows),flush=True)
    gains=np.array([r['actual_margin_error_reference']-r['actual_margin_error_candidate'] for r in rows],float)
    if len(gains):
        rng=np.random.default_rng(29014+year)
        boot=np.mean(rng.choice(gains,(3000,len(gains)),replace=True),axis=1)
        ci=np.percentile(boot,[2.5,97.5]).round(5).tolist()
    else:ci=None
    sorted_rush=sorted(rows,key=lambda x:abs(x['rush_advantage_continuous']),reverse=True)
    n_top=max(1,len(rows)//4) if rows else 0
    rank=sorted(rows,key=lambda x:abs(x['sos_margin_feature_unfitted']-x['reference_margin']),reverse=True)
    out={'schema':'cfb_continuous_rush_sos_paired_5000_v1',
        'year':year,'snapshot_at':manifest.get('captured_at_utc'),
        'source_sample_count':len(games),'cohort_screen':counts,
        'paired_games':len(rows),'missing':dict(missing),
        'all':_summarize(rows),
        'strongest_rush_advantage_quartile':_summarize(sorted_rush[:n_top]),
        'most_divergent_sos_vs_build16_quartile':_summarize(rank[:n_top]),
        'MAE_improvement_paired_bootstrap95':ci,
        'verified_spread_games':0,'historical_ats_hit_rate':None,
        'retrospective_source_capture_not_independently_2024_2025_asof':True,
        'simulations_per_game_per_side':5000,
        'model_status':'UNPROMOTED_SINGLE_BUILD16_RESEARCH',
        'games':rows}
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=('fit','evaluate'))
    p.add_argument('--snapshot',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--model')
    p.add_argument('--year',type=int)
    p.add_argument('--max-games',type=int,default=160)
    args=p.parse_args()
    if args.mode=='fit':
        z=training(args.snapshot,max_games=args.max_games)
    else:
        if not args.model or args.year not in (2025,2026):
            raise ValueError('Require frozen earlier-year model and later year')
        model=json.loads(Path(args.model).read_text())
        if 'frozen_model' in model:model=model['frozen_model']
        z=evaluate(args.snapshot,model,args.year,args.max_games)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(z,indent=2,sort_keys=True))
    summary={k:v for k,v in z.items() if k not in ('games','training_records')}
    print('SINGLE_BUILD16_CONTINUOUS_RUSH_SOS_RESULTS')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
