"""Forward-only, same-snapshot CFB Build16 vs 2023-fitted rushing shadow ledger.

Do not replace the independent earliest-ever original-forecast ledger. Its
earliest forecast may predate the challenger shadow, so the first *eligible
paired* forecast is a DISTINCT research cohort explicitly tagged as such.
Read only immutable, SHA-verified forecast/odds archives and after-kickoff
free-source final scores; never rerun either model to manufacture an outcome.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
from cfb.archive import verify_archive,load_all_archives,digest
from cfb.free_odds import load_quotes
from cfb.market import _date

SCHEMA='cfb_failure1_same_snapshot_frozen_pair_v1'
PROTOCOL='cfb-failure1-learned-rush-shadow-v1'

def _valid_num(x):
    return type(x) in (int,float) and math.isfinite(x)

def _triple(raw,game_id):
    if not isinstance(raw,(tuple,list)) or len(raw)!=3 or any(not _valid_num(v) for v in raw):
        raise ValueError('Bad frozen margin percentile triple for '+game_id)
    if not raw[0]<=raw[1]<=raw[2]:
        raise ValueError('Nonmonotone margin quantiles for '+game_id)
    return [float(z) for z in raw]

def load_first_frozen_pairs(archive_root):
    """Select earliest valid eligible research pair, not earliest-ever forecast."""
    root=Path(archive_root)
    results={}
    counts={'forecast_archive_files':0,'archive_game_records':0,
            'eligible_pair_records':0,'historical_without_fitted_shadow':0,
            'fitted_shadow_missing_evidence':0,'pair_before_kickoff':0}
    if not root.is_dir():return results,counts
    for file in sorted(root.glob('*.zip')):
        idx,records=verify_archive(file)
        counts['forecast_archive_files']+=1
        arch_sha=digest(file.read_bytes())
        for gid,p in records.items():
            if p.get('status')!='FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE':continue
            counts['archive_game_records']+=1
            z=p.get('failure1_learned_rush_shadow')
            if not isinstance(z,dict) or z.get('status')!='FROZEN_2023_FITTED_RUSH_SHADOW_5000':
                if isinstance(z,dict):counts['fitted_shadow_missing_evidence']+=1
                else:counts['historical_without_fitted_shadow']+=1
                continue
            ref=p.get('build16_challenger')
            if not isinstance(ref,dict) or ref.get('status')!='AVAILABLE_UNFITTED_BUILD16_RESEARCH_ONLY':
                raise ValueError('Frozen learned candidate without same-snapshot Build16 '+gid)
            if z.get('protocol')!=PROTOCOL or z.get('runs')!=5000 or z.get('fitted_year')!=2023:
                raise ValueError('Invalid frozen learned candidate protocol '+gid)
            if z.get('source_manifest_sha256')!=p.get('source_snapshot_manifest_sha256'):
                raise ValueError('Mismatched candidate and reference source '+gid)
            if not isinstance(z.get('frozen_fitted_model_sha256'),str) or len(z['frozen_fitted_model_sha256'])!=64:
                raise ValueError('Candidate lacks frozen 2023 model hash '+gid)
            common_seed=int(gid)%(2**32)
            if z.get('seed')!=common_seed:
                raise ValueError('Candidate random seed does not match matchup '+gid)
            sim=ref.get('simulation')
            if not isinstance(sim,dict) or sim.get('simulations')!=5000 or sim.get('seed')!=common_seed:
                raise ValueError('Reference not 5000 draws at candidate seed '+gid)
            ref_margin=ref.get('mean_margin_home')
            alt_margin=z.get('projected_margin_home')
            if not _valid_num(ref_margin) or not _valid_num(alt_margin):
                raise ValueError('Nonfinite frozen paired margins '+gid)
            pt=_date(p['prediction_generated_at_utc'],'paired issue time')
            kt=_date(p['kickoff_utc'],'fixture kickoff')
            if not pt<kt:raise ValueError('A purported pregame paired forecast was issued after kickoff '+gid)
            counts['eligible_pair_records']+=1
            counts['pair_before_kickoff']+=1
            row={**p,'paired_forecast_archive_run_id':str(idx['github_run_id']),
                'paired_forecast_zip_sha256':arch_sha}
            previous=results.get(gid)
            if previous is None or (pt,int(idx['github_run_id'])) < (
              _date(previous['prediction_generated_at_utc'],'paired issue time'),
              int(previous['paired_forecast_archive_run_id'])):
                results[gid]=row
    counts['distinct_first_paired_games']=len(results)
    return results,counts

def _outcome(actual,spread,pred):
    edge=pred+spread
    x=actual+spread
    return {'edge':round(edge,5),
      'side':'HOME' if edge>1e-8 else 'AWAY' if edge< -1e-8 else 'NO_EDGE',
      'result':'HOME' if x>1e-8 else 'AWAY' if x< -1e-8 else 'PUSH'}

def _metric(rows):
    n=len(rows)
    if not n:return {'games':0,'reference_mae':None,'learned_mae':None,
       'mae_gain_reference_minus_learned':None,'paired_bootstrap95_gain':None,
       'reference_brier':None,'learned_brier':None,'reference_margin80_coverage':None,
       'learned_margin80_coverage':None}
    r=np.asarray([x['build16_margin'] for x in rows],float)
    s=np.asarray([x['learned_margin'] for x in rows],float)
    y=np.asarray([x['actual_home_margin'] for x in rows],float)
    diff=np.abs(r-y)-np.abs(s-y)
    rng=np.random.default_rng(20261022)
    ix=rng.integers(0,n,size=(2000,n))
    ci=np.quantile(diff[ix].mean(axis=1),[.025,.975])
    def prob(key):
        v=[(x[key],int(x['actual_home_margin']>0)) for x in rows
              if _valid_num(x.get(key))]
        return round(float(np.mean([(p-z)**2 for p,z in v])),5) if v else None
    def coverage(key):
        v=[(x[key],x['actual_home_margin']) for x in rows if x.get(key) is not None]
        return round(sum(a[0]<=y<=a[2] for a,y in v)/len(v),5) if v else None
    return {'games':n,
        'reference_mae':round(float(np.mean(np.abs(r-y))),5),
        'learned_mae':round(float(np.mean(np.abs(s-y))),5),
        'reference_rmse':round(float(np.sqrt(np.mean((r-y)**2))),5),
        'learned_rmse':round(float(np.sqrt(np.mean((s-y)**2))),5),
        'mae_gain_reference_minus_learned':round(float(np.mean(diff)),5),
        'paired_bootstrap95_gain':[round(float(z),5) for z in ci],
        'reference_brier':prob('build16_home_win_probability'),
        'learned_brier':prob('learned_home_win_probability'),
        'reference_margin80_coverage':coverage('build16_margin_p10_p50_p90'),
        'learned_margin80_coverage':coverage('learned_margin_p10_p50_p90')}

def _ats_stats(rows):
    with_quote=[x for x in rows if x.get('quoted_spread')]
    decisions=[r for r in with_quote if
      r['quoted_spread']['build16']['side']!='NO_EDGE' and
      r['quoted_spread']['learned']['side']!='NO_EDGE' and
      r['quoted_spread']['build16']['result']!='PUSH']
    def record(key):
        w=sum(z['quoted_spread'][key]['side']==z['quoted_spread'][key]['result'] for z in decisions)
        return {'correct':w,'wrong':len(decisions)-w,
          'cover_accuracy':round(w/len(decisions),5) if decisions else None}
    different=[r for r in decisions if
      r['quoted_spread']['build16']['side']!=r['quoted_spread']['learned']['side']]
    return {'completed_games_with_joint_real_pregame_quote':len(with_quote),
       'fair_paired_decisive_spreads':len(decisions),
       'different_ATS_sides':len(different),
       'same_ATS_sides':len(decisions)-len(different),
       'reference':record('build16'),
       'learned':record('learned'),
       'different_side_game_ids':[r['game_id'] for r in different],
       'original_book_publication_time_verified':False,
       'spread_source_status':'observed ESPN free quote; no exact sportsbook quote publication attestation',
       'ATS_calibrated':False}

def evaluate(forecast_root,odds_root,schedule,captured_at,source_url,source_sha256):
    cutoff=_date(captured_at,'observed final source time')
    first_all=load_all_archives(forecast_root)
    pairs,load_info=load_first_frozen_pairs(forecast_root)
    quotes=load_quotes(odds_root)
    # Describe frozen pregame disagreement even before any final exists.
    # This is NOT a statement of predictive accuracy or ATS value.
    model_shifts=[]
    for gid,p in pairs.items():
        original=p['build16_challenger']
        learned=p['failure1_learned_rush_shadow']
        base=float(original['mean_margin_home'])
        alt=float(learned['projected_margin_home'])
        model_shifts.append({'game_id':gid,'home':p['home'],'away':p['away'],
            'kickoff_utc':p['kickoff_utc'],
            'build16_predicted_home_margin':round(base,5),
            'learned_predicted_home_margin':round(alt,5),
            'forecast_margin_change':round(alt-base,5),
            'severe_rushing_screen':bool((p.get('mandatory_rushing_audit') or {}).get('requires_full_rush_stress'))})
    average_abs_shift=(round(sum(abs(x['forecast_margin_change']) for x in model_shifts)/
                      len(model_shifts),5) if model_shifts else None)
    quoted_pairs=0
    for gid,p in pairs.items():
        q=quotes.get(gid)
        if q and q['home']==p['home'] and q['away']==p['away'] and (
             max(_date(q['observed_at_utc'],'market'),
                 _date(p['prediction_generated_at_utc'],'pair issued'))<
             _date(p['kickoff_utc'],'kickoff')):
            quoted_pairs+=1
    source={str(x.game_id):x for x in schedule.itertuples(index=False)}
    scored=[];pending=[]
    for gid,p in sorted(pairs.items(),key=lambda x:(x[1]['kickoff_utc'],x[0])):
        original=first_all.get(gid)
        original_id=original['original_forecast_run_id'] if original else None
        first_delay=bool(original_id!=p['paired_forecast_archive_run_id'])
        g=source.get(gid)
        if g is None:
            pending.append({'game_id':gid,'reason':'MISSING_IN_OBSERVED_SCHEDULE'});continue
        kickoff=_date(p['kickoff_utc'],'paired kickoff')
        schedule_at=pd.to_datetime(g.start_date,utc=True,errors='coerce')
        if pd.isna(schedule_at) or abs((kickoff-schedule_at.to_pydatetime()).total_seconds())>3600:
            pending.append({'game_id':gid,'reason':'KICKOFF_CHANGED_REQUIRES_REVIEW'});continue
        if (str(g.home_id)!=p['home_id'] or str(g.away_id)!=p['away_id']):
            raise ValueError('Paired pregame home/away team IDs mismatch '+gid)
        if cutoff<kickoff+timedelta(hours=3):
            pending.append({'game_id':gid,'reason':'AWAITING_MINIMUM_THREE_HOURS_AFTER_KICKOFF'});continue
        if not bool(g.completed) or pd.isna(g.home_points) or pd.isna(g.away_points):
            pending.append({'game_id':gid,'reason':'AWAITING_SOURCE_RECORDED_COMPLETED_FINAL'});continue
        home=float(g.home_points);away=float(g.away_points)
        if any(not math.isfinite(z) or z<0 or z>150 or z%1!=0 for z in (home,away)):
            raise ValueError('Invalid final score '+gid)
        actual=home-away
        b=p['build16_challenger']
        l=p['failure1_learned_rush_shadow']
        q_b=_triple(b.get('margin_p10_p50_p90'),gid)
        q_l=_triple(l.get('margin_p10_p50_p90'),gid)
        row={'game_id':gid,'home':p['home'],'away':p['away'],
          'kickoff_utc':p['kickoff_utc'],
          'first_original_forecast_run_id':original_id,
          'first_eligible_paired_run_id':p['paired_forecast_archive_run_id'],
          'pair_was_not_first_ever_forecast':first_delay,
          'frozen_pair_generated_at_utc':p['prediction_generated_at_utc'],
          'paired_archive_sha256':p['paired_forecast_zip_sha256'],
          'fitted_model_sha256':l['frozen_fitted_model_sha256'],
          'source_snapshot_manifest_sha256':p['source_snapshot_manifest_sha256'],
          'same_seed_5000_each':True,
          'actual_home_margin':actual,'actual_home_points':int(home),'actual_away_points':int(away),
          'build16_margin':float(b['mean_margin_home']),
          'learned_margin':float(l['projected_margin_home']),
          'margin_absolute_error_build16':round(abs(float(b['mean_margin_home'])-actual),5),
          'margin_absolute_error_learned':round(abs(float(l['projected_margin_home'])-actual),5),
          'frozen_adjustment_input_margin_points':l['margin_delta_points'],
          'build16_margin_p10_p50_p90':q_b,
          'learned_margin_p10_p50_p90':q_l,
          'build16_home_win_probability':b.get('simulation',{}).get('home_win_probability_raw'),
          'learned_home_win_probability':l.get('win_probability_home_raw'),
          'severe_rush_mismatch_frozen':bool((p.get('mandatory_rushing_audit') or {}).get('requires_full_rush_stress')),
          'quoted_spread':None}
        observed=quotes.get(gid)
        if observed is not None:
            if (observed['home']!=p['home'] or observed['away']!=p['away'] or
                _date(observed['kickoff_utc'],'quote kickoff')!=kickoff):
                raise ValueError('Quote does not match frozen paired matchup '+gid)
            obs=_date(observed['observed_at_utc'],'quote capture')
            produced=_date(p['prediction_generated_at_utc'],'forecast capture')
            if max(obs,produced)<kickoff:
                spread=observed['quote']['spread_home']
                if not _valid_num(spread):raise ValueError('Invalid observed spread '+gid)
                row['quoted_spread']={
                  'source':'Original free ESPN odds bytes; observed pre-kickoff',
                  'observed_at_utc':observed['observed_at_utc'],
                  'book':observed['quote']['book'],
                  'archive_sha256':observed['market_archive_sha256'],
                  'spread_home':spread,
                  'odds_home':observed['quote']['home_odds_american'],
                  'odds_away':observed['quote']['away_odds_american'],
                  'observed_after_paired_forecast':obs>produced,
                  'build16':_outcome(actual,spread,row['build16_margin']),
                  'learned':_outcome(actual,spread,row['learned_margin'])}
        scored.append(row)
    all_stats=_metric(scored)
    severe_stats=_metric([r for r in scored if r['severe_rush_mismatch_frozen']])
    result={'schema':SCHEMA,
        'status':'PROSPECTIVE_FROZEN_PAIRED_RUSH_SCORING_RESEARCH_ONLY',
        'final_schedule_source_url':source_url,
        'final_schedule_source_sha256':source_sha256,
        'final_schedule_observed_at_utc':cutoff.isoformat(),
        'source_archive_verification':load_info,
        'first_eligible_paired_forecast_games':len(pairs),
        'pregame_paired_model_differences':{
            'games':len(model_shifts),
            'avg_abs_change_in_predicted_margin':average_abs_shift,
            'most_divergent_upcoming_margins':sorted(model_shifts,
                 key=lambda x:-abs(x['forecast_margin_change']))[:5],
            'severe_rushing_screen_games':sum(x['severe_rushing_screen'] for x in model_shifts),
            'changed_outright_winner_direction':sum(
                 (x['build16_predicted_home_margin']>0)!=
                 (x['learned_predicted_home_margin']>0) for x in model_shifts),
            'not_a_forward_accuracy_measure':True},
        'first_eligible_paired_games_with_any_real_pre_kickoff_quote':quoted_pairs,
        'first_paired_not_first_ever_forecast':sum(
             bool(first_all.get(gid) and first_all[gid]['original_forecast_run_id']!=p['paired_forecast_archive_run_id'])
             for gid,p in pairs.items()),
        'qualified_completed_paired_games':len(scored),
        'pending':pending,'pending_games':len(pending),
        'margin':all_stats,'severe_mismatch_margin':severe_stats,
        'ATS_at_joint_observed_spread':_ats_stats(scored),
        'no_historical_forecast_backfill':True,
        'uses_same_snapshot_seed_and_runs':True,
        'ATS_calibrated':False,
        'automatic_challenger_promotion':False,
        'limitations':['May use first eligible paired forecast that is LATER than the first-ever forecast; never substitute it as original earliest forecast',
          'Bookmaker original quote-publication time not verified; only real ESPN observation time',
          'Mutable final source independently frozen by caller, not an official settlement feed',
          'No simulation reruns or model fitting after real scores are known',
          'Repeated-team games induce dependence; paired bootstrap CI is descriptive'],
        'scored_pairs':scored}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forecasts',required=True);p.add_argument('--odds',required=True)
    p.add_argument('--schedule',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    data=Path(args.schedule).read_bytes()
    manifest=json.loads(Path(args.manifest).read_text())
    source=manifest['datasets']['schedule']
    if hashlib.sha256(data).hexdigest()!=source['sha256']:
        raise ValueError('Source schedule does not match captured manifest')
    report=evaluate(args.forecasts,args.odds,pd.read_parquet(args.schedule),
      source['captured_at_utc'],source['url'],source['sha256'])
    dst=Path(args.out);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():raise FileExistsError('Refuse overwriting immutable paired audit')
    dst.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('scored_pairs','pending')},indent=2))
if __name__=='__main__':main()
