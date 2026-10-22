"""Failure 2: actual 2026 prior-game-only paired 5000-draw Build16 audit.

For each sampled fixture, compare the SAME unified challenger with only
Failure 2 scorer enabled/disabled, preserving frozen Failure 1 correction.
This is retrospective source snapshot research, not a verified original
point-in-time forecast and NOT an ATS backtest without sportsbook quotes.
"""
from __future__ import annotations
import argparse
import collections
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd

from cfb.pipeline import verify, normalize_pbp_teams, iso_utc
from cfb.research_backtest import observed_game
from cfb.mechanism_build16 import simulate_build16, _failure2_verified_scoring_histories
from cfb.failure1_learned_shadow import correction as frozen_failure1_correction

def cohort(snapshot,max_games=120):
    root=Path(snapshot)
    manifest=verify(snapshot)
    cutoff=iso_utc(manifest['captured_at_utc'])
    schedule=pd.read_parquet(root/'schedule.parquet')
    pbp=pd.read_parquet(root/'pbp.parquet')
    roster=pd.read_parquet(root/'roster.parquet')
    schedule['_kickoff']=pd.to_datetime(schedule.start_date,utc=True,errors='coerce')
    completed=(schedule.fbs_participant.fillna(False).eq(True) &
        schedule.completed.fillna(False).eq(True) &
        schedule['_kickoff'].lt(cutoff) &
        ~((schedule.home_division=='fcs') & (schedule.away_division=='fcs')) &
        schedule.home_points.notna() & schedule.away_points.notna())
    games=schedule.loc[completed].sort_values(['_kickoff','game_id']).copy()
    mapped=normalize_pbp_teams(pbp.loc[pbp.game_id.isin(games.game_id)],games)
    by_game={gid:part for gid,part in mapped.groupby('game_id',sort=False)}
    raw_by_game={gid:part for gid,part in pbp.loc[
        pbp.game_id.isin(games.game_id)].groupby('game_id',sort=False)}
    qb_ids={int(t):set(g.athlete_id.dropna()) for t,g in
            roster.loc[roster.position_abbreviation.eq('QB')].groupby('team_id')}
    historical=collections.defaultdict(list)
    pending=[]
    # Chronological all-eligible pool; select a deterministic even-spaced
    # retrospective sample using fixture order ONLY, never observed margins.
    targets=[]
    target_counts=collections.Counter()
    for kickoff, group in games.groupby('_kickoff',sort=True):
        ready=[item for item in pending if item[0]<=kickoff]
        pending=[item for item in pending if item[0]>kickoff]
        for _,team,row in ready:historical[team].append(row)
        additions=[]
        for _,g in group.iterrows():
            home,away=int(g.home_id),int(g.away_id)
            plays=by_game.get(g.game_id)
            if plays is None:continue
            h=observed_game(plays,home,away,g.home_points,g.away_points,
                qb_ids.get(home,set()),special_plays=raw_by_game.get(g.game_id),home_id=home)
            a=observed_game(plays,away,home,g.away_points,g.home_points,
                qb_ids.get(away,set()),special_plays=raw_by_game.get(g.game_id),home_id=home)
            if h is None or a is None:continue
            if len(historical[home])>=3 and len(historical[away])>=3:
                target_counts['pregame_prior_games_eligible']+=1
                # Histories frozen at this kickoff; no same-game outcomes.
                snap={t:list(rows) for t,rows in historical.items()}
                status=_failure2_verified_scoring_histories(home,away,snap)[2]['status']
                targets.append({'game_id':str(g.game_id),
                    'home':str(g.home_team),'away':str(g.away_team),
                    'home_id':home,'away_id':away,'kickoff':kickoff.isoformat(),
                    'neutral':bool(g.neutral_site),'actual_margin':float(g.home_points-g.away_points),
                    'historical':snap,'precheck':status})
                target_counts[status]+=1
            additions.extend(((home,h),(away,a)))
        for t,rec in additions:pending.append((kickoff+pd.Timedelta(hours=24),t,rec))
    if max_games<1:raise ValueError('Require positive cohort size')
    if len(targets)>max_games:
        idx=np.linspace(0,len(targets)-1,max_games,dtype=int)
        selected=[targets[int(i)] for i in idx]
    else:selected=targets
    return selected,dict(target_counts)

def evaluate(snapshot, max_games=120, runs=5000):
    if runs!=5000:raise ValueError('Must run exactly 5000 genuine simulations per variant')
    targets,count=cohort(snapshot,max_games)
    results=[]
    for game in targets:
        h,a=game['home_id'],game['away_id']
        histories=game['historical']
        same_seed=int(game['game_id'])%(2**32)
        fitted=frozen_failure1_correction(h,a,histories)
        rush_delta=fitted['margin_delta_points']
        common=dict(runs=5000,seed=same_seed,failure1_fit_margin_delta=rush_delta)
        reference=simulate_build16(h,a,histories,game['neutral'],failure2_attribution=False,**common)
        corrected=simulate_build16(h,a,histories,game['neutral'],failure2_attribution=True,**common)
        rs,cs=reference.get('simulation'),corrected.get('simulation')
        result={k:game[k] for k in ('game_id','home','away','kickoff','actual_margin','precheck')}
        result.update({'same_seed':same_seed,'runs_reference':rs.get('simulations') if rs else None,
           'runs_corrected':cs.get('simulations') if cs else None,
           'failure1_rush_delta_held_constant':rush_delta,
           'original_status':reference['status'],'corrected_status':corrected['status'],
           'failure2_status':corrected.get('failure2_scoring_attribution',{}).get('status')})
        if rs is not None and cs is not None:
            result.update({
                'original_mean_margin':rs['mean_margin_home'],
                'corrected_mean_margin':cs['mean_margin_home'],
                'original_mean_total':rs['mean_total'],
                'corrected_mean_total':cs['mean_total'],
                'original_score_home':rs['mean_score_home'],
                'corrected_score_home':cs['mean_score_home'],
                'original_score_away':rs['mean_score_away'],
                'corrected_score_away':cs['mean_score_away'],
                'original_p10_p50_p90':rs['margin_p10_p50_p90'],
                'corrected_p10_p50_p90':cs['margin_p10_p50_p90'],
                'corrected_nonoff_points_home':cs.get('simulated_nonoffensive_points_home'),
                'corrected_nonoff_points_away':cs.get('simulated_nonoffensive_points_away'),
                'mean_margin_change':round(cs['mean_margin_home']-rs['mean_margin_home'],4),
                'mean_total_change':round(cs['mean_total']-rs['mean_total'],4),
                'reference_margin_absolute_error':round(abs(rs['mean_margin_home']-game['actual_margin']),4),
                'corrected_margin_absolute_error':round(abs(cs['mean_margin_home']-game['actual_margin']),4),
                'paired_5000_verified':rs['simulations']==cs['simulations']==5000})
        results.append(result)
    paired=[x for x in results if x.get('paired_5000_verified')]
    active=[x for x in paired if x.get('failure2_status')==
          'ACTIVE_OFFENSE_ONLY_PPD_SEPARATE_NONOFFENSIVE_SCORING']
    def mae(name,rows):
        return round(float(np.mean([x[name] for x in rows])),5) if rows else None
    def bootstrap_gain(rows,iterations=3000):
        if len(rows)<2:return None
        gains=np.array([x['reference_margin_absolute_error']-
                        x['corrected_margin_absolute_error'] for x in rows],dtype=float)
        rng=np.random.default_rng(20261022)
        draws=rng.integers(0,len(gains),size=(iterations,len(gains)))
        return [round(float(x),5) for x in np.quantile(gains[draws].mean(axis=1),[.025,.975])]
    signed=[x['mean_margin_change'] for x in paired]
    absolute_shift=round(float(np.mean(np.abs(signed))),5) if signed else None
    return {'schema':'cfb_failure2_actual_source_paired_build16_v2',
      'snapshot':str(snapshot),
      'historical_pregame_cohort':count,
      'selected_games':len(targets),'paired_5000_game_comparisons':len(paired),
      'corrected_attribution_active_games':len(active),
      'all_comparisons_same_seed':all(x.get('paired_5000_verified') for x in paired),
      'reference_mae':mae('reference_margin_absolute_error',paired),
      'corrected_mae':mae('corrected_margin_absolute_error',paired),
      'active_reference_mae':mae('reference_margin_absolute_error',active),
      'active_corrected_mae':mae('corrected_margin_absolute_error',active),
      'mean_absolute_margin_change':absolute_shift,
      'paired_mae_gain_original_minus_corrected':round(
          mae('reference_margin_absolute_error',paired)-mae('corrected_margin_absolute_error',paired),5) if paired else None,
      'paired_mae_gain_bootstrap95':bootstrap_gain(paired),
      'active_paired_mae_gain_bootstrap95':bootstrap_gain(active),
      'n_changed_forecasts':sum(abs(x['mean_margin_change'])>0.001 for x in paired),
      'conclusion':('REAL_SAME_SEED_5000_SCORING_EFFECT_MEASURED' if active
                     else 'NO_ACTIVATED_REAL_SCORING_CORRECTION_IN_TEST_COHORT'),
      'caveats':['Retrospective 2026 source capture, not original pre-kickoff capture',
                 'No verified source sportsbook spreads, thus no ATS comparison',
                 'Fail-closed attribution may prevent any substantive change',
                 'Only previous games enter team histories at kickoff'],
      'games':[{k:v for k,v in x.items() if k!='historical'} for x in results]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--max-games',type=int,default=120)
    p.add_argument('--runs',type=int,default=5000)
    args=p.parse_args()
    report=evaluate(args.snapshot,args.max_games,args.runs)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({k:v for k,v in report.items() if k!='games'},indent=2))
    print('PAIRED_FIXTURES',json.dumps(report['games'][:12]))
if __name__=='__main__':main()
