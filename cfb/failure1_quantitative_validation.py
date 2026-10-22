"""Independent 2023/2024 blind-to-coefficient-study quantitative Failure 1 check.

Freeze EXACT existing 2026-10-22 'negative' response, before reading 2023/2024
results. Reference Build16 and paired Build16+negative correction use exactly
5000 genuine simulations with identical per-fixture seeds. No best-of-N tuning.
"""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from cfb.pipeline import verify,normalize_pbp_teams,iso_utc
from cfb.research_backtest import observed_game
from cfb.mechanism_build16 import simulate_build16
from cfb.rush_severity_source import extract,append_training_record
from cfb.mandatory_rushing_audit import audit as rushing_audit

def grade(rows,subset=None):
    if subset is not None:rows=[r for r in rows if subset(r)]
    summary={'games':len(rows)}
    if not rows:return summary
    truth=np.array([r['actual_home_margin'] for r in rows],float)
    total=np.array([r['actual_total'] for r in rows],float)
    win=(truth>0).astype(float)
    for label in ('reference','correction'):
        p=np.array([r[label]['margin'] for r in rows],float)
        t=np.array([r[label]['total'] for r in rows],float)
        prob=np.array([r[label]['home_win_probability_raw'] for r in rows],float)
        q=np.array([r[label]['margin_p10_p50_p90'] for r in rows],float)
        summary[label]={'mae':round(float(np.mean(abs(p-truth))),6),
                        'total_mae':round(float(np.mean(abs(t-total))),6),
                        'margin_bias':round(float(np.mean(p-truth)),6),
                        'brier':round(float(np.mean((prob-win)**2)),6),
                        'margin80_coverage':round(float(np.mean(
                          (truth>=q[:,0])&(truth<=q[:,2]))),6),
                        'winner_direction_accuracy':round(float(np.mean(
                          np.sign(truth)==np.sign(p))),6)}
    diff=np.array([abs(r['reference']['margin']-r['actual_home_margin'])-
       abs(r['correction']['margin']-r['actual_home_margin']) for r in rows],float)
    rng=np.random.default_rng(20261022)
    indices=rng.integers(0,len(rows),size=(3000,len(rows)))
    summary['paired_mae_gain']=round(float(diff.mean()),6)
    summary['paired_bootstrap95']=[round(float(z),6) for z in
                                  np.quantile(diff[indices].mean(axis=1),[.025,.975])]
    summary['matched_margin_change_abs_avg']=round(float(np.mean([
        abs(r['correction']['margin']-r['reference']['margin']) for r in rows])),6)
    return summary

def run(snapshot,out,*,max_games=180,runs=5000,min_prior_games=3):
    if runs!=5000:raise ValueError('Every paired variant must have exactly 5000 draws')
    root=Path(snapshot);m=verify(root)
    cutoff=iso_utc(m['captured_at_utc'])
    sched=pd.read_parquet(root/'schedule.parquet')
    pbp=pd.read_parquet(root/'pbp.parquet')
    roster=pd.read_parquet(root/'roster.parquet')
    sched['_kickoff']=pd.to_datetime(sched.start_date,utc=True,errors='coerce')
    good=(sched.fbs_participant.fillna(False).eq(True)&
          sched.completed.fillna(False).eq(True)&
          (sched['_kickoff']+pd.Timedelta(hours=24)<=cutoff)&
          ~((sched.home_division=='fcs')&(sched.away_division=='fcs'))&
          sched.home_points.notna()&sched.away_points.notna())
    games=sched.loc[good].sort_values(['_kickoff','game_id'])
    mapped=normalize_pbp_teams(pbp.loc[pbp.game_id.isin(games.game_id)],games)
    grouped={i:g for i,g in mapped.groupby('game_id',sort=False)}
    rawgroup={i:g for i,g in pbp.loc[pbp.game_id.isin(games.game_id)].groupby('game_id',sort=False)}
    qb={int(t):set(g.athlete_id.dropna()) for t,g in
        roster.loc[roster.position_abbreviation.eq('QB')].groupby('team_id')}
    history=defaultdict(list);pending=[];rows=[];fail=defaultdict(int)
    for kickoff,block in games.groupby('_kickoff',sort=True):
        eligible=[x for x in pending if x[0]<=kickoff]
        pending=[x for x in pending if x[0]>kickoff]
        for _,tid,record in eligible:history[tid].append(record)
        for _,g in block.iterrows():
            hid,aid=int(g.home_id),int(g.away_id)
            if len(rows)<max_games and min(len(history[hid]),len(history[aid]))>=min_prior_games:
                seed=int(g.game_id)%2**32
                ref=simulate_build16(hid,aid,history,bool(g.neutral_site),runs=runs,seed=seed)
                alt=simulate_build16(hid,aid,history,bool(g.neutral_site),
                                     runs=runs,seed=seed,mismatch_research='negative')
                if ref.get('simulation') is not None and alt.get('simulation') is not None:
                    def stats(x):
                        z=x['simulation']
                        return {'margin':z['mean_margin_home'],'total':z['mean_total'],
                        'home_win_probability_raw':z['home_win_probability_raw'],
                        'margin_p10_p50_p90':z['margin_p10_p50_p90']}
                    from cfb.matchups import matchup_signals
                    signal=matchup_signals(hid,aid,history)
                    screen=rushing_audit(hid,aid,history,signal)
                    rows.append({'game_id':str(g.game_id),'kickoff':kickoff.isoformat(),
                      'home':str(g.home_team),'away':str(g.away_team),
                      'prior_games_home':len(history[hid]),'prior_games_away':len(history[aid]),
                      'severe_screen_ex_ante':screen['requires_full_rush_stress'],
                      'reason_codes':screen['trigger_reasons'],
                      'home_negative_rate_excess':screen['home']['matched_negative_run_difference'],
                      'away_negative_rate_excess':screen['away']['matched_negative_run_difference'],
                      'actual_home_margin':float(g.home_points-g.away_points),
                      'actual_total':float(g.home_points+g.away_points),
                      'seed':seed,'reference':stats(ref),'correction':stats(alt),
                      'unfitted_ppd_home':alt['mismatch_research']['ppd_home_delta'],
                      'unfitted_ppd_away':alt['mismatch_research']['ppd_away_delta']})
                else:
                    fail['candidate_or_reference_missing']+=1
            elif len(rows)<max_games:fail['insufficient_prior_games']+=1
            p=grouped.get(g.game_id)
            if p is None:
                fail['pbp_missing']+=1;continue
            h=observed_game(p,hid,aid,g.home_points,g.away_points,qb.get(hid,set()),
                            special_plays=rawgroup.get(g.game_id))
            a=observed_game(p,aid,hid,g.away_points,g.home_points,qb.get(aid,set()),
                            special_plays=rawgroup.get(g.game_id))
            if h is None or a is None:continue
            observedh=extract(p,hid,aid);observeda=extract(p,aid,hid)
            append_training_record(h,observedh);append_training_record(a,observeda)
            h['opponent_id']=aid;a['opponent_id']=hid
            pending.extend(((kickoff+pd.Timedelta(hours=24),hid,h),
                            (kickoff+pd.Timedelta(hours=24),aid,a)))
        if len(rows)>=max_games:break
    report={'schema':'cfb_failure1_fixed_negative_run_unseen_season_validation_v1',
       'captured_at_utc':str(cutoff),
       'source_season':int(games['_kickoff'].dt.year.mode().iloc[0]),
       'fixed_mode':'negative',
       'coefficient_policy':'FROZEN existing rush_severity_research.py as of 2026-10-22',
       'no_coefficient_search':True,'simulations_per_fixture_variant':runs,
       'actual_margin_games':len(rows),'excluded':dict(fail),
       'metrics':{'all':grade(rows),
             'severe_ex_ante':grade(rows,lambda r:r['severe_screen_ex_ante']),
             'ordinary':grade(rows,lambda r:not r['severe_screen_ex_ante']),
             'negative_advantage_home':grade(rows,lambda r:r['home_negative_rate_excess'] is not None and r['home_negative_rate_excess']<-.03),
             'negative_advantage_away':grade(rows,lambda r:r['away_negative_rate_excess'] is not None and r['away_negative_rate_excess']<-.03)},
       'replays':rows,'promotion_automatically_authorized':False,
       'requires_untouched_prospective_ATS':True,
       'uncertainties':['Retroactively captured rows are not point-in-time proof',
          'No independently timestamped historical ATS odds quotes for past seasons',
          'New season after old coefficient freeze may be untested',
          'Raw simulations uncalibrated; original scoring contains rushing EPA already',
          'No selection based on target-game rush yards or final result']}
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError('Research output already exists')
    p.write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True)
    p.add_argument('--runs',type=int,default=5000);p.add_argument('--max-games',type=int,default=180)
    a=p.parse_args()
    d=run(a.snapshot,a.out,max_games=a.max_games,runs=a.runs)
    print(json.dumps({k:v for k,v in d.items() if k!='replays'},indent=2))
