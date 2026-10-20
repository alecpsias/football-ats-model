"""User-requested fresh CFB matchup research and matching postgame audit.

Pregame NEVER reads old forecasts, games after kickoff, or final results for the
requested game. It builds a new projection from the verified CURRENT snapshot's
eligible prior games, and archives it. Only postgame reads that saved prediction.
CFB-only; no NFL dependencies. Missing injuries/market inputs stay explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from cfb.engine import simulate
from cfb.advanced_scoring import simulate_challenger
from cfb.game_script import simulate_script, adverse_scripts
from cfb.integrated_scoring import simulate_integrated
from cfb.market import _date
from cfb.matchups import matchup_signals
from cfb.matchup_cycle import _market, postgame as compare_postgame, markdown as legacy_markdown, normalized
from cfb.pipeline import (verify, split_games, normalize_pbp_teams, team_features,
                          missing_features, iso_utc)
from cfb.research_backtest import observed_game, project
from cfb.free_odds import verify_odds_archive
from cfb.learn_forward import investigate
from cfb.personnel import quarterback_usage, load_official_statuses, correlate_availability
from cfb.personnel_effects import (quarterback_effects, skill_player_exposures,
                                   simulate_personnel_scenarios, other_position_availability)

SCHEMA='cfb_fresh_requested_matchup_v1'


def _json(obj):
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=str)+'\n'


def resolve_fixture(future, team_a, team_b, game_id=None):
    """Resolve the pair from the CURRENT schedule, not an old forecast index."""
    if not normalized(team_a) or not normalized(team_b) or normalized(team_a)==normalized(team_b):
        raise ValueError('Supply two distinct, nonempty school names')
    hits=[]
    for _, row in future.iterrows():
        if game_id and str(row.game_id)!=str(game_id):continue
        if {normalized(row.home_team),normalized(row.away_team)}=={normalized(team_a),normalized(team_b)}:
            hits.append(row)
    if not hits:
        raise ValueError(f'No upcoming eligible scheduled matchup for {team_a} vs {team_b}. '
                         'Use exact school names from the current schedule; no guesswork.')
    if len(hits)!=1:
        raise ValueError('Multiple upcoming fixtures; provide --game-id: '+
                         ', '.join(sorted(str(x.game_id) for x in hits)))
    return hits[0]


def histories_asof(schedule, pbp, roster, cutoff):
    """Recompute prior-games-only observations and opponent quality at capture."""
    history, future, past_plays=split_games(schedule,pbp,cutoff)
    # A completed flag alone does not prove final publication at its kickoff.
    eligible=history.loc[history['_kickoff']+timedelta(hours=24)<=cutoff].copy()
    past_plays=past_plays.loc[past_plays.game_id.isin(eligible.game_id)]
    mapped=normalize_pbp_teams(past_plays, eligible)
    by_game={gid:group for gid,group in mapped.groupby('game_id',sort=False)}
    if not {'position_abbreviation','team_id','athlete_id'}.issubset(roster.columns):
        raise ValueError('Verified roster missing required QB identity columns')
    qb_ids={int(t):set(g.athlete_id.dropna()) for t,g in
            roster.loc[roster.position_abbreviation.eq('QB')].groupby('team_id')}
    histories=defaultdict(list)
    for _,g in eligible.iterrows():
        part=by_game.get(g.game_id)
        if part is None or pd.isna(g.home_points) or pd.isna(g.away_points):continue
        hid,aid=int(g.home_id),int(g.away_id)
        h=observed_game(part,hid,aid,g.home_points,g.away_points,qb_ids.get(hid,set()))
        a=observed_game(part,aid,hid,g.away_points,g.home_points,qb_ids.get(aid,set()))
        if h is None or a is None:continue
        h['opponent_id']=aid; a['opponent_id']=hid
        h['source_game_id']=str(g.game_id);a['source_game_id']=str(g.game_id)
        histories[hid].append(h);histories[aid].append(a)
    return future, mapped, histories, eligible


def quote_asof(root, game_id, stamp):
    """Select source-verified archived quote visible BY the new analysis time.

    Always filter individual observations by timestamp, rather than dropping
    older valid quotes when a later (future-at-analysis) quote exists. The
    Build 7 recovered archive retains its ORIGINAL observation timestamp.
    """
    if not root:return None
    folder=Path(root)
    archives=sorted(folder.glob('*.zip'))+sorted((folder/'recovered').glob('*.zip'))
    found=[]
    for archive in archives:
        index,rows=verify_odds_archive(archive)
        for row in rows:
            if row['game_id']!=str(game_id) or row['status']!='OBSERVED_COMPLETE_SPREAD_AND_BOTH_PRICES':
                continue
            at=_date(row['observed_at_utc'],'market observation')
            if at>stamp or at>=_date(row['kickoff_utc'],'market kickoff'):continue
            found.append((at,str(index['run_id']),{
                'home':row['home'],'away':row['away'],'kickoff_utc':row['kickoff_utc'],
                'observed_at_utc':row['observed_at_utc'],'quote':row['quote'],
                'market_archive_run_id':index['run_id'],
                'market_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                'book_timestamp_verified':False,
                'source_url':row['source_url'],
                'raw_source_sha256':row['raw_source_sha256'],
                'raw_provider_sha256':row['raw_provider_sha256']}))
    return max(found,key=lambda x:(x[0],x[1]))[2] if found else None


def sensitivity(hp, ap, game_id, runs=500):
    """Hypothetical +/-10% scoring and +/-1 possession stress tests.

    These are risk checks, NOT event probabilities, injury assumptions, or
    fitted alternate predictions. Main official research estimate is unchanged.
    """
    n=max(100,min(int(runs),1200))
    cases=[('one_fewer_possession_each',-1,1.,1.),
           ('away_ppd_ten_percent_better',0,1.,1.10),
           ('home_ppd_ten_percent_worse',0,.90,1.),
           ('home_ppd_ten_percent_better',0,1.10,1.)]
    out=[]
    for j,(label,d,hm,am) in enumerate(cases):
        h={**hp,'_ppd':max(.35,min(5.,hp['_ppd']*hm)),
                 'expected_drives':max(6,min(19,hp['expected_drives']+d))}
        a={**ap,'_ppd':max(.35,min(5.,ap['_ppd']*am)),
                 'expected_drives':max(6,min(19,ap['expected_drives']+d))}
        r=simulate(h,a,runs=n,seed=(int(game_id)+j+1)%(2**32))
        out.append({'scenario':label,'mean_home_margin':r['mean_margin_home'],
                    'mean_home_score':r['mean_score_home'],
                    'mean_away_score':r['mean_score_away']})
    return {'status':'HYPOTHETICAL_SENSITIVITY_ONLY_NOT_CALIBRATED',
            'definition':'One fewer possession each, or illustrative 10% PPD deviation',
            'scenarios':out}


def analyze(snapshot, team_a, team_b, game_id=None, *, odds_root=None, runs=3000,
            min_games=3, now=None, github_run_id=None, git_sha=None,
            availability_root=None):
    """Run a FRESH two-team model investigation from latest captured data.

    Does not access a frozen forecast archive at all. Source must have been
    collected on or before the analysis and before target kickoff.
    """
    if type(runs) is not int or not 100<=runs<=300000:
        raise ValueError('Invalid simulation run count')
    if type(min_games) is not int or not 1<=min_games<=15:
        raise ValueError('Invalid minimum prior-game count')
    root=Path(snapshot);manifest=verify(root)
    cutoff=iso_utc(manifest['captured_at_utc'])
    stamp=_date(now.isoformat() if isinstance(now,datetime) else
                (now if now is not None else datetime.now(timezone.utc).isoformat()),'analysis time')
    if stamp<cutoff:raise ValueError('Model forecast predates snapshot capture')
    if stamp-cutoff>timedelta(hours=4):
        raise ValueError('Snapshot is stale; recapture current CFB statistics')
    schedule=pd.read_parquet(root/'schedule.parquet')
    pbp=pd.read_parquet(root/'pbp.parquet')
    roster=pd.read_parquet(root/'roster.parquet')
    future,mapped,histories,eligible=histories_asof(schedule,pbp,roster,cutoff)
    g=resolve_fixture(future,team_a,team_b,game_id)
    kickoff=_date(g['_kickoff'].isoformat(),'kickoff')
    if stamp>=kickoff:raise ValueError('Cannot make a new pregame prediction after kickoff')
    hid,aid=int(g.home_id),int(g.away_id)
    # Full current observed evidence is recomputed now, never drawn from old model output.
    home_stats=team_features(hid,mapped,roster=roster)
    away_stats=team_features(aid,mapped,roster=roster)
    qbs={'home':quarterback_usage(mapped,roster,eligible,hid),
         'away':quarterback_usage(mapped,roster,eligible,aid)}
    availability=load_official_statuses(availability_root,str(g.game_id),
        kickoff.isoformat(),stamp.isoformat(),expected_teams=[str(hid),str(aid)])
    risks={'home':correlate_availability(qbs['home'],availability),
           'away':correlate_availability(qbs['away'],availability)}
    qb_effects={'home':quarterback_effects(mapped,eligible,roster,hid,qbs['home'],availability),
                'away':quarterback_effects(mapped,eligible,roster,aid,qbs['away'],availability)}
    skill_exposures={'home':skill_player_exposures(mapped,eligible,roster,hid,availability),
                     'away':skill_player_exposures(mapped,eligible,roster,aid,availability)}
    other_availability={'home':other_position_availability(roster,availability,hid),
                        'away':other_position_availability(roster,availability,aid)}
    manifest_sha=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    result={'schema':SCHEMA,'mode':'pregame',
        'status':'INSUFFICIENT_PRIOR_OBSERVATIONS_NO_PREDICTION',
        'game_id':str(g.game_id), 'home':str(g.home_team),'away':str(g.away_team),
        'home_id':str(hid),'away_id':str(aid),'kickoff_utc':kickoff.isoformat(),
        'neutral_site':bool(g.neutral_site) if pd.notna(g.neutral_site) else None,
        'prediction_generated_at_utc':stamp.isoformat(),
        'source_snapshot_at_utc':cutoff.isoformat(),
        'source_manifest_sha256':manifest_sha,
        'source_sha256':{k:v['sha256'] for k,v in manifest['datasets'].items()},
        'github_run_id':str(github_run_id) if github_run_id is not None else None,
        'git_commit_sha':git_sha,
        'analysis_source':'NEW_VERIFIED_CAPTURED_CFB_STATS_NOT_ARCHIVED_PREDICTION',
        'model_variant':'UNMODIFIED_UNCALIBRATED_CFB_BASELINE',
        'historical_games_eligible_after_24h_embargo':len(eligible),
        'completed_prior_games_home':len(histories[hid]),
        'completed_prior_games_away':len(histories[aid]),
        'prior_game_ids_home':[r['source_game_id'] for r in histories[hid]],
        'prior_game_ids_away':[r['source_game_id'] for r in histories[aid]],
        'current_observed_statistics':{'home':home_stats,'away':away_stats},
        'missing_statistical_fields':{'home':missing_features(home_stats),
                                       'away':missing_features(away_stats)},
        'injuries_verified':False,'ATS_calibrated':False,'grade':'NOT_CALIBRATED',
        'quarterback_participation':qbs,
        'availability_evidence':availability,
        'quarterback_availability_risks':risks,
        'empirical_qb_replacement_evidence':qb_effects,
        'skill_player_availability_exposure':skill_exposures,
        'other_position_availability':other_availability,
        'conditional_personnel_scenarios':None,
        'personnel_adjustment_to_model_points':None,
        'market':{'status':'NO_SOURCE_OBSERVED_PREKICKOFF_SPREAD','ats_research_side':None},
        'locked_market_evidence':None,
        'limitations':['Quarterback usage measured from prior games; next-game starters are NOT verified',
          'Only source-backed optional availability snapshots are displayed; no uncalibrated injury point deductions',
          'Unverified current injuries and projected lineups are NOT incorporated',
          'Full pressure, special teams and penalty effects are missing/incomplete',
          'Prior-game row publication times are not independently proven by a later snapshot',
          'Possession model is uncalibrated; score estimates and ATS difference are research only',
          'No result or forecast from the target game was used as a model input']}
    if len(histories[hid])>=min_games and len(histories[aid])>=min_games:
        neutral=bool(g.neutral_site) if pd.notna(g.neutral_site) else False
        hp=project(histories[hid],histories[aid],neutral,'home',False)
        ap=project(histories[aid],histories[hid],neutral,'away',False)
        signals=matchup_signals(hid,aid,histories)
        q=quote_asof(odds_root,str(g.game_id),stamp)
        qmap={str(g.game_id):q} if q else {}
        sim=simulate(hp,ap,runs=runs,seed=int(g.game_id)%(2**32),
                     spread_home=q['quote']['spread_home'] if q else None)
        advanced=simulate_challenger(hp,ap,signals,len(histories[hid]),
                     len(histories[aid]),runs=runs,seed=int(g.game_id)%(2**32))
        integrated=simulate_integrated(hid,aid,histories,neutral,
                     runs=runs,seed=int(g.game_id)%(2**32),
                     spread_home=q['quote']['spread_home'] if q else None)
        integrated_script=simulate_integrated(hid,aid,histories,neutral,
                     runs=runs,seed=int(g.game_id)%(2**32),use_script=True,
                     spread_home=q['quote']['spread_home'] if q else None)
        script=simulate_script(hp,ap,histories[hid],histories[aid],runs=runs,
                     seed=int(g.game_id)%(2**32),
                     spread_home=q['quote']['spread_home'] if q else None)
        personnel=simulate_personnel_scenarios(hp,ap,qb_effects['home'],qb_effects['away'],
                     runs=runs,seed=int(g.game_id)%(2**32))
        # _market validates fixture, source observation and model creation time.
        proxy={**result,'simulated_margin_home':sim['mean_margin_home']}
        market=_market(proxy,qmap)
        scripts_stress=adverse_scripts(hp,ap,histories[hid],histories[aid],
                    runs=max(100,min(600,runs//5)),seed=int(g.game_id)%(2**32))
        result.update({'status':'FRESH_PREGAME_PROJECTION_LOCKED',
                       'game_script_adverse_sensitivity':scripts_stress,
                       'game_script_research_challenger':script,
                       'integrated_scoring_research_challenger':integrated,
                       'integrated_scoring_plus_script_research':integrated_script,
                       'integrated_challengers_not_primary':True,
                       'game_script_challenger_not_primary':True,
                       'advanced_research_challenger':advanced,
                       'advanced_challenger_not_primary':True,
                       'conditional_personnel_scenarios':personnel,
                       'home_projection':hp,'away_projection':ap,
                       'matchup_evidence':signals,'simulation':sim,
                       'stress_tests':sensitivity(hp,ap,g.game_id),
                       'simulated_margin_home':sim['mean_margin_home'],
                       'simulated_score_home':sim['mean_score_home'],
                       'simulated_score_away':sim['mean_score_away'],
                       'model_projected_winner':result['home'] if sim['mean_margin_home']>0
                           else result['away'] if sim['mean_margin_home']<0 else 'TIE',
                       'market':market,'locked_market_evidence':q})
    if now is None and datetime.now(timezone.utc)>=kickoff:
        raise ValueError('Kickoff occurred during analysis; refusing hindsight publication')
    return result


def select_saved_request(root, team_a, team_b, game_id=None):
    """Postgame ONLY: retrieve earliest archived USER-REQUESTED pregame report."""
    candidates=[]
    folder=Path(root)
    for file in folder.glob('*.json') if folder.is_dir() else []:
        raw=file.read_bytes()
        try:p=json.loads(raw)
        except (ValueError,TypeError):continue
        if p.get('schema')!=SCHEMA or p.get('mode')!='pregame':continue
        if p.get('status')!='FRESH_PREGAME_PROJECTION_LOCKED':continue
        if game_id and str(p.get('game_id'))!=str(game_id):continue
        if ({normalized(p.get('home')),normalized(p.get('away'))} !=
                {normalized(team_a),normalized(team_b)}):continue
        generated=_date(p['prediction_generated_at_utc'],'pregame timestamp')
        kickoff=_date(p['kickoff_utc'],'kickoff timestamp')
        if generated>=kickoff:continue
        candidates.append((kickoff,generated,str(p['game_id']),file.name,p,hashlib.sha256(raw).hexdigest()))
    if not candidates:raise ValueError('No saved, pre-kickoff USER-REQUESTED analysis for these two teams')
    ids={x[2] for x in candidates}
    if len(ids)>1:raise ValueError('Multiple games found; provide --game-id: '+', '.join(sorted(ids)))
    earliest=min(candidates,key=lambda x:(x[1],x[3]))
    return earliest[4],earliest[5],earliest[3]


def review(saved, saved_sha, saved_file, snapshot):
    """Compare ORIGINAL requested estimate to AFTER-game stats; never rerun model."""
    if saved.get('schema')!=SCHEMA or saved.get('status')!='FRESH_PREGAME_PROJECTION_LOCKED':
        raise ValueError('Only an original fresh-user-request pregame report can be reviewed')
    root=Path(snapshot);manifest=verify(root)
    cutoff=iso_utc(manifest['captured_at_utc'])
    if cutoff<=_date(saved['kickoff_utc'],'kickoff'):
        raise ValueError('Postgame snapshot not captured after the game started')
    schedule=pd.read_parquet(root/'schedule.parquet')
    pbp=pd.read_parquet(root/'pbp.parquet')
    roster=pd.read_parquet(root/'roster.parquet')
    # Legacy postgame statistics engine accepts the newly generated original
    # projection, but NEVER reuses a batch forecast or new after-game prediction.
    p={**saved,'status':'FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE',
       'original_forecast_run_id':saved.get('github_run_id') or 'DIRECT_REQUEST',
       'original_forecast_zip_sha256':saved_sha,
       'model_variant':saved['model_variant']}
    q=saved.get('locked_market_evidence')
    quotes={saved['game_id']:q} if q else {}
    report=compare_postgame(p,quotes,schedule,pbp,roster,cutoff.isoformat(),
                  hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())
    # Compare the SHA-locked challenger from the pregame file, never rerun it.
    advanced=saved.get('advanced_research_challenger')
    if report.get('status')=='FINAL_REVIEW_RECORDED' and isinstance(advanced,dict) and advanced.get('status')=='AVAILABLE_UNCALIBRATED_RESEARCH_ONLY':
        candidate=advanced.get('simulated_margin_home')
        actual=report['observed_final']['home_margin']
        if isinstance(candidate,(int,float)) and math.isfinite(float(candidate)):
            primary=float(saved['simulated_margin_home'])
            report['original_advanced_challenger_comparison']={
                'status':'LOCKED_RESEARCH_CHALLENGER_EVALUATED',
                'model_version':advanced.get('version'),
                'frozen_projected_home_margin':float(candidate),
                'actual_home_margin':float(actual),
                'baseline_absolute_error':round(abs(primary-actual),5),
                'advanced_absolute_error':round(abs(float(candidate)-actual),5),
                'not_deployed':True}
    script=saved.get('game_script_research_challenger')
    if report.get('status')=='FINAL_REVIEW_RECORDED' and isinstance(script,dict) and script.get('status')=='UNFITTED_GAME_SCRIPT_CHALLENGER_NOT_PRIMARY':
        cand=script.get('mean_margin_home')
        if isinstance(cand,(int,float)) and math.isfinite(float(cand)):
            actual=float(report['observed_final']['home_margin'])
            primary=float(saved['simulated_margin_home'])
            interval=script.get('margin_p10_p50_p90') or []
            report['original_game_script_comparison']={
                'status':'ORIGINAL_FROZEN_CANDIDATE_EVALUATED',
                'version':script.get('version'),
                'frozen_projected_home_margin':float(cand),
                'actual_home_margin':actual,
                'baseline_absolute_error':round(abs(primary-actual),5),
                'game_script_absolute_error':round(abs(float(cand)-actual),5),
                'inside_candidate_p10_p90':(float(interval[0])<=actual<=float(interval[2]) if len(interval)==3 else None),
                'recomputed_after_game':False,'not_deployed':True}
    # Compare only the original pregame archived integrated scoring estimates.
    if report.get('status')=='FINAL_REVIEW_RECORDED':
        for key,outkey in (('integrated_scoring_research_challenger','original_integrated_scoring_comparison'),
                            ('integrated_scoring_plus_script_research','original_integrated_script_comparison')):
            z=saved.get(key)
            if isinstance(z,dict) and z.get('status')=='AVAILABLE_UNFITTED_INTEGRATED_RESEARCH_ONLY':
                cand=z.get('mean_margin_home')
                if isinstance(cand,(int,float)) and math.isfinite(float(cand)):
                    actual=float(report['observed_final']['home_margin'])
                    primary=float(saved['simulated_margin_home'])
                    report[outkey]={'version':z.get('version'),'frozen_projected_home_margin':float(cand),
                         'actual_home_margin':actual,'baseline_absolute_error':round(abs(primary-actual),5),
                         'integrated_absolute_error':round(abs(float(cand)-actual),5),
                         'recomputed_after_game':False,'not_primary':True}
    # Frozen conditional personnel scenarios are not picks: the model did not
    # predict or attest who started. Record what was known, never retro-select a
    # matching alternate after observing the box score.
    evidence=saved.get('conditional_personnel_scenarios')
    if isinstance(evidence,dict):
        report['original_personnel_scenarios']={
            'status':'PRE_GAME_CONDITIONAL_ONLY_NOT_GRADED_AS_FORECAST',
            'version':evidence.get('version'),
            'scenario_count':len(evidence.get('scenarios',[])),
            'personnel_score_recomputed_after_game':False,
            'requires_proven_pregame_starting_lineup_to_measure_forecast_accuracy':True}
    report.update({'schema':'cfb_fresh_requested_postgame_v1',
                   'original_user_requested_pregame_file':saved_file,
                   'original_user_requested_pregame_sha256':saved_sha,
                   'original_pregame_source_manifest_sha256':saved['source_manifest_sha256'],
                   'original_pregame_information_cutoff_utc':saved['source_snapshot_at_utc'],
                   'postgame_does_not_rerun_the_original_model':True})
    return report


def requested_learning(review_folder, current=None):
    """Only genuine completed USER-REQUESTED games; no one-game promotions."""
    rows={}
    for file in sorted(Path(review_folder).glob('*.json')) if Path(review_folder).is_dir() else []:
        try:z=json.loads(file.read_text())
        except (ValueError,TypeError):continue
        if z.get('schema')!='cfb_fresh_requested_postgame_v1' or z.get('status')!='FINAL_REVIEW_RECORDED':continue
        gid=str(z['game_id'])
        try:
            pred=float(z['locked_original_forecast']['projected_home_margin'])
            actual=float(z['observed_final']['home_margin'])
            ts=_date(z['locked_original_forecast']['generated_at_utc'],'original generated')
            kickoff=_date(z['kickoff_utc'],'kickoff')
            if not ts<kickoff or not all(math.isfinite(x) for x in (pred,actual)):continue
        except (KeyError,TypeError,ValueError):continue
        rows.setdefault(gid,{'game_id':gid,'kickoff_utc':kickoff.isoformat(),
                             'projected_margin':pred,'actual_margin':actual})
    if current and current.get('schema')=='cfb_fresh_requested_postgame_v1' and current.get('status')=='FINAL_REVIEW_RECORDED':
        x=current;gid=str(x['game_id'])
        if gid not in rows:
            pred=float(x['locked_original_forecast']['projected_home_margin'])
            actual=float(x['observed_final']['home_margin'])
            created=_date(x['locked_original_forecast']['generated_at_utc'],'prediction')
            kickoff=_date(x['kickoff_utc'],'kickoff')
            if created<kickoff and math.isfinite(pred) and math.isfinite(actual):
                rows[gid]={'game_id':gid,'kickoff_utc':kickoff.isoformat(),
                           'projected_margin':pred,'actual_margin':actual}
    return investigate(list(rows.values()))


def markdown(report):
    if report['mode']=='postgame':
        output=legacy_markdown(report)+('\nOriginal user-requested file: '+
                   report['original_user_requested_pregame_file']+'\n')
        p=report.get('original_personnel_scenarios')
        if p:
            output+=('\n## Frozen personnel scenarios\n'
                      f"{p['scenario_count']} conditional QB scenario(s) archived. "
                      'Not graded as predictions because next-game starters were unverified.\n')
        c=report.get('original_advanced_challenger_comparison')
        if c:
            output+=('\n## Original research challenger comparison\n'
                f"Baseline absolute margin error: {c['baseline_absolute_error']:.2f}\n"
                f"Advanced challenger absolute margin error: {c['advanced_absolute_error']:.2f}\n"
                'Based only on pregame-saved challenger. No model promotion.\n')
        g=report.get('original_game_script_comparison')
        if g:
            output+=('\n## Original game-script research challenger\n'
                f"Baseline margin error: {g['baseline_absolute_error']:.2f}\n"
                f"Script challenger margin error: {g['game_script_absolute_error']:.2f}\n"
                'Compared using the original pregame record only. No automatic promotion.\n')
        for k in ('original_integrated_scoring_comparison','original_integrated_script_comparison'):
            c=report.get(k)
            if c:
                output+=(f"\n## {k.replace('_',' ').title()} (frozen only)\n"
                         f"Baseline margin error: {c['baseline_absolute_error']:.2f}; "
                         f"integrated error: {c['integrated_absolute_error']:.2f}.\n")
        return output
    if report['status']!='FRESH_PREGAME_PROJECTION_LOCKED':
        return '# CFB fresh matchup — insufficient observed data\n\n'+(
            f"{report['away']} at {report['home']}; prior games: "
            f"{report['completed_prior_games_away']} away, {report['completed_prior_games_home']} home.\n")
    m=report['market']
    quote=('No valid market quote before analysis' if m.get('home_spread') is None else
           f"{report['home']} {m['home_spread']:+g} ({m['provider']}); "
           f"research ATS side {m['ats_research_side']}")
    h=report['current_observed_statistics']['home']
    a=report['current_observed_statistics']['away']
    def show(z):return 'unavailable' if z is None else f'{z:.3f}' if isinstance(z,(int,float)) else str(z)
    lines=[f"# NEW CFB pregame investigation: {report['away']} at {report['home']}",
           '',f"Snapshot captured: {report['source_snapshot_at_utc']}",
           f"Historical games: {report['home']} {report['completed_prior_games_home']}, "
           f"{report['away']} {report['completed_prior_games_away']}",
           '',f"**Projected score:** {report['home']} {report['simulated_score_home']:.1f}, "
           f"{report['away']} {report['simulated_score_away']:.1f}",
           f"**Home margin:** {report['simulated_margin_home']:+.2f}",
           f"**Market:** {quote}. Confidence not calibrated.",
           '', '## Current opponent-context team statistics', '',
           f"| Metric | {report['home']} | {report['away']} |",'|---|---:|---:|']
    advanced=report.get('advanced_research_challenger',{})
    if advanced.get('simulation') is not None:
        lines += ['', '## Opponent-adjusted experimental challenger (NOT the primary pick)',
                  f"**Challenger score:** {report['home']} {advanced['simulated_score_home']:.1f}, "
                  f"{report['away']} {advanced['simulated_score_away']:.1f}",
                  f"**Challenger home margin:** {advanced['simulated_margin_home']:+.2f}",
                  'Uses leave-opponent-out pass/rush EPA, explosives, third down, QB-hurry proxy, '
                  'red-zone finishing, and field position where recorded.',
                  'Experimental coefficients are NOT validated; do not confuse this with the primary forecast.']
    else:
        lines += ['', '## Opponent-adjusted research challenger',
                  'Not scored: '+str(advanced.get('status','NO_RESULT'))]
    integrated=report.get('integrated_scoring_research_challenger') or {}
    if integrated.get('simulation') is not None:
        lines += ['', '## Integrated NFL-style CFB scoring challenger (research only)',
                  f"Integrated score: {report['home']} {integrated['mean_score_home']:.1f}, "
                  f"{report['away']} {integrated['mean_score_away']:.1f}; "
                  f"home margin {integrated['mean_margin_home']:+.2f}",
                  'Opponent-adjusted passing/rushing EPA drives scoring levels; '
                  'limited situational residuals and matched turnover effects. Unfitted.']
    else:lines += ['', '## Integrated CFB scoring challenger',
                  'Not scored: '+str(integrated.get('status','NOT_RUN'))]
    plus=report.get('integrated_scoring_plus_script_research') or {}
    if plus.get('simulation') is not None:
        lines += ['', '### Integrated scoring + Build 14 game script (research only)',
                  f"Combined home margin {plus['mean_margin_home']:+.2f}. Not the primary pick."]
    script=report.get('game_script_research_challenger') or {}
    if script.get('status')=='UNFITTED_GAME_SCRIPT_CHALLENGER_NOT_PRIMARY':
        lines += ['', '## Game-script distribution challenger (NOT the primary forecast)',
                  f"Script projected score: {report['home']} {script['mean_score_home']:.1f}, "
                  f"{report['away']} {script['mean_score_away']:.1f}",
                  f"Script home margin: {script['mean_margin_home']:+.2f}; "
                  f"p10/p50/p90 {script['margin_p10_p50_p90']}",
                  'Possession volatility, turnover short fields, shared scoring conditions, '
                  'and hypothetical halftime clock response. Unfitted; research only.']
        for name,item in script.get('scenario_distributions',{}).items():
            lines.append(f"- {name}: {item['share']:.1%} of draws, conditional home margin {item['mean_home_margin']}")
    stress=report.get('game_script_adverse_sensitivity') or {}
    if stress.get('scenarios'):
        lines += ['', '### Conditional game-script stress tests (not probabilities)']
        for name,scenario in stress['scenarios'].items():
            lines.append(f"- {name}: home margin {scenario['mean_home_margin']:+.2f}")
    personnel=report.get('conditional_personnel_scenarios') or {}
    lines += ['', '## Personnel replacement evidence (conditional, NOT the primary forecast)']
    for side in ('home','away'):
        x=(report.get('empirical_qb_replacement_evidence') or {}).get(side,{})
        if x.get('scenario'):
            s=x['scenario']
            lines.append(f"{report[side]}: IF {s['candidate_qb']} replaces recent volume leader "
                         f"{s['reference_qb']}, estimated PPD shift "
                         f"{s['estimated_change_in_points_per_drive']:+.3f} (research only)")
        else: lines.append(f"{report[side]}: no qualified replacement comparison "
                          f"({x.get('reason',x.get('status'))})")
    for case in personnel.get('scenarios',[]):
        lines.append(f"IF {case['candidate_qb']} assumes QB role: conditional home margin "
                     f"{case['candidate_home_margin']:+.2f}; NOT a starter forecast")
    for side in ('home','away'):
        flagged=(report.get('other_position_availability') or {}).get(side,{}).get('players',[])
        for player in flagged:
            lines.append(f"Reported {report[side]} {player['position']} {player['name']}: "
                         f"{player['reported_status']} (source not independently verified; "
                         'point impact not estimated)')
    for label,group,key in (
        ('Offensive pass EPA/play','offense','epa_pass'),
        ('Offensive rush EPA/play','offense','epa_rush'),
        ('Defensive pass EPA allowed/play','defense','epa_pass_allowed'),
        ('Defensive rush EPA allowed/play','defense','epa_rush_allowed'),
        ('Offensive pass success rate','offense','success_pass'),
        ('Offensive rush success rate','offense','success_rush'),
        ('Offensive explosive pass rate','offense','explosive_pass_rate'),
        ('Offensive explosive rush rate','offense','explosive_rush_rate'),
        ('Offensive third-down conversion','offense','third_down_rate'),
        ('Offensive red-zone TD rate','offense','redzone_td_rate'),
        ('Offensive drives/game','offense','drives_per_game'),
        ('Sack allowed / pass play','offense','sack_allowed'),
        ('Pass-defense success allowed','defense','success_pass_allowed'),
        ('Rush-defense success allowed','defense','success_rush_allowed')):
        lines.append(f"| {label} | {show(h[group].get(key))} | {show(a[group].get(key))} |")
    lines += ['', '## Quarterback involvement — NOT a confirmed depth chart', '']
    for side in ('home','away'):
        team=report[side]
        qbr=report.get('quarterback_participation',{}).get(side,{})
        lines.append(f'### {team}')
        if qbr.get('status')!='QB_PARTICIPATION_OBSERVED_NOT_STARTER_CONFIRMED':
            lines.append('Unavailable: '+str(qbr.get('status')))
        else:
            lines.append(f"Identified QB pass-play coverage: {qbr.get('identified_roster_qb_pass_play_coverage')}; "
                         'passer leader is NOT confirmed starter.')
            for qb in qbr.get('quarterbacks',[])[:3]:
                lines.append(f"- {qb['name']} ({qb['athlete_id']}): last "
                             f"{qbr['recent_games_window']} games: "
                             f"{qb['recent_recorded_pass_plays']} pass plays, "
                             f"share {qb['recent_share_of_all_team_pass_plays']}, "
                             f"mean EPA {qb['recent_pass_play_epa_avg']}, "
                             f"QB rushes {qb['recorded_qb_rushes']}")
    av=report.get('availability_evidence',{})
    lines += ['', '## Availability source evidence', '',
              'Report status: '+str(av.get('status'))]
    if av.get('source_url'):lines.append('Source URL: '+av['source_url'])
    for player in av.get('players',[]):
        lines.append(f"- {player['name']}: {player['status']} "
                     '(literal text in user-supplied source snapshot; original capture not independently attested)')
    for side in ('home','away'):
        for risk in report.get('quarterback_availability_risks',{}).get(side,[]):
            lines.append(f"**QB risk:** {risk['name']} {risk['status']}; "
                         'point impact unknown, not silently adjusted.')
    lines += ['', '## Opponent-adjusted signals', '',
              'Detailed matchup interactions and source-derived schedules are in the JSON.',
              '', '## Stress checks (hypothetical, not likely outcomes)', '']
    for t in report['stress_tests']['scenarios']:
        lines.append(f"- {t['scenario']}: projected home margin {t['mean_home_margin']:+.1f}")
    lines += ['', '**Known gaps:** Verified current starting QBs and full injury reports, pressure, '
              'special teams, penalties, and calibrated ATS probability are not integrated.',
              '', 'The model has calculated a NEW score projection; it did not look up an earlier prediction.']
    return '\n'.join(lines)+'\n'


def run(args):
    if args.mode=='pregame':
        data=analyze(args.snapshot,args.team_a,args.team_b,args.game_id,
                     odds_root=args.odds,runs=args.runs,min_games=args.min_games,
                     availability_root=getattr(args,'availability_dir',None),
                     github_run_id=args.github_run_id,git_sha=args.git_sha)
    else:
        p,sha,file=select_saved_request(args.reviews,args.team_a,args.team_b,args.game_id)
        data=review(p,sha,file,args.snapshot)
        data['forward_learning_research']=requested_learning(args.reviews,data)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    md=out.with_suffix('.md')
    if out.exists() or md.exists():raise FileExistsError('Cannot overwrite saved requested model analysis')
    out.write_text(_json(data),encoding='utf8')
    md.write_text(markdown(data),encoding='utf8')
    if data['mode']=='pregame':
        brief={k:v for k,v in data.items() if k not in
             ('current_observed_statistics','missing_statistical_fields','home_projection',
              'away_projection','simulation','matchup_evidence','locked_market_evidence')}
    else:
        brief={k:v for k,v in data.items() if k!='observed_game_statistics'}
    print(_json(brief))
    return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',required=True,choices=('pregame','postgame'))
    ap.add_argument('--team-a',required=True);ap.add_argument('--team-b',required=True)
    ap.add_argument('--game-id');ap.add_argument('--snapshot',required=True)
    ap.add_argument('--reviews',default='cfb/archive/reviews')
    ap.add_argument('--odds',default='cfb/archive/odds')
    ap.add_argument('--availability-dir',default='cfb/availability')
    ap.add_argument('--out',required=True)
    ap.add_argument('--runs',type=int,default=3000)
    ap.add_argument('--min-games',type=int,default=3)
    ap.add_argument('--github-run-id');ap.add_argument('--git-sha')
    args=ap.parse_args();run(args)

if __name__=='__main__':main()
