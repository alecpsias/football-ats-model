#!/usr/bin/env python3
"""Auditable unpriced NFL pregame projection for FUTURE games only (v0.7).

Never supplies invented sportsbook prices, claim of injury clearance, or
empirically calibrated betting probabilities. Neutral PERSONNEL SCENARIO is
an explicit assumption and not a verified injury adjustment.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from ingest import build, load_table, normalize_schedule
from nflverse_connector import verify
from integrate_advanced_features import build_overlay, attach_overlay, verified_snapshot
from collect_extra_metrics import parse_stamp
from model import MODEL_RULES, InputError, validate_team, projection, simulate
from mechanism_research import extract_mechanisms
from nfl_stateful_candidate import stateful_candidate_simulation
from nfl_qb_adjustments import estimate_qb_adjustments
from nfl_challenger_policy import (ACTIVE_CHALLENGER_VERSION, BASELINE_VERSION,
                                   require_single_active_challenger)

# All source capture and forecast generation must occur before the normal
# ~90-minute inactive announcement window. This is a policy cutoff, not
# a claim that each actual announcement time is known.
PREGAME_LOCK_MINUTES = 120

def enforce_pregame_lock(cutoff, kickoff, lock_minutes=PREGAME_LOCK_MINUTES):
    """Reject any data snapshot within the last lock_minutes before kickoff."""
    as_of=parse_stamp(cutoff)
    start=parse_stamp(kickoff)
    if as_of > start - timedelta(minutes=lock_minutes):
        raise InputError(f"PRE_INACTIVES_LOCK: data captured too late; require at least {lock_minutes} minutes before kickoff")
    return True
from measure_player_availability import measure, scenario_points

NEUTRAL_PERSONNEL={'pass_epa_shift':0.,'rush_epa_shift':0.,'pressure_allowed_shift':0.,
                   'qb_rush_multiplier':1.,'pace_multiplier':1.,'notes':'Neutral scenario only, NOT a verified active roster',
                   'source':'UNVERIFIED_NEUTRAL_SCENARIO','checked_at_utc':None}

def available_time(snapshot:Path, extra:Path, injuries:Path|None=None):
    base,metrics,_=verified_snapshot(snapshot,extra,metrics_time(extra))
    times=[base['datasets'][k]['captured_at_utc'] for k in ('pbp','schedule','roster')]
    times += [metrics['finalized_at_utc']]
    if injuries and (injuries/'injury_manifest.json').is_file():
        im=json.loads((injuries/'injury_manifest.json').read_text())
        times.append(im['retrieved_at_utc'])
    return max(parse_stamp(t) for t in times)

def metrics_time(extra:Path):
    return json.loads((extra/'metrics_manifest.json').read_text())['finalized_at_utc']

def injuries_for_game(folder:Path|None, home:str,away:str,week:int,deadline):
    result={'source':'nflverse weekly injury report (not official inactive confirmation)',
            'status':'NO_INJURY_FEED','confirmed_inactives':None,'personnel_point_adjustments_applied':False,
            'team_report_rows':{home:None,away:None}}
    if not folder or not (folder/'injury_manifest.json').is_file():return result
    m=json.loads((folder/'injury_manifest.json').read_text())
    retrieved=parse_stamp(m['retrieved_at_utc'])
    if retrieved>deadline:
        result['status']='POST_CUTOFF_INJURY_FEED_REJECTED';return result
    accepted=('COLLECTED','CAPTURED_SNAPSHOT_ONLY')
    if m['status'] not in accepted:
        result['status']='INJURY_FEED_UNAVAILABLE';result['warning']=m.get('warning');return result
    f=folder/m['file'];actual=hashlib.sha256(f.read_bytes()).hexdigest()
    if actual!=m['sha256']:raise InputError('Injury file failed SHA-256 verification')
    result['status']=('REPORT_ROWS_SNAPSHOT_ONLY_NOT_ROW_TIMESTAMP_VERIFIED'
                      if m['status']=='CAPTURED_SNAPSHOT_ONLY'
                      else 'REPORT_ROWS_OBSERVED_NOT_CONFIRMED_INACTIVES')
    result['row_timestamp_provenance']=m.get('timestamp_provenance','LEGACY_ROW_TIME')
    result['captured_at_utc']=m['retrieved_at_utc']
    for rec in m['observations']:
        if rec['team'] in (home,away) and int(rec['week'])==int(week):
            result['team_report_rows'][rec['team']]=int(rec['game_status_rows'])
    return result

def quote_reference(row:pd.Series,captured_at):
    """nflverse schedule spread_line is unsigned; NEVER infer a bet side."""
    keys=('spread_line','home_spread_odds','away_spread_odds','total_line')
    values={}
    for k in keys:
        v=row.get(k)
        values[k]=float(v) if pd.notna(v) and isinstance(v,(int,float)) and math.isfinite(float(v)) else None
    return {'source':'nflverse schedule feed: unverified original quote time/book',
            'first_observed_in_our_snapshot_at_utc':captured_at,
            **values,'signed_home_spread':None, 'executable_quote_verified':False,
            'note':'spread_line is unsigned; snapshot observation does not prove bookmaker quote age. Never calculate EV or cover side from it.'}

def _assemble(snapshot:Path, extra:Path, row:pd.Series, cutoff, *, pbp=None,schedule=None,roster=None):
    b=json.loads((snapshot/'manifest.json').read_text())
    home,away=str(row['home_team']),str(row['away_team'])
    kickoff=row['kickoff_utc'].to_pydatetime()
    if cutoff>=kickoff:raise InputError('Snapshot at/after kickoff')
    where=str(row.get('location','')).strip().lower()
    if where not in ('home','neutral'):raise InputError('Home-site/neutral-site flag unverified')
    config={'id':str(row['game_id']),'kickoff_utc':kickoff.isoformat(),
        'data_as_of_utc':cutoff.isoformat(),'home_code':home,'away_code':away,
        'home_name':home,'away_name':away,'neutral_site':where=='neutral',
        'pbp_source':'nflverse locally captured SHA256 snapshot '+str(snapshot),
        'pbp_source_available_at_utc':b['datasets']['pbp']['captured_at_utc'],
        'roster_source_available_at_utc':b['datasets']['roster']['captured_at_utc'],
        'market':{'spread_home':None,'home_odds_american':None,'away_odds_american':None,
                  'total':None,'as_of_utc':None,'source':None},
        'teams':{home:{'personnel':{}},away:{'personnel':{}}}}
    if pbp is None:pbp=load_table(snapshot/b['datasets']['pbp']['local_file'])
    if schedule is None:schedule=load_table(snapshot/b['datasets']['schedule']['local_file'])
    if roster is None:roster=load_table(snapshot/b['datasets']['roster']['local_file'])
    g,audit=build(pbp,schedule,config,participation=None,roster=roster)
    overlay=build_overlay(snapshot,extra,cutoff.isoformat(),(home,away))
    g,audit=attach_overlay(g,audit,overlay,{'home':home,'away':away})
    stat_issues=[x for x in audit['asof_issues'] if not x.startswith('market.as_of_utc:')]
    if stat_issues:
        raise InputError('Strict statistical provenance failed: '+str(stat_issues))
    # Neutral personnel scenarios are NOT observed roster status, and must be
    # documented as a counterfactual assumption. Only these technical copies
    # are used to call the scoring calculation; original audit retains gaps.
    for side in ('home','away'):
        t=g[side]
        for k in ('offense','defense','schedule'):
            if any(v is None for v in t[k].values()):
                raise InputError(f'{side}.{k} has unavailable essential statistics')
        t['personnel']=dict(NEUTRAL_PERSONNEL,checked_at_utc=cutoff.isoformat())
        validate_team(t,side)
        if t.get('pressure_source')!='PFR_QB_TIMES_PRESSURED_OVER_SAME_GAME_PBP_DROPBACKS':
            raise InputError(f'{side}: missing verified PFR pressure; core model requires consistent rate definition')
    ref=g['pressure_reference_rate']
    if ref is None or not .10<=float(ref)<=.65:raise InputError('Missing verified league PFR pressure reference')
    return g,audit,overlay

def forecast_one(snapshot:Path,extra:Path,row:pd.Series, cutoff,injuries=None,runs:int=5000,seed:int=20261018,pbp=None,schedule=None,roster=None):
    enforce_pregame_lock(cutoff,row['kickoff_utc'])
    g,audit,overlay=_assemble(snapshot,extra,row,cutoff,pbp=pbp,schedule=schedule,roster=roster)
    r=dict(MODEL_RULES['NFL']);r['pressure']=float(g['pressure_reference_rate'])
    hfa=0 if g['neutral_site'] else r['home_advantage_points']
    h=projection(g['home'],g['away'],r,'home',.5*hfa)
    a=projection(g['away'],g['home'],r,'away',-.5*hfa)
    sim=simulate(h,a,r,None,None,runs,seed)
    # Exactly one ACTIVE challenger: update this implementation in place in
    # future versions. v0.9 mechanisms are retained as features, not as a
    # competing simulated forecast. The baseline above remains untouched.
    if pbp is None or schedule is None:
        manifest=json.loads((snapshot/'manifest.json').read_text())
        if pbp is None:pbp=load_table(snapshot/manifest['datasets']['pbp']['local_file'])
        if schedule is None:schedule=load_table(snapshot/manifest['datasets']['schedule']['local_file'])
    challenger_candidate=None
    challenger_profile=None
    qb_adjustments = None
    challenger_status='UNAVAILABLE_RESEARCH_ONLY'
    challenger_reason=None
    try:
        challenger_profile=extract_mechanisms(
            pbp,schedule,cutoff,(g['home']['name'],g['away']['name']))
        qb_adjustments=estimate_qb_adjustments(
            pbp,schedule,cutoff,g['home']['name'],g['away']['name'],
            {g['home']['name']:h,g['away']['name']:a})
        challenger_candidate=stateful_candidate_simulation(
            h,a,challenger_profile,r,runs,seed,qb_adjustments=qb_adjustments)
        challenger_status='AVAILABLE_RESEARCH_ONLY'
    except (InputError, ValueError, KeyError, ArithmeticError) as exc:
        # Missing challenger features cannot contaminate the valid baseline.
        challenger_reason=str(exc)
    # Player evidence is source-verified and conditional QB impacts never silently
    # override the recorded neutral baseline forecast.
    personnel_evidence=measure(snapshot,injuries,cutoff,g['home']['name'],g['away']['name'],int(row['week']),pbp=pbp,roster=roster,metrics_dir=extra)
    conditional_scenarios=scenario_points(g,r,personnel_evidence,runs,seed)
    # Remove the internal scenario-only placeholder fields from user-visible results.
    def clean(p):
        return {k:v for k,v in p.items() if not k.startswith('_') and k not in ('personnel_adjustment_source','personnel_news_checked_at_utc','personnel_notes')}
    result={'schema':'football_ats_future_core_v0.8','game_id':g['id'],'home':g['home']['name'],'away':g['away']['name'],
       'generated_at_utc':datetime.now(timezone.utc).isoformat(),
       'execution':{'github_run_id':os.environ.get('GITHUB_RUN_ID'),
                    'git_commit_sha':os.environ.get('GITHUB_SHA')},
       'kickoff_utc':g['kickoff_utc'],'data_as_of_utc':g['data_as_of_utc'],
       'pre_inactives_lock_minutes':PREGAME_LOCK_MINUTES,
       'status':'EXPLORATORY_CORE_PROJECTION_UNCALIBRATED',
       'score_home':sim['mean_score_home'],'score_away':sim['mean_score_away'],
       'baseline_simulation':sim,
       'projected_home_margin':sim['mean_margin_home'],
       'projected_fair_spread_home':round(-sim['mean_margin_home'],2),
       'home_win_probability_RAW_UNCALIBRATED':sim['home_win_probability_raw'],
       'margin_interval_p10_p50_p90':sim['margin_p10_p50_p90'],
       'market_reference':quote_reference(row,cutoff.isoformat()),
       'spread_side':'NOT_EVALUATED_NO_VERIFIED_SIGNED_LINE','ats_roi':None,
       'team_projections':{'home':clean(h),'away':clean(a)},
       'injuries':injuries_for_game(injuries,g['home']['name'],g['away']['name'],int(row['week']),cutoff),
       'player_availability_evidence':personnel_evidence,
       'conditional_qb_scenarios':conditional_scenarios,
       'challenger':{'version':ACTIVE_CHALLENGER_VERSION,'candidate':challenger_candidate,
           'observations':challenger_profile,'qb_adjustments':qb_adjustments,
           'status':challenger_status,
           'missing_reason':challenger_reason,
           'baseline_locked_margin_home':sim['mean_margin_home'],
           'comparison_status':'ONLY_ACTIVE_CHALLENGER_SHRUNK_QB_FIT_NOT_ATS_VALIDATED; NO ATS EDGE'},
       'personnel_analysis_version':'0.8.0',
       'limitations':['No verified active/inactive lineup; all personnel shifts set to NEUTRAL SCENARIO, NOT measured zero impact',
           'No verified signed market spread; no ATS recommendation or EV',
           'Provisional model coefficients and raw probabilities not historically calibrated',
           'QB replacement scenarios are conditional and not the main forecast; no starter/inactive confirmation',
           'Exactly one active evolving challenger, v0.11, incorporating earlier mechanisms and shrunk QB form',
           'v0.11 QB form is inferred from previous completed game, not confirmed current starter; no in-game injury adjustment or calibrated ATS edge'],
       'audit':{'snapshot':str(snapshot),'derived_metrics':str(extra),'source_sha256':overlay['source_sha256'],
            'pressure_rate_reference':ref_to_number(g['pressure_reference_rate']),
            'advanced_missing':overlay['missing'],'statistics_provenance_issues':[x for x in audit['asof_issues'] if not x.startswith('market.as_of_utc:')],
            'model_version':f'{BASELINE_VERSION}+single_challenger_{ACTIVE_CHALLENGER_VERSION}','simulation_runs':runs,'seed':seed}}

    require_single_active_challenger(result)
    return result

def ref_to_number(x):return float(x)

def generate(snapshot:Path,extra:Path,out:Path,injuries:Path|None=None,window_hours:float=48,runs:int=5000,seed:int=20261018):
    if not 0<window_hours<=168:raise ValueError('Window must be 0-168 hours')
    if not 1000<=runs<=100000:raise ValueError('Runs out of bounds')
    snapshot,extra,out=Path(snapshot),Path(extra),Path(out)
    if verify(snapshot)['status']!='VERIFIED':raise InputError('Base snapshot manifest failed validation')
    cutoff=available_time(snapshot,extra,injuries)
    m=json.loads((snapshot/'manifest.json').read_text())
    pbp=load_table(snapshot/m['datasets']['pbp']['local_file'])
    schedule=load_table(snapshot/m['datasets']['schedule']['local_file'])
    roster=load_table(snapshot/m['datasets']['roster']['local_file'])
    table=normalize_schedule(schedule)
    # Normalize schedule returns game_id, home_team, away_team & kickoff_utc;
    # join verified source location, odds and week strictly by unique game_id.
    if schedule.duplicated('game_id').any():raise InputError('Duplicate schedule game_id')
    meta=schedule.set_index('game_id')
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for _,r in table.iterrows():
        kickoff=r['kickoff_utc'].to_pydatetime()
        if not cutoff<kickoff<=cutoff+timedelta(hours=window_hours):continue
        raw=meta.loc[r['game_id']]
        joined=r.copy()
        for key in ('week','location','spread_line','home_spread_odds','away_spread_odds','total_line'):
            joined[key]=raw.get(key)
        game_id=str(r['game_id'])
        try:
            obj=forecast_one(snapshot,extra,joined,cutoff,injuries=injuries,runs=runs,seed=seed,pbp=pbp,schedule=schedule,roster=roster)
            file=out/(game_id+'.json');file.write_text(json.dumps(obj,indent=2)+'\n')
            records.append({'game_id':game_id,'status':'PROJECTED_UNCALIBRATED','file':file.name,
                'mean_margin_home':obj['projected_home_margin'],'fair_spread_home':obj['projected_fair_spread_home'],
                'conditional_qb_scenarios':len(obj['conditional_qb_scenarios']),
                'challenger_version':obj['challenger']['version'],
                'challenger_margin_home':obj['challenger']['candidate']['mean_margin_home'] if obj['challenger']['candidate'] is not None else None,
                'reported_out_players':sum(len(t['reported_out']) for t in obj['player_availability_evidence']['teams'].values())})
        except (InputError,ValueError,KeyError,TypeError) as exc:
            records.append({'game_id':game_id,'status':'BLOCKED','reason':str(exc)})
    index={'schema':'future_core_index_v0.11','capture_cutoff_utc':cutoff.isoformat(),
      'window_hours':window_hours,'pre_inactives_lock_minutes':PREGAME_LOCK_MINUTES,
      'games':records,'n_projected':sum(x['status']=='PROJECTED_UNCALIBRATED' for x in records),
      'n_blocked':sum(x['status']=='BLOCKED' for x in records),
      'challenger_status':f'ONE_ACTIVE_CHALLENGER_V{ACTIVE_CHALLENGER_VERSION}_RESEARCH_ONLY_NOT_PROMOTED',
      'not_a_bet':True,'notes':'Predictions locked at least 120 minutes before kickoff; no official inactives incorporated. No artificial odds or personnel confirmation. Future games only; separately listed measured QB scenarios are conditional.'}
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    return index

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--metrics-dir',type=Path,required=True);p.add_argument('--injuries-dir',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--window-hours',type=float,default=48)
    p.add_argument('--runs',type=int,default=5000);p.add_argument('--seed',type=int,default=20261018)
    a=p.parse_args();print(json.dumps(generate(a.snapshot,a.metrics_dir,a.out,a.injuries_dir,a.window_hours,a.runs,a.seed),indent=2))
