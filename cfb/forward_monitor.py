"""Auto-grade independently archived CFB forecasts using *later-observed* finals.

Provenance: forecast ZIP frozen before kickoff; market ZIP fetched before kickoff;
final schedule bytes fetched after completion. SportsDataverse schedule is a free,
mutable public release and is NOT an authenticated result-at-final timestamp.
Records remain CFB research (not a calibrated profitable ATS product).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import zipfile
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from cfb.archive import load_all_archives, digest, _zipwrite, ID_RE
from cfb.free_odds import load_quotes
from cfb.pipeline import source_urls
from cfb.market import _date


def fetch_schedule(season):
    url=source_urls(season)['schedule']
    with urllib.request.urlopen(urllib.request.Request(url,headers={
                'User-Agent':'cfb-research-free-snapshot/1.0'}),timeout=90) as r:
        raw=r.read(15_000_000)
    if len(raw)<100 or raw[-4:]!=b'PAR1':
        raise ValueError('Invalid free schedule release')
    return raw,url


def grade(forecast_root, market_root, schedule, captured_at_utc, source_url,
          source_sha256, issued_at=None):
    cutoff=_date(captured_at_utc,'final schedule capture')
    if issued_at is not None and _date(issued_at,'audit issue')<cutoff:
        raise ValueError('Cannot issue a report predating its final-score source')
    required={'game_id','completed','start_date','home_id','away_id',
              'home_team','away_team','home_points','away_points'}
    if not required.issubset(schedule.columns):
        raise ValueError('Missing verified final score columns')
    if schedule.game_id.duplicated().any():
        raise ValueError('Duplicate schedule game ID')
    games={str(r.game_id):r for r in schedule.itertuples(index=False)}
    forecasts=load_all_archives(forecast_root)
    quotes=load_quotes(market_root)
    scored=[];pending=[]
    for gid,p in sorted(forecasts.items(),key=lambda x:(x[1]['kickoff_utc'],x[0])):
        game=games.get(gid)
        if game is None:
            pending.append({'game_id':gid,'reason':'GAME_NOT_IN_LATEST_FREE_SCHEDULE'});continue
        kickoff=_date(p['kickoff_utc'],'frozen kickoff')
        try:
            raw_date=pd.to_datetime(game.start_date,utc=True,errors='coerce')
            if pd.isna(raw_date):raise ValueError('Unparseable scheduled date')
            sched_kickoff=_date(raw_date.isoformat(),'schedule kickoff')
        except ValueError:
            raise ValueError('Unparseable schedule kickoff '+gid)
        if abs((kickoff-sched_kickoff).total_seconds())>3600:
            pending.append({'game_id':gid,'reason':'SCHEDULE_KICKOFF_CHANGED_REQUIRES_REVIEW'});continue
        # The original frozen matchup must match numeric IDs; never fuzzy-match.
        if str(game.home_id)!=p['home_id'] or str(game.away_id)!=p['away_id']:
            raise ValueError('Frozen schedule team IDs disagree '+gid)
        if kickoff>=cutoff or kickoff+timedelta(hours=3)>cutoff:
            pending.append({'game_id':gid,'reason':'INSUFFICIENT_ELAPSED_TIME'});continue
        if not (pd.notna(game.completed) and bool(game.completed)):
            pending.append({'game_id':gid,'reason':'FINAL_NOT_IN_FREE_RELEASE'});continue
        if pd.isna(game.home_points) or pd.isna(game.away_points):
            pending.append({'game_id':gid,'reason':'MISSING_FINAL_SCORES'});continue
        hs=float(game.home_points);ass=float(game.away_points)
        if not all(math.isfinite(x) and x>=0 and x<=150 and x.is_integer() for x in (hs,ass)):
            raise ValueError('Invalid recorded final scores '+gid)
        actual=hs-ass
        pred=float(p['simulated_margin_home'])
        if not math.isfinite(pred):raise ValueError('Invalid frozen projected margin '+gid)
        row={'game_id':gid,'home':p['home'],'away':p['away'],
             'kickoff_utc':p['kickoff_utc'],'forecast_generated_at_utc':p['prediction_generated_at_utc'],
             'forecast_run_id':p['original_forecast_run_id'],
             'forecast_zip_sha256':p['original_forecast_zip_sha256'],
             'predicted_home_margin':pred,'actual_home_margin':actual,
             'margin_error':round(pred-actual,5),'winner_direction_correct':((pred>0)==(actual>0) if actual!=0 and pred!=0 else None),
             'source_final_scores':{'home':int(hs),'away':int(ass)},
             'ATS':{'status':'NO_PREKICKOFF_SIGNED_SPREAD_OBSERVED'}}
        # Evaluate the PRE-RECORDED challenger; never reconstruct after a final.
        advanced=p.get('advanced_research_challenger')
        if isinstance(advanced,dict) and advanced.get('status')=='AVAILABLE_UNCALIBRATED_RESEARCH_ONLY':
            candidate=advanced.get('simulated_margin_home')
            if isinstance(candidate,(int,float)) and math.isfinite(candidate):
                row['advanced_challenger']={
                    'status':'ORIGINAL_FROZEN_RESEARCH_CANDIDATE',
                    'version':advanced.get('version'),
                    'projected_home_margin':float(candidate),
                    'margin_error':round(float(candidate)-actual,5),
                    'candidate_absolute_error':round(abs(float(candidate)-actual),5),
                    'baseline_absolute_error':round(abs(pred-actual),5),
                    'better_than_baseline_this_game':abs(float(candidate)-actual)<abs(pred-actual),
                    'not_deployed':True}
        script=p.get('game_script_research_challenger')
        if isinstance(script,dict) and script.get('status')=='UNFITTED_GAME_SCRIPT_CHALLENGER_NOT_PRIMARY':
            candidate=script.get('mean_margin_home')
            if isinstance(candidate,(int,float)) and math.isfinite(candidate):
                interval=script.get('margin_p10_p50_p90') or []
                row['game_script_challenger']={
                    'status':'ORIGINAL_FROZEN_GAME_SCRIPT_CANDIDATE',
                    'version':script.get('version'),
                    'projected_home_margin':float(candidate),
                    'candidate_absolute_error':round(abs(float(candidate)-actual),5),
                    'baseline_absolute_error':round(abs(pred-actual),5),
                    'margin_p10_p90_covered':(float(interval[0])<=actual<=float(interval[2]) if len(interval)==3 else None),
                    'not_deployed':True,'resimulated_after_kickoff':False}
        # Only evaluate challengers stored in the earliest original frozen record.
        for key,field in (('integrated_scoring_research_challenger','integrated_scoring_challenger'),
                          ('integrated_scoring_plus_script_research','integrated_script_challenger')):
            z=p.get(key)
            if isinstance(z,dict) and z.get('status')=='AVAILABLE_UNFITTED_INTEGRATED_RESEARCH_ONLY':
                cand=z.get('mean_margin_home')
                if isinstance(cand,(int,float)) and math.isfinite(float(cand)):
                    row[field]={
                        'status':'ORIGINAL_FROZEN_INTEGRATED_RESEARCH_CANDIDATE',
                        'version':z.get('version'),'projected_home_margin':float(cand),
                        'baseline_absolute_error':round(abs(pred-actual),5),
                        'candidate_absolute_error':round(abs(float(cand)-actual),5),
                        'not_primary':True,'recomputed_after_kickoff':False}
        challenger=p.get('build16_challenger')
        if isinstance(challenger,dict) and challenger.get('status')=='AVAILABLE_UNFITTED_BUILD16_RESEARCH_ONLY':
            value=challenger.get('mean_margin_home')
            if isinstance(value,(int,float)) and math.isfinite(float(value)):
                interval=challenger.get('margin_p10_p50_p90') or []
                row['build16_challenger']={
                    'status':'ORIGINAL_FROZEN_BUILD16_CHALLENGER',
                    'version':challenger.get('version'),
                    'projected_home_margin':float(value),
                    'baseline_absolute_error':round(abs(pred-actual),5),
                    'candidate_absolute_error':round(abs(float(value)-actual),5),
                    'margin_p10_p90_covered':(float(interval[0])<=actual<=float(interval[2])
                        if len(interval)==3 else None),
                    'recomputed_after_kickoff':False,'not_primary':True}
        # Personnel cases were frozen as counterfactual sensitivities. Without
        # independently confirmed pregame starter assignments they cannot be
        # selected with hindsight and scored as an honest forward prediction.
        pe=p.get('conditional_personnel_scenarios')
        if isinstance(pe,dict):
            row['conditional_personnel_research']={
                'status':'FROZEN_COUNTERFACTUAL_NOT_A_PREDICTION',
                'scenario_count':len(pe.get('scenarios',[])),
                'main_prediction_unchanged':True,
                'not_included_in_forward_mae':True}
        m=quotes.get(gid)
        if m is not None:
            if (m['home']!=p['home'] or m['away']!=p['away'] or
                    _date(m['kickoff_utc'],'market kickoff')!=kickoff):
                raise ValueError('Market and original frozen matchup disagree '+gid)
            q=m['quote']
            # Quotes may be captured after the frozen margin forecast but must
            # be BEFORE KICKOFF. Determining ATS side at capture is distinct
            # from claiming an ATS bet was decided when original forecast ran.
            recorded_at=_date(m['observed_at_utc'],'observed spread')
            projected_at=_date(p['prediction_generated_at_utc'],'original forecast')
            decision_time=max(projected_at,recorded_at)
            if decision_time>=kickoff:
                row['ATS']={'status':'MARKET_NOT_JOINTLY_AVAILABLE_BEFORE_KICKOFF'}
            else:
                spread=float(q['spread_home'])
                edge=pred+spread
                realized=actual+spread
                decision='HOME' if edge>1e-8 else 'AWAY' if edge< -1e-8 else 'NO_EDGE'
                outcome='HOME' if realized>1e-8 else 'AWAY' if realized< -1e-8 else 'PUSH'
                success=(decision==outcome) if decision!='NO_EDGE' and outcome!='PUSH' else None
                odds=q['home_odds_american'] if decision=='HOME' else q['away_odds_american']
                pnl=(100/abs(odds) if odds<0 else odds/100) if success else (-1 if success is False else 0)
                row['ATS']={'status':'OBSERVED_PREKICKOFF_ESPN_PROVISIONAL',
                       'book':q['book'],'spread_home':spread,
                       'odds_home':q['home_odds_american'],'odds_away':q['away_odds_american'],
                       'market_observed_at_utc':m['observed_at_utc'],
                       'ats_comparison_first_possible_at_utc':decision_time.isoformat(),
                       'market_observed_after_initial_margin_forecast':recorded_at>projected_at,
                       'market_archive_run_id':m['market_archive_run_id'],
                       'market_archive_sha256':m['market_archive_sha256'],
                       'raw_odds_source_sha256':m['raw_source_sha256'],
                       'original_book_quote_time_verified':False,
                       'predicted_edge_at_observed_line':round(edge,4),
                       'settled_margin_at_observed_line':round(realized,4),
                       'research_side':decision,'result':outcome,
                       'correct_excluding_push_or_no_edge':success,
                       'flat_stake_unit_profit_research_only':round(pnl,5)}
                # Same verified, jointly pre-kickoff quote: compare the ONE
                # frozen Build16 alternative independently of baseline.
                build16=row.get('build16_challenger')
                if isinstance(build16,dict):
                    bpred=build16['projected_home_margin'];bedge=bpred+spread
                    bside=('HOME' if bedge>1e-8 else 'AWAY'
                           if bedge< -1e-8 else 'NO_EDGE')
                    bcorrect=(bside==outcome if bside!='NO_EDGE' and outcome!='PUSH' else None)
                    row['ATS']['build16_research_comparison']={
                        'candidate_frozen_pre_kickoff':True,
                        'signed_home_spread':spread,
                        'predicted_edge_at_observed_line':round(bedge,4),
                        'research_side':bside,
                        'correct_excluding_push_or_no_edge':bcorrect,
                        'ats_calibrated':False,
                        'original_book_quote_time_verified':False}
        scored.append(row)
    if scored:
        errors=[r['margin_error'] for r in scored]
        margin={'games':len(scored),'mae':round(sum(abs(x) for x in errors)/len(errors),4),
                'rmse':round((sum(x*x for x in errors)/len(errors))**.5,4),
                'bias':round(sum(errors)/len(errors),4)}
        eligible_dir=[r['winner_direction_correct'] for r in scored if r['winner_direction_correct'] is not None]
        margin['winner_direction_accuracy_not_ats']=(round(sum(eligible_dir)/len(eligible_dir),4) if eligible_dir else None)
    else:margin={'games':0,'mae':None,'rmse':None,'bias':None,'winner_direction_accuracy_not_ats':None}
    paired=[r['advanced_challenger'] for r in scored if 'advanced_challenger' in r]
    advanced_forward={'status':'RESEARCH_ONLY_NEVER_AUTO_DEPLOY',
                      'games_with_frozen_challenger':len(paired),
                      'on_same_games_baseline_mae':(round(sum(p['baseline_absolute_error'] for p in paired)/len(paired),4) if paired else None),
                      'on_same_games_advanced_mae':(round(sum(p['candidate_absolute_error'] for p in paired)/len(paired),4) if paired else None),
                      'on_same_games_advanced_improves_mae':(sum(p['candidate_absolute_error'] for p in paired)<sum(p['baseline_absolute_error'] for p in paired) if paired else None)}
    script_paired=[r['game_script_challenger'] for r in scored if 'game_script_challenger' in r]
    script_forward={'status':'RESEARCH_ONLY_NEVER_AUTO_DEPLOY',
        'games_with_frozen_challenger':len(script_paired),
        'on_same_games_baseline_mae':(round(sum(p['baseline_absolute_error'] for p in script_paired)/len(script_paired),4) if script_paired else None),
        'on_same_games_script_mae':(round(sum(p['candidate_absolute_error'] for p in script_paired)/len(script_paired),4) if script_paired else None),
        'observed_p10_p90_coverage':(round(sum(p['margin_p10_p90_covered'] for p in script_paired if p['margin_p10_p90_covered'] is not None)/sum(p['margin_p10_p90_covered'] is not None for p in script_paired),4)
            if any(p['margin_p10_p90_covered'] is not None for p in script_paired) else None),
        'candidate_selected_automatically':False}
    personnel_forward={'status':'NOT_GRADED_UNCONFIRMED_PREGAME_STARTERS',
        'games_with_frozen_conditional_evidence':sum('conditional_personnel_research' in r for r in scored),
        'games_with_conditional_scenarios':sum(r.get('conditional_personnel_research',{}).get('scenario_count',0)>0 for r in scored),
        'prospective_personnel_accuracy_claims':False}
    ats=[r['ATS'] for r in scored if r['ATS']['status']=='OBSERVED_PREKICKOFF_ESPN_PROVISIONAL']
    graded=[q for q in ats if q['research_side']!='NO_EDGE']
    w=sum(q['correct_excluding_push_or_no_edge'] is True for q in graded)
    losses=sum(q['correct_excluding_push_or_no_edge'] is False for q in graded)
    pushes=len(graded)-w-losses
    output={'schema':'cfb_forward_research_audit_v1',
          'status':'FORWARD_MARGIN_RESEARCH_NO_VALIDATED_ATS_EDGE',
          'free_schedule_source_url':source_url,'free_schedule_bytes_sha256':source_sha256,
          'free_schedule_observed_at_utc':cutoff.isoformat(),
          'earliest_frozen_prediction_per_game':True,
          'archived_forecast_games':len(forecasts),'graded_final_games':len(scored),
          'pending_games':len(pending),'pending':pending,
          'forward_margin':margin,
          'build16_challenger_forward':{
              'status':'RESEARCH_ONLY_NOT_VALIDATED_OR_DEPLOYED',
              'paired_games':sum('build16_challenger' in r for r in scored),
              'paired_baseline_mae':(round(sum(r['build16_challenger']['baseline_absolute_error']
                  for r in scored if 'build16_challenger' in r)/
                  sum('build16_challenger' in r for r in scored),5)
                  if any('build16_challenger' in r for r in scored) else None),
              'paired_build16_mae':(round(sum(r['build16_challenger']['candidate_absolute_error']
                  for r in scored if 'build16_challenger' in r)/
                  sum('build16_challenger' in r for r in scored),5)
                  if any('build16_challenger' in r for r in scored) else None),
              'not_primary':True},
          'personnel_conditional_forward':personnel_forward,
          'ats_provisional':{'complete_observed_quoted_games':len(ats),
            'research_decisions':len(graded),'wins':w,'losses':losses,'pushes':pushes,
            'cover_accuracy_excluding_pushes':round(w/(w+losses),5) if w+losses else None,
            'flat_stake_roi_per_decision':round(sum(x['flat_stake_unit_profit_research_only'] for x in graded)/len(graded),5) if graded else None,
            'ats_validated':False},
          'build16_ats_provisional':{
              'games_with_joint_pre_kickoff_line':sum(
                 'build16_research_comparison' in x.get('ATS',{}) for x in scored),
              'cover_accuracy_excluding_pushes_and_no_edge':(
                 round(sum(x['ATS']['build16_research_comparison']['correct_excluding_push_or_no_edge'] is True
                     for x in scored if 'build16_research_comparison' in x.get('ATS',{})) /
                 sum(x['ATS']['build16_research_comparison']['correct_excluding_push_or_no_edge'] is not None
                     for x in scored if 'build16_research_comparison' in x.get('ATS',{})),5)
                 if any(x['ATS']['build16_research_comparison']['correct_excluding_push_or_no_edge'] is not None
                     for x in scored if 'build16_research_comparison' in x.get('ATS',{})) else None),
              'ats_validated':False},
          'scored_games':scored,
          'limitations':['No independently attested original sportsbook publication time',
            'GitHub archive commit must be independently checked to establish pregame durability',
            'Free SportsDataverse schedule may be revised after game; hash records observed bytes',
            'No empirically calibrated ATS probability, no wager recommendations',
            'Bookmaker price availability and coverage are uncontrolled',
            'Repeated observations/overlapping teams are not independent samples']}
    return output



def archive_final_source(file, raw, captured_at_utc, source_url):
    """Save the *actual* raw Parquet bytes, not only a claimed source hash."""
    target=Path(file)
    if not ID_RE.fullmatch(target.stem) or target.suffix!='.zip':
        raise ValueError('Final source archive must be numeric run_id.zip')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Refusing to overwrite original final-source evidence')
    if len(raw)<100 or raw[-4:]!=b'PAR1':
        raise ValueError('Cannot archive a non-Parquet final source')
    index={'schema':'cfb_finals_observed_source_v1',
        'observed_at_utc':_date(captured_at_utc,'final source observed').isoformat(),
        'source_url':source_url,'schedule_sha256':digest(raw),'size_bytes':len(raw)}
    with zipfile.ZipFile(target,'x') as z:
        _zipwrite(z,'evidence.json',(json.dumps(index,sort_keys=True)+'\n').encode())
        _zipwrite(z,'schedule.parquet',raw)
    return index


def verify_final_source(file):
    with zipfile.ZipFile(file) as z:
        if sorted(z.namelist())!=['evidence.json','schedule.parquet']:
            raise ValueError('Unexpected final-score archive structure')
        record=json.loads(z.read('evidence.json'))
        raw=z.read('schedule.parquet')
    if (record.get('schema')!='cfb_finals_observed_source_v1' or
        record.get('schedule_sha256')!=digest(raw) or
        record.get('size_bytes')!=len(raw)):
        raise ValueError('Final-score raw evidence digest mismatch')
    _date(record['observed_at_utc'],'observed')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forecasts',required=True);p.add_argument('--odds',required=True)
    p.add_argument('--out',required=True);p.add_argument('--season',type=int,default=2026)
    p.add_argument('--schedule',help='Local actual Parquet schedule from immutable snapshot')
    p.add_argument('--manifest',help='Manifest to verify local snapshot schedule bytes')
    p.add_argument('--source-archive',help='Append-only compressed original final schedule source')
    args=p.parse_args()
    if args.schedule:
        raw=Path(args.schedule).read_bytes()
        if args.manifest:
            manifest=json.loads(Path(args.manifest).read_text())
            evidence=manifest['datasets']['schedule']
            if digest(raw)!=evidence['sha256']:
                raise ValueError('Local schedule bytes mismatch source manifest')
            url=evidence['url']
            captured=evidence['captured_at_utc']
        else:
            url='local://supplied-parquet-no-provenance'
            captured=datetime.now(timezone.utc).isoformat()
    else:
        raw,url=fetch_schedule(args.season)
        captured=datetime.now(timezone.utc).isoformat()
    import io
    schedule=pd.read_parquet(io.BytesIO(raw))
    output=grade(args.forecasts,args.odds,schedule,captured,url,digest(raw))
    if args.source_archive:
        archive_final_source(args.source_archive,raw,captured,url)
        verify_final_source(args.source_archive)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Refuse overwrite of immutable postgame CFB audit')
    path.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('scored_games','pending')},indent=2))

if __name__=='__main__':main()
