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
from cfb.market import _date
from cfb.matchups import matchup_signals
from cfb.matchup_cycle import _market, postgame as compare_postgame, markdown as legacy_markdown, normalized
from cfb.pipeline import (verify, split_games, normalize_pbp_teams, team_features,
                          missing_features, iso_utc)
from cfb.research_backtest import observed_game, project
from cfb.free_odds import verify_odds_archive
from cfb.learn_forward import investigate

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
            min_games=3, now=None, github_run_id=None, git_sha=None):
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
        'market':{'status':'NO_SOURCE_OBSERVED_PREKICKOFF_SPREAD','ats_research_side':None},
        'locked_market_evidence':None,
        'limitations':['Unverified current injuries and projected lineups are NOT incorporated',
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
        # _market validates fixture, source observation and model creation time.
        proxy={**result,'simulated_margin_home':sim['mean_margin_home']}
        market=_market(proxy,qmap)
        result.update({'status':'FRESH_PREGAME_PROJECTION_LOCKED',
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
        return legacy_markdown(report)+('\nOriginal user-requested file: '+
                   report['original_user_requested_pregame_file']+'\n')
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
    lines += ['', '## Opponent-adjusted signals', '',
              'Detailed matchup interactions and source-derived schedules are in the JSON.',
              '', '## Stress checks (hypothetical, not likely outcomes)', '']
    for t in report['stress_tests']['scenarios']:
        lines.append(f"- {t['scenario']}: projected home margin {t['mean_home_margin']:+.1f}")
    lines += ['', '**Known gaps:** Verified current injuries and lineups, full pressure, '
              'special teams, penalties, and calibrated ATS probability are not integrated.',
              '', 'The model has calculated a NEW score projection; it did not look up an earlier prediction.']
    return '\n'.join(lines)+'\n'


def run(args):
    if args.mode=='pregame':
        data=analyze(args.snapshot,args.team_a,args.team_b,args.game_id,
                     odds_root=args.odds,runs=args.runs,min_games=args.min_games,
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
    ap.add_argument('--out',required=True)
    ap.add_argument('--runs',type=int,default=3000)
    ap.add_argument('--min-games',type=int,default=3)
    ap.add_argument('--github-run-id');ap.add_argument('--git-sha')
    args=ap.parse_args();run(args)

if __name__=='__main__':main()
