"""Two-team CFB research interface: archived pregame -> evidence-based postgame.

This is not a web chatbot, nor an auto-learning trained model. The same verified
frozen margin remains the baseline for the subsequent forensic analysis.
Only observed, source-attributed market prices support a provisional ATS lean.
No NFL imports. No FCS-v-FCS research.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from cfb.archive import load_all_archives
from cfb.free_odds import load_quotes
from cfb.market import _date
from cfb.pipeline import verify, normalize_pbp_teams
from cfb.research_backtest import observed_game


def normalized(name):
    return re.sub(r'[^a-z0-9]+', '', str(name).casefold())


def resolve(forecasts, team_a, team_b, game_id=None):
    """Require EXACT normalized school name pair; do not silently guess a game."""
    if not team_a or not team_b or normalized(team_a)==normalized(team_b):
        raise ValueError('Supply two distinct team names')
    names=(normalized(team_a),normalized(team_b))
    if not all(names):raise ValueError('Missing team names')
    matches=[]
    for gid,p in forecasts.items():
        if game_id and str(gid)!=str(game_id):continue
        if {normalized(p['home']),normalized(p['away'])}==set(names):
            matches.append(p)
    if not matches:
        examples=sorted({x['home']+' vs '+x['away'] for x in forecasts.values()})
        raise ValueError('No exact archived fixture for '+team_a+' / '+team_b+'. Available fixtures: '+
                         '; '.join(examples[:12]))
    if len(matches)!=1:
        raise ValueError('Multiple matches: provide numeric --game-id; game IDs '+
                         ', '.join(sorted(x['game_id'] for x in matches)))
    return matches[0]


def _valid_quote(p, quotes):
    q=quotes.get(p['game_id'])
    if not q:return None
    if (q['home']!=p['home'] or q['away']!=p['away'] or
        _date(q['kickoff_utc'],'quote kickoff')!=_date(p['kickoff_utc'],'kickoff')):
        raise ValueError('Archive mismatch: market quote belongs to different fixture')
    if max(_date(q['observed_at_utc'],'quote observed'),
           _date(p['prediction_generated_at_utc'],'forecast observed'))>=_date(p['kickoff_utc'],'kickoff'):
        return None
    return q


def _market(p, quotes):
    q=_valid_quote(p,quotes)
    if q is None:return {'status':'NO_VALID_FROZEN_PREKICKOFF_MARKET_OBSERVATION',
                          'ats_research_side':None,'confidence':'NOT_CALIBRATED'}
    sp=q['quote'];edge=float(p['simulated_margin_home'])+float(sp['spread_home'])
    return {'status':'PREKICKOFF_SOURCE_OBSERVED_RESEARCH_ONLY',
            'ats_research_side':'HOME' if edge>0 else 'AWAY' if edge<0 else 'NO_EDGE',
            'home_spread':sp['spread_home'],'provider':sp['book'],
            'home_odds':sp['home_odds_american'],'away_odds':sp['away_odds_american'],
            'predicted_margin_minus_market_margin_home':round(edge,4),
            'market_observed_at_utc':q['observed_at_utc'],
            'market_archive_run_id':q['market_archive_run_id'],
            'market_archive_sha256':q['market_archive_sha256'],
            'book_quote_original_publication_time_verified':False,
            'confidence':'NOT_CALIBRATED',
            'explanation':'Observed ESPN-listed odds; book timestamp unverified; no empirical cover probability'}


def pregame(p, quotes):
    if p['status']!='FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE':
        return {'schema':'cfb_matchup_request_v1','mode':'pregame',
                'status':'INSUFFICIENT_PRIOR_DATA_PASS', 'game_id':p['game_id'],
                'home':p['home'],'away':p['away'],
                'forecast_generated_at_utc':p['prediction_generated_at_utc'],
                'note':'Do not fabricate a margin or ATS side.'}
    if _date(p['prediction_generated_at_utc'],'forecast time')>=_date(p['kickoff_utc'],'kickoff'):
        raise ValueError('Archived forecast not pregame')
    m=_market(p,quotes)
    return {'schema':'cfb_matchup_request_v1', 'mode':'pregame',
            'status':'FROZEN_PREGAME_MODEL_RESEARCH',
            'game_id':p['game_id'],'home':p['home'],'away':p['away'],
            'kickoff_utc':p['kickoff_utc'],
            'forecast_generated_at_utc':p['prediction_generated_at_utc'],
            'original_forecast_run_id':p['original_forecast_run_id'],
            'original_forecast_zip_sha256':p['original_forecast_zip_sha256'],
            'model_variant':p['model_variant'],
            'projected_home_points':p['simulated_score_home'],
            'projected_away_points':p['simulated_score_away'],
            'projected_home_margin':p['simulated_margin_home'],
            'model_projected_winner':p['home'] if p['simulated_margin_home']>0 else p['away'] if p['simulated_margin_home']<0 else 'TIE',
            'prior_completed_games_home':p.get('completed_prior_games_home'),
            'prior_completed_games_away':p.get('completed_prior_games_away'),
            'home_projection':p.get('home_projection'),
            'away_projection':p.get('away_projection'),
            'opponent_adjusted_matchup_signals':p.get('matchup_evidence'),
            'market':m, 'injuries_verified':False,
            'grade':'NOT_CALIBRATED',
            'limitations':['Historical source publication times are not verified row by row',
                           'Player-specific availability and true injury reports are not integrated',
                           'Margin simulation is uncalibrated; market difference is not a validated betting edge',
                           'This is the original locked forecast, not a retroactively improved answer']}


def _team_actual(plays, team, opponent, points, conceded, qb_ids):
    x=observed_game(plays, team, opponent, points, conceded, qb_ids)
    if x is None:return {'status':'INSUFFICIENT_POSTGAME_PBP'}
    def rate(a,b):
        if x.get(a) is None or not x.get(b):return None
        return round(float(x[a])/float(x[b]),4)
    return {'status':'OBSERVED_PBP','points':x['points'],'drives':x['drives'],
            'points_per_recorded_drive':round(x['points']/x['drives'],4),
            'pass_epa_per_play':rate('off_pass_sum','off_pass_n'),
            'rush_epa_per_play':rate('off_rush_sum','off_rush_n'),
            'pass_explosives':x['pass_explosive_sum'],
            'rush_explosives':x['rush_explosive_sum'],
            'pass_plays_with_epa':x['off_pass_n'],'rush_plays_with_epa':x['off_rush_n'],
            'interceptions':x['interceptions'],'fumbles_lost':x['fumbles_lost'],
            'qb_rushes_identified':x['qb_rushes'],
            'qb_hurries':x['qb_hurries_sum'],
            'third_down_converted':x['third_conversions_sum'],
            'third_down_attempts':x['third_conversions_n'],
            'redzone_td_drives':x['redzone_tds'],
            'redzone_trips':x['redzone_trips'],
            'avg_start_field_position':x['start_field_position']}


def _hypotheses(p,home,away):
    """Predeclared diagnostic flags are proposed tests, not causal explanations."""
    candidates=[]
    for side,rec,key in [('home',home,'home_projection'),('away',away,'away_projection')]:
        proj=p.get(key) or {}
        if rec.get('status')!='OBSERVED_PBP':continue
        if proj.get('expected_drives') is not None and abs(rec['drives']-proj['expected_drives'])>=2.5:
            candidates.append({'driver':'POSSESSION_VOLUME','side':side,
               'observed':rec['drives'],'forecast_drives':proj['expected_drives'],
               'possible_model_test':'Test opponent-adjusted drive-count model; measure out-of-sample margin MAE impact'})
        if proj.get('_ppd') is not None and abs(rec['points_per_recorded_drive']-proj['_ppd'])>=1:
            candidates.append({'driver':'SCORING_PER_DRIVE','side':side,
               'observed':rec['points_per_recorded_drive'],'forecast_ppd':proj['_ppd'],
               'possible_model_test':'Test EPA/finishing-drives calibration and interaction terms on future holdouts'})
        if rec.get('interceptions') is not None and rec.get('fumbles_lost') is not None:
            turnovers=rec['interceptions']+rec['fumbles_lost']
            if turnovers>=3:
                candidates.append({'driver':'TURNOVER_UPPER_TAIL','side':side,'observed':turnovers,
                   'possible_model_test':'Test turnover tail distribution vs prior actual turnovers and market margins'})
        if rec.get('redzone_trips',0)>=2 and rec.get('redzone_td_drives') is not None:
            ratio=rec['redzone_td_drives']/rec['redzone_trips']
            if ratio<=.30 or ratio>=.90:
                candidates.append({'driver':'RED_ZONE_FINISHING','side':side,'observed_rate':round(ratio,3),
                   'possible_model_test':'Test shrinkage of red-zone TD conversion vs true points per trip'})
    # Paired game-level EPA gaps are descriptive; *neither* constitutes a
    # contemporaneously validated prediction feature until separately tested.
    if home.get('status')=='OBSERVED_PBP' and away.get('status')=='OBSERVED_PBP':
        for kind in ('pass','rush'):
            h=home.get(kind+'_epa_per_play');a=away.get(kind+'_epa_per_play')
            if h is not None and a is not None and abs(h-a)>=0.35:
                candidates.append({'driver':kind.upper()+'_EPA_DIFFERENTIAL',
                   'observed_home_minus_away_epa_per_play':round(h-a,4),
                   'possible_model_test':'Quantify pregame offense-vs-defense '+kind+' EPA interactions and shrinkage'})
        h=home.get('pass_explosives');a=away.get('pass_explosives')
        if h is not None and a is not None and abs(h-a)>=3:
            candidates.append({'driver':'EXPLOSIVE_PASS_DIFFERENTIAL',
                'observed_home_minus_away_explosive_pass_plays':h-a,
                'possible_model_test':'Evaluate explosive pass frequency distributions vs opponent coverage strength'})
        h=home.get('third_down_converted');a=away.get('third_down_converted')
        hn=home.get('third_down_attempts');an=away.get('third_down_attempts')
        if all(v is not None for v in (h,a,hn,an)) and hn>=4 and an>=4 and abs(h/hn-a/an)>=.25:
            candidates.append({'driver':'THIRD_DOWN_FINISHING_DIFFERENTIAL',
                'home_rate':round(h/hn,3),'away_rate':round(a/an,3),
                'possible_model_test':'Validate third-down success adjustments after regressing small samples'})
    return candidates


def postgame(p, quotes, schedule, pbp=None, roster=None, captured_at=None, source_sha256=None):
    """Original pregame result plus later source-observed final/PBP. Never rerun original forecast."""
    if p['status']!='FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE':
        raise ValueError('No original numeric forecast; cannot score hindsight')
    if captured_at is None:raise ValueError('Must provide real final snapshot capture time')
    when=_date(captured_at,'final snapshot time');kickoff=_date(p['kickoff_utc'],'kickoff')
    if when<kickoff+timedelta(hours=3):
        return {'schema':'cfb_matchup_request_v1','mode':'postgame','status':'FINAL_NOT_YET_ELIGIBLE',
                'game_id':p['game_id'],'reason':'Final source observed less than three hours after kickoff'}
    target=schedule.loc[schedule.game_id.astype(str).eq(p['game_id'])]
    if len(target)!=1:raise ValueError('Game ID not uniquely present in final schedule')
    g=target.iloc[0]
    if str(g.home_id)!=p['home_id'] or str(g.away_id)!=p['away_id']:
        raise ValueError('Numeric matchup team IDs changed')
    if str(g.home_division).casefold()=='fcs' and str(g.away_division).casefold()=='fcs':
        raise ValueError('FCS-v-FCS excluded')
    if not bool(g.completed) or pd.isna(g.home_points) or pd.isna(g.away_points):
        return {'schema':'cfb_matchup_request_v1','mode':'postgame','status':'FINAL_NOT_AVAILABLE',
                'game_id':p['game_id']}
    points=[float(g.home_points),float(g.away_points)]
    if not all(math.isfinite(z) and z>=0 and z<=150 and z.is_integer() for z in points):
        raise ValueError('Invalid observed scores')
    hh,aa=map(int,points)
    actual_margin=hh-aa;pred_margin=float(p['simulated_margin_home']);margin_error=pred_margin-actual_margin
    original=pregame(p,quotes)
    market=original['market'];ats={'status':'NO_SOURCE_OBSERVED_SPREAD'}
    if market['status']=='PREKICKOFF_SOURCE_OBSERVED_RESEARCH_ONLY':
        actual_cover=actual_margin+float(market['home_spread'])
        actual_side='HOME' if actual_cover>0 else 'AWAY' if actual_cover<0 else 'PUSH'
        side=market['ats_research_side']
        ats={'status':'PROVISIONAL_PREKICKOFF_ESPNSOURCED_ATS',
             'research_side':side,'actual_cover_side':actual_side,
             'covered':side==actual_side if side!='NO_EDGE' and actual_side!='PUSH' else None,
             'actual_ats_margin_home':round(actual_cover,4),
             'predicted_ats_margin_home':market['predicted_margin_minus_market_margin_home'],
             'original_book_timestamp_verified':False}
    actuals={'home':{'status':'NO_PBP_AVAILABLE'},'away':{'status':'NO_PBP_AVAILABLE'}}
    if pbp is not None and roster is not None:
        if 'game_id' not in pbp.columns or 'position_abbreviation' not in roster.columns:
            raise ValueError('PBP/roster columns absent')
        part=pbp.loc[pbp.game_id.astype(str).eq(p['game_id'])]
        if not part.empty:
            mapped=normalize_pbp_teams(part,target)
            qb={int(t):set(z.athlete_id.dropna()) for t,z in
                roster.loc[roster.position_abbreviation.eq('QB')].groupby('team_id')}
            hid,aid=int(p['home_id']),int(p['away_id'])
            actuals={'home':_team_actual(mapped,hid,aid,hh,aa,qb.get(hid,set())),
                     'away':_team_actual(mapped,aid,hid,aa,hh,qb.get(aid,set()))}
    return {'schema':'cfb_matchup_request_v1','mode':'postgame','status':'FINAL_REVIEW_RECORDED',
            'game_id':p['game_id'],'home':p['home'],'away':p['away'],
            'kickoff_utc':p['kickoff_utc'],
            'locked_original_forecast':{'run_id':p['original_forecast_run_id'],
               'forecast_zip_sha256':p['original_forecast_zip_sha256'],
               'generated_at_utc':p['prediction_generated_at_utc'],
               'projected_home':p['simulated_score_home'],'projected_away':p['simulated_score_away'],
               'projected_home_margin':pred_margin},
            'observed_final':{'home':hh,'away':aa,'home_margin':actual_margin},
            'margin_error_prediction_minus_actual':round(margin_error,4),
            'absolute_margin_error':round(abs(margin_error),4),
            'model_winner_direction_correct':None if pred_margin==0 or actual_margin==0 else ((pred_margin>0)==(actual_margin>0)),
            'ATS':ats,'observed_game_statistics':actuals,
            'candidate_mechanisms_to_test':_hypotheses(p,actuals['home'],actuals['away']),
            'future_model_update':{'status':'REQUIRES_INDEPENDENT_FORWARD_VALIDATION',
               'automatic_weight_change':False,
               'criterion':'Retain a modification only if it improves out-of-sample forward error, with sufficient sample size and no test leakage'},
            'final_schedule_snapshot_captured_at_utc':when.isoformat(),
            'source_manifest_sha256':source_sha256,
            'limitations':['Game-level deviations are observations, not proof of root cause',
               'Some turnovers/pressure and availability fields may be missing or approximated',
               'The original bookmaker timestamp remains unverified even if ESPN quote was observed before kickoff',
               'One game cannot validate or automatically promote a model change']}


def markdown(report):
    if report.get('mode')=='pregame' and report.get('status')=='FROZEN_PREGAME_MODEL_RESEARCH':
        m=report['market'];market=('No valid archived spread' if m.get('home_spread') is None else
                                 f"{report['home']} {m['home_spread']:+g} ({m['provider']}); lean {m['ats_research_side']}")
        return (f"# CFB pregame: {report['away']} at {report['home']}\n\n"
            f"Frozen forecast: {report['projected_home_points']:.1f}–{report['projected_away_points']:.1f}; "
            f"home margin {report['projected_home_margin']:+.2f}\n\n"
            f"Market: {market}. Research-only; confidence not calibrated.\n\n"
            f"Original archive: {report['original_forecast_run_id']} / {report['original_forecast_zip_sha256']}\n\n"
            f"Available matchup statistics and source limitations are in the accompanying JSON.\n")
    if report.get('mode')=='postgame' and report.get('status')=='FINAL_REVIEW_RECORDED':
        s=report['observed_final']; f=report['locked_original_forecast']
        return (f"# CFB postgame: {report['away']} at {report['home']}\n\n"
           f"Original frozen home margin {f['projected_home_margin']:+.2f}; actual home margin {s['home_margin']:+d}. "
           f"Absolute miss {report['absolute_margin_error']:.2f} points.\n\n"
           f"Final {s['home']}–{s['away']}; ATS {report['ATS']['status']}.\n\n"
           f"Candidate mechanisms proposed for future tests: {len(report['candidate_mechanisms_to_test'])}. "
           f"No model adjustment promoted from this game alone.\n")
    return '# CFB request\n\n'+report.get('status','UNKNOWN')+'\n'


def run(args):
    forecasts=load_all_archives(args.forecasts)
    p=resolve(forecasts,args.team_a,args.team_b,args.game_id)
    quotes=load_quotes(args.odds)
    if args.mode=='pregame':
        if datetime.now(timezone.utc)>=_date(p['kickoff_utc'],'kickoff'):
            raise ValueError('Kickoff has passed; request postgame instead of labeling an after-kickoff report pregame')
        report=pregame(p,quotes)
    else:
        manifest=verify(args.snapshot)
        folder=Path(args.snapshot)
        sc=pd.read_parquet(folder/'schedule.parquet')
        pbp=pd.read_parquet(folder/'pbp.parquet')
        roster=pd.read_parquet(folder/'roster.parquet')
        report=postgame(p,quotes,sc,pbp,roster,manifest['captured_at_utc'],
                        hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest())
    report['requested_team_a']=args.team_a; report['requested_team_b']=args.team_b
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    md=out.with_suffix('.md')
    if out.exists() or md.exists():raise ValueError('Refusing to overwrite an existing matchup review')
    out.write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+'\n')
    md.write_text(markdown(report))
    print(json.dumps({k:v for k,v in report.items() if k not in
                      ('home_projection','away_projection','opponent_adjusted_matchup_signals','observed_game_statistics')},indent=2,default=str))
    return report


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--mode',required=True,choices=('pregame','postgame'))
    pa.add_argument('--team-a',required=True);pa.add_argument('--team-b',required=True)
    pa.add_argument('--game-id',default=None)
    pa.add_argument('--forecasts',required=True);pa.add_argument('--odds',required=True)
    pa.add_argument('--snapshot',required=True);pa.add_argument('--out',required=True)
    args=pa.parse_args();run(args)
if __name__=='__main__':main()
