"""Independent CFB QB participation evidence and source-backed availability ledger.

No starter is inferred from season rosters, no missing injury report is treated as
healthy, and no injury point-value is invented. Future games/plays excluded by caller.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from cfb.market import _date

OFFICIAL_DOMAINS=(
    'secsports.com', 'bigten.org', 'big12sports.com', 'pac-12.com',
)
STATUSES={'AVAILABLE','PROBABLE','QUESTIONABLE','DOUBTFUL','OUT','OUT_1ST_HALF','GAME_TIME_DECISION'}
SCHEMA='cfb_official_availability_evidence_v1'


def _key(value):
    if pd.isna(value):return None
    try:
        return str(int(value)) if float(value).is_integer() else None
    except (ValueError,TypeError,OverflowError,AttributeError):
        return None


def _finite_mean(values):
    nums=pd.to_numeric(values,errors='coerce').dropna()
    return round(float(nums.mean()),4) if len(nums) else None


def quarterback_usage(mapped, roster, eligible, team_id, recent_games=3):
    """Describe identified passers from completed, eligible PRIOR games only.

    A roster QB is only an identity link, not a current starter or availability
    report. Pass-play volume is not identical to official pass attempts.
    """
    if not isinstance(recent_games,int) or not 1<=recent_games<=8:
        raise ValueError('Invalid recent games window')
    needed={'game_id','pos_team_id','pass','passer_player_id','EPA'}
    if not needed.issubset(mapped.columns):
        return {'status':'MISSING_QB_PASS_PLAY_FIELDS','missing_fields':sorted(needed-set(mapped.columns))}
    required={'team_id','athlete_id','position_abbreviation'}
    if not required.issubset(roster.columns):
        return {'status':'MISSING_QB_ROSTER_FIELDS','missing_fields':sorted(required-set(roster.columns))}
    required_games={'game_id','home_id','away_id','_kickoff'}
    if not required_games.issubset(eligible.columns):
        return {'status':'MISSING_GAME_IDENTITY_FIELDS','missing_fields':sorted(required_games-set(eligible.columns))}
    team=str(team_id)
    games=eligible.loc[(eligible.home_id.map(_key)==team)|(eligible.away_id.map(_key)==team)].sort_values(['_kickoff','game_id'])
    ids=[str(x) for x in games.game_id]
    if not ids:
        return {'status':'NO_PRIOR_ELIGIBLE_GAMES','team_id':team,'quarterbacks':[]}
    last=set(ids[-recent_games:]); full=set(ids)
    r=roster.loc[(roster.team_id.map(_key)==team)&roster.position_abbreviation.astype(str).str.upper().eq('QB')]
    qbids={_key(x) for x in r.athlete_id if _key(x)}
    if not qbids:
        return {'status':'NO_IDENTIFIED_ROSTER_QBS','team_id':team,'prior_games':len(ids),'quarterbacks':[]}
    names={_key(x.athlete_id):str(getattr(x,'display_name','') or getattr(x,'athlete_display_name','') or '') for _,x in r.iterrows()}
    offense=mapped.loc[(mapped.pos_team_id.map(_key)==team)&mapped.game_id.astype(str).isin(full)]
    pass_plays=offense.loc[offense['pass'].eq(True)].copy()
    pass_plays['pid']=pass_plays.passer_player_id.map(_key)
    known=pass_plays.loc[pass_plays.pid.isin(qbids)]
    unidentified=int(pass_plays.pid.isna().sum())
    other=int((pass_plays.pid.notna()&~pass_plays.pid.isin(qbids)).sum())
    recent=known.loc[known.game_id.astype(str).isin(last)]
    recent_all=pass_plays.loc[pass_plays.game_id.astype(str).isin(last)]
    rush_plays=offense.loc[offense.rush.eq(True)].copy() if 'rush' in offense.columns else offense.iloc[:0]
    if 'rusher_player_id' in rush_plays.columns:
        rush_plays['pid']=rush_plays.rusher_player_id.map(_key)
    else:
        rush_plays=rush_plays.iloc[:0]
    records=[]
    for qid in sorted(qbids):
        p=known.loc[known.pid==qid]; rp=recent.loc[recent.pid==qid]
        rush=rush_plays.loc[rush_plays.pid==qid] if 'pid' in rush_plays.columns else rush_plays
        if len(p)==0 and len(rush)==0:continue
        records.append({'athlete_id':qid,'name':names.get(qid) or 'Unnamed roster QB',
            'prior_games_with_recorded_pass_play':int(p.game_id.nunique()),
            'total_recorded_pass_plays':int(len(p)),
            'recent_recorded_pass_plays':int(len(rp)),
            'recent_share_of_all_team_pass_plays':round(len(rp)/len(recent_all),4) if len(recent_all) else None,
            'pass_play_epa_avg':_finite_mean(p.EPA),
            'recent_pass_play_epa_avg':_finite_mean(rp.EPA),
            'recorded_qb_rushes':int(len(rush)),
            'qb_rush_epa_avg':_finite_mean(rush.EPA) if len(rush) else None,
            'last_recorded_pass_game_id':next((gid for gid in reversed(ids) if gid in set(p.game_id.astype(str))),None)})
    records.sort(key=lambda x:(-x['recent_recorded_pass_plays'],-x['total_recorded_pass_plays'],x['athlete_id']))
    head=records[0] if records else None
    coverage=round((len(known)/len(pass_plays)),4) if len(pass_plays) else None
    return {'status':'QB_PARTICIPATION_OBSERVED_NOT_STARTER_CONFIRMED',
        'team_id':team,'prior_games':len(ids),'recent_games_window':min(len(ids),recent_games),
        'recent_prior_game_ids':ids[-recent_games:],
        'total_recorded_team_pass_plays':int(len(pass_plays)),
        'unidentified_pass_play_count':unidentified,
        'passer_id_not_matched_to_roster_count':other,
        'identified_roster_qb_pass_play_coverage':coverage,
        'leader_by_recent_pass_play_volume':head['athlete_id'] if head else None,
        'leader_is_confirmed_starter':False,
        'quarterbacks':records,
        'caution':'Recent passer volume is descriptive; cannot confirm next-game starter or medical availability.'}


def load_official_statuses(root, game_id, kickoff, analysis_time, expected_teams=None):
    """Load an optional RAW-SOURCE-EVIDENCED report, fail closed on unverifiable rows.

    This does not fetch reports; snapshots should be captured externally while
    published before kickoff. Captured text is paired with literal line-level
    evidence. A claimed captured time is not independently attested by this file.
    """
    none={'status':'NO_OFFICIAL_REPORT_CAPTURE','players':[],
          'unconfirmed_players':[], 'source_verified':False,'effects_applied_to_score':False}
    if not root:return none
    file=Path(root)/(str(game_id)+'.json')
    if not file.is_file():return none
    data=json.loads(file.read_text(encoding='utf8'))
    if data.get('schema')!=SCHEMA or str(data.get('game_id'))!=str(game_id):
        raise ValueError('Wrong report schema/game identity')
    url=data.get('source_url')
    p=urlparse(url if isinstance(url,str) else '')
    host=p.hostname or ''
    if p.scheme!='https' or p.username or p.password or p.port or host not in OFFICIAL_DOMAINS:
        raise ValueError('Report must cite allowed official conference HTTPS source')
    capture=_date(data.get('captured_at_utc'),'availability observed')
    k=_date(kickoff,'kickoff')
    at=_date(analysis_time,'analysis')
    if capture>at or capture>=k:
        raise ValueError('Availability report is not proven available by model analysis')
    raw=data.get('source_text')
    if not isinstance(raw,str) or not raw or len(raw)>400000:
        raise ValueError('Require original captured source text')
    actual_hash=hashlib.sha256(raw.encode('utf8')).hexdigest()
    if data.get('source_sha256')!=actual_hash:
        raise ValueError('Captured source checksum mismatch')
    players=[];teams=set(str(x) for x in expected_teams) if expected_teams else None
    for r in data.get('players',[]):
        if not isinstance(r,dict) or r.get('status') not in STATUSES:
            raise ValueError('Invalid player availability status')
        tid=str(r.get('team_id',''))
        if not tid.isdigit() or (teams and tid not in teams):
            raise ValueError('Availability player belongs to wrong team')
        name=r.get('name');excerpt=r.get('source_excerpt');qid=str(r.get('athlete_id',''))
        if not isinstance(name,str) or len(name)<3 or not isinstance(excerpt,str) or not excerpt:
            raise ValueError('Require athlete name and verbatim source excerpt')
        # Require exact quoted segment containing BOTH player name and status.
        if excerpt not in raw or name.casefold() not in excerpt.casefold() or r['status'].replace('_',' ').casefold() not in excerpt.casefold():
            raise ValueError('Player status not demonstrable from raw source text')
        if qid and not qid.isdigit():raise ValueError('Invalid player ID')
        players.append({'team_id':tid,'athlete_id':qid or None,'name':name,
                        'status':r['status'],'verbatim_source_excerpt':excerpt,
                        'is_reported_out':r['status']=='OUT', 'is_confirmed_out':False})
    return {'status':'OFFICIAL_REPORT_TEXT_MATCHED_USER_SUPPLIED_CAPTURE',
        'source_url':url,'source_sha256':actual_hash,'captured_at_utc':capture.isoformat(),
        'source_verified':False, # provenance cannot be attested from user-written capture file
        'players':players,'unconfirmed_players':[], 'effects_applied_to_score':False,
        'notes':['Page text matched supplied raw snapshot but capture time not independently attested',
                 'Only athletes explicitly listed are assigned statuses; others remain unknown',
                 'No unvalidated score adjustment or inferred starting lineup']}


def correlate_availability(usage, statuses):
    """Flag high-volume QB risk; never pretend to know point impact."""
    if not isinstance(usage,dict):return []
    qb=usage.get('quarterbacks',[])
    flagged=[]
    for q in qb:
        p=next((r for r in statuses.get('players',[]) if r.get('athlete_id')==q['athlete_id'] and
                r.get('team_id')==usage.get('team_id')),None)
        if p and p['status']!='AVAILABLE':
            flagged.append({'athlete_id':q['athlete_id'],'name':q['name'],
                            'status':p['status'],
                            'recent_share_of_team_pass_plays':q['recent_share_of_all_team_pass_plays'],
                            'risk':'QB_AVAILABILITY_MATERIAL_UNKNOWN_MARGIN_IMPACT',
                            'point_adjustment':None})
    return flagged
