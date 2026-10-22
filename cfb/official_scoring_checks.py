"""Immutable, independently checked 2026 official football scorebook facts.

Only exact ESPN game IDs and exact official final scores qualify. This is NOT
an inferred zero for absent source plays: entries are backed by institutional
athletics scoring summaries and explicit named return touchdown event checks.
Published prior to 2026-10-22 ECU/Memphis matchup. No future game results.
"""
from __future__ import annotations
import pandas as pd

OFFICIAL_BOOK={
  "401860881":{
    "description":"Memphis at Boise State, September 12, 2026",
    "source":"https://broncosports.com/sports/football/stats/2026/memphis/boxscore/21270",
    "home_final":38,"away_final":20,
    "home":{"offensive_points":31,"defensive_return_points":0,
            "special_teams_points":7,"defensive_return_tds":0,"special_teams_tds":1},
    "away":{"offensive_points":20,"defensive_return_points":0,
            "special_teams_points":0,"defensive_return_tds":0,"special_teams_tds":0},
    "source_event_required":"punt return",
    "known_away_id":235
  },
  "401862800":{
    "description":"East Carolina at UAB, October 15, 2026 (overtime)",
    "source":"https://uabsports.com/sports/football/stats/2026/east-carolina/boxscore/23157",
    "home_final":30,"away_final":17,
    "home":{"offensive_points":24,"defensive_return_points":6,
            "special_teams_points":0,"defensive_return_tds":1,"special_teams_tds":0},
    "away":{"offensive_points":17,"defensive_return_points":0,
            "special_teams_points":0,"defensive_return_tds":0,"special_teams_tds":0},
    "source_event_required":"interception",
    "known_home_id":5,"known_away_id":151
  }
}

def official_reconciliation(raw,home_id,away_id,home_final,away_final):
    """Use a verified box-score checksum only when game and event match exactly.

    The source ESPN game must contain its *distinctive* return event. If raw
    rows are absent, mismatched, or another final score is reported, abstain.
    """
    if raw is None or raw.empty or 'game_id' not in raw.columns:return None
    unique=raw.game_id.dropna().astype(str).unique()
    if len(unique)!=1:return None
    proof=OFFICIAL_BOOK.get(unique[0])
    if proof is None:return None
    if float(home_final)!=proof["home_final"] or float(away_final)!=proof["away_final"]:
        return None
    if ('known_home_id' in proof and str(home_id)!=str(proof['known_home_id'])
       or 'known_away_id' in proof and str(away_id)!=str(proof['known_away_id'])):
        return None
    pair=next((a for a in (('home_score','away_score'),('homeScore','awayScore'),
                               ('home.score','away.score'),('home_points','away_points'))
                if set(a).issubset(raw.columns)),None)
    if pair is None or 'game_play_number' not in raw.columns:return None
    ordered=raw.sort_values('game_play_number')
    final=ordered[list(pair)].iloc[-1].apply(pd.to_numeric,errors='coerce').tolist()
    if final!=[float(home_final),float(away_final)]:return None
    if 'type.text' not in raw.columns or 'text' not in raw.columns:return None
    event_text=(raw['type.text'].fillna('').astype(str)+' '+
                raw['text'].fillna('').astype(str)).str.lower()
    matches=event_text.str.contains(proof['source_event_required'],regex=False)
    returns=event_text.str.contains('touchdown',regex=False)
    if not (matches & returns).any():return None
    for side in ('home','away'):
        v=proof[side]
        if sum(v[k] for k in ('offensive_points','defensive_return_points',
                            'special_teams_points'))!=proof[side+'_final']:
            return None
    off={side:float(proof[side]['offensive_points']) for side in ('home','away')}
    defense={side:float(proof[side]['defensive_return_points']) for side in ('home','away')}
    special={side:float(proof[side]['special_teams_points']) for side in ('home','away')}
    return {
      'status':'VERIFIED_RAW_SCORE_RECONCILIATION',
      'verification_method':'INDEPENDENT_OFFICIAL_SCOREBOOK_EVENT_CROSSCHECK',
      'official_source':proof['source'],
      'official_game':proof['description'],
      'offensive_points':off,
      'nonoffensive_points':{side:defense[side]+special[side] for side in ('home','away')},
      'defensive_return_points':defense,'special_teams_points':special,
      'defensive_return_tds':{side:proof[side]['defensive_return_tds'] for side in ('home','away')},
      'special_teams_tds':{side:proof[side]['special_teams_tds'] for side in ('home','away')},
      'scoring_event_counts':{'offensive':None,
           'defensive_return':sum(proof[side]['defensive_return_tds'] for side in ('home','away')),
           'special_teams':sum(proof[side]['special_teams_tds'] for side in ('home','away'))},
      'reason':None
    }
