"""Failure 3: source-clock-verified durations, grouped by actual drive outcome.

Historical observations are stored on the ORIGINAL prior-game records, never
derived from the target matchup. Only drives with ESPN's elapsed field AND
agreeing independent start/end clocks are eligible. A complete-regulation
game-level coverage check is required. This does NOT claim independent
official-box TOP verification. We never manufacture missing durations.
"""
from __future__ import annotations
from collections import Counter
import math
import numpy as np
import pandas as pd
from cfb.clock_possession_validation import audit_game, _identifier

LABELS=('explosive_td','chain_score','turnover','stalled')
VERSION='cfb-verified-drive-clock-outcome-v1'

def _category(scrimmage,raw_part):
    """Partition into mutually exclusive outcomes, before using clock."""
    if scrimmage.empty:return None
    required={'down','yards_gained','pass','rush','pass_td','rush_td',
              'int','pos_fumble_lost','offense_score_play'}
    if not required.issubset(scrimmage.columns):return None
    if scrimmage[list(required)].isna().any().any():return None
    scrimmage=scrimmage.sort_values('game_play_number')
    td=(scrimmage.pass_td.eq(True)|scrimmage.rush_td.eq(True)) & scrimmage.offense_score_play.eq(True)
    scoring_td=bool(td.any())
    firsts=int(scrimmage.down.eq(1).sum())
    explosive=bool((scrimmage['pass'].eq(True) & scrimmage.yards_gained.ge(20)).any())
    if scoring_td and explosive and firsts<=2:return 'explosive_td'
    if scoring_td:return 'chain_score'
    if 'fg_made' in raw_part and raw_part.fg_made.eq(True).any():return 'chain_score'
    if (scrimmage['int'].eq(True)|scrimmage.pos_fumble_lost.eq(True)).any():
        return 'turnover'
    return 'stalled'

def observe(raw,plays,team,opponent,home_id):
    """Only use complete official-score-independent SOURCE drive clocks."""
    if raw is None or raw.empty or 'game_id' not in raw or home_id is None:
        return {'status':'NO_RAW_CLOCK_SOURCE'}
    game_ids=raw.game_id.dropna().unique()
    if len(game_ids)!=1:return {'status':'AMBIGUOUS_GAME_ID'}
    home_id=int(home_id)
    away_id=int(opponent) if int(team)==home_id else int(team)
    audit=audit_game(raw,game_ids[0],home_id,away_id)
    summary={'status':audit['status'],'clock_verification':'ESPN_ELAPSED_AND_ENDPOINTS',
             'game_id':str(game_ids[0]),
             'game_coverage':audit.get('regulation_clock_coverage'),
             'official_box_top_verified':False,
             'excluded_drive_reasons':audit.get('excluded',{})}
    if audit['status']!='COMPLETE_REGULATION_CLOCK_AUDIT':
        return summary
    raw_drive_column=_identifier(raw)
    if raw_drive_column is None:return {**summary,'status':'MISSING_RAW_DRIVE_IDENTIFIER'}
    clocks={str(d['drive_id']):d for d in audit['drives']
            if d['clock_verification']=='ELAPSED_AND_ENDPOINTS_AGREE'}
    groups={}
    for side,tid in (('off',team),('def',opponent)):
        buckets={key:[] for key in LABELS}
        scrim=plays.loc[plays.pos_team_id.eq(tid)]
        raw_drive=raw.loc[raw.pos_team_id.eq(tid)]
        for drive_id,g in scrim.groupby('drive_id',sort=False):
            clock=clocks.get(str(drive_id))
            if clock is None or clock['team_id']!=int(tid):continue
            p=raw_drive.loc[raw_drive[raw_drive_column].astype(str).eq(str(drive_id))]
            category=_category(g,p)
            duration=clock['duration_seconds']
            if category is None or not type(duration) in (int,float) or not 0<=duration<=900:
                continue
            # Require positive actual regulation duration. Zero-clock plays
            # cannot be used to model a normal sequence of possessions.
            if duration>0:buckets[category].append(int(duration))
        groups[side]=buckets
    n=sum(len(x) for x in groups['off'].values())
    m=sum(len(x) for x in groups['def'].values())
    summary.update({'status':'SOURCE_CLOCK_VERIFIED' if min(n,m)>=4
                         else 'INSUFFICIENT_VERIFIED_CLOCK_DRIVES',
                    'off':groups['off'],'def':groups['def'],
                    'observed_offense_drives':n,'observed_opponent_drives':m,
                    'categories':LABELS})
    return summary

def _aggregate(rows,side):
    games=0
    samples={k:[] for k in LABELS}
    for r in rows:
        c=r.get('drive_clock')
        if not isinstance(c,dict) or c.get('status')!='SOURCE_CLOCK_VERIFIED':continue
        if c.get('clock_verification')!='ESPN_ELAPSED_AND_ENDPOINTS':continue
        groups=c.get(side,{})
        if not all(isinstance(groups.get(k),list) for k in LABELS):continue
        if not all(type(t) is int and 1<=t<=900
                   for k in LABELS for t in groups[k]):continue
        games+=1
        for k in LABELS:samples[k].extend(groups[k])
    return games,samples

def matchup_clocks(home_id,away_id,histories):
    """Return historical empirical duration pools or fail closed."""
    pools={}
    for label,tid,opponent in (('home',home_id,away_id),('away',away_id,home_id)):
        og,off=_aggregate(histories.get(tid,[]),'off')
        dg,defense=_aggregate(histories.get(opponent,[]),'def')
        merged={k:off[k]+defense[k] for k in LABELS}
        overall=[v for k in LABELS for v in merged[k]]
        if min(og,dg)<2 or len(overall)<24:
            return {'status':'UNVERIFIED_MATCHED_HISTORICAL_CLOCKS',
                    'missing_for':label,'games_off':og,'games_opponent_def':dg,
                    'matched_clock_drives':len(overall)}
        # Classes with <3 observed durations fall back to matched REAL
        # possessions rather than fabricated fast/slow timing coefficients.
        pools[label]={'values':{k:(merged[k] if len(merged[k])>=3 else overall)
                                for k in LABELS},
                      'category_counts':{k:len(merged[k]) for k in LABELS},
                      'fallback_to_all_counts':{k:len(merged[k]) for k in LABELS if len(merged[k])<3},
                      'games_off':og,'games_opponent_def':dg,
                      'matched_clock_drives':len(overall)}
    return {'status':'ACTIVE_VERIFIED_SOURCE_DRIVE_CLOCKS',
            'source':'ESPN drive elapsed + independent start/end clocks; full regulation coverage',
            'official_box_top_verified':False,'home':pools['home'],'away':pools['away']}

def sample_seconds(rng,pool,category,mask):
    n=len(mask)
    out=np.zeros(n,dtype=float)
    if not np.any(mask):return out
    vals=pool['values'][category]
    if not isinstance(vals,list) or not vals or any(type(v) is not int or not 1<=v<=900 for v in vals):
        raise ValueError('Unavailable verified drive-clock pool')
    out[mask]=rng.choice(np.array(vals,dtype=float),size=int(mask.sum()))
    return out
