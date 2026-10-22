"""Failure 2: verifiable play-level offensive, defensive and ST scoring.

Read *raw* scoring events (including returns), require cumulative scores to
reconcile to the official final, and fail closed on ambiguous score sources.
Do NOT decide return-TD type from possession alone. Prior game records only.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from cfb.official_scoring_checks import official_reconciliation

SCORE_PAIRS=(('home_score','away_score'),('homeScore','awayScore'),
             ('home.score','away.score'),('home_points','away_points'))
SIDES=('home','away')


def _false_or_true(value):
    if pd.isna(value):return None
    if value == True:return True
    if value == False:return False
    return None


def attribute(raw, home_id, away_id, home_final, away_final):
    missing={'status':'UNVERIFIED_SCORE_ATTRIBUTION',
             'offensive_points':None,'nonoffensive_points':None,
             'defensive_return_points':None,'special_teams_points':None,
             'defensive_return_tds':None,'special_teams_tds':None}
    def reject(reason):return {**missing,'reason':reason}
    if raw is None or len(raw)==0:return reject('NO_RAW_PLAY_BY_PLAY')
    frame=raw.copy()
    pairs=[pair for pair in SCORE_PAIRS if set(pair).issubset(frame.columns)]
    if not pairs:return reject('NO_CUMULATIVE_SCORE_COLUMNS')
    required={'game_play_number','pos_team_id','offense_score_play','pass','rush',
              'int','pos_fumble_lost'}
    if not required.issubset(frame.columns):
        return reject('MISSING_PLAY_CLASSIFICATION_OR_TEAM_ID')
    if frame.game_play_number.isna().any() or frame.game_play_number.duplicated().any():
        return reject('UNVERIFIED_PLAY_ORDER')
    frame=frame.sort_values('game_play_number')
    scores=frame[list(pairs[0])].apply(pd.to_numeric,errors='coerce')
    if scores.isna().any().any():return reject('INCOMPLETE_CUMULATIVE_SCORES')
    values=scores.to_numpy(dtype=float)
    if (values<0).any() or not np.isfinite(values).all():
        return reject('INVALID_SCORE_VALUES')
    deltas=np.diff(np.vstack([np.zeros((1,2)),values]),axis=0)
    if tuple(values[-1])!=(float(home_final),float(away_final)):
        return reject('OFFICIAL_FINAL_RECONCILIATION_FAILED')
    # Independently verified institutional scorebooks correct known ESPN
    # delayed overtime updates and misclassified return scores. Only use
    # exact raw game IDs, finals and explicit return text; never global
    # team-level assumptions or guessed zero non-offensive scores.
    official=official_reconciliation(frame,home_id,away_id,home_final,away_final)
    if official is not None:return official
    if (deltas<0).any() or (deltas>8).any() or (deltas.sum(axis=1)>8).any():
        return reject('IMPOSSIBLE_SCORE_TRANSITIONS')
    offensive={side:0. for side in SIDES}
    defensive={side:0. for side in SIDES}
    special={side:0. for side in SIDES}
    defensive_tds={side:0 for side in SIDES}
    special_tds={side:0 for side in SIDES}
    event_counts={'offensive':0,'defensive_return':0,'special_teams':0}
    for k in range(len(frame)):
        dh,da=deltas[k]
        if not (dh or da):continue
        pos=frame.iloc[k]['pos_team_id']
        if pd.isna(pos) or str(pos) not in (str(home_id),str(away_id)):
            return reject('SCORING_TEAM_ID_MISMATCH')
        row=frame.iloc[k]
        scrimmage_pass=_false_or_true(row['pass'])
        scrimmage_rush=_false_or_true(row['rush'])
        was_int=_false_or_true(row['int'])
        was_fumble=_false_or_true(row['pos_fumble_lost'])
        offense_flag=_false_or_true(row['offense_score_play'])
        if any(x is None for x in (scrimmage_pass,scrimmage_rush,was_int,was_fumble,offense_flag)):
            return reject('INCOMPLETE_SCORING_EVENT_FLAGS')
        scrimmage=scrimmage_pass or scrimmage_rush
        turnover=was_int or was_fumble
        # A +7 score on a nonscrimmage/false-offense-flag row does NOT
        # establish a special teams TD. ESPN often attaches scores to
        # subsequent kickoff/PAT rows. Demand an explicit event type.
        event_type=str(row.get('type.text','')).strip().lower()
        detail=str(row.get('text','')).strip().lower()
        scoring_text=event_type+' '+detail
        special_return=(
            ('touchdown' in scoring_text) and (
             'punt return' in scoring_text or 'kickoff return' in scoring_text
             or 'kick return' in scoring_text or
             ('blocked' in scoring_text and ('punt' in scoring_text or
                                               'field goal' in scoring_text))))
        defensive_return=(
            ('touchdown' in scoring_text) and (
             'interception return' in scoring_text or
             'fumble return' in scoring_text))
        explicit_offense_td=(
            'passing touchdown' in event_type or 'rushing touchdown' in event_type)
        explicit_fg=('field goal good' in event_type or
                     'field goal is good' in detail or
                     bool(row.get('fg_made',False)) is True)
        for side,points,team in (('home',dh,home_id),('away',da,away_id)):
            if not points:continue
            # Two-point plays, safeties and split PAT scoring need explicitly
            # reliable type information not present in this raw schema.
            if points==2:return reject('TWO_POINT_SCORING_ATTRIBUTION_AMBIGUOUS')
            if points not in (3,6,7,8):return reject('AMBIGUOUS_SCORING_EVENT')
            own_possession=str(pos)==str(team)
            if special_return and points in (6,7,8):
                special[side]+=float(points)
                special_tds[side]+=1
                event_counts['special_teams']+=1
            elif (defensive_return and not own_possession and turnover
                  and points in (6,7,8)):
                defensive[side]+=float(points)
                defensive_tds[side]+=1
                event_counts['defensive_return']+=1
            elif (own_possession and not special_return and not defensive_return
                  and ((points==3 and explicit_fg)
                    or (points in (6,7,8) and not turnover
                        and (explicit_offense_td or (offense_flag and scrimmage))))):
                offensive[side]+=float(points)
                event_counts['offensive']+=1
            elif (not own_possession and scrimmage and turnover
                  and points in (6,7,8) and not offense_flag):
                defensive[side]+=float(points)
                defensive_tds[side]+=1
                event_counts['defensive_return']+=1
            else:
                return reject('SCORING_TYPE_UNVERIFIED_NOT_ASSUMED_SPECIAL_TEAMS')
    for side,final in (('home',float(home_final)),('away',float(away_final))):
        if offensive[side]+defensive[side]+special[side]!=final:
            return reject('SCORING_TOTAL_CHECK_FAILED')
    non={s:defensive[s]+special[s] for s in SIDES}
    return {'status':'VERIFIED_RAW_SCORE_RECONCILIATION',
            'verification_method':'EXPLICIT_PLAY_TYPE_AND_RAW_SCORE_DELTA',
            'offensive_points':offensive,'nonoffensive_points':non,
            'defensive_return_points':defensive,'special_teams_points':special,
            'defensive_return_tds':defensive_tds,'special_teams_tds':special_tds,
            'scoring_event_counts':event_counts,'reason':None}
