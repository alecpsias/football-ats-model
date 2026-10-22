"""Failure 2: strict, raw-play scoreboard attribution (no guessed points).

A game's raw PBP must expose cumulative scores AFTER each event, true
possession-team IDs, and an unambiguous order. Otherwise do not turn final
team scores into offensive production. Special teams and defense scoring
remain separate from historical offensive points-per-drive.
"""
from __future__ import annotations
import math
import pandas as pd

SCORE_PAIRS=(('home_score','away_score'),('homeScore','awayScore'),
             ('home.score','away.score'),('home_points','away_points'))
def attribute(raw, home_id, away_id, home_final, away_final):
    missing={'status':'UNVERIFIED_SCORE_ATTRIBUTION','offensive_points':None,
             'nonoffensive_points':None}
    if raw is None or len(raw)==0:return {**missing,'reason':'NO_RAW_PLAY_BY_PLAY'}
    frame=raw.copy()
    pairs=[pair for pair in SCORE_PAIRS if set(pair).issubset(frame.columns)]
    if not pairs:return {**missing,'reason':'NO_CUMULATIVE_SCORE_COLUMNS'}
    if 'game_play_number' not in frame or 'pos_team_id' not in frame:
        return {**missing,'reason':'MISSING_PLAY_ORDER_OR_POSSESSION_ID'}
    if frame.game_play_number.isna().any() or frame.game_play_number.duplicated().any():
        return {**missing,'reason':'UNVERIFIED_PLAY_ORDER'}
    frame=frame.sort_values('game_play_number')
    scores=frame[list(pairs[0])].apply(pd.to_numeric,errors='coerce')
    if scores.isna().any().any():return {**missing,'reason':'INCOMPLETE_CUMULATIVE_SCORES'}
    values=scores.to_numpy(dtype=float)
    if (values<0).any() or not all(math.isfinite(float(x)) for x in values.flat):
        return {**missing,'reason':'INVALID_SCORE_VALUES'}
    # Cumulative-score columns must represent post-play totals. Include
    # a zero-score origin and reconcile exactly with official finals.
    import numpy as np
    changes=np.diff(np.vstack([np.zeros((1,2)),values]),axis=0)
    if (changes < 0).any() or (changes>8).any() or (changes.sum(axis=1)>8).any():
        return {**missing,'reason':'IMPOSSIBLE_SCORE_TRANSITIONS'}
    if tuple(values[-1])!=(float(home_final),float(away_final)):
        return {**missing,'reason':'OFFICIAL_FINAL_RECONCILIATION_FAILED'}
    off=[0.,0.];non=[0.,0.]
    for k,row in enumerate(frame.itertuples(index=False,name=None)):
        dh,da=changes[k]
        if dh==0 and da==0:continue
        pos=frame.iloc[k]['pos_team_id']
        if pd.isna(pos):return {**missing,'reason':'UNKNOWN_SCORING_PLAY_POSSESSION'}
        if str(pos) not in (str(home_id),str(away_id)):
            return {**missing,'reason':'SCORING_TEAM_ID_MISMATCH'}
        for side,points,team in ((0,dh,home_id),(1,da,away_id)):
            if not points:continue
            # Safeties, point-after conversions, and defensive returns
            # can be scored during an offensive possession. Without an
            # explicit scoring-team and play-type proof, attribution is
            # unknown rather than determined solely from possession.
            is_off=(str(pos)==str(team))
            if points==2:
                return {**missing,'reason':'TWO_POINT_SCORING_ATTRIBUTION_AMBIGUOUS'}
            # On same-team possession, require provider flag that an
            # OFFENSE scored. A special-teams touchdown may otherwise
            # be incorrectly counted as sustainable offensive production.
            flag=frame.iloc[k].get('offense_score_play')
            if is_off:
                if pd.isna(flag) or flag not in (True,False):
                    return {**missing,'reason':'MISSING_OFFENSIVE_SCORE_CLASSIFICATION'}
                if bool(flag) and points in (3,6,7,8):off[side]+=float(points)
                elif not bool(flag) and points in (6,7,8):non[side]+=float(points)
                else:return {**missing,'reason':'AMBIGUOUS_SCORING_EVENT'}
            elif points in (6,7,8):non[side]+=float(points)
            else:return {**missing,'reason':'AMBIGUOUS_SCORING_EVENT'}
    if not (off[0]+non[0]==home_final and off[1]+non[1]==away_final):
        return {**missing,'reason':'SCORING_TOTAL_CHECK_FAILED'}
    return {'status':'VERIFIED_RAW_SCORE_RECONCILIATION',
            'offensive_points':{'home':off[0],'away':off[1]},
            'nonoffensive_points':{'home':non[0],'away':non[1]},
            'reason':None}
