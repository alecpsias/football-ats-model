"""CFB field-position and special-teams evidence from VERIFIED prior-game PBP.

Never use the scrimmage-only dataframe to infer kick/FG events: those events
were excluded during normalization. Caller must supply the raw, completed,
pre-cutoff game's plays and an authoritative schedule team ID. Unknown team
attribution, play type or kick result remains None, not fabricated zero.

These are observed descriptive stats, NOT charted special-teams EPA.
"""
from __future__ import annotations

import math
import pandas as pd


def _id(v):
    if v is None or pd.isna(v) or isinstance(v, bool):
        return None
    try:
        n = float(v)
        return int(n) if math.isfinite(n) and n.is_integer() else None
    except (ValueError, TypeError):
        return None


def _selected(plays, flags):
    """Select provider-observed event flags, never text/yardage heuristics."""
    for col in flags:
        if col in plays.columns and plays[col].notna().any():
            return plays.loc[plays[col].eq(True)].copy()
    return None


def _attributed(events, candidates, team, opponent):
    """Require a single reliable numeric kicking/receiving team ID per event."""
    if events is None:
        return None
    if events.empty:
        return events
    for col in candidates:
        if col not in events:
            continue
        ids=events[col].map(_id)
        if ids.notna().all() and ids.isin((team, opponent)).all():
            return events.loc[ids.eq(team)].copy()
    return None


def _metric(events, value, low, high):
    if events is None or value not in events:
        return None, None
    vals=pd.to_numeric(events[value],errors='coerce').dropna()
    if vals.empty:
        return None, None
    if not vals.between(low,high).all():
        return None,None
    return round(float(vals.sum()),6),int(len(vals))


def observed_special_teams(raw, team, opponent):
    """Return traceable per-game observations; no league assumptions.

    FG denominators include attempts only, returns include actual observed
    returns only (not touchbacks as zero-yard returns). Net punt yards are
    recorded ONLY if the raw provider explicitly supplies punt_net_yards;
    yds_punt_gained must NOT be mistaken for net punt distance.
    """
    default={k:None for k in ('fg_made','fg_attempts','fg_short_made',
        'fg_short_attempts','fg_medium_made','fg_medium_attempts',
        'fg_long_made','fg_long_attempts','punt_net_sum','punt_net_n',
        'kick_return_sum','kick_return_n')}
    if not isinstance(raw,pd.DataFrame) or raw.empty:
        return default
    if 'game_id' not in raw or raw['game_id'].nunique()!=1:
        return default
    fg=_attributed(_selected(raw,('fg_attempt','fg_inds')),
                   ('fg_team','kicking_team','pos_team_id'),team,opponent)
    if fg is not None and 'fg_made' in fg and fg['fg_made'].notna().all():
        result=dict(default)
        result['fg_attempts']=int(len(fg))
        result['fg_made']=int(fg['fg_made'].eq(True).sum())
        if len(fg) and 'yds_fg' in fg:
            distances=pd.to_numeric(fg['yds_fg'],errors='coerce')
            if distances.notna().all() and distances.between(15,75).all():
                for name,mask in (('short',distances.lt(40)),
                                  ('medium',distances.between(40,49)),
                                  ('long',distances.ge(50))):
                    subset=fg.loc[mask]
                    result['fg_'+name+'_attempts']=int(len(subset))
                    result['fg_'+name+'_made']=int(subset['fg_made'].eq(True).sum())
    else:
        result=dict(default)
    punts=_attributed(_selected(raw,('punt','punt_play')),
                      ('punt_team','kicking_team','pos_team_id'),team,opponent)
    result['punt_net_sum'],result['punt_net_n']=_metric(punts,'punt_net_yards',-35,110)
    net_sum,net_n=result['punt_net_sum'],result['punt_net_n']
    if net_n is not None and not 20 <= net_sum / net_n <= 70:
        result['punt_net_sum']=result['punt_net_n']=None
    returns=_attributed(_selected(raw,('kickoff_play',)),
                        ('kick_return_team','return_team'),team,opponent)
    result['kick_return_sum'],result['kick_return_n']=_metric(
        returns,'yds_kickoff_return',-20,110)
    return result
