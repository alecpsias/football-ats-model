"""Strict CFB play-selection and *observed* clock-possession audit.

Original ESPN raw PBP contains repeated drive.timeElapsed.displayValue,
drive.start/end clocks and play-type tags. The canonical scrimmage-only
model frame EXCLUDES punts/FGs and must NOT be used to time possessions.

Clock durations are from SOURCE PROVIDED drive elapsed times, never derived
from scrimmage play count. Group by drive_id/game_id; ambiguous ownership,
mismatched clocks, inconsistent drive metadata, overtime and missing events
fail closed. Source labels are retrospective, not permitted in pregame inputs.
"""
from __future__ import annotations
import math
import re
from collections import Counter
import numpy as np
import pandas as pd

SOURCE_TIME='drive.timeElapsed.displayValue'
START_PERIOD='drive.start.period.number'
START_CLOCK='drive.start.clock.displayValue'
END_PERIOD='drive.end.period.number'
END_CLOCK='drive.end.clock.displayValue'
POSSIBLE_ID=('drive_id','drive.id')
TIME_PARSE=re.compile(r'^([0-9]{1,2}):([0-9]{2})$')

def _seconds(s):
    if s is None or pd.isna(s):return None
    m=TIME_PARSE.fullmatch(str(s).strip())
    if m is None:return None
    minute,second=int(m[1]),int(m[2])
    if not 0<=second<60 or not 0<=minute<=60:return None
    return minute*60+second

def _reg_abs_clock(period,clock):
    try:
        q=float(period)
    except (TypeError, ValueError):return None
    sec=_seconds(clock)
    if not math.isfinite(q) or not q.is_integer() or not 1<=q<=4 or sec is None or sec>900:
        return None
    return (4-int(q))*900+sec

def _one(rows,col):
    if col not in rows:return None
    values=rows[col].dropna().drop_duplicates().tolist()
    return values[0] if len(values)==1 else None

def _identifier(raw):
    return next((c for c in POSSIBLE_ID if c in raw.columns),None)

def _as_bool(frame,key):
    if key not in frame:return None
    v=frame[key]
    if v.isna().any():return None
    if not v.isin((True,False)).all():return None
    return v.eq(True)

def audit_game(raw,game_id,home_id,away_id):
    """Produce observed drive durations and run shares; no score or result used.

    Observed elapsed clock must pass independent start/end clock cross-check
    where end points exist. Zero-length drives are allowed but kept visible.
    An official 3600-second TOP is NOT silently enforced; report coverage and
    compare with externally obtained game box score.
    """
    result={'game_id':str(game_id),'status':'UNAVAILABLE',
            'excluded':{},'excluded_drive_examples':[],'drives':[],'teams':{},'observed_seconds_total':None}
    if raw is None or raw.empty or 'game_id' not in raw:
        result['status']='NO_SOURCE';return result
    g=raw.loc[raw.game_id.eq(game_id)].copy()
    did=_identifier(g)
    required={'game_play_number','pos_team_id','pass','rush',SOURCE_TIME,
              START_PERIOD,START_CLOCK,END_PERIOD,END_CLOCK}
    if did is None or not required.issubset(g.columns):
        result['status']='MISSING_DRIVE_CLOCK_SCHEMA'
        result['missing']=sorted(required-set(g.columns))
        return result
    g=g.loc[g[did].notna()].sort_values('game_play_number')
    reasons=Counter()
    def reject(reason,drive_id,part):
        reasons[reason]+=1
        if len(result['excluded_drive_examples'])<30:
            x={'drive_id':str(drive_id),'reason':reason,
               'elapsed_raw':str(_one(part,SOURCE_TIME)),
               'start_period':str(_one(part,START_PERIOD)),
               'end_period':str(_one(part,END_PERIOD)),
               'start_clock':str(_one(part,START_CLOCK)),
               'end_clock':str(_one(part,END_CLOCK)),
               'pass_count':int(part['pass'].eq(True).sum()),
               'rush_count':int(part['rush'].eq(True).sum())}
            result['excluded_drive_examples'].append(x)
    groups=g.groupby(did,sort=False)
    for drive_id,part in groups:
        # ESPN drive metadata can be repeated on all plays in the drive;
        # require the owner from actual scrimmage plays rather than the
        # kicking/receiving team in a kickoff.
        p=_as_bool(part,'pass');r=_as_bool(part,'rush')
        if p is None or r is None:
            reject('unknown_play_classification',drive_id,part);continue
        scrim=part.loc[p^r]
        if scrim.empty:
            reject('no_scrimmage_plays',drive_id,part);continue
        ids=scrim.pos_team_id.dropna().drop_duplicates().tolist()
        if len(ids)!=1 or ids[0] not in (home_id,away_id):
            reject('ambiguous_offense_team',drive_id,part);continue
        team=int(ids[0])
        reg_start=_reg_abs_clock(_one(part,START_PERIOD),_one(part,START_CLOCK))
        reg_end=_reg_abs_clock(_one(part,END_PERIOD),_one(part,END_CLOCK))
        if reg_start is None or reg_end is None:
            reject('missing_regulation_clock',drive_id,part);continue
        difference=reg_start-reg_end
        if not 0<=difference<=900:
            reject('invalid_regulation_clock_delta',drive_id,part);continue
        elapsed=_seconds(_one(part,SOURCE_TIME))
        if elapsed is None or not 0<=elapsed<=900:
            reject('missing_or_invalid_source_elapsed',drive_id,part);continue
        if abs(difference-elapsed)>10:
            reject('source_clock_disagreement',drive_id,part);continue
        numrun=int(r.loc[scrim.index].sum())
        numpass=int(p.loc[scrim.index].sum())
        if numrun+numpass<1:
            reject('no_valid_run_pass_plays',drive_id,part);continue
        result['drives'].append({'drive_id':str(drive_id),'team_id':team,
               'duration_seconds':elapsed,'clock_delta_seconds':difference,
               'rush_plays':numrun,'pass_plays':numpass,
               'scrimmage_plays':numrun+numpass,
               'rush_share':round(numrun/(numrun+numpass),6)})
    result['excluded']=dict(reasons)
    result['source_drive_count']=int(groups.ngroups)
    result['valid_drive_count']=len(result['drives'])
    valid=result['drives']
    result['observed_seconds_total']=sum(d['duration_seconds'] for d in valid)
    for tid in (home_id,away_id):
        sub=[d for d in valid if d['team_id']==tid]
        n=len(sub);seconds=sum(d['duration_seconds'] for d in sub)
        run=sum(d['rush_plays'] for d in sub)
        pas=sum(d['pass_plays'] for d in sub)
        result['teams'][str(tid)]={'valid_drives':n,'observed_time_seconds':seconds,
            'seconds_per_drive':round(seconds/n,4) if n else None,
            'run_share':round(run/(run+pas),6) if run+pas else None,
            'rush_plays':run,'pass_plays':pas,
            'scrimmage_plays_per_drive':round((run+pas)/n,4) if n else None}
    # Game clock possession is a 60 min zero-sum allocation in regulation.
    # This coverage threshold is a conservative source completeness guard,
    # not a way of inventing missing seconds.
    result['regulation_clock_coverage']=round(result['observed_seconds_total']/3600,5)
    if (min(result['teams'][str(t)]['valid_drives'] for t in (home_id,away_id))>=6
            and len(valid)>=14 and
            3150<=result['observed_seconds_total']<=3650):
        result['status']='COMPLETE_REGULATION_CLOCK_AUDIT'
    else:
        result['status']='PARTIAL_CLOCK_COVERAGE'
    return result


def clock_reference_difference(game,reference):
    """Compare verified drive clocks against an INDEPENDENT official box."""
    if game['status']=='MISSING_DRIVE_CLOCK_SCHEMA':return {'status':'NO_SCHEMA'}
    errors={}
    for team_id,seconds in reference.items():
        got=game['teams'].get(str(team_id),{}).get('observed_time_seconds')
        if got is None:return {'status':'MISSING_TEAM'}
        errors[str(team_id)]=int(got-int(seconds))
    return {'status':'PASS_WITHIN_90_SEC' if max(abs(v) for v in errors.values())<=90
            else 'DISAGREES_WITH_OFFICIAL_TOP','difference_seconds_by_team':errors,
            'max_abs_error_seconds':max(abs(v) for v in errors.values())}
