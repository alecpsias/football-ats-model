"""Source-grounded *research-only* CFB early-rush / third-down / drive-chain data.

This is a separate contextual layer. Do not use provider 'firstD_by_yards'
as a sole conversion denominator: it is inconsistent with university box scores.
Use ordered same-drive next-scrimmage down resets and verified terminal TDs.
A missing terminal outcome is not a recorded failure. Never use a target game's
own observations as pregame inputs. Original pass/rush EPA labels unchanged.
"""
from __future__ import annotations
import math
from collections import Counter
import numpy as np
import pandas as pd

FEATURE_VERSION='cfb-third-chain-observation-v1'

def _finite(v):
    return type(v) in (float,int) and not isinstance(v,bool) and math.isfinite(v)

def _group(raw,team):
    if not {'pos_team_id','game_play_number','drive_id','down','distance','rush',
            'pass','yards_gained'}.issubset(raw.columns):
        return None
    x=raw.loc[raw.pos_team_id.eq(team)].copy()
    if x.empty or x.drive_id.isna().any() or x.game_play_number.isna().any():
        return None
    if x[['down','distance','rush','pass','yards_gained']].isna().any().any():
        return None
    x=x.loc[x.rush.eq(True)^x['pass'].eq(True)].sort_values('game_play_number')
    if x.empty:return None
    return x

def observed_chain(plays,team,opponent):
    """Return paired source statistics; exclude ambiguous outcomes, no guesses."""
    x=_group(plays,team);d=_group(plays,opponent)
    if x is None or d is None:return {'status':'UNAVAILABLE_SOURCE_FIELDS'}
    out={}
    def side(rows):
        early_good=early_total=early_rush=early_all=0
        early_down1_good=early_down1_n=early_down2_good=early_down2_n=0
        third_short_yes=third_short_n=third_medium_yes=third_medium_n=third_long_yes=third_long_n=0
        third_manage=third_total=third_yes=third_n=0
        third_official_yes=0
        terminal_unresolved=0
        long_drives=all_drives=0
        early_run_success_good=early_run_success_n=0
        third_distance_sum=0.
        for _,drive in rows.groupby('drive_id',sort=False):
            drive=drive.sort_values('game_play_number')
            valid=drive.loc[drive.down.isin([1,2,3,4])]
            if valid.empty:continue
            all_drives+=1
            # A drive beginning at first-and-goal is legitimately counted as
            # one series; we need 3 distinct first-and-X occurrences to call
            # a drive sustained. Penalty resets can create another series.
            firsts=int(valid.down.eq(1).sum())
            if firsts>=3:long_drives+=1
            for j,(_,z) in enumerate(valid.iterrows()):
                down=int(z.down)
                dist=float(z.distance)
                gain=float(z.yards_gained)
                if not 1<=dist<=60 or not -60<=gain<=110:continue
                if down<=2:
                    early_all+=1
                    # Standard early-down success: >=50% on first, >=70% on second.
                    good=gain >= (.5 if down==1 else .7)*dist
                    early_good+=int(good);early_total+=1
                    if down==1:
                        early_down1_good+=int(good);early_down1_n+=1
                    else:
                        early_down2_good+=int(good);early_down2_n+=1
                    if bool(z.rush):
                        early_rush+=1;early_run_success_n+=1
                        thresh=(.5 if down==1 else .7)*dist
                        if gain>=thresh:early_run_success_good+=1
                    continue
                if down!=3:continue
                third_total+=1
                third_distance_sum+=dist
                if dist<=4:third_manage+=1
                # An explicitly marked first down or TD is positive evidence,
                # but provider firstD_by_yards=FALSE isn't negative evidence.
                nextrow=valid.iloc[j+1] if j+1<len(valid) else None
                converted=None
                if nextrow is not None:
                    nextdown=int(nextrow.down)
                    if nextdown==1:converted=True
                    elif nextdown==4:converted=False
                if converted is None and any(
                    c in valid.columns and pd.notna(z[c]) and bool(z[c])
                    for c in ('firstD_by_yards','firstD_by_penalty','rush_td','pass_td')):
                    converted=True
                if converted is None and any(
                    c in valid.columns and z[c] is True
                    for c in ('int','pos_fumble_lost','downs_turnover')):
                    converted=False
                if converted is None:
                    terminal_unresolved+=1
                else:
                    third_n+=1
                    third_yes+=int(converted)
                    if dist<=3:
                        third_short_n+=1;third_short_yes+=int(converted)
                    elif dist<=6:
                        third_medium_n+=1;third_medium_yes+=int(converted)
                    else:
                        third_long_n+=1;third_long_yes+=int(converted)
        return {'early_success_yes':early_good,'early_success_n':early_total,
                'first_down_success_yes':early_down1_good,'first_down_success_n':early_down1_n,
                'second_down_success_yes':early_down2_good,'second_down_success_n':early_down2_n,
                'third_short_yes':third_short_yes,'third_short_n':third_short_n,
                'third_medium_yes':third_medium_yes,'third_medium_n':third_medium_n,
                'third_long_yes':third_long_yes,'third_long_n':third_long_n,
                'early_rush_success_yes':early_run_success_good,
                'early_rush_success_n':early_run_success_n,
                'early_rush_plays':early_rush,'early_offense_plays':early_all,
                'third_manageable_yes':third_manage,'third_attempts':third_total,
                'third_conversion_yes':third_yes,'third_resolved_n':third_n,
                'third_unresolved_n':terminal_unresolved,
                'third_distance_sum':round(third_distance_sum,5),
                'sustained_drives':long_drives,'observed_drives':all_drives}
    off=side(x);allowed=side(d)
    if min(off['observed_drives'],allowed['observed_drives'])<3:
        return {'status':'INSUFFICIENT_DRIVES','offense':off,'opponent':allowed}
    return {'status':'OBSERVED_QUALIFIED','version':FEATURE_VERSION,
            'offense':off,'opponent':allowed,
            'rule':('third outcome next within-drive scrimmage down reset / next fourth, '
                    'or source-positive marker / documented terminal turnover; '
                    'unknown terminal third outcomes excluded from conversion denominator'),
            'raw_positive_flag_not_trusted_as_negative':True}

def append_historical_chain(rec,observed,side):
    """Copy features to training record; operate on prior-game rows ONLY."""
    if observed.get('status')!='OBSERVED_QUALIFIED':return
    a=observed['offense' if side=='offense' else 'opponent']
    p='chain_off_' if side=='offense' else 'chain_def_'
    rec.update({p+key:value for key,value in a.items()})
