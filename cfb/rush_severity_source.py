"""Observed CFB rushing severity inputs: early-down rush success, nonpositive runs,
and pre-play-score-dependent rushing resilience, on BOTH sides of a past game.

The ESPN PBP 'rush' flag is a designed-rushing/EPA concept, not official NCAA
team rushing including sacks. Keep it original for rush efficiency. Use only
source-provided pre-play scoring state; never reconstruct state from final score.
All numeric attempts must have observed gains and 1st/2nd down-and-distance.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

VERSION='cfb-rush-severity-source-v1'
STATES=('trailing','within_8','leading')

def extract(plays, team_id, opponent_id):
    required={'game_play_number','pos_team_id','down','distance','yards_gained',
              'rush','pass','pos_score_diff_start'}
    if plays is None or not required.issubset(plays.columns):
        return {'status':'UNAVAILABLE_REQUIRED_PBP_FIELDS'}
    x=plays.loc[plays.pos_team_id.isin((team_id,opponent_id))].copy()
    if x.empty:return {'status':'UNAVAILABLE_NO_PLAYS'}
    x=x.loc[x.rush.eq(True)&x['pass'].eq(False)&x.down.isin((1,2))]
    for col in ('distance','yards_gained','pos_score_diff_start'):
        x[col]=pd.to_numeric(x[col],errors='coerce')
    # Keep score-state missing count visible; never label as tied or losing.
    data={}
    for tid,prefix in ((team_id,'off'),(opponent_id,'def')):
        t=x.loc[x.pos_team_id.eq(tid)]
        valid=t.loc[t.distance.between(1,60)&t.yards_gained.between(-60,110)].copy()
        valid['success']=valid.yards_gained >= np.where(valid.down.eq(1),
                                                         .5*valid.distance,
                                                         .7*valid.distance)
        valid['stuffed']=valid.yards_gained.le(0)
        valid['negative']=valid.yards_gained.lt(0)
        valid['trailing']=valid.pos_score_diff_start.le(-9)
        valid['within_8']=valid.pos_score_diff_start.between(-8,8)
        valid['leading']=valid.pos_score_diff_start.ge(9)
        data.update({prefix+'_early_rush_success_yes':int(valid.success.sum()),
            prefix+'_early_rush_n':len(valid),
            prefix+'_early_nonpositive_yes':int(valid.stuffed.sum()),
            prefix+'_early_negative_yes':int(valid.negative.sum()),
            prefix+'_rush_missing_distance_n':len(t)-len(valid),
            prefix+'_rush_missing_score_state_n':int(valid.pos_score_diff_start.isna().sum())})
        for state in STATES:
            z=valid.loc[valid.pos_score_diff_start.notna() & valid[state]]
            data[prefix+'_'+state+'_rush_good']=int(z.success.sum())
            data[prefix+'_'+state+'_rush_n']=len(z)
    if min(data['off_early_rush_n'],data['def_early_rush_n'])<3:
        return {'status':'INSUFFICIENT_OBSERVED_EARLY_RUSHES','features':data}
    return {'status':'OBSERVED_VALID','version':VERSION,
            'features':data,
            'score_state_from':'pos_score_diff_start',
            'negative_run':'yards_gained<0',
            'stuffed_run':'yards_gained<=0',
            'early_rush_success_definition':'1st >=50%, 2nd >=70% yards to gain',
            'EPA_labels_unchanged':True}

def append_training_record(record,observation):
    if observation.get('status')!='OBSERVED_VALID':return
    for key,value in observation['features'].items():
        record['mismatch_'+key]=value
