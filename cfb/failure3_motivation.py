"""Failure 3: no-fit forensic replay of NM State at Sam Houston.

The motivating game is EXCLUDED from paired validation and coefficient
selection. This tool merely asks whether frozen prior-game inputs would
activate the new sustainability interaction on that known miss.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from cfb.failure2_paired import cohort
from cfb.mechanism_build16 import simulate_build16
from cfb.failure1_learned_shadow import correction as prior_failure1

GAME_ID='401871070'
def diagnose(snapshot):
    full,counts=cohort(snapshot,max_games=300)
    matches=[x for x in full if x['game_id']==GAME_ID]
    if len(matches)!=1:
        return {'status':'MOTIVATING_GAME_NOT_ELIGIBLE_WITH_PRIOR_GAMES',
                'id':GAME_ID,'eligible_count':len(full)}
    g=matches[0];h=g['historical'];hi,ai=g['home_id'],g['away_id']
    seed=int(GAME_ID)%(2**32)
    rush=prior_failure1(hi,ai,h)['margin_delta_points']
    common=dict(runs=5000,seed=seed,failure1_fit_margin_delta=rush,failure2_attribution=True)
    old=simulate_build16(hi,ai,h,g['neutral'],failure3_sustainability=False,**common)
    new=simulate_build16(hi,ai,h,g['neutral'],failure3_sustainability=True,**common)
    a,b=old.get('simulation'),new.get('simulation')
    if not a or not b: return {'status':'BUILD16_UNAVAILABLE','old':old['status'],'new':new['status']}
    result={'status':'FORENSIC_DIAGNOSTIC_NOT_TRAINING_OR_HOLDOUT',
        'game_id':GAME_ID,'home':g['home'],'away':g['away'],
        'kickoff':g['kickoff'],'actual_margin_home':g['actual_margin'],
        'f1_held_constant':rush,
        'f2_held_constant':new['failure2_scoring_attribution']['status'],
        'previous_home_margin':a['mean_margin_home'],
        'failure3_home_margin':b['mean_margin_home'],
        'forecast_delta':round(b['mean_margin_home']-a['mean_margin_home'],5),
        'previous_total':a['mean_total'],'failure3_total':b['mean_total'],
        'before_interval':a['margin_p10_p50_p90'],
        'after_interval':b['margin_p10_p50_p90'],
        'sustainability_evidence':new['failure3_sustainability'],
        'simulations_each':5000,
        'no_target_game_used_as_predictor':True,
        'postgame_check_not_calibration':True}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    result=diagnose(a.snapshot)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
