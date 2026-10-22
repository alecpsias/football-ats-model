"""Separate non-offensive scoring mechanisms for the sole CFB Build16 challenger.

Defensive TDs are CONDITIONED on a simulated turnover and suppress a turnover
short-field reward. Special-teams TDs use their OWN bounded game rate. Rates
are conservatively smoothed priors, NOT trained/calibrated ATS probabilities.
"""
from __future__ import annotations
import math
import numpy as np

TURNOVER_RETURN_TD_PRIOR=0.04
TURNOVER_PRIOR_EXPOSURES=35.0
SPECIAL_TD_GAME_PRIOR=0.025
SPECIAL_PRIOR_GAMES=30.0
MAX_RETURN_RATE=.22
MAX_SPECIAL_GAME_RATE=.20

def _history(rows):
    if not isinstance(rows,list) or len(rows)<3:
        raise ValueError('Require three or more verified historical games')
    fields=('defensive_return_tds','defensive_return_tds_allowed',
            'defensive_return_points','special_teams_tds','special_teams_points',
            'offense_turnovers','defense_opportunities')
    totals={k:0. for k in fields}
    defensive_values=[]
    special_values=[]
    for r in rows:
        if not isinstance(r,dict):raise ValueError('Invalid return scoring row')
        for k in fields:
            x=r.get(k)
            if type(x) not in (float,int) or not math.isfinite(x) or x<0:
                raise ValueError('Missing or invalid verified return scoring data: '+k)
            totals[k]+=float(x)
        if (r['defensive_return_tds']>r['defense_opportunities']
                or r['defensive_return_tds_allowed']>r['offense_turnovers']
                or r['defensive_return_points']>8*r['defensive_return_tds']
                or r['special_teams_points']>8*r['special_teams_tds']):
            raise ValueError('Inconsistent return TD scoring attribution')
        if r['defensive_return_tds']:
            points=float(r['defensive_return_points'])/r['defensive_return_tds']
            if not 6<=points<=8:raise ValueError('Invalid defensive return points per TD')
            defensive_values += [round(points)]*int(r['defensive_return_tds'])
        if r['special_teams_tds']:
            points=float(r['special_teams_points'])/r['special_teams_tds']
            if not 6<=points<=8:raise ValueError('Invalid ST return points per TD')
            special_values += [round(points)]*int(r['special_teams_tds'])
    totals['games']=len(rows)
    return totals,defensive_values,special_values

def scoring_rates(source):
    if not isinstance(source,dict) or source.get('mode')!='turnover_linked_defensive_returns_and_separate_special_teams':
        raise ValueError('Invalid separated scoring evidence')
    home,h_def,h_st=_history(source['home'])
    away,a_def,a_st=_history(source['away'])
    def turn_rate(attack,defense):
        # Offense: past pick-six/fumble-return TDs allowed per turnover.
        # Defense: prior defensive return TDs per opponent turnover forced.
        attack_p=(attack['defensive_return_tds_allowed']+
                  TURNOVER_RETURN_TD_PRIOR*TURNOVER_PRIOR_EXPOSURES)/(attack['offense_turnovers']+TURNOVER_PRIOR_EXPOSURES)
        defense_p=(defense['defensive_return_tds']+
                   TURNOVER_RETURN_TD_PRIOR*TURNOVER_PRIOR_EXPOSURES)/(defense['defense_opportunities']+TURNOVER_PRIOR_EXPOSURES)
        return float(np.clip(.5*(attack_p+defense_p),.0,MAX_RETURN_RATE))
    def st_rate(info):
        return float(np.clip((info['special_teams_tds']+
             SPECIAL_TD_GAME_PRIOR*SPECIAL_PRIOR_GAMES)/(info['games']+SPECIAL_PRIOR_GAMES),
             0.,MAX_SPECIAL_GAME_RATE))
    return {'home_turnover_td_allowed_probability':turn_rate(home,away),
            'away_turnover_td_allowed_probability':turn_rate(away,home),
            'home_special_td_per_game':st_rate(home),
            'away_special_td_per_game':st_rate(away),
            'home_defense_return_points_sample':h_def,
            'away_defense_return_points_sample':a_def,
            'home_special_points_sample':h_st,
            'away_special_points_sample':a_st,
            'home_historical_games':int(home['games']),
            'away_historical_games':int(away['games']),
            'prior_rate_return_per_turnover':TURNOVER_RETURN_TD_PRIOR,
            'prior_rate_special_per_game':SPECIAL_TD_GAME_PRIOR,
            'ATS_calibrated':False}

def touchdown_points(rng,mask,observed):
    """Only model a TD when an actual event has been sampled."""
    result=np.zeros(mask.shape,dtype=np.int16)
    k=int(mask.sum())
    if not k:return result
    # An observed return supplies actual 6/7/8 values; otherwise the prior
    # NCAA XP model is explicit, not misrepresented as historical evidence.
    if observed:
        result[mask]=rng.choice(np.array(observed,dtype=np.int16),size=k)
    else:
        result[mask]=6+(rng.random(k)<.965).astype(np.int16)
    return result
