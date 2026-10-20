"""CFB-only opponent-adjusted *research challenger* for game-score distributions.

This module CONNECTS previously descriptive opponent-adjusted pass/rush EPA,
explosive plays, third-down, red-zone finishing, hurry proxy and field position
signals to points per drive. All contributions are experimental *hypotheses*,
not measured causal effects or fitted coefficients. Never silently promote to
the primary forecast. Required data are only games completed before cutoff.

Missing matchup measurements are recorded as missing, never imputed as 0.
For historical validation, the unchanged baseline is the control on EVERY
fixture. Candidate scores are paired with that same seed and data cutoff.
"""
from __future__ import annotations
import math
from copy import deepcopy
from types import MappingProxyType

from cfb.engine import simulate

VERSION = 'cfb-advanced-scoring-research-v1'
# Units: point-margin hypothesis per unit of each *home-minus-away* signal.
# No coefficients have been optimized on outcomes, and QB hurry != charted pressure.
COEFFICIENTS = MappingProxyType({
    'sos_pass_epa_differential': 5.0,
    'sos_rush_epa_differential': 3.0,
    'pass_explosive': 8.0,
    'rush_explosive': 5.0,
    'third_down': 6.0,
    'qb_hurries_allowed': -2.0,
    'redzone_td_rate': 3.0,
    'field_position': 0.10,
})
LIMITS = MappingProxyType({
    'sos_pass_epa_differential': .7,
    'sos_rush_epa_differential': .7,
    'pass_explosive': .25,
    'rush_explosive': .25,
    'third_down': .35,
    'qb_hurries_allowed': .40,
    'redzone_td_rate': .5,
    'field_position': 15.0,
})
MAX_MARGIN_SHIFT = 6.0


def _valid(value, name):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError('Invalid advanced signal '+name)
    try:
        v = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid advanced signal '+name) from exc
    if not math.isfinite(v):
        raise ValueError('Nonfinite advanced signal '+name)
    return v


def build_adjustment(signals, home_games, away_games, *, mode='all'):
    """Convert independent matchup signal differences into bounded margin hypothesis.

    An available opponent-adjusted pass/rush EPA signal is mandatory. Other
    context remains optional and missingness is disclosed. HFA and baseline PPD
    must be supplied separately. No game results may be supplied here.
    """
    if mode not in ('all','sos_only','context_only'):
        raise ValueError('Unsupported challenger ablation')
    if type(home_games) is not int or type(away_games) is not int or min(home_games,away_games)<1:
        raise ValueError('Positive completed-game counts required')
    if not isinstance(signals,dict) or not isinstance(signals.get('observed_context'),dict):
        raise ValueError('Matchup evidence missing')
    values={k:(signals.get(k) if k.startswith('sos_') else signals['observed_context'].get(k))
            for k in COEFFICIENTS}
    if all(values[k] is None for k in ('sos_pass_epa_differential','sos_rush_epa_differential')):
        return {'status':'UNAVAILABLE_NO_VERIFIED_OPPONENT_EPA','version':VERSION,
                'margin_shift_points':None,'missing_signals':list(COEFFICIENTS),
                'coefficients_not_validated':True,'mode':mode}
    n=min(home_games,away_games)
    shrink=n/(n+6.) # explicit early-season reliability shrink; hypothesis only.
    raw=0.
    contribution={}
    missing=[]
    for key,coefficient in COEFFICIENTS.items():
        v=_valid(values[key],key)
        if v is None:
            missing.append(key)
            continue
        if mode=='sos_only' and not key.startswith('sos_'):continue
        if mode=='context_only' and key.startswith('sos_'):continue
        bounded=max(-LIMITS[key],min(LIMITS[key],v))
        val=coefficient*bounded
        raw+=val
        contribution[key]={'observed':round(v,7),'bounded':round(bounded,7),
                           'coefficient_hypothesis':coefficient,
                           'raw_margin_points':round(val,5)}
    capped=max(-MAX_MARGIN_SHIFT,min(MAX_MARGIN_SHIFT,raw))
    shift=round(capped*shrink,6)
    return {'status':'AVAILABLE_UNCALIBRATED_RESEARCH_ONLY','version':VERSION,
            'mode':mode,'margin_shift_points':shift,
            'uncertainty_shrink':round(shrink,6),'effective_prior_games':n,
            'raw_shift_before_shrink':round(raw,6),
            'raw_shift_after_cap':round(capped,6),
            'contributions':contribution,'missing_signals':missing,
            'coefficients_not_validated':True,
            'not_production':True,
            'limitations':['No empirical approval for these feature coefficients',
                           'QB hurry is a limited proxy, NOT true charted pressure',
                           'Signals describe relative matchup advantages, not measured causal effects',
                           'No verified lineup or injuries are incorporated']}


def simulate_challenger(home_projection, away_projection, signals, home_games,
                        away_games, *, runs=3000, seed=1, mode='all'):
    """Re-simulate both teams from baseline inputs with bounded margin adjustment.

    Predictive input set is independent of results/market; no ATS variables.
    Preserves the original baseline projections, independent turnovers, drives,
    and scoring rules. The main baseline forecast must be recorded separately.
    """
    a=build_adjustment(signals,home_games,away_games,mode=mode)
    if a['margin_shift_points'] is None:
        return {**a,'simulation':None,'home_projection':None,'away_projection':None}
    hp=deepcopy(home_projection);ap=deepcopy(away_projection)
    for side,proj,sign in (('home',hp,1),('away',ap,-1)):
        if not isinstance(proj,dict) or 'expected_drives' not in proj or '_ppd' not in proj:
            raise ValueError('Missing '+side+' baseline projection')
        drives=float(proj['expected_drives'])
        if not math.isfinite(drives) or drives<=0:
            raise ValueError('Invalid '+side+' drives')
        new=float(proj['_ppd'])+sign*a['margin_shift_points']/(2.*drives)
        if not math.isfinite(new) or not .01<=new<=6.5:
            raise ValueError('Advanced scenario exceeds physical scoring bounds')
        proj['_ppd']=new
    r=simulate(hp,ap,runs=runs,seed=seed)
    return {**a,'simulation':r,'home_projection':hp,'away_projection':ap,
            'simulated_margin_home':r['mean_margin_home'],
            'simulated_score_home':r['mean_score_home'],
            'simulated_score_away':r['mean_score_away']}
