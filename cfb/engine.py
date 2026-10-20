#!/usr/bin/env python3
"""Independent college-football possession simulator (research; uncalibrated).

NO code, constants or imports from the NFL model. All numeric priors are
hypotheses, not backtested/fitted results. NCAA extra periods are implemented
as two series at the 25 (second OT mandates 2-point tries), then paired
alternating 2-point plays in the 3rd+ periods. Individual drive plays and
situational coaching tactics are simplified, not an NCAA rules emulator.

Points per drive incorporates field position via possession-to-possession
turnover effect. It does NOT assume a betting line or output a wager grade.
"""
from __future__ import annotations

import math
from types import MappingProxyType
import numpy as np

CFB_RULES = MappingProxyType({
    'league_ppd': 2.35,
    'league_to_drive': 0.13,
    'td_share_scoring_drives': 0.64,
    'home_advantage_points': 2.4,
    'shortfield_bonus_ppd': 0.80,
    'drive_noise_sd': 0.20,
    'regulation_drives_min': 6,
    'regulation_drives_max': 19,
    'ot_25_ppd_multiplier': 1.30,
    'two_point_conversion_reference': 0.44,
    'two_point_ppd_adjustment': 0.035,
    'max_overtime_periods': 30,
})

class CFBEngineError(ValueError):
    pass


def _rate(x, label, low, high):
    if isinstance(x, (bool, str)):
        raise CFBEngineError(f'{label} must be numeric')
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise CFBEngineError(f'{label} must be numeric') from exc
    if not math.isfinite(v) or not low <= v <= high:
        raise CFBEngineError(f'{label} outside [{low}, {high}] or nonfinite')
    return v


def validate_projection(p, label):
    if not isinstance(p, dict):
        raise CFBEngineError(f'{label} must be dict')
    return {
        '_ppd': _rate(p.get('_ppd'), label+'._ppd', .01, 6.5),
        '_to': _rate(p.get('_to'), label+'._to', 0, .48),
        '_td': _rate(p.get('_td'), label+'._td', .05, .95),
        'expected_drives': _rate(p.get('expected_drives'), label+'.expected_drives', 4, 22),
    }


def _rules(overrides):
    r = dict(CFB_RULES)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise CFBEngineError('rules must be a dict')
        extra = set(overrides) - set(r)
        if extra:
            raise CFBEngineError(f'Unknown CFB rules keys: {sorted(extra)}')
        r.update(overrides)
    for key in ('league_ppd','shortfield_bonus_ppd','drive_noise_sd',
                'ot_25_ppd_multiplier','home_advantage_points'):
        _rate(r[key], key, 0, 8)
    for key in ('league_to_drive','td_share_scoring_drives',
                'two_point_conversion_reference','two_point_ppd_adjustment'):
        _rate(r[key], key, 0, 1)
    if (not isinstance(r['max_overtime_periods'], int)
            or not 3 <= r['max_overtime_periods'] <= 60):
        raise CFBEngineError('Invalid maximum overtime periods')
    if (not isinstance(r['regulation_drives_min'], int)
            or not isinstance(r['regulation_drives_max'], int)
            or not 2 <= r['regulation_drives_min'] <= r['regulation_drives_max'] <= 25):
        raise CFBEngineError('Invalid drive-count bounds')
    return r


def _draw_possession(p, r, rng, active, shortfield, noise, multiplier=1.):
    """Single possession: TD, FG, turnover, or empty. Mean calibrated to PPD
    except under probability caps and possession-order effects. Return points/TO.
    """
    n = len(active)
    pts = (p['_ppd'] * multiplier + noise +
           r['shortfield_bonus_ppd'] * (shortfield.astype(float) - r['league_to_drive']))
    pts = np.clip(pts, 0.05, 6.25)
    tdshare = p['_td']
    avg_scoring_pts = tdshare * 6.98 + (1 - tdshare) * 3.
    pto = p['_to']
    pscore = np.clip(pts / avg_scoring_pts, 0, 1 - pto)
    draw = rng.random(n)
    turnover = active & (draw < pto)
    scoring = active & (draw >= pto) & (draw < pto + pscore)
    td = scoring & (rng.random(n) < tdshare)
    # CFB TD conversion treatment: kick usually scores one; rare failed kick
    # or coach-chosen two-point play. Exact selection behavior not observed.
    conversion = np.where(rng.random(n) < .965, 1, 0)
    score = np.where(td, 6 + conversion, np.where(scoring, 3, 0))
    return score.astype(np.int16), turnover


def _two_point_rate(p, r):
    """Provisional two-point success proxy; NOT empirically calibrated."""
    return float(np.clip(r['two_point_conversion_reference'] +
                         r['two_point_ppd_adjustment'] *
                         (p['_ppd'] - r['league_ppd']), .18, .72))


def _resolve_overtime(home, away, rules, rng, hs, ass):
    """Apply NCAA extra-period structure to tied-at-regulation draws.

    For OT 1 and 2, model paired series starting on opponent 25; OT 2
    TD follows mandatory 2-point try. OT 3+ paired two-point plays.
    Defensive scoring/penalty/coaching sequences are not fully modeled.
    """
    n = len(hs)
    entered = int(np.count_nonzero(hs == ass))
    period_reached = [0, 0, 0]
    for period in range(1, rules['max_overtime_periods'] + 1):
        tied = hs == ass
        if not np.any(tied):
            break
        if period <= 2:
            period_reached[period-1] = int(np.count_nonzero(tied))
            zeros = np.zeros(n, dtype=bool)
            noise = np.zeros(n, dtype=float)
            hp, _ = _draw_possession(home, rules, rng, tied, zeros, noise,
                                     rules['ot_25_ppd_multiplier'])
            ap, _ = _draw_possession(away, rules, rng, tied, zeros, noise,
                                     rules['ot_25_ppd_multiplier'])
            if period == 2:
                for p, points in ((home, hp), (away, ap)):
                    td = tied & (points >= 6)
                    two = rng.random(n) < _two_point_rate(p, rules)
                    points[td] = 6 + 2 * two[td].astype(np.int16)
            hs += hp
            ass += ap
        else:
            if period == 3:
                period_reached[2] = int(np.count_nonzero(tied))
            hp = tied & (rng.random(n) < _two_point_rate(home, rules))
            ap = tied & (rng.random(n) < _two_point_rate(away, rules))
            hs += 2 * hp.astype(np.int16)
            ass += 2 * ap.astype(np.int16)
    remaining = int(np.count_nonzero(hs == ass))
    return {'regulation_ties': entered, 'ot1': period_reached[0],
            'ot2': period_reached[1], 'ot3_or_later': period_reached[2],
            'unresolved_after_limit': remaining}


def simulate(home, away, rules=None, spread_home=None, total=None,
             runs=1500, seed=1, allow_overtime=True):
    """Independent CFB Monte Carlo scorer; no NFL module or market required.

    Implied scores are uncalibrated and should never be reported as ATS edges
    until separately validated against timestamped CFB sportsbook lines.
    """
    if type(runs) is not int or not 100 <= runs <= 300000:
        raise CFBEngineError('runs must be integer in 100..300000')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise CFBEngineError('seed must be 32-bit unsigned integer')
    h, a, r = validate_projection(home, 'home'), validate_projection(away, 'away'), _rules(rules)
    if spread_home is not None:
        spread_home = _rate(spread_home, 'spread_home', -100, 100)
    if total is not None:
        total = _rate(total, 'total', 0, 200)
    rng = np.random.default_rng(seed)
    tempo = np.clip(.5*(h['expected_drives']+a['expected_drives']),
                    r['regulation_drives_min'], r['regulation_drives_max'])
    drives = np.clip(np.rint(rng.normal(tempo, .9, runs)).astype(int),
                     r['regulation_drives_min'], r['regulation_drives_max'])
    hs, ass = np.zeros(runs, dtype=np.int32), np.zeros(runs, dtype=np.int32)
    short_h, short_a = np.zeros(runs, dtype=bool), np.zeros(runs, dtype=bool)
    noise = rng.normal(0, r['drive_noise_sd'], runs)
    # Each team's possession has the same count; turnovers influence the
    # opponent's NEXT possession, not a fictitious immediate extra drive.
    for i in range(r['regulation_drives_max']):
        active = i < drives
        ph, toh = _draw_possession(h, r, rng, active, short_h, noise)
        pa, toa = _draw_possession(a, r, rng, active, short_a, noise)
        hs += ph
        ass += pa
        short_h, short_a = toa, toh
    reg_ties = int(np.count_nonzero(hs == ass))
    if allow_overtime:
        overtime = _resolve_overtime(h, a, r, rng, hs, ass)
    else:
        overtime = {'regulation_ties':reg_ties,'ot1':0,'ot2':0,
                    'ot3_or_later':0,'unresolved_after_limit':reg_ties}
    margin = hs.astype(float) - ass
    line = margin + spread_home if spread_home is not None else None
    score = hs.astype(float) + ass
    return {
        'mean_score_home':round(float(hs.mean()),3),
        'mean_score_away':round(float(ass.mean()),3),
        'mean_margin_home':round(float(margin.mean()),3),
        'median_margin_home':float(np.median(margin)),
        'margin_standard_deviation':round(float(margin.std()),3),
        'margin_p10_p50_p90':[float(x) for x in np.quantile(margin,[.1,.5,.9])],
        'home_win_probability_raw':round(float(np.mean(margin>0)),5),
        'home_cover_probability_raw':round(float(np.mean(line>0)),5) if line is not None else None,
        'away_cover_probability_raw':round(float(np.mean(line<0)),5) if line is not None else None,
        'push_probability_raw':round(float(np.mean(line==0)),5) if line is not None else None,
        'mean_total':round(float(score.mean()),3),
        'over_probability_raw':round(float(np.mean(score>total)),5) if total is not None else None,
        'under_probability_raw':round(float(np.mean(score<total)),5) if total is not None else None,
        'overtime_summary':overtime,
        'simulations':runs, 'seed':seed,
        'engine':'cfb.engine.v1',
        'status':'CFB_RESEARCH_UNCALIBRATED_NO_WAGER_GRADE',
        'limitations':['Possessions and short fields approximated',
                       'Uncalibrated priors/TD distribution',
                       'NCAA overtime follows correct round structure but approximates drive outcomes',
                       'No situational coaching clock or play-by-play details',
                       'No ATS calibration or verified pregame market quote']
    }
