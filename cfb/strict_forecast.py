#!/usr/bin/env python3
"""Standalone CFB-only research forecast entrypoint; no NFL imports.

Expects explicitly supplied pregame CFB projection parameters (which must be
independently derived from observations). Does NOT turn missing raw statistics
into fabricated values. Emits no wager grades or claims of ATS calibration.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cfb.engine import simulate, CFBEngineError


def timestamp(x, field):
    if not isinstance(x, str):
        raise CFBEngineError(f'{field} requires ISO 8601 timestamp')
    try:
        date = datetime.fromisoformat(x.replace('Z','+00:00'))
    except ValueError as exc:
        raise CFBEngineError(f'Invalid {field}') from exc
    if date.tzinfo is None or date.utcoffset() is None:
        raise CFBEngineError(f'{field} must be timezone-aware')
    return date.astimezone(timezone.utc)


def forecast(game, runs=5000, seed=20261020):
    if game.get('league') != 'CFB':
        raise CFBEngineError('CFB forecast requires league=CFB')
    if (str(game.get('home_division','')).lower()=='fcs' and
            str(game.get('away_division','')).lower()=='fcs'):
        raise CFBEngineError('Exclude FCS-vs-FCS')
    kickoff=timestamp(game.get('kickoff_utc'), 'kickoff_utc')
    capture=timestamp(game.get('data_as_of_utc'), 'data_as_of_utc')
    if capture >= kickoff:
        raise CFBEngineError('Source observed at/after kickoff: cannot forecast pregame')
    for side in ('home','away'):
        if game.get(side,{}).get('projection_source') != 'VERIFIED_CFB_PRIOR_HISTORY':
            raise CFBEngineError(f'{side} projection provenance missing')
    market=game.get('market')
    spread=None
    total=None
    if market is not None:
        quote=timestamp(market.get('quoted_at_utc'),'market.quoted_at_utc')
        if quote >= kickoff or quote > capture:
            raise CFBEngineError('Market quote later than source capture or kickoff')
        spread=market.get('spread_home')
        total=market.get('total')
    result=simulate(game['home'],game['away'],spread_home=spread,total=total,
                    runs=runs,seed=seed)
    return {
        'game_id':str(game.get('id')),
        'league':'CFB',
        'model':'cfb.engine.v1',
        'status':'CFB_ONLY_RESEARCH_UNCALIBRATED',
        'data_as_of_utc':capture.isoformat(),
        'kickoff_utc':kickoff.isoformat(),
        'simulation':result,
        'grade':'UNRATED',
        'wager_decision':'NO_VALIDATED_ATS_EDGE',
        'note':'Input provenance label must be corroborated by saved source hashes; no independent live as-of verification by this wrapper.'
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--out',required=True)
    parser.add_argument('--runs',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=20261020)
    args=parser.parse_args()
    result=forecast(json.loads(Path(args.input).read_text()),args.runs,args.seed)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print('CFB-only research forecast:',result['status'])

if __name__=='__main__':main()
