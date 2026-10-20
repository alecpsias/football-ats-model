"""Grade *frozen* forward CFB forecasts, only with independently sourced results.

Compare CFB projected margins vs real home spreads, not winning-team picks.
No spread archive => no ATS record. This utility cannot independently attest
provider-page timestamps, therefore labels quote-origin verification as pending.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from cfb.market import load_quotes, _date, _odds_profit, CFBMarketError

RESULT_HEADERS=('game_id','home_points','away_points','final_observed_at_utc',
                'result_source_url','result_source_sha256')


def verified_frozen(folder):
    folder=Path(folder)
    index=json.loads((folder/'index.json').read_text())
    if index.get('status')!='CFB_PROSPECTIVE_RESEARCH_RECORDS_FROZEN':
        raise ValueError('Not a CFB prospective archive')
    out={}
    for item in index['predictions']:
        gid=str(item['game_id'])
        if gid in out or not gid.isdigit() or item['file']!=gid+'.json':
            raise ValueError('Duplicate or unsafe CFB archive game ID')
        raw=(folder/item['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('Frozen game JSON digest mismatch: '+gid)
        p=json.loads(raw)
        if p['game_id']!=gid or p['status']!=item['status']:
            raise ValueError('CFB frozen index disagrees with record')
        gen=_date(p['prediction_generated_at_utc'],'prediction generation')
        snap=_date(p['source_snapshot_at_utc'],'snapshot capture')
        kickoff=_date(p['kickoff_utc'],'kickoff')
        if not snap<=gen<kickoff:
            raise ValueError('Forecast record not captured before kickoff')
        if p['market_quote'] is not None or p['ats_prediction'] is not None:
            raise ValueError('Expected no invented market quote or ATS prediction')
        out[gid]=p
    return out


def load_results(file):
    with Path(file).open(newline='',encoding='utf8') as f:
        r=csv.DictReader(f)
        if r.fieldnames is None or not set(RESULT_HEADERS).issubset(r.fieldnames):
            raise ValueError('Invalid independent CFB result CSV headers')
        rows=list(r)
    out={}
    for row in rows:
        gid=row['game_id']
        if gid in out:raise ValueError('Duplicate result game id')
        for side in ('home_points','away_points'):
            raw=row[side]
            if not raw.isdigit() or not 0<=int(raw)<=150:
                raise ValueError('Invalid integer CFB final score')
        if not row['result_source_url'].startswith('https://'):
            raise ValueError('Final result must have independent HTTPS source')
        digest=row['result_source_sha256'].lower()
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Final source evidence hash required')
        out[gid]=row
    return out


def grade(archive, quotes_csv, results_csv):
    predictions=verified_frozen(archive)
    if not Path(quotes_csv).is_file() or not Path(results_csv).is_file():
        return {'status':'MISSING_FORWARDED_MARKET_QUOTES_OR_FINAL_RESULTS',
                'ats_validated':False,'games_graded':0,
                'limitations':['No invented lines or results']}
    quotes=load_quotes(quotes_csv)
    finals=load_results(results_csv)
    graded=[]
    for gid,p in sorted(predictions.items()):
        if p['status']!='FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE':continue
        if gid not in quotes or gid not in finals:continue
        q=quotes[gid];r=finals[gid]
        if q['home'].strip().casefold()!=p['home'].strip().casefold() or q['away'].strip().casefold()!=p['away'].strip().casefold():
            raise CFBMarketError('CFB matchup team mismatch '+gid)
        kickoff=_date(p['kickoff_utc'],'forecast kickoff')
        if q['kickoff_time']!=kickoff:raise CFBMarketError('Market kickoff mismatch '+gid)
        # A line first captured after the frozen margin projection is valid
        # for research if still PREKICKOFF. The ATS comparison is anchored
        # at quote capture, not retrospectively claimed for forecast time.
        observed=_date(q['captured_at_utc'],'market capture')
        prediction_time=_date(p['prediction_generated_at_utc'],'forecast generation')
        decision_time=max(prediction_time,observed)
        if decision_time>=kickoff:
            raise CFBMarketError('Quote and projection not jointly available before kickoff '+gid)
        if _date(r['final_observed_at_utc'],'final observed time')<=kickoff:
            raise ValueError('Claimed final result observed before game kickoff')
        pred=float(p['simulated_margin_home'])+float(q['spread_home'])
        actual=int(r['home_points'])-int(r['away_points'])+float(q['spread_home'])
        side=1 if pred>0 else -1 if pred<0 else 0
        outcome=1 if actual>0 else -1 if actual<0 else 0
        odds=q['home_odds_american'] if side>0 else q['away_odds_american']
        pnl=0 if side==0 or outcome==0 else (_odds_profit(odds) if side==outcome else -1)
        graded.append({'game_id':gid,'home':p['home'],'away':p['away'],
            'predicted_margin':p['simulated_margin_home'], 'spread_home':q['spread_home'],
            'ats_comparison_first_possible_at_utc':decision_time.isoformat(),
            'actual_margin':int(r['home_points'])-int(r['away_points']),
            'predicted_ats_margin':round(pred,3),'actual_ats_margin':round(actual,3),
            'decision':'home' if side>0 else 'away' if side<0 else 'no_edge',
            'correct':None if side==0 or outcome==0 else side==outcome,
            'pnl_per_1_unit':round(pnl,5),
            'forecast_sha256':next(x['sha256'] for x in json.loads((Path(archive)/'index.json').read_text())['predictions'] if x['game_id']==gid),
            'quote_evidence_sha256':q['evidence_sha256'],
            'final_evidence_sha256':r['result_source_sha256']})
    bets=[r for r in graded if r['correct'] is not None]
    return {'status':'PROSPECTIVE_ATS_RESEARCH_ONLY_ARCHIVE_PROVENANCE_NOT_INDEPENDENTLY_ATTESTED',
            'ats_validated':False,'games_graded':len(graded),'decisions':len(bets),
            'total_wins':sum(r['correct'] is True for r in bets),
            'total_losses':sum(r['correct'] is False for r in bets),
            'directional_ats_accuracy':round(sum(r['correct'] for r in bets)/len(bets),5) if bets else None,
            'flat_stake_roi':round(sum(r['pnl_per_1_unit'] for r in bets)/len(bets),5) if bets else None,
            'games':graded,
            'limitations':['Quote and final-result URL/hash are supplied claims; contents have not been independently verified',
                           'No statistical evidence of positive edge or calibrated cover probabilities',
                           'Archive retention must be durable before treating sample as long-run forward record']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True);p.add_argument('--quotes',required=True)
    p.add_argument('--results',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=grade(a.archive,a.quotes,a.results)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='games'},indent=2))
if __name__=='__main__':main()
