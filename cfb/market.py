"""Free-source CFB spread audit and pregame quote templates.

Never manufacture historical prices, never infer a timestamp from a postgame
Parquet field, and never use CFB winner accuracy as ATS cover accuracy.
Signed spread convention: home favourite -7.5 is spread_home=-7.5.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

FIELDS=('game_id','home','away','kickoff_utc','quoted_at_utc','captured_at_utc',
        'book','source_url','spread_home','home_odds_american','away_odds_american',
        'evidence_sha256')


class CFBMarketError(ValueError):
    pass


def _date(value,label):
    try: dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    except (ValueError,AttributeError) as exc:raise CFBMarketError('Invalid '+label) from exc
    if dt.tzinfo is None:raise CFBMarketError('Naive timestamp '+label)
    return dt.astimezone(timezone.utc)


def _american(value):
    try:
        raw=str(value).strip()
        if re.fullmatch(r'[+-]?[0-9]+',raw) is None:raise ValueError
        n=int(raw)
    except (TypeError,ValueError) as exc:
        raise CFBMarketError('American odds require signed integer') from exc
    if not (-10000 <= n <= -100 or 100 <= n <= 10000):
        raise CFBMarketError('Invalid American odds (zero/shorter than 100 disallowed)')
    return n


def _odds_profit(o):
    return 100/abs(o) if o<0 else o/100


def validate_quote(r):
    missing=[k for k in FIELDS if not str(r.get(k,'')).strip()]
    if missing:raise CFBMarketError('Quote missing fields: '+', '.join(missing))
    url=urlparse(r['source_url'])
    if url.scheme!='https' or not url.netloc:
        raise CFBMarketError('Source must have https URL')
    digest=r['evidence_sha256'].lower()
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise CFBMarketError('evidence_sha256 must be 64 hex characters')
    kickoff=_date(r['kickoff_utc'],'kickoff_utc')
    quoted=_date(r['quoted_at_utc'],'quoted_at_utc')
    captured=_date(r['captured_at_utc'],'captured_at_utc')
    if not (quoted <= captured < kickoff):
        raise CFBMarketError('Quote must be observed and recorded BEFORE kickoff')
    try:spread=float(r['spread_home'])
    except ValueError as exc:raise CFBMarketError('Invalid home spread') from exc
    if not math.isfinite(spread) or not -75<=spread<=75:
        raise CFBMarketError('Home spread implausible or nonfinite')
    if abs(2*spread-round(2*spread))>1e-8:
        raise CFBMarketError('Spread must be in half-point increments')
    home_odds=_american(r['home_odds_american'])
    away_odds=_american(r['away_odds_american'])
    return {**r,'spread_home':spread,'home_odds_american':home_odds,
            'away_odds_american':away_odds,'kickoff_time':kickoff}


def load_quotes(path):
    with Path(path).open(newline='',encoding='utf8') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None or any(k not in reader.fieldnames for k in FIELDS):
            raise CFBMarketError('Invalid quote CSV headers')
        quotes=[validate_quote(row) for row in reader]
    ids=[x['game_id'] for x in quotes]
    if len(ids)!=len(set(ids)):
        raise CFBMarketError('Duplicate game quotes; select one timestamped market BEFORE grading')
    return {x['game_id']:x for x in quotes}


def _norm(s):
    return ' '.join(str(s).casefold().split())


def make_template(index_path,output):
    index=json.loads(Path(index_path).read_text())
    games=index.get('upcoming_games',[])
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader()
        for game in games:
            info={'game_id':game['game_id'],'home':game['home'],
                  'away':game['away']}
            # Kickoff is factual and already present in this snapshot's game
            # evidence. All actual market information must still be supplied.
            evidence=Path(index_path).parent/(str(game['game_id'])+'.json')
            if evidence.is_file():
                info['kickoff_utc']=json.loads(evidence.read_text()).get('kickoff_utc','')
            w.writerow(info)
    return len(games)


def grade_replay(replay_path,calibration_path,quotes_path,output):
    """Grade only held-out margin directions vs pregame-attested captured lines.

    Evidence hashes/timestamps are claimed by the quote supplier: no online
    commit-validation performed here. Historical as-of model inputs remain
    unproven, so DO NOT label this as verified out-of-sample ATS validation.
    """
    rows=json.loads(Path(replay_path).read_text())
    cal=json.loads(Path(calibration_path).read_text())
    base={'status':'NO_USER_ATTESTED_PREKICKOFF_SPREADS','ats_validated':False,
          'limitations':['No independently verified market archive in this workflow',
                         'Historical replay features came from later-captured files']}
    if not Path(quotes_path).exists():
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(base,indent=2)+'\n')
        return base
    quotes=load_quotes(quotes_path)
    if cal.get('status')!='CFB_MARGIN_RESEARCH_VALIDATION_ONLY_NOT_ATS_VALIDATED':
        base['status']='CFB_MARGIN_CALIBRATION_UNAVAILABLE'
    else:
        cutoff=_date(cal['holdout_first_kickoff'],'holdout_first_kickoff')
        holdout=[r for r in rows if _date(r['kickoff'],'kickoff')>=cutoff]
        graded=[]; unmatched=[]
        for r in holdout:
            q=quotes.get(str(r['game_id']))
            if q is None:
                unmatched.append(str(r['game_id']));continue
            if _norm(q['home'])!=_norm(r['home']) or _norm(q['away'])!=_norm(r['away']):
                raise CFBMarketError('Team identity mismatch for '+str(r['game_id']))
            if q['kickoff_time']!=_date(r['kickoff'],'historical kickoff'):
                raise CFBMarketError('Kickoff timestamp mismatch for '+str(r['game_id']))
            actual=float(r['actual_margin'])+q['spread_home']
            pred=float(r['baseline_margin'])+q['spread_home']
            direction=1 if pred>0 else -1 if pred<0 else 0
            outcome=1 if actual>0 else -1 if actual<0 else 0
            stake_odds=q['home_odds_american'] if direction==1 else q['away_odds_american']
            profit=(0 if direction==0 or outcome==0 else
                    _odds_profit(stake_odds) if direction==outcome else -1.)
            graded.append({'game_id':str(r['game_id']),'actual_ats_margin':actual,
                           'predicted_ats_margin':round(pred,4),
                           'predicted_side':'home' if direction==1 else 'away' if direction==-1 else 'no_edge',
                           'outcome':'push' if outcome==0 else 'home' if outcome==1 else 'away',
                           'profit_per_unit_flat_stake_research_only':round(profit,5),
                           'correct':None if outcome==0 or direction==0 else direction==outcome})
        decisions=[r for r in graded if r['correct'] is not None]
        base={'status':'UNVERIFIED_ARCHIVE_ATS_RESEARCH_ONLY',
              'ats_validated':False,
              'quoted_holdout_games':len(graded),'gradable_games':len(decisions),
              'unmatched_holdout_games':len(unmatched),'coverage_games':len(holdout),
              'ats_directional_accuracy':round(sum(r['correct'] for r in decisions)/len(decisions),5) if decisions else None,
              'flat_stake_roi_research_only':round(sum(r['profit_per_unit_flat_stake_research_only'] for r in graded)/len(decisions),5) if decisions else None,
              'games':graded,
              'limitations':['User-attested quote timestamps and hashes were not independently verified against archival commits',
                             'After-the-fact source PBP is not a verified historical pregame snapshot',
                             'No calibrated cover probabilities; direction-only wagers are not a strategy',
                             'Do not claim proven ATS edge or profitability']}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(base,indent=2)+'\n')
    return base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('template');a.add_argument('--index',required=True);a.add_argument('--out',required=True)
    b=sub.add_parser('grade');b.add_argument('--replay',required=True);b.add_argument('--calibration',required=True)
    b.add_argument('--quotes',required=True);b.add_argument('--out',required=True)
    x=p.parse_args()
    if x.cmd=='template':print('Quote template games:',make_template(x.index,x.out))
    else:print(json.dumps(grade_replay(x.replay,x.calibration,x.quotes,x.out),indent=2))

if __name__=='__main__':main()
