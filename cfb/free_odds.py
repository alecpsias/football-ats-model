"""Capture *observed* free ESPN CFB odds, without backdating a source quote.

This is an unofficial/public ESPN core endpoint; it may change or omit markets.
No API key, paid feed, bookmaker account, purchase or subscription required.
A successful run with zero valid quotes is NOT an ATS result.

Important: observed_at_utc is when OUR job received the JSON. ESPN does not
promise that the listed book's price was originally published at that time.
Never represent it as an independently verified bookmaker timestamp.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import re
import urllib.request
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse
from cfb.archive import ZIP_TIME, _zipwrite, ID_RE, load_all_archives, digest
from cfb.market import _date, _american

CORE='https://sports.core.api.espn.com/v2/sports/football/leagues/college-football'
PREFERRED=('DraftKings','FanDuel','BetMGM','Caesars','ESPN BET')
# The real October 22 2026 ESPN CFB core odds endpoint reports provider
# 100 as 'Draft Kings' (with a space), not 'DraftKings'. Previously all
# such genuine complete markets were silently rejected. Restrict aliases to
# observed exact spellings, keep original unmodified source bytes in archive.
PROVIDER_ALIASES={'Draft Kings':'DraftKings'}
MAX_RESPONSE_BYTES=2_000_000


def endpoint(gid):
    if not ID_RE.fullmatch(str(gid)):
        raise ValueError('Unsafe game ID')
    return f'{CORE}/events/{gid}/competitions/{gid}/odds?limit=50'


def fetch(url, timeout=12):
    """Network helper enforces ESPN host and known CFB endpoint for safety."""
    parsed=urlparse(url)
    if (parsed.scheme!='https' or parsed.netloc!='sports.core.api.espn.com' or
            not parsed.path.startswith('/v2/sports/football/leagues/college-football/events/')):
        raise ValueError('Unexpected external odds response URL')
    with urllib.request.urlopen(urllib.request.Request(url, headers={
            'User-Agent':'cfb-research-free-snapshot/1.0',
            'Accept':'application/json'}),timeout=timeout) as r:
        body=r.read(MAX_RESPONSE_BYTES+1)
    if len(body)>MAX_RESPONSE_BYTES:
        raise ValueError('Oversized public odds response')
    return body


def _espn_american(value):
    """Accept ESPN's exact integral JSON floats (e.g. -110.0).

    Do not loosen the shared American-odds validator: reject fractional,
    Boolean, NaN, infinite, suspicious/out-of-range values. Strings must
    already satisfy the standard signed-integer parser.
    """
    if type(value) is float:
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError('ESPN float spread price must be an exact integer')
        return _american(int(value))
    return _american(value)


def _odds_item(item):
    """Require signed home-side spread AND BOTH actual spread prices."""
    if not isinstance(item,dict):return None
    p=item.get('provider',{})
    if not isinstance(p,dict) or not isinstance(p.get('name'),str):return None
    observed_book=p['name'].strip()
    book=PROVIDER_ALIASES.get(observed_book,observed_book)
    if book not in PREFERRED:return None
    try:
        spread=float(item['spread'])
        home=item['homeTeamOdds'];away=item['awayTeamOdds']
        home_odds=_espn_american(home['spreadOdds']);away_odds=_espn_american(away['spreadOdds'])
        if not math.isfinite(spread) or abs(spread)>75 or abs(spread*2-round(spread*2))>1e-8:
            return None
        h_fav=home.get('favorite');a_fav=away.get('favorite')
        # If favourite flags are known, confirm the signed spread convention.
        if not isinstance(h_fav,bool) or not isinstance(a_fav,bool):return None
        if spread<0 and not (h_fav and not a_fav):return None
        if spread>0 and not (a_fav and not h_fav):return None
        if spread==0 and h_fav==a_fav==True:return None
        # ESPN returns 'spread' as the signed HOME-team line in the
        # observed CFB core collection. When both current home/away team
        # point-spreads are present, cross-validate rather than assuming.
        h_current=home.get('current',{});a_current=away.get('current',{})
        if isinstance(h_current,dict) and isinstance(a_current,dict):
            hp=h_current.get('pointSpread');ap=a_current.get('pointSpread')
            if hp is not None or ap is not None:
                if not isinstance(hp,dict) or not isinstance(ap,dict):return None
                hline=float(hp['alternateDisplayValue'])
                aline=float(ap['alternateDisplayValue'])
                if not math.isfinite(hline) or not math.isfinite(aline):return None
                if abs(hline-spread)>1e-8 or abs(hline+aline)>1e-8:return None
    except (KeyError,TypeError,ValueError,OverflowError):
        return None
    return {'book':book,'spread_home':spread,'home_odds_american':home_odds,
            'away_odds_american':away_odds,'provider_id':str(p.get('id',''))}


def choose_espn_quote(body, fetcher=fetch, expected_gid=None):
    """Follow embedded/linked provider records, pick fixed priority irrespective of scores."""
    data=json.loads(body)
    if not isinstance(data,dict) or not isinstance(data.get('items'),list):
        raise ValueError('Missing ESPN odds item collection')
    matched=[]
    for i,item in enumerate(data['items'][:40]):
        provider_body=None
        src=None
        # The ESPN collection often includes COMPLETE odds inline *alongside*
        # a legacy http:// $ref. Read and validate the inline record FIRST.
        # Never discard valid source evidence because its optional link is HTTP.
        q=_odds_item(item)
        if q is None and isinstance(item,dict) and '$ref' in item:
            ref=item['$ref']
            if not isinstance(ref,str):
                continue
            # Follow same-origin references over HTTPS only. ESPN embeds
            # http:// links in otherwise HTTPS JSON; do not make HTTP calls.
            if ref.startswith('http://sports.core.api.espn.com/'):
                ref='https://'+ref[len('http://'):]
            prefix=(f'{CORE}/events/{expected_gid}/competitions/{expected_gid}/odds/'
                    if expected_gid is not None else CORE+'/events/')
            if not ref.startswith(prefix):
                continue
            try:
                provider_body=fetcher(ref)
                q=_odds_item(json.loads(provider_body))
                src=ref
            except (OSError,ValueError,TypeError,TimeoutError):
                continue
        if q:
            matched.append((PREFERRED.index(q['book']),i,q,provider_body,src))
    if not matched:return None,None,None
    selected=min(matched,key=lambda x:(x[0],x[1]))
    return selected[2],selected[3],selected[4]


def verify_odds_archive(file):
    file=Path(file)
    if not ID_RE.fullmatch(file.stem) or file.suffix!='.zip':
        raise ValueError('Odds archive must be run_id.zip')
    with zipfile.ZipFile(file) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or not names or names[0]!='index.json':
            raise ValueError('Bad odds archive paths')
        index=json.loads(z.read('index.json'))
        if index.get('schema')!='cfb_espn_observed_odds_archive_v1' or str(index.get('run_id'))!=file.stem:
            raise ValueError('Bad odds index')
        if len(index.get('records',[]))>1000:raise ValueError('Oversize odds index')
        expected={'index.json'}
        records=[]
        for record in index['records']:
            gid=str(record['game_id'])
            if not ID_RE.fullmatch(gid):raise ValueError('Unsafe odds game ID')
            if record['record_file']!=gid+'.json':raise ValueError('Invalid record path')
            expected.add(record['record_file'])
            raw=z.read(record['record_file'])
            if digest(raw)!=record['record_sha256']:
                raise ValueError('Odds record digest mismatch '+gid)
            row=json.loads(raw)
            if row['game_id']!=gid or row['status']!=record['status']:
                raise ValueError('Odds record mismatch')
            if row['status']=='OBSERVED_COMPLETE_SPREAD_AND_BOTH_PRICES':
                q=row['quote']
                valid=_odds_item({'provider':{'name':q['book']},'spread':q['spread_home'],
                     'homeTeamOdds':{'favorite':q['spread_home']<0,'spreadOdds':q['home_odds_american']},
                     'awayTeamOdds':{'favorite':q['spread_home']>0,'spreadOdds':q['away_odds_american']}})
                if valid is None:raise ValueError('Invalid archived spread quote')
                if not _date(row['observed_at_utc'],'captured')<_date(row['kickoff_utc'],'kickoff'):
                    raise ValueError('Archived odds observed after kickoff')
                if row['source_url']!=endpoint(gid) or not row.get('raw_source_file'):
                    raise ValueError('No original ESPN source URL/evidence for quote')
                # Verify quoted values are in the saved ORIGINAL source bytes,
                # not just in editable metadata or a claimed evidence hash.
                if row.get('raw_provider_file'):
                    confirmed=_odds_item(json.loads(z.read(row['raw_provider_file'])))
                else:
                    root=json.loads(z.read(row['raw_source_file']))
                    items=root.get('items',[]) if isinstance(root,dict) else []
                    confirmed=(q if any(_odds_item(item)==q for item in items)
                               else None)
                if confirmed!=q:
                    raise ValueError('Archived odds metadata disagrees with raw ESPN evidence')
                if row.get('provider_ref_url') and not row['provider_ref_url'].startswith(
                        f'{CORE}/events/{gid}/competitions/{gid}/odds/'):
                    raise ValueError('Provider response references wrong CFB event')
            if row.get('raw_source_file'):
                f=row['raw_source_file']
                if f!='raw/'+gid+'.json':raise ValueError('Unsafe raw source path')
                expected.add(f)
                if digest(z.read(f))!=row['raw_source_sha256']:
                    raise ValueError('Bad raw response evidence hash')
            if row.get('raw_provider_file'):
                f=row['raw_provider_file']
                if f!='provider/'+gid+'.json':raise ValueError('Unsafe provider source path')
                expected.add(f)
                if digest(z.read(f))!=row['raw_provider_sha256']:
                    raise ValueError('Bad selected provider evidence hash')
            records.append(row)
        if expected!=set(names):raise ValueError('Unexpected odds archive members')
    return index,records


def capture(forecast_root, archive_root, run_id, now=None, days=7, fetcher=fetch):
    if not ID_RE.fullmatch(str(run_id)):raise ValueError('Invalid odds capture run id')
    if not 1<=days<=31:raise ValueError('Invalid capture window')
    injected_clock=(now is not None)
    now=_date(now.isoformat() if isinstance(now,datetime) else
              (now if now is not None else datetime.now(timezone.utc).isoformat()),'capture clock')
    forecasts=load_all_archives(forecast_root)
    if not forecasts:raise ValueError('No frozen CFB forecasts to capture')
    selected=[(gid,r) for gid,r in forecasts.items()
              if now<_date(r['kickoff_utc'],'kickoff')<=now+timedelta(days=days)]
    selected.sort(key=lambda x:(x[1]['kickoff_utc'],x[0]))
    target=Path(archive_root)/(str(run_id)+'.zip')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Cannot overwrite odds capture '+str(run_id))
    contents={}
    records=[];summary=[]
    for gid,p in selected:
        ep=endpoint(gid)
        body=None;provider_raw=None;ref=None;q=None;error=None
        try:
            body=fetcher(ep)
            q,provider_raw,ref=choose_espn_quote(body,fetcher,expected_gid=gid)
        except (ValueError,TypeError,KeyError,OSError,TimeoutError) as exc:
            error=type(exc).__name__+': '+str(exc)[:200]
        # If the call straddles kickoff, omit the quote instead of backdating it.
        observed=(now if injected_clock else datetime.now(timezone.utc))
        if observed>=_date(p['kickoff_utc'],'kickoff'):
            q=None;error='Response not verified before kickoff'
        if body is not None:
            raw='raw/'+gid+'.json';contents[raw]=body
        else:raw=None
        if provider_raw is not None:
            provider_file='provider/'+gid+'.json';contents[provider_file]=provider_raw
        else:provider_file=None
        status=('OBSERVED_COMPLETE_SPREAD_AND_BOTH_PRICES' if q else
                'FETCH_ERROR' if error else 'NO_COMPLETE_VERIFIABLE_SPREAD_ODDS')
        rec={'game_id':gid,'status':status,'home':p['home'],'away':p['away'],
            'kickoff_utc':p['kickoff_utc'],'observed_at_utc':observed.isoformat(),
            'source_url':ep,'quote':q,'book_timestamp_verified':False,
            'observation_only':'Price appeared in ESPN public response; book publish time unknown',
            'raw_source_file':raw,'raw_source_sha256':digest(body) if body is not None else None,
            'raw_provider_file':provider_file,'raw_provider_sha256':digest(provider_raw) if provider_raw is not None else None,
            'provider_ref_url':ref,'error':error}
        b=(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n').encode()
        contents[gid+'.json']=b
        records.append({'game_id':gid,'record_file':gid+'.json','record_sha256':digest(b),'status':status})
        summary.append(status)
    index={'schema':'cfb_espn_observed_odds_archive_v1','run_id':str(run_id),
           'captured_at_utc':now.isoformat(),'source':'ESPN public core CFB odds endpoint; unofficial and not guaranteed',
           'selected_provider_priority':list(PREFERRED),'candidates':len(selected),
           'complete_spread_prices':summary.count('OBSERVED_COMPLETE_SPREAD_AND_BOTH_PRICES'),
           'missing_or_incomplete_prices':summary.count('NO_COMPLETE_VERIFIABLE_SPREAD_ODDS'),
           'fetch_errors':summary.count('FETCH_ERROR'),'records':records,
           'no_verified_original_book_quote_times':True}
    contents={'index.json':(json.dumps(index,sort_keys=True,separators=(',',':'))+'\n').encode(),**contents}
    with zipfile.ZipFile(target,'x') as z:
        for name,data in contents.items():_zipwrite(z,name,data)
    verify_odds_archive(target)
    return {'archive':str(target),'run_id':str(run_id),'quotes':index['complete_spread_prices'],
            'candidates':len(selected),'errors':index['fetch_errors'],
            'sha256':digest(target.read_bytes())}


def load_quotes(archive_root):
    """Choose latest captured valid quote before kickoff, fixed source priority."""
    result={}
    if not Path(archive_root).is_dir():return result
    for archive in sorted(Path(archive_root).rglob('*.zip')):
        index,rows=verify_odds_archive(archive)
        for row in rows:
            if row['status']!='OBSERVED_COMPLETE_SPREAD_AND_BOTH_PRICES':continue
            gid=row['game_id']
            evidence={'quote':row['quote'],'observed_at_utc':row['observed_at_utc'],
                    'home':row['home'],'away':row['away'],
                    'kickoff_utc':row['kickoff_utc'],
                    'source_url':row['source_url'],'raw_source_sha256':row['raw_source_sha256'],
                    'raw_provider_sha256':row['raw_provider_sha256'],
                    'market_archive_run_id':index['run_id'],
                    'market_archive_sha256':digest(archive.read_bytes()),
                    'book_timestamp_verified':False}
            prev=result.get(gid)
            if prev is None or (_date(evidence['observed_at_utc'],'observed'),int(index['run_id']))>(
                   _date(prev['observed_at_utc'],'observed'),int(prev['market_archive_run_id'])):
                result[gid]=evidence
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('capture');c.add_argument('--forecasts',required=True)
    c.add_argument('--out',required=True);c.add_argument('--run-id',required=True)
    c.add_argument('--days',type=int,default=7)
    v=sub.add_parser('verify');v.add_argument('zipfile')
    args=p.parse_args()
    if args.command=='capture': result=capture(args.forecasts,args.out,args.run_id,days=args.days)
    else:
        idx,recs=verify_odds_archive(args.zipfile)
        result={'run_id':idx['run_id'],'records':len(recs),'quotes':idx['complete_spread_prices']}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
