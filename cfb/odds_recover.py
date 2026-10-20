"""Losslessly repair CFB odds PARSING from already archived, pregame ESPN JSON.

Never fetch an external URL, guess a line, or alter the original capture.
A repaired quote inherits its source observation timestamp (NOT its bookmaker
publication timestamp). The repair time is separately recorded. Output is
append-only and independently checks original and repaired evidence hashes.
"""
from __future__ import annotations
import argparse
import copy
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from cfb.archive import digest, _zipwrite
from cfb.free_odds import choose_espn_quote, verify_odds_archive
from cfb.market import _date


def _no_network(_url):
    raise OSError('Recovery must never fetch a live price')


def recover_archive(original, repaired, now=None):
    """Re-parse saved raw ESPN JSON and create a provenance-linked NEW ZIP.

    Existing recovered archives are checked, but are never overwritten. Returns
    the original archive's count of verifiable inline, pre-kickoff market quotes.
    """
    original, repaired = Path(original), Path(repaired)
    if original.resolve() == repaired.resolve():
        raise ValueError('Cannot replace the original odds evidence')
    source_index, source_rows = verify_odds_archive(original)
    source_sha = digest(original.read_bytes())
    stamp = _date(now.isoformat() if isinstance(now,datetime) else
                  (now if now is not None else datetime.now(timezone.utc).isoformat()),
                  'reconstruction timestamp')
    if repaired.exists():
        repaired_index, _ = verify_odds_archive(repaired)
        if (repaired_index.get('original_archive_sha256') != source_sha or
                repaired_index.get('recovery_method') != 'REPARSE_SAVED_INLINE_ESPN_JSON_V1'):
            raise FileExistsError('Existing repaired archive has different provenance')
        return {'status':'ALREADY_ARCHIVED_UNCHANGED','original':str(original),
                'repaired':str(repaired),'source_archive_sha256':source_sha,
                'recoverable_quotes':repaired_index['complete_spread_prices']}
    repaired_rows=[]
    recovered=0
    with zipfile.ZipFile(original) as z:
        saved={name:z.read(name) for name in z.namelist() if name!='index.json'}
        for row in source_rows:
            updated=copy.deepcopy(row)
            gid=row['game_id']
            raw_file=row.get('raw_source_file')
            if (row['status']=='NO_COMPLETE_VERIFIABLE_SPREAD_ODDS'
                and raw_file and raw_file in saved
                and _date(row['observed_at_utc'],'original source observation')
                  < _date(row['kickoff_utc'],'original kickoff')):
                q, provider_raw, provider_url = choose_espn_quote(
                    saved[raw_file],fetcher=_no_network,expected_gid=gid)
                # The archived collection item must contain a *complete* quote
                # inline: provider-only refs are not re-fetched or backdated.
                if q is not None and provider_raw is None and provider_url is None:
                    updated['quote']=q
                    updated['status']='OBSERVED_COMPLETE_SPREAD_AND_BOTH_PRICES'
                    updated['original_parse_status']=row['status']
                    updated['reparsed_at_utc']=stamp.isoformat()
                    updated['original_source_archive_sha256']=source_sha
                    updated['observation_only']=(
                        'Quote recovered from unchanged ESPN response bytes '
                        'originally observed before kickoff; bookmaker publication '
                        'timestamp remains unverified')
                    recovered+=1
            repaired_rows.append(updated)
    # Copy every original raw response unchanged; update only row JSON + index.
    if recovered==0:
        return {'status':'NO_RECOVERABLE_INLINE_ODDS','original':str(original),
                'recovered':0, 'source_archive_sha256':source_sha}
    records=[]
    for row in repaired_rows:
        file=row['game_id']+'.json'
        data=(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode()
        saved[file]=data
        records.append({'game_id':row['game_id'],'record_file':file,
                        'record_sha256':digest(data),'status':row['status']})
    index=copy.deepcopy(source_index)
    index['records']=records
    index['complete_spread_prices']=sum(
        r['status']=='OBSERVED_COMPLETE_SPREAD_AND_BOTH_PRICES' for r in repaired_rows)
    index['missing_or_incomplete_prices']=sum(
        r['status']=='NO_COMPLETE_VERIFIABLE_SPREAD_ODDS' for r in repaired_rows)
    index['original_archive_sha256']=source_sha
    index['recovered_at_utc']=stamp.isoformat()
    index['recovered_quote_count']=recovered
    index['recovery_method']='REPARSE_SAVED_INLINE_ESPN_JSON_V1'
    index['source_observation_times_preserved']=True
    index['book_publish_times_independently_verified']=False
    index['note']='Source bytes, original observation times, and original ZIP remain unchanged'
    data=(json.dumps(index,sort_keys=True,separators=(',',':'))+'\n').encode()
    repaired.parent.mkdir(parents=True,exist_ok=True)
    # zipfile 'x' and source-first verification prevent silent overwrite.
    with zipfile.ZipFile(repaired,'x') as z:
        _zipwrite(z,'index.json',data)
        for name in sorted(saved):
            _zipwrite(z,name,saved[name])
    audit,_=verify_odds_archive(repaired)
    return {'status':'REPAIRED_FROM_ORIGINAL_PREKICKOFF_BYTES',
            'original':str(original),'repaired':str(repaired),
            'source_archive_sha256':source_sha,
            'repaired_archive_sha256':digest(repaired.read_bytes()),
            'recovered':recovered,'complete_spread_prices':audit['complete_spread_prices']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    print(json.dumps(recover_archive(args.original,args.out),indent=2))


if __name__=='__main__':main()
