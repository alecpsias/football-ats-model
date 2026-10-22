"""Reparse saved prekickoff ESPN CFB odds bytes; never backdate or fetch.

All prior odds zip files remain byte-for-byte immutable. This process does
not create odds for a game that wasn't originally observed before kickoff;
it only corrects deterministic processing of preserved original source JSON.
"""
import argparse,json
from pathlib import Path
from cfb.odds_recover import recover_archive

def backfill_preserved_originals(root):
    root=Path(root)
    targets=sorted(root.glob('*.zip'))
    all_records=[]
    for old in targets:
        output=root/'recovered'/old.name
        outcome=recover_archive(old,output)
        all_records.append(outcome)
    return {
        'status':'REPARSE_ORIGINAL_SAVED_PREGAME_BYTES_ONLY',
        'source_archives_inspected':len(targets),
        'source_files_never_changed':True,
        'new_recovered_quotes':sum(x.get('recovered',0) for x in all_records),
        'recoverable_original_quotes_including_previous_repairs':sum(
            x.get('recoverable_quotes',0) if x['status']=='ALREADY_ARCHIVED_UNCHANGED'
            else x.get('recovered',0) for x in all_records),
        'no_network_access':True,
        'no_synthetic_lines':True,
        'source_observed_timestamps_retained':True,
        'bookmaker_actual_published_time_unverified':True,
        'results':all_records
    }

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    a=p.parse_args()
    print(json.dumps(backfill_preserved_originals(a.root),indent=2))
