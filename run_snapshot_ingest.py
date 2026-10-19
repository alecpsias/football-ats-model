#!/usr/bin/env python3
"""Bridge a verified nflverse download to strict game ingestion, without paid feeds.

The code populates the play-by-play and roster source provenance from the
actual immutable manifest; user-supplied timestamps are NEVER trusted.
Missing licensed/specialist metrics remain missing and block full prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingest import build, load_table
from nflverse_connector import verify


def run(snapshot: Path, config_path: Path, output_prefix: Path, extra_dir: Path | None=None) -> tuple[dict,dict]:
    validated=verify(snapshot)
    if validated['status'] != 'VERIFIED':
        raise ValueError('Snapshot checksum verification failed: '+str(validated['problems']))
    manifest=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    config=json.loads(config_path.read_text(encoding='utf-8'))
    config['pbp_source']='nflverse via nflreadpy; immutable snapshot '+str(snapshot)
    config['pbp_source_available_at_utc']=manifest['datasets']['pbp']['captured_at_utc']
    config['roster_source_available_at_utc']=manifest['datasets']['roster']['captured_at_utc']
    # No participation file: nflverse currently does not establish a reliable
    # same-season 'was_pressure' source. Never silently substitute QB hits.
    config['participation_source_available_at_utc']=None
    config['free_data_mode']=True
    pbp_path=snapshot/manifest['datasets']['pbp']['local_file']
    schedule_path=snapshot/manifest['datasets']['schedule']['local_file']
    roster_path=snapshot/manifest['datasets']['roster']['local_file']
    game,audit=build(load_table(pbp_path),load_table(schedule_path),config,
                     participation=None,roster=load_table(roster_path))
    if extra_dir is not None:
        # Only use supplementary features when their manifest, PFR capture,
        # and the base source hashes are verified as-of this target's cutoff.
        from integrate_advanced_features import build_overlay,attach_overlay
        try:
            overlay=build_overlay(snapshot,extra_dir,config['data_as_of_utc'],
                                  (config['home_code'],config['away_code']))
            game,audit=attach_overlay(game,audit,overlay,
                {'home':config['home_code'],'away':config['away_code']})
        except (ValueError,KeyError,OSError) as exc:
            audit['supplementary_integration_error']=str(exc)
            # Fail closed for a requested supplemental collection; do not
            # claim it was consumed or silently run an unaugmented model.
            audit['status']='BLOCKED_SUPPLEMENTARY_VALIDATION'
    audit['source_manifest']=str(snapshot/'manifest.json')
    audit['source_manifest_checksums']={k:v['sha256'] for k,v in manifest['datasets'].items() if v['status']=='SUCCESS'}
    audit['free_only']=True
    audit['manifest_file_verification']=validated['status']
    output_prefix.parent.mkdir(parents=True,exist_ok=True)
    (output_prefix.parent/(output_prefix.name+'.audit.json')).write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    suffix='.game.json' if audit['status']=='READY_FOR_MODEL_UNCALIBRATED' else '.INCOMPLETE.json'
    (output_prefix.parent/(output_prefix.name+suffix)).write_text(json.dumps(game,indent=2)+'\n',encoding='utf-8')
    return game,audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--extra-dir',type=Path,help='Verified run-matched derived-metrics folder')
    args=p.parse_args()
    game,audit=run(args.snapshot,args.config,args.output_prefix,args.extra_dir)
    print(json.dumps({'status':audit['status'],'missing_count':audit['missing_count'],
                      'asof_issues':audit['asof_issues'],'audit':str(args.output_prefix)+'.audit.json'},indent=2))
    if audit['status']!='READY_FOR_MODEL_UNCALIBRATED':
        raise SystemExit(2)

if __name__=='__main__':main()
