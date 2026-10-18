#!/usr/bin/env python3
"""Audit collection and historical as-of evidence; always disclose retrospective status."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nflverse_connector import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--cutoff', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.snapshot, cutoff=args.cutoff)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    # A postgame snapshot must NOT be presented as a genuine pregame archive.
    if result['pregame_eligibility'] != 'CAPTURED_BEFORE_CUTOFF':
        print('NOTICE: File collection occurred too late to qualify as a strict pregame snapshot.')
    if result['status'] != 'VERIFIED':
        raise SystemExit(2)

if __name__ == '__main__':
    main()
