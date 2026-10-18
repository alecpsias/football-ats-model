#!/usr/bin/env python3
"""Summarize status without declaring an incomplete historical replay a prediction."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def compose(snapshot: Path, replay_audit: Path) -> str:
    lines = ['# Free nflverse ingest audit', '', 'No paid API or subscription was used.', '']
    m = snapshot/'manifest.json'
    if not m.exists():
        lines.extend(['**Collection:** FAILED / no manifest present.', ''])
    else:
        manifest = json.loads(m.read_text(encoding='utf-8'))
        lines.extend([
            f"**Collection:** {manifest.get('status','UNKNOWN')}",
            f"**Season:** {manifest.get('season','UNKNOWN')}",
            f"**Capture time:** {manifest.get('created_at_utc','UNKNOWN')}",
            '',
            '| Dataset | Status | Rows | SHA-256 |',
            '|---|---|---:|---|',
        ])
        for key, entry in manifest.get('datasets',{}).items():
            lines.append(f"| {key} | {entry.get('status','UNKNOWN')} | {entry.get('row_count','—')} | `{entry.get('sha256','missing')}` |")
        lines.append('')
    if replay_audit.exists():
        data=json.loads(replay_audit.read_text(encoding='utf-8'))
        lines.extend([
            '## Texans–Jaguars retrospective audit',
            '',
            f"**Status:** {data.get('status','UNKNOWN')}",
            f"**Missing required fields:** {data.get('missing_count','UNKNOWN')}",
            '',
            'A snapshot downloaded after the game cannot prove historical source availability,',
            'even when game-level play-by-play is correctly filtered to earlier completed games.',
            '',
        ])
        issues=data.get('asof_issues') or []
        if issues:
            lines.append('**Point-in-time issues:**')
            lines.extend(f'- {issue}' for issue in issues[:20])
            lines.append('')
    else:
        lines.append('**Replay:** Not requested or could not run. No pregame prediction has been generated.\n')
    lines.append('**Calibration:** NOT VALIDATED. Do not infer an ATS betting advantage from test success.\n')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--replay-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--github-summary',type=Path)
    args=parser.parse_args()
    result=compose(args.snapshot,args.replay_audit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(result,encoding='utf-8')
    if args.github_summary:
        with args.github_summary.open('a',encoding='utf-8') as handle:
            handle.write(result+'\n')
    print(result)

if __name__=='__main__':
    main()
