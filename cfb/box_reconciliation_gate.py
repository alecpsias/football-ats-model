"""Independent university box-score reconciliation gate for CFB clock study.

Two authoritative 2026 university boxes, four teams, eight rushing and
passing totals. Fail instead of forging agreement or changing event labels.
"""
import argparse
import json
from pathlib import Path

def check(report):
    boxes=(
       ('independent_official_validation',(
          ('pbp_box_rush_attempts_home','official_rushes_home'),
          ('pbp_box_rush_attempts_away','official_rushes_away'),
          ('pbp_box_pass_attempts_home','official_pass_attempts_home'),
          ('pbp_box_pass_attempts_away','official_pass_attempts_away'))),
       ('second_independent_official_validation',(
          ('pbp_box_rush_attempts_tcu','official_rush_attempts_tcu'),
          ('pbp_box_rush_attempts_unc','official_rush_attempts_unc'),
          ('pbp_box_pass_attempts_tcu','official_pass_attempts_tcu'),
          ('pbp_box_pass_attempts_unc','official_pass_attempts_unc'))))
    errors=[]
    for key,pairs in boxes:
        sub=report.get(key) or {}
        if sub.get('status')!='PASS_WITHIN_90_SEC':
            errors.append(key+': official clock not reconciled')
        for seen,official in pairs:
            if seen not in sub or official not in sub or sub[seen]!=sub[official]:
                errors.append(key+': '+seen+' != '+official)
    return {'status':'ALL_EIGHT_STAT_COUNTS_MATCH' if not errors else 'BLOCKED',
            'checks':8,'errors':errors,
            'official_source_games':2,
            'limitation':'This does not verify every game, or remove rare provider edits'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True)
    a=p.parse_args()
    value=check(json.loads(Path(a.study).read_text()))
    print(json.dumps(value,indent=2))
    if value['status']!='ALL_EIGHT_STAT_COUNTS_MATCH':
        raise SystemExit(1)
