"""Independently audit source-clock duration against USC's official TOP.

This historical result is exclusively a measurement validation anchor; it is
never a pregame feature or used to fit clock behavior.
Official box: https://usctrojans.com/sports/football/stats/2026/san-jos-state/boxscore/35264
USC TOP 37:15 (2235s); San Jose State TOP 22:45 (1365s), Aug 29 2026.
"""
import argparse,json
from pathlib import Path
import pandas as pd
from cfb.pipeline import verify
from cfb.clock_possession_validation import audit_game,clock_reference_difference

GAME_ID=401864494
USC_OFFICIAL_TOP=2235
SJS_OFFICIAL_TOP=1365
SOURCE_URL='https://usctrojans.com/sports/football/stats/2026/san-jos-state/boxscore/35264'

def evaluate(snapshot):
    root=Path(snapshot);verify(root)
    games=pd.read_parquet(root/'schedule.parquet')
    row=games.loc[games.game_id.eq(GAME_ID)]
    if len(row)!=1:return {'status':'GAME_UNAVAILABLE','id':GAME_ID}
    g=row.iloc[0]
    # Don't assign external box data to incorrectly matched game IDs.
    if 'usc' not in str(g.home_team).lower() or 'san jos' not in str(g.away_team).lower():
        return {'status':'OFFICIAL_SOURCE_TEAM_MISMATCH','home':str(g.home_team),
                'away':str(g.away_team)}
    pbp=pd.read_parquet(root/'pbp.parquet')
    raw=pbp.loc[pbp.game_id.eq(GAME_ID)]
    z=audit_game(raw,GAME_ID,int(g.home_id),int(g.away_id))
    d=clock_reference_difference(z,{int(g.home_id):USC_OFFICIAL_TOP,
                                    int(g.away_id):SJS_OFFICIAL_TOP})
    return {'schema':'cfb_clock_independent_official_box_usc_2026_v1',
        'game_id':str(GAME_ID),'official_source':SOURCE_URL,
        'official_top_seconds':{'USC':USC_OFFICIAL_TOP,
                                'San Jose State':SJS_OFFICIAL_TOP},
        'source_audit_status':z['status'],
        'source_covered_seconds':z['observed_seconds_total'],
        'source_valid_drives':z.get('valid_drive_count'),
        'source_only_elapsed_drives':z.get('source_only_elapsed_drive_count'),
        'official_comparison_status':d['status'],
        'official_comparison':d,
        'note':'Official-box comparison only, excluded from future features and tuning'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    r=evaluate(a.snapshot)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(r,indent=2,sort_keys=True))
    print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__':main()
