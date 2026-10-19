#!/usr/bin/env python3
"""Generate auditable pregame NFL matchup feature overlays automatically.

No market odds, injury states, or point predictions are invented. All targets
must kick off after the supplementary feed was actually captured.
"""
import argparse,json
from datetime import timedelta
from pathlib import Path
from integrate_advanced_features import build_overlay
from collect_extra_metrics import games_before_cutoff,parse_stamp,open_csv
from ingest import normalize_schedule, load_table


def generate(snapshot:Path, metrics_dir:Path, out:Path, window_hours:float=48):
    if window_hours<=0 or window_hours>168:raise ValueError('Window must be 0-168 hours')
    meta=json.loads((metrics_dir/'metrics_manifest.json').read_text())
    base=json.loads((snapshot/'manifest.json').read_text())
    cutoff=parse_stamp(meta['finalized_at_utc'])
    table=normalize_schedule(load_table(snapshot/base['datasets']['schedule']['local_file']))
    out.mkdir(parents=True,exist_ok=True)
    game_rows=[]
    from datetime import timezone
    for _,r in table.iterrows():
        kick=r['kickoff_utc'].to_pydatetime().astimezone(timezone.utc)
        if not (cutoff<kick<=cutoff+timedelta(hours=window_hours)):continue
        gid=str(r['game_id']);home=str(r['home_team']);away=str(r['away_team'])
        obj=build_overlay(snapshot,metrics_dir,cutoff.isoformat(),(home,away))
        obj['target_game_id']=gid
        obj['target_kickoff_utc']=kick.isoformat().replace('+00:00','Z')
        obj['note']='PRE-KICKOFF FEATURES ONLY; verified market/injury inputs still required. Provisional coefficients uncalibrated.'
        path=out/(gid+'.json');path.write_text(json.dumps(obj,indent=2)+'\n')
        game_rows.append({'game_id':gid,'home':home,'away':away,'kickoff_utc':obj['target_kickoff_utc'],
            'overlay_file':path.name,'missing':obj['missing']})
    index={'schema':'pregame_overlay_index_v0.6','capture_cutoff_utc':cutoff.isoformat().replace('+00:00','Z'),
        'matchups_found':len(game_rows),'window_hours':window_hours,'games':game_rows,
        'not_a_prediction':True}
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    return index

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--metrics-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--window-hours',type=float,default=48)
    a=p.parse_args();print(json.dumps(generate(a.snapshot,a.metrics_dir,a.out,a.window_hours),indent=2))
