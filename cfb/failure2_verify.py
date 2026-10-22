"""Real-source Failure 2 audit; no fabricated eligible fixtures or ATS claims."""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import pandas as pd
from cfb.scoring_attribution import attribute

def evaluate(snapshot):
    root=Path(snapshot)
    pbp=pd.read_parquet(root/'pbp.parquet')
    schedule=pd.read_parquet(root/'schedule.parquet')
    grouped={gid:g for gid,g in pbp.groupby('game_id',sort=False)}
    results=collections.Counter()
    examples=[]
    eligible=schedule.loc[schedule.completed.eq(True) &
        schedule.home_points.notna() & schedule.away_points.notna()]
    for _,g in eligible.iterrows():
        raw=grouped.get(g.game_id)
        if raw is None:continue
        if raw.empty:continue
        x=attribute(raw,g.home_id,g.away_id,g.home_points,g.away_points)
        reason=x.get('reason') or x['status']
        results[reason]+=1
        if x['status']=='VERIFIED_RAW_SCORE_RECONCILIATION' and len(examples)<12:
            examples.append({'game_id':str(g.game_id),
                  'official_home':float(g.home_points),'official_away':float(g.away_points),
                  'offensive':x['offensive_points'],
                  'nonoffensive':x['nonoffensive_points']})
    return {'schema':'failure2_real_raw_score_audit_v1',
      'games_with_pbp':sum(results.values()),
      'fully_verified_games':results['VERIFIED_RAW_SCORE_RECONCILIATION'],
      'scoring_attribution_status_counts':dict(results),
      'verified_examples':examples,
      'conclusion':'ACTIVE_ON_REAL_DATA' if results['VERIFIED_RAW_SCORE_RECONCILIATION']>0
                   else 'NOT_VERIFIED_ON_REAL_DATA',
      'note':'Unit tests are not proof of non-offensive scoring frequency or 5000-run delta. Missing cumulative score fields prevent attribution.'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--snapshot',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    result=evaluate(a.snapshot)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
