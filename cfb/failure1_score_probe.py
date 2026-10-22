"""Inspect actual public PBP pre-play scoreboard, never backfill guessed score."""
import argparse,json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from cfb.pipeline import verify,normalize_pbp_teams

def probe(root):
    root=Path(root); verify(root)
    schema=pq.read_schema(root/'pbp.parquet').names
    keys=[c for c in schema if any(w in c.lower() for w in
      ('score','points','homeaway','clock','period','lead','start.time','playtext'))]
    print('SCORE_COLS',json.dumps(keys))
    needed=list(dict.fromkeys(['game_id','game_play_number','pos_team_id','def_pos_team_id','pass','rush',
                              'home','away','pos_score_diff_start','homeScore','awayScore',
                              'home_score','away_score','homeScoreBefore','awayScoreBefore',
                              'homeScoreAfter','awayScoreAfter','scoreValue',
                              'homeScoreStart','awayScoreStart','offense_score','defense_score',
                              'offense_score_play','td_play']+[x for x in keys if 'score' in x.lower()]))
    f=pd.read_parquet(root/'pbp.parquet',columns=[x for x in needed if x in schema])
    report={'schema':'cfb_raw_score_provenance_probe_v1','all_columns':len(schema),'candidate_columns':keys,
      'rows':len(f),'availability':{},'sample_first_game':{}}
    for c in [x for x in needed if x in f]:
        report['availability'][c]={'not_null':int(f[c].notna().sum()),
            'distinct':int(f[c].nunique(dropna=True))}
    gid=f.game_id.value_counts().index[0]
    sub=f.loc[f.game_id.eq(gid)].sort_values('game_play_number').head(14)
    sc=[x for x in sub if x in keys or x in ('game_id','game_play_number','pos_team_id','home','away','rush','pass')]
    report['sample_first_game']={'game_id':str(gid),'rows':sub[sc].astype(object).where(pd.notna(sub[sc]),None).to_dict('records')}
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    r=probe(a.snapshot)
    f=Path(a.out);f.parent.mkdir(parents=True,exist_ok=True)
    f.write_text(json.dumps(r,indent=2,default=str)+'\n')
    print('SCORE_SUMMARY',json.dumps({k:v for k,v in r.items() if k!='sample_first_game'},indent=2))
    print('SCORE_SAMPLE',json.dumps(r['sample_first_game'],default=str)[:8000])
