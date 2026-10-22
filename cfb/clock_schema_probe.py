"""Audit actual CFB 2026 PBP event-clock and classification fields, no models."""
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from cfb.pipeline import verify

def audit(snapshot):
    verify(snapshot)
    path=Path(snapshot)/"pbp.parquet"
    schema=pq.read_schema(path)
    cols=list(schema.names)
    selected=[x for x in cols if any(k in x.casefold() for k in
      ("clock","time","period","drive","play_type","type.","pos_team","rush","pass","kickoff","punt","wall","half","firstd","score"))]
    needed=["game_id","game_play_number","drive_id","pos_team_id","def_pos_team_id",
            "pass","rush","period","half","clock_minutes","clock_seconds",
            "clock.displayValue","period.number","start.TimeSecsRem",
            "drive_time_of_possession","type.text","type.id","scoring_play"]
    present=[x for x in needed if x in cols]
    t=pd.read_parquet(path,columns=present)
    ids=[401871070] if 401871070 in set(t.game_id.dropna()) else []
    extra=t.loc[~t.game_id.isin(ids),"game_id"].dropna().drop_duplicates().head(2).tolist()
    ids+=list(extra)
    details=[]
    for gid in ids:
        g=t.loc[t.game_id.eq(gid)].sort_values("game_play_number").head(35)
        pick=[x for x in present if x!="game_id"]
        sample=g[pick].astype(object).where(pd.notna(g[pick]),None)
        details.append({"game_id":str(gid),"count":int(t.game_id.eq(gid).sum()),
                        "sample":sample.to_dict("records")[:15]})
    info={"schema":"cfb_actual_clock_source_probe_v1","all_fields_count":len(cols),
          "candidate_clock_event_fields":selected,"present_required":present,
          "missing_required":[x for x in needed if x not in cols],
          "sample_games":details}
    return info

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    r=audit(a.snapshot)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(r,indent=2,default=str)+"\n")
    print(json.dumps(r,indent=2,default=str))
