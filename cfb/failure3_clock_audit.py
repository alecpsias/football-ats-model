"""Research-only coverage diagnostic for 2026 SOURCE-verified possession time."""
import argparse,collections,json
from pathlib import Path
import pandas as pd
from cfb.clock_possession_validation import audit_game
from cfb.pipeline import verify

def audit(snapshot):
    root=Path(snapshot)
    verify(root)
    pbp=pd.read_parquet(root/'pbp.parquet')
    sched=pd.read_parquet(root/'schedule.parquet')
    source={k:g for k,g in pbp.groupby('game_id',sort=False)}
    count=collections.Counter();good=[];bad=[]
    for _,g in sched.loc[sched.completed.fillna(False)].iterrows():
        raw=source.get(g.game_id)
        if raw is None:continue
        z=audit_game(raw,g.game_id,int(g.home_id),int(g.away_id))
        count[z['status']]+=1
        count['source_eligible_regulation_drives']+=len(z.get('drives',()))
        count['independently_endpoint_verified_drives']+=sum(
            d['clock_verification']=='ELAPSED_AND_ENDPOINTS_AGREE'
            for d in z.get('drives',()))
        if z['status']=='COMPLETE_REGULATION_CLOCK_AUDIT':
            count['complete_game_verified_drives']+=len(z['drives'])
            if len(good)<8:good.append({'game_id':str(g.game_id),
                'coverage':z.get('regulation_clock_coverage'),
                'drive_count':z['valid_drive_count'],
                'source_seconds':z['observed_seconds_total']})
        elif len(bad)<8:bad.append({'game_id':str(g.game_id),
                    'status':z['status'],
                    'coverage':z.get('regulation_clock_coverage'),
                    'excluded':z.get('excluded')})
    return {'schema':'cfb_failure3_source_regulation_clock_audit_v1',
            'counts':dict(count),'complete_game_examples':good,
            'partial_examples':bad,
            'clock_provenance':'ESPN source elapsed and start/end match, complete game coverage',
            'independent_official_box_time_of_possession_checked':False,
            'no_unverified_clock_imputation':True}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    x=audit(a.snapshot)
    out=Path(a.out);out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(x,indent=2,sort_keys=True))
    print(json.dumps(x,indent=2,sort_keys=True))
if __name__=='__main__':main()
