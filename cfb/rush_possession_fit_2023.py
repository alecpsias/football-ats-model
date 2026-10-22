"""Train exactly one 2023-only sustained-drive residual coefficient."""
import argparse,json
from pathlib import Path
from cfb.rush_possession_response import fit

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-2023',required=True)
    p.add_argument('--out',required=True);args=p.parse_args()
    s=json.loads(Path(args.train_2023).read_text())
    if s['season']!=2023 or not s['training_only']:
        raise ValueError('Cannot fit on holdout or partially fit source')
    rows=s['study_rows']
    if any(z['year']!=2023 or z['candidate'] is not None for z in rows):
        raise ValueError('Data leakage into training rows')
    result=fit(rows)
    o=Path(args.out);o.parent.mkdir(parents=True,exist_ok=True)
    if o.exists():raise FileExistsError('Frozen coefficient immutable')
    o.write_text(json.dumps(result,indent=2)+'\n')
    print('FROZEN_2023_POSSESSION_FIT',json.dumps(result))
if __name__=='__main__':main()
