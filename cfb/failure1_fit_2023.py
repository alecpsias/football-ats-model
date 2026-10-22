"""Fit one frozen supervised rushing correction exclusively to 2023 pregame records.

No 2024+ results may be imported at this step.
"""
import argparse,json
from pathlib import Path
from cfb.failure1_ridge_correction import fit

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-2023',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    d=json.loads(Path(a.train_2023).read_text())
    if d['source_season']!=2023 or d['fitted_model_year'] is not None:
        raise ValueError('Training requires unfitted, exclusively 2023 research')
    rows=[r for r in d['replays'] if r['features'] is not None]
    if any(r['year']!=2023 for r in rows):raise ValueError('Future leakage')
    model=fit(rows)
    target=Path(a.out)
    if target.exists():raise FileExistsError('Refuse overwrite frozen 2023 fit')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(model,indent=2)+'\n')
    print(json.dumps({'status':'FROZEN_ON_2023_ONLY','model':model},indent=2))
if __name__=='__main__':main()
