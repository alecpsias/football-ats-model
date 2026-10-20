#!/usr/bin/env python3
"""Run existing common ATS engine on fully verified CFB game snapshot only."""
import argparse
import json
from pathlib import Path
from model import InputError,run

def forecast(game,runs=20000,seed=20261019):
    if game.get('league')!='CFB':
        raise InputError('CFB forecast requires league=CFB')
    if game.get('home_division')=='fcs' and game.get('away_division')=='fcs':
        raise InputError('FCS-vs-FCS game is excluded')
    return run(game, runs=runs, seed=seed)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input');p.add_argument('--out',required=True)
    p.add_argument('--runs',type=int,default=20000);p.add_argument('--seed',type=int,default=20261019)
    a=p.parse_args()
    result=forecast(json.loads(Path(a.input).read_text()),a.runs,a.seed)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print('CFB projection generated. Grade:',result.get('grade'),'Calibration:',result.get('audit',{}).get('probabilities_empirically_calibrated'))
if __name__=='__main__':main()
