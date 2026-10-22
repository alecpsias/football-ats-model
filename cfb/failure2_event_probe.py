"""Diagnostic of target 2026 games: raw event score transitions and type flags."""
import argparse
import pandas as pd
from pathlib import Path
from cfb.pipeline import verify
from cfb.scoring_attribution import SCORE_PAIRS

def run(snapshot):
    root=Path(snapshot);verify(root)
    pbp=pd.read_parquet(root/'pbp.parquet')
    for game in (401860881,401862800):
        p=pbp.loc[pbp.game_id.eq(game)].sort_values('game_play_number').reset_index(drop=True)
        print('EVENT_PROBE_GAME',game,'rows',len(p))
        pair=next((x for x in SCORE_PAIRS if set(x).issubset(p.columns)),None)
        if pair is None:raise ValueError('No source score pair')
        score=p[list(pair)].apply(pd.to_numeric,errors='coerce')
        delta=score.diff().fillna(score)
        interesting=delta.ne(0).any(axis=1)
        if 'td_play' in p.columns:interesting|=p.td_play.fillna(False).eq(True)
        print('FIELDS',[(col,str(p[col].dtype)) for col in p.columns if any(x in col for x in ('type','text','touchdown','td','score','period','pos_team','game_play_number','fg_'))][:90])
        for i in p.index[interesting]:
            r=p.loc[i]
            print('EVT',str(int(r.game_play_number)),
             'delta',tuple(delta.loc[i].to_list()),
             'score',tuple(score.loc[i].to_list()),
             'type',r.get('type.text'), 'text',str(r.get('text'))[:220],
             'pos',r.get('pos_team_id'), 'offense_score_play',r.get('offense_score_play'),
             'td_play',r.get('td_play'),'pass',r.get('pass'),'rush',r.get('rush'),
             'int',r.get('int'),'fum',r.get('pos_fumble_lost'),'FG',r.get('fg_made'),
             'period',r.get('period.number'), 'prior',
             str(p.loc[i-1,'text'])[:130] if i else '')
        print('END_GAME_PROBE')

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True)
    args=p.parse_args();run(args.snapshot)

if __name__=='__main__':main()
