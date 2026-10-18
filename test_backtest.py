import copy
import datetime
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import InputError, run
from backtest import replay

BASE=json.loads((Path(__file__).resolve().parents[1]/'examples'/'synthetic_matchup.json').read_text())

def make_historical_jsonl(path, number=8):
    with open(path,'w',encoding='utf8') as out:
        for i in range(number):
            g=copy.deepcopy(BASE)
            start=datetime.datetime(2026,5,3,20,25,tzinfo=datetime.timezone.utc)+datetime.timedelta(weeks=i)
            g['id']=f'SYNTHETIC-HIST-{i}'
            g['kickoff_utc']=start.isoformat()
            g['data_as_of_utc']=(start-datetime.timedelta(hours=2)).isoformat()
            g['market']['as_of_utc']=(start-datetime.timedelta(hours=3)).isoformat()
            for side in ('home','away'):
                g[side]['stat_as_of_utc']=(start-datetime.timedelta(days=1)).isoformat()
                g[side]['personnel']['checked_at_utc']=(start-datetime.timedelta(hours=3)).isoformat()
            out.write(json.dumps({'game':g,'actual_home_score':20+(i%5), 'actual_away_score':24+(i%4)})+'\n')

class HistoricalChecks(unittest.TestCase):
    def test_chronological_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            fn=Path(tmp)/'fake_history.jsonl'
            make_historical_jsonl(fn)
            report=replay(fn,1000,42, allow_synthetic_for_software_tests=True)
            self.assertEqual(report['games_excluding_pushes'],8)
            self.assertEqual(report['league'],'NFL')
            self.assertIn('INSUFFICIENT',report['validation_status'])
            self.assertTrue(report['holdout_first_kickoff_utc']>report['train_last_kickoff_utc'])

    def test_failed_calibrator_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            fn=Path(tmp)/'fake_history.jsonl'
            make_historical_jsonl(fn)
            cal=replay(fn,1000,42,allow_synthetic_for_software_tests=True)
            with self.assertRaises(InputError):run(copy.deepcopy(BASE),1000,1,False,cal)

    def test_synthetic_history_cannot_claim_real_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            fn=Path(tmp)/'fake_history.jsonl'
            make_historical_jsonl(fn)
            with self.assertRaises(InputError):replay(fn,1000,42)

    def test_first_half_price_required(self):
        g=copy.deepcopy(BASE)
        g['market']['first_half']={'spread_home':-3,'total':22}
        with self.assertRaises(InputError):run(g,1000,1,False)

    def test_stale_personnel_check_fails(self):
        g=copy.deepcopy(BASE)
        g['home']['personnel']['checked_at_utc']='2026-10-15T19:00:00Z'
        with self.assertRaises(InputError):run(g,1000,1,False)

    def test_home_spread_monotonicity(self):
        a=run(copy.deepcopy(BASE),4000,2)
        s=a['stress_tests']['spread_home_sensitivity']
        vals=[s[k]['home_cover_raw'] for k in sorted(s,key=float)]
        self.assertEqual(vals,sorted(vals))

if __name__=='__main__':unittest.main()
