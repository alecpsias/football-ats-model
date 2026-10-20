"""Build 5 CFB-only diagnostics and frozen genuine pregame record tests."""
import ast
import hashlib
import json
import tempfile
import unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from cfb.diagnostics import investigate
from cfb.prospective import freeze
from cfb.forward_grade import verified_frozen, grade
from test_cfb_pipeline import schedule,plays


def dummy(n=120):
    start=pd.Timestamp('2026-01-01T12:00:00Z')
    return [{'game_id':str(i),'kickoff':(start+pd.Timedelta(days=i)).isoformat(),
             'home':'A','away':'B','baseline_margin':float((i%10)-5),
             'epa_candidate_margin':float((i%10)-5)+.2,
             'actual_margin':float((i%10)-5)*2,
             'sos_margin_feature':None,'context_margin_feature':None}
            for i in range(n)]


class Build5Tests(unittest.TestCase):
    def test_no_nfl_imports(self):
        for name in ('prospective.py','diagnostics.py','forward_grade.py'):
            tree=ast.parse((Path(__file__).parent/name).read_text())
            for n in ast.walk(tree):
                if isinstance(n,ast.Import):
                    self.assertFalse(any(x.name=='model' or x.name.startswith('nfl') for x in n.names))
                if isinstance(n,ast.ImportFrom):
                    self.assertFalse(n.module=='model' or (n.module or '').startswith('nfl'))

    def test_training_selection_ignores_holdout_results(self):
        rows=dummy()
        original=investigate(rows)
        for r in rows[-24:]:r['actual_margin']+=200
        changed=investigate(rows)
        self.assertEqual(original['champion_selected_using_validation_only'],
                         changed['champion_selected_using_validation_only'])
        self.assertEqual(original['candidate_validation_only'],changed['candidate_validation_only'])
        self.assertNotEqual(original['untouched_test_selected'],changed['untouched_test_selected'])
        self.assertFalse(original['ats_validated'])

    def test_diagnostic_reports_big_misses_without_claiming_cause(self):
        rows=dummy();rows[-1]['actual_margin']+=100
        r=investigate(rows)
        self.assertEqual(r['test_games'],24)
        self.assertEqual(r['worst_15_test_misses_post_selection_diagnostics_only'][0]['game_id'],'119')
        self.assertIn('missed_winner',r['test_error_buckets_after_model_selection'])

    def test_diagnostic_rejects_duplicates_and_insufficient(self):
        rows=dummy();rows[4]['game_id']=rows[3]['game_id']
        with self.assertRaises(ValueError):investigate(rows)
        self.assertEqual(investigate(dummy(4))['status'],'INSUFFICIENT_GAMES')

    @staticmethod
    def snapshot_data(root):
        target=Path(root);target.mkdir(parents=True,exist_ok=True)
        cutoff='2026-10-19T12:00:00+00:00'
        m={'schema':'cfb_source_snapshot_v2','captured_at_utc':cutoff,
           'datasets':{k:{'sha256':'b'*64,'file':k+'.parquet'}
                       for k in ('pbp','schedule','roster')}}
        (target/'manifest.json').write_text(json.dumps(m))
        s=schedule().copy()
        s['home_points']=21;s['away_points']=14
        s.loc[s.completed.eq(False),['home_points','away_points']]=None
        p=plays()
        q=pd.DataFrame([{'team_id':1,'position_abbreviation':'QB','athlete_id':'11'},
                        {'team_id':2,'position_abbreviation':'QB','athlete_id':'22'}])
        return m,{'schedule':s,'pbp':p,'roster':q}

    def _run(self, snap, output, m, datasets, min_games=2):
        def fake_parquet(path):return datasets[Path(path).stem]
        with patch('cfb.prospective.verify',return_value=m),\
             patch('cfb.prospective.pd.read_parquet',side_effect=fake_parquet):
            return freeze(snap,output,runs=120,days=7,min_games=min_games,
                now='2026-10-19T12:10:00+00:00',git_sha='a'*40,workflow_run_id='123')

    def test_forward_freezes_realistic_case_with_sha_and_no_spread(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);snap=base/'snap';m,df=self.snapshot_data(snap)
            out=base/'forecast'
            result=self._run(snap,out,m,df)
            self.assertEqual(result['game_records'],2) # 103 and 106, not FCS-FCS
            self.assertEqual(result['margin_projections'],1)
            self.assertFalse(result['ats_validated'])
            f=json.loads((out/'103.json').read_text())
            self.assertEqual(f['status'],'FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE')
            self.assertIsNone(f['market_quote']);self.assertIsNone(f['ats_prediction'])
            self.assertLess(f['prediction_generated_at_utc'],f['kickoff_utc'])
            self.assertNotIn('model.py',(Path(__file__).parent/'prospective.py').read_text())
            self.assertEqual(result['predictions'][0]['sha256'],hashlib.sha256((out/'103.json').read_bytes()).hexdigest())
            result2=self._run(snap,base/'again',m,df)
            self.assertEqual(result['predictions'],result2['predictions'])
            with self.assertRaises(ValueError):self._run(snap,out,m,df)

    def test_missing_team_history_is_frozen_pass(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);snap=base/'snap';m,df=self.snapshot_data(snap)
            r=self._run(snap,base/'preds',m,df,min_games=5)
            self.assertEqual(r['margin_projections'],0)
            self.assertEqual(r['insufficient_data'],2)
            z=json.loads((base/'preds'/'103.json').read_text())
            self.assertNotIn('simulated_margin_home',z)

    def test_future_scores_do_not_enter_history(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);snap=base/'snap';m,df=self.snapshot_data(snap)
            df['schedule'].loc[df['schedule'].game_id.eq(103),'home_points']=9999
            df['schedule'].loc[df['schedule'].game_id.eq(103),'away_points']=0
            a=self._run(snap,base/'a',m,df)
            df['schedule'].loc[df['schedule'].game_id.eq(103),'home_points']=1
            b=self._run(snap,base/'b',m,df)
            self.assertEqual(a['predictions'],b['predictions'])


    def test_frozen_digest_protects_forward_record(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);snap=base/'snap';m,df=self.snapshot_data(snap)
            self._run(snap,base/'preds',m,df)
            self.assertEqual(len(verified_frozen(base/'preds')),2)
            f=base/'preds'/'103.json';f.write_bytes(f.read_bytes()+b'x')
            with self.assertRaises(ValueError):verified_frozen(base/'preds')

    def test_forward_grading_never_invents_odds_and_safely_grades(self):
        from cfb.market import FIELDS,CFBMarketError
        import csv
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);snap=base/'snap';m,df=self.snapshot_data(snap)
            self._run(snap,base/'preds',m,df)
            empty=grade(base/'preds',base/'quotes.csv',base/'final.csv')
            self.assertFalse(empty['ats_validated']);self.assertEqual(empty['games_graded'],0)
            pred=json.loads((base/'preds'/'103.json').read_text())
            q={'game_id':'103','home':'SCHOOL_A','away':'SCHOOL_B',
               'kickoff_utc':pred['kickoff_utc'],
               'quoted_at_utc':'2026-10-19T10:00:00Z',
               'captured_at_utc':'2026-10-19T11:00:00Z',
               'book':'Public source','source_url':'https://example.org/odds',
               'spread_home':'-7.5','home_odds_american':'-110',
               'away_odds_american':'-110','evidence_sha256':'a'*64}
            with (base/'quotes.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerow(q)
            final={'game_id':'103','home_points':'21','away_points':'7',
                   'final_observed_at_utc':'2026-10-25T00:00:00Z',
                   'result_source_url':'https://example.org/final',
                   'result_source_sha256':'b'*64}
            with (base/'final.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(final));w.writeheader();w.writerow(final)
            result=grade(base/'preds',base/'quotes.csv',base/'final.csv')
            self.assertEqual(result['games_graded'],1)
            self.assertFalse(result['ats_validated'])
            self.assertEqual(result['games'][0]['actual_ats_margin'],6.5)
            q['captured_at_utc']='2026-10-19T12:11:00Z'
            with (base/'quotes.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerow(q)
            with self.assertRaises(CFBMarketError):grade(base/'preds',base/'quotes.csv',base/'final.csv')

if __name__=='__main__':unittest.main()
