"""Build 8: human-requested CFB pregame -> frozen prediction -> postgame -> learning gate."""
from __future__ import annotations
import ast
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from cfb.archive import pack_forecasts,load_all_archives,digest
from cfb.matchup_cycle import resolve,pregame,postgame,_hypotheses,markdown,normalized
from cfb.learn_forward import observations,investigate
from cfb.test_cfb_build6 import frozen,G,K
from cfb.test_cfb_pipeline import plays


def fixture(folder):
    p=frozen(folder)
    p.update({'model_variant':'UNMODIFIED_UNCALIBRATED_BASELINE',
              'simulated_score_home':28.6,'simulated_score_away':22.6,
              'home_projection':{'_ppd':2.4,'expected_drives':10},
              'away_projection':{'_ppd':2.0,'expected_drives':10},
              'matchup_evidence':{'sos_margin_feature':None},
              'completed_prior_games_home':6,'completed_prior_games_away':6})
    b=(json.dumps(p,sort_keys=True)+'\n').encode()
    (Path(folder)/(G+'.json')).write_bytes(b)
    ix=json.loads((Path(folder)/'index.json').read_text())
    ix['predictions'][0]['sha256']=digest(b)
    (Path(folder)/'index.json').write_text(json.dumps(ix)+'\n')
    return p


def final_schedule(done=True):
    return pd.DataFrame([dict(game_id=int(G),completed=done,home_id=333,away_id=245,
        home_division='fbs',away_division='fbs',
        start_date=K,home_team='Alabama',away_team='Texas A&M',home_points=21 if done else None,
        away_points=24 if done else None)])


class Build8Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        base=Path(self.temp.name);folder=base/'fresh'
        fixture(folder)
        pack_forecasts(folder,base/'archives')
        self.archive=base/'archives'
        self.p=load_all_archives(self.archive)[G]

    def test_team_selection_is_order_independent_exact(self):
        self.assertEqual(resolve({G:self.p},'texas a&m','ALABAMA')['game_id'],G)
        self.assertEqual(normalized('Texas A&M'),normalized('texas a&m'))

    def test_unmatched_school_is_not_guessed(self):
        with self.assertRaisesRegex(ValueError,'No exact archived fixture'):
            resolve({G:self.p},'Alabama','Tennessee')

    def test_reject_same_school(self):
        with self.assertRaises(ValueError):resolve({G:self.p},'Alabama','ALABAMA')

    def test_game_id_restriction(self):
        with self.assertRaises(ValueError):resolve({G:self.p},'Alabama','Texas A&M','401999999')

    def test_pregame_is_original_locked_projection_not_updated(self):
        report=pregame(self.p,{})
        self.assertEqual(report['projected_home_margin'],6)
        self.assertEqual(report['grade'],'NOT_CALIBRATED')
        self.assertEqual(report['market']['ats_research_side'],None)
        self.assertEqual(report['original_forecast_zip_sha256'],digest((self.archive/'1234567.zip').read_bytes()))
        self.assertIn('28.6',markdown(report))

    def test_market_quote_must_be_pre_kickoff(self):
        fake={G:{'home':'Alabama','away':'Texas A&M','kickoff_utc':K,
          'observed_at_utc':'2026-10-25T01:00:00+00:00',
          'market_archive_run_id':'887766','market_archive_sha256':'h',
          'quote':{'spread_home':-3,'home_odds_american':-110,
                  'away_odds_american':-110,'book':'DraftKings'}}}
        self.assertIsNone(pregame(self.p,fake)['market']['ats_research_side'])
        fake[G]['observed_at_utc']='2026-10-21T01:00:00+00:00'
        report=pregame(self.p,fake)
        self.assertEqual(report['market']['ats_research_side'],'HOME')
        self.assertEqual(report['market']['predicted_margin_minus_market_margin_home'],3)

    def test_postgame_compares_original_and_actual_margin_and_does_not_learn_from_one(self):
        source='2026-10-25T03:00:00+00:00'
        report=postgame(self.p,{},final_schedule(),captured_at=source,source_sha256='abcd')
        self.assertEqual(report['status'],'FINAL_REVIEW_RECORDED')
        self.assertEqual(report['observed_final']['home_margin'],-3)
        self.assertEqual(report['margin_error_prediction_minus_actual'],9)
        self.assertFalse(report['future_model_update']['automatic_weight_change'])
        self.assertIn('Absolute miss 9.00',markdown(report))

    def test_postgame_rejects_early_and_incomplete(self):
        x=postgame(self.p,{},final_schedule(),captured_at='2026-10-24T20:00:00+00:00')
        self.assertEqual(x['status'],'FINAL_NOT_YET_ELIGIBLE')
        x=postgame(self.p,{},final_schedule(False),captured_at='2026-10-25T04:00:00+00:00')
        self.assertEqual(x['status'],'FINAL_NOT_AVAILABLE')

    def test_postgame_cannot_change_numeric_team_id(self):
        x=final_schedule();x['home_id']=999
        with self.assertRaisesRegex(ValueError,'team IDs'):
            postgame(self.p,{},x,captured_at='2026-10-25T04:00:00+00:00')

    def test_postgame_with_play_by_play_diagnostics(self):
        x=plays();x=x.loc[x.game_id.eq(103)].copy();x['game_id']=int(G)
        roster=pd.DataFrame([{'team_id':333,'athlete_id':12,'position_abbreviation':'QB'}])
        # legacy labeled play fixtures still verified against schedule game IDs
        sched=final_schedule()
        report=postgame(self.p,{},sched,x,roster,captured_at='2026-10-25T04:00:00+00:00')
        self.assertEqual(report['observed_game_statistics']['home']['status'],'OBSERVED_PBP')
        self.assertEqual(report['observed_game_statistics']['home']['drives'],2)
        self.assertGreater(len(report['candidate_mechanisms_to_test']),0)

    def test_forward_training_gate_requires_prospective_sample(self):
        z=investigate([{'game_id':str(i),'kickoff_utc':f'2026-10-{i+1:02d}T00:00:00+00:00'} for i in range(2)])
        self.assertEqual(z['status'],'INSUFFICIENT_GENUINE_FORWARD_GAMES')
        self.assertFalse(z['deploy_automatically'])

    def test_learning_deduplicates_repeated_grades(self):
        out=Path(self.temp.name)/'grades';out.mkdir()
        d={'schema':'cfb_forward_research_audit_v1',
           'free_schedule_observed_at_utc':'2026-10-27T10:00:00+00:00',
           'scored_games':[{'game_id':G,'kickoff_utc':K,
                            'forecast_generated_at_utc':'2026-10-20T15:10:37+00:00',
                            'predicted_home_margin':6,'actual_home_margin':-3}]}
        (out/'12345.json').write_text(json.dumps(d))
        later={**d,'free_schedule_observed_at_utc':'2026-10-28T10:00:00+00:00',
               'scored_games':[{**d['scored_games'][0],'actual_home_margin':99}]}
        (out/'12346.json').write_text(json.dumps(later))
        rows=observations(out)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['actual_margin'],-3)

    def test_learning_holds_final_test_out_of_candidate_selection(self):
        from datetime import timedelta
        start=datetime(2025,8,1,tzinfo=timezone.utc)
        rows=[]
        for i in range(140):
            kickoff=(start+timedelta(days=i)).isoformat()
            rows.append({'game_id':str(800000+i),'kickoff_utc':kickoff,
                         'projected_margin':float(i%20-10),
                         'actual_margin':float(i%20-10)+1})
        first=investigate(rows)
        self.assertNotEqual(first['status'],'INSUFFICIENT_GENUINE_FORWARD_GAMES')
        altered=[dict(r) for r in rows]
        for r in altered[-25:]:r['actual_margin']+=70
        second=investigate(altered)
        self.assertEqual(first['best_candidate'],second['best_candidate'])
        self.assertEqual(first['validation_mae_improvement'],second['validation_mae_improvement'])
        self.assertEqual(first['passes_validation_gate'],second['passes_validation_gate'])
        self.assertEqual(first['train_coefficients_research_only'],second['train_coefficients_research_only'])
        self.assertFalse(first['deploy_automatically'])

    def test_forward_learner_rejects_forecasts_issued_after_kickoff(self):
        out=Path(self.temp.name)/'grades2';out.mkdir()
        d={'schema':'cfb_forward_research_audit_v1',
           'free_schedule_observed_at_utc':'2026-10-25T10:00:00+00:00',
           'scored_games':[{'game_id':G,'kickoff_utc':K,
                            'forecast_generated_at_utc':'2026-10-25T00:00:00+00:00',
                            'predicted_home_margin':100,'actual_home_margin':-3}]}
        (out/'12345.json').write_text(json.dumps(d))
        self.assertEqual(observations(out),[])

    def test_postgame_provisional_ats_is_based_on_original_quoted_line(self):
        quote={G:{'home':'Alabama','away':'Texas A&M','kickoff_utc':K,
          'observed_at_utc':'2026-10-21T01:00:00+00:00',
          'market_archive_run_id':'887766','market_archive_sha256':'h',
          'quote':{'spread_home':-3,'home_odds_american':-110,
                  'away_odds_american':-110,'book':'DraftKings'}}}
        report=postgame(self.p,quote,final_schedule(),captured_at='2026-10-25T04:00:00+00:00')
        self.assertEqual(report['ATS']['research_side'],'HOME')
        self.assertEqual(report['ATS']['actual_cover_side'],'AWAY')
        self.assertFalse(report['ATS']['covered'])
        self.assertFalse(report['ATS']['original_book_timestamp_verified'])

    def test_no_nfl_imports(self):
        for n in ('matchup_cycle.py','learn_forward.py'):
            tree=ast.parse((Path(__file__).parent/n).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('nfl'))
                    self.assertNotEqual(node.module,'model')

if __name__=='__main__':unittest.main()
