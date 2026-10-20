"""Build 9 regression tests: fresh matchup requested, no old predictions at pregame."""
from __future__ import annotations
import ast
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from cfb.fresh_matchup import (SCHEMA, analyze, resolve_fixture, quote_asof, select_saved_request,
                              review, markdown, requested_learning, run)
from cfb.test_cfb_pipeline import schedule, plays, CUTOFF
from cfb.test_cfb_build6 import odds, frozen
from cfb.archive import pack_forecasts
from cfb.free_odds import capture

RUN_AT='2026-10-19T12:01:00+00:00'
AFTER='2026-10-25T04:00:00+00:00'


def data_fixture(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    sched=schedule().copy()
    sched['home_points']=sched['game_id'].map({100:30,101:28,102:24})
    sched['away_points']=sched['game_id'].map({100:17,101:14,102:10})
    roster=pd.DataFrame([{'team_id':1,'position_abbreviation':'QB','athlete_id':444},
                         {'team_id':2,'position_abbreviation':'QB','athlete_id':555}])
    blob=b'fake snapshot manifest for mocked PBP and schedule'
    (root/'manifest.json').write_bytes(blob)
    manifest={'captured_at_utc':CUTOFF,'datasets':{x:{'sha256':'abc'+x} for x in ('schedule','pbp','roster')}}
    return {'schedule.parquet':sched,'pbp.parquet':plays(),'roster.parquet':roster},manifest


class FreshBuild9(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'snapshot'
        self.tables,self.manifest=data_fixture(self.root)
        def reader(path):return self.tables[Path(path).name].copy()
        self.p1=patch('cfb.fresh_matchup.verify',return_value=self.manifest)
        self.p2=patch('cfb.fresh_matchup.pd.read_parquet',side_effect=reader)
        self.p1.start();self.p2.start();self.addCleanup(self.p1.stop);self.addCleanup(self.p2.stop)

    def make(self,**kwargs):
        return analyze(self.root,'SCHOOL_B','school_a',runs=150,min_games=1,
                       now=RUN_AT,github_run_id='123456789',**kwargs)

    def test_resolves_from_current_schedule_no_old_forecast(self):
        r=self.make()
        self.assertEqual(r['status'],'FRESH_PREGAME_PROJECTION_LOCKED')
        self.assertEqual(r['game_id'],'103')
        self.assertEqual(r['home'],'SCHOOL_A')
        self.assertEqual(r['away'],'SCHOOL_B')
        self.assertEqual(r['analysis_source'],'NEW_VERIFIED_CAPTURED_CFB_STATS_NOT_ARCHIVED_PREDICTION')
        self.assertIn('epa_pass',r['current_observed_statistics']['home']['offense'])
        self.assertEqual(r['github_run_id'],'123456789')
        self.assertIn('No result or forecast from the target game',r['limitations'][-1])
        self.assertIn('Projected score',markdown(r))

    def test_input_statistical_history_respects_24_hour_embargo(self):
        r=self.make()
        self.assertEqual(r['completed_prior_games_home'],2)
        self.assertEqual(r['completed_prior_games_away'],2)
        self.assertNotIn('103',r['prior_game_ids_home'])
        self.assertNotIn('105',r['prior_game_ids_home'])
        self.assertNotIn('104',r['prior_game_ids_home']) # FCS-FCS excluded

    def test_future_target_plays_do_not_change_prediction(self):
        first=self.make()
        pbp=self.tables['pbp.parquet'].copy()
        pbp.loc[pbp.game_id.eq(103),'EPA']=99999.0
        self.tables['pbp.parquet']=pbp
        second=self.make()
        self.assertEqual(first['simulated_margin_home'],second['simulated_margin_home'])
        self.assertEqual(first['current_observed_statistics'],second['current_observed_statistics'])

    def test_no_fake_atsside_when_no_quote(self):
        r=self.make()
        self.assertIsNone(r['market']['ats_research_side'])
        self.assertFalse(r['ATS_calibrated'])
        self.assertEqual(r['grade'],'NOT_CALIBRATED')

    def test_named_forecast_uses_only_available_market_observation(self):
        quote={'home':'SCHOOL_A','away':'SCHOOL_B',
               'kickoff_utc':'2026-10-24T20:00:00+00:00',
               'observed_at_utc':'2026-10-19T11:00:00+00:00',
               'market_archive_run_id':'1234567',
               'market_archive_sha256':'abcd',
               'quote':{'spread_home':-3.5,'home_odds_american':-110,
                        'away_odds_american':-110,'book':'DraftKings'}}
        with patch('cfb.fresh_matchup.quote_asof',return_value=quote):
            r=self.make()
        self.assertEqual(r['market']['ats_research_side'],'HOME')
        self.assertAlmostEqual(r['market']['predicted_margin_minus_market_margin_home'],
                               r['simulated_margin_home']-3.5,places=3)
        self.assertEqual(r['locked_market_evidence'],quote)
        self.assertIsNotNone(r['simulation']['home_cover_probability_raw'])
        self.assertFalse(r['ATS_calibrated'])

    def test_market_asof_chooses_older_valid_quoted_line_not_future_one(self):
        root=Path(self.temp.name);raw=root/'frozen';forecasts=root/'forecasts';markets=root/'markets'
        frozen(raw,run='12345678')
        pack_forecasts(raw,forecasts)
        def response(spread):
            return lambda _url:json.dumps({'items':[odds(spread)]}).encode()
        capture(forecasts,markets,'99999911',now=datetime.fromisoformat('2026-10-20T15:30:00+00:00'),
                fetcher=response(-3.5))
        capture(forecasts,markets,'99999922',now=datetime.fromisoformat('2026-10-22T15:30:00+00:00'),
                fetcher=response(-7.0))
        before=quote_asof(markets,'401856727',datetime.fromisoformat('2026-10-21T00:00:00+00:00'))
        after=quote_asof(markets,'401856727',datetime.fromisoformat('2026-10-22T18:00:00+00:00'))
        self.assertEqual(before['quote']['spread_home'],-3.5)
        self.assertEqual(after['quote']['spread_home'],-7.0)
        self.assertEqual(before['market_archive_run_id'],'99999911')

    def test_insufficient_games_pass_not_fabricated_score(self):
        r=analyze(self.root,'SCHOOL_B','school_a',runs=100,min_games=4,now=RUN_AT)
        self.assertEqual(r['status'],'INSUFFICIENT_PRIOR_OBSERVATIONS_NO_PREDICTION')
        self.assertNotIn('simulated_margin_home',r)
        self.assertIsNone(r['market']['ats_research_side'])

    def test_no_fixture_guessed(self):
        with self.assertRaisesRegex(ValueError,'No upcoming eligible scheduled matchup'):
            self.make(game_id='999')
        with self.assertRaisesRegex(ValueError,'No upcoming eligible scheduled matchup'):
            analyze(self.root,'SCHOOL_A','Michigan',runs=100,min_games=1,now=RUN_AT)

    def test_never_predict_after_kickoff(self):
        self.manifest['captured_at_utc']='2026-10-25T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError,'No upcoming eligible scheduled matchup'):
            analyze(self.root,'SCHOOL_A','SCHOOL_B',runs=100,min_games=1,
                    now='2026-10-25T00:01:00+00:00')

    def test_stale_snapshot_blocks(self):
        with self.assertRaisesRegex(ValueError,'stale'):
            analyze(self.root,'SCHOOL_A','SCHOOL_B',runs=100,min_games=1,
                    now='2026-10-20T12:00:00+00:00')

    def test_no_old_forecast_dependency_even_if_archive_empty(self):
        # Current direct analysis must not even CALL load_all_archives.
        with patch('cfb.archive.load_all_archives',side_effect=RuntimeError('legacy not allowed')):
            r=self.make()
        self.assertEqual(r['status'],'FRESH_PREGAME_PROJECTION_LOCKED')

    def test_postgame_loads_earliest_requested_report(self):
        r=self.make()
        dest=Path(self.temp.name)/'reviews';dest.mkdir()
        (dest/'111111.json').write_text(json.dumps(r))
        later={**r,'prediction_generated_at_utc':'2026-10-19T13:00:00+00:00',
               'simulated_margin_home':12345.0}
        (dest/'222222.json').write_text(json.dumps(later))
        chosen,sha,file=select_saved_request(dest,'school_a','school_b')
        self.assertEqual(file,'111111.json')
        self.assertEqual(chosen['simulated_margin_home'],r['simulated_margin_home'])
        self.assertEqual(sha,hashlib.sha256((dest/file).read_bytes()).hexdigest())
        self.assertEqual(SCHEMA,chosen['schema'])

    def test_postgame_cannot_use_batch_archives_instead_of_saved_request(self):
        with self.assertRaisesRegex(ValueError,'No saved, pre-kickoff USER-REQUESTED'):
            select_saved_request(Path(self.temp.name)/'empty','school_a','school_b')

    def test_postgame_computes_without_reprediction(self):
        r=self.make()
        p=root=Path(self.temp.name)/'reviews';p.mkdir(exist_ok=True)
        old=p/'111111.json';old.write_text(json.dumps(r))
        sha=hashlib.sha256(old.read_bytes()).hexdigest()
        # Final result arrives only in the NEW later-captured game snapshot.
        self.manifest['captured_at_utc']=AFTER
        schedule=self.tables['schedule.parquet'].copy()
        schedule.loc[schedule.game_id.eq(103),'completed']=True
        schedule.loc[schedule.game_id.eq(103),'home_points']=14
        schedule.loc[schedule.game_id.eq(103),'away_points']=35
        self.tables['schedule.parquet']=schedule
        with patch('cfb.fresh_matchup.project',side_effect=AssertionError('postgame reran model')), \
             patch('cfb.fresh_matchup.simulate',side_effect=AssertionError('postgame reran simulation')):
            result=review(r,sha,'111111.json',self.root)
        self.assertEqual(result['status'],'FINAL_REVIEW_RECORDED')
        self.assertEqual(result['locked_original_forecast']['projected_home_margin'],
                         r['simulated_margin_home'])
        self.assertEqual(result['observed_final']['home_margin'],-21)
        self.assertTrue(result['postgame_does_not_rerun_the_original_model'])
        self.assertFalse(result['future_model_update']['automatic_weight_change'])
        x=requested_learning(p,result)
        self.assertEqual(x['completed_forward_games'],1)
        self.assertFalse(x['deploy_automatically'])

    def test_requested_postgame_must_follow_saved_source(self):
        r=self.make()
        with self.assertRaisesRegex(ValueError,'Postgame snapshot'):
            review(r,'sha','111111.json',self.root)

    def test_no_nfl_imports_and_workflow_uses_fresh_path(self):
        x=ast.parse((Path(__file__).parent/'fresh_matchup.py').read_text())
        for n in ast.walk(x):
            if isinstance(n,ast.ImportFrom):
                self.assertFalse((n.module or '').startswith('nfl') or n.module=='model')
        wf=(Path(__file__).parent.parent/'.github/workflows/cfb-free-data.yml').read_text()
        self.assertIn('python -m cfb.fresh_matchup',wf)
        self.assertNotIn('python -m cfb.matchup_cycle \\',wf)

if __name__=='__main__':unittest.main()
