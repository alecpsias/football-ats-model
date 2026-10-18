"""Local stand-ins test the official nflreadpy method contracts; NO network access."""
import json
import sys
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nflverse_connector import collect, verify, standardize, CollectionError
from run_snapshot_ingest import run
from test_ingest import fake_data, CUTOFF


class PolarsLike:
    def __init__(self,df): self.df=df
    def to_pandas(self): return self.df.copy()


class FakeNflreadpy:
    def __init__(self):
        self.pbp,self.schedule,_,self.roster,self.cfg=fake_data()
        self.roster=self.roster.rename(columns={'player_id':'gsis_id'})
        self.counts={}
    def load_pbp(self,seasons):
        self.counts['pbp']=seasons
        return PolarsLike(self.pbp)
    def load_schedules(self,seasons):
        self.counts['schedule']=seasons
        return PolarsLike(self.schedule)
    def load_rosters_weekly(self,seasons):
        self.counts['roster']=seasons
        return PolarsLike(self.roster)
    def load_player_stats(self,seasons):
        self.counts['player_stats']=seasons
        return PolarsLike(pd.DataFrame({'season':[seasons[0]],'player_id':['p1']}))


class TestNflverseConnector(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=FakeNflreadpy()
    @staticmethod
    def when(): return datetime(2026,10,17,12,0,tzinfo=timezone.utc)
    def download(self, after=False, optional=False):
        moment=(lambda:datetime(2026,10,19,12,0,tzinfo=timezone.utc)) if after else self.when
        return collect(2026,self.root/'snapshot',nfl_module=self.source,capture_now=moment,include_player_stats=optional)
    def test_nflreadpy_calls_methods_with_season_filter(self):
        manifest=self.download(optional=True)
        self.assertEqual(manifest['status'],'COLLECTED')
        self.assertEqual(self.source.counts,{'pbp':[2026],'schedule':[2026],'roster':[2026],'player_stats':[2026]})
        self.assertEqual(verify(self.root/'snapshot')['status'],'VERIFIED')
        self.assertEqual(verify(self.root/'snapshot',CUTOFF)['pregame_eligibility'],'CAPTURED_BEFORE_CUTOFF')
        self.assertTrue((self.root/'snapshot'/'pbp_2026.csv.gz').exists())
    def test_capture_after_cutoff_cannot_count_as_pregame(self):
        self.download(after=True)
        self.assertEqual(verify(self.root/'snapshot',CUTOFF)['pregame_eligibility'],'NOT_VERIFIED_PREGAME')
    def test_no_overwrite(self):
        self.download()
        with self.assertRaises(CollectionError):
            self.download()
    def test_file_tampering_is_detected(self):
        self.download()
        with (self.root/'snapshot'/'schedule_2026.csv.gz').open('ab') as f: f.write(b'garbage')
        check=verify(self.root/'snapshot')
        self.assertEqual(check['status'],'BLOCKED')
    def test_missing_metric_rejected_not_imputed(self):
        self.source.pbp=self.source.pbp.drop(columns=['epa'])
        report=self.download()
        self.assertEqual(report['status'],'INCOMPLETE')
        self.assertIn('pbp missing essential',report['datasets']['pbp']['error'])
    def test_bridge_with_free_data_reports_missing_pressure(self):
        self.download()
        cfg=self.source.cfg
        path=self.root/'input_config.json'
        path.write_text(json.dumps(cfg),encoding='utf8')
        game,audit=run(self.root/'snapshot',path,self.root/'out'/'prediction')
        self.assertEqual(audit['status'],'BLOCKED_MISSING_INPUTS_OR_ASOF')
        self.assertTrue(any(x.endswith('pressure_allowed') for x in audit['missing_fields']))
        self.assertFalse(any('source_available_at_utc: after' in s for s in audit['asof_issues']))
        self.assertEqual(audit['manifest_file_verification'],'VERIFIED')
    def test_bridge_cannot_fake_historical_pregame(self):
        self.download(after=True)
        cfg=self.source.cfg
        cfg['pbp_source_available_at_utc']=CUTOFF # forged config timestamp must not be used
        config=self.root/'cfg.json'
        config.write_text(json.dumps(cfg))
        game,audit=run(self.root/'snapshot',config,self.root/'out'/'prediction')
        self.assertEqual(audit['status'],'BLOCKED_MISSING_INPUTS_OR_ASOF')
        self.assertTrue(any('pbp_source_available_at_utc' in s for s in audit['asof_issues']))
    def test_free_mode_does_not_create_runnable_prediction(self):
        self.download()
        cfg_path=self.root/'cfg.json'
        cfg_path.write_text(json.dumps(self.source.cfg))
        game,audit=run(self.root/'snapshot',cfg_path,self.root/'out'/'prediction')
        self.assertFalse((self.root/'out'/'prediction.game.json').exists())
        self.assertTrue((self.root/'out'/'prediction.INCOMPLETE.json').exists())
    def test_incomplete_snapshot_refuses_run(self):
        self.source.roster=self.source.roster.drop(columns=['position'])
        self.download()
        cfg_path=self.root/'cfg.json'
        cfg_path.write_text(json.dumps(self.source.cfg))
        with self.assertRaises(ValueError):
            run(self.root/'snapshot',cfg_path,self.root/'out'/'prediction')

if __name__=='__main__':unittest.main()
