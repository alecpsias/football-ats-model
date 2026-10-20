"""Contract tests: separate CFB engine, NCAA OT structure, reproducibility, replay."""
import ast
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from cfb.engine import (CFB_RULES, simulate, _resolve_overtime,
                        _two_point_rate, CFBEngineError)
from cfb.research_backtest import metrics, project
from cfb.strict_forecast import forecast

P={'_ppd':2.4,'_to':.10,'_td':.62,'expected_drives':11.5}
Q={'_ppd':2.1,'_to':.15,'_td':.58,'expected_drives':11}

class IndependentEngineTests(unittest.TestCase):
    def test_never_imports_nfl_module(self):
        root=Path(__file__).resolve().parent
        for name in ('engine.py','research_backtest.py','strict_forecast.py'):
            module=ast.parse((root/name).read_text())
            for node in ast.walk(module):
                if isinstance(node,ast.Import):
                    self.assertNotIn('model',[x.name for x in node.names],name)
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse(node.module == 'model' or (node.module or '').startswith('nfl'),name)
    def test_deterministic(self):
        a=simulate(P,Q,runs=1200,seed=4)
        b=simulate(P,Q,runs=1200,seed=4)
        self.assertEqual(a,b)
        self.assertNotEqual(a['mean_margin_home'],simulate(P,Q,runs=1200,seed=5)['mean_margin_home'])
        self.assertEqual(a['engine'],'cfb.engine.v1')
    def test_market_not_fabricated(self):
        a=simulate(P,Q,runs=1000,seed=8)
        self.assertIsNone(a['home_cover_probability_raw'])
        self.assertIsNone(a['away_cover_probability_raw'])
        self.assertEqual(a['status'],'CFB_RESEARCH_UNCALIBRATED_NO_WAGER_GRADE')
    def test_signed_spread_settlement(self):
        a=simulate(P,Q,spread_home=-2.5,total=50,runs=2000,seed=3)
        self.assertAlmostEqual(a['home_cover_probability_raw']+a['away_cover_probability_raw']+a['push_probability_raw'],1.,places=4)
        self.assertIsNotNone(a['over_probability_raw'])
    def test_sensitive_to_ppd_matchup(self):
        weaker=dict(P,_ppd=1.0)
        stronger=dict(P,_ppd=4.0)
        a=simulate(stronger,weaker,runs=2500,seed=10)
        b=simulate(weaker,stronger,runs=2500,seed=10)
        self.assertGreater(a['mean_margin_home'],10)
        self.assertLess(b['mean_margin_home'],-10)
    def test_invalid_inputs_fail_closed(self):
        for p in (dict(P,_ppd=None),dict(P,_ppd=float('nan')),dict(P,_to=1.0),dict(P,expected_drives=0)):
            with self.assertRaises(CFBEngineError):simulate(p,Q)
        with self.assertRaises(CFBEngineError):simulate(P,Q,rules={'nfl_pressure_rate':.33})
        with self.assertRaises(CFBEngineError):simulate(P,Q,spread_home=float('nan'))
    def test_global_cfb_rules_not_changed(self):
        before=dict(CFB_RULES)
        simulate(P,Q,dict(CFB_RULES,drive_noise_sd=.4),runs=200)
        self.assertEqual(before,dict(CFB_RULES))
        with self.assertRaises(TypeError): CFB_RULES['league_ppd']=1
    def test_no_automatic_nfl_calibration(self):
        result=simulate(P,Q,runs=500)
        self.assertNotIn('grade',result)
        self.assertTrue('UNCALIBRATED' in result['status'])
    def test_first_second_third_overtime_structure(self):
        hs=np.zeros(10000,dtype=np.int32)
        ass=np.zeros(10000,dtype=np.int32)
        rng=np.random.default_rng(99)
        result=_resolve_overtime(P,P,dict(CFB_RULES),rng,hs,ass)
        self.assertEqual(result['regulation_ties'],10000)
        self.assertEqual(result['ot1'],10000)
        self.assertGreater(result['ot2'],0)
        self.assertGreater(result['ot3_or_later'],0)
        self.assertLess(result['unresolved_after_limit'],10)
        self.assertFalse(np.any(hs<0) or np.any(ass<0))
    def test_mandatory_two_point_changes_2ot_scoring(self):
        # Force both teams into 2OT and check TDs add 6 or 8, never 7.
        # Patch draws: first period both 0, second both touchdowns, third 2pt.
        from cfb import engine
        calls=[0]
        def draw(p,r,rng,active,shortfield,noise,multiplier=1):
            calls[0]+=1
            if calls[0] <= 2: return np.zeros(len(active),dtype=np.int16),np.zeros(len(active),bool)
            return np.full(len(active),7,dtype=np.int16),np.zeros(len(active),bool)
        h=np.zeros(500,dtype=np.int32); a=np.zeros(500,dtype=np.int32)
        rules=dict(CFB_RULES)
        with patch.object(engine,'_draw_possession',side_effect=draw):
            engine._resolve_overtime(P,Q,rules,np.random.default_rng(1),h,a)
        # 2OT touchdowns get only 6 or 8 plus later 2pt periods.
        self.assertTrue(set(h.tolist()).issubset(set(range(6,80,2))))
        self.assertTrue(set(a.tolist()).issubset(set(range(6,80,2))))
    def test_future_replay_not_ats_claim(self):
        good={'actual_margin':7,'baseline_margin':5,'epa_candidate_margin':6}
        result=metrics([good],'epa_candidate_margin')
        self.assertEqual(result['games'],1)
        self.assertEqual(result['mae'],1)
        self.assertNotIn('ats_accuracy',result)
    def test_project_cfb_parameters_only(self):
        r={'points':28,'conceded':21,'drives':11,'faced':11,'off_pass_sum':4,'off_pass_n':30,
           'off_rush_sum':1,'off_rush_n':20,'def_pass_sum':2,'def_pass_n':30,
           'def_rush_sum':1,'def_rush_n':20,'interceptions':1,'fumbles_lost':0}
        p=project([r,r,r],[r,r,r],False,'home',True)
        self.assertGreater(p['expected_drives'],0)
        self.assertEqual(set(p),set(P))
    def test_replay_end_to_end_with_fixture_and_chronological_history(self):
        """Mocked parquet fixture: check actual orchestration, not real 2026 data."""
        from tempfile import TemporaryDirectory
        from cfb.test_cfb_pipeline import schedule as make_schedule, plays as make_plays, CUTOFF
        from cfb.research_backtest import replay
        import pandas as pd
        fixtures=make_schedule()
        fixtures['home_points']=21
        fixtures['away_points']=14
        fixtures.loc[fixtures.game_id==103,'completed']=True
        fixtures.loc[fixtures.game_id==103,'start_date']='2026-10-17T20:00:00Z'
        data=make_plays()
        roster=pd.DataFrame({'team_id':[1,2,3,4],
                             'position_abbreviation':['QB']*4,
                             'athlete_id':[101,201,301,401]})
        def fake_read(path):
            p=str(path)
            return data if p.endswith('/pbp.parquet') else roster if p.endswith('/roster.parquet') else fixtures
        with TemporaryDirectory() as tmp:
            with patch('cfb.research_backtest.verify', return_value={'captured_at_utc':CUTOFF}), \
                 patch('cfb.research_backtest.pd.read_parquet', side_effect=fake_read):
                summary=replay(tmp,Path(tmp)/'output',runs=200,min_games=1)
            self.assertEqual(summary['engine'],'cfb.engine.v1')
            self.assertEqual(summary['chronologically_evaluated_games'],1)
            self.assertEqual(summary['holdout_games'],1)
            self.assertEqual(summary['nfl_code_imported'],False)
            self.assertTrue((Path(tmp)/'output'/'margin_replay.json').exists())

    def test_wrapper_rejects_late_data_and_fcs(self):
        item={'id':'1','league':'CFB','home_division':'fbs','away_division':'fbs',
              'kickoff_utc':'2026-11-01T17:00:00Z','data_as_of_utc':'2026-10-20T13:00:00Z',
              'home':dict(P,projection_source='VERIFIED_CFB_PRIOR_HISTORY'),
              'away':dict(Q,projection_source='VERIFIED_CFB_PRIOR_HISTORY')}
        self.assertEqual(forecast(item,runs=200)['model'],'cfb.engine.v1')
        with self.assertRaises(CFBEngineError):forecast(dict(item,data_as_of_utc='2026-11-02T00:00:00Z'))
        with self.assertRaises(CFBEngineError):forecast(dict(item,home_division='fcs',away_division='fcs'))
        with self.assertRaises(CFBEngineError):forecast(dict(item,market={'spread_home':-4,'quoted_at_utc':'2026-10-21T13:00:00Z'}))

if __name__=='__main__':unittest.main()
