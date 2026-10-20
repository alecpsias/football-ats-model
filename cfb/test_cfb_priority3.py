"""Build 14: CFB game-script research simulation, frozen-provenance and no promotion."""
from __future__ import annotations
import ast
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cfb.engine import simulate
from cfb.game_script import simulate_script, profile, adverse_scripts, VERSION
from cfb.test_cfb_engine import P,Q
from cfb.test_cfb_build5 import Build5Tests
from cfb.test_cfb_fresh_build9 import data_fixture,RUN_AT
from cfb.test_cfb_build6 import frozen,G,K
from cfb.archive import pack_forecasts
from cfb.forward_monitor import grade
from cfb.fresh_matchup import analyze,review,markdown
from cfb.prospective import freeze


def history(volatility=0):
    rows=[]
    for i in range(6):
        p=25+int(volatility*((-1)**i))
        rows.append({'points':p,'conceded':26+int(volatility*((-1)**(i+1))),
          'drives':11+i%3,'faced':12+i%3,'interceptions':(i%4==0)*2,
          'fumbles_lost':int(i==3),'opponent_id':100+i,'source_game_id':f'prior-{i}'})
    return rows

class Priority3(unittest.TestCase):
    def test_only_prior_game_records_and_versions(self):
        r=profile(history(),history(),P,Q)
        self.assertEqual(r['games_home'],6)
        self.assertEqual(r['games_away'],6)
        self.assertFalse(r['distribution_coefficients_fitted'])
        self.assertIn('not measured persistence',' '.join(r['assumptions']))

    def test_deterministic_and_preserves_primary(self):
        hp=copy.deepcopy(P); ap=copy.deepcopy(Q); hs=history(); aw=history()
        baseline=simulate(hp,ap,runs=400,seed=16)
        x=simulate_script(hp,ap,hs,aw,runs=400,seed=16)
        y=simulate_script(hp,ap,hs,aw,runs=400,seed=16)
        self.assertEqual(x,y)
        self.assertEqual((hp,ap),(P,Q))
        self.assertEqual(hs,history())
        self.assertEqual(x['status'],'UNFITTED_GAME_SCRIPT_CHALLENGER_NOT_PRIMARY')
        self.assertEqual(baseline,simulate(P,Q,runs=400,seed=16))

    def test_distribution_outputs_produced(self):
        r=simulate_script(P,Q,history(),history(),runs=800,seed=44)
        self.assertEqual(r['version'],VERSION)
        self.assertEqual(r['simulations'],800)
        self.assertEqual(len(r['margin_p10_p50_p90']),3)
        self.assertGreater(r['margin_p10_p50_p90'][2],r['margin_p10_p50_p90'][0])
        self.assertGreater(r['simulated_turnovers_home'],0)
        self.assertGreater(r['simulated_turnovers_away'],0)
        self.assertGreater(r['simulated_regulation_drives_per_side'],6)
        self.assertIsNone(r['home_cover_probability_raw'])
        self.assertFalse(r['ATS_calibrated'])
        self.assertTrue(r['not_auto_deployed'])
        for z in r['scenario_distributions'].values():
            self.assertTrue(0<=z['share']<=1)
            self.assertTrue(0<=z['simulations']<=800)

    def test_independent_overtime_off_and_eligible_raw_market_only(self):
        x=simulate_script(P,Q,history(),history(),runs=300,seed=3,allow_overtime=False,spread_home=-3.5,total=48)
        self.assertEqual(x['overtime_summary']['ot1'],0)
        self.assertIsNotNone(x['home_cover_probability_raw'])
        self.assertIsNotNone(x['over_probability_raw'])
        self.assertFalse(x['ATS_calibrated'])

    def test_small_sample_shrink_and_extra_volatility_changes_profile(self):
        low=profile(history(0),history(0),P,Q)
        high=profile(history(20),history(20),P,Q)
        self.assertGreater(high['home_game_efficiency_ppd_sd'],low['home_game_efficiency_ppd_sd'])
        self.assertLessEqual(high['home_game_efficiency_ppd_sd'],.55)

    def test_slow_fast_and_turnover_scripts(self):
        r=simulate_script(P,Q,history(),history(),runs=5000,seed=18)
        s=r['scenario_distributions']
        self.assertGreater(s['slow_possessions']['simulations'],0)
        self.assertGreater(s['fast_possessions']['simulations'],0)
        self.assertGreater(s['home_two_more_turnovers']['simulations'],0)
        self.assertGreater(s['away_two_more_turnovers']['simulations'],0)
        self.assertLess(s['home_two_more_turnovers']['mean_home_margin'],
                        s['away_two_more_turnovers']['mean_home_margin'])

    def test_hypothetical_clock_mechanism_toggle_changes_tail(self):
        with_clock=simulate_script(P,Q,history(),history(),runs=12000,seed=6)
        without_clock=simulate_script(P,Q,history(),history(),runs=12000,seed=6,clock_response=False)
        self.assertGreater(with_clock['scenario_distributions']['runaway_clock_sensitivity_triggered']['share'],0)
        self.assertEqual(without_clock['scenario_distributions']['runaway_clock_sensitivity_triggered']['share'],0)
        self.assertLess(with_clock['simulated_regulation_drives_per_side'],
                        without_clock['simulated_regulation_drives_per_side'])

    def test_stronger_offense_increases_expected_margin(self):
        hp=copy.deepcopy(P);hp['_ppd']+=.8
        a=simulate_script(P,Q,history(),history(),runs=12000,seed=18)
        b=simulate_script(hp,Q,history(),history(),runs=12000,seed=18)
        self.assertGreater(b['mean_margin_home'],a['mean_margin_home']+3)

    def test_missing_turnover_not_imputed_as_zero(self):
        h=history();h[0]['interceptions']=None
        p=profile(h,history(),P,Q)
        self.assertEqual(p['observed_turnover_games_home'],5)

    def test_stress_tests_are_explicit_not_postgame_selected(self):
        r=adverse_scripts(P,Q,history(),history(),runs=1500,seed=25)
        self.assertTrue(r['non_probabilistic_counterfactuals'])
        self.assertTrue(r['not_auto_deployed'])
        self.assertEqual(r,adverse_scripts(P,Q,history(),history(),runs=1500,seed=25))
        scripts=r['scenarios']
        self.assertLess(scripts['home_turnover_adversity']['mean_home_margin'],
                        scripts['reference']['mean_home_margin'])
        self.assertGreater(scripts['away_turnover_adversity']['mean_home_margin'],
                           scripts['reference']['mean_home_margin'])
        self.assertLess(scripts['slower_game']['mean_home_score'],
                        scripts['faster_game']['mean_home_score'])
        self.assertLess(scripts['slower_game']['mean_away_score'],
                        scripts['faster_game']['mean_away_score'])
        self.assertLess(scripts['home_scoring_adversity']['mean_home_margin'],
                        scripts['reference']['mean_home_margin'])

    def test_invalid_inputs_fail_closed(self):
        for invalid in [None,[],[{}], [{'points':float('nan'), 'drives':5,'faced':5,'conceded':10}]]:
            with self.assertRaises(ValueError):simulate_script(P,Q,invalid,history(),runs=100)
        with self.assertRaises(ValueError):simulate_script(P,Q,history(),history(),runs=99)
        with self.assertRaises(ValueError):simulate_script(P,Q,history(),history(),seed=-1)
        with self.assertRaises(ValueError):simulate_script(P,Q,history(),history(),spread_home=float('nan'))

    def test_batch_pregame_archives_script_but_not_in_primary(self):
        with tempfile.TemporaryDirectory() as temp:
            snap=Path(temp)/'snapshot';manifest,tables=Build5Tests.snapshot_data(snap)
            def reader(p):return tables[Path(p).stem].copy()
            with patch('cfb.prospective.verify',return_value=manifest),\
                 patch('cfb.prospective.pd.read_parquet',side_effect=reader):
                freeze(snap,Path(temp)/'out',runs=100,days=7,min_games=2,
                    now='2026-10-19T12:10:00+00:00')
            obj=json.loads((Path(temp)/'out'/'103.json').read_text())
            self.assertEqual(obj['status'],'FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE')
            self.assertEqual(obj['build16_challenger']['version'],'cfb-build16-unified-situational-v1')
            self.assertTrue(obj['build16_not_primary'])
            self.assertEqual(obj['simulated_margin_home'],obj['simulation']['mean_margin_home'])

    def test_fresh_matchup_records_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap=Path(tmp)/'snap';tables,meta=data_fixture(snap)
            def reader(f):return tables[Path(f).name].copy()
            with patch('cfb.fresh_matchup.verify',return_value=meta),\
                 patch('cfb.fresh_matchup.pd.read_parquet',side_effect=reader):
                data=analyze(snap,'School A','School B',runs=100,min_games=2,now=RUN_AT)
            self.assertEqual(data['status'],'FRESH_PREGAME_PROJECTION_LOCKED')
            self.assertEqual(data['build16_challenger']['version'],'cfb-build16-unified-situational-v1')
            self.assertIn('Build 16',markdown(data))
            self.assertEqual(data['simulated_margin_home'],data['simulation']['mean_margin_home'])

    def test_forward_grader_only_uses_frozen_candidate(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp); folder=base/'raw';frozen(folder)
            file=folder/(G+'.json');row=json.loads(file.read_text())
            row['build16_challenger']={'version':'cfb-build16-unified-situational-v1',
                'status':'AVAILABLE_UNFITTED_BUILD16_RESEARCH_ONLY',
                'mean_margin_home':12.,'margin_p10_p50_p90':[-2,12,26]}
            file.write_text(json.dumps(row))
            index=json.loads((folder/'index.json').read_text())
            index['predictions'][0]['sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            (folder/'index.json').write_text(json.dumps(index))
            pack_forecasts(folder,base/'archive')
            observed=pd.DataFrame([{'game_id':int(G),'completed':True,'start_date':K,
                'home_id':333,'away_id':245,'home_team':'Alabama','away_team':'Texas A&M',
                'home_points':40,'away_points':20}])
            with patch('cfb.game_script.simulate_script',side_effect=AssertionError('Never recompute after final')):
                r=grade(base/'archive',base/'no-odds',observed,'2026-10-25T17:00:00+00:00',
                    'https://github.com/sportsdataverse/data','a'*64)
            self.assertEqual(r['graded_final_games'],1)
            self.assertEqual(r['forward_margin']['mae'],14.)
            self.assertEqual(r['build16_challenger_forward']['paired_build16_mae'],8.)
            self.assertEqual(r['build16_challenger_forward']['paired_baseline_mae'],14.)
            self.assertTrue(r['scored_games'][0]['build16_challenger']['margin_p10_p90_covered'])

    def test_postgame_uses_only_saven_original(self):
        source=(Path(__file__).parent/'fresh_matchup.py').read_text()
        self.assertIn("script=saved.get('game_script_research_challenger')",source)
        self.assertIn("'recomputed_after_game':False",source)
        self.assertIn("'game_script_absolute_error'",source)

    def test_no_nfl_import_and_workflow_compatibility(self):
        folder=Path(__file__).parent
        for name in ('game_script.py','fresh_matchup.py','prospective.py','research_backtest.py','forward_monitor.py'):
            t=ast.parse((folder/name).read_text())
            for node in ast.walk(t):
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse(node.module=='model' or (node.module or '').startswith('nfl'))
        yaml=(folder.parent/'.github/workflows/cfb-free-data.yml').read_text()
        self.assertIn('python -m cfb.research_backtest',yaml)
        self.assertIn('python -m cfb.prospective',yaml)

if __name__=='__main__':unittest.main()
