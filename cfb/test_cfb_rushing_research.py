"""Research-only CFB rushing matchup ablation: tests, not fitted edges."""
import copy
import math
import unittest

from cfb.engine import simulate
from cfb.mechanism_build16 import simulate_build16
from cfb.rushing_interaction_research import ABLATIONS,VERSION,experiment
from cfb.research_backtest import project
from cfb.test_cfb_integrated_build15 import histories
from cfb.integrated_scoring import project_integrated


def clocked_histories():
    h=histories()
    for games in h.values():
        for rec in games:
            rec.update({'clock_run_n':25,'clock_pass_n':35,
                        'clock_def_run_n':26,'clock_def_pass_n':34,
                        'clock_time_sum':1750.0,'clock_drive_n':10,
                        'clock_def_time_sum':1820.0,'clock_def_drive_n':10})
    return h


def inputs():
    h=clocked_histories()
    return h,project_integrated(1,2,h,False,'home')['components'],project_integrated(2,1,h,False,'away')['components']


class RushingResearchTests(unittest.TestCase):
    def test_three_declared_ablation_modes_and_no_auto_deployment(self):
        h,home,away=inputs()
        for mode in ABLATIONS:
            v=experiment(1,2,h,home,away,mode)
            self.assertEqual(v['version'],VERSION)
            self.assertEqual(v['status'],'UNFITTED_SCENARIO_READY')
            self.assertFalse(v['coefficients_fitted'])
            self.assertFalse(v['ATS_calibrated'])
            self.assertTrue(v['not_deployed'])
        m=experiment(1,2,h,home,away,'mix')
        p=experiment(1,2,h,home,away,'pace')
        self.assertEqual(m['shared_tempo_delta'],0)
        self.assertEqual(p['ppd_home_delta'],0)
        self.assertEqual(p['ppd_away_delta'],0)

    def test_counterfactual_ppd_counts_only_added_run_share(self):
        h,home,away=inputs()
        home=copy.deepcopy(home)
        home['matched_rushing']['matched_epa_shrunk']=.20
        home['matched_passing']['matched_epa_shrunk']=-.1
        x=experiment(1,2,h,home,away,'combined')
        self.assertGreater(x['home']['rush_share_delta'],0)
        self.assertGreater(x['ppd_home_delta'],0)
        self.assertLessEqual(abs(x['ppd_home_delta']),.120001)
        # Same efficiency gap but no shift in play mix MUST give zero score
        # increment; avoids adding rushing EPA directly a second time.
        self.assertAlmostEqual(x['ppd_home_delta'],
            round(x['home']['old_plays_per_drive']*
                  x['home']['rush_share_delta']*
                  (.20-(-.1)),6),delta=.0002)

    def test_clock_mode_uses_actual_source_seconds_not_play_proxy(self):
        h,home,away=inputs()
        x=experiment(1,2,h,home,away,'combined')
        self.assertTrue(x['measured_clock_duration'])
        self.assertIsNotNone(x['home']['clock'])
        self.assertAlmostEqual(x['home']['clock']['own_seconds_per_drive'],175.0)
        self.assertAlmostEqual(x['home']['clock']['opponent_allowed_seconds_per_drive'],182.0)
        self.assertIn('projected_seconds_per_drive',x['home'])
        self.assertTrue(abs(x['shared_tempo_delta'])<=.75)
        missing=copy.deepcopy(h)
        for row in missing[1]:row.pop('clock_time_sum')
        invalid=experiment(1,2,missing,home,away,'pace')
        self.assertEqual(invalid['status'],'UNAVAILABLE_SOURCE_VERIFIED_DRIVE_CLOCK')
        self.assertNotIn('shared_tempo_delta',invalid)

    def test_missing_input_fails_closed_and_bad_configuration_raises(self):
        h,home,away=inputs()
        del h[1][0]['off_rush_n']
        v=experiment(1,2,h,home,away,'combined')
        self.assertEqual(v['status'],'UNAVAILABLE_REQUIRED_EVIDENCE')
        self.assertNotIn('ppd_home_delta',v)
        with self.assertRaises(ValueError):
            experiment(1,2,h,home,away,'game_outcome_fitted')

    def test_same_seed_and_no_mutation_of_primary(self):
        h=clocked_histories();before=copy.deepcopy(h)
        baseline=project(h[1],h[2],False,'home',False)
        opposing=project(h[2],h[1],False,'away',False)
        frozen=simulate(baseline,opposing,runs=200,seed=77)
        original=simulate_build16(1,2,h,False,runs=200,seed=77)
        same=simulate_build16(1,2,h,False,runs=200,seed=77)
        self.assertEqual(original,same)
        for mode in ABLATIONS:
            a=simulate_build16(1,2,h,False,runs=200,seed=77,rush_research=mode)
            b=simulate_build16(1,2,h,False,runs=200,seed=77,rush_research=mode)
            self.assertEqual(a,b)
            self.assertEqual(a['simulation']['simulations'],200)
            self.assertEqual(a['rush_research']['mode'],mode)
            self.assertTrue(math.isfinite(a['mean_margin_home']))
        self.assertEqual(h,before)
        self.assertEqual(project(h[1],h[2],False,'home',False),baseline)
        self.assertEqual(simulate(baseline,opposing,runs=200,seed=77),frozen)
        self.assertNotIn('rush_research',original)

    def test_no_hidden_lookahead_or_nfl_import(self):
        from pathlib import Path
        src=(Path(__file__).parent/'rushing_interaction_research.py').read_text()
        self.assertNotIn('from model import',src)
        self.assertNotIn('final_score',src)
        self.assertNotIn('actual_margin',src)
        self.assertNotIn('game_id',src)
        self.assertNotIn('market_spread',src)


if __name__=='__main__':
    unittest.main()
