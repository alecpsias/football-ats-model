"""Regression tests: observed rush-control, opposing chain fragility, true 5000 draws."""
import copy
import unittest
import numpy as np
from cfb.test_cfb_severity_research import decorated
from cfb.coupled_rush_possessions import (
    parameters,draw_coupled_state,MAX_SCORING_FRACTION)
from cfb.mechanism_build16 import simulate_build16

def qualified():
    hist=decorated()
    for tid,rows in hist.items():
        for i,r in enumerate(rows):
            # Real games have distinct rushing success rates; avoid an
            # artificial zero-variance fixture that cannot exercise the
            # coupled scoring draw.
            swing=(-7,0,6)[i%3]
            r['mismatch_off_early_rush_success_yes']=max(0,min(35,
                r['mismatch_off_early_rush_success_yes']+swing))
            r['mismatch_def_early_rush_success_yes']=max(0,min(35,
                r['mismatch_def_early_rush_success_yes']-swing))
            # Synthetic *prior* source records test mechanics only. Real
            # GitHub pairing must obtain these from actual source PBP.
            home=tid==1
            r['sustainability']={'status':'OBSERVED',
                'off':{'drives':12,'sustained':7 if home else 2,
                    'low_chain_non_td':2 if home else 8,
                    'early_success':26 if home else 13,'early_plays':44,
                    'touchdown_drives':4 if home else 2,
                    'short_explosive_pass_tds':1,
                    'pass_explosive_plays':2,'pass_plays':30},
                'def':{'drives':12,'sustained':6 if home else 3,
                    'low_chain_non_td':3 if home else 6,
                    'early_success':25 if home else 17,'early_plays':44,
                    'touchdown_drives':3,
                    'short_explosive_pass_tds':1,
                    'pass_explosive_plays':2,'pass_plays':30}}
    return hist

class CoupledRushTests(unittest.TestCase):
    def test_opponent_strength_and_two_way_fragility_are_observed(self):
        hist=qualified();original=copy.deepcopy(hist)
        m=parameters(1,2,hist)
        self.assertEqual(m['status'],'ACTIVE_OBSERVED_COUPLED_RUSH_CHAIN_SCRIPTS')
        self.assertGreater(m['home_two_way_strength'],0)
        self.assertEqual(m['away_two_way_strength'],0.)
        self.assertGreater(m['relative_matchup_rush_advantage'],0)
        self.assertGreater(m['home']['chain_fragility_opponent'],0)
        self.assertFalse(m['fitted'])
        self.assertEqual(hist,original)

    def test_missing_rush_observations_do_not_get_zero_filled(self):
        hist=qualified()
        for row in hist[1]:
            row.pop('mismatch_off_early_rush_n')
        evidence=parameters(1,2,hist)
        self.assertEqual(evidence['status'],'INSUFFICIENT_PRIOR_RUSH_CONTROL')
        other=qualified()
        for row in other[2]:
            row.pop('sustainability')
        self.assertEqual(parameters(1,2,other)['status'],
                         'INSUFFICIENT_MATCHED_CHAIN_HISTORY')

    def test_matched_source_bootstrap_zero_center_and_opposite_direction(self):
        config=parameters(1,2,qualified())
        draw=draw_coupled_state(config,np.random.default_rng(7159),5000)
        h=draw['home_ppd_fraction'];a=draw['away_ppd_fraction']
        self.assertTrue(np.allclose(h,-a))
        self.assertLess(abs(float(h.mean())),.01)
        self.assertGreater(float(np.std(h)),0)
        self.assertLessEqual(float(np.max(np.abs(h))),MAX_SCORING_FRACTION)
        self.assertFalse(draw['fitted_coefficient'])

    def test_default_build16_actual_same_seed_5000_has_coupled_results(self):
        hist=qualified();snapshot=copy.deepcopy(hist)
        old=simulate_build16(1,2,hist,False,runs=5000,seed=11571,
            coupled_rush_possessions=False,continuous_rush_sos_enabled=False)
        new=simulate_build16(1,2,hist,False,runs=5000,seed=11571,
            coupled_rush_possessions=True,continuous_rush_sos_enabled=False)
        again=simulate_build16(1,2,hist,False,runs=5000,seed=11571,
            coupled_rush_possessions=True,continuous_rush_sos_enabled=False)
        self.assertEqual(hist,snapshot)
        self.assertEqual(new,again)
        self.assertEqual(new['simulation']['simulations'],5000)
        self.assertEqual(new['coupled_rush_possession_interaction']['status'],
            'ACTIVE_OBSERVED_COUPLED_RUSH_CHAIN_SCRIPTS')
        self.assertEqual(old['coupled_rush_possession_interaction']['status'],
            'DISABLED_FOR_PAIRED_RUSH_POSSESSION_REFERENCE')
        self.assertTrue(new['simulation']['coupled_rush_chain_active'])
        self.assertFalse(old['simulation']['coupled_rush_chain_active'])
        self.assertIn('home_rushing_control_tail',new['simulation']['scenario_distributions'])
        self.assertNotIn('home_rushing_control_tail',old['simulation']['scenario_distributions'])
        self.assertNotEqual(new['simulation']['mean_margin_home'],
                            old['simulation']['mean_margin_home'])
        self.assertFalse(new['simulation']['coupled_rush_chain_empirical_clock_compounding'])
        self.assertTrue(new['simulation']['coupled_rush_chain_clock_unverified_fallback'])
        self.assertFalse(new['coupled_rush_possession_interaction']['scoring_ppd_added_in_advance'])

    def test_invalid_boolean_and_research_mode_ablation_safety(self):
        with self.assertRaises(ValueError):
            simulate_build16(1,2,qualified(),False,runs=100,seed=7,
                             coupled_rush_possessions='yes')
        # No independent research ablation can silently combine with it.
        r=simulate_build16(1,2,qualified(),False,runs=100,seed=7,
            coupled_rush_possessions=True,mismatch_research='negative')
        self.assertEqual(r['coupled_rush_possession_interaction']['status'],
                         'DISABLED_DURING_OTHER_RESEARCH_ABLATION')

if __name__=='__main__':unittest.main()
