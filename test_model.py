import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import InputError, run, validate_game

SAMPLE=json.loads((Path(__file__).resolve().parents[1]/'examples'/'synthetic_matchup.json').read_text())

class ModelChecks(unittest.TestCase):
    def test_deterministic_and_audit(self):
        a=run(copy.deepcopy(SAMPLE), 3000, seed=12, perform_stress=False)
        b=run(copy.deepcopy(SAMPLE), 3000, seed=12, perform_stress=False)
        self.assertEqual(a['simulation'], b['simulation'])
        self.assertEqual(a['grade'],'UNRATED')
        self.assertFalse(a['audit']['probabilities_empirically_calibrated'])
        self.assertTrue(a['audit']['opponent_strength_adjusted'])
        self.assertFalse(a['audit']['stress_tests'])

    def test_probability_conservation(self):
        out=run(copy.deepcopy(SAMPLE), 3000, seed=4, perform_stress=False)
        s=out['simulation']
        self.assertAlmostEqual(s['home_cover_probability_raw']+s['away_cover_probability_raw']+s['push_probability'],1,places=4)
        self.assertTrue(0 <= s['home_win_probability_raw'] <= 1)

    def test_missing_required_stat_fails_hard(self):
        g=copy.deepcopy(SAMPLE)
        g['home']['offense']['epa_pass']=None
        with self.assertRaises(InputError): validate_game(g)

    def test_no_future_information(self):
        g=copy.deepcopy(SAMPLE)
        g['away']['stat_as_of_utc']='2026-10-18T20:26:00Z'
        with self.assertRaises(InputError): validate_game(g)

    def test_wrong_units_fails(self):
        g=copy.deepcopy(SAMPLE)
        g['away']['offense']['interception_rate']=2.5
        with self.assertRaises(InputError): validate_game(g)

    def test_personnel_changes_need_evidence(self):
        g=copy.deepcopy(SAMPLE)
        g['away']['personnel']['pass_epa_shift']=.1
        g['away']['personnel']['source']=''
        with self.assertRaises(InputError):validate_game(g)

    def test_sos_affects_projections(self):
        base=run(copy.deepcopy(SAMPLE), 3000, seed=1, perform_stress=False)
        g=copy.deepcopy(SAMPLE)
        g['home']['schedule']['opp_def_pass_epa_allowed_avg']=-.15  # tougher historical schedule
        nxt=run(g, 3000, seed=1, perform_stress=False)
        self.assertGreater(nxt['projection_home']['matched_pass_epa'],base['projection_home']['matched_pass_epa'])

    def test_qb_rushing_sensitivity(self):
        base=run(copy.deepcopy(SAMPLE), 3000, seed=1, perform_stress=False)
        g=copy.deepcopy(SAMPLE)
        g['home']['personnel']['qb_rush_multiplier']=1.5
        g['home']['personnel']['notes']='Hypothetical game plan adjustment'
        g['home']['personnel']['source']='SYNTHETIC DEMO'
        new=run(g, 3000, seed=1, perform_stress=False)
        self.assertGreater(new['projection_home']['qb_rushes_per_game'],base['projection_home']['qb_rushes_per_game'])
        self.assertNotEqual(new['projection_home']['raw_points_per_drive'],base['projection_home']['raw_points_per_drive'])

    def test_full_stress_is_not_blank(self):
        out=run(copy.deepcopy(SAMPLE), 3000, seed=1)
        self.assertTrue(out['audit']['stress_tests'])
        self.assertIn('home_adverse_offensive_script',out['stress_tests']['scenario_results'])
        self.assertEqual(len(out['stress_tests']['spread_home_sensitivity']),5)

    def test_first_half_optional(self):
        g=copy.deepcopy(SAMPLE)
        g['market']['first_half']={
            'spread_home':-2.5,'home_odds_american':-110,'away_odds_american':-110,
            'total':23.5,'source':'SYNTHETIC','as_of_utc':'2026-10-18T18:55:00Z'}
        out=run(g,3000,seed=1,perform_stress=False)
        self.assertTrue(out['audit']['first_half_market_evaluated'])
        self.assertTrue(out['first_half']['simulation']['mean_total'] < out['simulation']['mean_total'])

if __name__=='__main__': unittest.main()
