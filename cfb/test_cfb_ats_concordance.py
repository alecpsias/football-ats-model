"""ATS baseline-market concordance: pregame-only input, no invented quote."""
import unittest
from cfb.ats_concordance import assess

class ATSConcordanceTests(unittest.TestCase):
    def test_nc_state_stanford_frozen_pregame_edge_is_conflict(self):
        # Home Stanford +6.5: both margins are PRE-game frozen from Oct 23.
        x=assess(-6.462,-8.869,6.5)
        self.assertEqual(x['status'],'HOLD_CONFLICTING_BASELINE_CHALLENGER_SPREAD_SIDES')
        self.assertAlmostEqual(x['baseline_edge_vs_spread_points'],.038)
        self.assertAlmostEqual(x['challenger_edge_vs_spread_points'],-2.369)
        self.assertTrue(x['material_model_disagreement'])
        self.assertTrue(x['model_spread_side_conflict'])
        self.assertFalse(x['automatic_bet_recommendation'])

    def test_when_native_quote_missing_holds_even_with_gap(self):
        x=assess(-6.462,-8.869)
        self.assertEqual(x['status'],'HOLD_NO_VERIFIED_QUOTE_AND_MATERIAL_MODEL_GAP')
        self.assertIsNone(x['baseline_edge_vs_spread_points'])

    def test_baseline_no_edge_neutral_leans_remain_hold(self):
        x=assess(3.4,6.5,-3.5)
        self.assertEqual(x['status'],'HOLD_CHALLENGER_ONLY_EDGE_BASELINE_AT_MARKET')

    def test_both_models_same_direction_still_not_a_calibrated_bet(self):
        x=assess(8,9,-4.5)
        self.assertEqual(x['status'],'MODELS_NOT_DISCORDANT_NO_CALIBRATED_BET')
        self.assertFalse(x['calibrated_ats_probability'])
        self.assertFalse(x['automatic_bet_recommendation'])

    def test_small_edge_remains_hold(self):
        self.assertEqual(assess(4,4.5,-4.)['status'],'HOLD_SMALL_CHALLENGER_MARGIN_BUFFER')

    def test_reject_nonfinite_and_invented_spread_values(self):
        for args in ((float('nan'),5.,3.),
                     (5.,float('inf'),3.),
                     (5.,4.,80.),
                     (5.,4.,float('nan'))):
            with self.assertRaises(ValueError):
                assess(*args)

if __name__=='__main__':unittest.main()
