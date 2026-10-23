"""Regression fixture for the *frozen* 2026 Liberty–Kennesaw ATS miss.

Does not refit model coefficients or use postgame outcomes as pregame inputs.
Prevents later reports from silently changing which pregame model was graded,
from misattributing official scoring and from declaring Failure1 stress ran.
"""
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).parent/'postgame_audits/2026_10_22_liberty_kennesaw.json'

class LockedPostmortem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=json.loads(ROOT.read_text())
        cls.p=cls.x['frozen_prediction']
        cls.a=cls.x['observed']
        cls.e=cls.x['evaluation']

    def test_unmodified_original_5000_simulation_and_line(self):
        self.assertEqual(self.x['fixture']['game_id'],'401871056')
        self.assertEqual(self.p['build16']['simulations'],5000)
        self.assertEqual(self.p['baseline']['simulations'],5000)
        self.assertEqual(self.p['reference_market_kennesaw_plus'],10.5)
        self.assertEqual(self.p['build16']['kennesaw_plus_10_5_cover_share_raw'],.6972)
        self.assertLess(self.p['source_generated_utc'],
                        self.p['estimated_official_kickoff_utc'])

    def test_official_points_and_rushing_box_reconciled(self):
        a=self.a
        self.assertEqual(a['score_liberty']-a['score_kennesaw'],a['margin_liberty'])
        self.assertEqual(a['score_liberty']+a['score_kennesaw'],a['total'])
        self.assertEqual(sum(a['possession_seconds'].values()),3600)
        self.assertEqual(a['rush_yards']['liberty']-a['rush_yards']['kennesaw'],153)
        self.assertEqual(sum(a['turnovers_lost'].values()),2)
        self.assertEqual(a['kennesaw_interceptions'],2)
        self.assertEqual(sum(a['point_sequences_after_lost_possessions']),20)
        self.assertEqual(a['failed_fourth_downs_kennesaw'],2)

    def test_actual_minus_frozen_score_errors_not_posthoc_revisions(self):
        p=self.p['build16'];a=self.a;e=self.e
        self.assertAlmostEqual(a['margin_liberty']-p['margin_liberty'],
                               e['build16_liberty_margin_underprediction'],places=3)
        self.assertAlmostEqual(a['score_kennesaw']-p['score_kennesaw'],
                               -e['build16_kennesaw_score_overprediction'],places=3)
        self.assertAlmostEqual(a['total']-p['total'],
                               -e['build16_total_overprediction'],places=3)
        self.assertAlmostEqual(a['margin_liberty']-self.p['baseline']['margin_liberty'],
                               e['baseline_liberty_margin_underprediction'],places=3)
        self.assertTrue(e['actual_margin_outside_pregame_build16_10th_90th_interval'])
        self.assertFalse(e['ats_direction_correct'])

    def test_predeclared_mismatch_screen_indeed_did_not_trigger(self):
        z=self.p['source_information']
        self.assertFalse(z['full_5000_rush_stress_triggered'])
        self.assertEqual(z['nonsevere_stress_runs_per_counterfactual'],400)
        self.assertLess(z['rush_joint_severity_away'],z['rush_joint_trigger'])
        self.assertLess(abs(z['sos_rush_epa_difference_home']),
                        z['sos_rush_epa_severe_threshold'])
        self.assertEqual(z['failure3_clock_status'],
                         'FALLBACK_LEGACY_BUILD16_NO_VERIFIED_CLOCKS')
        self.assertEqual(z['failure2_status'],
                         'ACTIVE_LINKED_DEFENSE_AND_SPECIAL_TEAMS_SCORING')

    def test_no_hindsight_retraining(self):
        x=self.x
        self.assertTrue(x['guardrails']['dont_leak_target_result_into_model_features'])
        self.assertTrue(x['guardrails']['baseline_preserved'])
        self.assertTrue(x['guardrails']['one_CFB_challenger_only'])
        self.assertTrue(x['guardrails']['no_scoring_coefficients_tuned_by_this_postmortem'])
        self.assertTrue(x['guardrails']['do_not_claim_validated_improvement'])

if __name__=='__main__':
    unittest.main()
