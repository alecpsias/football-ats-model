"""Failure-1 mandatory audit is required before CFB ATS confidence."""
import copy
import unittest
from cfb.mandatory_rushing_audit import audit,gate
from cfb.test_cfb_severity_research import decorated

class RequiredRushingAuditTests(unittest.TestCase):
    def test_severe_rushing_matchup_and_negative_prevention_audited(self):
        h=decorated();before=copy.deepcopy(h)
        screen=audit(1,2,h)
        self.assertTrue(screen['mandatory_for_every_cfb_pregame_prediction'])
        self.assertEqual(screen['status'],'SEVERE_RUSH_MISMATCH_REQUIRES_5000_RUN_STRESS')
        self.assertTrue(screen['requires_full_rush_stress'])
        self.assertTrue(screen['home']['severe_signal'])
        self.assertFalse(screen['away']['severe_signal'])
        self.assertGreater(screen['home']['early_rush_success']['trials'],35)
        self.assertIsNotNone(screen['home']['matched_negative_run_difference'])
        self.assertTrue(screen['home']['own_trailing_9plus_rush_success'])
        self.assertEqual(screen['required_adverse_script_simulations_each'],5000)
        self.assertEqual(h,before)

    def test_incomplete_rush_data_stays_hold_and_no_invented_stats(self):
        screen=audit(1,2,{1:[],2:[]})
        self.assertEqual(screen['status'],'REQUIRES_HUMAN_REVIEW_INCOMPLETE_RUSH_EVIDENCE')
        self.assertIsNone(screen['home']['early_rush_success'])
        x=gate(screen,None,None)
        self.assertEqual(x['ats_recommendation_status'],'HOLD_INCOMPLETE_RUSH_EVIDENCE')
        self.assertIsNone(x['stress_ats_conflict_with_central'])

    def test_shallow_stress_does_not_qualify_as_5000_draws(self):
        screen=audit(1,2,decorated())
        x=gate(screen,{'simulation':{'simulations':5000},
                       'mean_margin_home':3.,
                       'adverse_sensitivity':{'counterfactuals':{
                           str(k):{'mean_home_margin':i} for k,i in enumerate(range(6))},
                           'simulations_per_scenario':400}}, {'home_spread':-2.5})
        self.assertEqual(x['ats_recommendation_status'],
                         'HOLD_MANDATORY_5000_RUN_STRESS_NOT_VERIFIED')

    def test_complete_adverse_scripts_produce_hold_if_side_flips(self):
        screen=audit(1,2,decorated())
        b={'simulation':{'simulations':5000},'mean_margin_home':3.,
           'adverse_sensitivity':{'simulations_per_scenario':5000,
               'counterfactuals':{f'script_{i}':{'mean_home_margin':m}
                                  for i,m in enumerate((4.,3.,2.,1.,6.,7.))}}}
        x=gate(screen,b,{'home_spread':-2.5})
        self.assertEqual(x['ats_recommendation_status'],'HOLD_CONFLICTING_ADVERSE_SCRIPTS')
        self.assertEqual(x['stress_runs_per_scenario_verified'],5000)
        self.assertTrue(x['stress_ats_conflict_with_central'])
        self.assertEqual(x['stress_home_margin_min'],1.)
        self.assertEqual(x['stress_home_margin_max'],7.)
        self.assertIsNone(x.get('new_scoring_coefficient'))
        y=gate(screen,b,None)
        self.assertEqual(y['ats_recommendation_status'],
                         'HOLD_NO_VERIFIED_SPREAD_OR_MANUAL_REVIEW')

    def test_other_matchup_sos_ega_gap_also_triggers_even_if_partial(self):
        x=audit(1,2,{}, {'sos_rush_epa_differential':.13})
        self.assertTrue(x['requires_full_rush_stress'])
        self.assertIn('OPPONENT_QUALITY_ADJUSTED_RUSH_EPA_DIFFERENTIAL',x['trigger_reasons'])
        self.assertEqual(x['status'],'REQUIRES_HUMAN_REVIEW_INCOMPLETE_RUSH_EVIDENCE')

if __name__=='__main__':unittest.main()
