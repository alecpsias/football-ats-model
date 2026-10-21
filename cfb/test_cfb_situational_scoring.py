"""Build16 source-backed 4th-down, finishing, nonthird penalty effects."""
import unittest
import copy
from cfb.situational_scoring import matchup_effects
from cfb.test_cfb_build16 import decorated
from cfb.test_cfb_integrated_build15 import histories
from cfb.mechanism_build16 import simulate_build16

def enriched():
    h=decorated()
    for group in h.values():
        for r in group:
            m=r['mechanism']
            m['fourth_off']={'attempted':6,'converted':5}
            m['fourth_def']={'attempted':6,'converted':1}
            m['scoring_off']={'opportunities':8,'td':5}
            m['scoring_def']={'opportunities':8,'td':2}
            m['penalty_first_down_off']={'awarded':1,'plays':90}
            m['penalty_first_down_def']={'allowed':10,'plays':90}
    return h

class ScoredSituationalEffects(unittest.TestCase):
    def test_missing_evidence_has_zero_effect(self):
        h=histories();r=matchup_effects(h[1],h[2])
        self.assertEqual(r['status'],'NO_SITUATIONAL_EFFECT_AVAILABLE')
        self.assertEqual(r['delta_ppd'],0)
        self.assertEqual(r['delta_td_share'],0)
    def test_three_mechanisms_change_bounded_projection(self):
        h=enriched();r=matchup_effects(h[1],h[2])
        self.assertEqual(r['status'],'OBSERVED_BOUNDED_UNFITTED')
        self.assertTrue(all(r[k]['status']=='OBSERVED_UNFITTED' for k in
             ('fourth_down','finishing','penalty_first_down')))
        self.assertLessEqual(abs(r['delta_ppd']),.085)
        self.assertLessEqual(abs(r['delta_td_share']),.032)
        self.assertFalse(r['fitted'])
    def test_only_challenger_changes_not_input(self):
        h=enriched();baseline=decorated();copy_h=copy.deepcopy(h)
        x=simulate_build16(1,2,baseline,False,runs=300,seed=4)
        y=simulate_build16(1,2,h,False,runs=300,seed=4)
        self.assertNotEqual(x['mean_score_home'],y['mean_score_home'])
        self.assertEqual(h,copy_h)
        self.assertEqual(y,simulate_build16(1,2,h,False,runs=300,seed=4))
        self.assertTrue(y['not_primary'])
    def test_insufficient_defense_does_not_create_finishing_impact(self):
        h=enriched()
        for r in h[2]:r['mechanism']['scoring_def']=None
        self.assertEqual(matchup_effects(h[1],h[2])['finishing']['status'],'UNAVAILABLE')
if __name__=='__main__':unittest.main()
