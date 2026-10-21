"""NFL v0.9 must score situational matchups without changing NFL baseline."""
import copy
import unittest
from model import MODEL_RULES
from mechanism_research import extract_mechanisms,candidate_simulation
from test_mechanism_research import fixture,projections,CUTOFF
from nfl_situational_scoring import scored_situations

class SituationalNFLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plays,sched=fixture()
        cls.profile=extract_mechanisms(plays,sched,CUTOFF,('GB','DAL'))
    def test_missing_fourth_and_retain_observed_finishing(self):
        z=scored_situations(self.profile,'GB','DAL')
        self.assertEqual(z['fourth']['status'],'UNAVAILABLE')
        self.assertIn(z['finishing']['status'],('UNAVAILABLE','OBSERVED_UNFITTED'))
        self.assertFalse(z['fitted'])
    def test_three_scored_components_are_opponent_matched(self):
        p=copy.deepcopy(self.profile)
        o=p['offense']['GB'];d=p['defense']['DAL'];lg=p['league']
        o.update(fourth_attempts=20,fourth_failed=3,scoring_opportunities=35,
                 scoring_opportunity_touchdowns=24,penalty_nonthird_firsts_earned=3,snaps_taken=280)
        d.update(fourth_attempts=20,fourth_failed=18,scoring_opportunities=35,
                 scoring_opportunity_touchdowns=8,penalty_nonthird_firsts_allowed=22,snaps_faced=280)
        lg.update(fourth_attempts=60,fourth_failed=25,scoring_opportunities=100,
                  scoring_opportunity_touchdowns=50,penalty_nonthird_firsts_earned=18,snaps_taken=600)
        z=scored_situations(p,'GB','DAL')
        self.assertEqual(z['status'],'OBSERVED_BOUNDED_UNFITTED')
        self.assertTrue(all(z[k]['status']=='OBSERVED_UNFITTED' for k in
            ('fourth','finishing','penalty_first_down')))
        self.assertLessEqual(abs(z['total_delta_ppd']),.085)
        self.assertLessEqual(abs(z['total_delta_td_share']),.032)
        h,a=projections();copy_h=copy.deepcopy(h)
        x=candidate_simulation(h,a,self.profile,MODEL_RULES['NFL'],1000,19)
        y=candidate_simulation(h,a,p,MODEL_RULES['NFL'],1000,19)
        self.assertNotEqual(x['score_home'],y['score_home'])
        self.assertEqual(h,copy_h)
        self.assertFalse(y['promoted_to_primary'])
        self.assertEqual(y,candidate_simulation(h,a,p,MODEL_RULES['NFL'],1000,19))
    def test_no_defense_opportunities_means_no_finishing_effect(self):
        p=copy.deepcopy(self.profile);p['defense']['DAL']['scoring_opportunities']=0
        self.assertEqual(scored_situations(p,'GB','DAL')['finishing']['delta_td_share'],0)
if __name__=='__main__':unittest.main()
