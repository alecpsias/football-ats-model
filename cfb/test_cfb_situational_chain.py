"""Regression checks for opponent-adjusted early/third/distance chain mode."""
import unittest
from cfb.chain_interaction_research import experiment

def history(off=True):
    p='chain_off_' if off else 'chain_def_'
    return {
        p+'early_rush_success_yes':40,p+'early_rush_success_n':90,
        p+'early_rush_plays':90,p+'early_offense_plays':180,
        p+'third_manageable_yes':30,p+'third_attempts':80,
        p+'sustained_drives':30,p+'observed_drives':60,
        p+'early_success_yes':90,p+'early_success_n':180,
        p+'third_short_yes':12,p+'third_short_n':20,
        p+'third_medium_yes':8,p+'third_medium_n':20,
        p+'third_long_yes':5,p+'third_long_n':40,
    }

def row():
    return dict(history(True),**history(False))

class SituationalChainTests(unittest.TestCase):
    def test_matched_source_only(self):
        hs={1:[row() for _ in range(3)],2:[row() for _ in range(3)]}
        result=experiment(1,2,hs,'situational')
        self.assertEqual(result['status'],'UNFITTED_SCENARIO_READY')
        self.assertAlmostEqual(result['ppd_home_delta'],0.,places=5)
        self.assertAlmostEqual(result['ppd_away_delta'],0.,places=5)

    def test_missing_fails_closed(self):
        hs={1:[row() for _ in range(3)],2:[row() for _ in range(3)]}
        del hs[1][0]['chain_off_third_short_yes']
        self.assertEqual(experiment(1,2,hs,'situational')['status'],
                         'UNAVAILABLE_REQUIRED_HISTORICAL_CHAIN_EVIDENCE')

    def test_increased_opponent_third_allowance(self):
        hs={1:[row() for _ in range(3)],2:[row() for _ in range(3)]}
        for r in hs[2]:
            r['chain_def_third_short_yes']=18
            r['chain_def_third_medium_yes']=15
            r['chain_def_third_long_yes']=18
            r['chain_def_early_success_yes']=120
        result=experiment(1,2,hs,'situational')
        self.assertEqual(result['status'],'UNFITTED_SCENARIO_READY')
        self.assertGreater(result['ppd_home_delta'],0.)
        self.assertLessEqual(result['ppd_home_delta'],.055)

if __name__=='__main__':
    unittest.main()
