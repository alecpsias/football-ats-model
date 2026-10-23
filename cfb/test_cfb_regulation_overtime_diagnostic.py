"""Regression tests for exact regulation vs overtime decomposition of Build16."""
import unittest
from cfb.game_script import simulate_script

class RegulationOvertimeDiagnosticsTests(unittest.TestCase):
    def test_disjoint_regulation_and_overtime_totals(self):
        hist=[{'drives':12,'faced':12,'points':28.,'conceded':28.,
               'interceptions':1,'fumbles_lost':0} for _ in range(4)]
        team={'_ppd':2.2,'_to':.10,'_td':.60,'expected_drives':12.}
        a=simulate_script(team,team,hist,hist,runs=5000,seed=401862807)
        b=simulate_script(team,team,hist,hist,runs=5000,seed=401862807)
        self.assertEqual(a,b)
        self.assertGreaterEqual(a['mean_overtime_points_added'],0.)
        self.assertAlmostEqual(a['regulation_mean_score_home']+
                               a['regulation_mean_score_away']+
                               a['mean_overtime_points_added'],
                               a['mean_total'],delta=.003)
        self.assertAlmostEqual(a['regulation_mean_margin_home'],0.,delta=2.)
        self.assertEqual(a['regulation_tie_probability_raw'],
                         round(a['overtime_summary']['regulation_ties']/5000,5))
        self.assertGreater(a['regulation_tie_probability_raw'],0.)
        self.assertEqual(a['simulations'],5000)

if __name__=='__main__':unittest.main()
