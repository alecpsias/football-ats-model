"""Paired score metrics and no-bookmaker/no-outcome model data contracts."""
import unittest
from cfb.rushing_study import paired_metrics,EXPERIMENTS,study


class PairedRushingStudyTests(unittest.TestCase):
    def test_paired_comparisons_same_cohort_and_reproducible_bootstrap(self):
        rows=[]
        for i in range(20):
            margin=3.0 if i%2 else -4.0
            experiments={}
            for mode,err in (('reference',4),('mix',2),('pace',3),('combined',1)):
                experiments[mode]={'margin':margin+err,'total':45+err,
                    'win_probability_raw':.6 if margin>0 else .35,
                    'p10_p50_p90':[-12,0,20],'drives':11,
                    'seed':i,'simulations':5000}
            rows.append({'actual_home_margin':margin,'actual_total':45,'experiments':experiments})
        x=paired_metrics(rows)
        self.assertEqual(x,paired_metrics(rows))
        self.assertEqual(set(EXPERIMENTS),set(('reference','mix','pace','combined')))
        self.assertEqual(x['reference']['margin_mae'],4.0)
        self.assertEqual(x['combined']['margin_mae'],1.0)
        self.assertEqual(x['combined']['paired_mae_improvement_vs_reference'],3.0)
        self.assertEqual(x['combined']['bootstrap95_improvement_interval'],[3.,3.])
        self.assertNotIn('ats_cover_probability',x['combined'])

    def test_empty_cohort_and_hard_run_limit(self):
        self.assertEqual(paired_metrics([]),{'games':0})
        with self.assertRaises(ValueError):
            study('/not/a/snapshot','/tmp/unused.json',runs=4000)
        with self.assertRaises(ValueError):
            study('/not/a/snapshot','/tmp/unused.json',runs=5000,max_games=0)

if __name__=='__main__':
    unittest.main()
