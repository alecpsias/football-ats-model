"""One early-year-fitted CFB rushing score interaction; never modify baseline."""
import copy,unittest
from cfb.failure1_ridge_correction import feature_vector,fit,score,NAMES,VERSION
from cfb.test_cfb_severity_research import decorated
from cfb.mechanism_build16 import simulate_build16
from cfb.failure1_quantitative_validation import grade

class RidgeRushingCorrectionTests(unittest.TestCase):
    def test_fits_only_earlier_year_with_fixed_penalty(self):
        samples=[{'features':[i/100.0 for _ in NAMES],
            'year':2023,'actual_margin':(i%9)-3,
            'reference_margin':(i%9)-3.4} for i in range(1,131)]
        x=fit(samples)
        self.assertEqual(x['fit_year'],2023)
        self.assertEqual(x['n_training_games'],130)
        self.assertEqual(x['ridge_lambda'],250.0)
        self.assertLess(max(abs(v) for v in x['coefficients']),3)
        with self.assertRaises(ValueError):
            fit(samples[:119])

    def test_fitted_delta_changes_simulated_distribution_but_not_default(self):
        h=decorated()
        snapshot=copy.deepcopy(h)
        reference=simulate_build16(1,2,h,False,runs=350,seed=18)
        altered=simulate_build16(1,2,h,False,runs=350,seed=18,
                                failure1_fit_margin_delta=2.0)
        unchanged=simulate_build16(1,2,h,False,runs=350,seed=18)
        self.assertEqual(reference,unchanged)
        self.assertNotEqual(reference['mean_margin_home'],altered['mean_margin_home'])
        self.assertEqual(altered['simulation']['simulations'],350)
        self.assertTrue(altered['not_deployed'])
        self.assertEqual(h,snapshot)
        with self.assertRaises(ValueError):
            simulate_build16(1,2,h,False,runs=350,seed=18,
                            failure1_fit_margin_delta=10.)
        with self.assertRaises(ValueError):
            simulate_build16(1,2,h,False,runs=350,seed=18,
                            failure1_fit_margin_delta=1.,mismatch_research='negative')

    def test_rejects_corrupted_or_missing_model_inputs(self):
        model={'version':VERSION,'feature_names':list(NAMES),
               'coefficients':[0.5]*len(NAMES),'fit_year':2023,
               'n_training_games':130}
        missing={'status':'UNAVAILABLE_REQUIRED_ASOF_RUSH_INPUTS'}
        self.assertIsNone(score(model,missing)['margin_points'])
        v={'status':'AVAILABLE_PREGAME_RUSH_VECTOR','vector':[20.]*len(NAMES)}
        self.assertEqual(score(model,v)['margin_points'],3.0)
        with self.assertRaises(ValueError):
            score({**model,'version':'tampered'},v)

    def test_feature_missing_does_not_invent_opponent_quality(self):
        h=decorated()
        z=feature_vector(1,2,h)
        self.assertEqual(z['status'],'UNAVAILABLE_REQUIRED_ASOF_RUSH_INPUTS')
        self.assertFalse(z.get('no_outcomes_used',False))

    def test_grade_empty_without_winner_claim(self):
        self.assertEqual(grade([]),{'games':0})

if __name__=='__main__':unittest.main()
