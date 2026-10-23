"""No hindsight/double-EPA and genuine Build16 scoring-path tests."""
import copy,unittest
import numpy as np
from cfb.continuous_rush_sos_residual import (
    VERSION,NAMES,TRAIN_YEAR,fit,score,feature_vector,MAX_ABS_DELTA)
from cfb.test_cfb_severity_research import decorated
from cfb.mechanism_build16 import simulate_build16

class ContinuousRushSOSTests(unittest.TestCase):
    def test_fit_one_earlier_year_and_never_use_later_outcomes(self):
        v=np.ones(len(NAMES))
        rows=[]
        for i in range(100):
            f=(v*np.sin(i/8.)+i/160).tolist()
            rows.append({'year':2024,'game_id':str(30000+i),
                'simulations':5000,'feature_vector':f,
                'reference_margin':2.,'actual_margin':2.+f[0]*1.2})
        m=fit(rows)
        self.assertEqual(m['training_season'],TRAIN_YEAR)
        self.assertEqual(m['training_games'],100)
        self.assertFalse(m['promotion_authorized'])
        self.assertEqual(len(m['coefficients']),len(NAMES))
        z=score(m,{'status':'AVAILABLE_PREGAME_CONTINUOUS_RUSH_SOS',
                   'vector':rows[0]['feature_vector']})
        self.assertEqual(z['status'],'FROZEN_EARLIER_SEASON_RESIDUAL_AVAILABLE')
        self.assertLessEqual(abs(z['margin_delta_points']),MAX_ABS_DELTA)
        with self.assertRaises(ValueError):fit([{**r,'year':2026} for r in rows])
        with self.assertRaises(ValueError):fit(rows[:79])
        with self.assertRaises(ValueError):fit(rows+[rows[0]])
        with self.assertRaises(ValueError):score({**m,'version':'wrong'},{
            'status':'AVAILABLE_PREGAME_CONTINUOUS_RUSH_SOS',
            'vector':rows[0]['feature_vector']})

    def test_missing_opponent_quality_fails_closed(self):
        hist=decorated()
        for rows in hist.values():
            for r in rows:r.pop('opponent_id',None)
        f=feature_vector(1,2,hist)
        self.assertEqual(f['status'],'UNAVAILABLE_PREGAME_RUSH_OR_OPPONENT_QUALITY')
        self.assertTrue(f['missing_instead_of_zero'])

    def test_strength_sign_is_relative_not_absolute(self):
        h=decorated()
        f=feature_vector(1,2,h)
        if f['status']=='AVAILABLE_PREGAME_CONTINUOUS_RUSH_SOS':
            reverse=feature_vector(2,1,h)
            self.assertEqual(reverse['status'],f['status'])
            for v,w in zip(f['vector'],reverse['vector']):
                self.assertAlmostEqual(v,-w,places=5)

    def test_same_seed_real_sim_changes_margin_without_changing_reference(self):
        hist=decorated()
        copy_before=copy.deepcopy(hist)
        original=simulate_build16(1,2,hist,False,runs=5000,seed=2026)
        alt=simulate_build16(1,2,hist,False,runs=5000,seed=2026,
            continuous_rush_sos_residual_delta=3.4)
        repeated=simulate_build16(1,2,hist,False,runs=5000,seed=2026)
        self.assertEqual(original,repeated)
        self.assertEqual(alt['simulation']['simulations'],5000)
        self.assertGreater(alt['simulation']['mean_margin_home'],
                           original['simulation']['mean_margin_home'])
        self.assertEqual(hist,copy_before)
        self.assertTrue(alt['continuous_rush_sos_residual_research']['not_default'])
        self.assertTrue(alt['not_deployed'])
        for invalid in (float('nan'),float('inf'),7.):
            with self.assertRaises(ValueError):
                simulate_build16(1,2,hist,False,runs=200,seed=1,
                    continuous_rush_sos_residual_delta=invalid)
        with self.assertRaises(ValueError):
            simulate_build16(1,2,hist,False,runs=200,seed=1,
                continuous_rush_sos_residual_delta=1.0,mismatch_research='negative')

if __name__=='__main__':unittest.main()
