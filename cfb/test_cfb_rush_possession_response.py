"""Verify physical rushing possession chain source, clock and 5000-run opt-in."""
import copy,unittest
from cfb.rush_possession_response import signal,fit,response
from cfb.mechanism_build16 import simulate_build16
from cfb.test_cfb_severity_research import decorated

def observed():
    h=decorated()
    for team,rows in h.items():
        for r in rows:
            own=team==1
            r.update({
            'chain_off_early_rush_success_yes':27 if own else 18,
            'chain_off_early_rush_success_n':45,
            'chain_def_early_rush_success_yes':25 if own else 34,
            'chain_def_early_rush_success_n':45,
            'chain_off_early_rush_plays':45,
            'chain_off_early_offense_plays':90,
            'chain_off_sustained_drives':7 if own else 3,
            'chain_off_observed_drives':12,
            'chain_def_sustained_drives':4 if own else 9,
            'chain_def_observed_drives':12,
            'chain_off_third_manageable_yes':4 if own else 7,
            'chain_off_third_attempts':12,
            'chain_def_third_manageable_yes':8 if own else 3,
            'chain_def_third_attempts':12,
            'possession_observed_time_sec':2100 if own else 1700,
            'possession_verified_drive_n':12,
            'possession_def_observed_time_sec':1700 if own else 2100,
            'possession_def_verified_drive_n':12})
    return h

class PossessionTests(unittest.TestCase):
    def test_opponent_chains_come_from_actual_prior_observed_samples(self):
        h=observed();old=copy.deepcopy(h)
        e=signal(1,2,h)
        self.assertEqual(e['status'],'AVAILABLE_PREGAME_DRIVE_CONTINUATION')
        self.assertGreater(e['home_continuation_delta'],0)
        self.assertEqual(e['home']['clock_data_status'],'SOURCE_VERIFIED')
        self.assertTrue(e['clock_both_sides_qualified'])
        self.assertLessEqual(abs(e['shared_verified_clock_drives_delta']),.65)
        self.assertEqual(h,old)

    def test_missing_clock_does_not_invent_possession_duration(self):
        h=observed()
        for rows in h.values():
            for r in rows:
                for k in list(r):
                    if k.startswith('possession_'):r.pop(k)
        e=signal(1,2,h)
        self.assertEqual(e['status'],'AVAILABLE_PREGAME_DRIVE_CONTINUATION')
        self.assertFalse(e['clock_both_sides_qualified'])
        self.assertEqual(e['shared_verified_clock_drives_delta'],0)

    def test_missing_chain_no_numerical_imputation(self):
        h=observed()
        for r in h[1]:r.pop('chain_off_sustained_drives')
        self.assertEqual(signal(1,2,h)['status'],'UNAVAILABLE_VERIFIED_DRIVE_CHAIN')

    def test_fit_prior_year_only_and_cap_no_composite_experiments(self):
        h=observed();snap=signal(1,2,h)
        data=[{'signal':snap,'year':2023,
          'reference_expected_drives':11.,
          'reference_margin':3.+i/5,
          'actual_margin':4.+i/5} for i in range(85)]
        model=fit(data);z=response(model,snap)
        self.assertEqual(model['train_games'],85)
        self.assertEqual(z['status'],'AVAILABLE_FROZEN_POSSESSION_RESPONSE')
        self.assertLessEqual(abs(z['home_ppd_shift']),.22)
        self.assertLessEqual(abs(z['away_ppd_shift']),.22)
        original=simulate_build16(1,2,h,False,runs=350,seed=17)
        alternative=simulate_build16(1,2,h,False,runs=350,seed=17,possession_response=z)
        self.assertNotEqual(original['mean_margin_home'],alternative['mean_margin_home'])
        self.assertEqual(original,simulate_build16(1,2,h,False,runs=350,seed=17))
        self.assertEqual(alternative['simulation']['simulations'],350)
        with self.assertRaises(ValueError):
            simulate_build16(1,2,h,False,runs=350,seed=17,
                             possession_response=z,failure1_fit_margin_delta=1.)
        with self.assertRaises(ValueError):
            simulate_build16(1,2,h,False,runs=350,seed=17,
              possession_response={**z,'shared_clock_drives_shift':.5,'real_clock_both_sides_valid':False})

if __name__=='__main__':unittest.main()
