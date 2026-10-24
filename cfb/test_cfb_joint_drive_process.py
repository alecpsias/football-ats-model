"""Regression tests: observed joint-drive coupling inside the only CFB challenger."""
import copy
import unittest
import numpy as np

from cfb.empirical_drive_clock import observe
from cfb.joint_drive_process import matchup_joint_drives, sample_joint
from cfb.mechanism_build16 import simulate_build16
from cfb.test_cfb_empirical_drive_clock import complete_game
from cfb.test_cfb_build16 import decorated
from cfb.test_cfb_failure3_sustainability import rec


def qualified_histories():
    history=decorated()
    raw=complete_game()
    # A mixed run/pass history with verified durations is essential:
    # do not infer rush counts from team rush yardage totals.
    for ix in (1, 2, 5, 6, 9, 10, 13, 14):
        raw.loc[ix,'pass']=False
        raw.loc[ix,'rush']=True
        raw.loc[ix,'pass_td']=False
        raw.loc[ix,'rush_td']=bool(raw.loc[ix,'offense_score_play'])
    clock=observe(raw,raw,10,20,10)
    assert clock['status']=='SOURCE_CLOCK_VERIFIED'
    for entries in history.values():
        for i,item in enumerate(entries):
            item['sustainability']=rec()
            item['drive_clock']=copy.deepcopy(clock)
            item['source_game_id']=str(900000+i)
    return history


class JointDriveTests(unittest.TestCase):
    def test_observed_pairs_are_real_single_drive_records(self):
        raw=complete_game()
        clock=observe(raw,raw,10,20,10)
        self.assertEqual(clock['status'],'SOURCE_CLOCK_VERIFIED')
        self.assertEqual(len(clock['joint_drives']['off']),8)
        self.assertTrue(all(r['seconds']==200 for r in clock['joint_drives']['off']))
        self.assertTrue(all(r['rush_plays']+r['pass_plays']==1
                            for r in clock['joint_drives']['off']))

    def test_matchup_fail_closed_without_joint_history(self):
        h=qualified_histories()
        matched=matchup_joint_drives(1,2,h)
        self.assertEqual(matched['status'],'ACTIVE_SOURCE_JOINT_DRIVE_BOOTSTRAP')
        self.assertTrue(matched['score_mean_preserved'])
        self.assertGreaterEqual(matched['home']['total_observed_drives'],24)
        h[1][0]['drive_clock'].pop('joint_drives')
        self.assertEqual(matchup_joint_drives(1,2,h)['status'],
                         'ACTIVE_SOURCE_JOINT_DRIVE_BOOTSTRAP')
        for x in h[1]:x['drive_clock'].pop('joint_drives',None)
        self.assertEqual(matchup_joint_drives(1,2,h)['status'],
                         'INSUFFICIENT_VERIFIED_JOINT_DRIVES')

    def test_ensemble_preserves_rush_clock_correlation_and_state_dependence(self):
        pool={'pools':{'stalled':[
           {'seconds':80,'rush_plays':1,'pass_plays':9,'rush_share':.1},
           {'seconds':110,'rush_plays':1,'pass_plays':9,'rush_share':.1},
           {'seconds':350,'rush_plays':9,'pass_plays':1,'rush_share':.9},
           {'seconds':390,'rush_plays':9,'pass_plays':1,'rush_share':.9}]}}
        n=10000
        mask=np.ones(n,dtype=bool)
        slow=sample_joint(np.random.default_rng(21),pool,'stalled',mask,
                          np.full(n,.9))
        fast=sample_joint(np.random.default_rng(21),pool,'stalled',mask,
                          np.full(n,.1))
        self.assertGreater(float(slow[1].mean()),float(fast[1].mean())+1.)
        self.assertGreater(float(slow[0].mean()),float(fast[0].mean())+20.)
        self.assertTrue(np.all(~slow[3]))
        self.assertEqual(float(slow[0][slow[1]==9].min()),350.)

    def test_qualified_build16_process_is_active_and_reproducible(self):
        h=qualified_histories()
        frozen=copy.deepcopy(h)
        control=simulate_build16(1,2,h,False,runs=5000,seed=341,
                                 joint_drive_enabled=False)
        active=simulate_build16(1,2,h,False,runs=5000,seed=341,
                                joint_drive_enabled=True)
        again=simulate_build16(1,2,h,False,runs=5000,seed=341,
                               joint_drive_enabled=True)
        self.assertEqual(active,again)
        self.assertEqual(h,frozen)
        self.assertFalse(control['simulation']['joint_drive_active'])
        self.assertTrue(active['simulation']['joint_drive_active'])
        self.assertTrue(active['joint_drive_active_by_default'])
        self.assertEqual(active['simulation']['simulations'],5000)
        self.assertLessEqual(active['simulation']['joint_drive_avg_possession_seconds_home']+
                             active['simulation']['joint_drive_avg_possession_seconds_away'],3600.01)
        self.assertGreater(active['simulation']['joint_drive_avg_rush_plays_home'],0)
        self.assertGreater(active['simulation']['joint_drive_avg_pass_plays_away'],0)

    def test_original_path_unchanged_if_joint_records_missing(self):
        h=qualified_histories()
        for items in h.values():
            for rec_ in items:
                rec_['drive_clock'].pop('joint_drives')
        a=simulate_build16(1,2,h,False,runs=200,seed=78,
                           joint_drive_enabled=False)
        b=simulate_build16(1,2,h,False,runs=200,seed=78,
                           joint_drive_enabled=True)
        self.assertFalse(b['joint_drive_active_by_default'])
        self.assertEqual(a['simulation'],b['simulation'])
        with self.assertRaises(ValueError):
            simulate_build16(1,2,h,False,runs=200,seed=78,
                             joint_drive_enabled='yes')


if __name__=='__main__':
    unittest.main()
