"""Failure3 possession-level scoring and regulation-clock regression checks."""
import unittest
import numpy as np
from cfb.drive_sustainability_process import (
    VERSION, parameters, chain_draws, draw_possession, duration_multiplier)
from cfb.game_script import simulate_script
from cfb.test_cfb_engine import P,Q
from cfb.test_cfb_priority3 import history

def fixture(share=.75,chain=.17):
    return {s:{"chain_probability":chain,"explosive_td_share":share,
               "matched_drives":60.,"chain_concentration":18.6}
            for s in ("home","away")}

class Failure3Possession(unittest.TestCase):
    def test_no_duplicate_explosive_points_and_reconstructed_baseline_ppd(self):
        p=dict(P)
        p["_to"]=0.0
        n=150000
        rng=np.random.default_rng(14)
        factor=np.ones(n)
        config=fixture()["home"]
        pts,to,explosive,chain=draw_possession(
            p,rng,np.full(n,.05),np.zeros(n),np.zeros(n),
            np.ones(n,dtype=bool),config,factor)
        self.assertEqual(int((explosive&chain).sum()),0)
        self.assertEqual(int((explosive&to).sum()),0)
        self.assertEqual(int((chain&to).sum()),0)
        self.assertEqual(int(np.count_nonzero(pts)),int(explosive.sum()+chain.sum()))
        expected=min(p["_ppd"]/(p["_td"]*6.965+(1-p["_td"])*3.),.95)*(
            p["_td"]*6.965+(1-p["_td"])*3.)
        self.assertLess(abs(float(pts.mean())-expected),.065)

    def test_chain_variance_is_separate_from_explosive_path(self):
        n=10000
        x=chain_draws(fixture()["home"],np.random.default_rng(3),n)
        self.assertGreater(float(x.std()),.1)
        self.assertTrue(np.all((x>=.4)&(x<=1.6)))
        self.assertLess(abs(float(x.mean())-1.),.04)

    def test_timing_proxies_consume_more_clock_for_chain_than_explosive(self):
        pts=np.array([7,7,0,0])
        turnover=np.array([False,False,True,False])
        explosive=np.array([True,False,False,False])
        d=duration_multiplier(pts,turnover,explosive)
        self.assertLess(d[0],d[1])
        self.assertLess(d[2],d[3])

    def test_same_seed_repeatable_and_possessions_change(self):
        mechanism={"home":{},"away":{},"turnover_field":{"status":"UNAVAILABLE"},
                   "failure3_score_process":fixture()}
        original=simulate_script(P,Q,history(),history(),runs=5000,seed=23)
        corrected=simulate_script(P,Q,history(),history(),runs=5000,seed=23,mechanisms=mechanism)
        repeated=simulate_script(P,Q,history(),history(),runs=5000,seed=23,mechanisms=mechanism)
        self.assertEqual(corrected,repeated)
        self.assertTrue(corrected["failure3_possession_process_active"])
        self.assertFalse(original["failure3_possession_process_active"])
        self.assertNotEqual(corrected["mean_total"],original["mean_total"])
        self.assertGreater(corrected["failure3_explosive_touchdowns_home"],0)
        self.assertGreater(corrected["failure3_chain_scoring_drives_home"],0)
        self.assertTrue(6<=corrected["failure3_possessions_home"]<=19)
        self.assertTrue(6<=corrected["failure3_possessions_away"]<=19)
        self.assertTrue(3300<=corrected["failure3_mean_elapsed_seconds"]<=3800)

    def test_invalid_unsupported_data_fails_closed(self):
        self.assertIsNone(parameters({"status":"MISSING_PRIOR_DRIVE_SUSTAINABILITY"}))
        with self.assertRaises(ValueError):
            parameters({"status":"ACTIVE_BUILD16_UNCALIBRATED_FAILURE3",
              "home":{"status":"AVAILABLE_BOUNDED_UNFITTED",
                      "chain_share_matched":2,"short_explosive_pass_td_share_shrunk":0.5,"matched_drives":55},
              "away":{"status":"AVAILABLE_BOUNDED_UNFITTED",
                      "chain_share_matched":.4,"short_explosive_pass_td_share_shrunk":.5,"matched_drives":55}})

if __name__=="__main__":unittest.main()
