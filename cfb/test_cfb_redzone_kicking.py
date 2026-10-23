"""Red-zone / exact distance kicker regression: Build16 invariants."""
import unittest
import numpy as np
from cfb.redzone_kicking import (matchup,draw_missed_field_goals,
                                  gross_up_for_misses)
from cfb.game_script import simulate_script
from cfb.engine import _resolve_overtime,CFB_RULES

def record(*, made=(1,1,1,1), defense_rz=4):
    return {"redzone_trips":10,"redzone_tds":6,
       "def_redzone_trips":10,"def_redzone_tds":defense_rz,
       "drives":13,"faced":13,"points":30.,"conceded":24.,
       "special_teams":{"fg_distance_events":[
          [28.,made[0]],[38.,made[1]],[45.,made[2]],[55.,made[3]]]}}
def hs(defense_rz=4,made=(1,1,1,1)):
    return {1:[record(made=made) for _ in range(3)],
            2:[record(defense_rz=defense_rz) for _ in range(3)]}

class RedzoneKickingTests(unittest.TestCase):
    def test_all_perfect_kicks_regressed_below_one(self):
        a=matchup(1,2,hs())
        self.assertEqual(a["status"],"ACTIVE_VERIFIED_EVIDENCE")
        self.assertGreater(a["field_goals"]["predicted_make_rate"],0.)
        self.assertLess(a["field_goals"]["predicted_make_rate"],1.)
        self.assertLess(a["fg_regression_ppd_delta"],0.)
        self.assertEqual(len(a["field_goals"]["distance_bins"]),5)
        self.assertAlmostEqual(sum(b["weight"] for b in
            a["field_goals"]["distance_bins"]),1.,places=7)

    def test_bad_kicking_regresses_upward(self):
        a=matchup(1,2,hs(made=(0,0,0,0)))
        self.assertGreater(a["field_goals"]["predicted_make_rate"],0.)
        self.assertGreater(a["fg_regression_ppd_delta"],0.)

    def test_opponent_specific_redzone_td_probability(self):
        low=matchup(1,2,hs(defense_rz=2))
        high=matchup(1,2,hs(defense_rz=9))
        self.assertLess(low["redzone"]["matched_redzone_td_probability"],
                        high["redzone"]["matched_redzone_td_probability"])
        self.assertLess(low["redzone_ppd_delta"],high["redzone_ppd_delta"])

    def test_no_source_does_not_fabricate_kicks(self):
        h={1:[{"drives":12}],2:[{"drives":12}]}
        result=matchup(1,2,h)
        self.assertEqual(result["status"],"UNAVAILABLE_LEAVE_EXISTING_MODEL_UNCHANGED")
        self.assertIsNone(result["field_goals"])
        self.assertEqual(result["delta_ppd"],0.)

    def test_fg_miss_is_not_offensive_td_or_turnover(self):
        fg={"predicted_make_rate":.1,
            "distance_bins":[{"weight":1.,"probability":.1,"distance":45.}]}
        outcomes=np.array([7,3,3,0,3,6,3]*100,dtype=np.int16)
        new,miss=draw_missed_field_goals(outcomes,fg,np.random.default_rng(13))
        self.assertTrue(np.all(new[outcomes==7]==7))
        self.assertTrue(np.all(new[outcomes==6]==6))
        self.assertTrue(np.all(new[outcomes==0]==0))
        self.assertTrue(np.all(new[miss]==0))
        self.assertGreater(miss.sum(),100)

    def test_expected_gross_to_net_ppd_normalization(self):
        p={"_ppd":2.3,"_td":.65}
        fg={"predicted_make_rate":.6}
        gross=gross_up_for_misses(p,fg)
        self.assertGreater(gross["_ppd"],p["_ppd"])
        base_avg=.65*6.965+.35*3.
        net_avg=.65*6.965+.35*3.*.6
        self.assertAlmostEqual(gross["_ppd"]*net_avg/base_avg,2.3,places=8)

    def test_regulation_and_overtime_are_kick_aware(self):
        h=hs()
        me=matchup(1,2,h)
        fg=me["field_goals"]
        mechanisms={"home":{},"away":{},"turnover_field":{"status":"UNAVAILABLE"},
                     "redzone_kicking":{"home":{"field_goals":fg},
                                        "away":{"field_goals":fg}}}
        proj={"_ppd":2.15,"_to":.12,"_td":.58,"expected_drives":12.}
        result=simulate_script(proj,proj,h[1],h[2],runs=5000,seed=411,
                               mechanisms=mechanisms)
        self.assertTrue(result["field_goal_distance_regression_active"])
        self.assertGreater(result["simulated_missed_field_goals_regulation_home"],0.)
        self.assertGreater(result["simulated_missed_field_goals_regulation_away"],0.)
        self.assertEqual(result["simulations"],5000)

if __name__=="__main__": unittest.main()
