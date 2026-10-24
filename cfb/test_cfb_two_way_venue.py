"""Improvement 1: two-sided source-backed route and location effects."""
import copy
import unittest
from cfb.engine import CFB_RULES, simulate
from cfb.research_backtest import project
from cfb.test_cfb_integrated_build15 import histories,record
from cfb.two_way_venue import matchup, _venue_feature, LEAGUE_VENUE_EPA
from cfb.mechanism_build16 import simulate_build16


def venue_history(venue_signal=True):
    h=histories()
    for team in (1,2):
        rows=[]
        for venue in ("home","away"):
            for opponent in (3,4,3,4):
                item=record(opponent,
                    off_pass=.24 if team==1 else -.1,
                    off_rush=-.08 if team==1 else .24,
                    def_pass=.02,def_rush=.03)
                item["venue"]=venue
                # Stand-in source-verified observed per-game home and away
                # effects. Same opponents at both venues: SOS cancels exactly.
                v=1 if venue=="home" else -1
                for kind in ("pass","rush"):
                    o_n=item["off_"+kind+"_n"]
                    d_n=item["def_"+kind+"_n"]
                    main=LEAGUE_VENUE_EPA
                    item["off_"+kind+"_sum"]+=v*main*o_n
                    item["def_"+kind+"_sum"]-=v*main*d_n
                    if venue_signal and team==1:
                        item["off_"+kind+"_sum"]+=v*.12*o_n
                        item["def_"+kind+"_sum"]-=v*.09*d_n
                item["qb_rushes"]=10 if team==2 else 3
                item["qb_rush_epa"]=.4 if team==2 else .07
                rows.append(item)
        h[team]=rows
    return h

class TwoWayVenueTests(unittest.TestCase):
    def test_opponent_adjusted_venue_signal_matches_true_direction(self):
        h=venue_history()
        x=_venue_feature(1,h,"off")
        y=_venue_feature(1,h,"def")
        self.assertEqual(x["status"],"SOURCE_ADJUSTED_VENUE_CONTRAST")
        self.assertGreater(x["half_home_minus_away_epa"],0)
        self.assertLess(y["half_home_minus_away_epa"],0)
        self.assertEqual(x["home_games"],4)
        self.assertTrue(x["league_home_field_removed"])

    def test_no_double_count_of_existing_league_hfa(self):
        h=venue_history(venue_signal=False)
        for role in ("off","def"):
            for team in (1,2):
                x=_venue_feature(team,h,role)
                self.assertEqual(x["status"],"SOURCE_ADJUSTED_VENUE_CONTRAST")
                self.assertLess(abs(x["half_home_minus_away_epa"]),.00001)

    def test_neutral_location_bypasses_venue_but_retains_route_competition(self):
        h=venue_history()
        x=matchup(1,2,h,False)
        n=matchup(1,2,h,True)
        self.assertTrue(x["venue_component_active"])
        self.assertFalse(n["venue_component_active"])
        self.assertEqual(n["venue"]["status"],"NEUTRAL_SITE_NO_TEAM_VENUE_ADJUSTMENT")
        self.assertEqual(n["delta_ppd_home"],-n["delta_ppd_away"])
        self.assertTrue(n["home_route"]["main_epa_counted_only_by_existing_integrated_scorer"])
        self.assertTrue(n["away_route"]["qb_rushing_used_as_historical_route_not_personnel"])

    def test_missing_venue_history_fails_closed(self):
        h=histories()
        x=matchup(1,2,h,False)
        self.assertFalse(x["venue_component_active"])
        self.assertEqual(x["delta_ppd_home"],-x["delta_ppd_away"])
        self.assertEqual(x["venue"]["home_off"]["status"],
                         "MISSING_INDEPENDENT_OPPONENTS_OR_VENUES")
        for item in h[1]:item["venue"]="home"
        y=matchup(1,2,h,False)
        self.assertFalse(y["venue_component_active"])

    def test_actual_five_thousand_draw_replay_changes_score_not_baseline(self):
        h=venue_history()
        before=copy.deepcopy(h)
        hp=project(h[1],h[2],False,"home",False)
        ap=project(h[2],h[1],False,"away",False)
        base=simulate(hp,ap,runs=5000,seed=876)
        old=simulate_build16(1,2,h,False,runs=5000,seed=876,
                             two_way_venue_enabled=False)
        new=simulate_build16(1,2,h,False,runs=5000,seed=876,
                             two_way_venue_enabled=True)
        self.assertEqual(new["simulation"]["simulations"],5000)
        self.assertEqual(old["simulation"]["simulations"],5000)
        self.assertNotEqual(new["mean_margin_home"],old["mean_margin_home"])
        self.assertTrue(new["two_way_venue_active_by_default"])
        self.assertTrue(new["two_way_venue_improvement1"]["venue_component_active"])
        self.assertFalse(old["two_way_venue_active_by_default"])
        self.assertEqual(new,simulate_build16(1,2,h,False,runs=5000,seed=876,
                             two_way_venue_enabled=True))
        self.assertEqual(simulate(hp,ap,runs=5000,seed=876),base)
        self.assertEqual(h,before)

    def test_toggle_requires_bool(self):
        with self.assertRaises(ValueError):
            simulate_build16(1,2,histories(),False,runs=50,
                             two_way_venue_enabled=1)

if __name__=="__main__":unittest.main()
