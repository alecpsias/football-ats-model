"""No-leakage, incumbent, shrinkage and baseline-separation tests for NFL QB fit."""
from datetime import datetime, timezone
import copy
import unittest

import numpy as np
import pandas as pd

from model import MODEL_RULES, simulate
from nfl_qb_adjustments import (
    VERSION, pre_cutoff_qb_plays, adjusted_dropback_epa,
    qb_effect_for_team, estimate_qb_adjustments,
)
from nfl_stateful_candidate import stateful_candidate_simulation
from test_mechanism_research import fixture, projections, CUTOFF
from mechanism_research import extract_mechanisms


def sample():
    schedule = pd.DataFrame([
        dict(game_id="g1", home_team="CHI", away_team="NE", week=1,
             kickoff_utc="2026-09-10T20:00:00Z", home_score=14, away_score=21),
        dict(game_id="g2", home_team="NE", away_team="CHI", week=2,
             kickoff_utc="2026-09-17T20:00:00Z", home_score=21, away_score=14),
        dict(game_id="g3", home_team="CHI", away_team="NE", week=3,
             kickoff_utc="2026-09-24T20:00:00Z", home_score=21, away_score=10),
        dict(game_id="future", home_team="CHI", away_team="NE", week=4,
             kickoff_utc="2026-10-02T20:00:00Z", home_score=40, away_score=0),
        dict(game_id="near", home_team="CHI", away_team="NE", week=4,
             kickoff_utc="2026-09-26T18:00:00Z", home_score=50, away_score=7),
        dict(game_id="unfinished", home_team="CHI", away_team="NE", week=3,
             kickoff_utc="2026-09-24T21:00:00Z", home_score=np.nan, away_score=np.nan),
    ])
    rows = []
    for gid, week in [("g1",1),("g2",2),("g3",3),("future",4),("near",4),("unfinished",3)]:
        for team, opponent in [("CHI","NE"),("NE","CHI")]:
            for i in range(40):
                backup = team == "CHI" and ((week == 1 and i < 30) or
                                             (week == 3 and i < 8))
                qb = ("T.Backup" if backup else
                      ("C.Williams" if team == "CHI" else "D.Maye"))
                rows.append(dict(game_id=gid, week=week, posteam=team,
                    defteam=opponent, qb_dropback=1, qb_scramble=1 if i==0 else 0,
                    passer_player_name=np.nan if i == 0 else qb,
                    rusher_player_name=qb if i == 0 else np.nan,
                    play_type="run" if i == 0 else "pass",
                    epa=(.25 if qb=="C.Williams" else -.2 if backup else .05)
                          + i*.0005))
    return pd.DataFrame(rows),schedule


class QBAdjustmentTests(unittest.TestCase):
    def setUp(self):
        self.pbp,self.schedule=sample()
        self.cutoff=pd.Timestamp("2026-09-26T20:00:00Z")

    def test_excludes_future_unfinished_and_too_recent_games(self):
        x=pre_cutoff_qb_plays(self.pbp,self.schedule,self.cutoff)
        self.assertEqual(set(x.game_id),{"g1","g2","g3"})
        self.assertEqual(len(x),240)
        self.assertEqual(len(x[(x.posteam=="CHI")&(x.qb=="C.Williams")]),82)

    def test_scramble_attributed_to_qb_and_opponent_adjustment(self):
        x=pre_cutoff_qb_plays(self.pbp,self.schedule,self.cutoff)
        self.assertEqual(len(x[x.qb=="D.Maye"]),120)
        self.assertTrue((x[x.qb=="C.Williams"].play_type=="run").any())
        y=adjusted_dropback_epa(x)
        self.assertTrue(np.isfinite(y.adj_epa).all())
        self.assertEqual(len(y),len(x))

    def test_leading_last_game_qb_has_effect_but_not_confirmed(self):
        projection={"pass_rate":.55,"plays_per_drive":5.95}
        z=estimate_qb_adjustments(self.pbp,self.schedule,self.cutoff,"CHI","NE",
            {"CHI":projection,"NE":projection})
        self.assertEqual(z["CHI"]["status"],"APPLIED")
        self.assertEqual(z["CHI"]["projected_qb_name"],"C.Williams")
        self.assertGreater(z["CHI"]["adjusted_ppd_residual"],0)
        self.assertFalse(z["CHI"]["projected_starter_confirmed"])
        self.assertEqual(z["CHI"]["version"],VERSION)
        self.assertEqual(z["NE"]["status"],"APPLIED")
        self.assertLessEqual(abs(z["NE"]["adjusted_ppd_residual"]),.4)

    def test_ambiguous_or_small_latest_game_cannot_invent_starter(self):
        x=adjusted_dropback_epa(pre_cutoff_qb_plays(
            self.pbp,self.schedule,self.cutoff))
        changed=x.copy()
        latest=(changed.posteam=="CHI") & (changed.week==3)
        indexes=changed[latest].index[:18]
        changed.loc[indexes,"qb"]="T.Backup"
        est=qb_effect_for_team(changed,"CHI",{"pass_rate":.55,"plays_per_drive":5.95})
        self.assertEqual(est["status"],"NOT_APPLIED")
        self.assertEqual(est["reason"],"AMBIGUOUS_LAST_GAME_QB_USAGE")

    def test_missing_source_columns_fail_no_silent_fabrication(self):
        with self.assertRaisesRegex(ValueError,"QB_PBP_MISSING_COLUMNS"):
            pre_cutoff_qb_plays(self.pbp.drop(columns=["qb_dropback"]),
                               self.schedule,self.cutoff)

    def test_new_qb_delta_applies_only_to_existing_challenger(self):
        pbp,sched=fixture()
        profile=extract_mechanisms(pbp,sched,CUTOFF,("GB","DAL"))
        home,away=projections()
        home_before=copy.deepcopy(home)
        base_before=simulate(home,away,MODEL_RULES["NFL"],None,None,1000,11)
        raw=stateful_candidate_simulation(home,away,profile,MODEL_RULES["NFL"],1000,11)
        adjusted=stateful_candidate_simulation(home,away,profile,MODEL_RULES["NFL"],1000,11,
            qb_adjustments={"GB":{"status":"APPLIED","adjusted_ppd_residual":.25},
                            "DAL":{"status":"APPLIED","adjusted_ppd_residual":-.10}})
        self.assertGreater(adjusted["mean_margin_home"],raw["mean_margin_home"])
        self.assertEqual(adjusted["qb_form_ppd_adjustments"],{"home":.25,"away":-.10})
        self.assertEqual(home,home_before)
        self.assertEqual(base_before,
            simulate(home,away,MODEL_RULES["NFL"],None,None,1000,11))

    def test_non_finite_or_unbounded_injected_qb_adjustment_rejected(self):
        pbp,sched=fixture()
        profile=extract_mechanisms(pbp,sched,CUTOFF,("GB","DAL"))
        home,away=projections()
        for bad in (.41,float("nan"),float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(Exception,"Invalid QB residual"):
                    stateful_candidate_simulation(home,away,profile,MODEL_RULES["NFL"],
                        1000,15,qb_adjustments={
                            "GB":{"status":"APPLIED","adjusted_ppd_residual":bad}})


if __name__=="__main__":
    unittest.main()
