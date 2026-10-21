"""NFL-only v0.10 research invariant and forward-evaluator tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model import InputError, MODEL_RULES, simulate
from mechanism_research import extract_mechanisms
from nfl_stateful_candidate import (
    STATES, state_scoring_residuals, stateful_candidate_simulation,
)
from test_mechanism_research import CUTOFF, fixture, projections
from evaluate_locked_forecasts import evaluate


class StatefulCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pbp, sched = fixture()
        cls.profile = extract_mechanisms(pbp, sched, CUTOFF, ("GB", "DAL"))

    def test_no_post_kickoff_target_game_is_used(self):
        pbp, sched = fixture(include_target=True)
        contaminated = extract_mechanisms(pbp, sched, CUTOFF, ("GB", "DAL"))
        self.assertEqual(contaminated["league"]["drive_states"],
                         self.profile["league"]["drive_states"])
        self.assertEqual(contaminated["league"]["completed_team_game_ppd"],
                         self.profile["league"]["completed_team_game_ppd"])

    def test_complete_drive_outcomes_not_mislabeled_scores(self):
        league = self.profile["league"]
        self.assertGreaterEqual(sum(x["drives"] for x in league["drive_states"].values()), 90)
        self.assertEqual(sum(x["points_proxy"] for x in league["drive_states"].values()), 0)
        self.assertGreaterEqual(len(league["completed_team_game_ppd"]), 16)

    def test_state_residuals_are_shrunk_and_centered(self):
        p = copy.deepcopy(self.profile)
        # Synthetic PRE-CUTOFF points, not results of Washington-SF or target fixture.
        p["offense"]["GB"]["drive_states"]["trailing_9_plus"]["points_proxy"] += 70
        p["defense"]["DAL"]["drive_states"]["trailing_9_plus"]["points_proxy"] += 70
        p["league"]["drive_states"]["trailing_9_plus"]["points_proxy"] += 70
        result = state_scoring_residuals(p, "GB", "DAL")
        d = result["ppd_delta"]
        self.assertGreater(d["trailing_9_plus"], d["within_8"])
        self.assertTrue(all(-.4 <= d[s] <= .4 for s in STATES))
        self.assertEqual(result["status"], "OBSERVED_SHRUNK_UNFITTED")

    def test_no_fabricated_profile_or_game_history(self):
        p = copy.deepcopy(self.profile)
        del p["league"]["drive_states"]
        with self.assertRaises(InputError):
            state_scoring_residuals(p, "GB", "DAL")
        p = copy.deepcopy(self.profile)
        p["league"]["completed_team_game_ppd"] = []
        h, a = projections()
        with self.assertRaisesRegex(InputError, "variance"):
            stateful_candidate_simulation(h, a, p, MODEL_RULES["NFL"], 1000, 13)

    def test_deterministic_research_only_and_baseline_unchanged(self):
        h, a = projections()
        original = copy.deepcopy((h, a, self.profile))
        x = stateful_candidate_simulation(h, a, self.profile, MODEL_RULES["NFL"], 1000, 81)
        y = stateful_candidate_simulation(h, a, self.profile, MODEL_RULES["NFL"], 1000, 81)
        self.assertEqual(x, y)
        self.assertEqual((h, a, self.profile), original)
        self.assertTrue(x["research_only"])
        self.assertFalse(x["promoted_to_primary"])
        self.assertEqual(x["schema"], "nfl_stateful_challenger_v0.10")
        self.assertTrue(0 < x["score_home"] < 60)
        self.assertTrue(0 < x["score_away"] < 60)
        self.assertIn("in-game injuries", " ".join(x["limitations"]).lower())
        self.assertNotIn("home_cover_probability_calibrated", x)
        self.assertNotIn("ats_roi", x)

    def test_model_simulate_stays_separate_from_v010(self):
        p = dict(expected_drives=10.5, _ppd=2.1, _td=.6, _to=.10)
        before = simulate(p, p, MODEL_RULES["NFL"], None, None, 1000, 55)
        h, a = projections()
        stateful_candidate_simulation(h, a, self.profile, MODEL_RULES["NFL"], 1000, 55)
        after = simulate(p, p, MODEL_RULES["NFL"], None, None, 1000, 55)
        self.assertEqual(before, after)

    def test_requires_minimum_simulations(self):
        h, a = projections()
        with self.assertRaises(ValueError):
            stateful_candidate_simulation(h, a, self.profile, MODEL_RULES["NFL"], 999, 1)

    def test_forward_evaluator_reports_separate_team_errors(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "frozen.json"
            pred = dict(
                status="EXPLORATORY_CORE_PROJECTION_UNCALIBRATED",
                game_id="2026_game", home="GB", away="DAL",
                kickoff_utc="2026-10-18T20:00:00Z",
                data_as_of_utc="2026-10-18T17:00:00Z",
                projected_home_margin=4.0, score_home=24, score_away=20,
                mechanism_research={"candidate": dict(
                    mean_margin_home=3., score_home=25., score_away=22.,
                    promoted_to_primary=False)},
                stateful_research={"candidate": dict(
                    mean_margin_home=5., score_home=23., score_away=18.,
                    promoted_to_primary=False)},
            )
            f.write_text(json.dumps(pred))
            scores = pd.DataFrame([dict(game_id="2026_game", home_team="GB",
                away_team="DAL", home_score=23., away_score=17.,
                result=6., gameday="2026-10-18", gametime="16:00")])
            e = evaluate([f], scores)
            self.assertEqual(e["completed_games"], 1)
            self.assertEqual(e["stateful_completed_games"], 1)
            self.assertEqual(e["stateful_mae_margin"], 1)
            self.assertEqual(e["stateful_mae_total"], 1)
            self.assertEqual(e["base_mae_home_score"], 1)
            self.assertEqual(e["base_mae_away_score"], 3)
            self.assertEqual(e["ats_record"], "NOT_MEASURED_NO_VALIDATED_PRICED_SIGNED_LINES")
            pred["stateful_research"]["candidate"]["promoted_to_primary"] = True
            f.write_text(json.dumps(pred))
            with self.assertRaisesRegex(InputError, "incorrectly promoted"):
                evaluate([f], scores)


if __name__ == "__main__":
    unittest.main()
