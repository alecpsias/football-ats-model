"""NFL invariants: precisely one evolving challenger and immutable baseline."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from model import InputError
from nfl_challenger_policy import (
    ACTIVE_CHALLENGER_VERSION, BASELINE_VERSION,
    require_single_active_challenger,
)
from forecast_upcoming_core import forecast_one
from evaluate_locked_forecasts import evaluate


class ChallengerPolicyTests(unittest.TestCase):
    def _example(self):
        return {
            "challenger": {
                "version": ACTIVE_CHALLENGER_VERSION,
                "candidate": {"promoted_to_primary": False, "mean_margin_home": 3.0},
            }
        }

    def test_single_current_challenger_passes(self):
        require_single_active_challenger(self._example())
        self.assertEqual(BASELINE_VERSION, "0.8.1")

    def test_parallel_challengers_are_rejected(self):
        for legacy in ("mechanism_research", "stateful_research"):
            record = self._example()
            record[legacy] = {"candidate": {"promoted_to_primary": False}}
            with self.assertRaisesRegex(InputError, "Multiple NFL challenger"):
                require_single_active_challenger(record)

    def test_stale_version_and_implicit_promotion_are_rejected(self):
        record = self._example()
        record["challenger"]["version"] = "0.9.0"
        with self.assertRaisesRegex(InputError, "stale"):
            require_single_active_challenger(record)
        record = self._example()
        record["challenger"]["candidate"]["promoted_to_primary"] = True
        with self.assertRaisesRegex(InputError, "silently replace"):
            require_single_active_challenger(record)

    def test_live_forecast_calls_only_current_challenger(self):
        row = pd.Series({"kickoff_utc": pd.Timestamp("2026-10-23T00:15:00Z"),
                         "week": 7})
        cutoff = pd.Timestamp("2026-10-22T18:00:00Z").to_pydatetime()
        game = {
            "id": "one_test", "kickoff_utc": "2026-10-23T00:15:00Z",
            "data_as_of_utc": "2026-10-22T18:00:00Z",
            "neutral_site": False, "pressure_reference_rate": .32,
            "home": {"name": "SF"}, "away": {"name": "WAS"},
        }
        projected_home = {"team": "SF", "_ppd": 2.2}
        projected_away = {"team": "WAS", "_ppd": 1.9}
        baseline = {"mean_score_home": 25., "mean_score_away": 22.,
                    "mean_margin_home": 3., "home_win_probability_raw": .6,
                    "margin_p10_p50_p90": [-12, 3, 18]}
        stateful = {"promoted_to_primary": False, "mean_margin_home": 4.,
                    "score_home": 25., "score_away": 21.}
        profiles = {"mock": "verified"}
        people = {"teams": {"SF": {"reported_out": []},
                            "WAS": {"reported_out": []}}}
        with (patch("forecast_upcoming_core._assemble", return_value=(
                    game, {"asof_issues": []},
                    {"source_sha256": {}, "missing": []})),
              patch("forecast_upcoming_core.projection", side_effect=[
                  projected_home, projected_away]),
              patch("forecast_upcoming_core.simulate", return_value=baseline),
              patch("forecast_upcoming_core.extract_mechanisms", return_value=profiles),
              patch("forecast_upcoming_core.stateful_candidate_simulation",
                    return_value=stateful) as challenger_call,
              patch("forecast_upcoming_core.measure", return_value=people),
              patch("forecast_upcoming_core.scenario_points", return_value=[]),
              patch("forecast_upcoming_core.injuries_for_game", return_value={})):
            pred = forecast_one(None, None, row, cutoff, pbp=object(),
                                schedule=object(), runs=1000)
        require_single_active_challenger(pred)
        self.assertIn("challenger", pred)
        self.assertNotIn("mechanism_research", pred)
        self.assertNotIn("stateful_research", pred)
        self.assertEqual(pred["challenger"]["version"], ACTIVE_CHALLENGER_VERSION)
        self.assertEqual(pred["challenger"]["candidate"]["mean_margin_home"], 4.)
        self.assertEqual(pred["projected_home_margin"], 3.)
        challenger_call.assert_called_once()

    def test_legacy_two_sections_are_one_archived_challenger_not_active(self):
        # Frozen historical records stay immutable; evaluator picks ONE.
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "legacy.json"
            record = {
                "status": "EXPLORATORY_CORE_PROJECTION_UNCALIBRATED",
                "game_id": "old_game", "home": "SF", "away": "WAS",
                "kickoff_utc": "2026-10-20T00:15:00Z",
                "data_as_of_utc": "2026-10-19T20:00:00Z",
                "score_home": 27., "score_away": 21.,
                "projected_home_margin": 6.,
                "mechanism_research": {"candidate": {
                    "promoted_to_primary": False, "mean_margin_home": 4.,
                    "score_home": 24., "score_away": 20.}},
                "stateful_research": {"candidate": {
                    "promoted_to_primary": False, "mean_margin_home": 5.,
                    "score_home": 24., "score_away": 19.}}
            }
            f.write_text(json.dumps(record))
            actual = pd.DataFrame([{
                "game_id": "old_game", "home_team": "SF", "away_team": "WAS",
                "home_score": 23., "away_score": 17., "result": 6.,
                "gameday": "2026-10-19", "gametime": "20:15"
            }])
            result = evaluate([f], actual)
            self.assertEqual(result["completed_games"], 1)
            self.assertEqual(result["current_challenger"]["challenger_games"], 1)
            self.assertEqual(result["games"][0]["challenger_margin_home"], 5.)
            self.assertEqual(result["archived_challenger_versions"], {})


if __name__ == "__main__":
    unittest.main()
