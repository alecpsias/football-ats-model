"""Free NFL prospective-artifact scoring: fixture checks, no network."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from nfl_forward_ledger import extract_forecasts
from evaluate_locked_forecasts import evaluate
from model import InputError
import pandas as pd


def zip_bytes(entries):
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as z:
        for name, payload in entries.items():
            z.writestr(name, payload)
    return bio.getvalue()


class ArchivedForecastTests(unittest.TestCase):
    def setUp(self):
        self.p = {
            "status": "EXPLORATORY_CORE_PROJECTION_UNCALIBRATED",
            "game_id": "2026_test", "home": "SF", "away": "WAS",
            "kickoff_utc": "2026-10-19T20:15:00Z",
            "data_as_of_utc": "2026-10-19T17:00:00Z",
            "projected_home_margin": 6.0, "score_home": 27,
            "score_away": 21, "challenger": {"version": "0.10.0", "candidate": {
                "mean_margin_home": 6.3, "score_home": 24.0, "score_away": 18.0,
                "promoted_to_primary": False}}
        }

    def test_only_prediction_files_extracted_and_original_bytes_retained(self):
        raw = json.dumps(self.p, separators=(",", ":")).encode()
        entries = {
            "future-predictions/run-1/index.json": '{"games":[]}',
            "future-predictions/run-1/2026_test.json": raw,
            "cfb/2026_test.json": raw,
            "snapshots/2026/metadata.json": raw,
            "future-predictions/run-2/not-an-actual-forecast.json": "{}",
        }
        with tempfile.TemporaryDirectory() as d:
            paths = extract_forecasts(zip_bytes(entries), Path(d), 700)
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].read_bytes(), raw)

    def test_after_lock_source_is_rejected_during_evaluation(self):
        self.p["data_as_of_utc"] = "2026-10-19T19:15:01Z"
        raw = json.dumps(self.p)
        finals = pd.DataFrame([dict(game_id="2026_test", home_team="SF",
                                   away_team="WAS", home_score=23,
                                   away_score=17, result=6,
                                   gameday="2026-10-19", gametime="16:15")])
        with tempfile.TemporaryDirectory() as d:
            paths = extract_forecasts(zip_bytes({
                "future-predictions/run-1/2026_test.json": raw}), Path(d), 123)
            with self.assertRaisesRegex(InputError, "After-lock"):
                evaluate(paths, finals)

    def test_frozen_fixture_scores_baseline_and_one_challenger_without_ats_claim(self):
        finals = pd.DataFrame([dict(game_id="2026_test", home_team="SF",
                                   away_team="WAS", home_score=23,
                                   away_score=17, result=6,
                                   gameday="2026-10-19", gametime="16:15")])
        with tempfile.TemporaryDirectory() as d:
            paths = extract_forecasts(zip_bytes({
                "future-predictions/run-1/2026_test.json": json.dumps(self.p)
            }), Path(d), 123)
            x = evaluate(paths, finals)
            self.assertEqual(x["completed_games"], 1)
            self.assertEqual(x["current_challenger"]["challenger_games"], 1)
            self.assertEqual(x["all_baseline"]["baseline_mae_margin"], 0)
            self.assertEqual(x["current_challenger"]["challenger_mae_margin"], .3)
            self.assertEqual(x["archived_challenger_versions"], {})
            self.assertEqual(x["ats_record"], "NOT_MEASURED_NO_VALIDATED_PRICED_SIGNED_LINES")


if __name__ == "__main__":
    unittest.main()
