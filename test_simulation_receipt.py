"""Regression tests for the 5,000-run execution-proof gate."""
from __future__ import annotations
import copy
import unittest

from simulation_receipt import verify
from nfl_challenger_policy import ACTIVE_CHALLENGER_VERSION

SHA = "a" * 40
DIGEST = "f" * 64


def cfb():
    return {
        "mode": "pregame", "status": "FRESH_PREGAME_PROJECTION_LOCKED",
        "game_id": "401871070", "home": "Sam Houston",
        "away": "New Mexico State", "github_run_id": "123",
        "git_commit_sha": SHA, "source_manifest_sha256": DIGEST,
        "source_snapshot_at_utc": "2026-10-21T03:00:00+00:00",
        "prediction_generated_at_utc": "2026-10-21T03:01:00+00:00",
        "kickoff_utc": "2026-10-22T00:00:00+00:00",
        "model_variant": "UNMODIFIED_UNCALIBRATED_CFB_BASELINE",
        "simulated_margin_home": 2.04, "simulated_score_home": 25.7,
        "simulated_score_away": 23.66,
        "simulation": {"simulations": 5000},
        "build16_not_primary": True,
        "build16_challenger": {
            "version": "cfb-build16-unified-situational-v1",
            "mean_margin_home": 1.56,
            "simulation": {"simulations": 5000}
        }
    }


def nfl():
    return {
        "status": "EXPLORATORY_CORE_PROJECTION_UNCALIBRATED",
        "game_id": "2026_07_NE_CHI", "home": "CHI", "away": "NE",
        "generated_at_utc": "2026-10-21T15:01:00+00:00",
        "data_as_of_utc": "2026-10-21T15:00:00+00:00",
        "kickoff_utc": "2026-10-23T00:15:00+00:00",
        "execution": {"github_run_id": "123", "git_commit_sha": SHA},
        "baseline_simulation": {"simulations": 5000},
        "projected_home_margin": 4.02,
        "audit": {"simulation_runs": 5000,
                  "source_sha256": {"pbp": DIGEST},
                  "model_version": "0.8.1+single_challenger_"+ACTIVE_CHALLENGER_VERSION},
        "challenger": {"version": ACTIVE_CHALLENGER_VERSION, "candidate": {
            "n": 5000, "promoted_to_primary": False,
            "mean_margin_home": 3.7}}
    }


class ProofTests(unittest.TestCase):
    def test_cfb_receipt(self):
        result = verify(cfb(), "CFB", "123", SHA, DIGEST)
        self.assertEqual(result["primary_runs"], 5000)
        self.assertEqual(result["challenger_runs"], 5000)
        self.assertFalse(result["ats_calibrated"])

    def test_nfl_receipt(self):
        result = verify(nfl(), "NFL", "123", SHA, DIGEST)
        self.assertEqual(result["primary_home_margin"], 4.02)
        self.assertEqual(result["github_run_id"], "123")

    def test_block_cfb_short_challenger(self):
        x = cfb()
        x["build16_challenger"]["simulation"]["simulations"] = 3000
        with self.assertRaisesRegex(ValueError, "not 5000"):
            verify(x, "CFB", "123", SHA, DIGEST)

    def test_block_nfl_short_baseline(self):
        x = nfl()
        x["baseline_simulation"]["simulations"] = 3000
        with self.assertRaisesRegex(ValueError, "not 5000"):
            verify(x, "NFL", "123", SHA, DIGEST)

    def test_block_nfl_wrong_commit(self):
        with self.assertRaisesRegex(ValueError, "commit"):
            verify(nfl(), "NFL", "123", "b" * 40, DIGEST)

    def test_block_cfb_postkickoff(self):
        x = cfb()
        x["prediction_generated_at_utc"] = "2026-10-22T00:01:00+00:00"
        with self.assertRaisesRegex(ValueError, "kickoff"):
            verify(x, "CFB", "123", SHA, DIGEST)

    def test_block_nfl_missing_challenger(self):
        x = nfl()
        x["challenger"]["candidate"] = None
        with self.assertRaises(ValueError):
            verify(x, "NFL", "123", SHA, DIGEST)


if __name__ == "__main__":
    unittest.main()
