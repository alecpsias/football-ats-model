"""Regression tests: an incomplete slate can never be published as complete."""
import json
import tempfile
import unittest
from pathlib import Path

from cfb.investigation_gate import _blob_sha, audit_day


class TestMandatoryRankingGate(unittest.TestCase):
    DATE = "2026-10-24"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.record = self.root / "cfb/frozen_pregame_audits/2026_10_24_sample_team.json"
        self.record.parent.mkdir(parents=True)
        self.workflow = self.root / ".github/workflows/cfb-sample-team-20261024.yml"
        self.workflow.parent.mkdir(parents=True)
        self.workflow.write_text("name: sample\n", encoding="utf-8")
        self.data = {
            "fixture": {"game_id": "401999999", "away": "Sample", "home": "Team",
                        "kickoff_utc": "2026-10-24T19:00:00Z"},
            "provenance": {"snapshot_utc": "2026-10-24T16:00:00Z"},
            "simulation": {"draws_each": 5000},
            "decision": {"ats_side": "Sample +5.5", "grade": "B",
                         "confidence": 6.2, "bet": "Conditional lean"},
        }
        self.entry = {
            "game_id": "401999999", "away": "Sample", "home": "Team",
            "model": "CFB Build16",
            "audit_file": "cfb/frozen_pregame_audits/2026_10_24_sample_team.json",
            "workflow_file": ".github/workflows/cfb-sample-team-20261024.yml",
            "evidence": {
                "game_id": "fixture.game_id", "away": "fixture.away", "home": "fixture.home",
                "captured_at": "provenance.snapshot_utc", "kickoff": "fixture.kickoff_utc",
                "draws": "simulation.draws_each", "side": "decision.ats_side",
                "grade": "decision.grade", "confidence": "decision.confidence",
                "bet_decision": "decision.bet",
            },
        }
        self.registry = self.root / "cfb/investigation_registry/2026-10-24.json"
        self.registry.parent.mkdir(parents=True)
        self.save()

    def save(self):
        raw = (json.dumps(self.data, indent=2) + "\n").encode()
        self.record.write_bytes(raw)
        self.entry["blob_sha"] = _blob_sha(raw)
        self.registry.write_text(json.dumps({
            "schema": "cfb_investigation_registry_v1",
            "date": self.DATE, "entries": [self.entry],
        }, indent=2), encoding="utf-8")

    def run_gate(self):
        return audit_day(self.root, self.DATE)

    def test_verified_ranking_is_sorted_and_pinned(self):
        report = self.run_gate()
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertEqual(report["registered"], 1)
        self.assertEqual(report["verified"], 1)
        self.assertEqual(report["ranking"][0]["ats_selection"], "Sample +5.5")

    def test_missing_audit_blocks_without_partial_ranking(self):
        self.record.unlink()
        report = self.run_gate()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["ranking"], [])
        self.assertTrue(any("Registered audit is missing" in e for e in report["errors"]))

    def test_changed_frozen_record_blocks_even_when_valid_json(self):
        self.data["decision"]["confidence"] = 7.4
        self.record.write_text(json.dumps(self.data))
        report = self.run_gate()
        self.assertEqual(report["ranking"], [])
        self.assertTrue(any("differ from registered" in e for e in report["errors"]))

    def test_missing_simulation_draws_blocks(self):
        self.data["simulation"]["draws_each"] = 4999
        self.save()
        self.assertEqual(self.run_gate()["status"], "BLOCKED")

    def test_post_kickoff_data_blocks(self):
        self.data["provenance"]["snapshot_utc"] = "2026-10-24T20:01:00Z"
        self.save()
        self.assertEqual(self.run_gate()["status"], "BLOCKED")

    def test_unregistered_audit_blocks(self):
        (self.record.parent / "2026_10_24_unregistered.json").write_text("{}")
        self.assertTrue(any("Unregistered frozen pregame audit" in e
                            for e in self.run_gate()["errors"]))

    def test_unregistered_dated_workflow_blocks(self):
        (self.workflow.parent / "cfb-extra-20261024.yml").write_text("name: extra\n")
        self.assertTrue(any("workflow not registered" in e
                            for e in self.run_gate()["errors"]))

    def test_duplicate_registry_id_blocks(self):
        document = json.loads(self.registry.read_text())
        document["entries"].append(dict(self.entry))
        self.registry.write_text(json.dumps(document))
        self.assertEqual(self.run_gate()["status"], "BLOCKED")

    def test_missing_registry_blocks(self):
        self.registry.unlink()
        report = self.run_gate()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["ranking"], [])

    def test_directional_record_does_not_become_a_bet(self):
        del self.entry["evidence"]["bet_decision"]
        self.save()
        r = self.run_gate()
        self.assertEqual(r["status"], "PASS", r["errors"])
        self.assertEqual(r["ranking"][0]["bet_decision"],
                         "NO_EXPLICIT_WAGER_DECISION_IN_FROZEN_RECORD")


if __name__ == "__main__":
    unittest.main()
