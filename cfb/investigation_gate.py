"""Fail-closed, read-only CFB investigation-completeness and ranking gate.

This does not re-run simulations or edit frozen forecasts. Every investigated
game must be registered and its exact frozen file pinned by Git blob SHA.
A blocked report NEVER exposes an official ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "cfb_investigation_registry_v1"
ID = re.compile(r"^[0-9]{8,12}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
GRADE = {"A", "B", "C"}


def _read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _get(obj, dotted):
    if not isinstance(dotted, str) or not dotted:
        raise ValueError("Missing evidence field path")
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            raise ValueError("Missing evidence field: " + dotted)
        obj = obj[part]
    return obj


def _when(value):
    if not isinstance(value, str):
        raise ValueError("Source and kickoff require timestamps")
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("Timestamp lacks timezone")
    return t.astimezone(timezone.utc)


def _blob_sha(raw):
    # Same content-addressed SHA-1 definition used for Git blobs.
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\\0" + raw).hexdigest()


def _normalize_team(name):
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def _game_record(root, date, entry):
    game_id = str(entry["game_id"])
    name = entry["audit_file"]
    if not re.fullmatch(r"cfb/frozen_pregame_audits/2026_\\d{2}_\\d{2}_[a-z0-9_]+\\.json", name):
        raise ValueError("Unsafe or non-standard frozen audit path")
    if not name.startswith("cfb/frozen_pregame_audits/" + date.replace("-", "_") + "_"):
        raise ValueError("Audit filename belongs to different game date")
    pinned = entry.get("blob_sha")
    if not isinstance(pinned, str) or not HEX40.fullmatch(pinned):
        raise ValueError("Missing immutable Git blob pin")
    path = root / name
    if not path.is_file():
        raise ValueError("Registered audit is missing")
    raw = path.read_bytes()
    if _blob_sha(raw) != pinned:
        raise ValueError("Frozen audit bytes differ from registered Git blob")
    data = json.loads(raw)
    mapping = entry["evidence"]
    actual_game_id = str(_get(data, mapping["game_id"]))
    if actual_game_id != game_id:
        raise ValueError("Game ID mismatch")
    if _normalize_team(_get(data, mapping["away"])) != _normalize_team(entry["away"]):
        raise ValueError("Away team mismatch")
    if _normalize_team(_get(data, mapping["home"])) != _normalize_team(entry["home"]):
        raise ValueError("Home team mismatch")
    capture, kickoff = _when(_get(data, mapping["captured_at"])), _when(_get(data, mapping["kickoff"]))
    if capture >= kickoff or kickoff.date().isoformat() != date:
        raise ValueError("Invalid pregame cutoff or fixture date")
    draws = _get(data, mapping["draws"])
    if type(draws) is not int or draws != 5000:
        raise ValueError("Missing actual 5,000-draw simulation evidence")
    side = _get(data, mapping["side"])
    confidence = _get(data, mapping["confidence"])
    if not isinstance(side, str) or not side.strip():
        raise ValueError("No frozen directional ATS selection")
    if type(confidence) not in (int, float) or not (0 <= confidence <= 10):
        raise ValueError("Missing or invalid confidence rating")
    if "grade" in mapping:
        g = _get(data, mapping["grade"])
        if not isinstance(g, str) or not g or g[0].upper() not in GRADE:
            raise ValueError("Missing grade")
        grade = g[0].upper()
    else:
        grade = entry.get("grade_override")
        phrase = _get(data, mapping["grade_evidence"])
        if grade not in GRADE or not isinstance(phrase, str) or not re.search(r"\\b" + grade + r"-grade\\b", phrase, re.I):
            raise ValueError("Unsubstantiated conditional grade override")
    decision = _get(data, mapping["bet_decision"])
    if not isinstance(decision, str) or not decision.strip():
        raise ValueError("Missing wager/pass decision")
    return {
        "game_id": game_id, "away": entry["away"], "home": entry["home"],
        "ats_selection": side.strip(), "grade": grade,
        "confidence_10": float(confidence), "bet_decision": decision,
        "draws_each": draws, "source_captured_utc": capture.isoformat(),
        "kickoff_utc": kickoff.isoformat(), "model": entry["model"],
        "frozen_record": name, "git_blob_sha": pinned,
        "record_url": "https://github.com/alecpsias/football-ats-model/blob/main/" + name,
    }


def audit_day(root, date, registry_path=None):
    root = Path(root)
    if not re.fullmatch(r"20\\d{2}-\\d{2}-\\d{2}", date):
        raise ValueError("Date must be YYYY-MM-DD")
    manifest = Path(registry_path) if registry_path else root / "cfb" / "investigation_registry" / (date + ".json")
    if not manifest.is_file():
        return {"status": "BLOCKED", "date": date, "registered": 0, "verified": 0,
                "ranking": [], "errors": ["Investigation registry is missing: " + str(manifest)]}
    doc = _read_json(manifest)
    if doc.get("schema") != SCHEMA or doc.get("date") != date or not isinstance(doc.get("entries"), list):
        raise ValueError("Invalid investigation registry schema/date")
    errors = []
    rows = []
    ids, paths, workflows = set(), set(), set()
    for index, entry in enumerate(doc["entries"]):
        identifier = str(entry.get("game_id", ""))
        if not ID.fullmatch(identifier):
            errors.append("Entry " + str(index) + ": invalid game ID")
            continue
        if identifier in ids:
            errors.append(identifier + ": duplicate game ID")
        ids.add(identifier)
        path = entry.get("audit_file", "")
        if path in paths:
            errors.append(identifier + ": duplicate audit path")
        paths.add(path)
        workflow = entry.get("workflow_file", "")
        if not isinstance(workflow, str) or not re.fullmatch(r"\\.github/workflows/cfb-[a-z0-9-]+-20\\d{6}\\.yml", workflow):
            errors.append(identifier + ": missing dated workflow reference")
        else:
            if workflow in workflows:
                errors.append(identifier + ": duplicated workflow reference")
            workflows.add(workflow)
            if not (root / workflow).is_file():
                errors.append(identifier + ": referenced workflow file missing")
        try:
            rows.append(_game_record(root, date, entry))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, OSError) as exc:
            errors.append(identifier + " (" + entry.get("away", "?") + " at " +
                          entry.get("home", "?") + "): " + str(exc))
    # An archived game that has not been registered must not disappear from a ranking.
    known_files = {p.as_posix().removeprefix(root.as_posix().rstrip("/") + "/")
                   for p in (root / "cfb" / "frozen_pregame_audits").glob(date.replace("-", "_") + "_*.json")}
    for extra in sorted(known_files - paths):
        errors.append("Unregistered frozen pregame audit: " + extra)
    if not doc["entries"]:
        errors.append("Empty registry cannot certify a complete slate")
    rows.sort(key=lambda r: (-r["confidence_10"], {"A": 0, "B": 1, "C": 2}[r["grade"]],
                             r["away"].casefold(), r["home"].casefold()))
    # A blocked report must never be mistaken for an official ranked slate.
    return {"status": "BLOCKED" if errors else "PASS", "date": date,
            "registered": len(doc["entries"]), "verified": len(rows),
            "ranking": [] if errors else rows,
            "errors": errors, "verification": "strict-pinned-frozen-ats-v1",
            "note": ("No official ranking while any registered game is unresolved"
                     if errors else "Complete ranking from frozen archived records")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--registry")
    parser.add_argument("--out", default="cfb_artifacts/investigation_ranking_gate.json")
    args = parser.parse_args()
    result = audit_day(args.root, args.date, args.registry)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
