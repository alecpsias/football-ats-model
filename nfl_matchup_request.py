"""Owner-only, JSON-only NFL matchup request from GitHub Issues.

The issue supplies identities only, never odds, stats, model parameters or scores.
We match against a newly collected, pre-kickoff forecast; missing match fails.
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
from pathlib import Path

TITLE = "NFL Matchup Request"
KIND = "nfl_matchup_request_v1"
CODE = re.compile(r"^[A-Z]{2,3}$")
GAME_ID = re.compile(r"^20[0-9]{2}_[0-9]{2}_[A-Z]{2,3}_[A-Z]{2,3}$")


def parse_issue(event):
    issue = event.get("issue")
    repo = event.get("repository")
    if not isinstance(issue, dict) or not isinstance(repo, dict):
        raise ValueError("Not a GitHub issue")
    author = (issue.get("user") or {}).get("login")
    owner = (repo.get("owner") or {}).get("login")
    if not owner or author != owner or issue.get("title") != TITLE:
        raise ValueError("Only repository-owner NFL matchup requests are allowed")
    body = issue.get("body")
    if not isinstance(body, str) or len(body) > 2048:
        raise ValueError("Invalid issue body")
    try:
        request = json.loads(body)
    except (ValueError, TypeError) as exc:
        raise ValueError("Issue body must contain JSON") from exc
    if not isinstance(request, dict) or set(request) != {
        "kind", "team_a", "team_b", "game_id"}:
        raise ValueError("Unknown request fields")
    if request["kind"] != KIND:
        raise ValueError("Unknown request kind")
    a, b = request["team_a"], request["team_b"]
    if not isinstance(a, str) or not CODE.fullmatch(a) or not isinstance(b, str) or not CODE.fullmatch(b) or a == b:
        raise ValueError("Two distinct uppercase NFL team codes required")
    game_id = request["game_id"]
    if game_id is None:
        game_id = ""
    if not isinstance(game_id, str) or (game_id and not GAME_ID.fullmatch(game_id)):
        raise ValueError("Invalid optional NFL game ID")
    return {"NFL_TEAM_A": a, "NFL_TEAM_B": b, "NFL_GAME_ID": game_id}


def select(directory, team_a, team_b, game_id, out):
    folder = Path(directory)
    index = json.loads((folder / "index.json").read_text())
    candidates = []
    for row in index.get("games", []):
        game = row.get("game_id")
        if not isinstance(game, str) or not GAME_ID.fullmatch(game):
            continue
        if game_id and game != game_id:
            continue
        # Match both exact teams in any ordering; never infer unspecified games.
        tokens = game.split("_")
        if set(tokens[-2:]) != {team_a, team_b}:
            continue
        if row.get("status") != "PROJECTED_UNCALIBRATED":
            raise ValueError("Requested NFL game blocked: " + str(row.get("reason")))
        file_name = row.get("file")
        if file_name != game + ".json":
            raise ValueError("Unsafe/mismatched forecast filename")
        candidates.append(folder / file_name)
    if len(candidates) != 1:
        raise ValueError(f"Expected one new pregame NFL forecast, got {len(candidates)}")
    source = candidates[0]
    record = json.loads(source.read_text())
    if record.get("game_id") != source.stem or {record.get("home"), record.get("away")} != {team_a, team_b}:
        raise ValueError("NFL selected record identity mismatch")
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError("Requested NFL archive already exists")
    shutil.copyfile(source, dest)
    return {"status": "SELECTED_UNVERIFIED_MODEL_OUTPUT", "game_id":record["game_id"],
            "output":str(dest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event")
    parser.add_argument("--env-file")
    parser.add_argument("--predictions")
    parser.add_argument("--team-a")
    parser.add_argument("--team-b")
    parser.add_argument("--game-id", default="")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.event:
        if not args.env_file or any((args.predictions, args.out)):
            raise ValueError("--event requires --env-file only")
        fields = parse_issue(json.loads(Path(args.event).read_text()))
        with Path(args.env_file).open("a", encoding="utf8") as env:
            env.write("".join(f"{k}={v}\n" for k, v in fields.items()))
        print(json.dumps({"status": "NFL_REQUEST_VALIDATED", **fields}))
    else:
        if not all((args.predictions, args.team_a, args.team_b, args.out)):
            raise ValueError("Select requires --predictions --team-a --team-b --out")
        if not CODE.fullmatch(args.team_a) or not CODE.fullmatch(args.team_b):
            raise ValueError("Unsafe NFL team names")
        print(json.dumps(select(args.predictions, args.team_a, args.team_b,
                                args.game_id, args.out)))


if __name__ == "__main__":
    main()
