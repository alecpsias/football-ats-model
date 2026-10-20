"""Retrospective CFB scoring-margin replay. Research only; no ATS picks.

Input: immutable CFB pipeline snapshot captured after the studied games.
The replay filters game observations chronologically but CANNOT establish
that each source row was published before that historical kickoff.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cfb.pipeline import verify, normalize_pbp_teams, iso_utc
from model import MODEL_RULES, simulate


def total(records, key):
    return sum(row[key] for row in records)


def observed_game(plays, team, opponent, points, conceded, qb_ids):
    off = plays.loc[plays.pos_team_id == team]
    opp = plays.loc[plays.pos_team_id == opponent]
    if off.empty or opp.empty:
        return None

    def epa(group, kind):
        x = group.loc[group[kind].fillna(False).eq(True), "EPA"]
        x = pd.to_numeric(x, errors="coerce").dropna()
        return float(x.sum()), len(x)

    op, pn = epa(off, "pass")
    or_, rn = epa(off, "rush")
    dp, dpn = epa(opp, "pass")
    dr, drn = epa(opp, "rush")
    drives = off.drive_id.nunique()
    faced = opp.drive_id.nunique()
    if not drives or not faced or not pn or not rn:
        return None

    interceptions = off["int"].fillna(False).eq(True).sum()
    if "pos_fumble_lost" in off and off.pos_fumble_lost.notna().all():
        fumbles = int(off.pos_fumble_lost.eq(True).sum())
    else:
        fumbles = None

    qb_rushes = None
    qb_rush_epa = None
    if "rusher_player_id" in off and qb_ids:
        runs = off.loc[off.rush.fillna(False).eq(True)]
        matches = runs.loc[runs.rusher_player_id.isin(qb_ids)]
        qb_rushes = int(len(matches))
        qb_rush_epa = (
            float(matches.EPA.mean()) if
            len(matches) and matches.EPA.notna().all() else None
        )

    return {
        "points": float(points),
        "conceded": float(conceded),
        "drives": int(drives),
        "faced": int(faced),
        "off_pass_sum": op, "off_pass_n": pn,
        "off_rush_sum": or_, "off_rush_n": rn,
        "def_pass_sum": dp, "def_pass_n": dpn,
        "def_rush_sum": dr, "def_rush_n": drn,
        "interceptions": int(interceptions),
        "fumbles_lost": fumbles,
        "qb_rushes": qb_rushes,
        "qb_rush_epa": qb_rush_epa,
    }


def mean_epa(records, kind):
    count = total(records, kind + "_n")
    return total(records, kind + "_sum") / count if count else None


def project(team, opponent, neutral, side, use_epa):
    rules = MODEL_RULES["CFB"]
    prior = rules["league_ppd"]
    shrink_drives = 25.0

    off = (total(team, "points") + prior * shrink_drives) / (
        total(team, "drives") + shrink_drives
    )
    defense = (total(opponent, "conceded") + prior * shrink_drives) / (
        total(opponent, "faced") + shrink_drives
    )
    ppd = prior + 0.5 * ((off - prior) + (defense - prior))

    if use_epa:
        pass_off = mean_epa(team, "off_pass")
        pass_def = mean_epa(opponent, "def_pass")
        rush_off = mean_epa(team, "off_rush")
        rush_def = mean_epa(opponent, "def_rush")
        if None not in (pass_off, pass_def, rush_off, rush_def):
            pass_n = total(team, "off_pass_n")
            rush_n = total(team, "off_rush_n")
            pass_weight = pass_n / (pass_n + rush_n)
            matched_epa = (
                pass_weight * (pass_off + pass_def) / 2 +
                (1 - pass_weight) * (rush_off + rush_def) / 2
            )
            # Candidate coefficient, NOT empirically fitted.
            ppd += float(np.clip(1.5 * matched_epa, -0.45, 0.45))

    pace_off = total(team, "drives") / len(team)
    pace_def = total(opponent, "faced") / len(opponent)
    expected_drives = float(np.clip((pace_off + pace_def) / 2, 7, 17))

    hfa = 0 if neutral else rules["home_advantage_points"]
    ppd += (hfa / 2 if side == "home" else -hfa / 2) / expected_drives

    recorded_turnovers = [
        r for r in team if r["fumbles_lost"] is not None
    ]
    if recorded_turnovers:
        turnovers = sum(
            r["interceptions"] + r["fumbles_lost"]
            for r in recorded_turnovers
        )
        opportunities = sum(r["drives"] for r in recorded_turnovers)
        to_rate = (
            rules["league_to_drive"] * 20 + turnovers
        ) / (20 + opportunities)
    else:
        # Explicit modeling prior, NOT a measured team turnover rate.
        to_rate = rules["league_to_drive"]

    return {
        "_ppd": float(np.clip(ppd, 0.35, 5.0)),
        "_to": float(np.clip(to_rate, 0.01, 0.35)),
        "_td": rules["td_share_scoring_drives"],
        "expected_drives": expected_drives,
    }


def metrics(rows, field):
    if not rows:
        return {"games": 0}
    actual = np.array([r["actual_margin"] for r in rows])
    predicted = np.array([r[field] for r in rows])
    error = predicted - actual
    return {
        "games": len(rows),
        "mae": round(float(np.abs(error).mean()), 3),
        "rmse": round(float(np.sqrt(np.mean(error ** 2))), 3),
        "bias": round(float(error.mean()), 3),
        "margin_direction_accuracy": round(float(
            np.mean((predicted > 0) == (actual > 0))
        ), 4),
    }


def replay(snapshot, out, runs=1500, min_games=3):
    manifest = verify(snapshot)
    cutoff = iso_utc(manifest["captured_at_utc"])
    folder = Path(snapshot)

    schedule = pd.read_parquet(folder / "schedule.parquet")
    pbp = pd.read_parquet(folder / "pbp.parquet")
    roster = pd.read_parquet(folder / "roster.parquet")

    schedule["_kickoff"] = pd.to_datetime(
        schedule.start_date, utc=True, errors="coerce"
    )
    eligible = (
        schedule.fbs_participant.fillna(False).eq(True) &
        schedule.completed.fillna(False).eq(True) &
        schedule["_kickoff"].lt(cutoff) &
        ~(
            schedule.home_division.eq("fcs") &
            schedule.away_division.eq("fcs")
        ) &
        schedule.home_points.notna() &
        schedule.away_points.notna()
    )
    games = schedule.loc[eligible].sort_values(
        ["_kickoff", "game_id"]
    ).copy()

    data = normalize_pbp_teams(
        pbp.loc[pbp.game_id.isin(games.game_id)], games
    )
    pbp_by_game = {
        game_id: part for game_id, part in data.groupby("game_id")
    }

    quarterbacks = {}
    qb = roster.loc[
        roster.position_abbreviation.eq("QB")
    ]
    for team_id, group in qb.groupby("team_id"):
        quarterbacks[int(team_id)] = set(group.athlete_id.dropna())

    history = defaultdict(list)
    predictions = []
    rules = dict(MODEL_RULES["CFB"])

    # Forecast the entire simultaneous-kickoff group BEFORE adding results.
    for kickoff, group in games.groupby("_kickoff", sort=True):
        additions = []
        for _, game in group.iterrows():
            home = int(game.home_id)
            away = int(game.away_id)
            plays = pbp_by_game.get(game.game_id)
            if plays is None:
                continue

            h = observed_game(
                plays, home, away, game.home_points,
                game.away_points, quarterbacks.get(home, set())
            )
            a = observed_game(
                plays, away, home, game.away_points,
                game.home_points, quarterbacks.get(away, set())
            )
            if h is None or a is None:
                continue

            if (len(history[home]) >= min_games and
                    len(history[away]) >= min_games):
                item = {
                    "game_id": str(game.game_id),
                    "kickoff": str(kickoff),
                    "home": str(game.home_team),
                    "away": str(game.away_team),
                    "actual_margin": float(
                        game.home_points - game.away_points
                    ),
                }
                neutral = bool(game.neutral_site)
                for candidate, label in (
                    (False, "baseline_margin"),
                    (True, "epa_candidate_margin"),
                ):
                    hproj = project(
                        history[home], history[away],
                        neutral, "home", candidate
                    )
                    aproj = project(
                        history[away], history[home],
                        neutral, "away", candidate
                    )
                    result = simulate(
                        hproj, aproj, rules, None, None,
                        runs, seed=int(game.game_id) % (2 ** 32),
                    )
                    item[label] = result["mean_margin_home"]
                predictions.append(item)

            additions.append((home, h))
            additions.append((away, a))

        for team, record in additions:
            history[team].append(record)

    predictions.sort(key=lambda x: (x["kickoff"], x["game_id"]))
    split = int(len(predictions) * 0.7)
    while (1 < split < len(predictions) and
           predictions[split-1]["kickoff"] ==
           predictions[split]["kickoff"]):
        split -= 1

    holdout = predictions[split:]
    baseline = metrics(holdout, "baseline_margin")
    challenger = metrics(holdout, "epa_candidate_margin")
    benchmark = [
        dict(r, constant_home_margin=rules["home_advantage_points"])
        for r in holdout
    ]

    summary = {
        "status": "RETROSPECTIVE_RESEARCH_NOT_VALIDATED_ATS",
        "source_snapshot_utc": str(cutoff),
        "total_replay_games": len(predictions),
        "chronological_holdout_games": len(holdout),
        "holdout_baseline": baseline,
        "holdout_epa_challenger": challenger,
        "holdout_constant_home_advantage": metrics(
            benchmark, "constant_home_margin"
        ),
        "candidate_improves_mae": (
            challenger["mae"] < baseline["mae"]
            if holdout else None
        ),
        "warnings": [
            "Source files were captured after the historical games.",
            "This is retrospective replay, not verified as-of pregame data.",
            "Pressure is omitted; QB rushing is observational only.",
            "Model coefficients and probabilities are not calibrated.",
            "College overtime is only approximately simulated.",
            "No historical signed betting lines; ATS ROI not evaluated."
        ],
    }

    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "margin_replay.json").write_text(
        json.dumps(predictions, indent=2) + "\n"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=1500)
    args = parser.parse_args()
    replay(args.snapshot, args.out, args.runs)
