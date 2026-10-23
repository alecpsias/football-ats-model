#!/usr/bin/env python3
"""NFL play-by-play -> strict, timestamp-audited model v0.1 input.

Does NOT fetch data, invent player availability, impute missing pressure, or
pretend a subsequently downloaded PBP file is a point-in-time archive.
Use normalized schedules and nflverse-compatible PBP CSV/Parquet.
"""
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from model import OFF_KEYS, DEF_KEYS, SOS_KEYS, CONTEXT_KEYS, PERSONNEL_KEYS, InputError, utc, validate_game, VERSION

ET = ZoneInfo("America/New_York")


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in (".parquet", ".pq"):
        try:
            return pd.read_parquet(path)
        except ImportError as e:
            raise InputError("Parquet requires pyarrow; pip install pyarrow") from e
    raise InputError("Supported input formats: CSV and Parquet")


def datetime_as_utc(x: str) -> pd.Timestamp:
    return pd.Timestamp(utc(x, "source timestamp"))


def normalize_schedule(df: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "home_team", "away_team"}
    if required - set(df):
        raise InputError(f"Schedule missing columns: {sorted(required - set(df))}")
    df = df.copy()
    df["game_id"] = df["game_id"].astype(str)
    if "kickoff_utc" not in df.columns:
        if not {"gameday", "gametime"} <= set(df):
            raise InputError("Schedule requires kickoff_utc OR gameday and gametime (ET)")
        def make_kick(row):
            s = f"{row['gameday']} {str(row['gametime']).strip()}"
            if s.endswith('nan'):
                raise InputError("No verified game time in schedule")
            try:
                return pd.Timestamp(datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=ET)).tz_convert("UTC")
            except ValueError as exc:
                raise InputError(f"Cannot read kickoff time: {s}") from exc
        df["kickoff_utc"] = df.apply(make_kick, axis=1)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="raise")
    if df["game_id"].duplicated().any():
        raise InputError("Schedule contains duplicate game_id")
    return df


def norm_flag(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).eq(1)


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def rate(df: pd.DataFrame, col: str) -> float | None:
    if col not in df or df.empty:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.isna().any():
        return None
    return float(s.mean())


def mean_or_none(df: pd.DataFrame, col: str) -> float | None:
    if col not in df or df.empty:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.isna().any():
        return None
    return float(s.mean())


def safe_ratio(n: float | int, d: float | int) -> float | None:
    return float(n/d) if d else None


def subset_epa(df: pd.DataFrame, play: str) -> tuple[float | None, int]:
    sub = df[df["_type"].eq(play)]
    return mean_or_none(sub, "epa"), len(sub)


def weighted_sos(pbp: pd.DataFrame, team: str, offense: bool, play: str) -> float | None:
    """Play-weighted strength of opponents, excluding games against target team.

    Offense team: average opponents' defensive EPA allowed vs OTHER teams.
    Defense team: average opponents' offensive EPA vs OTHER defenses.
    No fallback to same head-to-head games, which would leak the team's own
    EPA back into its opponent adjustment.
    """
    if offense:
        faced = pbp[(pbp.posteam == team) & (pbp._type == play)]
        opponents = faced.groupby("defteam").size().to_dict()
        field = "defteam"
        exclude_field = "posteam"
    else:
        faced = pbp[(pbp.defteam == team) & (pbp._type == play)]
        opponents = faced.groupby("posteam").size().to_dict()
        field = "posteam"
        exclude_field = "defteam"
    weighted_total = 0.0
    denom = 0.0
    for opp, weight in opponents.items():
        other = pbp[(pbp[field] == opp) & (pbp[exclude_field] != team) & (pbp._type == play)]
        avg = mean_or_none(other, "epa")
        if avg is None:
            return None
        weighted_total += avg * int(weight)
        denom += int(weight)
    return weighted_total / denom if denom else None


def _drives(team_plays: pd.DataFrame, off: bool) -> pd.DataFrame | None:
    if "drive" not in team_plays or "game_id" not in team_plays:
        return None
    if team_plays["drive"].isna().any():
        return None
    return team_plays.sort_values(["game_id", "play_id"] if "play_id" in team_plays else ["game_id", "drive"]).groupby(["game_id", "drive"], sort=False)


def _rz(team_pbp: pd.DataFrame) -> tuple[int | None, float | None]:
    if not {"drive", "yardline_100", "pass_touchdown", "rush_touchdown"} <= set(team_pbp):
        return None, None
    if team_pbp["drive"].isna().any():
        return None, None
    z = team_pbp.copy()
    z["_in_redzone"] = numeric(z, "yardline_100").le(20)
    z["_off_td"] = norm_flag(z["pass_touchdown"]) | norm_flag(z["rush_touchdown"])
    drive = z.groupby(["game_id", "drive"]).agg(rz=("_in_redzone", "any"), td=("_off_td", "any"))
    visits = drive[drive.rz]
    return len(visits), float(visits.td.mean()) if len(visits) else None


def _third(team_pbp: pd.DataFrame) -> float | None:
    if not {"third_down_converted", "third_down_failed"} <= set(team_pbp):
        return None
    c, f = norm_flag(team_pbp.third_down_converted), norm_flag(team_pbp.third_down_failed)
    valid = c | f
    return safe_ratio(int(c[valid].sum()), int(valid.sum()))


def _pass_rush_frames(pbp: pd.DataFrame, team: str, by_offense: bool) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    teamplays = pbp[pbp.posteam.eq(team) if by_offense else pbp.defteam.eq(team)]
    return teamplays, teamplays[teamplays._type.eq("pass")], teamplays[teamplays._type.eq("rush")]


def _team_stats(pbp: pd.DataFrame, team: str, is_offense: bool) -> dict:
    sub, p, r = _pass_rush_frames(pbp, team, is_offense)
    pass_n, rush_n = len(p), len(r)
    n = len(sub)
    dv = _drives(sub, is_offense)
    dcount = int(dv.ngroups) if dv is not None else 0
    games = int(sub.game_id.nunique())
    rz_n, rz_p = _rz(sub)
    yards_pass = numeric(p, "yards_gained")
    yards_rush = numeric(r, "yards_gained")
    pass_expl = float(yards_pass.ge(15).mean()) if len(p) and yards_pass.notna().all() else None
    rush_expl = float(yards_rush.ge(10).mean()) if len(r) and yards_rush.notna().all() else None
    press = rate(p, "was_pressure")
    sack = rate(p, "sack")
    int_rate = rate(p, "interception")
    fumble_lost = rate(sub, "fumble_lost")
    if is_offense:
        # QB rushing requires a verified QB position mapping, or complete labeled
        # qb_designed_run + qb_scramble fields. Scrambles are rushes for this count.
        qbr = r[r.qb_rush.eq(1)] if "qb_rush" in r else None
        return dict(
            games=games, plays=n, epa_pass=mean_or_none(p,"epa"), epa_rush=mean_or_none(r,"epa"),
            pass_rate=safe_ratio(pass_n,n), success_pass=rate(p,"success"), success_rush=rate(r,"success"),
            explosive_pass_rate=pass_expl, explosive_rush_rate=rush_expl,
            pressure_allowed=press, sack_allowed=sack, interception_rate=int_rate,
            fumble_lost_rate=fumble_lost, redzone_td_rate=rz_p, redzone_trips=rz_n,
            third_down_rate=_third(sub), drives_per_game=safe_ratio(dcount,games),
            plays_per_drive=safe_ratio(n,dcount), qb_rushes_per_game=safe_ratio(len(qbr),games) if qbr is not None else None,
            # A *verified* QB-position join with zero QB-designed rushing attempts
            # has zero contribution to the QB-rush scoring term. Assign the
            # structural neutral value 0, not a fabricated per-carry estimate.
            # Unknown roster/position joins STILL leave the input unavailable.
            qb_rush_epa=(mean_or_none(qbr,"epa") if qbr is not None and not qbr.empty
                         else 0.0 if qbr is not None else None),
            qb_rush_sample_count=(len(qbr) if qbr is not None else None),
            qb_rush_epa_provenance=("OBSERVED_QB_RUSH_MEAN" if qbr is not None and not qbr.empty
                                    else "ZERO_OBSERVED_QB_RUSHES_STRUCTURAL_NEUTRAL"
                                    if qbr is not None else "UNKNOWN_QB_POSITION_NO_ESTIMATE"),
        )
    return dict(
        plays=n, epa_pass_allowed=mean_or_none(p,"epa"), epa_rush_allowed=mean_or_none(r,"epa"),
        success_pass_allowed=rate(p,"success"), success_rush_allowed=rate(r,"success"),
        explosive_pass_allowed=pass_expl, explosive_rush_allowed=rush_expl,
        pressure_generated=press, sack_generated=sack, interception_rate_forced=int_rate,
        fumble_recovery_rate=fumble_lost, redzone_td_allowed=rz_p, redzone_trips_faced=rz_n,
        third_down_allowed=_third(sub), drives_faced_per_game=safe_ratio(dcount,games),
        plays_per_drive_allowed=safe_ratio(n,dcount),
    )


def _prepare_pbp(pbp: pd.DataFrame, schedule: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame,dict]:
    need = {"game_id","posteam","defteam","epa","pass","rush","qb_dropback","sack","interception","success","yards_gained","qb_kneel","qb_spike","fumble_lost","play_id"}
    missing = sorted(need - set(pbp))
    if missing:
        raise InputError(f"PBP missing indispensable columns: {missing}")
    if pbp["game_id"].isna().any():
        raise InputError("PBP contains rows without game_id")
    pbp = pbp.copy()
    if "no_play" not in pbp.columns:
        # Current nflfastR / nflverse releases encode nullified plays as
        # play_type == 'no_play'; they do not necessarily ship a no_play flag.
        # Reconstruct ONLY that explicit source label, never assume that
        # missing labels mean genuine plays. Preserve raw snapshots unchanged.
        if "play_type" not in pbp.columns:
            raise InputError("PBP missing no_play flag and play_type source classification")
        eligible = (pbp.posteam.notna() & pbp.defteam.notna() &
                    (norm_flag(pbp.qb_dropback) | norm_flag(pbp["pass"]) | norm_flag(pbp["rush"])))
        labels = pbp["play_type"].astype("string").str.strip().str.lower()
        if labels.loc[eligible].isna().any():
            raise InputError("PBP cannot classify null play_type among offensive candidate plays")
        pbp["no_play"] = labels.eq("no_play").fillna(False).astype("int8")
    pbp["game_id"] = pbp["game_id"].astype(str)
    # Require a conservative six-hour completion buffer. Merely kicking off
    # before the cutoff does not establish that a game was finished.
    covered = schedule[schedule.kickoff_utc.lt(cutoff - pd.Timedelta(hours=6))]
    if "game_type" in covered:
        covered = covered[covered.game_type.eq("REG")]
    if "season_type" in covered:
        covered = covered[covered.season_type.eq("REG")]
    pbp = pbp.merge(covered[["game_id","kickoff_utc"]], on="game_id", how="inner", validate="many_to_one")
    # Exclude nullified plays, special teams, kneels, spikes and clock artifacts.
    pbp = pbp[pbp.posteam.notna() & pbp.defteam.notna()]
    pbp = pbp[~norm_flag(pbp.no_play) & ~norm_flag(pbp.qb_kneel) & ~norm_flag(pbp.qb_spike)]
    dropback = norm_flag(pbp.qb_dropback) | norm_flag(pbp["pass"])
    rush = norm_flag(pbp["rush"]) & ~dropback
    pbp["_type"] = np.select([dropback, rush],["pass","rush"],default="other")
    pbp = pbp[pbp._type.ne("other")].copy()
    for c in ("epa","success","sack","interception","yards_gained","fumble_lost"):
        if numeric(pbp,c).isna().any():
            raise InputError(f"PBP contains missing or nonnumeric {c} among qualified plays")
    if pbp.duplicated(["game_id","play_id"]).any():
        raise InputError("Duplicate PBP game_id/play_id (could inflate rates)")
    return pbp, {"completed_games_in_window":int(covered.game_id.nunique()), "eligible_scrimmage_plays":len(pbp), "last_eligible_kickoff":str(covered.kickoff_utc.max())}


def _participation_join(pbp: pd.DataFrame, part: pd.DataFrame | None) -> tuple[pd.DataFrame,dict]:
    if part is None:
        pbp["was_pressure"] = np.nan
        return pbp, {"pressure_join":"NO_PARTICIPATION_FILE"}
    if "game_id" not in part and "nflverse_game_id" in part:
        part = part.rename(columns={"nflverse_game_id":"game_id"})
    req = {"game_id","play_id","was_pressure"}
    if req - set(part):
        raise InputError(f"Participation file missing: {sorted(req-set(part))}")
    part = part.copy()
    part.game_id = part.game_id.astype(str)
    if part.duplicated(["game_id","play_id"]).any():
        raise InputError("Duplicate participation game_id/play_id")
    part["was_pressure"] = part["was_pressure"].map({True:1,False:0,1:1,0:0,"TRUE":1,"FALSE":0,"true":1,"false":0})
    joined = pbp.merge(part[["game_id","play_id","was_pressure"]], on=["game_id","play_id"], how="left", validate="one_to_one")
    eligible_pass = joined._type.eq("pass")
    coverage = float(joined.loc[eligible_pass,"was_pressure"].notna().mean()) if eligible_pass.any() else 0
    # Coverage below 98% is not a trustworthy pressure rate; missing values
    # remain NA and cause the model to block rather than be treated as 0.
    if coverage < .98:
        joined["was_pressure"] = np.nan
    return joined, {"pressure_join":"OK" if coverage >= .98 else "INSUFFICIENT_COVERAGE", "pressure_pass_coverage":round(coverage,4)}


def _qb_join(pbp: pd.DataFrame, roster: pd.DataFrame | None) -> tuple[pd.DataFrame,dict]:
    pbp["qb_rush"] = np.nan
    if roster is None:
        return pbp, {"qb_position_join":"NO_POSITION_FILE"}
    if "player_id" not in roster and "gsis_id" in roster:
        roster = roster.rename(columns={"gsis_id":"player_id"})
    required = {"player_id","position"}
    if required - set(roster):
        raise InputError(f"QB roster map missing: {sorted(required-set(roster))}")
    if "rusher_player_id" not in pbp:
        return pbp, {"qb_position_join":"PBP_LACKS_RUSHER_ID"}
    ids = set(roster.loc[roster.position.eq("QB"),"player_id"].astype(str))
    rush = pbp._type.eq("rush")
    has_id = pbp.loc[rush,"rusher_player_id"].notna().all()
    if not has_id:
        return pbp, {"qb_position_join":"INCOMPLETE_RUSHER_IDS"}
    pbp.loc[rush,"qb_rush"] = pbp.loc[rush,"rusher_player_id"].astype(str).isin(ids).astype(int)
    return pbp, {"qb_position_join":"OK", "qb_ids":len(ids)}


def _valid_timestamp_or_missing(value: object, field: str, cutoff: datetime, issues: list[str]) -> str | None:
    if value is None:
        issues.append(f"{field}: missing timestamp")
        return None
    try:
        t = utc(value,field)
    except (InputError,TypeError):
        issues.append(f"{field}: invalid timestamp")
        return None
    if t > cutoff:
        issues.append(f"{field}: after snapshot cutoff")
        return None
    return t.isoformat().replace("+00:00","Z")


def build(pbp_raw: pd.DataFrame, schedule_raw: pd.DataFrame, config: dict,
          participation: pd.DataFrame | None=None, roster: pd.DataFrame | None=None) -> tuple[dict,dict]:
    for key in ("id","kickoff_utc","data_as_of_utc","home_code","away_code","home_name","away_name","neutral_site","pbp_source","pbp_source_available_at_utc","market","teams"):
        if key not in config:
            raise InputError(f"Game config missing {key}")
    kickoff = utc(config["kickoff_utc"], "kickoff_utc")
    cutoff = utc(config["data_as_of_utc"], "data_as_of_utc")
    if cutoff >= kickoff:
        raise InputError("Forecast cutoff must precede kickoff")
    schedule = normalize_schedule(schedule_raw)
    # Check intended game from schedule (including neutral venue via config).
    match = schedule[(schedule.home_team == config["home_code"]) &
                     (schedule.away_team == config["away_code"]) &
                     (schedule.kickoff_utc == pd.Timestamp(kickoff))]
    if len(match) != 1:
        raise InputError("Target game and kickoff not uniquely verified in schedule")
    pbp, coverage = _prepare_pbp(pbp_raw, schedule, pd.Timestamp(cutoff))
    issues: list[str] = []
    if participation is not None:
        _valid_timestamp_or_missing(config.get("participation_source_available_at_utc"),
                                    "participation_source_available_at_utc",cutoff,issues)
    if roster is not None:
        _valid_timestamp_or_missing(config.get("roster_source_available_at_utc"),
                                    "roster_source_available_at_utc",cutoff,issues)
    pbp, pressure = _participation_join(pbp, participation)
    pbp, qb = _qb_join(pbp, roster)
    source_time = _valid_timestamp_or_missing(config.get("pbp_source_available_at_utc"),"pbp_source_available_at_utc",cutoff,issues)
    if source_time is None:
        issues.append("PBP source is NOT independently verified as available before prediction cutoff; retrospective event filtering alone is insufficient")
    g = {"id":config["id"],"league":"NFL","kickoff_utc":config["kickoff_utc"],
         "data_as_of_utc":config["data_as_of_utc"],"neutral_site":config["neutral_site"],
         "market":deepcopy(config["market"])}
    for key,code,name in (("home", config["home_code"],config["home_name"]),
                          ("away", config["away_code"],config["away_name"])):
        cfg = config["teams"].get(code,{})
        o = _team_stats(pbp,code,True)
        d = _team_stats(pbp,code,False)
        sos = {"opp_def_pass_epa_allowed_avg":weighted_sos(pbp,code,True,"pass"),
               "opp_def_rush_epa_allowed_avg":weighted_sos(pbp,code,True,"rush"),
               "opp_off_pass_epa_avg":weighted_sos(pbp,code,False,"pass"),
               "opp_off_rush_epa_avg":weighted_sos(pbp,code,False,"rush")}
        # Start field position is measured on first scrimmage play of each
        # offensive drive; do not use average yardline over all plays.
        offense = pbp[pbp.posteam == code].sort_values(["game_id","play_id"])
        first = offense.groupby(["game_id","drive"],sort=False).head(1) if "drive" in offense else pd.DataFrame()
        if len(first) and "yardline_100" in first and numeric(first,"yardline_100").notna().all():
            start = float((100-numeric(first,"yardline_100")).mean())
        else:
            start = None
        context = {"avg_start_yardline":start, "penalty_epa_per_game":None, "special_teams_epa_per_game":None}
        for k in CONTEXT_KEYS:
            if k in cfg.get("context",{}):
                # Context-file values are not invented; require source record
                # including published-before-cutoff metadata outside the model.
                item = cfg["context"][k]
                if isinstance(item,dict) and "value" in item:
                    when = _valid_timestamp_or_missing(item.get("available_at_utc"),f"{code}.context.{k}",cutoff,issues)
                    if when is not None and item.get("source"):
                        context[k] = item["value"]
                    else:
                        issues.append(f"{code}.context.{k}: no valid source and time")
        p = {"pass_epa_shift":None,"rush_epa_shift":None,"pressure_allowed_shift":None,
             "qb_rush_multiplier":None,"pace_multiplier":None,"notes":None,"source":None,"checked_at_utc":None}
        for k in PERSONNEL_KEYS:
            if k in cfg.get("personnel",{}):
                p[k] = cfg["personnel"][k]
        if p.get("checked_at_utc") is not None:
            _valid_timestamp_or_missing(p["checked_at_utc"],f"{code}.personnel.checked_at_utc",cutoff,issues)
        g[key]={"name":name, "stat_source":config["pbp_source"],
                "stat_as_of_utc":source_time, "offense":o,"defense":d,"schedule":sos,
                "context":context,"personnel":p}
    m = g["market"]
    _valid_timestamp_or_missing(m.get("as_of_utc"),"market.as_of_utc",cutoff,issues)
    # Validate *every* input even when blocked, accumulating all missing fields.
    missing = []
    for side in ("home","away"):
        team=g[side]
        for section, keys in (("offense",OFF_KEYS),("defense",DEF_KEYS),("schedule",SOS_KEYS),
                              ("context",("avg_start_yardline",)),("personnel",PERSONNEL_KEYS)):
            missing.extend(f"{side}.{section}.{k}" for k in keys if team[section].get(k) is None)
        missing.extend(f"{side}.{k}" for k in ("stat_as_of_utc",) if team.get(k) is None)
    missing.extend(f"market.{k}" for k in ("spread_home","home_odds_american","away_odds_american","total","as_of_utc","source") if m.get(k) is None)
    audit = {"status":"BLOCKED_MISSING_INPUTS_OR_ASOF" if missing or issues else "READY_TO_VALIDATE",
             "source_lineage":{"pbp_source":config["pbp_source"],"source_available_at_utc":source_time,
                               "snapshot_cutoff":config["data_as_of_utc"],"target_kickoff":config["kickoff_utc"]},
             "coverage":{**coverage,**pressure,**qb},"missing_fields":sorted(set(missing)),
             "missing_count":len(set(missing)),"asof_issues":issues,
             "field_definitions":{"pass":"qb_dropback or pass=1, excludes nullified plays and kneels",
                                  "rush":"rush=1 excluding dropbacks/kneels",
                                  "explosive_pass":"gain >= 15 yd per dropback",
                                  "explosive_rush":"gain >= 10 yd per rush",
                                  "pressure":"participation was_pressure aligned by game_id/play_id",
                                  "drives":"unique game_id/drive; red-zone visit = drive reaches <=20 yards to goal",
                                  "opponent_strength":"leave-opponent-game-out, play weighted"},
             "model_version":VERSION}
    if audit["status"]=="READY_TO_VALIDATE":
        try:
            validate_game(g)
            audit["status"]="READY_FOR_MODEL_UNCALIBRATED"
        except (InputError,TypeError,ValueError) as e:
            audit["status"]="BLOCKED_SCHEMA_VALIDATION"
            audit["validation_error"]=str(e)
    return g,audit


def main() -> None:
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--pbp",type=Path,required=True)
    pa.add_argument("--schedule",type=Path,required=True)
    pa.add_argument("--config",type=Path,required=True)
    pa.add_argument("--participation",type=Path)
    pa.add_argument("--qb-roster",type=Path)
    pa.add_argument("--output-prefix",type=Path,required=True)
    args=pa.parse_args()
    cfg=json.loads(args.config.read_text(encoding="utf-8"))
    game,audit=build(load_table(args.pbp),load_table(args.schedule),cfg,
                     load_table(args.participation) if args.participation else None,
                     load_table(args.qb_roster) if args.qb_roster else None)
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    (args.output_prefix.parent/(args.output_prefix.name+".audit.json")).write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    # Do not create a file that looks runnable unless the full schema validates.
    suffix=".game.json" if audit["status"]=="READY_FOR_MODEL_UNCALIBRATED" else ".INCOMPLETE.json"
    (args.output_prefix.parent/(args.output_prefix.name+suffix)).write_text(json.dumps(game,indent=2)+"\n",encoding="utf-8")
    print(f"Ingestion: {audit['status']}; {audit['missing_count']} missing required fields; {len(audit['asof_issues'])} timestamp/provenance issues")
    print(f"Audit: {args.output_prefix}.audit.json")
    if audit["status"]!="READY_FOR_MODEL_UNCALIBRATED":
        raise SystemExit(2)


if __name__=="__main__":
    main()
