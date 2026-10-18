# NFL data ingestion — v0.2 scaffold (model engine remains v0.1.0)

This module extracts reproducible pregame input fields and refuses to manufacture missing data. **This is a working offline ingestion/validation component, not proof of real-game predictive accuracy.** No real-game data files are bundled. Fields still missing must be sourced, or the model must return BLOCKED. Inputs from a downloaded file obtained after kickoff do **not** count as a verified pre-kickoff archive, even when rows from the target game are excluded.

## Source files

Primary source: nflverse data GitHub releases:

- PBP: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.parquet`
- Schedules: `https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet`
- Weekly rosters: `https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_2026.parquet`

PBP and schedules are time-changing live release assets; acquire and freeze each file *before* the intended prediction cutoff, saving download hashes and timestamps. For a historical replay from an after-game download, record it as retrospective/event-filtered, never a strictly contemporaneous archived snapshot.

Optional pressure feed: participation with `was_pressure`, `game_id`/`nflverse_game_id` and `play_id`; this is generally distributed only **after the season** (2023 onward FTN provenance) and cannot be assumed available pregame during 2026. To create a valid 2026 pregame pressure estimate, obtain a separately timestamped contemporary charting source with comparable dropback definitions, or explicitly develop a documented, independently calibrated historical prior in a future version of the model. Do not substitute sack rate for pressure rate under the current model.

QB identity: position map with `player_id`/`gsis_id` and `position`, matched against the PBP's `rusher_player_id`. In v0.1, QB rushing uses the model's **rush play category**, excluding QB scrambles classified as dropbacks. This is a methodological limitation in the original schema to revisit in v0.2; do not claim designed runs plus scrambles are fully accounted for.

Penalty EPA: PBP penalty indicator alone **does not** quantify net penalty EPA; deriving true counterfactual penalty effect requires an independent model or a precomputed verified report. Special-teams EPA also needs a sign-checked, possession-normalized source. These are not set to 0 by default. For both, use sourced records with `value`, `source`, and `available_at_utc` in the external game config.

## Install and optionally download

```bash
python -m pip install -r requirements-ingestion.txt
python fetch_nflverse.py --season 2026 --out data/2026_snapshots
```

Downloads are optional. `fetch_nflverse.py` is not executed automatically by `ingest.py` and needs internet access. It produces `download_manifest.json` with SHA-256 hashes, completed download times, source URLs, and failures. The downloader does not fetch proprietary market data, injuries, or live participation pressure.

For a strict historical pregame replay, a snapshot taken **before kickoff** must already exist. A file downloaded after the kickoff fails the historical-source-as-of requirement even if it is restricted to prior games. This protects against silent retroactive stat corrections.

## Run ingestion

```bash
python ingest.py \
  --pbp data/2026_snapshots/play_by_play_2026.parquet \
  --schedule data/2026_snapshots/games.parquet \
  --qb-roster data/qb_position_map.csv \
  --participation data/pregame_pressure_charting.csv \
  --config replays/hou_jax_2026_10_18/ingest_config_UNVERIFIED.json \
  --output-prefix replays/hou_jax_2026_10_18/ingested
```

The current `ingest_config_UNVERIFIED.json` is an **honest, intentionally incomplete** example. The source timestamps, historical same-time odds and archived pressure are unknown. The tool will issue a blocking report until those are resolved, and will never produce a runnable-looking `.game.json` unless the full model validation passes.

The same command without optional roster or pressure flags produces a useful `.audit.json` with missing field names, but will not run the predictive model.

If all requirements pass, run:

```bash
python model.py replays/hou_jax_2026_10_18/ingested.game.json \
  --runs 50000 --output replays/hou_jax_2026_10_18/simulation.json \
  --markdown replays/hou_jax_2026_10_18/simulation.md
```

## Preconditions and data definitions

- PBP `game_id`, play_id, posteam, defteam, epa, pass, rush, qb_dropback, sack, interception, success, yards_gained, no_play, qb_kneel, qb_spike, fumble_lost must exist. Missing mandatory statistics on qualifying plays block ingestion.
- Target `home_team`, `away_team`, and exact kickoff UTC must match a schedule entry. If schedule does not provide kickoff UTC, `gameday` + `gametime` are treated as US Eastern; DST is handled.
- Only regular-season games that **started at least six hours before** forecast cutoff are eligible. Nullified plays, kneels, spikes and special teams are excluded from offensive play efficiency. This conservative time buffer prevents accidental inclusion of in-progress games.
- Pass category = dropback OR pass flag; rushing category = rush and not a dropback. Sack rate and interception rate are defined per dropback; offensive fumbles lost per scrimmage play.
- Explosive passes = dropbacks gaining 15+ yards; explosive rushes = rushes gaining 10+ yards. Success is provider-defined `success` and is not imputed from EPA.
- Offensive red-zone TD rate = offensive TD per drive reaching the opponent 20; visit counts come from `drive` and `yardline_100`. Defensive rate is the counterpart from opposing possessions. Defensive return TDs are excluded by using passing/rushing offensive TD flags.
- Third-down conversions use `third_down_converted`/`third_down_failed` denominator, not just arbitrary third-down plays.
- Plays/drive and starts are calculated from scrimmage plays by unique `game_id + drive`. This is an approximation: ensure fumbles, defense scores, end-of-half drives and source drive semantics are consistent before production.
- Opponent adjustments are **play-weighted leave-head-to-head-out** opponent EPA. If there is no independent prior opponent sample, the adjusted statistic is missing and the model blocks; later development can estimate offseason priors rather than fill with zero silently.
- Pressure participation files and QB roster files must carry documented *source availability times* that precede the cutoff, not only rows with early game dates. When pressure coverage is below 98% of eligible pass plays, pressure remains missing.
- `market` must provide spread, both American prices, total, source and timestamp from a synchronized pregame market quotation. Never mix odds from different books or dates.
- Personnel needs sourced game-day confirmation within 24h of prediction snapshot; a neutral numeric shift is allowed if documented, but it does not mean players are known healthy.

## Coverage and exact outstanding work

The pipeline calculates most PBP-derived offensive, defensive, situational and SOS fields. The following require sources or methodology decisions beyond PBP:

1. **2026 pressure**: independent contemporary play-level/weighted team charting source with timestamp and team IDs, because nflverse's participation release is postseason-lagged.
2. **Penalty and special teams EPA**: consistently signed source or verified separate computation, with audit and points-per-game normalization.
3. **Market**: synchronized timestamped spread, total and both sides' American odds before kickoff; closing number after kickoff is disallowed.
4. **Injury/lineup**: team/NFL reports captured before kickoff plus reproducible **estimated numerical impact** rules. Injury source availability is not solved by nflverse for 2026.
5. **Historical snapshot archive**: create pregame downloaded/hashed files going forward. The 2026 Texans–Jaguars retrospective replay cannot be retroactively made a perfect point-in-time archive if the files were only collected afterward.
6. **Model-specific validation**: time-based holdouts, league mean priors, pressure distribution fits, drive simulations, and calibrated ATS edges still need robust backtests. v0.1's coefficients are provisional.

The code is **NFL-first**, not a universal CFB ingestion implementation. CFB needs a distinct verified source map, pressure and drive definitions, and team-name reconciliation before claims of equivalence.

## Tests

```bash
python -m unittest discover -s tests -v
```

The ingestion suite uses SYNTHETIC data, including deliberately impossible high EPA in target-game rows, to test nonleakage. Do not confuse these tests with real-game statistical validation.
