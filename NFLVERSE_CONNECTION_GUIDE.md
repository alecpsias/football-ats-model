# Free nflverse Connection — v0.3.0

## State of implementation

**Code integration complete; live remote download NOT confirmed in this execution environment.**

`nflverse_connector.py` invokes the public functions of the open `nflreadpy` library:

- `load_pbp([season])` -> play-by-play (EPA, success, rushing and passing play data)
- `load_schedules([season])` -> games, dates and opponents
- `load_rosters_weekly([season])` -> QB position mapping and weekly roster records
- `load_player_stats([season])` -> optional reference dataset

All saved files are compressed CSV (`.csv.gz`), without needing `pyarrow`. Every file has a recorded SHA-256 checksum, capture timestamp, source function, row count and byte count in a persistent `manifest.json`.

**No paid APIs, subscriptions, sportsbook API keys, or data purchases are used anywhere in this new connector.** This does not erase the separate market fields in the existing prediction model: these remain unsatisfied until a permissible FREE source is verified.

## One-time free installation

Use Python 3.10+ (3.11–3.13 recommended). Open a terminal in the unzipped `football_ats_model` folder:

```bash
python -m pip install -r requirements-nflverse.txt
```

The package requires an Internet connection when downloading data. The current ChatGPT execution environment did not have outbound GitHub/DNS access and did not have `nflreadpy` installed, so no actual 2026 nflverse data was downloaded here. Integration was tested with stand-in `nflreadpy` objects using the documented method signatures. 

## Download and archive the season

Choose a unique snapshot directory each run. For a user-facing run on October 18:

```bash
python nflverse_connector.py collect --season 2026 --out data/snapshots/2026-10-18-afternoon
```

For an optional separate player-stat file:

```bash
python nflverse_connector.py collect --season 2026 --include-player-stats --out data/snapshots/2026-10-19-morning
```

A run will never overwrite an existing snapshot. If a source fails, the manifest shows `INCOMPLETE`, and the bridge will refuse it.

Verify the saved data's integrity:

```bash
python nflverse_connector.py verify --snapshot data/snapshots/2026-10-18-afternoon
```

Check whether a snapshot was **captured before** a game cutoff:

```bash
python nflverse_connector.py verify --snapshot data/snapshots/2026-10-18-afternoon --cutoff 2026-10-18T13:20:00Z
```

The cutoff is an **example** for the October 18 Texans–Jaguars 09:30 ET kickoff, not the game time itself. A snapshot captured after 13:20 UTC will correctly report `NOT_VERIFIED_PREGAME`, even if the program can filter out the game being replayed.

## Connect the snapshot to our prediction model

Use a valid game-specific config, with real pregame market and personnel records. The existing historical config is a **partial/UNVERIFIED** example, not a complete pregame audit:

```bash
python run_snapshot_ingest.py \
    --snapshot data/snapshots/2026-10-18-afternoon \
    --config replays/hou_jax_2026_10_18/ingest_config_UNVERIFIED.json \
    --output-prefix outputs/hou_jax
```

The bridge checks the raw file hashes; it overwrites any config-supplied PBP/roster source timestamps with **actual** manifest capture timestamps. It calls the existing strict `ingest.build(...)` pipeline and writes:

- `outputs/hou_jax.audit.json` listing missing fields and provenance failures
- `outputs/hou_jax.INCOMPLETE.json` when mandatory inputs are missing, OR
- `outputs/hou_jax.game.json` only when every field is verified and validated

Some mandatory inputs are not provided by the free sources above: true pass-pressure rates, reliably timestamped official injury adjustments, market quotes with price/time, some special-teams and penalty EPA fields. Their absence blocks the **full** model by design. It does not silently claim that sacks equal pressures or fill odds with fictional lines.

## Additional standards

1. The downloader is a **snapshot collector**, not a historical-publication-time oracle. Today's download is not proof that identical data existed before an earlier game.
2. Local integrity hashes guard against accidental changes, not against inaccurate original data.
3. A strict pregame forecast requires pre-cutoff capture, validated personnel news, and verified prices as well as football data.
4. The provider's license and attribution requirements must be followed for each dataset. See https://github.com/nflverse/nflverse-data.
5. Never treat synthetic unit tests as historical model calibration.

## Test the integration (offline, no fees)

```bash
python -m unittest discover -s tests -v
```

This version passed **35 total tests**, including 9 new connector tests with a stand-in `nflreadpy` object. Those tests do not demonstrate live Internet retrieval or real-game predictive accuracy.

## If live collection fails

- `ModuleNotFoundError: nflreadpy`: install using the free requirements file.
- `NameResolutionError`/`ConnectionError`: the machine cannot reach the nflverse release host. Use a network with access to public GitHub releases.
- `INCOMPLETE` manifest: inspect individual `datasets.*.error` messages.
- `NOT_VERIFIED_PREGAME`: the snapshot was downloaded after the forecast cutoff; do not backdate it.
- Missing `was_pressure`: current-season nflverse participation availability is not established; leave this input missing until a valid, permitted free pressure source is verified.

## v0.3.1: Free direct-file ingestion when nflreadpy cannot be installed

If GitHub works on a computer but this ChatGPT runtime cannot connect, download the open nflverse 2026 season files manually and use `import_local_nflverse.py`. It supports `.csv` or `.csv.gz` without Polars or PyArrow and feeds the existing validation/forecast bridge. Example command and limitations are documented in `replays/hou_jax_2026_10_18/ACTUAL_COLLECTION_REPORT.md`.

**2026-10-18 result:** Live connector was actually invoked. Runtime DNS prevented GitHub / package-index access; zero real 2026 PBP rows collected. 38 tests now pass, but they do not prove remote access or model prediction accuracy.
