# Live 2026 nflverse collection attempt — Oct 18, 2026

## Outcome

**2026 source release confirmed; actual season PBP files NOT collected.** An unsuccessful network call is not a data download. No real 2026 PBP rows have been verified or passed into the full prediction engine. No 2026 ATS probability or new score is claimed.

- Official source: https://github.com/nflverse/nflverse-data/releases/tag/pbp
- Schedule/results source: https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv
- Observed schedule record: `2026_06_HOU_JAX`, neutral site, Houston 34–31, listed JAX -2.5 and total 42.5. The schedule record is **postgame** and cannot establish when its market numbers were first publicly available.
- Current local execution environment: package index and GitHub DNS access blocked; failed direct download. The existing `nflreadpy` collector cannot run without the free package.

## What was actually executed

1. `python nflverse_connector.py collect --season 2026 --out /mnt/data/nflverse_real_2026_attempt`: rejected due to missing `nflreadpy` package (log in this directory).
2. `pip install nflreadpy pyarrow`: unavailable package-index DNS; no packages installed.
3. `curl` for the public 2026 PBP release: GitHub DNS resolution blocked.
4. Managed file downloads: rejected/failed.
5. `python -m unittest discover -s tests -q`: **38 tests passed**, including 3 new tests for direct import of official CSV files (tests are synthetic; not evidence of a 2026 live data download).

## Free fallback added

`import_local_nflverse.py` imports actual public CSV / CSV.GZ files directly, so `nflreadpy`, Polars and pyarrow are not required once those files are present. It validates required schemas, rejects duplicate plays and season mismatches, archives immutable snapshots with SHA-256 file hashes, records actual capture times and works with the existing `run_snapshot_ingest.py` bridge.

Example from the `football_ats_model` folder:

```bash
python import_local_nflverse.py --season 2026 \
  --pbp /path/to/play_by_play_2026.csv.gz \
  --schedule /path/to/games.csv \
  --roster /path/to/roster_weekly_2026.csv.gz \
  --out data/snapshots/actual-2026

python nflverse_connector.py verify --snapshot data/snapshots/actual-2026 \
  --cutoff 2026-10-18T12:30:00Z

python run_snapshot_ingest.py \
  --snapshot data/snapshots/actual-2026 \
  --config replays/hou_jax_2026_10_18/ingest_config_UNVERIFIED.json \
  --output-prefix outputs/hou_jax_actual
```

Use the corresponding CSV or CSV.GZ assets in the free nflverse GitHub releases: `pbp/play_by_play_2026.csv.gz`, `schedules/games.csv`, and `weekly_rosters/roster_weekly_2026.csv.gz` (verify available formats on their release pages). For a retrospectively downloaded file, the cutoff check must report **NOT_VERIFIED_PREGAME**. The pipeline filters post-cutoff game rows but cannot prove original file publication timing without dated archives. The strict model will still flag unresolved pressure, market timestamp, personnel, penalties and special-teams fields; do not fill these with made-up values.

## Independent partial replay unchanged

The earlier verified-feature-only EPA diagnostic projected **Jacksonville 25.40, Houston 19.77**, without full scoring simulation or confidence estimate. Actual result was Houston 34–31. This diagnostic did not get new real-PBP inputs, and no parameters were tuned after the game.

**Next step strictly requiring externally accessible raw data:** provide/download the free official nflverse 2026 PBP, schedules, and weekly roster CSV files. No subscription or transaction is required.
