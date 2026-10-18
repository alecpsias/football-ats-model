# v0.3.0 — Free nflverse Integration

- Added `nflverse_connector.py`: official nflreadpy play-by-play, schedules, weekly-roster and optional player-stat readers; immutable compressed-CSV snapshots; SHA-256 hashes; time/cutoff verification; no paid API calls.
- Added `run_snapshot_ingest.py`: loads verified files into `ingest.build` with manifest-authenticated capture timestamps. Required market and personnel data remain explicitly missing where absent.
- Added `requirements-nflverse.txt` and `NFLVERSE_CONNECTION_GUIDE.md`.
- Updated `ingest.load_table` to read `.csv.gz` so free dependencies can be used without optional Parquet conversion.
- Added nine offline integration tests; all 35 model and ingestion tests passing in this execution environment.
- Source-download limitation: outbound GitHub access and the live package are unavailable from this execution environment; live download was attempted but not completed. No actual NFL 2026 dataset is included.
- No predictive coefficients changed, and no betting probabilities are claimed calibrated.
