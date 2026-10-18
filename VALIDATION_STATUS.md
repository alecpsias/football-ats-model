## v0.3 free nflverse connection (2026-10-18)
- Implemented: official nflreadpy interface for free PBP, schedules, weekly rosters, optional player stats; immutable compressed-CSV snapshot, checksums, per-dataset capture times, data-as-of verification, provenance-aware bridge to strict `ingest.build()`.
- Tests: 35 passed, 9 new connector tests, all local stubbed data; existing 26 regressions passed.
- Live data collection: NOT VERIFIED. This container cannot resolve the GitHub release host and the nflreadpy package is not installed here. No claim is made that real 2026 season play-by-play was retrieved.
- Still missing: legitimate contemporaneous pressure charting, injuries and market provenance, some context EPA inputs. Full-model prediction remains blocked for Texans–Jaguars; no calibration claimed.
- Free-only requirement: no purchases, paid APIs, subscriptions or transactions.

# Implementation status — v0.1.0

**Delivered:** Fixed data contract, executable EPA/SOS-based matchup calculations, QB rushing, pressure/turnover coupling, situational effects, explicit regression, possession simulation, first-half option, ATS price evaluation, adverse-script sensitivity, 16 passing software tests, and chronological backtest/calibration machinery.

**Not delivered or claimed:** Historical training against actual verified NFL/CFB pregame data, calibrated model error, tested predictive performance, trustworthy A/B/C grade thresholds, real-game data collection or current injuries, and proven betting profitability. Neither the model nor its test outputs constitute betting recommendations.

The `examples/` files are wholly synthetic and the backtest refuses them as real calibration data. The blank real-matchup template contains required nulls so that unknown numbers cannot silently become fabricated inputs.

**Next real milestone:** populate and verify time-stamped historical pregame inputs (separate NFL/CFB), replay them chronologically, inspect the holdout results, and revise/version the formulas before relying on the grades. Never use observed final-game information to fill pregame fields.

## Ingestion layer added (2026-10-18)

- `ingest.py` and `fetch_nflverse.py` added; v0.1 prediction coefficients unchanged.
- Tested source-cutoff gate, postgame future-row exclusion, blocked missing pressure and QB identifiers, future-file metadata, validation against the model schema, and an end-to-end synthetic ingestion -> model run.
- 26 software tests pass (16 existing + 10 ingestion); zero real game ingestion validations so far.
- No genuine pre-kickoff 2026 HOU-JAX PBP snapshot was acquired in this offline runtime. A retrospective source publication check still requires independent proof; penalty/ST EPA and separate pressure and injury records remain outstanding. No posthoc scoring changes were made.

## v0.3.1 actual 2026 data attempt (2026-10-18)
- Ran actual 2026 collection CLI; blocked by absent `nflreadpy`. Attempted free installation and official release retrieval; DNS/network restrictions blocked both. No real season PBP file downloaded.
- Web independently established that official 2026 PBP release and public schedule/results record `2026_06_HOU_JAX` exist. Those are not complete downloaded PBP inputs.
- Added `import_local_nflverse.py`, which imports user-supplied official open CSV / CSV.GZ files offline, standardizes, hashes and allows the existing strict replay pipeline to run. 38 total offline unit tests pass.
- Full Texans–Jaguars replay still blocked. Existing EPA-only diagnostic unchanged; no invented stats, postgame tuning, paid provider or fabricated ATS percentage.
