# Mandatory archived investigation gate (CFB)

This is the **authoritative, fail-closed path** for CFB ATS rankings and frozen-record retrieval. It adds a gate to the existing Build16 process; it does **not** change simulations, weights, source data, challenger count, or any previously frozen record.

## Contracts

1. **Register the game at investigation start.** Each date has `cfb/investigation_registry/YYYY-MM-DD.json`. Register a unique provider game ID, actual teams, a dated matchup workflow, expected frozen audit path, model version, and provenance field paths. A workflow is not itself an archived ATS selection.
2. **Run the established investigation.** Historical statistics first, then the required 5,000 baseline and challenger simulations. Keep the original frozen evidence unchanged.
3. **Freeze and pin.** Write `cfb/frozen_pregame_audits/YYYY_MM_DD_<slug>.json` with pregame capture timestamp, authoritative kickoff, team/game mapping, simulation count, ATS directional pick, conditional spread, grade, confidence, and explicit PASS/conditional recommendation if recorded. Compute the Git blob SHA with `git hash-object <file>`, and set `blob_sha` in that game's registry entry. Existing pinned records must not be edited for postgame reconstruction.
4. **Require the gate before output.** A full-slate ranking comes *only* from a `PASS` gate result. A single pregame/postgame investigation must verify its registered original archive using `--game-id`. Do not substitute memory, historical hindsight, or an unpinned artifact.
5. **Fail closed.** Missing registry, missing audit, changed bytes, duplicate game, mismatched teams/IDs, missing dated workflows, unregistered dated workflows, invalid timestamps, non-5,000 draws, or missing ATS selection/confidence/grade all return `BLOCKED` and a nonzero exit status. A blocked slate has `ranking: []`. Diagnostic counts must never be presented as an official ranking.
6. **Separate directional picks from wagers.** Missing an explicit bet decision does not convert an ATS lean into a recommendation; it is labeled `NO_EXPLICIT_WAGER_DECISION_IN_FROZEN_RECORD`. The underlying model's raw cover frequencies remain uncalibrated.

## Run the mandatory checks

```bash
python -m unittest cfb.test_investigation_gate -v

# Verify this day's *complete* investigated slate:
python -m cfb.investigation_gate --date 2026-10-24 --out cfb_artifacts/oct24-verified-ranking.json

# Verify one archived investigation before analysis or postgame review:
python -m cfb.investigation_gate --date 2026-10-24 --game-id 401858498 --out cfb_artifacts/rutgers-record.json

# Verify every future dated CFB workflow is registered and fully archived:
python -m cfb.investigation_gate --all-dates-from 2026-10-24 --out cfb_artifacts/all-future-rankings.json
```

The `CFB mandatory frozen-investigation ranking gate` GitHub Actions workflow runs tests and all-date reconciliation when relevant code, workflows, registry, or audit files change. It preserves the failure diagnostic as an Actions artifact. Red status is intentional while any required game remains unarchived.

## Initial Oct 24 reconciliation

The Oct 24 registry declares **13 investigations**. At rollout, **12 individual ATS audit files are present and pinned**. Oregon at Illinois (provider ID `401858495`) has a *successful* pregame 5,000-draw GitHub workflow, including an uploaded simulation artifact (run `40107807252`, artifact `12565885092`), but lacks a pinned individual **ATS decision / grade / confidence** in `cfb/frozen_pregame_audits`. It therefore remains **blocked**. Do not infer the historical chat pick from a raw simulation and do not silently omit this game.

California at SMU is graded at the verifiably quoted **Cal +18.5 (C, 5.7/10)** level; +19.5 (borderline B, 6.0/10) was conditional and not independently confirmed available. This distinction is retained in the original frozen record.

## Limits and governance

This code enforces **repository-produced outputs**. It cannot alter the ChatGPT runtime to physically prevent an unsupported free-form answer. For a verifiable output, require the GitHub gate result and its source record links rather than accepting an independently reconstructed chat ranking.

A registry entry must be created for a **chat-only investigation**, too: software cannot discover an unlogged chat investigation without a durable event. Adding a dated GitHub workflow without a corresponding registry will trigger a failure, but a chat-only statement with no GitHub artifact will not be discoverable automatically.

This version covers CFB; do **not** imply that it implements NFL archival enforcement. The NFL archive schema needs its own adapter and completeness rules before making that claim.
