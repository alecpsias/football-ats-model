# Failure 1 — empirical rushing-to-possession scoring correction

**Scope:** Independent CFB Build16 only. The NFL model and CFB baseline are untouched.
The correction is an opt-in within the **existing single Challenger** via
`simulate_build16(...,possession_response=...)`; it is NOT active in the
standard CFB pregame command, and no additional published challenger exists.

## What has actually been implemented

The old model already opponent-adjusts rushing EPA and third-down conversions.
To avoid counting those gains twice, this implementation isolates additional
**pre-kickoff opponent-specific sustained-possession residual** using:

- Actual designed rushing success on early downs (first down >=50% to gain,
  second down >=70%) from completed past games
- The upcoming defense's *observed allowed* rush success and rate of drives
  sustaining **three or more distinct first-down series** (not just yardage)
- The offense's own historic rate of sustained drives and share of designed
  rushing attempts
- Actual observed third-and-manageable rates and opponent allowed rates
- **Real recorded, independently cross-checked regulation possession elapsed
  times** for both teams. A game lacking verified clocks contributes NO guessed
  seconds and NO shared-tempo adjustment

A capped, historically fitted residual changes home/away points per drive
before *actual* sequential scoring draws. Verified possession-clock effects
change expected drives for **both teams together**, since they share the
same regulation clock. Real football plays and drives are not invented.

The model uses a 24-hour embargo on previous completed games; none of the
target game's final outcome or measured runs is a pregame feature.
The public historical data were collected retrospectively, so this does not
certify source row publication timing at each historic kickoff.

## Locked training and holdout

- Trained precisely one ridge residual coefficient on **180 games from 2023**
  with 5,000 base simulations per historical fixture.
- 2023 coefficient `-1.662395245` points per extra expected sustained drive;
  it has a **negative** sign. This is the reason it is not acceptable to force
  a positive rush-scoring weight merely on intuition or one motivating game.
- Held coefficient unchanged against 180 games each from **2024, 2025,
  2026**, each with **5,000 baseline and 5,000 possession-adjusted draws at
  exactly the same game seed**.
- All three later years showed **tiny, statistically inconclusive** margin
  MAE gains, respectively +0.051, +0.015, +0.068 points per game.
- Pooled 540 games: Build16 margin MAE 14.0783 versus new 14.0333,
  apparent improvement +0.0450, paired bootstrap95 [-0.0123,+0.1003].
- Severe rushing interactions, 406 games: apparent gain +0.0666,
  bootstrap95 [-0.0082,+0.1345].
- With source-verified opponent drive clock on both teams, 305 games:
  apparent gain +0.0465, bootstrap95 [-0.0289,+0.1147].
- Normal Build16 and the baseline are **unmodified by default**.
  The new research-only `possession_response` requires independently
  frozen, previous-season learned evidence, and cannot be combined with
  another experimental scoring adjustment.
- No 2023–26 retrospective spread can substitute for an actual documented
  *pre-kickoff* public sportsbook quote for this exact model version;
  prospective ATS evidence is not available yet.

## Decision

The **physical feature extraction, fitted scoring conversion, clock
coupling, simulation integration, 307 regression tests, and all holdout
studies have been implemented and executed.** The idea was *not*
promoted to active predictions because the data did not establish a reliable
improvement, particularly given the fitted negative residual sign and
missing prospectively validated ATS evidence.

**Do not advertise Failure 1 as fixed.** Improvement 1 still enforces
the mandatory rushing mismatch audit and 5,000-run adverse-scenario gate.
The next active Build16 main-line estimate remains unchanged. Further work
should test verified first-down-chain probabilities against new actual
pregame outcomes and their original observed line, without fitting new
coefficient variants on any of these already-inspected holdouts.

Original runs:
- [2023 fit](https://github.com/alecpsias/football-ats-model/actions/runs/39843826367)
- [Three later seasons](https://github.com/alecpsias/football-ats-model/actions/runs/39844166211)
- [Prespecified 540-game decision gate](https://github.com/alecpsias/football-ats-model/actions/runs/39845044570)
