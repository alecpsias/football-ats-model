# Default CFB Build16: pregame player-availability corrections

**Status:** Active as a *source-backed conditional component* in the single
Build16 challenger. The independent NFL engine and original CFB baseline
are unchanged.

## Entrypoints

- `cfb.fresh_matchup.analyze` freezes qualified completed prior games, obtains
  player-use statistics, captures a linked-conference report **during the run**
  from `cfb.live_availability` when no locally supplied official archive is
  present, and derives `build16_active_personnel_evidence`.
- `cfb.personnel_active.estimate` uses historical *identified player* play
  counts, per-play EPA, last-three-game opportunity shares, and same-team
  replacement observations. These are regressions with finite sample priors,
  not claimed causal player values.
- `cfb.mechanism_build16.simulate_build16` applies the regressed incremental
  points/drive to the appropriate current-team offense before the single
  source-timed possession and scoring simulation. It also supports kicker
  fallback only where a specific historical FG kicker's PBP actor ID matches
  a reported-out primary kicker.
- `personnel_enabled=False` is a paired diagnostic ablation; it leaves the
  historical default Build16 inputs untouched.
- The research workflow `cfb-personnel-build16-default.yml` checks the full
  CFB regression suite and compares exactly 5,000 matched-seed draws with
  and without the personnel layer on the frozen NC State–Stanford pregame.

## Strict safety and traceability

1. No future or target-game play-by-play enters the feature history. The
   ordinary caller applies the 24-hour pregame embargo.
2. A report that arrives after kickoff is rejected, not backdated.
3. The free website is a **third-party transcription** linked to its conference,
   not independently certified official data or proof of today's starters.
   Its response is captured with URL, actual fetch time and SHA256; a missing
   page means no inferred injuries. Reports inside `cfb/availability` remain
   source-constrained but cannot self-attest an invented capture time.
4. Player IDs are matched conservatively. A name without an unambiguous
   roster identity does not receive a numerical adjustment.
5. For QB changes, the alternate must have measured prior passing data.
   A proposed replacement is **not** claimed to be the actual starting QB.
6. For RB/WR/TE, loss magnitude is restricted by observed personal production,
   teammate performance, recent opportunities and explicit shrinkage and caps.
   The team's pregame projection already contains many previous absences;
   there is no artificial whole-season stat subtraction.
7. Kicker effects require observed PBP-attributed individual kicking
   history; team kicking statistics alone do not identify an absent kicker.
8. Missing offensive-line blocking, defense/snap, pressure and special-teams
   participant measurements are **unmodeled and explicitly reported**. No
   flat injury points are invented for OL/DL/LB/DB/special teams.
9. The baseline is always unchanged. There is one evolving CFB challenger.
   No large historical validation gate blocks future measured improvements,
   but raw simulation ATS probabilities are not independently calibrated.

## Next postgame evaluation

Check both ATS at the frozen market quote **and** margin/total error using
the exact frozen pregame snapshot. Preserve any source/report or missing-role
limitations and record which actual roster played; do not adjust frozen
predictions retrospectively.
