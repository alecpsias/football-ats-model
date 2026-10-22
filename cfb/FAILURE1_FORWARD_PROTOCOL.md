# Failure 1 forward validation — negative designed-run avoidance

Status: **one shadow ablation of the existing single Build16 CFB Challenger**,
not a separate live challenger, not a new baseline. The only comparison
eligible for a future promotion decision is **Build16 unchanged versus the
same Build16 plus its exactly frozen negative-run adjustment**.

## Source for the score-state repair

The actual October 22, 2026 public ESPN PBP release has 505 columns,
88,293 events and nonmissing `pos_score_diff_start` on all 88,293.
Provider fields `start.pos_score_diff` and
`start.pos_team_score - start.def_pos_team_score` match that column on
**all** 88,293 events: zero missing and zero disagreements in the
source-level double-check. See archived Actions probe
`cfb-failure1-score-probe`.

The historical 2025/2026 zero qualifying game-script matches were caused
by a **research aggregation bug**: for every game, even if a team did not
rush while down nine or more, the score-state grouping correctly emitted
a zero-denominator count. The aggregator then rejected all history
containing *any* zero denominator. It now ignores valid zero-count
game-state entries as having no trials, while rejecting impossible
numerators, missing/contradictory play-score fields, and invalid rates.
No final-score-derived score states or backfilled rushing plays.

This repair should be validated using its new 180-game matched historical
study before interpreting trailing resilience.

## Frozen candidate and prospective evaluation

- Frozen coefficients exactly as recorded on 2026-10-22 in
  `cfb/rush_severity_research.py`: negative-run relative rate factor
  -1.2, nonpositive-run relative rate factor -0.35, prior of 160
  early rushes for reliability, and absolute PPD effect cap 0.17.
  These are fixed *hypotheses* and were **not fit on untouched future games**.
- Negative designed rushing run is yards gained <0 on provider `rush`
  (excluding QB sacks classified as pass concepts). Nonpositive run is
  <=0. Same offense-versus-opponent defense prior-game inputs.
- Existing daily/weekly free CFB snapshot pipeline
  `.github/workflows/cfb-free-data.yml` freezes both ordinary Build16
  and an **opt-in within-Build16 negative-run shadow** before actual
  scheduled kickoff. Historical results are limited to >=24 hours before
  the current source snapshot; matchup itself is never used.
- Both variants use **5,000 actual Monte Carlo outcomes per game** with
  exactly the same seed, and unchanged scripts/existing defensive
  adjustments. Standard Build16 is the ONLY live challenger.
- Pre-kickoff source manifest, code configuration hash, generated time,
  seed and mean/p10/p50/p90 margin are frozen together. The first
  original prediction for each game must be independently archived.
- Public ESPN odds are **captured, not inferred**. If an observable
  signed spread and American prices are unavailable ahead of kickoff,
  no ATS record is created. Quote observed timestamp is authentic
  API capture time, but original sportsbook publication time is
  **unverified**. Label as provisional research, not a sharp-book
  closing-line estimate or certified ATS pick.
- Postgame processing grades only the **unchanged frozen predictions**
  against newly observed completed-game scores and the one
  same signed pre-kickoff quote. It must NEVER resimulate once actual
  final scores are known. Compare paired margin MAE, residual bias,
  80% predictive interval coverage, and paired ATS classification
  without pushes/no-edge, with distribution/rating calibration
  explicitly unvalidated.
- Freeze now; keep all subsequent prospective games fully separated
  from calibration. **No auto-promotion** until at least 120
  genuinely paired forward margin cases, 40 cases from separate game dates
  in a later untouched time block, and appropriate improvement across
  entire slate plus high rushing mismatch stratum; ATS requires enough
  independently captured real-line cases to support a meaningful
  uncertainty interval.
- Do not promote on a tiny 0.1-pt retrospective effect, a winner-direction
  gain, a single NM State–Sam Houston replay, or data selected using
  postgame information.

## Caveats

The underlying past-game release is a real October 22 snapshot but it
does not prove which *historic* individual rows were originally published
at each past game's kickoff. The new forecasts themselves are captured
ahead of future game start, resolving this particular lookahead issue
for prospective evaluation. Feature-source clock and missingness gates
must still pass. The free-source odds endpoints are not guarantees of
availability; absent odds do not turn into hypothetical ATS statistics.
