# CFB rushing-interaction test protocol — research, not model promotion

**Objective.** Test whether the *same* independent CFB Build16 challenger
benefits from changing expected run/pass share, drive length and available
possessions when a specific offense faces a specific rushing defense.
**Do not alter `cfb/engine.py`, `cfb/research_backtest.project` or the NFL
model. Do not deploy an unvalidated new challenger.** Build16 is still the
single active CFB challenger; all new arguments are opt-in laboratory toggles.

## Hypotheses and units

1. **Rush matchup:** Prior-game-only leave-opponent-out pass/rush EPA, offensive
rush share, defensive rush share allowed; reliability-shrink both. More favorable
rush/pass EPA and opponent rush tendency may increase offensive rush share.
2. **Scoring interaction:** Build16 already blends both rushing and passing EPA.
Only the **incremental shift in the share** times the difference in matched
shrunk rush/pass EPA changes expected PPD. Directly adding rush advantage
to PPD again is prohibited.
3. **Drive length:** Prior offensive and opponent defensive **scrimmage plays
per drive** estimate future plays/drive. This is **not** actual elapsed clock.
Test PBP quarter/clock parsing versus official time of possession before
calling this clock-consumption evidence.
4. **Tempo:** Both offenses share a game clock. A change in expected plays
per drive changes the **same** number of paired regulation possessions for
both sides (capped). Do not award an extra possession to only one team.
5. **Margin distribution:** Run the same sequential CFB game-script engine,
same seed, 5,000 draws per scenario. Evaluate the entire margin quantile
distribution, not only predicted winner or mean score.

The staged research coefficients are stated in
`cfb/rushing_interaction_research.py` as **unfitted hypotheses**, never
identified from the already-examined NM State–Sam Houston game.

## Four within-challenger ablations

- `reference`: Existing Build16, unchanged.
- `mix`: Opponent-specific rushing/pass selection and mix-only efficiency
residual; no tempo change.
- `pace`: Observed matchup plays/drive and common possession change only.
- `combined`: Both changes, with game-state passing/turnover exposure.

They are **NOT** four published challengers. The user continues to see one
unchanged baseline and one current Build16 challenger.

## Test process and pass/fail criteria

**Phase 0 – data-contract:** Confirm source PBP drive IDs, scrimmage run/pass
classifications and elapsed-clock fields using official game-by-game data.
Where kickoff, punt or penalty-only possessions cannot be resolved, mark
missing rather than imputing fictitious drives. Run same-seed determinism,
home/away neutral-site swap, no-score-leakage and baseline immutability tests.

**Phase 1 – retrospective exploratory:** Rebuild 2026 team observations as
of each historical kickoff with a 24h final embargo; preserve source hashes
and game IDs. Collect only the earliest consecutive eligible scheduled
fixtures, without outcome or spread-based selection. Pair all four
simulations on the **same** fixtures and 5,000 seeds/draws each. Measure:
- Margin MAE, bias and RMSE
- Scoring total MAE
- Raw outright Brier score (winner probability is not ATS probability)
- 10th–90th margin interval empirical coverage and excessive tail width
- Observed versus projected rush share, plays/drive and regulation possessions
- Error in high-rushing-advantage vs low-advantage matchup cohorts

**Phase 2 – fit without leakage:** Estimate coefficients using ONLY a
chronologically earlier, sufficiently large training period. Select ONE
variant on a later validation period. Freeze that choice and all shrinkage
parameters **before** reading an untouched test cohort. Require improvement
over the **same** current Build16 comparator, not over a strawman baseline.

**Phase 3 – prospective confirmation:** Because the NM State game motivated
these hypotheses, the current 2026 historical snapshot is NOT a genuinely
untouched holdout. Keep the existing challenger live until there is adequate
new pre-kickoff frozen forward evidence. Suggested promotion gate:
at least 120 completed eligible paired forward FBS games; >=0.3-point lower
paired margin MAE; 95% paired-bootstrap interval excluding zero in the
improving direction; no material worsening in outright Brier score or
margin-tail coverage. ATS claims require timestamped pregame sportsbook
lines available before kickoff and an independently sized ATS holdout.
Do not optimize on the once-unblinded forward test. If no improvement, reject.

**Stress and negative controls:** Opponent-label shuffle should kill
matchup-specific signal; substitute zero shift to reproduce reference;
run +/− turnover, tempo and run-share shocks; check neutral-site team-swap
symmetry; check small-sample loss of evidence does not turn into numeric zero.

**Current status:** The study code and opt-in simulator path are staged;
unit tests establish software correctness only. No fitted coefficients,
no validated gain and no automatic deployment.

## October 22, 2026 — verified-clock, NCAA box-stat audit outcome

- 2026 ESPN source classifications must be kept in TWO views:
  original provider `pass`/`rush` EPA concepts for the existing
  scoring model; NCAA **box-statistical** attempts assign provider sacks
  to team rushing rather than pass attempts. Do **not** relabel historical
  EPA, sacks, pressure, passer EPA or TD play type.
- Exact university-box verification: USC 40 rush/36 pass; San Jose
  State 24/32; TCU 39/32; North Carolina 29/28. All eight counts
  matched after reclassifying 2/1/3/3 verified quarterback sacks.
- Possession clock: provider `drive.timeElapsed.displayValue`, checked
  against drive start/end periods/clocks when present. Separately flag
  up to two source-only elapsed drives per game. Two official game
  possession totals within 13 seconds or less per team. Do not claim
  510/510 games are complete: 321 passed, 189 partial.
- Measured seconds, NOT scrimmage plays per drive, now feed the
  **opt-in** pace and combined research modes. These fail closed if
  pre-kickoff verified clock history is absent.
- Chronological retrospective 92-game paired comparison, 5,000
  actual draws per variant: original Build16 MAE 13.7508; mix 13.7549;
  measured-clock pace 13.7754; combined 13.7716. None improved.
  Bootstrap 95% intervals of paired gains included zero. The
  stronger rushing model is **NOT DEPLOYED**.
- Model and simulation independence preserved; unchanged baseline and
  unchanged default Build16. Original user-investigation process and
  5,000-run protocol are unaffected.
- All 264 CFB unit tests passed in the research workflow. The
  evaluation used a later archived source without independently
  proved original row-publication times. It has NO bookmaker ATS
  calibration and is not a prospective blind test.

Permanent reproducibility receipt:
`cfb/CLOCK_RUSH_VALIDATION_20261022.json`.
Github Actions run `39801019522`, artifact
`cfb-verified-clock-100-game-study`. Due to only 92
genuinely eligible paired historical games, no missing clock data
were substituted with zero and the study did not manufacture 100 games.

**Current decision:** Reject these uncalibrated numerical effects for
the live Challenger, retain the research code for prospective testing,
and continue searching for better validated run/pass and scoring
interactions. Pre-declare any future coefficient revision rather
than fitting the postgame NM State result.
