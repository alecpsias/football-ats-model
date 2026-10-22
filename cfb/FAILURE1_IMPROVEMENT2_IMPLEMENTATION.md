# CFB Failure 1 — Improvement 2 scoring correction

**Status:** Implemented quantitatively in the single Build16 engine as an
opt-in, pregame-fitted rushing matchup response, **NOT promoted** because
a broad, reliable ATS/margin improvement has not yet been demonstrated.

## Two quantitatively tested mechanisms

1. **Previously frozen negative-run coefficient** using own negative
   designed runs versus the opponent's allowed negatives. Same-seed
   5,000-run reference/candidate comparisons on 180 fixtures each year:
   2023 negative by 0.078 MAE points; 2024 negative by 0.034; 2025
   +0.035; 2026 +0.098. The year-to-year sign reverses. This fixed
   scoring coefficient is RETIRED for new forecast shadow runs.

2. **One 2023-fitted opponent-specific rushing residual.** Seven
   features (own-vs-defense early rushing success, own negative run
   avoidance, opposing defense negatives allowed, quality-adjusted
   rush EPA, run usage, and joint severe mismatch) are computed only
   from eligible earlier games. Train ONE ridge model without intercept
   on 180 chronologically eligible 2023 games. Regularization fixed
   at 250 before observing subsequent predictions; fitted model
   frozen byte-for-byte in `FAILURE1_RIDGE_2023_FROZEN.json`.
   Scoring response is capped at +/-3.0 points of signed margin,
   converted to equal/opposite changes in teams' points **per drive**
   before simulating **5,000 genuine games** at the same seed.
   This changes actual score/margin distributions, not merely the
   printed expected margin.

   Entirely later calendar year tests, 180 paired games each:

   | Season | Existing Build16 MAE | 2023-fitted MAE | Gain |
   |---|---:|---:|---:|
   | 2024 | 13.80487 | 13.60782 | +0.19706 |
   | 2025 | 13.50224 | 13.47102 | +0.03122 |
   | 2026 | 14.92766 | 14.93041 | -0.00275 |
   | **Pooled 540** | **14.07826** | **14.00308** | **+0.07517** |

   Pooled 95% paired-bootstrap gain CI **[-0.01807,+0.17002]**.
   Severe pregame screen (312): +0.12153 points,
   CI **[-0.03148,+0.27226]**. No convincing gain. Winner Brier
   falls .213583 -> .211837 and p10/p90 coverage improves 73.7%
   -> 75.2%, but ATS is not measured.

## Source protections and model policy

- Model trained only from 2023 as-of-by-season replay, never from
  target-game margins; past play-by-play captured retrospectively, so
  cannot prove individual 2023 rows were genuinely present at old
  game kickoffs. Do not represent this as a verified live forward test.
- Every scored 2024–26 reference and candidate used identical
  individual game seeds and 5,000 Monte Carlo simulations.
- Real quote captured **before kickoff** is required for ATS
  validation. Missing sportsbook lines are not invented.
- Existing CFB baseline remains frozen. The default Build16
  calculations remain unchanged; `failure1_fit_margin_delta=None`
  means identical results to the previous code.
- No second active challenger: only **one** previously-fitted shadow
  of Build16 is kept in new prospective forecasts; the earlier
  unvalidated negative-run shadow is retired for new games. Archived
  prior results remain immutable.
- Shadow stores source-data manifest SHA256, 2023 frozen coefficient
  SHA256, seed, time and actual 5,000-draw distribution. Grader
  compares earliest pre-kickoff frozen forecasts and genuine sportsbook
  capture time against later final scores, NEVER reruns after results.
- Neither scoring residual nor any alternative receives an A-grade,
  cover probability calibration, or automatic active promotion.

**Improvement 2 conclusion:** A quantitative mechanism is implemented
and tested, but its evidence **fails the predetermined performance
promotion gate** (MAE benefit <0.2, pooled 95% interval includes zero,
and no certified timestamped forward ATS record). Therefore it is
**not currently enabled for standard ATS picks**. Improvement 1, the
mandatory rushing audit plus high-severity 5,000-run stress tests,
remains active.

Full results and source references:
`cfb/FAILURE1_IMPROVEMENT2_VALIDATION_20261022.json`.
