# CFB early-down rushing → third-down distance → sustained drives

**Status:** Research-only within the single Build16 CFB challenger. Nothing in
this experiment changes `cfb/engine.py`, the NFL model, or the standard
`cfb/fresh_matchup.py` call.

## Preregistered hypothesis

Early-down rushing success against the *specific opponent's* historic rush
success allowed can make third downs more manageable, reducing exposure to
long passing downs. A defense that historically allows opponents to chain
first-down series can extend possessions. These **interactions**, rather than
raw rush EPA itself, might improve expected points per drive and the **whole
margin distribution**.

## Source definitions and safeguards

- Early rush success: first down >=50% of needed yards; second down >=70%;
  must be source-observed rush, numeric gain/distance, never imagined.
- Manageable third: third-and-4 or shorter **at the snap**; use play-by-play
  `down` and `distance`. Quarter/game-time and score states stay original.
- Third-down conversion: a next-play same-drive reset to first down; an
  explicit TD/first-down marker; failed fourth-down situation or actual
  turnover. Ambiguous third-down terminal plays are excluded from conversion
  percentages, **not** misclassified as failures. The source's historical
  `firstD_by_yards = False` is not treated as a verified failure.
- Drive sustenance: 3+ source-observed first-down series within one actual
  drive; descriptive, not the number of seconds elapsed or a verified
  official first-down tally. Penalty-first downs may create more series.
- Defense: use the opponent's prior *allowed* early rushing success rate,
  opponent third-down distance distribution and sustained-drive allowance.
- No scoring double count: current Build16 already has rushing EPA and
  third conversion adjustments. Optional additions are shrunk bounded
  deviations from OWN historic share, not extra raw EPA points.
- Future-game outcomes never enter the predictors. Games completed at least
  24 hours previously may enter the next game's historical record.
- The NM State–Sam Houston game that motivated this work is omitted from
  the retrospective study. Do not optimize coefficients on that result.

## Tests

1. Verify third-down attempts and resolvable conversions against external
   university box scores before trusting the new measurement as independent.
   Reports retain unresolved plays and both known and unknown attempts.
2. 4 same-fixture, same-seed ablations:
   `reference`, `distance`, `sustain`, `combined`; exactly **5,000**
   draws per eligible game/variant.
3. Metrics: margin MAE/RMSE/bias, total MAE, outright Brier, central 80%
   margin interval coverage, paired bootstrap 95% improvement intervals.
   No bookmaker ATS assessment is possible without pre-kickoff timestamped
   spread data.
4. Cross-check feature availability and whether source conversion counts
   undercount official boxes. Report missingness explicitly, never zero-fill.
5. Prospective holdout promotion gate: freeze one variant/coefficients after
   earlier training & validation, then compare on new actual pregame
   snapshots. *Not* enough to improve the in-sample fit.

**Never automatically promote.** If evidence is inconclusive or worse,
retain Build16 unchanged and archive the negative result.

## October 22 empirical study — completed, rejected for production

GitHub Actions run [39807158589](https://github.com/alecpsias/football-ats-model/actions/runs/39807158589)
passed all 268 CFB unit tests and executed **160 complete paired games**,
4 variants × 5,000 seeded draws = **3,200,000 actual simulated outcomes**.
The source-cohort's last fixture occurred October 10, before the motivating
October 21 NM State–Sam Houston game (excluded from both replay and fitting).

| Laboratory setting | Margin MAE | +/- paired 95% improvement interval |
| --- | ---: | --- |
| Existing Build16 | 15.35353 | reference |
| Third-down distance | 15.35558 | [-0.02951, +0.02322] |
| Sustained-drive | 15.38519 | [-0.08340, +0.01854] |
| Combined | 15.36998 | [-0.07407, +0.03821] |

No tested version improved margin MAE. Any nominal improvement in outright
winner classification in the latter two scenarios is not an ATS advantage.
The combined variant's typical absolute home PPD change was ~0.022, with
average absolute implied **relative score spread** change roughly 0.29 points.

**University box cross-check:** USC converted 9/13 and SJSU 5/11 third downs.
The same ESPN PBP extraction counted precisely **13 USC attempts with 9
positively identified conversions**, **11 SJSU attempts with 5 positively
identified conversions**, but only **12/13 USC** and **6/11 SJSU**
attempts had definitive 0/1 outcomes (six terminal situations unresolved
across both teams). Do not treat those unresolved third-down snaps as
known failures. The experiment predicts the *manageable third-down share*
from known pre-snap down/distance and uses *sustained-drive series* rather
than building a fake conversion rate from unverified terminal flags.

**Outcome:** DO NOT enable `chain_research` in ordinary games. Tests cover
the code path, not fitted coefficients. The pilot uses retrospective
provider releases, not an independently frozen historical as-of feed, and
contains no authentic pre-kickoff ATS lines. Next attempt should test
verified end-of-drive events, game state, and offensive rush-success
interaction on earlier training periods, with a real later holdout rather
than retroactively tuning this experiment.

Archived report: `cfb/THIRD_CHAIN_VALIDATION_20261022.json`.
