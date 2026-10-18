# Postgame lessons → reproducible model rules (v0.1)

**Meaning of status:** *Executed* means there is real deterministic code, a trace and tests; it does **not** mean the coefficient is historically calibrated or improves forecasting accuracy. *Partial* means the causal relationship is approximated without fully validated coefficients. *Not yet implemented* is explicitly acknowledged.

| Recent postgame investigation | Specific causal lesson | Where quantified | Implementation status |
|---|---|---|---|
| Bills–Rams | Mobile quarterback can create upper-tail rushing possessions; can't project purely from passing averages | `projection`: QB rushing per game, relative usage shift and adjusted rush EPA; `stress_test`: score upside | Executed approximation; no QB-vs-front empirical rush model |
| Memphis–Tulane | Pressure can drive sacks → turnovers → field position → scoring; yardage alone misleads | `projection`: pressure, sacks, interception and fumble probabilities; `simulate`: turnovers and short fields | Partial: correlations beyond drive events remain provisional |
| Jacksonville State–FIU; Kennesaw State–Missouri State | Raw opponent stats and rush success may not transfer; separate play-specific offense/defense and shrink small samples | `projection`: schedule-adjusted pass/rush EPA, sample-size weighting and match-specific models | Executed; coefficients unvalidated |
| Texans–Jaguars | Passing offense can exploit a vulnerable secondary even when the run game is weak; regress early elite red-zone defense | `projection`: independent passing/rushing estimates and red-zone shrink by opportunities | Executed; match coefficients provisional |
| Cincinnati–West Virginia; Western Michigan–Central Michigan | Unfavorable or late-game scripts and ATS backdoors can overturn a correct winner read | `simulate`: margin distribution, `stress_test`: offensive shocks and spread sensitivity | Partial: endgame scripts not yet possession-optimized |
| Bills–Rams; Colts–Chiefs | Superior overall team not equivalent to correct spread side | `run`: both cover probabilities, market break-even, pushes and price sensitivity | Executed; no wagering grade until validated |
| Delaware–Middle Tennessee; Tulane–Memphis | QB availability and late injury news can fundamentally change the game | `validate_game`: 24h personnel-news timestamp; `projection`: documented injury EPA, pressure and pace changes | Executed data gate; impact adjustments supplied, not automatically learned |
| FIU–Jacksonville State | Turnover extremes regress; avoid relying on early unsustainable rates | `rate_match`: sample shrink; `projection`: INT/fumble mix; `stress_test`: turnover-neutral scenario | Partial; requires historical turnover regression validation |
| Sam Houston–Western Kentucky | A half-point in the market can reverse the ATS outlook | `stress_test`: 5 alternate home spreads; vig and push handling | Executed |
| Multiple 2026 audits | Need to evaluate first-half only if stronger than full-game | Optional `market.first_half` and half-possession simulation | Partial: no recommendation without separately fitted first-half calibration |
| Broad prior methodology | Third down, red zone, special teams, penalties, pace, field position, explosives all matter | `projection` and per-game audit | Executed as provisional approximate features |
| All postgame work | Calibration must be tested, not merely proclaimed | `backtest.py`: chronological train/test, holdout probability diagnostics, input as-of checks | Engine exists; no historical real-data validation has been performed |

## Explicitly absent in v0.1

- Real play-by-play EPA ingestion and provider-specific processing.
- Opponent-adjusted personnel-level route/coverage matchup models, O-line lineup snaps and pass-rush win probabilities.
- Full NFL drive-state Markov transitions or rules-accurate overtime and 4th-down coaching simulation.
- Historical fitting of weights and validated ATS confidence-to-grade mapping.
- A verified dated NFL/CFB historical pregame archive; the included example is fully synthetic.
- Automated Circa ticket/handle collection and consensus line-movement history.

**Future changes should add a model version and a failing-then-passing unit/backtest case. Otherwise a discussion of a lesson is not implementation.**
