# CFB Failure 1 — opponent rush mismatch severity audit

**Main objective:** Determine whether nonlinear offense-versus-defense
rushing success gaps, rushing-negative-play avoidance, and the ability
to continue productive rushing when trailing improve the **full final
margin distribution**, especially for games with a materially large
pregame rushing edge.

This is an extension of the same independent Build16 CFB challenger,
**NOT** an additional active challenger. Baseline, NFL model, published
pregame Build16 settings, and expected 5,000-run protocol remain frozen.

## Definitions and source protections

- Designed rushing uses the provider's original `rush=True, pass=False`
  concept for efficiency, **excluding sacks**. Official NCAA rush counts
  elsewhere include sacks; neither labeling rule is silently changed.
- First-down rushing success = >=50% of yards-to-gain; second down >=70%.
  Valid, observed gains/distance only; exclude missing.
- A negative early rushing play = observed yards gained <0;
  nonpositive/stuffed = yards gained <=0. These are not synonyms.
- Situational rushing performance uses **recorded pre-play**
  `pos_score_diff_start`, never the final result: trailing by 9+
  versus within 8, with the opponent's allowed rate under matching
  offensive score state.
- Initial V1 "severe joint edge" required both offensive and
  defensive-allowed early-down success >0.50 and a strict excess >=0.08.
  In the 2026 source, only 2 of 360 observed team sides cleared the
  positive-signal threshold, so **no** severe games qualified.
- Exploratory V2 cutpoints were selected using only observed marginal
  source feature distributions (offensive median ~0.43, defense-allowed
  median ~0.38), NOT independently specified before viewing those
  distributions: offense >=0.42 and defense-allowed >=0.38, with a
  strict joint-excess >=0.04. Report 23 strict games and an exact
  45-game top-ranked feature quartile (remaining 135 games). All
  retrospective V2 performance comparisons are exploratory; cutpoints
  are not evidence of untouched validation.
- To avoid adding rushing EPA again: scoring changes are modest,
  bounded *counterfactual interaction* hypotheses, not duplicate raw EPA.
  Severity's nonlinear coefficient, negative-play coefficient, and
  trailing score-state coefficient are NOT statistically fitted.
- The trailing resilience change affects only **actual simulated**
  possessions in which that side is trailing >=9. It does not reward
  the team on tied/leading drives or modify regular Build16 behavior.
- All inputs must precede kickoff under a minimum 24h historical embargo.
  Do not use the NM State–Sam Houston motivating game (401871070), or
  any game from Oct 21 onward, as fitting/test evidence. Source
  observations are from retrospective public release, however, without
  provable original per-row publication timestamps.

## Four within-challenger mechanisms

`severity`: nonlinear jointly favorable early-down rushing success.

`negative`: opponent-specific negative and nonpositive rushing
play prevention, separate from overall rushing EPA.

`resilience`: source-observed trailing-state rushing effectiveness,
changing only simulated trailing possessions.

`combined`: all three, each bounded. Compared to `reference`
unchanged Build16 on identical games, seeds, and 5,000 simulations.

Metrics: overall and strict severe / top-quartile / other-matchups
margin MAE, RMSE, bias, total MAE, winner Brier and p10–p90 margin
interval coverage. Paired bootstrap 95% intervals assess instability.

**Decision rule:** Do not adopt unless the improvement survives a
predeclared, chronologically later, independent forward holdout and
does not worsen broad-slate accuracy or interval coverage. A historical
retrospective experiment is a pilot, not certified ATS validation.
No line-backed ATS probability can be claimed without timestamped,
pre-kickoff actual sharp-book spreads.

If no gains, archive exact negative result and leave Build16 unchanged.

## October 22 empirical diagnostics

The source-observed early rush success rates were considerably lower
than V1's absolute 50%/50% assumed threshold (offense median 42.86%,
defense-allowed median 38.49%). Under V2's feature-derived cutpoints,
the source-matched 2026 historical study included 180 games:

- Existing Build16 margin MAE: 14.92766.
- Severity nonlinear v2: 14.90531, paired improvement +0.02234,
  bootstrap95 [-0.02881,+0.07608]. Not statistically persuasive.
- Negative-designed-run prevention: 14.82965, improvement +0.09801,
  paired bootstrap95 [+0.01251,+0.18416], *exploratory* and
  multiple-comparison-sensitive.
- Strict severe group 23 games: Build16 15.02413 versus severity
  14.87791, +0.14622 error improvement, 95% interval crosses zero
  [-0.20310,+0.49216]; **winner direction got worse**
  (78.26% to 69.56%). No credible ATS edge from this subgroup.
- Exact 45-game top-quartile group: mean absolute margin 16.83587
  versus 16.71798 (severity), an inconclusive +0.11789.
- Score-state resilience and full combined mode: **ZERO valid
  matched games** at the prescribed pre-play-score sampling
  minimum. Do not claim these were validated.
- The 180 eligible games were played before the motivating
  NM State–Sam Houston game, which was explicitly excluded;
  however the archived data capture is retrospective and its
  exact historical row publication timing was unverified.

**Promotion decision:** neither severity nor negative-run prevention
is activated in the standard Challenger. The apparent 0.098-point
negative-play improvement deserves a distinct-season and later
forward-confirmation study; threshold and coefficient fitting must
happen only on chronologically earlier data. Do not substitute
unobserved trailing plays or fit against the motivating outcome.

See `cfb/FAILURE1_SEVERITY_VALIDATION_20261022.json`.
