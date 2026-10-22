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
- "Severe joint edge" is a preregistered historical offense early-run
  success and opposing defense allowed early-run success both >0.50.
  The **strict large advantage cohort** uses max joint excess >=0.08.
  Also report upper-quartile ex-ante joint signal versus remainder.
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
