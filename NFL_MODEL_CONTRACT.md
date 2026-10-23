# NFL Model Contract — Baseline + One Evolving Challenger

This is the **permanent model-development policy** for the NFL engine. The
college-football model remains independently maintained under `cfb/`.

## Exactly two active forecast outputs

1. **Baseline:** existing locked v0.8.1 scoring core from `model.py`.
   Preserve it as the fixed experimental control; never silently recalculate
   or replace a previously frozen result.
2. **One challenger:** currently v0.11.0, defined by
   `nfl_stateful_candidate.py` and the constant in
   `nfl_challenger_policy.py`. This single challenger accumulates improvements
   from every NFL postgame audit and prior research version.

There is **no separate v0.9 runner**. Its useful state/pass/turnover/field-position
mechanisms survive as extracted features and scoring components used by the
v0.11 challenger. Git commit history preserves the old implementation.

For any **future** revision (e.g. v0.12):

- Update **the one existing challenger implementation in place** and bump
  `ACTIVE_CHALLENGER_VERSION`; **do not** run v0.10 and v0.11 together as
  active challenger candidates.
- Keep the baseline unchanged until the user expressly authorizes baseline
  retirement. A challenger version may change without changing the baseline.
- Preserve previously recorded, time-stamped forecasts and exact SHA-256 bytes.
  They are immutable **archive entries**, not additional active models.
- Each new forecast JSON must contain exactly one top-level `challenger`
  object, with `version`, `candidate`, `status`, and limitations.
  `require_single_active_challenger` rejects multiple sections and implicit
  baseline promotion.
- If a required challenger input is missing, report `candidate: null` and a
  reason; do **not** fall back to a retired contender or invent a statistic.
- The evaluator chooses a single challenger per previously frozen game. Legacy
  records are read-only; old v0.9/v0.10 fields are supported for historical
  comparison, but cannot re-enable the retired model. Segment accuracy **by
  version**; never pool unequal versions and claim a calibrated improvement.
- Use free/open sources only. Without verified **frozen signed spreads** and
  sample-size/chronological validation, there are no calibrated ATS
  probabilities, A-grade selections, or ROI claims.
- Distinguish individual team-score error, total error, margin error,
  missing-projection rates, and turnover/drive-script calibration.
- NFL tests run only root NFL suites; never import or alter the separate CFB
  engine.

**Current operational structure:** `v0.8.1 baseline + v0.11.0 one challenger`.

## v0.11 starter-specific QB-form adjustment

The one current NFL challenger, and **not the baseline**, adds a bounded
opponent-adjusted quarterback performance residual in points per drive.
It includes sacks and QB scrambles, finds the likely incumbent from the last
completed game (not an assertion of current starter confirmation), shrinks
QB skill relative to the team's observed QB average, and applies a smaller
one-week recency term. Regularization constants (30 skill dropbacks and
200 form dropbacks; opponent-defense 150-dropback prior) were selected during
an earlier limited Weeks 4-6 walk-forward QB-EPA study; they are not
independently calibrated to winning margins or ATS returns. Effect is
**0.5 × QB EPA residual × team dropback share × plays per drive**, bounded at
±0.40 PPD to avoid adding the entire QB effect on top of already-modeled team EPA.
It intentionally does not separately add scramble rush EPA or change turnover
probabilities, avoiding unsupported double counting. If the source columns,
completed games, recent QB usage or sufficient sample are unavailable,
record an explicit NOT_APPLIED reason and use zero residual, not an
imaginary replacement quarterback. Missing mandatory QB source columns
make the challenger unavailable while the baseline still runs.

All historic v0.10 predictions remain immutable archived versions, evaluated
separately from v0.11. Only **one** challenger is invoked for new forecasts.
