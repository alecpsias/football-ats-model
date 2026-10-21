# NFL Model Contract — Baseline + One Evolving Challenger

This is the **permanent model-development policy** for the NFL engine. The
college-football model remains independently maintained under `cfb/`.

## Exactly two active forecast outputs

1. **Baseline:** existing locked v0.8.1 scoring core from `model.py`.
   Preserve it as the fixed experimental control; never silently recalculate
   or replace a previously frozen result.
2. **One challenger:** currently v0.10.0, defined by
   `nfl_stateful_candidate.py` and the constant in
   `nfl_challenger_policy.py`. This single challenger accumulates improvements
   from every NFL postgame audit and prior research version.

There is **no separate v0.9 runner**. Its useful state/pass/turnover/field-position
mechanisms survive as extracted features and scoring components used by the
v0.10 challenger. Git commit history preserves the old implementation.

For any **future** revision (e.g. v0.11):

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

**Current operational structure:** `v0.8.1 baseline + v0.10.0 one challenger`.
