# Synthetic Away at Synthetic Home — ATS model v0.1.0

**Game:** SYNTHETIC-DEMO-DO-NOT-BET | **Snapshot:** 2026-10-18T19:00:00Z
**Input SHA-256:** `167752c77b67065ef96c598290c0cdc8707b7cb9d2c7ac6c658443a5fd684ccb`
**Status:** SYNTHETIC_EXAMPLE_UNCALIBRATED | **Grade:** UNRATED | **Decision:** PASS_UNTIL_BACKTESTED

## Model estimates (NOT a betting recommendation)

**Expected score:** Synthetic Home 24.5, Synthetic Away 22.6
**Estimated margin:** +1.92 home; median +3.0
**Market:** home spread -3.5, total 45.5
**Raw ATS:** home 46.4%, away 53.6%, push 0.0%
**Raw total over:** 53.0%
**Simulation:** 50,000 trials; reproducible seed 20261018

## Matchup calculations

| Adjusted component | Home offense | Away offense |
|---|---:|---:|
| Pass EPA/play | 0.0142 | 0.0052 |
| Rush EPA/play | -0.0035 | -0.0072 |
| Pressure rate | 0.3239 | 0.3239 |
| Expected sacks | 2.526 | 2.526 |
| QB carries | 3.314 | 3.314 |
| Pass success rate | 0.4435 | 0.4423 |
| Rush success rate | 0.4373 | 0.4365 |
| Explosive pass rate | 0.1029 | 0.1029 |
| Explosive rush rate | 0.1024 | 0.1024 |
| Third-down conversions | 0.395 | 0.395 |
| Red-zone TD rate | 0.5703 | 0.5703 |
| Turnover probability per drive | 0.1164 | 0.1164 |
| EPA added points per drive | 0.0384 | -0.0009 |
| Red-zone residual points per drive | -0.0018 | -0.0018 |
| Third-down residual points per drive | -0.0 | -0.0 |
| Field-position points per drive | 0.044 | 0.044 |
| Penalties net points/game | 0.0 | 0.0 |
| Special teams net points/game | 0.0 | 0.0 |
| Home-field component points/game | 0.75 | -0.75 |
| Estimated points per drive | 2.2291 | 2.0528 |

## Stress tests (hypothetical cases, NOT calibrated likelihoods)

| Scenario | Home cover | Away cover | Mean margin |
|---|---:|---:|---:|
| baseline resampled | 47.2% | 52.8% | +2.1 |
| turnover rates regress to league | 47.1% | 52.9% | +2.1 |
| red zone reversion | 47.2% | 52.8% | +2.0 |
| home adverse offensive script | 30.6% | 69.4% | -4.0 |
| away adverse offensive script | 63.7% | 36.3% | +8.1 |
| pass matchup swings toward away | 32.3% | 67.7% | -3.4 |
| pass matchup swings toward home | 62.1% | 37.9% | +7.6 |

### Spread sensitivity

| Home spread | Home cover | Away cover | Push |
|---|---:|---:|---:|
| -5 | 41.8% | 56.6% | 1.6% |
| -4 | 43.4% | 52.8% | 3.8% |
| -3.5 | 47.2% | 52.8% | 0.0% |
| -3 | 47.2% | 48.3% | 4.5% |
| -2 | 51.7% | 46.6% | 1.6% |

## Audit / hard gates

| Gate | Status |
|---|---|
| schema complete | Complete |
| strict source dates checked | Complete |
| opponent strength adjusted | Complete |
| small sample shrinkage | Complete |
| pass and rush marginal models | Complete |
| qb rush volume adjusted | Complete |
| pass pressure sack interaction | Complete |
| success and explosives checked | Complete |
| turnover interaction | Complete |
| redzone regression | Complete |
| third down situations | Complete |
| special teams and penalties | Complete |
| field position and pace | Complete |
| personnel adjustments traced | Complete |
| drive level score distribution | Complete |
| spread and vig checked | Complete |
| first half market evaluated | Not complete / unavailable |
| stress tests | Complete |
| historical out of sample validation | Not complete / unavailable |
| probabilities empirically calibrated | Not complete / unavailable |

## Source and assumption notes

Home personnel: No personnel override applied (source: SYNTHETIC: news checked for demo only)
Away personnel: No personnel override applied (source: SYNTHETIC: news checked for demo only)

**Limitations:** Model priors, regression weights, and possession mapping remain provisional. Scores and probabilities are illustrative research outputs until prospective or chronological out-of-sample validation confirms calibration and predictive performance. The model does not automatically fetch stats or injury reports.
