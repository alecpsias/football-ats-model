# Free nflverse connector — v0.3.0

**NEW:** `nflverse_connector.py` loads nflverse play-by-play, schedules and weekly rosters using the officially documented `nflreadpy` Python functions, saves immutable `.csv.gz` snapshots with manifests and SHA-256 hashes, and tests point-in-time eligibility. `run_snapshot_ingest.py` bridges verified snapshots to the unchanged strict forecast-ingestion engine. No paid sources are used. **The live GitHub dataset was not downloadable from this environment; the integration is software-tested with local stub data, not with actual 2026 nflverse files.**

See [NFLVERSE_CONNECTION_GUIDE.md](NFLVERSE_CONNECTION_GUIDE.md).

# Football ATS Prediction Engine — v0.1.0

**Status: reproducible research prototype, not a trained or validated betting model.**

This is an actual executable Python model designed to enforce the *same* evidence gathering, calculations, and decision gates for every NFL or college-football game. It is not merely a prompt or a checklist. It has a strict pregame input format; repeatable possession-level Monte Carlo simulation; a transparent calculation audit; scenario and point-spread sensitivity; and a chronological historical replay/calibration tool.

**It does not automatically retrieve football statistics, authenticate cited sources, verify injuries, or estimate its own coefficients from data.** The current league reference rates and adjustment coefficients are disclosed **provisional illustrative assumptions**, not empirical results. Until real out-of-sample validation is performed, the engine deliberately labels its probabilities **RAW UNCALIBRATED**, the pick **UNRATED**, and the betting decision **PASS**. Unit tests prove code consistency—not forecasting accuracy.

## Files

| File | Function |
|---|---|
| `model.py` | Strict input validation, opponent adjustment, matchup math, simulations, uncertainty, scenario & market sensitivity, optional chronological calibration |
| `backtest.py` | Replays historical **pregame** snapshots against final ATS results; fits a regularized probability mapping on the first 70% and checks it on the later 30% |
| `report.py` | Produces a human-readable Markdown summary, matchup adjustments and completion audit |
| `examples/real_matchup_TEMPLATE.json` | Mandatory data-entry structure; `null` values must be replaced with verified statistics |
| `examples/synthetic_matchup.json` | **Entirely artificial figures, including prices and team names**, intended solely for software testing |
| `examples/synthetic_report.md` | Example readable model output; **not a real football prediction** |
| `examples/synthetic_report.json` | Detailed machine-readable trace for the fake matchup |
| `tests/` | Automated reproducibility, completeness, leakage and sensitivity tests |
| `LESSONS_TO_RULES.md` | Trace from this week's postgame lessons to executable modules and limitations |
| `requirements.txt` | Python dependency |

## Installation and first run

Use Python 3.10+ and NumPy. From inside this folder:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python model.py examples/synthetic_matchup.json --runs 50000 --seed 20261018 --output sample_result.json --markdown sample_result.md
```

On Windows, `py` can be used in place of `python`.

The example game is **synthetic**. For a real game, fill in a copy of `examples/real_matchup_TEMPLATE.json` using trustworthy pre-kickoff stats and sportsbook quotes. To prevent guessing, mandatory unknowns are represented by `null`. The model errors rather than filling gaps with invented values.

```bash
python model.py my_verified_game.json --runs 50000 --seed 1234 --output results.json --markdown results.md
```

## Input requirements and conventions

All per-play efficiency statistics must use a consistent methodology across teams (for example, one data provider's EPA definitions). **All rate inputs are proportions from 0 to 1, not percentages.** Offense/defense statistics must only include games completed before the forecast timestamp.

- **`league`**: `NFL` or `CFB`. Separate sets of provisional league assumptions are provided. They are **not** fitted to either league's empirical data.
- **`kickoff_utc`, `data_as_of_utc`**: timezone-aware timestamps. The snapshot must be before kickoff. No post-kickoff statistics or quotes accepted.
- **`home`, `away`**: team identity, source, statistics timestamp, opponent-quality context, field position, penalties, special teams, and recent personnel check.
- **`offense.epa_pass` / `.epa_rush`**: offense EPA per **dropback / rush**, consistent with `pass_rate` denominator. Define consistently whether sacks and scrambles belong to passes or rushes.
- **`defense.epa_pass_allowed` / `.epa_rush_allowed`**: allowed EPA on corresponding opposing plays. Negative allowed EPA means strong defense.
- **`schedule.opp_def_pass_epa_allowed_avg` / `.opp_def_rush...`**: average allowed EPA of the opponents that this offense faced, ideally weighted by offensive plays against each.
- **`schedule.opp_off_pass_epa_avg` / `.opp_off_rush_epa_avg`**: average passing/rushing EPA of opponents faced by the defense, preferably weighted by plays.
- **`success_*` / `explosive_*`**: use common data-provider definitions consistently. The engine uses small *residual* adjustments, not full double counts of effects already in EPA.
- **`pressure_allowed` / `pressure_generated`, `sack_allowed` / `sack_generated`**: rate of pressured/sacked **dropbacks**. For pressure, use a single consistent denominator. Sacks are a subset of pressures.
- **`interception_rate` / `interception_rate_forced`**: interceptions per **dropback**, not per attempt, to match the engine.
- **`fumble_lost_rate` / `fumble_recovery_rate`**: relevant lost/recovered fumbles per **offensive play**; exclude interception turnovers.
- **`redzone_td_rate` / `redzone_td_allowed`**: touchdowns / red-zone trips; also supply the sample count `redzone_trips` or `redzone_trips_faced` for shrinkage.
- **`qb_rushes_per_game`, `qb_rush_epa`**: QB rushing designed/scramble usage; use a consistent treatment of kneels and sacks across all teams.
- **`context.penalty_epa_per_game`, `context.special_teams_epa_per_game`**: *net signed* values from the chosen provider. Avoid mixing penalties yards and penalty EPA units. Adjustments are deliberately attenuated because some effects are already in EPA.
- **`context.avg_start_yardline`**: average own-yardline starting field position, 29 = own 29; not distance to opponent end zone.
- **`personnel`**: news check time must be within 24 hours of snapshot. Any non-neutral changes to expected passing/rushing EPA, protection, QB rushing or pace require an explanatory note and a source. Neutral values are not evidence of confirmed full health; news should still be checked and cited.
- **`market.spread_home`**: **negative if the HOME team is favored**. Example: home -6.5 is `-6.5`; home +3 is `3`. American odds required for each side, plus source/timestamp.
- **Optional `market.first_half`**: same fields as full market. First-half model runs separately but cannot claim a better edge until first-half probabilities are independently calibrated.
- **`neutral_site`**: if `true`, model removes the placeholder home-field advantage.

Do not populate the blank template with real scores or a closing line observed **after** the forecast time. That would produce look-ahead bias.

## How the model calculates (fixed order)

1. **Check validity and timestamps**. Rejects missing fields, bad rate units, impossible rates, stale (over 24-hour) injury/news checks, and data newer than the prediction snapshot.
2. **Opponent-adjusted efficiency**. Passing and rushing offense EPA are adjusted by the quality of opposing defenses; defensive EPA is adjusted by offenses faced. Each is shrunk toward a league baseline using sample play counts.
3. **Matchup composition**. The adjusted offense and opposing defense are combined separately for pass and rush. Models pressure, sack expectation, explosive plays, play success, third downs, QB rushing volume, and projected interception/fumble rates.
4. **Regression and personnel**. Red-zone finishing rates shrink toward a common baseline by number of opportunities. Personnel changes are applied only via documented numeric assumptions.
5. **Expected points per possession**. Play-level EPA differences are converted to expected drive points; residual situational, net special-teams/penalty effects, field position, and a fixed placeholder home edge are included without assuming all historic effects are independently additive.
6. **Stochastic game simulation**. Generates TD/FG/turnover/empty possessions, shared game variance and short fields after turnovers. Draws the number of drives as a game-level random variable. Approximate overtime is used for full game; no overtime for first half.
7. **Spread/price assessment**. Reports score distributions, home/away cover and push probabilities, total over/under, vig-adjusted break-even, and raw EV as an explicitly **unverified model output**.
8. **Stress and sensitivity**. Recomputes scoring under turnover regression, red-zone regression, adverse offensive scripts, passing-efficiency shocks and multiple alternate spread prices. These are *conditional scenarios*, not learned scenario probabilities.
9. **Mandatory limits**. No A/B/C bet grade or positive wager recommendation is emitted without further empirical verification.

## Historical replay and calibration

Create a JSON Lines file with one game per line. **Each object must contain the FULL pregame snapshot from when the prediction would have been made.** Do not reconstruct pregame features from season-end statistics.

```json
{"game":{"...":"A complete game object from the template, populated with historical pregame data"}, "actual_home_score":24, "actual_away_score":20}
```

The `"..."` is documentation shorthand, *not* a valid runnable input file. The sample history above has illustrative results and must not be used to claim a model hit rate.

Then run:

```bash
python backtest.py my_historical_pregame_games.jsonl --runs 5000 --output my_backtest.json
```

The program sorts games chronologically, puts the earlier ~70% into training, and holds the later ~30% out. It computes raw and calibrated holdout Brier score, log loss and directional accuracy. A validation flag is only available with **at least 100 training and 40 holdout non-push games**, and an improved Brier score. Even then, that only establishes a limited calibration diagnostic, **not profitable wagering or guaranteed future performance**.

If that historical report passes, the forecast command can additionally use:

```bash
python model.py my_verified_game.json --calibration my_backtest.json --output results.json --markdown results.md
```

The model rejects cross-league, different-version, failed, or future-leaking calibration reports. Even after a holdout passes, grades remain conservative/unrated until independently validated betting-return criteria exist.

## Limitations and next milestones

**What's real now:** The code, formulas, required-field validation, opponent adjustments, stochastic drive engine, error checks, scenario sensitivity, archived input hashes, chronological replay, optional probability mapping, and unit tests.

**What is not yet established:** Real-data accuracy, empirical coefficients for pressure/turnover/red-zone interplay, drive-structure calibration, trustworthy cover probability, calibrated first-half comparisons, and any A/B/C confidence reliability. A placeholder home edge and generic NFL/CFB priors are NOT authoritative league figures. The simulation is not a play-by-play Markov drive model, and the overtime approximation is deliberately simplified. No automatic collection of live stats, betting markets, Circa splits, or game-day personnel is included.

**Next required validation work:** Construct an archive of verified *timestamped pregame* games (NFL and CFB separately), run chronological and then forward-season holdouts, verify error calibration and realized ATS performance at actually available sportsbook prices, and fit/replace the provisional coefficients. All changes should be versioned and tested rather than silently altering the rules.

See `LESSONS_TO_RULES.md` for how the recent postgame findings map to executable code and what remains unresolved.

## NEW: NFL ingestor (v0.2 scaffold; prediction engine still v0.1.0)

The `ingest.py` command constructs timestamped per-game model inputs from nflverse-compatible play-by-play, a verified schedule, an optional pressure charting dataset, an optional QB position map, and a game configuration for injuries/market/context. It writes a machine-readable audit and refuses to create a ready-to-predict file when required statistics or source timestamps are absent.

Read `INGESTION_GUIDE.md` for source URLs, data definitions, setup, blocking conditions, and example commands. `fetch_nflverse.py` optionally downloads public PBP/schedules/weekly rosters and records SHA-256 hashes plus capture times. **Neither downloader nor ingestor can establish that a file collected after a historical kickoff was available before that kickoff.** It therefore intentionally blocks the October 18 retrospective Texans–Jaguars replay until suitable archived sources, 2026 pressure, contextual EPA, injuries and synchronized market prices have been obtained.

New dependencies for ingestion: `pip install -r requirements-ingestion.txt`. New tests: `tests/test_ingest.py`. The full suite (26 tests) is synthetic software verification, *not* a historical ATS model-performance test.
