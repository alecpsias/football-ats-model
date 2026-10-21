# CFB open-data engine — Build 2

This folder extends the existing NFL/CFB prediction engine; it is **not** a validated CFB ATS model or an automated provider of market bets.

## Source compatibility finding

The freely published SportsDataverse ESPN CFB PBP *producer code* defines released columns including:

- Team labels: `home`, `away`, `pos_team`, `def_pos_team` (not PBP numeric `pos_team_id`)
- Drives: `drive_id` (not raw `drive.id`)
- Offense: `yards_gained`, `yards_to_goal`, `EPA`, `pass`, `rush`, `sack`, `int`, `down`, `distance`, `game_play_number`
- First downs and scores: `firstD_by_yards`, `firstD_by_penalty`, `offense_score_play`, `td_play`, `pass_td`, `rush_td`
- The released `turnover` indicator includes interceptions; it cannot stand in for *lost fumbles*.

The schema identification comes from the upstream producer at [sportsdataverse/cfbfastR-cfb-data — python/cfb_data_build/pbp.py](https://github.com/sportsdataverse/cfbfastR-cfb-data/blob/main/python/cfb_data_build/pbp.py) and [schedules_unified.py](https://github.com/sportsdataverse/cfbfastR-cfb-data/blob/main/python/cfb_data_build/schedules_unified.py). **Actual downloaded 2026 Parquet files must still pass `inspect` and the report step.**

## Field position and special teams — CFB-only Build 17 candidate (uncalibrated)

- `cfb/field_special.py` reads ONLY the original verified historical PBP, **before** the scrimmage-only normalization drops kick/punt events. It strictly attributes events by provider team IDs and exposes made/attempted field goals, distance ranges, net punt yards, and observed kickoff return yards (not touchback-imputed zeros).
- `cfb/research_backtest.observed_game` stores observed offensive **and defensive** drive-start yards, with sample counts, plus kick/punt observations from the same eligible game. The 24-hour historical embargo and fixed snapshot cutoff still apply across fresh, prospective and replay paths.
- The independent `cfb/integrated_scoring.py` challenger adjusts offense expected points-per-drive ONLY for the opponent-matched change from its own historic starting position, with capped reliability shrinkage. FG finishing is regressed from observed success toward an explicitly provisional baseline (bounded to 0.055 PPD), rather than treating observed makes as a second scoring bonus.
- Punt net yards and kickoff returns are reported as descriptive source evidence; their scoring path is through observed starting field position. **No double addition of a punting/return EPA component.** Charted special-teams EPA, exact return field-position attribution, player-level kicker adjustment, penalty EPA and validated ATS probabilities are not available. Missing events fail closed.
- `cfb/test_cfb_field_special.py` covers source-team attribution, correct FG attempt denominators, prior-game field matchups, regression bounds, missingness and non-duplication. These are unit tests, not a prospective accuracy validation.

## Guardrails and limitations

- Unified schedule release includes `game_id`, `start_date`, FBS classifications, schedule team IDs, and status. PBP is joined using same-source ESPN `home/away/pos_team` labels **by game_id** to avoid fuzzy guesses about team naming. Any mismatch fails and must be corrected using verified metadata.
- Completed pre-capture games only, all sources hashed, future rows excluded. A dated capture is not evidence that every row existed at any earlier timestamp.
- Offense/defense success means 50%/70%/100% yards-to-gain on 1st/2nd/3rd+ downs; explosives mean 20+ pass or 10+ rush. Those are specific definitions, not blindly copied into a vendor's `success`.
- Scrimmage-only drives are approximations and can omit penalty-only possessions, kicks and punts; true drive-state reconciliation remains outstanding.
- Historical opponent passing/rushing EPA is play weighted across opponents' **other games**; when an opponent has no other observed games, the adjustment remains unknown rather than made up.
- Mandatory charted pressures, complete QB rushing assignment, precise injury effects, independently calculated special-teams EPA and signed sportsbook quotes remain missing. No ATS direction, confidence or betting recommendation is produced by this evidence-only module.
- The `strict_forecast.py` file only hands fully populated, pregame-verified CFB inputs to the existing model; its current CFB coefficients are provisional and have never been established by out-of-sample calibration.

## Running

See the root `INSTALL.md`. `python -m cfb.pipeline --help` lists `collect`, `verify`, `inspect`, `report`.
