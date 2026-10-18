# Historical replay 01 — Houston Texans at Jacksonville Jaguars, 2026-10-18

**Research classification:** Verified-feature partial replay. **Not a full v0.1.0 simulation, not a calibrated forecast, not an A/B/C bet.** The outcome was already known when this case was selected; this is an *in-sample development diagnostic*, not genuine blinded prospective validation.

## Data firewall

- **Kickoff:** 2026-10-18 13:30 UTC (9:30 a.m. ET), at Wembley Stadium, **neutral site**.
- **Archived week-1-to-5 team snapshot:** https://gridironandwine.com/predictions/nfl/2026-10-18-houston-texans-vs-jacksonville-jaguars.html — explicitly labeled season-to-date and captured before kickoff. Its *exact* statistical capture timestamp is not supplied.
- **Market snapshot:** https://edraft.com/nfl/games/2026-week-6-hou-at-jax?scoring=ppr — Jacksonville -2.5, 42.5 total listed October 18 at 7:18 a.m. ET, before kickoff. Exact juice not used in the limited calculation. A separately dated BetMGM Oct 14 quote lists JAX -2.5 (-112), HOU +2.5 (-105); these prices must not be misrepresented as same-time quotes.
- **Pregame news:** https://www.houstontexans.com/news/texans-inactives-week-6-at-jacksonville-jaguars ; https://www.reuters.com/sports/reports-jaguars-lt-anton-harrison-knee-undergo-surgery--flm-2026-10-13/ ; https://www.houstonchronicle.com/sports/texans/article/marvin-harrison-houston-texans-22478180.php . News is documented but **no arbitrary EPA/points uplift** assigned.
- **Actual outcome (held separate):** Houston 34, Jacksonville 31; https://www.jaguars.com/news/game-report-2026-week-6-texans-34-jaguars-31 . Do not feed this value into the pregame feature file.

## Verified raw inputs

| Feature | Houston | Jacksonville |
|---|---:|---:|
| Offensive EPA/dropback | +0.110 | +0.269 |
| Offensive EPA/rush | -0.193 | +0.008 |
| Defensive EPA/dropback allowed | +0.011 | -0.024 |
| Defensive EPA/rush allowed | -0.141 | -0.217 |
| Dropback share | 65.3% | 54.9% |
| Offensive 15+ yard explosive passes | 19.6% | 20.3% |
| Defensive 15+ yard explosive passes allowed | 20.3% | 19.3% |
| Off red-zone TD rate | 52.6% | 64.3% |
| Def red-zone TD rate allowed | 64.3% | 33.3% |

League 15+ passing explosion baseline is 15.2% in source.

## What was actually executed

Ran `python replays/hou_jax_2026_10_18/replay_partial.py` using only archived features and **unchanged** v0.1.0 NFL league assumptions: 2.08 points per drive, 5.8 plays per drive, 10.8 drives per team. This is a submodel diagnostic, not `model.run()`.

The *direct pregame EPA matchup equation* is:

`matched pass EPA = (offense pass EPA + opponent defensive pass EPA allowed)/2`;
`matched rush EPA = (offense rush EPA + opponent defensive rush EPA allowed)/2`;
`matched total EPA = pass_rate × matched pass + (1 - pass_rate) × matched rush`;
`points = fixed 10.8 drives × [2.08 + 5.8 × matched total EPA]`.

- Houston pass +0.0430, rush -0.2050, blend -0.0431 ⇒ **19.77 provisional points**.
- Jacksonville pass +0.1400, rush -0.0665, blend +0.0469 ⇒ **25.40 provisional points**.
- **Estimated JAX margin +5.63, total 45.17**; direct season-scoring diagnostic independently yields JAX +5.5.
- Market JAX -2.5, total 42.5.
- Actual margin JAX -3, total 65. Replay diagnostic **missed margin by 8.63 points**, total by **19.83 points**, Houston by **14.23** and Jacksonville by **5.60**.
- The raw explosive 15+ pass matchup averages **19.45% Houston** and **20.30% Jacksonville**, **1.28× and 1.34×** the archived league reference 15.2%. Both teams had an upward-tail passing warning even though Jacksonville had strong season-long scoring defense.

**Do not interpret +5.63 as a bet edge**: the underlying derivation lacks opponent-quality corrections and full possession distribution. It signals a baseline that would have leaned the wrong way, *not* a successful predicted upset.

## Why full engine did not run

The strict v0.1.0 engine rejects the incomplete input. In `strict_input_INCOMPLETE.json`, **60 mandatory fields or timestamps remain unverified**. Among the important gaps: opponent-strength schedule EPA, precise drive and field-position rates, QB-specific rushing EPA, pressure rates (distinct from QB hits), red-zone **attempt counts**, per-play penalty and special teams EPA, and exact timestamps for archived statistics and lineup verification. The code purposely does not replace those unknowns with fabricated 2026 statistics. This is a valid quality gate, but it means an actual full-model prediction **has not yet been produced**.

## What this case proves and doesn't

**Proves:** We can archive pregame sources, execute fixed arithmetic repeatably, measure score/margin errors honestly, and document a two-sided explosive passing interaction that the mean EPA diagnostic did not capture well.

**Does not prove:** That the full executable model would have selected Houston, that higher explosive rates necessarily imply a specific 65-point total, or that adding one rule will improve out-of-sample accuracy.

**Proposed v0.2 experiment (do not silently change v0.1):** Evaluate a joint explosive-pass tail adjustment to scoring **variance** (not simply adding a deterministic points bonus) and test whether it improves out-of-sample game-total calibration and ATS cover probabilities across many games. Also quantify passing/rushing injury adjustments using actual game-day snap expectations. Freeze formulas before reviewing holdout outcomes.

### Reproducibility

From `football_ats_model` run:

```bash
python replays/hou_jax_2026_10_18/replay_partial.py
python replays/hou_jax_2026_10_18/create_strict_input.py
python -m unittest discover -s tests -v
```

Inputs: `pregame_verified_partial.json` and `strict_input_INCOMPLETE.json`. Output: `partial_diagnostic.json`. The raw v0.1.0 prediction code was **not** modified to fit this known result.
