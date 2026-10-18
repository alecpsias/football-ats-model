"""Deterministic human-readable view of the complete prediction audit."""

def render(result: dict) -> str:
    h=result['projection_home']; a=result['projection_away']
    s=result['simulation']; m=result['market']; aud=result['audit']
    lines=[
        f"# {a['team']} at {h['team']} — ATS model v{result['version']}",
        '',
        f"**Game:** {result['game_id']} | **Snapshot:** {result['snapshot_utc']}",
        f"**Input SHA-256:** `{result['input_sha256']}`",
        f"**Status:** {result['model_status']} | **Grade:** {result['grade']} | **Decision:** {result['wager_decision']}",
        '',
        '## Model estimates (NOT a betting recommendation)',
        '',
        f"**Expected score:** {h['team']} {s['mean_score_home']:.1f}, {a['team']} {s['mean_score_away']:.1f}",
        f"**Estimated margin:** {s['mean_margin_home']:+.2f} home; median {s['median_margin_home']:+.1f}",
        f"**Market:** home spread {m['spread_home']:+g}, total {m['total']}",
        f"**Raw ATS:** home {s['home_cover_probability_raw']:.1%}, away {s['away_cover_probability_raw']:.1%}, push {s['push_probability']:.1%}",
        f"**Raw total over:** {s['over_probability_raw']:.1%}",
        f"**Simulation:** {s['simulations']:,} trials; reproducible seed {s['seed']}",
        '',
        '## Matchup calculations',
        '',
        '| Adjusted component | Home offense | Away offense |',
        '|---|---:|---:|',
    ]
    comps = [
        ('Pass EPA/play','matched_pass_epa'), ('Rush EPA/play','matched_rush_epa'),
        ('Pressure rate','pressure_rate'),('Expected sacks','expected_sacks_per_game'),
        ('QB carries','qb_rushes_per_game'),('Pass success rate','pass_success_rate'),
        ('Rush success rate','rush_success_rate'),('Explosive pass rate','explosive_pass_rate'),
        ('Explosive rush rate','explosive_rush_rate'),('Third-down conversions','third_down_conversion'),
        ('Red-zone TD rate','redzone_td_rate'),('Turnover probability per drive','turnover_drive_probability'),
        ('EPA added points per drive','epa_points_per_drive_adjustment'),
        ('Red-zone residual points per drive','redzone_residual_ppd'),
        ('Third-down residual points per drive','third_down_residual_ppd'),
        ('Field-position points per drive','field_position_ppd'),
        ('Penalties net points/game','net_penalties_ppg_component'),
        ('Special teams net points/game','net_special_teams_ppg_component'),
        ('Home-field component points/game','home_adv_ppg_component'),
        ('Estimated points per drive','raw_points_per_drive'),
    ]
    for label,key in comps:
        lines.append(f'| {label} | {h[key]} | {a[key]} |')
    lines += ['', '## Stress tests (hypothetical cases, NOT calibrated likelihoods)', '',
              '| Scenario | Home cover | Away cover | Mean margin |', '|---|---:|---:|---:|']
    st = result.get('stress_tests')
    if st:
        for name, row in st['scenario_results'].items():
            lines.append(f"| {name.replace('_',' ')} | {row['home_cover_raw']:.1%} | {row['away_cover_raw']:.1%} | {row['mean_margin_home']:+.1f} |")
        lines.extend(['','### Spread sensitivity', '', '| Home spread | Home cover | Away cover | Push |','|---|---:|---:|---:|'])
        for spread,row in st['spread_home_sensitivity'].items():
            lines.append(f"| {spread} | {row['home_cover_raw']:.1%} | {row['away_cover_raw']:.1%} | {row['push']:.1%} |")
    else:
        lines.append('*Stress testing skipped (historical replay mode).*')
    if result.get('first_half'):
        fh=result['first_half']
        lines += ['', '## First half (unrated)', '',
                 f"1H spread home {fh['market']['spread_home']:+g}; raw home cover {fh['simulation']['home_cover_probability_raw']:.1%}.",
                 'Do not promote the 1H side over the full-game side without separate historical calibration.']
    if result.get('calibration'):
        c=result['calibration']
        lines += ['', '## Chronological holdout calibration', '',
                  f"Home {c['calibrated_home_cover_probability']:.1%}; away {c['calibrated_away_cover_probability']:.1%}. ",
                  f"Holdout games: {c['historical_holdout_games']}; Brier: {c['calibration_holdout_brier']}. ",
                  'Calibration alone is not evidence of positive ROI.']
    lines += ['', '## Audit / hard gates', '', '| Gate | Status |', '|---|---|']
    for k,v in aud.items():
        lines.append(f'| {k.replace("_"," ")} | {"Complete" if v else "Not complete / unavailable"} |')
    lines += ['', '## Source and assumption notes', '',
              f"Home personnel: {h['personnel_notes']} (source: {h['personnel_adjustment_source']})", 
              f"Away personnel: {a['personnel_notes']} (source: {a['personnel_adjustment_source']})",
              '',
              '**Limitations:** Model priors, regression weights, and possession mapping remain provisional. '
              'Scores and probabilities are illustrative research outputs until prospective or chronological out-of-sample '
              'validation confirms calibration and predictive performance. The model does not automatically fetch stats or injury reports.',
             ]
    return '\n'.join(lines)
