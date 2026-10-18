"""One historical NFL game; verified-feature-only diagnostic; NOT full prediction model.

Never supplement missing pregame variables with the actual final score or fabricate values.
The only modeling assumptions are the unchanged provisional NFL league priors,
EPA mixing weights and drives/play proxies from model.py v0.1.0.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import MODEL_RULES

base = Path(__file__).resolve().parent
s = json.loads((base / 'pregame_verified_partial.json').read_text())
r = MODEL_RULES['NFL']
h, a = s['Jacksonville Jaguars'], s['Houston Texans']

def calculate(off, de):
    pass_match = .5*(off['off_pass_epa'] + de['def_pass_epa_allowed'])
    rush_match = .5*(off['off_rush_epa'] + de['def_rush_epa_allowed'])
    mix = off['pass_rate']*pass_match + (1-off['pass_rate'])*rush_match
    # Substitute only fixed v0.1 provisional league reference quantities,
    # NOT missing team facts. This is a simplified EPA projection diagnostic.
    ppd = r['league_ppd'] + r['plays_per_drive']*mix
    points = r['drives_per_team'] * ppd
    naive_sack_proxy = .5*(off['off_sack_rate']+de['def_sack_rate'])
    baseline_scoring = .5*(off['points_for_per_game'] + de['points_allowed_per_game'])
    return {
      'matched_pass_epa_raw':round(pass_match,4),
      'matched_rush_epa_raw':round(rush_match,4),
      'matched_blended_epa_raw':round(mix,4),
      'fixed_reference_plays_per_drive':r['plays_per_drive'],
      'fixed_reference_drives_per_team':r['drives_per_team'],
      'provisional_points_per_drive':round(ppd,4),
      'provisional_score_points':round(points,2),
      'unadjusted_points_for_vs_allowed_baseline':round(baseline_scoring,2),
      'two_rate_sack_proxy_NOT_pressure_model':round(naive_sack_proxy,4),
      'pass_rate_5pp_up_changes_points':round(.05*(pass_match-rush_match)*r['plays_per_drive']*r['drives_per_team'],2),
      'matched_explosive_pass_15plus_simple_raw':round(.5*(off['off_explosive_pass_rate']+de['def_explosive_pass_allowed']),4),
      'matched_explosive_pass_vs_league_ratio':round(.5*(off['off_explosive_pass_rate']+de['def_explosive_pass_allowed'])/.152,3)
    }

out = {
 'status':'PARTIAL_DIAGNOSTIC_NOT_FULL_ATS_SIMULATION',
 'version':'0.1.0',
 'source':s['source'],
 'pre_kickoff_stats_only':True,
 'neutral_site':s['neutral_site'],
 'model_parameters':'unchanged MODEL_RULES NFL league priors; no postgame tuning',
 'home_JAX':calculate(h,a),
 'away_HOU':calculate(a,h),
 'missing_fields':s['mandatory_fields_not_yet_verified'],
 'fitted_probabilities':False,
 'bet_grade':'UNRATED'
}
out['home_margin'] = round(out['home_JAX']['provisional_score_points']-out['away_HOU']['provisional_score_points'],2)
out['implied_total'] = round(out['home_JAX']['provisional_score_points']+out['away_HOU']['provisional_score_points'],2)
out['market_home_margin'] = -s['market']['spread_home']
out['market_total'] = s['market']['total']
out['scoring_baseline_home_margin'] = round(out['home_JAX']['unadjusted_points_for_vs_allowed_baseline'] - out['away_HOU']['unadjusted_points_for_vs_allowed_baseline'],2)
out['actual_result_sealed_separately'] = 'Houston 34 Jacksonville 31 (not used in calculations)'
(base/'partial_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
