"""Mandatory CFB Failure-1 pregame rushing mismatch / ATS risk audit.

This is an evidence and decision gate, NEVER an extra scoring coefficient.
Called in both on-demand and batch pregame workflows from the same
24-hour-embargoed history used by Build16. When severe, require SAME-SEED
5,000-draw Build16 counterfactual stress cases. Findings cannot be silently
omitted before presenting an ATS lean; unresolved conflicts block upgrading
confidence, but no current ATS probability is calibrated.
"""
from __future__ import annotations
import math

VERSION='cfb-failure1-required-rushing-audit-v1'
EARLY_OFF_THRESHOLD=.42
EARLY_DEF_THRESHOLD=.38
JOINT_EXCESS_TRIGGER=.04  # V2 exploratory source-feature heuristic, NOT calibrated
SOS_RUSH_EPA_TRIGGER=.085  # predeclared stress screen, NOT calibrated
MIN_EARLY_RUSHES=35
MIN_PRIOR_GAMES=3
REQUIRED_STRESS_RUNS=5000

def _rate(rows,prefix,numerator,denominator,minimum=MIN_EARLY_RUSHES,minimum_games=MIN_PRIOR_GAMES):
    fields=('mismatch_'+prefix+numerator,'mismatch_'+prefix+denominator)
    selected=[]
    for r in rows:
        if any(f not in r for f in fields):continue
        x,y=r[fields[0]],r[fields[1]]
        if (type(x) not in (int,float) or type(y) not in (int,float)
            or not math.isfinite(x) or not math.isfinite(y)
            or x<0 or y<0 or x>y):
            return None
        if y>0:selected.append((float(x),float(y)))
    if len(selected)<minimum_games:return None
    sx=sum(x for x,_ in selected);sy=sum(y for _,y in selected)
    return {'rate':sx/sy,'trials':int(sy),'observed_games':len(selected)} if sy>=minimum else None

def _side(team_id,opponent_id,histories):
    offense=histories.get(team_id,[])
    defense=histories.get(opponent_id,[])
    out={'status':'INSUFFICIENT_OBSERVED_RUSH_SUCCESS','offense_games':len(offense),
         'defense_games':len(defense),'early_rush_success':None,
         'opponent_early_rush_success_allowed':None,
         'offense_negative_run_rate':None,
         'opponent_negative_run_rate_allowed':None,
         'offense_nonpositive_run_rate':None,
         'opponent_nonpositive_run_rate_allowed':None,
         'own_trailing_9plus_rush_success':None,
         'defense_trailing_9plus_rush_success_allowed':None,
         'joint_severity_excess':None,
         'matched_negative_run_difference':None,'severe_signal':False,
         'missing_or_insufficient':[]}
    cols=[
       ('early_rush_success','off_','early_rush_success_yes','early_rush_n'),
       ('opponent_early_rush_success_allowed','def_','early_rush_success_yes','early_rush_n'),
       ('offense_negative_run_rate','off_','early_negative_yes','early_rush_n'),
       ('opponent_negative_run_rate_allowed','def_','early_negative_yes','early_rush_n'),
       ('offense_nonpositive_run_rate','off_','early_nonpositive_yes','early_rush_n'),
       ('opponent_nonpositive_run_rate_allowed','def_','early_nonpositive_yes','early_rush_n'),
       ('own_trailing_9plus_rush_success','off_','trailing_rush_good','trailing_rush_n'),
       ('defense_trailing_9plus_rush_success_allowed','def_','trailing_rush_good','trailing_rush_n'),
       ('own_within_8_rush_success','off_','within_8_rush_good','within_8_rush_n'),
       ('defense_within_8_rush_success_allowed','def_','within_8_rush_good','within_8_rush_n')]
    for key,prefix,num,den in cols:
        source=defense if prefix=='def_' else offense
        is_script='trailing_' in key or 'within_8' in key
        r=_rate(source,prefix,num,den,12 if is_script else MIN_EARLY_RUSHES,
                2 if is_script else MIN_PRIOR_GAMES)
        if r is None:
            out['missing_or_insufficient'].append(key)
        else:out[key]=r
    own=out['early_rush_success'];allowed=out['opponent_early_rush_success_allowed']
    ownnegative=out['offense_negative_run_rate'];allownegative=out['opponent_negative_run_rate_allowed']
    if own and allowed:
        severity=max(0.,min(own['rate']-EARLY_OFF_THRESHOLD,
                            allowed['rate']-EARLY_DEF_THRESHOLD))
        out['joint_severity_excess']=round(severity,6)
        out['severe_signal']=severity>=JOINT_EXCESS_TRIGGER
        out['matched_early_success_mean']=round((own['rate']+allowed['rate'])/2,6)
        out['status']='EVIDENCE_OBSERVED'
    if ownnegative and allownegative:
        out['matched_negative_run_difference']=round(
            ownnegative['rate']-allownegative['rate'],6)
    out['script_resilience_status']=(
        'TRAILING_HISTORY_OBSERVED' if
        out['own_trailing_9plus_rush_success'] and
        out['defense_trailing_9plus_rush_success_allowed'] and
        out.get('own_within_8_rush_success') else
        'TRAILING_OR_NEUTRAL_HISTORY_INSUFFICIENT_NO_ASSUMPTIONS')
    return out

def audit(home_id,away_id,histories,matchup_signals=None):
    if home_id==away_id:raise ValueError('Two distinct teams required')
    h=_side(home_id,away_id,histories)
    a=_side(away_id,home_id,histories)
    sos=(matchup_signals or {}).get('sos_rush_epa_differential')
    if type(sos) not in (float,int) or not math.isfinite(sos):sos=None
    trigger_reasons=[]
    for label,z in (('home',h),('away',a)):
        if z['severe_signal']:
            trigger_reasons.append(label+'_JOINT_DESIGNED_RUSH_SUCCESS_VS_DEFENSE')
    if sos is not None and abs(sos)>=SOS_RUSH_EPA_TRIGGER:
        trigger_reasons.append('OPPONENT_QUALITY_ADJUSTED_RUSH_EPA_DIFFERENTIAL')
    sufficient=h['status']=='EVIDENCE_OBSERVED' and a['status']=='EVIDENCE_OBSERVED'
    if not sufficient:status='REQUIRES_HUMAN_REVIEW_INCOMPLETE_RUSH_EVIDENCE'
    elif trigger_reasons:status='SEVERE_RUSH_MISMATCH_REQUIRES_5000_RUN_STRESS'
    else:status='SCREENED_NO_SEVERE_TRIGGER'
    return {'version':VERSION,'status':status,
        'mandatory_for_every_cfb_pregame_prediction':True,
        'home':h,'away':a,'sos_rush_epa_differential':sos,
        'thresholds_exploratory_not_fitted':{
          'offense_early_success':EARLY_OFF_THRESHOLD,
          'opponent_defense_success_allowed':EARLY_DEF_THRESHOLD,
          'joint_excess_to_trigger':JOINT_EXCESS_TRIGGER,
          'opponent_adjusted_rush_epa_differential':SOS_RUSH_EPA_TRIGGER},
        'trigger_reasons':trigger_reasons,
        'requires_full_rush_stress':bool(trigger_reasons),
        'required_adverse_script_simulations_each':REQUIRED_STRESS_RUNS,
        'source':'past completed games only, >=24h embargo required at caller',
        'designed_rush_EPA_concept_not_NCAA_sack_rush_count':True,
        'scoring_coefficients_changed':False,
        'independently_validated_ATS_probability':False,
        'no_missing_data_zero_filling':True,
        'objective':'adverse-rush scripts must be checked before any ATS confidence upgrade'}

def gate(check,challenger,market=None):
    """Refuse implicit high-confidence recommendation if audit is unresolved."""
    result={**check,'ats_recommendation_status':None,
        'risk_conclusion':None,'stress_runs_per_scenario_verified':None,
        'stress_home_margin_min':None,'stress_home_margin_max':None,
        'stress_ats_conflict_with_central':None,'stress_scenario_margins':{}}
    if not check['requires_full_rush_stress']:
        result['ats_recommendation_status']=(
            'HOLD_INCOMPLETE_RUSH_EVIDENCE' if
            check['status'].startswith('REQUIRES_HUMAN') else
            'SCREENED_NO_EXTRA_RUSH_STRESS_NO_CALIBRATED_BET')
        result['risk_conclusion']='No numerical rushing scoring adjustment.'
        return result
    if not isinstance(challenger,dict) or challenger.get('simulation') is None:
        result['ats_recommendation_status']='HOLD_BUILD16_UNAVAILABLE'
        result['risk_conclusion']='Cannot validate severe-mismatch stress without Build16 result.'
        return result
    stress=(challenger.get('adverse_sensitivity') or {}).get('counterfactuals') or {}
    if (len(stress)<6 or any(not isinstance(v,dict) or
            type(v.get('mean_home_margin')) not in (float,int) or
            not math.isfinite(v['mean_home_margin']) for v in stress.values()) or
            (challenger.get('adverse_sensitivity') or {}).get('simulations_per_scenario') != REQUIRED_STRESS_RUNS):
        result['ats_recommendation_status']='HOLD_MANDATORY_5000_RUN_STRESS_NOT_VERIFIED'
        result['risk_conclusion']='Scenario set not proven to be 5,000 draws per script.'
        return result
    values={k:round(float(v['mean_home_margin']),5) for k,v in stress.items()}
    result['stress_scenario_margins']=values
    result['stress_runs_per_scenario_verified']=REQUIRED_STRESS_RUNS
    result['stress_home_margin_min']=min(values.values())
    result['stress_home_margin_max']=max(values.values())
    signed=(market or {}).get('home_spread')
    mean=challenger['mean_margin_home']
    if type(signed) in (float,int) and math.isfinite(signed):
        base=mean+signed
        sides={k:(v+signed) for k,v in values.items()}
        conflict=any((base>0 and x<0) or (base<0 and x>0) for x in sides.values())
        result['stress_ats_conflict_with_central']=conflict
        result['market_home_spread_observed']=signed
        result['ats_recommendation_status']=(
            'HOLD_CONFLICTING_ADVERSE_SCRIPTS' if conflict else
            'HOLD_SEVERE_RUSH_MISMATCH_MANUAL_REVIEW')
    else:
        result['ats_recommendation_status']='HOLD_NO_VERIFIED_SPREAD_OR_MANUAL_REVIEW'
    result['risk_conclusion']=(
        'Completed all six 5,000-run Build16 stress distributions; '
        'counterfactual margins are NOT scenario probabilities, '
        'and this audit never boosts a confidence grade automatically.')
    return result
