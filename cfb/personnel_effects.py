"""CFB-only empirical quarterback replacement *conditional* scoring research.

The source is completed, eligible pre-cutoff play-by-play, matched by numeric
team/athlete/game IDs.  QB participation is NOT proof of a future starter.

No injury points are invented: the conditional difference is estimated from
actual QB pass-play EPA and individual QB rushing EPA, then conservatively
shrunk and bounded. These conversion/shrink choices are unvalidated research
hypotheses and MUST NOT replace the frozen primary forecast. Play EPA is not
an isolated causal estimate of QB talent (opponents, scheme, teammates differ).
A manually supplied official-report snapshot is NOT independent proof of a
starter/absence or of its reported capture time.
"""
from __future__ import annotations
from copy import deepcopy
import math
import pandas as pd
from cfb.engine import simulate
from cfb.personnel import _key

VERSION = 'cfb-personnel-replacement-research-v1'
MIN_LEADER_PASS_PLAYS = 35
MIN_ALTERNATE_PASS_PLAYS = 20
PRIOR_EQUIVALENT_PASS_PLAYS = 90.0
MAX_RAW_PASS_EPA_DIFFERENCE = .35
MAX_ABS_POINTS_PER_DRIVE_SHIFT = .40


def _numeric_epa(frame):
    return pd.to_numeric(frame['EPA'], errors='coerce').dropna().astype(float)


def _player_name(roster, qid, team):
    if not {'team_id','athlete_id'}.issubset(roster.columns): return None
    matched=roster.loc[(roster.team_id.map(_key)==str(team)) &
                       (roster.athlete_id.map(_key)==qid)]
    if matched.empty:return None
    for col in ('display_name','athlete_display_name','full_name'):
        if col in matched:
            values=matched[col].dropna().astype(str)
            if not values.empty and values.iloc[0].strip():return values.iloc[0].strip()
    return None


def _team_plays(mapped, eligible, team_id):
    required={'game_id','pos_team_id','pass','EPA','passer_player_id','drive_id'}
    missing=required-set(mapped.columns)
    if missing: return None, {'status':'MISSING_REQUIRED_QB_PLAY_FIELDS',
                              'missing_fields':sorted(missing)}
    if not {'game_id','home_id','away_id','_kickoff'}.issubset(eligible.columns):
        return None, {'status':'MISSING_PRIOR_GAME_IDENTITY'}
    team=str(team_id)
    recent=eligible.loc[(eligible.home_id.map(_key)==team)|
                        (eligible.away_id.map(_key)==team)].sort_values(['_kickoff','game_id'])
    ids=set(recent.game_id.astype(str))
    offense=mapped.loc[(mapped.pos_team_id.map(_key)==team) &
                       mapped.game_id.astype(str).isin(ids)].copy()
    return offense, {'status':'PRIOR_PBP_EVIDENCE','prior_game_ids':sorted(ids),
                     'prior_games':len(ids)}


def quarterback_effects(mapped, eligible, roster, team_id, qb_usage, statuses):
    """Build one possible replacement scenario, or an explicit evidence-based PASS.

    Historical pass-play EPA is compared only for QBs on the same team.  We
    expose confounding, missing participation and each observed sample count.
    Shrink only measured DIFFERENCES; never infer that OUT == zero production.
    """
    abstain={'version':VERSION,'status':'UNAVAILABLE_NOT_ENOUGH_REPLACEMENT_EVIDENCE',
             'team_id':str(team_id),'scenario':None,'baseline_auto_adjusted':False,
             'not_calibrated':True}
    offense,meta=_team_plays(mapped,eligible,team_id)
    if offense is None:return {**abstain,**meta}
    if qb_usage.get('status')!='QB_PARTICIPATION_OBSERVED_NOT_STARTER_CONFIRMED':
        return {**abstain,'reason':'No identified QB participation'}
    players=qb_usage.get('quarterbacks') or []
    if len(players)<2:return {**abstain,'reason':'Fewer than two recorded roster quarterbacks'}
    leader_id=qb_usage.get('leader_by_recent_pass_play_volume')
    leader=next((q for q in players if q['athlete_id']==leader_id),None)
    if leader is None:return {**abstain,'reason':'Recent passer leader unidentified'}
    status=next((x for x in statuses.get('players',[]) if
                 x.get('team_id')==str(team_id) and x.get('athlete_id')==leader_id),None)
    reported=(status or {}).get('status')
    passer=offense.loc[offense['pass'].eq(True)].copy()
    passer['qb_id']=passer.passer_player_id.map(_key)
    team_pass=len(passer)
    mapped_pass=int(passer.qb_id.isin([q['athlete_id'] for q in players]).sum())
    if team_pass==0 or mapped_pass/team_pass<.75:
        return {**abstain,'reason':'Incomplete QB identity on team pass plays',
                'qb_identity_coverage':round(mapped_pass/team_pass,4) if team_pass else None}
    valid_drives=offense.loc[offense.drive_id.notna()].groupby('game_id').drive_id.nunique().sum()
    if valid_drives<3:return {**abstain,'reason':'Too few identified team possessions'}
    prior_pass_per_drive=team_pass/valid_drives
    if not .05<=prior_pass_per_drive<=15:
        return {**abstain,'reason':'Implausible team passing volume / possession'}
    lead_rows=passer.loc[passer.qb_id==leader_id]
    lead_epa=_numeric_epa(lead_rows)
    if len(lead_epa)<MIN_LEADER_PASS_PLAYS:
        return {**abstain,'reason':'Recent passer leader below minimum measured pass-play sample',
                'leader_pass_plays_with_epa':len(lead_epa)}
    # Pick only the most-observed alternate, not an asserted next-game starter.
    candidate=None
    for alt in sorted((p for p in players if p['athlete_id']!=leader_id),
                      key=lambda p:(-p['total_recorded_pass_plays'],p['athlete_id'])):
        ar=passer.loc[passer.qb_id==alt['athlete_id']]
        ap=_numeric_epa(ar)
        if len(ap)>=MIN_ALTERNATE_PASS_PLAYS:
            candidate=(alt,ar,ap);break
    if candidate is None:
        return {**abstain,'reason':'No alternate with minimum observed passing sample',
                'minimum_alternate_pass_plays':MIN_ALTERNATE_PASS_PLAYS}
    alt,alt_rows,alt_epa=candidate
    # Distinguish small overlap of game environments from truly comparable reps.
    leader_games=set(lead_rows.game_id.astype(str))
    alternate_games=set(alt_rows.game_id.astype(str))
    overlap=sorted(leader_games & alternate_games)
    lead_mean=float(lead_epa.mean());alt_mean=float(alt_epa.mean())
    diff=alt_mean-lead_mean
    bound=max(-MAX_RAW_PASS_EPA_DIFFERENCE,min(MAX_RAW_PASS_EPA_DIFFERENCE,diff))
    effective=min(len(lead_epa),len(alt_epa))
    shrink=effective/(effective+PRIOR_EQUIVALENT_PASS_PLAYS)
    # A change in observed QB pass-play EPA changes expected EPA, not a verified
    # causal talent difference. Scale using measured team pass-plays per drive.
    raw_ppd=bound*shrink*prior_pass_per_drive
    pass_ppd=max(-MAX_ABS_POINTS_PER_DRIVE_SHIFT,min(MAX_ABS_POINTS_PER_DRIVE_SHIFT,raw_ppd))
    # Rush accounting: an actual QB rush record is descriptive; do NOT impute
    # different designed-carry volumes for an unconfirmed replacement.
    rush_detail={'status':'NOT_PROJECTED_INSUFFICIENT_QB_RUSH_SAMPLE',
                 'leader_recorded_qb_rushes':leader.get('recorded_qb_rushes'),
                 'alternate_recorded_qb_rushes':alt.get('recorded_qb_rushes'),
                 'leader_mean_rush_epa':leader.get('qb_rush_epa_avg'),
                 'alternate_mean_rush_epa':alt.get('qb_rush_epa_avg')}
    rush_ppd=0.0
    ln=leader.get('recorded_qb_rushes') or 0
    an=alt.get('recorded_qb_rushes') or 0
    le=leader.get('qb_rush_epa_avg')
    ae=alt.get('qb_rush_epa_avg')
    if (ln>=12 and an>=12 and le is not None and ae is not None
            and math.isfinite(float(le)) and math.isfinite(float(ae))):
        # Same opportunity count is a conditional assumption, not a verified
        # future play-calling distribution. Do not boost the primary forecast.
        rush_epa_delta=max(-.4,min(.4,float(ae)-float(le)))
        rush_shrink=min(ln,an)/(min(ln,an)+50.0)
        observed_leader_rush_per_team_drive=ln/valid_drives
        rush_ppd=max(-.12,min(.12,rush_epa_delta*rush_shrink*observed_leader_rush_per_team_drive))
        rush_detail={'status':'MEASURED_RUSH_EPA_CONDITIONAL_FIXED_VOLUME',
                     'leader_recorded_qb_rushes':ln,'alternate_recorded_qb_rushes':an,
                     'leader_mean_rush_epa':le,'alternate_mean_rush_epa':ae,
                     'observed_leader_qb_rushes_per_team_drive':round(observed_leader_rush_per_team_drive,5),
                     'rush_ppd_difference_fixed_volume':round(rush_ppd,6),
                     'hypothesis_not_validated':True}
    combined=max(-MAX_ABS_POINTS_PER_DRIVE_SHIFT,min(MAX_ABS_POINTS_PER_DRIVE_SHIFT,pass_ppd+rush_ppd))
    availability_basis=('REPORTED_OUT_UNATTESTED_SOURCE_CAPTURE' if reported=='OUT'
                        else 'UNCONFIRMED_STARTER_OR_ABSENCE')
    return {'version':VERSION,'status':'EMPIRICAL_CONDITIONAL_SCENARIO_UNCALIBRATED',
            'team_id':str(team_id),'prior_game_ids':meta['prior_game_ids'],
            'baseline_auto_adjusted':False,'not_calibrated':True,
            'scenario':{'reference_qb_id':leader_id,'reference_qb':leader['name'],
                 'candidate_qb_id':alt['athlete_id'],'candidate_qb':alt['name'],
                 'conditional':'IF_CANDIDATE_TAKES_REPLACEMENT_QB_ROLE',
                 'reported_reference_status':reported,
                 'availability_basis':availability_basis,
                 'report_source_verified':bool(statuses.get('source_verified')),
                 'reference_pass_play_epa':round(lead_mean,5),
                 'candidate_pass_play_epa':round(alt_mean,5),
                 'observed_epa_difference_per_pass_play':round(diff,5),
                 'reference_pass_play_sample':len(lead_epa),
                 'candidate_pass_play_sample':len(alt_epa),
                 'shared_prior_game_ids':overlap,
                 'team_pass_plays_per_recorded_drive':round(prior_pass_per_drive,5),
                 'shrunk_bounded_pass_epa_difference':round(bound*shrink,6),
                 'shrink_factor':round(shrink,6),
                 'estimated_change_in_points_per_drive':round(combined,6),
                 'passing_component_change_in_ppd':round(pass_ppd,6),
                 'rushing_component_change_in_ppd':round(rush_ppd,6),
                 'qb_rushing_evidence':rush_detail,
                 'confounding':'Not a causal QB-value estimate; teammates/opponents, game scripts and play calling differ',
                 'not_game_day_starter_confirmation':True,
                 'not_primary_model':True}}


def skill_player_exposures(mapped,eligible,roster,team_id,statuses,recent_games=3):
    """Measurable RB carries / pass targets, never unsupported point deductions."""
    offense,meta=_team_plays(mapped,eligible,team_id)
    if offense is None:return {'status':meta['status'],'players':[]}
    if 'rush' not in offense.columns or 'rusher_player_id' not in offense.columns or 'receiver_player_id' not in offense.columns:
        return {'status':'SKILL_PLAYER_IDENTIFIERS_MISSING','players':[]}
    scheduled=eligible.loc[(eligible.home_id.map(_key)==str(team_id))|
                           (eligible.away_id.map(_key)==str(team_id))].sort_values(['_kickoff','game_id'])
    last=set(scheduled.game_id.astype(str).iloc[-recent_games:])
    lastplays=offense.loc[offense.game_id.astype(str).isin(last)]
    rush=lastplays.loc[lastplays.rush.eq(True)]
    target=lastplays.loc[lastplays['pass'].eq(True)]
    rush_ids=rush.rusher_player_id.map(_key).value_counts()
    # receiver ID on *pass attempts* may be absent for throwaways/sacks;
    # receiver share is among only identified targets, not all attempts.
    target_ids=target.receiver_player_id.map(_key).value_counts()
    identities=sorted(set(rush_ids.index)|set(target_ids.index))
    status_map={(p['team_id'],p.get('athlete_id')):p for p in statuses.get('players',[])}
    results=[]
    for aid in identities:
        count_r=int(rush_ids.get(aid,0));count_t=int(target_ids.get(aid,0))
        if count_r+count_t<4:continue
        q=status_map.get((str(team_id),aid))
        results.append({'athlete_id':aid,'name':_player_name(roster,aid,team_id),
                        'recent_rushes':count_r,'recent_identified_targets':count_t,
                        'share_of_identified_recent_rushes':round(count_r/len(rush),4) if len(rush) else None,
                        'share_of_identified_recent_targets':round(count_t/sum(target_ids),4) if sum(target_ids) else None,
                        'reported_availability':q['status'] if q else None,
                        'point_adjustment':None,'not_verified_starter':True})
    results.sort(key=lambda x:-(x['recent_rushes']+x['recent_identified_targets']))
    return {'status':'DESCRIPTIVE_ONLY_NO_SKILL_PLAYER_SCORE_ADJUSTMENT',
            'eligible_prior_games':len(scheduled),'recent_prior_game_ids':sorted(last),
            'players':results[:12],
            'missing_report_is_not_healthy':True}


def simulate_personnel_scenarios(home_projection,away_projection,home_evidence,away_evidence, *, runs=3000,seed=1):
    """Freeze conditional scores; never modify primary or claim realized lineup."""
    original_home=deepcopy(home_projection);original_away=deepcopy(away_projection)
    base=simulate(original_home,original_away,runs=runs,seed=seed)
    cases=[]
    for side,evidence in (('home',home_evidence),('away',away_evidence)):
        if evidence.get('status')!='EMPIRICAL_CONDITIONAL_SCENARIO_UNCALIBRATED':continue
        scenario=evidence['scenario']
        hp=deepcopy(original_home);ap=deepcopy(original_away)
        team_proj=hp if side=='home' else ap
        value=float(team_proj['_ppd'])+scenario['estimated_change_in_points_per_drive']
        if not .01<=value<=6.5:continue
        team_proj['_ppd']=value
        alt=simulate(hp,ap,runs=runs,seed=seed)
        cases.append({'team_side':side,'condition':scenario['conditional'],
                      'reference_qb':scenario['reference_qb'],
                      'candidate_qb':scenario['candidate_qb'],
                      'candidate_qb_id':scenario['candidate_qb_id'],
                      'frozen_reference_report_status':scenario['reported_reference_status'],
                      'availability_basis':scenario['availability_basis'],
                      'candidate_score_home':alt['mean_score_home'],
                      'candidate_score_away':alt['mean_score_away'],
                      'candidate_home_margin':alt['mean_margin_home'],
                      'home_margin_change_vs_same_seed_baseline':round(alt['mean_margin_home']-base['mean_margin_home'],5),
                      'conditional_only':True,
                      'not_pre_game_starting_qb_prediction':True,
                      'no_verified_coverage_or_accuracy_claim':True})
    return {'version':VERSION,'status':'CONDITIONAL_SENSITIVITY_ONLY_NOT_A_PRIMARY_FORECAST',
            'baseline_same_seed_margin':base['mean_margin_home'],
            'scenarios':cases,'requires_verified_lineup_for_predictive_grading':True,
            'not_auto_deployed':True}


def other_position_availability(roster, statuses, team_id):
    """Track reported OL/DL/DB/LB/TE/K availability without invented point values.

    The released PBP does not supply reliable player-level blocking, pressure,
    pass coverage or snap participation, so do not assign a numeric effect.
    """
    required={'team_id','athlete_id','position_abbreviation'}
    if not required.issubset(roster.columns):
        return {'status':'MISSING_ROSTER_POSITION_IDENTITIES','players':[]}
    positions={_key(x.athlete_id):str(x.position_abbreviation).upper() for _,x in
               roster.loc[roster.team_id.map(_key)==str(team_id)].iterrows()}
    relevant={'OL','OT','OG','C','T','G','DE','DT','DL','EDGE','LB','MLB',
              'OLB','ILB','CB','S','SS','FS','DB','TE','K','PK','P'}
    players=[]
    for p in statuses.get('players',[]):
        if p.get('team_id')!=str(team_id):continue
        pos=positions.get(p.get('athlete_id'))
        if pos not in relevant:continue
        players.append({'athlete_id':p.get('athlete_id'),'name':p.get('name'),
                        'position':pos,'reported_status':p.get('status'),
                        'source_verified':bool(statuses.get('source_verified')),
                        'actual_inactive_verified':False,'point_adjustment':None})
    return {'status':'REPORT_STATUS_ONLY_NO_PLAYER_LEVEL_EFFECT_ESTIMATE',
            'players':players,'no_report_does_not_mean_healthy':True,
            'missing_position_snap_data':['offensive line blocking','individual pass rush',
                                          'coverage matchup','verified lineup']}
