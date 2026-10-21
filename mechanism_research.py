#!/usr/bin/env python3
"""v0.9 candidate mechanisms, separate from production core.

All empirical material is read from snapshots captured before the forecast time.
A 6-hour completed-game buffer is a conservative convention, not independent
verification of every historical final whistle. No official inactives are used.

IMPORTANT: Descriptive qb_hit is *not* PFR pressure, and play EPA on a penalty
is *not* isolated penalty EPA. No candidate coefficient has been ATS fitted.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import math
import numpy as np
import pandas as pd

from collect_extra_metrics import parse_stamp
from model import InputError, clip, shrink, logit, sigmoid
from nfl_situational_scoring import scored_situations

REQUIRED = {'game_id','play_id','posteam','defteam','fixed_drive','play_type','down',
            'ydstogo','third_down_converted','first_down_penalty','yardline_100',
            'interception','fumble_lost','score_differential','qb_dropback',
            'qb_hit','sack','fixed_drive_result','penalty','penalty_team','penalty_yards',
            'fourth_down_converted','fourth_down_failed'}


def _num(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None


def _bucket(value):
    y=_num(value)
    if y is None or y<=0:return None
    return 'short_1_2' if y<=2 else 'medium_3_6' if y<=6 else 'long_7_plus'


def _zone(yards_to_goal):
    y=_num(yards_to_goal)
    if y is None or not 0<=y<=100:return None
    return 'opponent_40_or_closer' if y<=40 else 'outside_opponent_40'


def _state(score_diff):
    v=_num(score_diff)
    if v is None:return None
    return 'trailing_9_plus' if v<=-9 else 'leading_9_plus' if v>=9 else 'within_8'


def _blank():
    return {'third':{b:[0,0] for b in ('short_1_2','medium_3_6','long_7_plus')},
            'pass_states':{b:[0,0] for b in ('trailing_9_plus','leading_9_plus','within_8')},
            'qb_hit_dropbacks':0,'dropbacks':0,'sacks':0,'interceptions':0,
            'hit_interceptions':0,'clean_interceptions':0,
            'turnovers':0,'turnovers_near_scoring':0,'fourth_attempts':0,'fourth_failed':0,
            'snaps_taken':0,'penalty_nonthird_firsts_earned':0,
            'penalty_nonthird_firsts_allowed':0,
            'penalty_auto_first_down_conceded':0,'penalties_conceded':0,
            'penalty_yards_conceded':0,'snaps_faced':0,
            'drive_states':{s:{'drives':0,'points_proxy':0,'touchdowns':0,'field_goals':0}
                            for s in ('trailing_9_plus','within_8','leading_9_plus')},
            'completed_team_game_ppd':[],
            'scoring_opportunities':0,'scoring_opportunity_touchdowns':0,
            'scoring_opportunity_field_goals':0,'scoring_opportunity_turnovers':0,
            'scoring_opportunity_fourth_down_failures':0}


def extract_mechanisms(pbp:pd.DataFrame,schedule:pd.DataFrame,cutoff,teams:tuple[str,str], buffer_hours:int=6)->dict:
    """Capture-safe observed NFL mechanisms. No fitted causal coefficients.

    Returns team and league observed counts, and real next-possession field
    positions *after* takeaways. The target game cannot be eligible.
    """
    if not isinstance(pbp,pd.DataFrame) or not isinstance(schedule,pd.DataFrame):
        raise InputError('Mechanism research requires loaded PBP and schedule tables')
    miss=REQUIRED-set(pbp.columns)
    if miss:raise InputError('Missing mechanism PBP columns: '+str(sorted(miss)))
    for c in ('game_id','home_team','away_team'):
        if c not in schedule.columns:raise InputError('Schedule missing '+c)
    t=parse_stamp(str(cutoff))
    if len(set(teams))!=2:raise InputError('Teams must be distinct')
    if not 4<=buffer_hours<=24:raise InputError('Buffer outside 4-24 hours')
    if schedule.game_id.duplicated().any():raise InputError('Duplicate schedule game IDs')
    from ingest import normalize_schedule
    norm=normalize_schedule(schedule)
    cutoff_eligible=t-timedelta(hours=buffer_hours)
    allowed=set(norm.loc[pd.to_datetime(norm.kickoff_utc,utc=True)<=pd.Timestamp(cutoff_eligible),'game_id'].astype(str))
    valid=pbp.loc[pbp.game_id.astype(str).isin(allowed)].sort_values(['game_id','play_id']).copy()
    # Malformed team's possessions are excluded, rather than assigned to a wrong opponent.
    if not valid.empty:
        gm=schedule.set_index('game_id')[['home_team','away_team']]
        valid=valid.drop(columns=[c for c in ('home_team','away_team') if c in valid.columns]).join(gm,on='game_id',how='inner')
        valid=valid.loc[((valid.posteam==valid.home_team)&(valid.defteam==valid.away_team))|
                        ((valid.posteam==valid.away_team)&(valid.defteam==valid.home_team))]
    if len(valid)<350 or valid.game_id.nunique()<8:
        raise InputError('Insufficient completed pre-cutoff PBP for mechanism estimation')
    off=defaultdict(_blank);deff=defaultdict(_blank)
    lg=_blank()
    # Save observed following-possession yardline by location of prior turnover.
    next_starts={b:[] for b in ('opponent_40_or_closer','outside_opponent_40')}
    drives=defaultdict(list)
    # Pre-cutoff, completed-drive PPD observations for a separate v0.10
    # research challenger. Never import the target game's play-by-play.
    game_points=defaultdict(lambda:[0.,0])
    for row in valid.itertuples(index=False):
        gid=str(row.game_id);a=str(row.posteam);d=str(row.defteam)
        x=off[a];y=deff[d]
        # Count a drive only by its actual first offensive snap, never kickoff.
        if row.play_type!='kickoff' and _num(row.fixed_drive) is not None:
            drives[gid].append(row)
        if row.play_type in ('run','pass'):
            x['snaps_taken']+=1;lg['snaps_taken']+=1
            state=_state(row.score_differential)
            if state is not None:
                is_pass=int(_num(row.qb_dropback)==1 or row.play_type=='pass')
                for obj in (x,lg):
                    obj['pass_states'][state][0]+=is_pass
                    obj['pass_states'][state][1]+=1
            y['snaps_faced']+=1
        if _num(row.qb_dropback)==1:
            is_hit=int(_num(row.qb_hit)==1)
            picked=int(_num(row.interception)==1)
            # A QB hit is an OBSERVED hit, not every kind of pressure.
            for obj in (x,y,lg):
                obj['dropbacks']+=1
                obj['qb_hit_dropbacks']+=is_hit
                obj['sacks']+=int(_num(row.sack)==1)
                obj['interceptions']+=picked
                obj['hit_interceptions']+=picked*is_hit
                obj['clean_interceptions']+=picked*(1-is_hit)
        if _num(row.down)==3:
            b=_bucket(row.ydstogo)
            if b is not None:
                converted=int(_num(row.third_down_converted)==1 or _num(row.first_down_penalty)==1)
                for obj in (x,y,lg):
                    obj['third'][b][0]+=converted
                    obj['third'][b][1]+=1
        if _num(row.down)==4 and row.play_type in ('run','pass','no_play'):
            failed=int(_num(row.fourth_down_failed)==1)
            if failed or _num(row.fourth_down_converted)==1:
                for obj in (x,y,lg):
                    obj['fourth_attempts']+=1;obj['fourth_failed']+=failed
        if _num(row.interception)==1 or _num(row.fumble_lost)==1:
            z=_zone(row.yardline_100)
            for obj in (x,lg):
                obj['turnovers']+=1
                obj['turnovers_near_scoring']+=int(z=='opponent_40_or_closer')
        if (_num(row.first_down_penalty)==1 and _num(row.down)!=3
                and row.play_type in ('run','pass','no_play')):
            x['penalty_nonthird_firsts_earned']+=1
            lg['penalty_nonthird_firsts_earned']+=1
            if str(row.penalty_team)==d:
                y['penalty_nonthird_firsts_allowed']+=1
        if _num(row.penalty)==1 and str(row.penalty_team)==d:
            amount=_num(row.penalty_yards)
            for obj in (y,lg):
                obj['penalties_conceded']+=1
                obj['penalty_auto_first_down_conceded']+=int(_num(row.first_down_penalty)==1)
                if amount is not None and amount>=0:obj['penalty_yards_conceded']+=amount
    # Actual next offense location is NOT interchangeable with location of pick.
    for gid,rows in drives.items():
        prior=None; ordered=[]; seen=set()
        for row in rows:
            key=(str(row.posteam),row.fixed_drive)
            if key not in seen:
                ordered.append((key,row));seen.add(key)
        bydrive={k:[] for k,_ in ordered}
        for row in rows:bydrive[(str(row.posteam),row.fixed_drive)].append(row)
        # Summarize completed possession results for drives that reached
        # scoring position. These are observation counts, NOT isolated points.
        for (k,initial) in ordered:
            drive_rows=bydrive[k]
            outcome=str(next((q.fixed_drive_result for q in reversed(drive_rows)
                              if hasattr(q,'fixed_drive_result') and pd.notna(q.fixed_drive_result)),'')).lower()
            state=_state(initial.score_differential)
            # Only recognize explicit FINAL drive outcomes. A missing result
            # is not silently relabeled a scoreless drive.
            observed={'touchdown','field goal','punt','turnover',
                      'turnover on downs','missed field goal',
                      'end of half','end of game'}
            if state and outcome in observed:
                points=7 if outcome=='touchdown' else 3 if outcome=='field goal' else 0
                for obj in (off[k[0]],deff[str(initial.defteam)],lg):
                    entry=obj['drive_states'][state]
                    entry['drives']+=1
                    entry['points_proxy']+=points
                    entry['touchdowns']+=int(outcome=='touchdown')
                    entry['field_goals']+=int(outcome=='field goal')
                game_points[(gid,k[0])][0]+=points
                game_points[(gid,k[0])][1]+=1
            if not any((_num(p.yardline_100) is not None and
                        _num(p.yardline_100)<=40) for p in drive_rows):continue
            outcome=str(next((q.fixed_drive_result for q in reversed(drive_rows)
                              if hasattr(q,'fixed_drive_result') and pd.notna(q.fixed_drive_result)),'')).lower()
            for obj in (off[k[0]],deff[str(initial.defteam)],lg):
                obj['scoring_opportunities']+=1
                obj['scoring_opportunity_touchdowns']+=int(outcome=='touchdown')
                obj['scoring_opportunity_field_goals']+=int(outcome=='field goal')
                obj['scoring_opportunity_turnovers']+=int(outcome=='turnover')
                obj['scoring_opportunity_fourth_down_failures']+=int(outcome=='turnover on downs')
        for j,(key,first) in enumerate(ordered[:-1]):
            cur=bydrive[key]
            last=cur[-1]
            # Defensive return TD and halftime can break this simple next-drive join:
            # require the next offensive drive to belong to the opposing team.
            if str(ordered[j+1][0][0])==key[0]:continue
            turnover=next((r for r in reversed(cur) if _num(r.interception)==1 or _num(r.fumble_lost)==1),None)
            if turnover is None:continue
            z=_zone(turnover.yardline_100)
            start=_num(ordered[j+1][1].yardline_100)
            if z and start is not None and 1<=start<=99:
                next_starts[z].append(round(100-start,2))
    for (gid,team),(points,attempts) in sorted(game_points.items()):
        if attempts>=5:
            lg['completed_team_game_ppd'].append(round(points/attempts,6))
    if not next_starts['opponent_40_or_closer'] and not next_starts['outside_opponent_40']:
        raise InputError('No observed next-drive field positions after turnovers')
    # Each team should have at least three legitimate prior games, not 3 tagged plays.
    gcounts={t:valid.loc[valid.posteam==t,'game_id'].nunique() for t in teams}
    if any(n<3 for n in gcounts.values()):
        raise InputError('Insufficient previous games for at least one target team')
    return {'schema':'ats_mechanism_observations_v0.9','as_of_utc':t.isoformat().replace('+00:00','Z'),
        'completion_buffer_hours':buffer_hours,'eligible_game_count':int(valid.game_id.nunique()),
        'eligible_play_count':len(valid),'target_teams':list(teams),'target_team_games':gcounts,
        'league':lg,'offense':{x:off[x] for x in teams},
        'defense':{x:deff[x] for x in teams},
        'next_start_own_yardline_after_takeaway_by_prior_offense_zone':next_starts,
        'limits':['QB hits are NOT PFR pressure or all disruptions',
                  'Automatic first downs from penalties describe events, not isolated penalty EPA',
                  'Next-possession field position is observational, not causal penalty/turnover value',
                  'Six-hour kickoff buffer is conservative; only files captured by as-of are valid']}


def matchup_third_down(profile:dict,off_team:str,def_team:str):
    """Shrink offense and defense buckets to league, without fitting on target game."""
    result={};off=profile['offense'][off_team]['third'];df=profile['defense'][def_team]['third'];lg=profile['league']['third']
    for b in lg:
        succ,n=lg[b];prior=(succ+8*.395)/(n+8)
        os,on=off[b];ds,dn=df[b]
        pa=(os+25*prior)/(on+25);pd_=(ds+25*prior)/(dn+25)
        expected=clip(sigmoid(logit(prior)+.5*(logit(pa)-logit(prior))+.5*(logit(pd_)-logit(prior))),.03,.92)
        result[b]={'rate':round(expected,5),'league_rate':round(prior,5),
            'offense_trials':on,'defense_trials':dn,'offense_conversions':os,'defense_allowed_conversions':ds}
    weights=np.array([max(0,off[b][1]) for b in lg],dtype=float)
    if weights.sum()==0:weights=np.array([v[1] for v in lg.values()],dtype=float)
    weights=weights/weights.sum()
    result['weighted_conversion_rate']=round(sum(weights[i]*result[b]['rate'] for i,b in enumerate(lg)),5)
    result['distance_mix']={b:round(float(weights[i]),4) for i,b in enumerate(lg)}
    return result


def _passrate_for_state(profile,team,state,neutral_projection):
    lg=profile['league']['pass_states'];tm=profile['offense'][team]['pass_states']
    # Only an additive STATE delta: team pass baseline remains the existing model.
    def posterior(st):
        good,n=tm[st];base,bn=lg[st]
        prior=(base+15*.57)/(bn+15)
        return (good+40*prior)/(n+40)
    return clip(neutral_projection+posterior(state)-posterior('within_8'),.18,.90)


def turnover_following_field_bonus(profile,turnover_yards_to_goal,r,rng):
    """Draw *actual* next possession field position conditional on TO zone.

    A pick at the opponent 17 often gives the defense a LONG field, unlike a
    generic '+0.9 short-field points' rule.
    """
    z=_zone(turnover_yards_to_goal)
    values=profile['next_start_own_yardline_after_takeaway_by_prior_offense_zone'].get(z,[])
    if not values:
        both=[v for x in profile['next_start_own_yardline_after_takeaway_by_prior_offense_zone'].values() for v in x]
        values=both
    start=float(values[int(rng.integers(0,len(values)))])
    bonus=clip((start-r['ref_start_yardline'])*r['field_position_points_per_yard'],-1.25,2.2)
    return bonus,start


def _turnover_zone(profile,team,rng):
    a=profile['offense'][team];lg=profile['league']
    frac=(a['turnovers_near_scoring']+12*(lg['turnovers_near_scoring']+1)/(lg['turnovers']+2))/(a['turnovers']+12)
    return 25 if rng.random()<frac else 72


def candidate_simulation(home:dict,away:dict,profile:dict,r:dict,runs:int,seed:int)->dict:
    """Explicit state-dependent alternative, NOT production or calibrated ATS.

    Reuses the established projections; experimental drive resolution adds:
    1) 3rd-down distance mix corrections (bounded residual only),
    2) trailing/leading passing volume and exposure to interception,
    3) measured next-drive field position following a turnover.
    High-leverage penalty events and hit rates are OUTPUT AS DIAGNOSTICS ONLY,
    because their isolated causal point coefficients have not been estimated.
    """
    if not 1000<=runs<=100000:raise ValueError('Candidate runs must be 1000-100000')
    for team in (home['team'],away['team']):
        if team not in profile['offense']:raise InputError('Missing team '+team)
    rng=np.random.default_rng(seed)
    third_h=matchup_third_down(profile,home['team'],away['team'])
    third_a=matchup_third_down(profile,away['team'],home['team'])
    # Existing scalar matchup already included in _ppd: only the difference is added.
    r_h=clip(1.25*(third_h['weighted_conversion_rate']-home['third_down_conversion']),-.12,.12)
    r_a=clip(1.25*(third_a['weighted_conversion_rate']-away['third_down_conversion']),-.12,.12)
    starts=profile['next_start_own_yardline_after_takeaway_by_prior_offense_zone']
    observed=[v for a in starts.values() for v in a]
    empirical_avg_bonus=float(np.mean([clip((v-r['ref_start_yardline'])*r['field_position_points_per_yard'],-1.25,2.2) for v in observed]))
    situ_h=scored_situations(profile,home['team'],away['team'])
    situ_a=scored_situations(profile,away['team'],home['team'])
    def adjusted(p,s):
        # A shift toward TDs rather than FGs changes expected points too.
        # Keep the original scoring-event frequency fixed; do not let the
        # score normalization cancel the conditional finishing adjustment.
        scoring_event_rate=clip(p['_ppd']/(3.0+3.95*p['_td']),.01,.8)
        finishing_ppd=3.95*scoring_event_rate*s['total_delta_td_share']
        s['applied_finishing_delta_ppd']=round(finishing_ppd,6)
        return {**p,'_ppd':clip(p['_ppd']+s['total_delta_ppd']+finishing_ppd,.16,5.4),
                '_td':clip(p['_td']+s['total_delta_td_share'],.10,.95)}
    home_sim=adjusted(home,situ_h)
    away_sim=adjusted(away,situ_a)
    score_h=np.zeros(runs,dtype=float);score_a=np.zeros(runs,dtype=float)
    to_h=np.zeros(runs,dtype=int);to_a=np.zeros(runs,dtype=int)
    state_pass_h=defaultdict(list);state_pass_a=defaultdict(list)
    drive_count=np.clip(np.rint(rng.normal(.5*(home['expected_drives']+away['expected_drives']),1.,runs)),7,17).astype(int)
    opening=rng.integers(0,2,size=runs)
    next_bonus_h=np.zeros(runs);next_bonus_a=np.zeros(runs)
    # Correlation-only, shrinkable QB-hit vs interception interaction.
    # This is deliberately a small residual to the existing PFR-adjusted INT
    # matchup, NOT another additive pressure or sack main effect.
    lg=profile['league']
    overall=(lg['interceptions']+2)/(lg['dropbacks']+80)
    hit_count=lg['qb_hit_dropbacks'];clean_count=lg['dropbacks']-hit_count
    hit_int=(lg['hit_interceptions']+60*overall)/(hit_count+60)
    clean_int=(lg['clean_interceptions']+60*overall)/(clean_count+60)
    association=clip(hit_int-clean_int,-.04,.04)
    hit_prior=(hit_count+12*.08)/(lg['dropbacks']+12)
    def hit_residual(p,defteam):
        off=profile['offense'][p['team']];defen=profile['defense'][defteam]
        off_hit=(off['qb_hit_dropbacks']+80*hit_prior)/(off['dropbacks']+80)
        def_hit=(defen['qb_hit_dropbacks']+80*hit_prior)/(defen['dropbacks']+80)
        return clip(((off_hit+def_hit)/2-hit_prior)*association,-.004,.004)
    hit_h=hit_residual(home,away['team']);hit_a=hit_residual(away,home['team'])
    # Precompute mixing per score STATE. All probabilities independent of target result.
    pass_h={s:_passrate_for_state(profile,home['team'],s,home['pass_rate']) for s in ('trailing_9_plus','leading_9_plus','within_8')}
    pass_a={s:_passrate_for_state(profile,away['team'],s,away['pass_rate']) for s in ('trailing_9_plus','leading_9_plus','within_8')}
    def one(p,team_side,run_index,bh,ba):
        if team_side=='home':d=float(score_h[run_index]-score_a[run_index]);advance=next_bonus_h;to_counts=to_h;track=state_pass_h
        else:d=float(score_a[run_index]-score_h[run_index]);advance=next_bonus_a;to_counts=to_a;track=state_pass_a
        state=_state(d);passp=(pass_h if team_side=='home' else pass_a)[state]
        track[state].append(passp)
        # Turnover risk changes only through share of passes with existing INT/fumble matchup rates.
        adjusted_int=clip(p['interception_rate_per_dropback']+(hit_h if team_side=='home' else hit_a),.001,.14)
        turnover_p=clip(1-math.exp(-p['plays_per_drive']*(passp*adjusted_int+p['fumble_lost_rate_per_play'])),.005,.42)
        # Mean-center bonus using league-average observed next-possession field position.
        mean_ppd=clip(p['_ppd'] + (r_h if team_side=='home' else r_a)
                     -r['league_to_drive']*empirical_avg_bonus+advance[run_index],.16,5.4)
        advance[run_index]=0
        is_to=rng.random()<turnover_p
        if is_to:
            to_counts[run_index]+=1
            _,_=None,None
            to_goal=_turnover_zone(profile,p['team'],rng)
            bonus,start=turnover_following_field_bonus(profile,to_goal,r,rng)
            if team_side=='home':next_bonus_a[run_index]=bonus
            else:next_bonus_h[run_index]=bonus
            return 0
        avg_score=p['_td']*6.95+(1-p['_td'])*3.
        score_p=clip(mean_ppd/avg_score*(1-turnover_p)/(1-r['league_to_drive']),.01,.96)
        if rng.random()>=score_p:return 0
        if rng.random()<p['_td']:return 7 if rng.random()<.96 else 6
        return 3
    for i in range(17):
        for idx in np.flatnonzero(drive_count>i):
            if opening[idx]==0:
                score_h[idx]+=one(home_sim,'home',idx,next_bonus_h,next_bonus_a)
                score_a[idx]+=one(away_sim,'away',idx,next_bonus_h,next_bonus_a)
            else:
                score_a[idx]+=one(away_sim,'away',idx,next_bonus_h,next_bonus_a)
                score_h[idx]+=one(home_sim,'home',idx,next_bonus_h,next_bonus_a)
    margins=score_h-score_a
    result={'schema':'ats_candidate_mechanism_simulation_v0.9','research_only':True,'promoted_to_primary':False,
            'score_home':round(float(score_h.mean()),2),'score_away':round(float(score_a.mean()),2),
            'mean_margin_home':round(float(margins.mean()),2),
            'margin_p10_p50_p90':[float(x) for x in np.quantile(margins,[.1,.5,.9])],
            'projected_third_down_by_distance':{'home':third_h,'away':third_a},
            'third_down_ppd_residual_not_fitted':{'home':round(r_h,4),'away':round(r_a,4)},
            'scored_situational_outcomes':{'home':situ_h,'away':situ_a},
            'passing_share_given_score_state':{'home':pass_h,'away':pass_a},
            'empirical_turnover_following_start':{k:{'sample_count':len(v),'mean_own_yardline':round(float(np.mean(v)),2) if v else None} for k,v in starts.items()},
            'sampled_turnovers_per_game':{'home':round(float(to_h.mean()),3),'away':round(float(to_a.mean()),3)},
            'observed_hit_interception_association_not_causal':{'hit_int_rate_league':round(hit_int,5),'nonhit_int_rate_league':round(clean_int,5),'matched_interception_residual_home':round(hit_h,6),'matched_interception_residual_away':round(hit_a,6)},
            'measured_qb_hit_not_pressure':{'home':profile['offense'][home['team']]['qb_hit_dropbacks'],
                                       'away':profile['offense'][away['team']]['qb_hit_dropbacks']},
            'high_leverage_penalties_descriptive_only':{'home_defense':profile['defense'][home['team']]['penalty_auto_first_down_conceded'],
                                                     'away_defense':profile['defense'][away['team']]['penalty_auto_first_down_conceded']},
            'n':runs,'seed':seed,
            'limitations':['Exploratory uncalibrated model; must not supplant locked primary prediction until out-of-sample improvement',
              'QB hit is not full pressure. No unmeasured penalty or scoring-opportunity EPA is invented.',
              'Turnover field location uses empirically sampled next possession, not an assumed short field',
              'New fourth-down, finishing and nonthird penalty effects are bounded unfitted hypotheses, not measured causal EPA']}
    return result
