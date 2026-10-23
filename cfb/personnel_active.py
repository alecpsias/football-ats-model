"""Default Build16 personnel corrections based on frozen prior-game play data.

A qualifying pregame report is necessary but not sufficient: impact requires
source-identified player ID, previously observed role usage and identifiable
same-team alternative performance. No arbitrary generic position point values.
Never treat absent players as zero-production plays; only replace their
historically demonstrated opportunities. Effects are hypotheses, not ATS fits.
"""
from __future__ import annotations

from collections import Counter
import math
import numpy as np
import pandas as pd
from cfb.personnel import _key

VERSION="cfb-build16-personnel-source-regressed-v1"
MIN_PLAYER_EVENTS=6
MIN_REPLACEMENT_EVENTS=12
PRIOR_EQUIVALENT_EVENTS=55.
MAX_SKILL_DELTA_PPD=.30
MAX_QB_DELTA_PPD=.30
MAX_TEAM_DELTA_PPD=.36

def _as_numerics(a):
    return pd.to_numeric(a,errors='coerce').dropna().astype(float)

def _position(p):
    x=str(p.get('position') or '').upper()
    if x in ('RB','FB','HB'):return 'RB'
    if x in ('WR','TE'):return 'RECEIVER'
    if x in ('QB',):return 'QB'
    if x in ('K','PK'):return 'K'
    if x in ('C','OL','OT','OG','G','T','LT','RT','LG','RG'):return 'OL'
    if x in ('P','LS'):return 'SPECIAL'
    if x:return 'DEFENSE_OR_OTHER'
    return 'UNKNOWN'

def _records(mapped,eligible,team_id):
    needed={'game_id','pos_team_id','drive_id','EPA','rush','pass'}
    if mapped is None or not needed.issubset(mapped.columns):
        return None,[]
    if eligible is None or not {'home_id','away_id','game_id','_kickoff'}.issubset(eligible.columns):
        return None,[]
    team=str(team_id)
    games=eligible.loc[(eligible.home_id.map(_key)==team)|
                       (eligible.away_id.map(_key)==team)].sort_values(['_kickoff','game_id'])
    last=list(games.game_id.astype(str).iloc[-3:])
    ids=set(games.game_id.astype(str))
    rows=mapped.loc[(mapped.pos_team_id.map(_key)==team)&
                    (mapped.game_id.astype(str).isin(ids))].copy()
    return rows,last

def _bounded_skill_delta(plays,players,category,last):
    """Sample-shrunk difference vs observed available same-team alternatives.

    All source outs for the same role are excluded from replacement pool.
    Never count target share as share of all passes when receiver IDs missing.
    """
    col='rusher_player_id' if category=='RB' else 'receiver_player_id'
    mode='rush' if category=='RB' else 'pass'
    if col not in plays:return [],0.,['MISSING_PLAYER_IDENTIFIER_'+col]
    frame=plays.loc[plays[mode].eq(True)].copy()
    frame['pid']=frame[col].map(_key)
    frame['EPA_num']=pd.to_numeric(frame['EPA'],errors='coerce')
    frame=frame.loc[frame.pid.notna()&frame.EPA_num.notna()]
    if len(frame)<25:return [],0.,['INSUFFICIENT_TEAM_IDENTIFIED_'+mode+'_PLAYS']
    recent=frame.loc[frame.game_id.astype(str).isin(last)]
    if len(recent)<10:return [],0.,['INSUFFICIENT_RECENT_IDENTIFIED_'+mode+'_PLAYS']
    role_out={x['athlete_id'] for x in players if x.get('athlete_id') and _position(x)==category and x['status']=='OUT'}
    pool=frame.loc[~frame.pid.isin(role_out)]
    if len(pool)<MIN_REPLACEMENT_EVENTS:return [],0.,['NO_OBSERVED_SAME_TEAM_REPLACEMENT_'+category]
    # True prior drive count, never assume 12 possessions by fiat.
    ndrives=plays.loc[plays.drive_id.notna()].groupby('game_id').drive_id.nunique().sum()
    if ndrives<=0:return [],0.,['MISSING_PRIOR_DRIVE_DENOMINATOR']
    mode_plays_per_drive=float(len(frame)/ndrives)
    pool_epa=float(pool.EPA_num.mean())
    total=0.;events=[];gaps=[]
    for p in players:
        if p['status']!='OUT' or _position(p)!=category:continue
        pid=p.get('athlete_id')
        if not pid:
            gaps.append('UNKNOWN_PLAYER_ID_'+p['name']);continue
        own=frame.loc[frame.pid==pid]
        rec=recent.loc[recent.pid==pid]
        if len(own)<MIN_PLAYER_EVENTS or len(rec)<3:
            gaps.append('NO_RECENT_MEASURED_ROLE_'+p['name']);continue
        # Only currently active recent opportunity share is replaced; last
        # year's starter with no recent participation gets no double penalty.
        share=len(rec)/len(recent)
        if share<=0:continue
        player_epa=float(own.EPA_num.mean())
        raw=max(0.,min(.28,player_epa-pool_epa))
        sample=min(len(own),len(pool))
        shrink=sample/(sample+PRIOR_EQUIVALENT_EVENTS)
        delta=-raw*shrink*share*mode_plays_per_drive
        delta=max(-.18,min(0.,delta))
        total+=delta
        events.append({'player':p['name'],'athlete_id':pid,'position':category,
          'reported_status':'OUT','source_events':len(own),
          'replacement_pool_events':len(pool),'recent_player_events':len(rec),
          'recent_share_of_identified_mode':round(share,6),
          'player_epa_per_play':round(player_epa,6),
          'observed_other_team_players_epa_per_play':round(pool_epa,6),
          'sample_reliability':round(shrink,6),
          'bounded_ppd_delta':round(delta,6),
          'no_imputed_snap_counts':True})
    return events,round(float(max(-MAX_SKILL_DELTA_PPD,min(0.,total))),6),gaps

def _qb_delta(players,qb_evidence):
    qb_out={p['athlete_id'] for p in players
            if p.get('athlete_id') and p['status']=='OUT' and _position(p)=='QB'}
    s=(qb_evidence or {}).get('scenario')
    if not s or (qb_evidence or {}).get('status')!='EMPIRICAL_CONDITIONAL_SCENARIO_UNCALIBRATED':
        return 0.,{'status':'NO_SAMPLED_REPLACEMENT_QB_EVIDENCE'}
    if s.get('reference_qb_id') not in qb_out:
        return 0.,{'status':'NO_REPORTED_OUT_RECENT_STARTER_QB'}
    v=float(s['estimated_change_in_points_per_drive'])
    if not math.isfinite(v):raise ValueError('Nonfinite QB effect')
    # Replacement identity itself may not be confirmed; the observed
    # alternate is a scenario, not a guaranteed starting lineup.
    v=max(-MAX_QB_DELTA_PPD,min(MAX_QB_DELTA_PPD,v))
    return round(v,6),{'status':'SOURCE_OUT_PROVISIONAL_SAMPLED_QB_REPLACEMENT',
          'reference':s['reference_qb'],'candidate':s['candidate_qb'],
          'candidate_confirmed_starter':False,
          'historical_pbp_effect_ppd':round(v,6),
          'historical_overlap_games':s.get('shared_prior_game_ids',[])}

def _kicker_flag(players,histories,team_id):
    """Do not attribute team FG record to named starter without kicker IDs."""
    outs=[p for p in players if p['status']=='OUT' and _position(p)=='K']
    if not outs:return {'status':'NO_REPORTED_KICKER_OUT','override':False}
    ids=Counter()
    for r in histories.get(team_id,[]):
        for q in (r.get('special_teams') or {}).get('fg_kicker_ids') or []:
            if q: ids[str(q)]+=1
    attempted=sum(ids.values())
    matching=[p for p in outs if p.get('athlete_id') and ids[str(p['athlete_id'])]>0]
    if not attempted or not matching:
        return {'status':'UNIDENTIFIED_FG_ATTEMPT_KICKER_NO_NUMERIC_OVERRIDE',
          'override':False,'out_names':[p['name'] for p in outs],
          'source_kicker_attributions':attempted}
    majority=[p for p in matching if ids[str(p['athlete_id'])]/attempted>=.5]
    if not majority:
        return {'status':'NO_DOMINANT_OUT_KICKER_NO_OVERRIDE','override':False}
    return {'status':'VERIFIED_PRIMARY_KICKER_OUT_REPLACE_WITH_LEAGUE_PRIOR',
       'override':True,'out_athlete_id':majority[0]['athlete_id'],
       'attributed_team_fg_attempts':attempted,
       'share_prior_attempts':round(ids[str(majority[0]['athlete_id'])]/attempted,6),
       'replacement_kicker_skill_unobserved':True}

def estimate(mapped,eligible,roster,home_id,away_id,statuses,qb_effects,histories):
    """Return default-path personnel PPD increments, or safe zero evidence.

    A secondary linked-conference transcription is admitted provisionally
    only when captured during this exact pregame analysis. Local unverified
    files never silently activate point effects.
    """
    safe=bool(statuses.get('source_captured_during_forecast')
              and statuses.get('secondary_report_captured')
              and statuses.get('captured_at_utc'))
    # Other source adapters can be added once independent verification exists.
    active={};totals={};all_rows=statuses.get('players') or []
    for side,tid in (('home',home_id),('away',away_id)):
        players=[p for p in all_rows if p.get('team_id')==str(tid)]
        rows,last=_records(mapped,eligible,tid)
        audit={'team_id':str(tid),'source_type':statuses.get('status'),
               'reported_out':len([p for p in players if p.get('status')=='OUT']),
               'unknown_starters_not_treated_as_out':True,
               'missing_evidence':[],'affected_players':[],
               'position_groups_unmodeled':[],
               'baseline_prior_game_performance_includes_previous_injuries':True}
        ppd=0.
        if not safe or rows is None:
            audit['status']='NO_TRUSTED_CURRENT_CAPTURE_OR_PBP'
            audit['missing_evidence'].append('LIVE_REPORT_OR_PBP_MISSING')
            audit['applied_ppd']=0.
            audit['kicker']={'status':'NOT_VERIFIED'}
            active[side]=audit;totals[side]=0.;continue
        for kind in ('RB','RECEIVER'):
            events,delta,gaps=_bounded_skill_delta(rows,players,kind,last)
            ppd+=delta
            audit['affected_players']+=events
            audit['missing_evidence']+=gaps
        qb,qe=_qb_delta(players,(qb_effects or {}).get(side))
        ppd+=qb;audit['qb_replacement']=qe
        # OL/DEFENSE: no trustworthy player snap participation or on/off EPA.
        # Keep visible risk but never turn the count of outs into a scoring
        # deduction that was simply invented.
        audit['position_groups_unmodeled']=[{'name':p['name'],
            'position':p.get('position'),'reason':'NO_PLAYER_LEVEL_SNAP_OR_PRESSURE_EVIDENCE'}
            for p in players if p['status']=='OUT'
            and _position(p) in ('OL','DEFENSE_OR_OTHER','SPECIAL')]
        audit['kicker']=_kicker_flag(players,histories,tid)
        ppd=round(float(max(-MAX_TEAM_DELTA_PPD,min(MAX_QB_DELTA_PPD,ppd))),6)
        audit['applied_ppd']=ppd
        audit['status']=('ACTIVE_SOURCE_BACKED_PROVISIONAL_PERSONNEL'
                         if ppd or audit['kicker'].get('override')
                         else 'NO_QUANTIFIABLE_PLAYER_EFFECT')
        active[side]=audit;totals[side]=ppd
    return {'version':VERSION,'status':('ACTIVE' if any(
        x['status']=='ACTIVE_SOURCE_BACKED_PROVISIONAL_PERSONNEL'
        for x in active.values()) else 'NO_QUALIFIED_NUMERIC_ADJUSTMENT'),
        'home':active['home'],'away':active['away'],
        'ppd_home':totals['home'],'ppd_away':totals['away'],
        'conservative_source_mode':'LIVE_SECONDARY_VERIFIED_CAPTURE_NOT_OFFICIAL',
        'fitted_on_player_EPA_opportunity_data':True,
        'causal_injury_point_values_verified':False,'ATS_calibrated':False}
