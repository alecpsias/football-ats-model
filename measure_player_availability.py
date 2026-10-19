#!/usr/bin/env python3
"""v0.8: future-only, measured injury and replacement evidence.

A current report's OUT designation is a verified snapshot *observation*, not
proof of the final inactive list. Replacement identity is NOT certified by a
roster ACT flag. Only an explicit, conditional quarterback scenario may move
points; no generic per-position injury-point multipliers are used.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import pandas as pd
from collect_extra_metrics import games_before_cutoff, parse_stamp
from model import InputError, MODEL_RULES, projection, simulate

MIN_MAIN_QB_SHARE=0.52
MIN_MAIN_QB_DROPS=25
MIN_BACKUP_QB_DROPS=25
PRIOR_EFFECTIVE_DROPBACKS=80
MAX_ABS_QB_PASS_EPA_SHIFT=0.12  # safety bound, NOT empirically calibrated

def _eligible_injury_rows(folder:Path|None, cutoff, team_codes:tuple[str,str],week:int):
    state={'status':'UNAVAILABLE','time_provenance':None,'rows':[], 'warning':None}
    if folder is None: return state
    folder=Path(folder);meta_file=folder/'injury_manifest.json'
    if not meta_file.is_file():return state
    meta=json.loads(meta_file.read_text(encoding='utf-8'))
    if parse_stamp(meta['retrieved_at_utc'])>parse_stamp(cutoff):
        state['status']='REJECTED_AFTER_CUTOFF';return state
    if meta.get('status') not in ('CAPTURED_SNAPSHOT_ONLY','COLLECTED'):
        state['warning']=meta.get('warning');return state
    source=folder/meta['file']
    if hashlib.sha256(source.read_bytes()).hexdigest()!=meta['sha256']:
        raise InputError('Injury snapshot SHA256 mismatch')
    frame=pd.read_csv(source,compression='infer',low_memory=False)
    need={'team','week','gsis_id','position','full_name','report_status'}
    if not need.issubset(frame.columns):
        raise InputError('Injury snapshot missing player-identity or status columns')
    frame=frame[frame.team.isin(team_codes)&(pd.to_numeric(frame.week,errors='coerce')==week)].copy()
    if 'date_modified' in frame.columns and meta.get('row_timestamp_column')=='date_modified':
        t=pd.to_datetime(frame['date_modified'],utc=True,errors='coerce');frame=frame[t.notna()&(t<=pd.Timestamp(cutoff))]
    state.update(status='SNAPSHOT_OBSERVED_NOT_FINAL_INACTIVES',
        time_provenance=meta.get('timestamp_provenance'),
        rows=frame.to_dict('records'))
    return state

def _safe_number(value):
    try:
        f=float(value)
        return f if math.isfinite(f) else None
    except (TypeError,ValueError):return None

def _qb_profiles(pbp:pd.DataFrame,team:str,games:dict):
    data=pbp[pbp.game_id.isin(games)&pbp.posteam.eq(team)&pd.to_numeric(pbp.qb_dropback,errors='coerce').eq(1)].copy()
    if 'play_type' in data:
        data=data[~data.play_type.eq('no_play')]
    data=data[data.passer_player_id.notna()&pd.to_numeric(data.epa,errors='coerce').notna()]
    if data.empty:return {},[],0
    # Normalized unique eligible game ordering; don't sort by fictional future roster changes.
    valid_games=sorted(data.game_id.unique(),key=lambda x:games[x]['kickoff_utc'])
    latest_games=set(valid_games[-3:])
    profiles={}
    for player,rows in data.groupby('passer_player_id'):
        re=rows[rows.game_id.isin(latest_games)]
        profiles[str(player)]={'name':str(rows.passer_player_name.dropna().iloc[-1]) if rows.passer_player_name.notna().any() else str(player),
          'dropbacks':int(len(rows)), 'epa_per_dropback_observed':float(rows.epa.mean()),
          'games_sampled':int(rows.game_id.nunique()),'recent_three_game_dropbacks':int(len(re)),
          'most_recent_game':str(rows.game_id.iloc[-1])}
    return profiles,valid_games,int(sum(v['recent_three_game_dropbacks'] for v in profiles.values()))

def _eligible_roster(roster:pd.DataFrame,team:str,week:int):
    f=roster[(roster.team==team)&(pd.to_numeric(roster.week,errors='coerce')<=week)]
    if f.empty:return {}
    recent=int(f.week.max())
    f=f[f.week.eq(recent)].drop_duplicates('player_id')
    return {str(row.player_id):{'name':str(row.full_name),'position':str(row.position),'status':str(row.status),'pfr_id':str(row.pfr_id) if pd.notna(row.pfr_id) else None,'week':recent} for row in f.itertuples()}

def _skill_usage(pbp,team,games,roster):
    frame=pbp[pbp.game_id.isin(games)&pbp.posteam.eq(team)].copy()
    if 'play_type' in frame:frame=frame[~frame.play_type.eq('no_play')]
    results={}
    for k,player_col,minimum in (('targets','receiver_player_id',8),('carries','rusher_player_id',12)):
        group=frame[frame[player_col].notna()].copy()
        eligible=len(group)
        if not eligible:results[k]=[];continue
        counts=group.groupby(player_col).agg(opportunities=(player_col,'size'),games=('game_id','nunique'),mean_team_play_epa=('epa','mean'))
        summary=[]
        for pid,row in counts.sort_values('opportunities',ascending=False).iterrows():
            if str(pid) not in roster:continue
            desc=roster[str(pid)]
            n=int(row.opportunities)
            if n<minimum:continue
            summary.append({'player_id':str(pid),'name':desc['name'],'position':desc['position'],
                'opportunities':n,'team_share':round(n/eligible,4),'games':int(row.games),
                'observed_team_play_epa':round(float(row.mean_team_play_epa),4) if pd.notna(row.mean_team_play_epa) else None,
                'warning':'This is TEAM play EPA on a player touch/target, not player-causal EPA or an injury point effect'})
        results[k]=summary
    return results

def _skill_replacements(absent, usage, roster):
    comparisons=[]
    for entry in absent:
        pos=entry['position'].upper()
        metric='targets' if pos in ('WR','TE') else 'carries' if pos in ('RB','FB') else None
        if metric is None:continue
        observed={x['player_id']:x for x in usage[metric]}
        previous=observed.get(entry['id'])
        # Rank prior-volume peers at same roster position. This is evidence
        # of an available HISTORY, not an availability/role forecast.
        excluded={r['id'] for r in absent}
        peers=[x for x in usage[metric] if x['player_id'] not in excluded and x['position']==pos and
               roster.get(x['player_id'],{}).get('status')=='ACT']
        comparisons.append({'reported_out':entry,'history':previous,
             'top_measured_same_position_peers':peers[:3],
             'predicted_target_or_carry_redistribution':None,
             'point_adjustment':None,
             'status':'MEASURED_ROLE_NOT_CAUSAL_REPLACEMENT',
             'reason':'Team play EPA on a target/carry is not an isolated player effect; no confirmed new target allocation'})
    return comparisons


def _pass_rush_replacements(metrics_dir,cutoff,games,team,roster,absent):
    """PFR per-game defensive pressure evidence, no automatic point translation."""
    if metrics_dir is None:return []
    d=Path(metrics_dir);mfile=d/'metrics_manifest.json'
    if not mfile.exists():return []
    m=json.loads(mfile.read_text())
    source=m.get('pfr',{}).get('def',{})
    if source.get('status')!='COLLECTED':return []
    if parse_stamp(source['capture_time_utc'])>parse_stamp(cutoff):raise InputError('Defensive pressure file captured after cutoff')
    path=d/source['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:
        raise InputError('PFR defense file checksum mismatch')
    data=pd.read_csv(path,compression='infer')
    require={'game_id','team','pfr_player_id','def_pressures'}
    if not require.issubset(data):raise InputError('PFR defense file missing pressure columns')
    data=data[(data.team==team)&data.game_id.isin(games)].copy()
    data['def_pressures']=pd.to_numeric(data.def_pressures,errors='coerce')
    data=data[data.def_pressures.notna() & data.def_pressures.ge(0)]
    counts=data.groupby('pfr_player_id').agg(pressures=('def_pressures','sum'),games=('game_id','nunique'))
    profiles={}
    for gsis,info in roster.items():
        pfr_id=info.get('pfr_id')
        if not pfr_id or pfr_id not in counts.index:continue
        row=counts.loc[pfr_id]
        profiles[gsis]={'name':info['name'],'pressure_events':int(row.pressures),
             'games_observed':int(row.games),
             'pressures_per_game_appearance':round(float(row.pressures)/row.games,4),
             'denominator_not_snaps':True}
    output=[]
    excluded={x['id'] for x in absent}
    for entry in absent:
        if entry['position'].upper() not in ('DE','DT','DL','EDGE','LB','OLB','ILB'):continue
        peers=[(pid,p) for pid,p in profiles.items() if pid not in excluded and roster[pid]['status']=='ACT' and
               roster[pid]['position'].upper() in ('DE','DT','DL','EDGE','LB','OLB','ILB') and p['games_observed']>=2]
        peers.sort(key=lambda item:item[1]['pressures_per_game_appearance'],reverse=True)
        output.append({'reported_out':entry,'observed_pressure_profile':profiles.get(entry['id']),
             'potential_peer_pressure_profiles':[p for _,p in peers[:3]],
             'point_adjustment':None,'status':'OBSERVED_PRESSURE_ONLY_NO_SNAP_BASE',
             'note':'PFR player pressure counts are not distinct team pressure plays; without defensive snaps and replacement assignments no causal rate or points adjustment is justified'})
    return output


def measure(snapshot:Path,injury_dir:Path|None,cutoff,home:str,away:str,week:int,pbp=None,roster=None,metrics_dir=None):
    """Outputs explicit conditional QB replacement signals and descriptive roles."""
    from nflverse_connector import verify
    from ingest import load_table
    base=Path(snapshot)
    if verify(base).get('status')!='VERIFIED':raise InputError('Unverified base source')
    meta=json.loads((base/'manifest.json').read_text())
    for k in ('pbp','schedule','roster'):
        if parse_stamp(meta['datasets'][k]['captured_at_utc'])>parse_stamp(cutoff):raise InputError(f'{k} source after cutoff')
    inj=_eligible_injury_rows(injury_dir,cutoff,(home,away),week)
    if pbp is None:pbp=load_table(base/meta['datasets']['pbp']['local_file'])
    if roster is None:roster=load_table(base/meta['datasets']['roster']['local_file'])
    schedule_file=base/meta['datasets']['schedule']['local_file']
    games=games_before_cutoff(schedule_file,parse_stamp(cutoff),hours=6)
    report={'schema':'player_availability_evidence_v0.8','as_of_utc':parse_stamp(cutoff).isoformat(),
      'injury_source_status':inj['status'],'injury_time_provenance':inj['time_provenance'],
      'teams':{},'assumptions':['OUT is a collected injury report designation, not an official inactive list',
         'An ACT weekly roster status is not confirmation of a starting assignment',
         'QB effects use observational per-dropback EPA and explicit shrinkage; uncalibrated',
         'WR/RB/OL/defender reports have no automatic scoring impact without defensible position-level attribution']}
    for team in (home,away):
        player_roster=_eligible_roster(roster,team,week)
        profiles,games_recent,recent_total=_qb_profiles(pbp,team,games)
        historical_total=sum(q['dropbacks'] for q in profiles.values())
        qb_recent=[(pid,p) for pid,p in profiles.items() if p['recent_three_game_dropbacks']>=MIN_MAIN_QB_DROPS and recent_total and p['recent_three_game_dropbacks']/recent_total>=MIN_MAIN_QB_SHARE]
        qb_recent.sort(key=lambda x:x[1]['recent_three_game_dropbacks'],reverse=True)
        main_id=qb_recent[0][0] if len(qb_recent)==1 else None
        reports=[i for i in inj['rows'] if i['team']==team]
        absent=[{'id':str(i.get('gsis_id')),'name':str(i.get('full_name')),
                 'position':str(i.get('position')),'reported_status':'OUT'} for i in reports if str(i.get('report_status','')).strip().lower()=='out']
        uncertain=[{'id':str(i.get('gsis_id')),'name':str(i.get('full_name')),'position':str(i.get('position')),
                    'reported_status':str(i.get('report_status'))} for i in reports if str(i.get('report_status','')).strip().lower() in ('questionable','doubtful')]
        evidence={'reported_out':absent,'uncertain_status':uncertain,'historical_main_qb_id':main_id,
          'historical_main_qb_name':profiles.get(main_id,{}).get('name'),
          'qb_eligible_games':len(games_recent),'qb_profiles':profiles,
          'skill_position_usage':_skill_usage(pbp,team,games,player_roster),
          'conditional_qb_replacement':None,'warnings':[]}
        evidence['skill_replacement_evidence']=_skill_replacements(absent,evidence['skill_position_usage'],player_roster)
        evidence['pass_rush_replacement_evidence']=_pass_rush_replacements(metrics_dir,cutoff,games,team,player_roster,absent)
        if inj['status']!='SNAPSHOT_OBSERVED_NOT_FINAL_INACTIVES':evidence['warnings'].append('Injury report unavailable or post cutoff; no availability-driven adjustment')
        if main_id and main_id in {x['id'] for x in absent}:
            # Select candidate by measured prior dropbacks, WITH current roster presence.
            choices=[(pid,x) for pid,x in profiles.items() if pid!=main_id and x['dropbacks']>=MIN_BACKUP_QB_DROPS and pid in player_roster and
                player_roster[pid]['status']=='ACT' and player_roster[pid]['position']=='QB']
            choices.sort(key=lambda item:item[1]['dropbacks'],reverse=True)
            if not choices:
                evidence['warnings'].append('Starting QB reported OUT but no eligible measured replacement; cannot quantify')
            elif len(choices)>1 and choices[1][1]['dropbacks'] >= .8*choices[0][1]['dropbacks']:
                evidence['warnings'].append('Multiple plausible measured replacement QBs; no singular scenario')
            else:
                pid,b=choices[0]
                observed_mix=sum(x['dropbacks']*x['epa_per_dropback_observed'] for x in profiles.values())/max(historical_total,1)
                raw_difference=b['epa_per_dropback_observed']-observed_mix
                weight=b['dropbacks']/(b['dropbacks']+PRIOR_EFFECTIVE_DROPBACKS)
                shrink_delta=max(-MAX_ABS_QB_PASS_EPA_SHIFT,min(MAX_ABS_QB_PASS_EPA_SHIFT,raw_difference*weight))
                # An identified backup is only an inferred candidate; replacement assignment NOT verified.
                evidence['conditional_qb_replacement']={'status':'CONDITIONAL_NOT_CONFIRMED_STARTER',
                   'reported_out_id':main_id,'reported_out_name':profiles[main_id]['name'],
                   'candidate_id':pid,'candidate_name':player_roster[pid]['name'],
                   'candidate_measured_dropbacks':b['dropbacks'],
                   'candidate_observed_epa_per_dropback':round(b['epa_per_dropback_observed'],5),
                   'team_history_qb_mix_epa_per_dropback':round(observed_mix,5),
                   'raw_epa_difference':round(raw_difference,5),
                   'shrinkage_weight':round(weight,5),
                   'pass_epa_shift_if_candidate_starts':round(shrink_delta,5),
                   'model_effect_is_capped':abs(raw_difference*weight)>MAX_ABS_QB_PASS_EPA_SHIFT,
                   'source':'timestamped nflverse injury snapshot, roster, and eligible completed-game PBP',
                   'notes':'Conditional scenario ONLY. Noncausal adjusted-EPA proxy, not a verified starter or calibrated player WAR.'}
        elif any(x['position']=='QB' for x in absent):
            evidence['warnings'].append('Reported OUT QB is not sole qualified recent primary QB; no starter-loss point adjustment')
        report['teams'][team]=evidence
    return report

def scenario_points(game:dict, rules:dict, player_measurement:dict,runs:int,seed:int):
    """Compare same-seed projection to a measured QB scenario; baseline is unchanged."""
    from copy import deepcopy
    outcomes=[]
    base=deepcopy(game)
    hfa=0 if base['neutral_site'] else float(rules['home_advantage_points'])
    def calc(g):
        ph=projection(g['home'],g['away'],rules,'home',.5*hfa)
        pa=projection(g['away'],g['home'],rules,'away',-.5*hfa)
        return simulate(ph,pa,rules,None,None,runs,seed)
    ref=calc(base)
    for side in ('home','away'):
        code=base[side]['name']
        candidate=player_measurement['teams'][code]['conditional_qb_replacement']
        if candidate is None:continue
        delta=float(candidate['pass_epa_shift_if_candidate_starts'])
        modified=deepcopy(base)
        modified[side]['personnel']['pass_epa_shift']=delta
        modified[side]['personnel']['source']=candidate['source']
        modified[side]['personnel']['notes']=candidate['notes']
        result=calc(modified)
        outcomes.append({'team':code,'condition':f"IF {candidate['candidate_name']} starts for {candidate['reported_out_name']}",
            'candidate':candidate,'projected_points_home':result['mean_score_home'],
            'projected_points_away':result['mean_score_away'],
            'home_margin':result['mean_margin_home'],
            'delta_home_margin_vs_neutral':round(result['mean_margin_home']-ref['mean_margin_home'],3),
            'not_base_forecast':True})
    return outcomes
