"""Audit actual 2026 defensive / special-teams points for BOTH matchup teams.

Retains verified point subtotals AND failed-game details separately.
No points are imputed for unreconciled games and nothing changes model inputs.
"""
from __future__ import annotations
import argparse,collections,json
from pathlib import Path
import pandas as pd
from cfb.pipeline import verify,iso_utc
from cfb.fresh_matchup import histories_asof
from cfb.mechanism_build16 import _failure2_verified_scoring_histories
from cfb.scoring_attribution import attribute

def evaluate(snapshot,game_id="401862807"):
    root=Path(snapshot)
    manifest=verify(root)
    cutoff=iso_utc(manifest['captured_at_utc'])
    schedule=pd.read_parquet(root/'schedule.parquet')
    pbp=pd.read_parquet(root/'pbp.parquet')
    roster=pd.read_parquet(root/'roster.parquet')
    target=schedule.loc[schedule.game_id.astype(str).eq(str(game_id))]
    if len(target)!=1:raise ValueError("Expected exactly one target fixture")
    g=target.iloc[0]
    if str(g.home_team)!="Memphis" or str(g.away_team)!="East Carolina":
        raise ValueError("Incorrect matchup teams or venue")
    if cutoff>=pd.Timestamp(g.start_date).tz_convert('UTC'):
        raise ValueError("This is a retrospective capture, NOT an as-of pregame audit")
    _,_,histories,eligible=histories_asof(schedule,pbp,roster,cutoff)
    by_game={str(gid):chunk for gid,chunk in pbp.groupby('game_id',sort=False)}
    ids={"Memphis":int(g.home_id),"East Carolina":int(g.away_id)}
    team_reports={}
    for name,team_id in ids.items():
        rows=histories.get(team_id,[])
        schedule_games=eligible.loc[
            eligible.home_id.eq(team_id)|eligible.away_id.eq(team_id)]
        summary={'all_history_games':len(rows),'eligible_completed_games':len(schedule_games),
                 'verified_games':0,'unverified_games':0,
                 'verified_offensive_points':0,
                 'verified_defensive_return_points':0,
                 'verified_special_teams_points':0,
                 'verified_nonoffensive_points':0,
                 'verified_defensive_return_tds':0,
                 'verified_special_teams_tds':0,
                 'games':[]}
        for _,p in schedule_games.sort_values('_kickoff').iterrows():
            gid=str(p.game_id)
            is_home=int(p.home_id)==team_id
            team_score=float(p.home_points if is_home else p.away_points)
            opp_score=float(p.away_points if is_home else p.home_points)
            opp_name=str(p.away_team if is_home else p.home_team)
            record=next((x for x in rows if x.get('source_game_id')==gid),None)
            pbp_game=by_game.get(gid)
            scoring=attribute(pbp_game,int(p.home_id),int(p.away_id),
                             p.home_points,p.away_points)
            is_ok=scoring['status']=='VERIFIED_RAW_SCORE_RECONCILIATION'
            label='home' if is_home else 'away'
            item={'game_id':gid,'opponent':opp_name,
                  'game_date':str(p._kickoff if hasattr(p,'_kickoff') else p.start_date),
                  'team_score':team_score,'opponent_score':opp_score,
                  'used_by_Build16_histories':record is not None,
                  'status':scoring['status'],
                  'reason':scoring.get('reason'),
                  'offensive_points':scoring['offensive_points'][label] if is_ok else None,
                  'defensive_return_points':scoring['defensive_return_points'][label] if is_ok else None,
                  'special_teams_points':scoring['special_teams_points'][label] if is_ok else None,
                  'nonoffensive_points':scoring['nonoffensive_points'][label] if is_ok else None,
                  'defensive_return_tds':scoring['defensive_return_tds'][label] if is_ok else None,
                  'special_teams_tds':scoring['special_teams_tds'][label] if is_ok else None}
            if is_ok:
                summary['verified_games']+=1
                for key in ('offensive_points','defensive_return_points',
                            'special_teams_points','nonoffensive_points',
                            'defensive_return_tds','special_teams_tds'):
                    summary['verified_'+key]+=int(item[key])
            else:
                summary['unverified_games']+=1
                if pbp_game is not None:
                    # Include original scoring transitions for manual reconciliation.
                    pair=next((x for x in [('home_score','away_score'),
                                         ('homeScore','awayScore')]
                               if set(x).issubset(pbp_game.columns)),None)
                    if pair is not None:
                        t=pbp_game.sort_values('game_play_number')
                        score=t[list(pair)].apply(pd.to_numeric,errors='coerce')
                        delta=score.diff().fillna(score)
                        events=t.loc[delta.gt(0).any(axis=1)]
                        fields=['game_play_number','pos_team_id','pass','rush','int',
                            'pos_fumble_lost','offense_score_play','text','play.text',
                            'type.text','play_type','home_score','away_score']
                        present=[col for col in fields if col in events.columns]
                        item['original_scoring_events']=events[present].head(20).astype(
                            str).to_dict('records')
            summary['games'].append(item)
        summary['verified_games_in_used_history']=sum(
            x['status']=='VERIFIED_RAW_SCORE_RECONCILIATION'
            and x['used_by_Build16_histories'] for x in summary['games'])
        summary['unverified_games_in_used_history']=sum(
            x['status']!='VERIFIED_RAW_SCORE_RECONCILIATION'
            and x['used_by_Build16_histories'] for x in summary['games'])
        summary['verified_nonoffensive_points_not_full_season_total']=(
            summary['unverified_games']>0)
        team_reports[name]=summary
    h,a=ids['Memphis'],ids['East Carolina']
    _,_,activation=_failure2_verified_scoring_histories(h,a,histories)
    return {'schema':'cfb_failure2_actual_2026_team_scoring_exceptions_v1',
            'fixture_id':str(game_id),'snapshot_at_utc':str(cutoff),
            'build16_failure2_status':activation['status'],
            'strict_rule':'Requires every historical game for BOTH teams verified; a single ambiguous game disables mechanism',
            'figures_verified_subtotals_only_until_failed_games_reconciled':True,
            'teams':team_reports}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--game-id',default='401862807')
    args=p.parse_args()
    report=evaluate(args.snapshot,args.game_id)
    out=Path(args.out)
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True))
    console={k:v for k,v in report.items() if k!='teams'}
    console['teams']={}
    for name,team in report['teams'].items():
        console['teams'][name]={k:v for k,v in team.items() if k!='games'}
        console['teams'][name]['games']=[{k:v for k,v in g.items() if k!='original_scoring_events'}
                                           for g in team['games']]
    print('FAILURE2_ECU_MEMPHIS_ACTUAL_2026_TEAM_AUDIT')
    print(json.dumps(console,indent=2,sort_keys=True))
    print('UNVERIFIED_ORIGINAL_SCORING_EVENT_COUNT')
    for name,team in report['teams'].items():
        for g in team['games']:
            if g['status']!='VERIFIED_RAW_SCORE_RECONCILIATION':
                print(json.dumps({'team':name,'game_id':g['game_id'],
                    'opponent':g['opponent'],'reason':g['reason'],
                    'event_examples':g.get('original_scoring_events',[])[:10]}))

if __name__=='__main__':main()
