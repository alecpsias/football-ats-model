"""Build16 free ESPN play-text disruption proxy. Not charted pressure or all dropbacks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

VERSION='cfb-build16-recorded-disruption-v1'
REQUIRED={'pos_team_id','pass','sack','qb_hurry'}

def observed_disruption(plays,team_id):
    """Count each observed sack/hurry pass play once; null is not clean-pocket evidence."""
    base={'status':'UNAVAILABLE','version':VERSION,'proxy':'PASS_OR_SACK_PLAYS_NOT_ALL_DROPBACKS',
          'denominator':0,'annotated':0,'sacks':None,'hurries':None,
          'non_sack_hurries':None,'disrupted_unique':None,'annotation_coverage':None,
          'limitation':'ESPN text-based hurry annotation is not full charted pressure.'}
    if not isinstance(plays,pd.DataFrame) or not REQUIRED.issubset(plays.columns):
        return {**base,'reason':'MISSING_REQUIRED_SOURCE_COLUMNS'}
    off=plays.loc[plays.pos_team_id.eq(team_id)]
    eligible=off.loc[off['pass'].eq(True)|off.sack.eq(True)]
    n=len(eligible)
    if not n:return {**base,'reason':'NO_IDENTIFIABLE_PASS_OR_SACK_PLAYS'}
    complete=eligible.loc[eligible.qb_hurry.notna() & eligible.sack.notna()]
    coverage=len(complete)
    base.update({'denominator':n,'annotated':coverage,'annotation_coverage':round(coverage/n,6)})
    if coverage!=n:
        return {**base,'reason':'PARTIAL_HURRY_OR_SACK_EVENT_COVERAGE'}
    sack=complete.sack.eq(True);hurry=complete.qb_hurry.eq(True)
    return {**base,'status':'OBSERVED_TEXT_FLAGGED_DISRUPTION_PROXY','reason':None,
            'sacks':int(sack.sum()),'hurries':int(hurry.sum()),
            'non_sack_hurries':int((hurry & ~sack).sum()),
            'disrupted_unique':int((hurry|sack).sum())}

def matchup_disruption(offense_history,defense_history):
    """Offense allowed vs opponent defense generated; pregame evidence only."""
    result={'status':'UNAVAILABLE_INCOMPLETE_PROXY','version':VERSION,
            'recorded_disruption_match_rate':None,
            'recorded_non_sack_hurry_match_rate':None,
            'offense_allowed_rate':None,'defense_generated_rate':None,
            'offense_pass_or_sack_plays':0,'defense_pass_or_sack_plays':0,
            'scoring_adjustment_ppd':0.0,'fitted_incremental_scoring_effect':False,
            'limitations':['Not a charted pressure rate or all dropbacks',
                           'Unreported hurries may be mislabeled as clean pockets',
                           'No fitted incremental scoring effect; diagnostic only']}
    def usable(rows,key):
        result=[]
        for r in rows:
            s=r.get(key) if isinstance(r,dict) else None
            if isinstance(s,dict) and s.get('status')=='OBSERVED_TEXT_FLAGGED_DISRUPTION_PROXY' and s.get('annotation_coverage')==1 and s.get('denominator',0)>0:
                result.append(s)
        return result
    a=usable(offense_history,'disruption_off')
    d=usable(defense_history,'disruption_def')
    an=sum(v['denominator'] for v in a);dn=sum(v['denominator'] for v in d)
    result.update({'offense_pass_or_sack_plays':an,'defense_pass_or_sack_plays':dn,
                   'offense_prior_games':len(a),'defense_prior_games':len(d)})
    if len(a)<3 or len(d)<3 or min(an,dn)<60:return result
    oa=sum(v['disrupted_unique'] for v in a);dd=sum(v['disrupted_unique'] for v in d)
    oh=sum(v['non_sack_hurries'] for v in a);dh=sum(v['non_sack_hurries'] for v in d)
    pooled=(oa+dd)/(an+dn);pooled_h=(oh+dh)/(an+dn)
    matched=((oa+100*pooled)/(an+100)+(dd+100*pooled)/(dn+100))/2
    hurry=((oh+100*pooled_h)/(an+100)+(dh+100*pooled_h)/(dn+100))/2
    return {**result,'status':'OBSERVED_DESCRIPTIVE_ONLY_NOT_CHARTED_PRESSURE',
            'recorded_disruption_match_rate':round(matched,6),
            'recorded_non_sack_hurry_match_rate':round(hurry,6),
            'offense_allowed_rate':round(oa/an,6),'defense_generated_rate':round(dd/dn,6),
            'offense_recorded_events':oa,'defense_recorded_events':dd,
            'offense_recorded_non_sack_hurries':oh,'defense_recorded_non_sack_hurries':dh}

def audit_snapshot(snapshot,out):
    """Audit actual FBS-involving PBP before allowing any modeled effect."""
    from cfb.pipeline import verify,normalize_pbp_teams,split_games,iso_utc
    root=Path(snapshot);manifest=verify(root)
    at=iso_utc(manifest['captured_at_utc'])
    schedule=pd.read_parquet(root/'schedule.parquet')
    plays=pd.read_parquet(root/'pbp.parquet')
    games,_,past=split_games(schedule,plays,at)
    mapped=normalize_pbp_teams(past.loc[past.game_id.isin(games.game_id)],games)
    teams={str(int(t)):observed_disruption(mapped,int(t))
           for t in sorted(mapped.pos_team_id.dropna().unique())}
    eligible=[v for v in teams.values() if v['status']=='OBSERVED_TEXT_FLAGGED_DISRUPTION_PROXY']
    total=sum(v['denominator'] for v in eligible)
    hurry=sum(v['hurries'] for v in eligible)
    sack=sum(v['sacks'] for v in eligible)
    unique=sum(v['disrupted_unique'] for v in eligible)
    result={'status':'SOURCE_AUDIT_NO_SCORING_PROMOTION','version':VERSION,
            'snapshot_captured_at_utc':at.isoformat(),
            'pbp_rows':len(mapped),'school_count':len(teams),
            'schools_with_complete_observed_proxy':len(eligible),
            'proxy_play_count':total,'sack_flag_count':sack,'recorded_hurry_flag_count':hurry,
            'unique_sack_or_hurry_plays':unique,
            'observed_disruption_rate_not_charted_pressure':round(unique/total,6) if total else None,
            'observed_hurry_rate':round(hurry/total,6) if total else None,
            'school_observations':teams,
            'limitations':['Does not capture all dropbacks or full charted pressure',
                           'Retrospective source snapshot cannot prove past publication times',
                           'No new scoring coefficient applied without independent validation']}
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='school_observations'},indent=2))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();audit_snapshot(a.snapshot,a.out)

if __name__=='__main__':main()
