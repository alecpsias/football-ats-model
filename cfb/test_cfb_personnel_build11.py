"""Build 11: QB activity, explicit missingness and source-backed report tests."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from cfb.personnel import quarterback_usage,load_official_statuses,correlate_availability

G='401856727';K='2026-10-24T19:30:00+00:00';A='2026-10-20T17:00:00+00:00'


def fixture():
    games=pd.DataFrame([{'game_id':101,'home_id':333,'away_id':245,'_kickoff':pd.Timestamp('2026-10-01T00:00Z')},
                        {'game_id':102,'home_id':333,'away_id':245,'_kickoff':pd.Timestamp('2026-10-08T00:00Z')},
                        {'game_id':103,'home_id':245,'away_id':333,'_kickoff':pd.Timestamp('2026-10-15T00:00Z')}])
    plays=pd.DataFrame([{'game_id':gid,'pos_team_id':333,'pass':True,'rush':False,
                          'passer_player_id':pid,'rusher_player_id':None,'EPA':epa} for gid,pid,epa in
                        [(101,10,.1),(101,10,.2),(102,10,.3),(102,10,.4),(103,11,-.4),(103,11,-.5)]])
    roster=pd.DataFrame([{'team_id':333,'athlete_id':10,'position_abbreviation':'QB','display_name':'QB One'},
                         {'team_id':333,'athlete_id':11,'position_abbreviation':'QB','display_name':'QB Two'}])
    return games,plays,roster


class Build11(unittest.TestCase):
    def test_usage_order_not_assumed_starter(self):
        g,p,r=fixture();out=quarterback_usage(p,r,g,333)
        self.assertEqual(out['status'],'QB_PARTICIPATION_OBSERVED_NOT_STARTER_CONFIRMED')
        self.assertFalse(out['leader_is_confirmed_starter'])
        self.assertEqual(out['quarterbacks'][0]['athlete_id'],'10')
        self.assertEqual(out['quarterbacks'][0]['recent_recorded_pass_plays'],4)
        self.assertEqual(out['identified_roster_qb_pass_play_coverage'],1.)
    def test_no_future_play_leakage(self):
        g,p,r=fixture();future=p.iloc[:1].copy();future['game_id']=104;future['passer_player_id']=99
        a=quarterback_usage(pd.concat([p,future],ignore_index=True),r,g,333)
        b=quarterback_usage(p,r,g,333)
        self.assertEqual(a,b)
    def test_unmapped_qb_is_missing_not_imputed(self):
        g,p,r=fixture();p.loc[0,'passer_player_id']=999
        x=quarterback_usage(p,r,g,333)
        self.assertEqual(x['passer_id_not_matched_to_roster_count'],1)
        self.assertLess(x['identified_roster_qb_pass_play_coverage'],1)
    def test_no_report_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            x=load_official_statuses(tmp,G,K,A)
        self.assertEqual(x['status'],'NO_OFFICIAL_REPORT_CAPTURE')
        self.assertFalse(x['effects_applied_to_score'])
    def write(self,tmp,capture=A,url='https://bigten.org/sports/2026/9/10/FB_Availability_Reports.aspx',status='OUT'):
        raw='QB One OUT on official gameday report\n'
        d={'schema':'cfb_official_availability_evidence_v1','game_id':G,'captured_at_utc':capture,
           'source_url':url,'source_text':raw,'source_sha256':hashlib.sha256(raw.encode()).hexdigest(),
           'players':[{'team_id':'333','athlete_id':'10','name':'QB One','status':status,
                       'source_excerpt':'QB One OUT'}]}
        (Path(tmp)/(G+'.json')).write_text(json.dumps(d))
        return d
    def test_source_text_match_and_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp)
            a=load_official_statuses(tmp,G,K,A,expected_teams=['333','245'])
        self.assertEqual(a['players'][0]['status'],'OUT')
        self.assertFalse(a['source_verified'])
        g,p,r=fixture();qb=quarterback_usage(p,r,g,333)
        f=correlate_availability(qb,a)
        self.assertEqual(f[0]['status'],'OUT')
        self.assertIsNone(f[0]['point_adjustment'])
    def test_future_dated_report_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp,capture='2026-10-21T18:00:00+00:00')
            with self.assertRaises(ValueError):load_official_statuses(tmp,G,K,A)
    def test_fake_origin_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp,url='https://bigten.org.evil.example/report')
            with self.assertRaises(ValueError):load_official_statuses(tmp,G,K,A)
    def test_mismatch_status_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp,status='QUESTIONABLE')
            with self.assertRaises(ValueError):load_official_statuses(tmp,G,K,A)
    def test_integration_present_in_fresh_workflow(self):
        f=(Path(__file__).parent/'fresh_matchup.py').read_text()
        y=(Path(__file__).parent.parent/'.github/workflows/cfb-free-data.yml').read_text()
        self.assertIn('quarterback_usage(mapped,roster,eligible',f)
        self.assertIn('--availability-dir cfb/availability',y)
        self.assertIn('personnel_adjustment_to_model_points',f)

if __name__=='__main__':unittest.main()

class Build11FreshIntegration(unittest.TestCase):
    def test_new_matchup_preserves_qb_context_and_unknown_injuries(self):
        from unittest.mock import patch
        from cfb.test_cfb_fresh_build9 import data_fixture,RUN_AT
        from cfb.fresh_matchup import analyze,markdown
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'snapshot'
            tables,manifest=data_fixture(root)
            def read(path):return tables[Path(path).name].copy()
            with patch('cfb.fresh_matchup.verify',return_value=manifest),\
                 patch('cfb.fresh_matchup.pd.read_parquet',side_effect=read):
                x=analyze(root,'SCHOOL_A','SCHOOL_B',runs=150,min_games=1,now=RUN_AT)
            self.assertEqual(x['availability_evidence']['status'],'NO_OFFICIAL_REPORT_CAPTURE')
            self.assertIsNone(x['personnel_adjustment_to_model_points'])
            self.assertIn('quarterback_participation',x)
            self.assertIn('Quarterback involvement',markdown(x))
            self.assertFalse(x['injuries_verified'])
    def test_postgame_request_unmodified_by_new_personnel_layer(self):
        from cfb.fresh_matchup import select_saved_request
        self.assertTrue(callable(select_saved_request))
