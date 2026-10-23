"""Deterministic tests: source capture + sample-backed default personnel effects."""
from __future__ import annotations
import io
import unittest
import pandas as pd
from cfb.live_availability import parse_report,fetch_matchup
from cfb.personnel_active import estimate
from cfb.mechanism_build16 import simulate_build16

HTML="""<html><body><a href="https://theacc.com/sports/2025/8/28/availability-reporting-football.aspx">official</a>
<h3>NC State</h3><table><tr><th>Player</th><th>Pos</th><th>Status</th></tr>
<tr><td>Jayden &ldquo;Duke&rdquo; Scott #1</td><td>RB</td><td>Out</td></tr>
<tr><td>Nick Konieczynski #91</td><td>K</td><td>Out</td></tr>
<tr><td>Rylan Vann #55</td><td>OL</td><td>Out</td></tr></table>
<h3>Stanford</h3><table><tr><th>Player</th><th>Pos</th><th>Status</th></tr>
<tr><td>Matt Rose #35</td><td>ILB</td><td>Out</td></tr>
<tr><td>Micah Ford #20</td><td>RB</td><td>Probable</td></tr></table>
<h2>Unofficial Injury Notes</h2>
<table><tr><td>Someone #4</td><td>WR</td><td>Out</td></tr></table></body></html>"""
class Stream(io.BytesIO):
    pass
def mock_open(req,timeout=9):
    assert req.full_url=='https://kickoffstatus.com/games/2026-10-23-nc-state-at-stanford/'
    return Stream(HTML.encode())

def prior_data():
    roster=pd.DataFrame([
        {'team_id':152,'athlete_id':11,'display_name':'Jayden Scott','position_abbreviation':'RB'},
        {'team_id':152,'athlete_id':22,'display_name':'Available Back','position_abbreviation':'RB'},
        {'team_id':152,'athlete_id':91,'display_name':'Nick Konieczynski','position_abbreviation':'K'},
        {'team_id':152,'athlete_id':55,'display_name':'Rylan Vann','position_abbreviation':'OL'},
        {'team_id':24,'athlete_id':35,'display_name':'Matt Rose','position_abbreviation':'ILB'},
        {'team_id':24,'athlete_id':20,'display_name':'Micah Ford','position_abbreviation':'RB'}])
    games=pd.DataFrame([
        {'game_id':100+i,'home_id':152,'away_id':24,'_kickoff':pd.Timestamp('2026-09-01',tz='UTC')+pd.Timedelta(days=i*7)}
        for i in range(3)])
    plays=[]
    for i in range(3):
        for j in range(24):
            out=(j%2==0)
            plays.append({'game_id':100+i,'pos_team_id':152,'drive_id':1+j//4,
                'EPA':.60 if out else -.10,
                'rush':True,'pass':False,
                'rusher_player_id':11 if out else 22,
                'receiver_player_id':None})
    return pd.DataFrame(plays),games,roster

class LivePersonnel(unittest.TestCase):
    def test_capture_linked_source_and_roster_disambiguation(self):
        pbp,eligible,roster=prior_data()
        got=fetch_matchup('NC State','Stanford',152,24,
          '2026-10-24T02:30:00+00:00',roster=roster,
          observed_at='2026-10-23T18:30:00+00:00',opener=mock_open)
        self.assertEqual(got['status'],'PREGAME_SECONDARY_CONFERENCE_TRANSCRIPTION_CAPTURED')
        self.assertTrue(got['source_captured_during_forecast'])
        self.assertFalse(got['source_verified'])
        self.assertEqual(len(got['players']),5)
        self.assertEqual(next(x for x in got['players'] if x['name'].startswith('Jayden'))['athlete_id'],'11')
        self.assertTrue(next(x for x in got['players'] if x['name']=='Nick Konieczynski')['is_reported_out'])
        self.assertFalse(any(p['name']=='Someone' for p in got['players']))
    def test_pre_kickoff_only_and_missing_report_fail_closed(self):
        late=fetch_matchup('NC State','Stanford',152,24,'2026-10-24T02:30:00+00:00',
                           observed_at='2026-10-24T02:31:00+00:00',opener=mock_open)
        self.assertEqual(late['status'],'NO_VERIFIED_LIVE_AVAILABILITY')
    def test_skills_numeric_only_when_observed_replacements(self):
        mapped,games,roster=prior_data()
        cap=fetch_matchup('NC State','Stanford',152,24,
          '2026-10-24T02:30:00+00:00',roster=roster,
          observed_at='2026-10-23T18:30:00+00:00',opener=mock_open)
        x=estimate(mapped,games,roster,24,152,cap,{}, {152:[],24:[]})
        self.assertLess(x['ppd_away'],0.)
        self.assertGreaterEqual(x['ppd_away'],-.36)
        self.assertEqual(x['ppd_home'],0.)
        self.assertTrue(any(p['position']=='RB' for p in x['away']['affected_players']))
        self.assertEqual(x['away']['kicker']['status'],'UNIDENTIFIED_FG_ATTEMPT_KICKER_NO_NUMERIC_OVERRIDE')
        self.assertTrue(any(p['position']=='OL' for p in x['away']['position_groups_unmodeled']))
        self.assertEqual(x['home']['status'],'NO_QUANTIFIABLE_PLAYER_EFFECT')
        # A fabricated report without in-run source attestation is disallowed.
        cap['source_captured_during_forecast']=False
        y=estimate(mapped,games,roster,24,152,cap,{}, {152:[],24:[]})
        self.assertEqual(y['ppd_away'],0.)
    def test_qb_out_requires_sampled_replacement_and_named_id(self):
        mapped,games,roster=prior_data()
        cap={'status':'PREGAME_SECONDARY_CONFERENCE_TRANSCRIPTION_CAPTURED',
             'captured_at_utc':'2026-10-23T18:30:00+00:00',
             'source_captured_during_forecast':True,'secondary_report_captured':True,
             'players':[{'team_id':'152','athlete_id':'80','name':'QB One',
                         'position':'QB','status':'OUT'}]}
        qb={'away':{'status':'EMPIRICAL_CONDITIONAL_SCENARIO_UNCALIBRATED',
                   'scenario':{'reference_qb_id':'80','reference_qb':'QB One',
                   'candidate_qb':'QB Two','estimated_change_in_points_per_drive':-.18}}}
        x=estimate(mapped,games,roster,24,152,cap,qb,{24:[],152:[]})
        self.assertAlmostEqual(x['ppd_away'],-.18,places=5)
        cap['players'][0]['status']='QUESTIONABLE'
        y=estimate(mapped,games,roster,24,152,cap,qb,{24:[],152:[]})
        self.assertEqual(y['ppd_away'],0.)
    def test_no_report_preserves_no_change(self):
        mapped,games,roster=prior_data()
        x=estimate(mapped,games,roster,24,152,{'players':[]},{},{})
        self.assertEqual(x['status'],'NO_QUALIFIED_NUMERIC_ADJUSTMENT')
        self.assertEqual(x['ppd_home'],0.)
        self.assertEqual(x['ppd_away'],0.)

if __name__=='__main__':unittest.main()
