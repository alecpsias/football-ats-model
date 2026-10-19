from __future__ import annotations
import copy, hashlib, json, tempfile, unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from collect_injury_reports import collect
from measure_player_availability import (measure, scenario_points, _pass_rush_replacements,
                                              MIN_BACKUP_QB_DROPS)
from model import InputError, MODEL_RULES

NOW=datetime(2026,10,19,1,7,tzinfo=timezone.utc)
CUTOFF='2026-10-19T01:10:00Z'
TEAM='SF'
GAMES={f'2026_0{week}_WAS_SF':{'home':'SF','away':'WAS','kickoff_utc':f'2026-10-{week+9:02}T18:00:00+00:00'} for week in (1,2,3)}

class Stub:
    def __init__(self,rows):self.rows=rows
    def load_injuries(self,season):return pd.DataFrame(self.rows)

class PersonnelEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.base=self.root/'snapshot';self.base.mkdir()
        t={'captured_at_utc': '2026-10-18T20:00:00Z','local_file':'none.csv.gz'}
        (self.base/'manifest.json').write_text(json.dumps({'datasets':{x:t for x in ('pbp','schedule','roster')}}))
        self.inj=self.root/'injuries'
        self.roster=pd.DataFrame([{'team':TEAM,'week':6,'player_id':i,'full_name':name,'status':'ACT','position':p,'pfr_id':pfr} for i,name,p,pfr in
          [('QB1','Primary QB','QB','pQB1'),('QB2','Secondary QB','QB','pQB2'),('QB3','Third QB','QB','pQB3'),
           ('W1','Injured Receiver','WR','pW1'),('W2','Active Receiver','WR','pW2'),
           ('D1','Injured Edge','DL','pD1'),('D2','Active Edge','DL','pD2')]])
        plays=[]
        for week,gid in enumerate(GAMES,1):
            for i in range(30): plays.append(dict(game_id=gid,posteam=TEAM,qb_dropback=1,passer_player_id='QB1',passer_player_name='Primary QB',epa=.2,receiver_player_id='W1' if i<10 else 'W2',rusher_player_id=None,play_type='pass'))
            for i in range(10): plays.append(dict(game_id=gid,posteam=TEAM,qb_dropback=1,passer_player_id='QB2',passer_player_name='Secondary QB',epa=-.1,receiver_player_id='W2',rusher_player_id=None,play_type='pass'))
        self.pbp=pd.DataFrame(plays)
    def make_injuries(self,status='Out',id='QB1',name='Primary QB',position='QB'):
        rows=[dict(season=2026,week=6,team=TEAM,gsis_id=id,full_name=name,position=position,report_status=status)]
        return collect(2026,self.inj,Stub(rows),now=NOW)
    def eval(self,cutoff=CUTOFF,metrics_dir=None):
        with patch('nflverse_connector.verify',return_value={'status':'VERIFIED'}),patch('measure_player_availability.games_before_cutoff',return_value=GAMES):
            return measure(self.base,self.inj,cutoff,TEAM,'WAS',6,pbp=self.pbp,roster=self.roster,metrics_dir=metrics_dir)
    def test_out_major_qb_has_measured_conditional_impact(self):
        self.make_injuries()
        q=self.eval()['teams'][TEAM]['conditional_qb_replacement']
        self.assertIsNotNone(q)
        self.assertEqual(q['candidate_id'],'QB2')
        self.assertEqual(q['candidate_measured_dropbacks'],30)
        self.assertTrue(-.12<=q['pass_epa_shift_if_candidate_starts']<0)
        self.assertEqual(q['status'],'CONDITIONAL_NOT_CONFIRMED_STARTER')
    def test_questionable_not_assumed_out(self):
        self.make_injuries(status='Questionable')
        self.assertIsNone(self.eval()['teams'][TEAM]['conditional_qb_replacement'])
    def test_reported_out_backup_not_equal_starter_loss(self):
        self.make_injuries(id='QB2',name='Secondary QB')
        q=self.eval()['teams'][TEAM]
        self.assertIsNone(q['conditional_qb_replacement'])
        self.assertIn('not sole qualified recent primary QB',' '.join(q['warnings']))
    def test_no_measured_replacement_no_guess(self):
        self.make_injuries();self.pbp=self.pbp[self.pbp.passer_player_id.eq('QB1')].copy()
        q=self.eval()['teams'][TEAM]
        self.assertIsNone(q['conditional_qb_replacement'])
        self.assertIn('no eligible measured replacement',' '.join(q['warnings']))
    def test_no_post_cutoff_injuries(self):
        self.make_injuries();out=self.eval('2026-10-18T23:59:00Z')
        self.assertEqual(out['injury_source_status'],'REJECTED_AFTER_CUTOFF')
        self.assertIsNone(out['teams'][TEAM]['conditional_qb_replacement'])
    def test_hash_tampering_detected(self):
        meta=self.make_injuries();(self.inj/meta['file']).write_bytes(b'tampered')
        with self.assertRaisesRegex(InputError,'SHA256'):self.eval()
    def test_missing_report_no_confirmation(self):
        out=self.eval();self.assertEqual(out['injury_source_status'],'UNAVAILABLE')
        self.assertEqual(out['teams'][TEAM]['reported_out'],[])
    def test_receiver_opportunities_and_peer_not_assigned_points(self):
        self.make_injuries(id='W1',name='Injured Receiver',position='WR')
        report=self.eval()['teams'][TEAM]['skill_replacement_evidence'][0]
        self.assertEqual(report['history']['opportunities'],30)
        self.assertEqual(report['top_measured_same_position_peers'][0]['name'],'Active Receiver')
        self.assertIsNone(report['point_adjustment'])
    def test_no_fictional_games_in_qb_history(self):
        self.make_injuries()
        frame=pd.concat([self.pbp,pd.DataFrame([{'game_id':'2026_06_WAS_SF','posteam':TEAM,'qb_dropback':1,'passer_player_id':'QB2','passer_player_name':'Secondary QB','epa':-100,'play_type':'pass'}])],ignore_index=True)
        self.pbp=frame
        q=self.eval()['teams'][TEAM]['conditional_qb_replacement']
        self.assertEqual(q['candidate_measured_dropbacks'],30)
    def test_noncausal_defensive_profile_has_no_point_shift(self):
        import gzip
        data=pd.DataFrame([dict(game_id=gid,team=TEAM,pfr_player_id=pid,def_pressures=n)
            for gid in GAMES for pid,n in [('pD1',2),('pD2',3)]])
        folder=self.root/'extra';folder.mkdir()
        file=folder/'def.csv.gz';data.to_csv(file,index=False,compression='gzip')
        meta={'pfr':{'def':{'status':'COLLECTED','capture_time_utc':'2026-10-19T01:08:00Z',
             'path':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}}}
        (folder/'metrics_manifest.json').write_text(json.dumps(meta))
        self.make_injuries(id='D1',name='Injured Edge',position='DL')
        report=self.eval(metrics_dir=folder)['teams'][TEAM]['pass_rush_replacement_evidence'][0]
        self.assertEqual(report['observed_pressure_profile']['pressure_events'],6)
        self.assertEqual(report['potential_peer_pressure_profiles'][0]['pressure_events'],9)
        self.assertIsNone(report['point_adjustment'])
    def test_defensive_pressure_after_cutoff_rejected(self):
        folder=self.root/'extra';folder.mkdir()
        f=folder/'def.csv.gz';pd.DataFrame([dict(game_id=list(GAMES)[0],team=TEAM,pfr_player_id='pD1',def_pressures=3)]).to_csv(f,index=False,compression='gzip')
        (folder/'metrics_manifest.json').write_text(json.dumps({'pfr':{'def':{'status':'COLLECTED','capture_time_utc':'2026-10-20T01:08:00Z','path':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}}}))
        self.make_injuries(id='D1',name='Injured Edge',position='DL')
        with self.assertRaisesRegex(InputError,'after cutoff'):self.eval(metrics_dir=folder)
    def test_scenario_changes_points_only_conditionally(self):
        self.make_injuries();m=self.eval()
        game={'home':{'name':TEAM,'personnel':{'pass_epa_shift':0.0,'notes':'neutral','source':'neutral'}},
              'away':{'name':'WAS','personnel':{'pass_epa_shift':0.0,'notes':'neutral','source':'neutral'}},
              'neutral_site':False}
        original=copy.deepcopy(game)
        def fake_projection(atk,defend,rules,side,hfa):return {'_ppd':2.0+10*atk['personnel']['pass_epa_shift'],'_td':.5,'_to':.1,'expected_drives':10}
        def fake_sim(h,a,r,spread,total,runs,seed):return {'mean_score_home':10*h['_ppd'],'mean_score_away':10*a['_ppd'],'mean_margin_home':10*(h['_ppd']-a['_ppd'])}
        with patch('measure_player_availability.projection',side_effect=fake_projection),patch('measure_player_availability.simulate',side_effect=fake_sim):
            result=scenario_points(game,MODEL_RULES['NFL'],m,1000,123)
        self.assertEqual(game,original)
        self.assertEqual(len(result),1)
        self.assertLess(result[0]['delta_home_margin_vs_neutral'],0)
        self.assertTrue(result[0]['not_base_forecast'])

if __name__=='__main__':unittest.main()
