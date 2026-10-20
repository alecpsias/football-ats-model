"""Deterministic schema-contract and no-leakage tests for the CFB connector."""
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from cfb.pipeline import (
    CFBError, iso_utc, source_urls, verify, split_games, normalize_pbp_teams,
    team_features, missing_features, play_stats, require, capture, report, REQUIRED_PBP,
    validate_pbp_columns, standardize_pbp_columns,
)

CUTOFF = '2026-10-19T12:00:00Z'


def schedule():
    row = lambda gid, when, completed, home, away, home_id, away_id, hd='fbs', ad='fbs': dict(
        game_id=gid, start_date=when, completed=completed, home_id=home_id,
        away_id=away_id, home_team=home, away_team=away,
        home_division=hd, away_division=ad, fbs_participant=(hd=='fbs' or ad=='fbs'),
        neutral_site=False, start_time_tbd=False)
    return pd.DataFrame([
        row(100,'2026-10-03T20:00:00Z',True,'SCHOOL_A','SCHOOL_B',1,2),
        row(101,'2026-10-10T20:00:00Z',True,'SCHOOL_A','SCHOOL_C',1,3),
        row(102,'2026-10-10T20:00:00Z',True,'SCHOOL_B','SCHOOL_D',2,4),
        row(103,'2026-10-24T20:00:00Z',False,'SCHOOL_A','SCHOOL_B',1,2),
        row(104,'2026-10-04T20:00:00Z',True,'FCS_X','FCS_Y',11,12,hd='fcs',ad='fcs'),
        row(105,'2026-10-17T20:00:00Z',False,'SCHOOL_A','SCHOOL_B',1,2),
        row(106,'2026-10-25T20:00:00Z',False,'SCHOOL_C','FCS_X',3,11,hd='fbs',ad='fcs'),
    ])


def plays():
    out=[]
    for game,home,away in [(100,'A espn','B espn'),(101,'A espn','C espn'),(102,'B espn','D espn'),
                           (103,'A espn','B espn'),(104,'FCS X','FCS Y'),(105,'A espn','B espn')]:
        for side in (0,1):
            team,other=(home,away) if side==0 else (away,home)
            for i in range(8):
                passing=(i%2==0)
                out.append(dict(game_id=game,game_play_number=side*8+i,
                    home=home,away=away,pos_team=team,def_pos_team=other,drive_id=i//4+side*2,
                    **{'pass':passing,'rush':not passing},EPA=(0.2 if side==0 else -0.1),
                    down=(i%4)+1,distance=10,yards_gained=12,yards_to_goal=75-i*10,
                    sack=False,**{'int':False},firstD_by_yards=True,firstD_by_penalty=False,
                    rush_td=False,pass_td=(i==7),offense_score_play=(i==7),td_play=(i==7),
                    turnover=False))
    return pd.DataFrame(out)


class TestCFBPipeline(unittest.TestCase):
    def setUp(self):
        self.schedule=schedule()
        self.pbp=plays()

    def test_2026_urls_verified_format(self):
        u=source_urls(2026)
        self.assertTrue(u['pbp'].endswith('play_by_play_2026.parquet'))
        self.assertTrue(u['schedule'].endswith('cfb_schedules_2026.parquet'))
        self.assertTrue(u['roster'].endswith('cfb_rosters_2026.parquet'))

    def test_requires_offset(self):
        with self.assertRaises(CFBError):iso_utc('2026-10-19T12:00:00')

    def test_no_future_and_no_incomplete(self):
        hist,new,old=split_games(self.schedule,self.pbp,CUTOFF)
        self.assertEqual(set(hist.game_id),{100,101,102})
        self.assertEqual(set(new.game_id),{103,106})
        self.assertEqual(set(old.game_id),{100,101,102})
        self.assertEqual(len(old),48)

    def test_fcs_vs_fcs_rejected_fbs_vs_fcs_kept(self):
        hist,new,_=split_games(self.schedule,self.pbp,CUTOFF)
        self.assertNotIn(104,set(hist.game_id))
        self.assertIn(106,set(new.game_id))

    def test_actual_release_schema_contract(self):
        require(self.pbp,REQUIRED_PBP,'fixture')
        self.assertIn('drive_id',self.pbp.columns)
        self.assertNotIn('drive.id',self.pbp.columns)

    def test_espn_id_resolution_without_school_name_alias(self):
        hist,_,old=split_games(self.schedule,self.pbp,CUTOFF)
        mapped=normalize_pbp_teams(old,hist)
        self.assertEqual(set(mapped[mapped.game_id==100].pos_team_id),{1,2})
        self.assertTrue((mapped.pos_team_id!=mapped.def_pos_team_id).all())

    def test_no_fuzzy_guessing_for_ambiguous_label(self):
        hist,_,old=split_games(self.schedule,self.pbp,CUTOFF)
        old.loc[old.index[0],'pos_team']='not a valid provider team label'
        with self.assertRaises(CFBError):normalize_pbp_teams(old,hist)

    def test_missing_actual_provider_column_fails(self):
        with self.assertRaises(CFBError):split_games(self.schedule,self.pbp.drop(columns='yards_gained'),CUTOFF)

    def test_real_2026_release_aliases_and_numeric_ids(self):
        # Contract reconstructed from the ACTUAL 2026 schema log, not guesswork.
        p=self.pbp.rename(columns={
            'drive_id':'drive.id',
            'yards_gained':'statYardage',
            'yards_to_goal':'start.yardsToEndzone',
        }).drop(columns=['home','away','turnover'])
        # IDs are ESPN game-side IDs, unlike potentially abbreviated team names.
        ids={100:(1,2),101:(1,3),102:(2,4),103:(1,2),104:(11,12),105:(1,2)}
        p['pos_team_id']=p.apply(lambda r: ids[r.game_id][0] if 'A espn' in r.pos_team and r.game_id!=102
          or r.game_id==102 and r.pos_team=='B espn'
          or r.game_id==104 and r.pos_team=='FCS X'
          or r.game_id in (100,101,103,105) and r.pos_team=='A espn'
          else ids[r.game_id][1],axis=1)
        # For the purpose of the numeric ID test, names may disagree with schedule.
        p['def_pos_team_id']=p.apply(lambda r: ids[r.game_id][0]+ids[r.game_id][1]-r.pos_team_id,axis=1)
        p['pos_team']='Provider Name That Does Not Match Schedule'
        p['def_pos_team']='Other Provider Alias'
        validate_pbp_columns(p.columns)
        hist,_,old=split_games(self.schedule,p,CUTOFF)
        mapped=normalize_pbp_teams(old,hist)
        self.assertIn('drive_id',mapped)
        self.assertIn('yards_gained',mapped)
        self.assertIn('yards_to_goal',mapped)
        self.assertEqual(set(mapped[mapped.game_id==100].pos_team_id),{1,2})
        self.assertGreater(team_features(1,mapped)['offense']['plays'],0)

    def test_numeric_ids_must_agree_with_schedule(self):
        p=self.pbp.rename(columns={
            'drive_id':'drive.id','yards_gained':'statYardage',
            'yards_to_goal':'start.yardsToEndzone',
        }).drop(columns=['home','away','turnover'])
        # Valid-looking IDs for wrong schools are rejected instead of fuzzy matched.
        p['pos_team_id']=9999
        p['def_pos_team_id']=8888
        hist,_,old=split_games(self.schedule,p,CUTOFF)
        with self.assertRaises(CFBError):normalize_pbp_teams(old,hist)

    def test_null_missing_pressure_qb_rushing_and_fumble_rate(self):
        h,_,old=split_games(self.schedule,self.pbp,CUTOFF)
        mapped=normalize_pbp_teams(old,h)
        t=team_features(1,mapped)
        self.assertIsNone(t['offense']['pressure_allowed'])
        self.assertIsNone(t['defense']['pressure_generated'])
        self.assertIsNone(t['offense']['qb_rush_epa'])
        self.assertIsNone(t['offense']['fumble_lost_rate'])
        self.assertGreater(t['offense']['plays'],0)
        self.assertIn('offense.pressure_allowed',missing_features(t))

    def test_schedule_strength_excludes_own_matchup(self):
        h,_,old=split_games(self.schedule,self.pbp,CUTOFF)
        t=team_features(1,normalize_pbp_teams(old,h))
        # B played D; C has no other opponent, so aggregate cannot be verified.
        self.assertIsNone(t['schedule']['opp_def_pass_epa_allowed_avg'])

    def test_invalid_snapshot_sha_blocks(self):
        with TemporaryDirectory() as td:
            folder=Path(td)
            m=dict(schema='cfb_source_snapshot_v2',captured_at_utc=CUTOFF,datasets={})
            for k in ('pbp','schedule','roster'):
                (folder/(k+'.parquet')).write_bytes(b'fake contents')
                m['datasets'][k]=dict(file=k+'.parquet',sha256='not-match',size_bytes=13,
                                      captured_at_utc=CUTOFF)
            (folder/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(CFBError):verify(folder)

    def test_snapshot_captured_after_manifest_blocks(self):
        with TemporaryDirectory() as td:
            folder=Path(td)
            m=dict(schema='cfb_source_snapshot_v2',captured_at_utc=CUTOFF,datasets={})
            for k in ('pbp','schedule','roster'):
                payload=b'fourth fake parquet!'
                (folder/(k+'.parquet')).write_bytes(payload)
                m['datasets'][k]=dict(file=k+'.parquet',sha256=hashlib.sha256(payload).hexdigest(),
                                      size_bytes=len(payload),captured_at_utc='2026-10-19T13:00:00Z')
            (folder/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(CFBError):verify(folder)

    def test_empty_directory_required_for_immutable_capture(self):
        with TemporaryDirectory() as td:
            (Path(td)/'existing.txt').write_text('no overwrite')
            with self.assertRaises(CFBError):capture(2026,td)

    def test_no_invented_first_half_or_ats_market(self):
        h,_,old=split_games(self.schedule,self.pbp,CUTOFF)
        t=team_features(1,normalize_pbp_teams(old,h))
        self.assertNotIn('spread_home',t)
        self.assertNotIn('cover_probability',t)

    def test_end_to_end_evidence_report_fixture_without_parquet_download(self):
        """A mocked reader exercises manifest -> mappings -> game JSON, not live data."""
        with TemporaryDirectory() as td:
            root=Path(td);src=root/'snap';dest=root/'report';src.mkdir()
            m={'schema':'cfb_source_snapshot_v2','captured_at_utc':CUTOFF,'datasets':{}}
            for kind in ('pbp','schedule','roster'):
                payload=(kind+' data, synthetic only').encode()
                (src/(kind+'.parquet')).write_bytes(payload)
                m['datasets'][kind]={'file':kind+'.parquet','sha256':hashlib.sha256(payload).hexdigest(),
                    'size_bytes':len(payload),'captured_at_utc':CUTOFF}
            (src/'manifest.json').write_text(json.dumps(m))
            def read_parquet(path):
                return self.schedule if Path(path).name=='schedule.parquet' else self.pbp
            with patch('cfb.pipeline.pd.read_parquet',side_effect=read_parquet):
                index=report(src,dest,window_days=7)
            self.assertEqual(index['completed_games_used'],3)
            self.assertEqual({x['game_id'] for x in index['upcoming_games']},{'103','106'})
            game=json.loads((dest/'103.json').read_text())
            self.assertEqual(game['status'],'INCOMPLETE_CORE_INPUTS_NO_ATS_PICK')
            self.assertEqual(game['home']['id'],'1')
            self.assertIsNone(game['home']['features']['offense']['pressure_allowed'])
            self.assertEqual(game['market_status'],'NO_TIMESTAMPED_VERIFIED_QUOTE')

    def test_unmapped_special_teams_play_is_ignored_not_assigned(self):
        hist,_,old=split_games(self.schedule,self.pbp,CUTOFF)
        invalid=old.iloc[0].copy()
        invalid['pos_team']='NONE';invalid['pass']=False;invalid['rush']=False
        result=normalize_pbp_teams(pd.concat([old,pd.DataFrame([invalid])],ignore_index=True),hist)
        self.assertEqual(len(result),len(old))

    def test_strict_model_when_parent_available(self):
        try:
            from cfb.strict_forecast import forecast
            from model import InputError
        except ImportError:
            self.skipTest('parent NFL/CFB model.py not installed in standalone test')
        with self.assertRaises(InputError):forecast({'league':'CFB'})

if __name__ == '__main__': unittest.main()
