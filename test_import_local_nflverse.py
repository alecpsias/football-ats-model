import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from import_local_nflverse import import_files
from nflverse_connector import verify, CollectionError

class TestLocalImport(unittest.TestCase):
    def test_uses_real_local_csv_and_hash_verification(self):
        with TemporaryDirectory() as td:
            t=Path(td)
            pbp=pd.DataFrame([{'season':2026,'game_id':'2026_01_A_B','play_id':1,'posteam':'A','defteam':'B','epa':.1,'qb_dropback':1,'pass':1,'rush':0,'success':1}])
            schedule=pd.DataFrame([{'season':2026,'game_id':'2026_01_A_B','home_team':'B','away_team':'A','gameday':'2026-09-10','gametime':'20:20'}])
            roster=pd.DataFrame([{'season':2026,'gsis_id':'00-TESTQB','position':'QB','week':1}])
            for n,fr in [('pbp',pbp),('schedule',schedule),('roster',roster)]:fr.to_csv(t/f'{n}.csv',index=False)
            at=datetime(2026,10,18,21,15,tzinfo=timezone.utc)
            out=t/'snapshot'
            m=import_files(2026,out,{k:t/f'{k}.csv' for k in ('pbp','schedule','roster')},now=lambda:at)
            self.assertEqual(m['status'],'COLLECTED')
            self.assertEqual(verify(out)['status'],'VERIFIED')
            self.assertEqual(verify(out,'2026-10-18T13:20:00Z')['pregame_eligibility'],'NOT_VERIFIED_PREGAME')
            self.assertEqual(verify(out,'2026-10-18T23:00:00Z')['pregame_eligibility'],'CAPTURED_BEFORE_CUTOFF')
            self.assertEqual(len(pd.read_csv(out/'pbp_2026.csv.gz')),1)
    def test_blocks_missing_roster_before_snapshot_creation(self):
        with TemporaryDirectory() as td:
            t=Path(td)
            with self.assertRaises(CollectionError):
                import_files(2026,t/'snapshot',{'pbp':t/'absent.csv','schedule':t/'absent2.csv','roster':t/'absent3.csv'})
            self.assertFalse((t/'snapshot').exists())
    def test_blocks_duplicate_plays(self):
        with TemporaryDirectory() as td:
            t=Path(td)
            pbp=pd.DataFrame([{'season':2026,'game_id':'g','play_id':1,'posteam':'A','defteam':'B','epa':.1,'qb_dropback':1,'pass':1,'rush':0,'success':1}]*2)
            pbp.to_csv(t/'pbp.csv',index=False)
            pd.DataFrame([{'season':2026,'game_id':'g','home_team':'B','away_team':'A','gameday':'2026-09-10','gametime':'20:20'}]).to_csv(t/'schedule.csv',index=False)
            pd.DataFrame([{'season':2026,'gsis_id':'QB','position':'QB'}]).to_csv(t/'roster.csv',index=False)
            with self.assertRaises(CollectionError):
                import_files(2026,t/'snapshot',{k:t/f'{k}.csv' for k in ('pbp','schedule','roster')})
            self.assertFalse((t/'snapshot').exists())
if __name__=='__main__':unittest.main()
