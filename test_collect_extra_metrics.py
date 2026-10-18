"""Offline schema, attribution, source-time and missing-data tests for extra metrics."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from collect_extra_metrics import collect, games_before_cutoff, pfr_team_pressure, parse_stamp


class ExtraMetricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.snap=self.root/'base'
        self.snap.mkdir()
        self.schedule=[
            {'game_id':'2026_06_GB_DAL','gameday':'2026-10-17','gametime':'13:00','home_team':'DAL','away_team':'GB'},
            {'game_id':'2026_07_GB_DAL','gameday':'2026-10-18','gametime':'20:20','home_team':'GB','away_team':'DAL'},
        ]
        self.pbp=[
            {'game_id':'2026_06_GB_DAL','posteam':'DAL','defteam':'GB','play_type':'no_play',
             'penalty':'1','penalty_team':'DAL','penalty_yards':'10','epa':'-0.8'},
            {'game_id':'2026_06_GB_DAL','posteam':'DAL','defteam':'GB','play_type':'field_goal',
             'field_goal_attempt':'1','field_goal_result':'made','epa':'0.6'},
            {'game_id':'2026_06_GB_DAL','posteam':'GB','defteam':'DAL','play_type':'punt',
             'punt_attempt':'1','punt_inside_twenty':'1','punt_blocked':'0',
             'return_yards':'12','kick_distance':'48','epa':'-0.5'},
            {'game_id':'2026_06_GB_DAL','posteam':'DAL','defteam':'GB','play_type':'kickoff',
             'kickoff_attempt':'1','touchback':'0','return_yards':'21','epa':'0.2'},
            {'game_id':'2026_07_GB_DAL','posteam':'DAL','defteam':'GB','play_type':'no_play',
             'penalty':'1','penalty_team':'DAL','penalty_yards':'99','epa':'-1'},
        ]
        self.pfr_pass=[
            {'game_id':'2026_06_GB_DAL','team':'DAL','opponent':'GB','times_pressured':'3','week':'6'},
            {'game_id':'2026_06_GB_DAL','team':'GB','opponent':'DAL','times_pressured':'6','week':'6'},
            {'game_id':'2026_07_GB_DAL','team':'DAL','opponent':'GB','times_pressured':'999','week':'7'},
        ]
        self.pfr_def=[{'game_id':'2026_06_GB_DAL','team':'GB','opponent':'DAL','def_times_hurried':'3','week':'6'}]
        self.paths={}
        for kind,records in [('schedule',self.schedule),('pbp',self.pbp)]:
            p=self.snap/f'{kind}_2026.csv.gz'
            # The fixture is plain CSV under .gz extension only when compressed.
            import gzip
            keys=sorted(set().union(*(set(x) for x in records)))
            with gzip.open(p,'wt',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
            self.paths[kind]=p
        manifest={'season':2026,'status':'COLLECTED','created_at_utc':'2026-10-18T23:00:00Z',
                  'datasets':{k:{'local_file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in self.paths.items()}}
        (self.snap/'manifest.json').write_text(json.dumps(manifest))
        self.passfile=self.root/'pass.csv'; self.deffile=self.root/'def.csv'
        for p,rows in [(self.passfile,self.pfr_pass),(self.deffile,self.pfr_def)]:
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    def test_completed_only_and_correct_team_attribution(self):
        out=self.root/'out'
        result=collect(self.snap,out,pfr_files={'pass':self.passfile,'def':self.deffile})
        self.assertEqual(result['eligible_schedule_games'],1)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertIn('sha256',result['pfr']['pass'])
        with (out/'team_extra_metrics.csv').open() as f:
            rows={r['team']:r for r in csv.DictReader(f)}
        self.assertEqual(rows['DAL']['penalty_yards_observed'],'10.0')
        self.assertEqual(rows['DAL']['field_goals_made'],'1')
        self.assertEqual(rows['GB']['punts'],'1')
        self.assertEqual(rows['GB']['punt_distance_mean'],'48.0')
        self.assertEqual(rows['GB']['kickoff_return_yards_allowed'],'21.0')
        self.assertEqual(rows['DAL']['pfr_qb_pressures_allowed'],'3')
        self.assertEqual(rows['GB']['pfr_opponent_qb_pressures_generated'],'3')
        self.assertEqual(rows['DAL']['pfr_opponent_qb_pressures_generated'],'6')
        self.assertNotIn('999',str(rows))
        self.assertIn('not',result['asof_policy'].lower())

    def test_missing_pfr_means_missing_pressure_not_zero(self):
        bad=self.root/'baddef.csv';bad.write_text('game_id,team\n')
        result=collect(self.snap,self.root/'partial',pfr_files={'pass':self.passfile,'def':bad})
        self.assertEqual(result['status'],'PARTIAL_PFR')
        self.assertEqual(result['pfr']['def']['status'],'UNAVAILABLE')

    def test_missing_pressure_column_blocks_pressure_dataset(self):
        bad=self.root/'badpass.csv';bad.write_text('game_id,team,opponent\n2026_06_GB_DAL,DAL,GB\n')
        result=collect(self.snap,self.root/'partial',pfr_files={'pass':bad,'def':self.deffile})
        self.assertEqual(result['pfr']['pass']['status'],'UNAVAILABLE')
        self.assertEqual(result['status'],'PARTIAL_PFR')
        with (self.root/'partial'/'team_extra_metrics.csv').open() as f:
            rows=list(csv.DictReader(f))
        self.assertTrue(all(r['pfr_qb_pressures_allowed']=='' for r in rows))

    def test_corrupt_base_file_fails_before_computation(self):
        with self.paths['pbp'].open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            collect(self.snap,self.root/'out',pfr_files={'pass':self.passfile,'def':self.deffile})

    def test_time_zone_window(self):
        games=games_before_cutoff(self.paths['schedule'],parse_stamp('2026-10-18T23:00:00Z'))
        self.assertEqual(list(games),['2026_06_GB_DAL'])

    def test_pressure_counts_are_opponent_attributed(self):
        games={'A':{'home':'DAL','away':'GB'}}
        stats,cov=pfr_team_pressure([{'game_id':'A','team':'DAL','opponent':'GB','times_pressured':2}],games)
        self.assertEqual(stats['GB']['pfr_opponent_qb_pressures_generated'],2)
        self.assertEqual(stats['DAL']['pfr_qb_pressures_allowed'],2)


if __name__=='__main__': unittest.main()
