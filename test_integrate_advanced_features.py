import copy,csv,json,tempfile,unittest
from pathlib import Path
from model import InputError,run,validate_game
from integrate_advanced_features import valid_pressure_rates,build_overlay,attach_overlay,verified_snapshot

EXAMPLE=json.loads((Path(__file__).resolve().parents[1]/'examples'/'synthetic_matchup.json').read_text())

class AdvancedFeatureTests(unittest.TestCase):
    def advanced(self,penalties,made,attempts):
        return {'penalty_yards_per_game':penalties,'penalty_games':5,
            'field_goals_made':made,'field_goals_attempted':attempts,'punt_gross_yards':49.5,
            'source':'VERIFIED_SYNTHETIC_FIXTURE','as_of_utc':'2026-10-18T18:40:00Z',
            'status':'DESCRIPTIVE_UNCALIBRATED'}

    def base(self):
        g=copy.deepcopy(EXAMPLE)
        for k in ('home','away'):
            g[k]['context']['penalty_epa_per_game']=None
            g[k]['context']['special_teams_epa_per_game']=None
        return g

    def test_missing_epa_does_not_become_fabricated_zero(self):
        g=self.base();validate_game(g)
        self.assertIsNone(g['home']['context']['penalty_epa_per_game'])
        x=run(g,1000,seed=6,perform_stress=False)
        self.assertTrue(x['audit']['optional_causal_penalty_or_specialteams_epa_omitted'])
        self.assertEqual(x['projection_home']['net_penalties_ppg_component'],0)

    def test_penalty_direction_monotonic_in_points_per_drive(self):
        g=self.base();g['home']['advanced']=self.advanced(50,8,10)
        g['away']['advanced']=self.advanced(90,8,10)
        a=run(g,1000,seed=6,perform_stress=False)
        g['home']['advanced']['penalty_yards_per_game']=100
        b=run(g,1000,seed=6,perform_stress=False)
        self.assertGreater(a['projection_home']['advanced_penalty_yards_proxy_ppg'], b['projection_home']['advanced_penalty_yards_proxy_ppg'])
        self.assertGreater(a['projection_home']['raw_points_per_drive'],b['projection_home']['raw_points_per_drive'])
        self.assertFalse(a['audit']['supplemental_proxy_coefficients_empirically_calibrated'])

    def test_fg_accuracy_shrunk_and_capped(self):
        g=self.base();g['home']['advanced']=self.advanced(70,10,10)
        a=run(g,1000,seed=6,perform_stress=False)
        g['home']['advanced']['field_goals_made']=5
        b=run(g,1000,seed=6,perform_stress=False)
        self.assertGreater(a['projection_home']['advanced_fg_accuracy_proxy_ppg'],b['projection_home']['advanced_fg_accuracy_proxy_ppg'])
        self.assertLessEqual(abs(a['projection_home']['advanced_fg_accuracy_proxy_ppg']),.2)

    def test_reject_late_advanced_stats(self):
        g=self.base();g['home']['advanced']=self.advanced(70,9,10)
        g['home']['advanced']['as_of_utc']='2026-10-18T20:30:00Z'
        with self.assertRaisesRegex(InputError,'later than'):validate_game(g)

    def test_mismatched_pressure_definitions_fail(self):
        g=self.base();g['home']['pressure_source']='PFR_QB_TIMES_PRESSURED_OVER_SAME_GAME_PBP_DROPBACKS'
        g['pressure_reference_rate']=.25
        with self.assertRaisesRegex(InputError,'same pressure definition'):run(g,1000,perform_stress=False)

    def test_pressure_uses_same_game_dropback_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            games={f'G{i}':{'home':'H','away':'A'} for i in range(1,4)}
            with (p/'pbp.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['game_id','posteam','play_type','qb_dropback','pass','qb_kneel','qb_spike'])
                w.writeheader()
                for g in games:
                    for team in ('H','A'):
                        for _ in range(10):w.writerow(dict(game_id=g,posteam=team,play_type='pass',qb_dropback=1,**{'pass':1},qb_kneel=0,qb_spike=0))
                        # A nullified pass must not inflate denominator.
                        w.writerow(dict(game_id=g,posteam=team,play_type='no_play',qb_dropback=1,**{'pass':1},qb_kneel=0,qb_spike=0))
            with (p/'pfr.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['game_id','team','opponent','times_pressured','pfr_player_id'])
                w.writeheader()
                for g in games:
                    w.writerow(dict(game_id=g,team='H',opponent='A',times_pressured=2,pfr_player_id='qbH'))
                    w.writerow(dict(game_id=g,team='A',opponent='H',times_pressured=4,pfr_player_id='qbA'))
            out=valid_pressure_rates(p/'pbp.csv',p/'pfr.csv',games)
            self.assertAlmostEqual(out['H']['offense']['rate'],.2)
            self.assertAlmostEqual(out['H']['defense']['rate'],.4)
            self.assertEqual(out['H']['offense']['dropbacks'],30)
            self.assertEqual(out['A']['defense']['pressures'],6)

    def test_legitimate_extreme_qb_rush_epa_shrunk_not_rejected(self):
        g=self.base();g['home']['offense']['qb_rush_epa']=-1.598
        validate_game(g)
        res=run(g,1000,seed=6,perform_stress=False)
        self.assertTrue(-.065<=res['projection_home']['qb_rushing_epa_adjustment']<=.065)

    def test_legacy_causal_context_prevents_double_count(self):
        g=copy.deepcopy(EXAMPLE)
        g['home']['advanced']=self.advanced(50,10,10)
        g['away']['advanced']=self.advanced(100,8,10)
        r=run(g,1000,seed=6,perform_stress=False)
        self.assertEqual(r['projection_home']['advanced_penalty_yards_proxy_ppg'],0)
        self.assertEqual(r['projection_home']['advanced_fg_accuracy_proxy_ppg'],0)

    def test_missing_pfr_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);games={f'G{i}':{'home':'H','away':'A'} for i in range(1,4)}
            (p/'pbp.csv').write_text('game_id,posteam,play_type,qb_dropback,pass,qb_kneel,qb_spike\nG1,H,pass,1,1,0,0\n')
            (p/'pfr.csv').write_text('game_id,team,opponent,times_pressured,pfr_player_id\n')
            out=valid_pressure_rates(p/'pbp.csv',p/'pfr.csv',games)
            self.assertEqual(out['H']['offense']['status'],'INSUFFICIENT_COVERAGE')
            self.assertNotIn('rate',out['H']['offense'])

if __name__=='__main__':unittest.main()
