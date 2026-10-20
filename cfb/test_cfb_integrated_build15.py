"""Build 15 test suite: scoring independence, provenance and safe integration."""
import copy
import math
import unittest
from pathlib import Path

from cfb.integrated_scoring import project_integrated, simulate_integrated, VERSION
from cfb.research_backtest import project


def record(opponent, *, off_pass=.15, off_rush=.07, def_pass=.03, def_rush=.02, points=27.,
           sacks=4, third=9, redzone=3, turnovers=1):
    return {'opponent_id':opponent,'points':points,'conceded':23.,'drives':11,
            'faced':11,'off_pass_sum':off_pass*35,'off_pass_n':35,
            'off_rush_sum':off_rush*25,'off_rush_n':25,
            'def_pass_sum':def_pass*33,'def_pass_n':33,
            'def_rush_sum':def_rush*25,'def_rush_n':25,
            'third_conversions_sum':third,'third_conversions_n':18,
            'def_third_conversions_sum':8,'def_third_conversions_n':18,
            'redzone_tds_sum':redzone,'redzone_tds_n':4,
            'def_redzone_tds_sum':2,'def_redzone_tds_n':4,
            'int_sum':turnovers,'int_n':35,'def_int_sum':1,'def_int_n':33,
            'fumble_sum':0,'fumble_n':60,'def_fumble_sum':0,'def_fumble_n':58,
            'sack_sum':sacks,'sack_n':35,'def_sack_sum':2,'def_sack_n':33,
            'interceptions':turnovers,'fumbles_lost':0,
            'qb_rushes':3,'qb_rush_epa':.1,
            'start_field_position':28.}


def histories():
    # Every school's opponents have third-party observations: SOS can be
    # calculated without using target-game outcomes or a school's own games.
    return {1:[record(3),record(4,points=30),record(3)],
            2:[record(4,off_pass=-.16,off_rush=-.10,points=15),
               record(3,off_pass=-.16,off_rush=-.10,points=15),
               record(4,off_pass=-.16,off_rush=-.10,points=15)],
            3:[record(4,def_pass=.12,def_rush=.03),record(4)],
            4:[record(3,def_pass=.10,def_rush=.06),record(3)]}


class IntegratedBuild15(unittest.TestCase):
    def test_never_import_nfl(self):
        base=Path(__file__).parent
        for name in ('integrated_scoring.py','fresh_matchup.py','prospective.py',
                     'research_backtest.py','forward_monitor.py'):
            src=(base/name).read_text()
            self.assertNotIn('from model import',src)
            self.assertNotIn('import nflverse_connector',src)

    def test_version_and_not_primary(self):
        h=histories(); p=project_integrated(1,2,h,False,'home')
        self.assertEqual(p['version'],VERSION)
        self.assertFalse(p['primary_forecast_modified'])
        self.assertFalse(p['coefficients_empirically_fitted'])
        self.assertIn('matched_passing',p['components'])
        self.assertIn('matched_rushing',p['components'])
        self.assertIn('baseline_ppd_anchor',p['components'])
        self.assertIn('epa_ppd_level',p['components'])

    def test_two_team_scores_change_from_actual_epa(self):
        h=histories(); stronger=project_integrated(1,2,h,False,'home')
        weaker=copy.deepcopy(h)
        for r in weaker[1]: r['off_pass_sum']-=35*.4
        v=project_integrated(1,2,weaker,False,'home')
        self.assertGreater(stronger['projection']['_ppd'],v['projection']['_ppd'])

    def test_defensive_quality_affects_expected_ppd(self):
        h=histories(); a=project_integrated(1,2,h,False,'home')
        weaker=copy.deepcopy(h)
        for r in weaker[2]:r['def_pass_sum']+=33*.3
        b=project_integrated(1,2,weaker,False,'home')
        self.assertGreater(b['projection']['_ppd'],a['projection']['_ppd'])

    def test_no_mutation_and_reproducible_same_seed(self):
        h=histories();before=copy.deepcopy(h)
        a=simulate_integrated(1,2,h,False,runs=200,seed=51)
        b=simulate_integrated(1,2,h,False,runs=200,seed=51)
        self.assertEqual(a,b)
        self.assertEqual(h,before)
        self.assertTrue(a['not_primary'])
        self.assertTrue(math.isfinite(a['mean_margin_home']))

    def test_game_script_plus_integrated_independent(self):
        h=histories()
        a=simulate_integrated(1,2,h,False,runs=300,seed=21)
        b=simulate_integrated(1,2,h,False,runs=300,seed=21,use_script=True)
        self.assertEqual(b['simulation_version'],'CFB_GAME_SCRIPT')
        self.assertEqual(a['simulation_version'],'CFB_BASELINE_ENGINE')
        self.assertIsNotNone(b['simulation'])

    def test_missing_opponent_quality_never_fabricates(self):
        h={1:[record(2)]*3,2:[record(1)]*3}
        a=project_integrated(1,2,h,False,'home')
        self.assertEqual(a['status'],'UNAVAILABLE_MISSING_LEAVE_OPPONENT_OUT_EPA')
        self.assertIsNone(a['projection'])
        self.assertIsNone(simulate_integrated(1,2,h,False,runs=110)['simulation'])

    def test_missing_optional_measurement_is_disclosed(self):
        h=histories()
        for group in h.values():
            for r in group:
                for k in ('sack_sum','sack_n','def_sack_sum','def_sack_n',
                          'third_conversions_sum','third_conversions_n',
                          'def_third_conversions_sum','def_third_conversions_n'):
                    r[k]=None
        x=project_integrated(1,2,h,False,'home')
        self.assertIsNotNone(x['projection'])
        self.assertIn('sack_matchup_not_full_pressure',x['missing_features'])
        self.assertIn('third_down_matched',x['missing_features'])
        self.assertIn('full_pressure_rate_not_source_backed',x['missing_features'])
        self.assertNotIn('sack_residual_ppd',x['components'])

    def test_turnover_matchup_changes_turnover_probability_not_twice_ppd(self):
        h=histories()
        x=project_integrated(1,2,h,False,'home')
        g=copy.deepcopy(h)
        for r in g[1]:r['int_sum']=3
        y=project_integrated(1,2,g,False,'home')
        self.assertGreater(y['projection']['_to'],x['projection']['_to'])
        self.assertEqual(y['projection']['_ppd'],x['projection']['_ppd'])
        self.assertEqual(x['components']['turnover_source'],'OFFENSE_VERSUS_OPPONENT_DEFENSE')

    def test_bad_optional_counts_fail_closed(self):
        h=histories();h[1][0]['int_sum']=36
        with self.assertRaises(ValueError):project_integrated(1,2,h,False,'home')

    def test_home_field_and_neutral_affect_projection(self):
        h=histories()
        home=project_integrated(1,2,h,False,'home')['projection']['_ppd']
        neutral=project_integrated(1,2,h,True,'home')['projection']['_ppd']
        self.assertGreater(home,neutral)

    def test_primary_existing_projection_does_not_change(self):
        h=histories()
        before=project(h[1],h[2],False,'home',False)
        project_integrated(1,2,h,False,'home')
        after=project(h[1],h[2],False,'home',False)
        self.assertEqual(before,after)

    def test_codepaths_present(self):
        base=Path(__file__).parent
        for n in ('fresh_matchup.py','prospective.py','research_backtest.py','forward_monitor.py'):
            s=(base/n).read_text()
            self.assertIn('integrated_scoring',s)
        self.assertIn('integrated_scoring_plus_script_research',
            (base/'fresh_matchup.py').read_text())

if __name__=='__main__':unittest.main()

class IntegratedForwardAudit(unittest.TestCase):
    def test_learning_reads_only_original_pre_kickoff_integrated_records(self):
        import json
        import tempfile
        from cfb.learn_forward import observations,investigate
        with tempfile.TemporaryDirectory() as tmp:
            row={'game_id':'401000002','kickoff_utc':'2026-10-25T19:00:00+00:00',
                'forecast_generated_at_utc':'2026-10-25T12:00:00+00:00',
                'predicted_home_margin':3.0,'actual_home_margin':-5.0,
                'integrated_scoring_challenger':{
                    'status':'ORIGINAL_FROZEN_INTEGRATED_RESEARCH_CANDIDATE',
                    'projected_home_margin':1.0},
                'integrated_script_challenger':{
                    'status':'ORIGINAL_FROZEN_INTEGRATED_RESEARCH_CANDIDATE',
                    'projected_home_margin':2.0}}
            Path(tmp,'1.json').write_text(json.dumps({
                'schema':'cfb_forward_research_audit_v1',
                'free_schedule_observed_at_utc':'2026-10-26T04:00:00+00:00',
                'scored_games':[row]}))
            rows=observations(tmp)
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['integrated_projected_margin'],1.0)
            self.assertEqual(rows[0]['integrated_script_projected_margin'],2.0)
            result=investigate(rows)
            self.assertFalse(result['deploy_automatically'])
            self.assertEqual(result['integrated_projected_margin_paired_games'],1)

    def test_learning_rejects_forecast_after_kickoff(self):
        import json
        import tempfile
        from cfb.learn_forward import observations
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'1.json').write_text(json.dumps({
                'schema':'cfb_forward_research_audit_v1',
                'free_schedule_observed_at_utc':'2026-10-26T04:00:00+00:00',
                'scored_games':[{'game_id':'1',
                    'kickoff_utc':'2026-10-25T19:00:00+00:00',
                    'forecast_generated_at_utc':'2026-10-25T20:00:00+00:00',
                    'predicted_home_margin':3,'actual_home_margin':-5,
                    'integrated_scoring_challenger':{
                        'status':'ORIGINAL_FROZEN_INTEGRATED_RESEARCH_CANDIDATE',
                        'projected_home_margin':15.0}}]}))
            self.assertEqual(observations(tmp),[])

class InteractionMeasurementTests(unittest.TestCase):
    def test_success_and_explosive_interactions_are_observed_and_bounded(self):
        h=histories()
        for rows in h.values():
            for r in rows:
                for typ,offs,defense in (
                    ('pass_success',18,16),('rush_success',13,11),
                    ('pass_explosive',6,4),('rush_explosive',4,3)):
                    den=35 if typ.startswith('pass') else 25
                    if 'explosive' in typ:
                        if typ.startswith('pass'):
                            r['pass_explosive_sum']=offs;r['pass_explosive_n']=den
                            r['def_pass_explosive_sum']=defense;r['def_pass_explosive_n']=den
                        else:
                            r['rush_explosive_sum']=offs;r['rush_explosive_n']=den
                            r['def_rush_explosive_sum']=defense;r['def_rush_explosive_n']=den
                    else:
                        r[typ+'_sum']=offs;r[typ+'_n']=den
                        r['def_'+typ+'_sum']=defense;r['def_'+typ+'_n']=den
        result=project_integrated(1,2,h,False,'home')
        for key in ('pass_success_residual_ppd','rush_success_residual_ppd',
                    'pass_explosive_residual_ppd','rush_explosive_residual_ppd'):
            self.assertIn(key,result['components'])
            self.assertLessEqual(abs(result['components'][key]),.020001)
