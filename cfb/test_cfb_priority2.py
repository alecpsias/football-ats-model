"""Build 13: conditional personnel evidence, no fictional QB or injury points."""
import ast
import copy
import unittest
from pathlib import Path
import pandas as pd
from cfb.personnel import quarterback_usage
from cfb.personnel_effects import (quarterback_effects,skill_player_exposures,
                                    simulate_personnel_scenarios,other_position_availability,MAX_ABS_POINTS_PER_DRIVE_SHIFT)
from cfb.engine import simulate
from cfb.test_cfb_engine import P,Q


def example():
    games=[];rows=[]
    for i in range(4):
        gid=100+i
        games.append({'game_id':gid,'home_id':777,'away_id':888,'_kickoff':pd.Timestamp('2026-09-01T00:00Z')+pd.Timedelta(days=i*7)})
        # Both passers play in each prior completed game; still not proof of
        # future starter and not a same-role causal ability comparison.
        for j in range(20):
            rows.append({'game_id':gid,'pos_team_id':777,'pass':True,'rush':False,
                         'passer_player_id':10,'rusher_player_id':None,
                         'receiver_player_id':31 if j%3 else None,
                         'drive_id':f'{gid}-{j%11}','EPA':.30})
        for j in range(12):
            rows.append({'game_id':gid,'pos_team_id':777,'pass':True,'rush':False,
                         'passer_player_id':11,'rusher_player_id':None,
                         'receiver_player_id':31 if j%2 else None,
                         'drive_id':f'{gid}-{j%11}','EPA':-.35})
        for qb,n,epa in [(10,5,.20),(11,4,-.25)]:
            for j in range(n):
                rows.append({'game_id':gid,'pos_team_id':777,'pass':False,'rush':True,
                         'passer_player_id':None,'rusher_player_id':qb,
                         'receiver_player_id':None,
                         'drive_id':f'{gid}-{j%11}','EPA':epa})
        for j in range(7):
            rows.append({'game_id':gid,'pos_team_id':777,'pass':False,'rush':True,
                         'passer_player_id':None,'rusher_player_id':41,
                         'receiver_player_id':None,
                         'drive_id':f'{gid}-{j%11}','EPA':.1})
    roster=pd.DataFrame([
        {'team_id':777,'athlete_id':10,'position_abbreviation':'QB','display_name':'Main QB'},
        {'team_id':777,'athlete_id':11,'position_abbreviation':'QB','display_name':'Reserve QB'},
        {'team_id':777,'athlete_id':31,'position_abbreviation':'WR','display_name':'Leading WR'},
        {'team_id':777,'athlete_id':41,'position_abbreviation':'RB','display_name':'Leading RB'},
    ])
    frame=pd.DataFrame(rows);sched=pd.DataFrame(games)
    qbs=quarterback_usage(frame,roster,sched,777)
    return frame,sched,roster,qbs


class PersonnelPriorityTwo(unittest.TestCase):
    def test_starts_as_conditional_no_lineup_assumption(self):
        p,g,r,q=example()
        e=quarterback_effects(p,g,r,777,q,{'status':'NO_OFFICIAL_REPORT_CAPTURE','players':[]})
        self.assertEqual(e['status'],'EMPIRICAL_CONDITIONAL_SCENARIO_UNCALIBRATED')
        self.assertIsNone(e['scenario']['reported_reference_status'])
        self.assertEqual(e['scenario']['availability_basis'],'UNCONFIRMED_STARTER_OR_ABSENCE')
        self.assertFalse(e['baseline_auto_adjusted'])
        self.assertTrue(e['scenario']['not_game_day_starter_confirmation'])
        self.assertEqual(e['scenario']['candidate_qb'],'Reserve QB')

    def test_sample_sizes_and_historical_games_recorded(self):
        p,g,r,q=example()
        s=quarterback_effects(p,g,r,777,q,{'players':[]})['scenario']
        self.assertEqual(s['reference_pass_play_sample'],80)
        self.assertEqual(s['candidate_pass_play_sample'],48)
        self.assertEqual(len(s['shared_prior_game_ids']),4)
        self.assertEqual(len(set(s['shared_prior_game_ids'])),4)

    def test_underperforming_backup_reduces_expected_points(self):
        p,g,r,q=example()
        e=quarterback_effects(p,g,r,777,q,{'players':[]})
        self.assertLess(e['scenario']['estimated_change_in_points_per_drive'],0)
        self.assertLess(e['scenario']['passing_component_change_in_ppd'],0)
        self.assertEqual(e['scenario']['qb_rushing_evidence']['status'],'MEASURED_RUSH_EPA_CONDITIONAL_FIXED_VOLUME')
        self.assertLess(e['scenario']['rushing_component_change_in_ppd'],0)
        self.assertLessEqual(abs(e['scenario']['estimated_change_in_points_per_drive']),MAX_ABS_POINTS_PER_DRIVE_SHIFT)

    def test_signed_effect_flips_when_alternate_better(self):
        p,g,r,q=example()
        p.loc[p.passer_player_id.eq(10),'EPA']=-.4
        p.loc[p.passer_player_id.eq(11),'EPA']=.4
        p.loc[p.rusher_player_id.eq(10),'EPA']=-.2
        p.loc[p.rusher_player_id.eq(11),'EPA']=.2
        q=quarterback_usage(p,r,g,777)
        e=quarterback_effects(p,g,r,777,q,{'players':[]})
        self.assertGreater(e['scenario']['estimated_change_in_points_per_drive'],0)

    def test_reported_out_is_not_confirmed_actual_inactive(self):
        p,g,r,q=example()
        status={'source_verified':False,'players':[{'team_id':'777','athlete_id':'10','status':'OUT'}]}
        e=quarterback_effects(p,g,r,777,q,status)
        self.assertEqual(e['scenario']['availability_basis'],'REPORTED_OUT_UNATTESTED_SOURCE_CAPTURE')
        self.assertFalse(e['scenario']['report_source_verified'])
        self.assertFalse(e['baseline_auto_adjusted'])

    def test_pending_questionable_never_asserted_out(self):
        p,g,r,q=example()
        status={'players':[{'team_id':'777','athlete_id':'10','status':'QUESTIONABLE'}]}
        e=quarterback_effects(p,g,r,777,q,status)
        self.assertEqual(e['scenario']['reported_reference_status'],'QUESTIONABLE')
        self.assertEqual(e['scenario']['availability_basis'],'UNCONFIRMED_STARTER_OR_ABSENCE')

    def test_ignores_data_for_future_game(self):
        p,g,r,q=example()
        orig=quarterback_effects(p,g,r,777,q,{'players':[]})
        later=p.iloc[:60].copy()
        later['game_id']=105;later['EPA']=10000.
        joined=pd.concat([p,later],ignore_index=True)
        later_effect=quarterback_effects(joined,g,r,777,q,{'players':[]})
        self.assertEqual(orig,later_effect)

    def test_abstains_without_backup_sample(self):
        p,g,r,q=example()
        p=p.loc[~((p.passer_player_id==11)&(p.game_id!=100))].copy()
        q=quarterback_usage(p,r,g,777)
        effect=quarterback_effects(p,g,r,777,q,{'players':[]})
        self.assertEqual(effect['status'],'UNAVAILABLE_NOT_ENOUGH_REPLACEMENT_EVIDENCE')
        self.assertIsNone(effect['scenario'])

    def test_abstains_if_qb_identity_coverage_too_low(self):
        p,g,r,q=example()
        p.loc[p.passer_player_id.eq(10),'passer_player_id']=9999
        q=quarterback_usage(p,r,g,777)
        effect=quarterback_effects(p,g,r,777,q,{'players':[]})
        self.assertNotEqual(effect['status'],'EMPIRICAL_CONDITIONAL_SCENARIO_UNCALIBRATED')

    def test_missing_fields_abstain(self):
        p,g,r,q=example()
        effect=quarterback_effects(p.drop(columns=['drive_id']),g,r,777,q,{'players':[]})
        self.assertEqual(effect['status'],'MISSING_REQUIRED_QB_PLAY_FIELDS')
        self.assertIsNone(effect['scenario'])

    def test_skill_usage_is_descriptive_and_uses_identified_targets(self):
        p,g,r,q=example()
        status={'players':[{'team_id':'777','athlete_id':'31','status':'OUT'}]}
        x=skill_player_exposures(p,g,r,777,status)
        self.assertEqual(x['status'],'DESCRIPTIVE_ONLY_NO_SKILL_PLAYER_SCORE_ADJUSTMENT')
        receiver=next(z for z in x['players'] if z['athlete_id']=='31')
        self.assertEqual(receiver['reported_availability'],'OUT')
        self.assertEqual(receiver['name'],'Leading WR')
        self.assertIsNone(receiver['point_adjustment'])
        self.assertTrue(x['missing_report_is_not_healthy'])
        back=next(z for z in x['players'] if z['athlete_id']=='41')
        self.assertEqual(back['name'],'Leading RB')
        self.assertGreater(back['recent_rushes'],0)

    def test_line_defense_absences_reported_without_point_invention(self):
        p,g,r,q=example()
        r=pd.concat([r,pd.DataFrame([{'team_id':777,'athlete_id':99,
           'position_abbreviation':'OT','display_name':'Left Tackle'}])],ignore_index=True)
        st={'source_verified':False,'players':[{'team_id':'777','athlete_id':'99',
             'name':'Left Tackle','status':'OUT'}]}
        x=other_position_availability(r,st,777)
        self.assertEqual(x['players'][0]['position'],'OT')
        self.assertIsNone(x['players'][0]['point_adjustment'])
        self.assertFalse(x['players'][0]['actual_inactive_verified'])
        self.assertTrue(x['no_report_does_not_mean_healthy'])

    def test_same_seed_conditional_preserves_inputs_and_primary(self):
        p,g,r,q=example();e=quarterback_effects(p,g,r,777,q,{'players':[]})
        a=copy.deepcopy(P);b=copy.deepcopy(Q)
        baseline=simulate(a,b,runs=500,seed=51)
        x=simulate_personnel_scenarios(a,b,e,{'status':'UNAVAILABLE'},runs=500,seed=51)
        self.assertEqual((a,b),(P,Q))
        self.assertEqual(x['baseline_same_seed_margin'],baseline['mean_margin_home'])
        self.assertEqual(len(x['scenarios']),1)
        self.assertLess(x['scenarios'][0]['candidate_home_margin'],baseline['mean_margin_home'])
        self.assertTrue(x['scenarios'][0]['conditional_only'])
        self.assertTrue(x['not_auto_deployed'])
        self.assertEqual(x,simulate_personnel_scenarios(a,b,e,{},runs=500,seed=51))

    def test_weak_evidence_produces_no_scenarios(self):
        z=simulate_personnel_scenarios(P,Q,{'status':'UNAVAILABLE'}, {'status':'UNAVAILABLE'},runs=200,seed=11)
        self.assertEqual(z['scenarios'],[])
        self.assertTrue(z['requires_verified_lineup_for_predictive_grading'])

    def test_fresh_matchup_exposes_evidence_without_moving_primary_prediction(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from cfb.test_cfb_fresh_build9 import data_fixture, RUN_AT
        from cfb.fresh_matchup import analyze, markdown
        with TemporaryDirectory() as temp:
            snap=Path(temp)/'snap'; tables,manifest=data_fixture(snap)
            def read(f):return tables[Path(f).name].copy()
            with patch('cfb.fresh_matchup.verify',return_value=manifest),\
                 patch('cfb.fresh_matchup.pd.read_parquet',side_effect=read):
                x=analyze(snap,'SCHOOL_A','SCHOOL_B',runs=140,min_games=1,now=RUN_AT)
            self.assertEqual(x['status'],'FRESH_PREGAME_PROJECTION_LOCKED')
            self.assertIsNone(x['personnel_adjustment_to_model_points'])
            self.assertEqual(x['conditional_personnel_scenarios']['scenarios'],[])
            self.assertIn('empirical_qb_replacement_evidence',x)
            self.assertIn('skill_player_availability_exposure',x)
            self.assertIn('other_position_availability',x)
            self.assertIn('Personnel replacement evidence',markdown(x))

    def test_frozen_batch_archive_records_personnel_without_changing_margin(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from cfb.test_cfb_build5 import Build5Tests
        from cfb.prospective import freeze
        import json
        with TemporaryDirectory() as temp:
            t=Path(temp);snap=t/'snap';manifest,tables=Build5Tests.snapshot_data(snap)
            def reader(f):return tables[Path(f).stem].copy()
            with patch('cfb.prospective.verify',return_value=manifest),\
                 patch('cfb.prospective.pd.read_parquet',side_effect=reader):
                freeze(snap,t/'forecasts',runs=100,days=7,min_games=2,
                       now='2026-10-19T12:10:00+00:00')
            r=json.loads((t/'forecasts'/'103.json').read_text())
            self.assertEqual(r['status'],'FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE')
            self.assertIsNone(r['personnel_adjustment_to_model_points'])
            self.assertEqual(r['conditional_personnel_scenarios']['scenarios'],[])
            self.assertIsInstance(r['simulated_margin_home'],float)
            self.assertTrue(r['advanced_challenger_not_primary'])

    def test_final_grader_does_not_score_counterfactual_as_prediction(self):
        from tempfile import TemporaryDirectory
        from cfb.test_cfb_build6 import frozen,G,K
        from cfb.archive import pack_forecasts
        from cfb.forward_monitor import grade
        import json
        with TemporaryDirectory() as temp:
            base=Path(temp);folder=base/'raw';frozen(folder)
            file=folder/(G+'.json');row=json.loads(file.read_text())
            row['conditional_personnel_scenarios']={'version':'research',
                 'scenarios':[{'candidate_qb':'Other','candidate_home_margin':20.0}]}
            file.write_text(json.dumps(row))
            index=json.loads((folder/'index.json').read_text());import hashlib
            index['predictions'][0]['sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
            (folder/'index.json').write_text(json.dumps(index))
            pack_forecasts(folder,base/'zip')
            sched=pd.DataFrame([{'game_id':int(G),'completed':True,'start_date':K,
                 'home_id':333,'away_id':245,'home_team':'Alabama','away_team':'Texas A&M',
                 'home_points':40,'away_points':20}])
            report=grade(base/'zip',base/'missing_odds',sched,'2026-10-25T17:00:00+00:00',
                         'https://github.com/sportsdataverse/data','a'*64)
            self.assertEqual(report['graded_final_games'],1)
            self.assertEqual(report['forward_margin']['mae'],14.0)
            self.assertEqual(report['personnel_conditional_forward']['games_with_conditional_scenarios'],1)
            self.assertTrue(report['scored_games'][0]['conditional_personnel_research']['not_included_in_forward_mae'])

    def test_cfb_independent_and_workflow_has_no_invalid_flags(self):
        folder=Path(__file__).parent
        for mod in ('personnel_effects.py','personnel.py','fresh_matchup.py','prospective.py'):
            x=ast.parse((folder/mod).read_text())
            for n in ast.walk(x):
                if isinstance(n,ast.ImportFrom):
                    self.assertNotEqual(n.module,'model')
                    self.assertFalse((n.module or '').startswith('nfl'))
        wf=(folder.parent/'.github/workflows/cfb-free-data.yml').read_text()
        self.assertNotIn('python -m cfb.research_backtest \\\n            --availability-dir',wf)
        self.assertIn('--availability-dir cfb/availability',wf)

    def test_request_and_batch_archive_both_personnel(self):
        src=(Path(__file__).parent/'fresh_matchup.py').read_text()
        batch=(Path(__file__).parent/'prospective.py').read_text()
        monitor=(Path(__file__).parent/'forward_monitor.py').read_text()
        for code in (src,batch):
            self.assertIn('conditional_personnel_scenarios',code)
            self.assertIn('skill_player_availability_exposure',code)
            self.assertIn('personnel_adjustment_to_model_points',code)
        self.assertIn('FROZEN_COUNTERFACTUAL_NOT_A_PREDICTION',monitor)
        self.assertIn('not_included_in_forward_mae',monitor)

if __name__=='__main__':unittest.main()
