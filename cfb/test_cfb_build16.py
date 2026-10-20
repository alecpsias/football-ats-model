"""Build16 policy, source-observed mechanisms, and preserved primary scorer."""
from __future__ import annotations
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from cfb.mechanism_build16 import (VERSION,observe_game_context,mechanism_profile,simulate_build16)
from cfb.test_cfb_integrated_build15 import histories
from cfb.research_backtest import project
from cfb.engine import simulate
from cfb.fresh_matchup import analyze, markdown, review
from cfb.test_cfb_fresh_build9 import data_fixture, RUN_AT, AFTER
from cfb.forward_monitor import grade
from cfb.test_cfb_build6 import frozen,G,K
from cfb.archive import pack_forecasts, digest


def decorated():
    h=histories()
    for group in h.values():
        for rec in group:
            rec['mechanism']={
               'third_off':{'short_1_2':[7,10],'medium_3_6':[5,10],'long_7_plus':[3,12]},
               'third_def':{'short_1_2':[6,10],'medium_3_6':[4,10],'long_7_plus':[2,12]},
               'pass_states_off':{'trailing_9_plus':[17,20],
                                  'within_8':[30,55],'leading_9_plus':[8,25]},
               'turnover_zone_off':{'opponent_40_or_closer':2,'outside_opponent_40':6},
               'turnover_next_field_by_zone':{'opponent_40_or_closer':[22,28],
                                               'outside_opponent_40':[58,65]},
               'fourth_off':{'attempted':2,'converted':1},
               'scoring_off':{'td':2,'fg':1,'turnover':1,'other':1,'opportunities':5}}
    return h


class Build16Tests(unittest.TestCase):
    def test_primary_is_unmodified_and_one_challenger(self):
        h=decorated();before=copy.deepcopy(h)
        hp=project(h[1],h[2],False,'home',False)
        ap=project(h[2],h[1],False,'away',False)
        expected=simulate(hp,ap,runs=300,seed=83)
        z=simulate_build16(1,2,h,False,runs=300,seed=83)
        self.assertEqual(z['version'],VERSION)
        self.assertTrue(z['not_primary'])
        self.assertFalse(z['ATS_calibrated'])
        self.assertTrue(z['simulation']['build16_mechanisms_enabled'])
        self.assertEqual(h,before)
        self.assertEqual(project(h[1],h[2],False,'home',False),hp)
        self.assertEqual(simulate(hp,ap,runs=300,seed=83),expected)
        self.assertTrue(math.isfinite(z['mean_margin_home']))
        self.assertEqual(z,simulate_build16(1,2,h,False,runs=300,seed=83))

    def test_game_state_changes_observed_pass_mix(self):
        h=decorated();p=mechanism_profile(1,2,h,{},{} )
        mix=p['home']['pass_states']['by_state']
        self.assertGreater(mix['trailing_9_plus'],mix['leading_9_plus'])
        self.assertEqual(p['turnover_field']['status'],'OBSERVED')
        self.assertGreater(p['turnover_field']['samples'],3)
        self.assertLessEqual(abs(p['home']['third']['residual_ppd']),.055)

    def test_missing_mechanisms_are_explicit_and_no_invented_proxies(self):
        h=histories();r=simulate_build16(1,2,h,False,runs=200,seed=3)
        self.assertEqual(r['mechanism_evidence']['home']['third']['status'],'UNAVAILABLE')
        self.assertEqual(r['mechanism_evidence']['home']['pass_states']['status'],'INSUFFICIENT')
        self.assertEqual(r['mechanism_evidence']['turnover_field']['status'],'INSUFFICIENT')
        self.assertIn('full_pressure_rate_not_source_backed',r['home']['missing_features'])
        self.assertEqual(r,simulate_build16(1,2,h,False,runs=200,seed=3))

    def test_new_mechanisms_affect_simulation_not_just_report_text(self):
        from cfb.integrated_scoring import simulate_integrated
        h=decorated();combined=simulate_integrated(1,2,h,False,runs=1500,seed=51,use_script=True)
        new=simulate_build16(1,2,h,False,runs=1500,seed=51)
        self.assertNotEqual(new['mean_margin_home'],combined['mean_margin_home'])
        self.assertTrue(new['simulation']['build16_mechanisms_enabled'])
        self.assertEqual(combined['simulation']['build16_mechanisms_enabled'],False)
        self.assertEqual(len(new['mechanism_evidence']['home']['pass_states']['by_state']),3)

    def test_six_adverse_scripts_are_within_single_challenger(self):
        z=simulate_build16(1,2,decorated(),False,runs=130,seed=29,stress_runs=100)
        self.assertEqual(len(z['adverse_sensitivity']['counterfactuals']),6)
        self.assertTrue(z['adverse_sensitivity']['not_fitted'])
        self.assertTrue(z['not_primary'])
        self.assertEqual(z['version'],VERSION)
        self.assertEqual(len(z['margin_p10_p50_p90']),3)
        with self.assertRaises(ValueError):
            simulate_build16(1,2,decorated(),False,runs=100,stress_runs=99)

    def test_no_fake_scores_when_opponent_sos_missing(self):
        from cfb.test_cfb_integrated_build15 import record
        h={1:[record(2)]*3,2:[record(1)]*3}
        z=simulate_build16(1,2,h,False,runs=100)
        self.assertEqual(z['status'],'UNAVAILABLE_REQUIRED_OPPONENT_EPA')
        self.assertIsNone(z['simulation'])

    def test_source_context_uses_preplay_score_and_next_actual_drive(self):
        def row(seq,team,drive,distance,diff,kind,*,turnover=False,converted=False,yl=50,down=1):
            return {'game_play_number':seq,'pos_team_id':team,'def_pos_team_id':2 if team==1 else 1,
             'drive_id':drive,'yards_to_goal':yl,'half':1,'down':down,'distance':distance,
             'pass':kind=='pass','rush':kind=='rush','pos_score_diff_start':diff,
             'firstD_by_yards':converted,'firstD_by_penalty':False,'rush_td':False,'pass_td':False,
             'offense_score_play':False,'td_play':False,'fg_made':False,'downs_turnover':False,
             'pos_fumble_lost':False,'int':turnover,'penalty_1st_conv':False}
        rows=[row(1,1,'a',3,-10,'pass',converted=True,down=3),
              row(2,1,'a',8,-10,'pass',turnover=True,yl=30,down=3),
              row(3,2,'b',10,10,'rush',yl=80),
              row(4,2,'b',6,10,'pass',down=3),
              row(5,1,'c',1,14,'rush',converted=True,down=3)]
        observed=observe_game_context(pd.DataFrame(rows),1,2)
        self.assertEqual(observed['third_off']['short_1_2'],[1,1])
        self.assertEqual(observed['third_off']['long_7_plus'],[0,1])
        self.assertEqual(observed['third_off']['medium_3_6'],[1,1])
        self.assertEqual(observed['pass_states_off']['trailing_9_plus'],[2,2])
        self.assertEqual(observed['pass_states_off']['leading_9_plus'],[0,1])
        self.assertEqual(observed['turnover_zone_off']['opponent_40_or_closer'],1)
        self.assertEqual(observed['turnover_next_field_by_zone']['opponent_40_or_closer'],[20.0])

    def test_fresh_report_contains_baseline_and_only_build16(self):
        with tempfile.TemporaryDirectory() as td:
            snap=Path(td)/'snap';tables,manifest=data_fixture(snap)
            with patch('cfb.fresh_matchup.verify',return_value=manifest),\
                 patch('cfb.fresh_matchup.pd.read_parquet',side_effect=lambda path:tables[Path(path).name].copy()):
                r=analyze(snap,'SCHOOL_A','SCHOOL_B',runs=120,min_games=1,now=RUN_AT)
            self.assertIn('build16_challenger',r)
            for old in ('advanced_research_challenger','game_script_research_challenger',
                        'integrated_scoring_research_challenger','integrated_scoring_plus_script_research'):
                self.assertNotIn(old,r)
            self.assertEqual(r['simulated_margin_home'],r['simulation']['mean_margin_home'])
            md=markdown(r)
            self.assertIn('Build 16',md)
            self.assertNotIn('Integrated NFL-style CFB scoring challenger',md)

    def test_frozen_build16_is_never_recomputed_after_final(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'forecasts';frozen(folder)
            file=folder/(G+'.json');data=json.loads(file.read_text())
            data['build16_challenger']={'status':'AVAILABLE_UNFITTED_BUILD16_RESEARCH_ONLY',
                'version':VERSION,'mean_margin_home':12.,'margin_p10_p50_p90':[-4,12,26]}
            file.write_text(json.dumps(data)+'\n')
            index=json.loads((folder/'index.json').read_text())
            index['predictions'][0]['sha256']=digest(file.read_bytes())
            (folder/'index.json').write_text(json.dumps(index)+'\n')
            pack_forecasts(folder,Path(td)/'packed')
            schedule=pd.DataFrame([{'game_id':int(G),'completed':True,'start_date':K,
               'home_id':333,'away_id':245,'home_team':'Alabama','away_team':'Texas A&M',
               'home_points':40,'away_points':20}])
            with patch('cfb.mechanism_build16.simulate_build16',side_effect=AssertionError('postgame replay forbidden')):
                result=grade(Path(td)/'packed',Path(td)/'odds',schedule,
                    '2026-10-25T17:00:00Z','https://example.com', 'a'*64)
            self.assertEqual(result['build16_challenger_forward']['paired_games'],1)
            self.assertEqual(result['build16_challenger_forward']['paired_build16_mae'],8.)
            self.assertEqual(result['build16_challenger_forward']['paired_baseline_mae'],14.)
            self.assertFalse(result['scored_games'][0]['build16_challenger']['recomputed_after_kickoff'])

    def test_live_new_forecasts_only_two_models(self):
        files=('prospective.py','fresh_matchup.py')
        for file in files:
            text=(Path(__file__).parent/file).read_text()
            self.assertIn("'build16_challenger'",text)
        self.assertIn("'build16_candidate_margin'",(Path(__file__).parent/'research_backtest.py').read_text())
        self.assertIn("'build16_projected_margin'",(Path(__file__).parent/'learn_forward.py').read_text())

class Build16HistoricalReplayCompatibility(unittest.TestCase):
    def test_calibration_and_diagnostics_accept_only_build16(self):
        from cfb.calibration import calibrate
        from cfb.diagnostics import investigate
        from datetime import datetime, timedelta, timezone
        start=datetime(2026,1,1,tzinfo=timezone.utc)
        games=[]
        for i in range(120):
            baseline=float((i%11)-5)
            games.append({'game_id':str(401800000+i),
                          'kickoff':(start+timedelta(days=i)).isoformat(),
                          'home':'Alpha','away':'Bravo',
                          'actual_margin':float(2*baseline+1),
                          'baseline_margin':baseline,
                          'build16_candidate_margin':baseline+.25,
                          'sos_margin_feature':None,
                          'context_margin_feature':None})
        result=calibrate(games)
        self.assertEqual(result['status'],'CFB_MARGIN_RESEARCH_VALIDATION_ONLY_NOT_ATS_VALIDATED')
        self.assertEqual(result['untouched_holdout_games'],24)
        self.assertIn('holdout_build16_challenger',result)
        self.assertNotIn('holdout_old_epa_challenger',result)
        self.assertIn('build16_delta',result['coefficients'])
        diagnostic=investigate(games)
        self.assertEqual(diagnostic['status'],'CFB_RETROSPECTIVE_DIAGNOSTICS_NO_ATS_VALIDATION')
        self.assertIn('build16_delta',diagnostic['feature_coverage'])
        self.assertNotIn('epa_delta',diagnostic['feature_coverage'])

    def test_unavailable_build16_not_misrepresented_as_zero_effect(self):
        from cfb.calibration import _design
        with self.assertRaises(ValueError):
            _design([{'baseline_margin':3.0,'build16_candidate_margin':None}])


if __name__=='__main__':unittest.main()
