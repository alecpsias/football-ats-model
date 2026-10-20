"""Priority 1: materially used CFB advanced features, never untested promotion."""
import ast
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from cfb.advanced_scoring import COEFFICIENTS, build_adjustment,simulate_challenger,MAX_MARGIN_SHIFT
from cfb.engine import simulate
from cfb.test_cfb_engine import P,Q
from cfb.test_cfb_fresh_build9 import data_fixture, RUN_AT
from cfb.fresh_matchup import analyze, markdown
from cfb.forward_monitor import grade
from cfb.learn_forward import investigate
from cfb.research_backtest import metrics
from cfb.test_cfb_build6 import frozen
from cfb.archive import pack_forecasts


def observed(pass_epa=.4):
    return {'sos_pass_epa_differential':pass_epa,'sos_rush_epa_differential':.13,
            'observed_context':{'pass_explosive':.08,'rush_explosive':.02,
                'third_down':.07,'qb_hurries_allowed':-.04,
                'redzone_td_rate':.08,'field_position':2.}}


class AdvancedPriorityOne(unittest.TestCase):
    def test_cfb_only_does_not_import_nfl(self):
        for name in ('advanced_scoring.py','research_backtest.py','prospective.py','fresh_matchup.py'):
            t=ast.parse((Path(__file__).parent/name).read_text())
            for node in ast.walk(t):
                if isinstance(node,ast.Import):
                    self.assertFalse(any(x.name in ('model','nflverse_connector') or x.name.startswith('nfl') for x in node.names))
                elif isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith('nfl') or node.module=='model')

    def test_opponent_adjusted_epa_and_context_actually_change_points(self):
        report=simulate_challenger(P,Q,observed(),5,6,runs=1800,seed=27)
        baseline=simulate(P,Q,runs=1800,seed=27)
        self.assertEqual(report['status'],'AVAILABLE_UNCALIBRATED_RESEARCH_ONLY')
        self.assertGreater(report['simulated_margin_home'],baseline['mean_margin_home'])
        self.assertGreater(report['simulated_score_home'],baseline['mean_score_home'])
        self.assertLess(report['simulated_score_away'],baseline['mean_score_away'])
        self.assertTrue(report['not_production'])
        self.assertIsNone(report['simulation']['home_cover_probability_raw'])

    def test_hypothetical_coefficients_are_not_fitted(self):
        x=build_adjustment(observed(),4,4)
        self.assertTrue(x['coefficients_not_validated'])
        self.assertEqual(set(x['contributions']),set(COEFFICIENTS))
        self.assertLessEqual(abs(x['margin_shift_points']),MAX_MARGIN_SHIFT)

    def test_inputs_not_mutated_and_repeatable(self):
        h=copy.deepcopy(P);a=copy.deepcopy(Q);s=observed()
        x=simulate_challenger(h,a,s,5,5,runs=200,seed=5)
        y=simulate_challenger(h,a,s,5,5,runs=200,seed=5)
        self.assertEqual(x,y)
        self.assertEqual(h,P);self.assertEqual(a,Q)
        self.assertEqual(s,observed())

    def test_no_opponent_history_returns_unavailable_no_fabrication(self):
        z=observed();z['sos_pass_epa_differential']=None;z['sos_rush_epa_differential']=None
        r=simulate_challenger(P,Q,z,5,5,runs=200)
        self.assertIsNone(r['simulation'])
        self.assertIsNone(r['margin_shift_points'])
        self.assertEqual(r['status'],'UNAVAILABLE_NO_VERIFIED_OPPONENT_EPA')

    def test_missing_optional_context_is_reported_not_silently_imputed(self):
        z=observed();z['observed_context']['third_down']=None
        r=build_adjustment(z,5,5)
        self.assertIn('third_down',r['missing_signals'])
        self.assertNotIn('third_down',r['contributions'])

    def test_early_season_small_samples_shrink(self):
        x=build_adjustment(observed(),3,3)
        y=build_adjustment(observed(),11,11)
        self.assertGreater(y['margin_shift_points'],x['margin_shift_points'])
        self.assertLess(y['margin_shift_points'],MAX_MARGIN_SHIFT)

    def test_no_market_or_injury_assumptions_used(self):
        d=simulate_challenger(P,Q,observed(),5,5,runs=200,seed=100)
        self.assertIn('No verified lineup',d['limitations'][-1])
        self.assertFalse(d.get('ats_validated',False))

    def test_negative_matchup_does_not_always_favor_home(self):
        z=observed(-.4);z['sos_rush_epa_differential']=-.15
        z['observed_context']={k:-v for k,v in z['observed_context'].items()}
        self.assertLess(build_adjustment(z,5,5)['margin_shift_points'],0)

    def test_bad_numbers_and_invalid_samples_rejected(self):
        z=observed();z['sos_pass_epa_differential']=float('nan')
        with self.assertRaises(ValueError):build_adjustment(z,5,5)
        with self.assertRaises(ValueError):build_adjustment(observed(),0,5)
        with self.assertRaises(ValueError):build_adjustment(observed(),5,5,mode='future')

    def test_ablation_differentiates_mechanisms(self):
        x=build_adjustment(observed(),5,5,mode='sos_only')
        y=build_adjustment(observed(),5,5,mode='context_only')
        z=build_adjustment(observed(),5,5,mode='all')
        self.assertIn('sos_pass_epa_differential',x['contributions'])
        self.assertNotIn('sos_pass_epa_differential',y['contributions'])
        self.assertAlmostEqual(z['margin_shift_points'],x['margin_shift_points']+y['margin_shift_points'],places=5)

    def test_fresh_matchup_adds_challenger_without_changing_primary_score(self):
        with tempfile.TemporaryDirectory() as temp:
            snap=Path(temp)/'snapshot';tables,manifest=data_fixture(snap)
            def reader(path):return tables[Path(path).name].copy()
            with patch('cfb.fresh_matchup.verify',return_value=manifest),\
                 patch('cfb.fresh_matchup.pd.read_parquet',side_effect=reader):
                x=analyze(snap,'SCHOOL_A','SCHOOL_B',runs=180,min_games=1,now=RUN_AT)
                base=x['simulated_margin_home']; adv=x['advanced_research_challenger']
                self.assertEqual(x['status'],'FRESH_PREGAME_PROJECTION_LOCKED')
                self.assertTrue(x['advanced_challenger_not_primary'])
                self.assertEqual(x['simulated_margin_home'],base)
                self.assertIn('advanced_research_challenger',x)
                self.assertIn('experimental challenger',markdown(x)) if adv['simulation'] else self.assertIn('Not scored',markdown(x))
                first_source=x['source_manifest_sha256']
                # Future target play alterations cannot change the original OR candidate.
                tables['pbp.parquet'].loc[tables['pbp.parquet'].game_id.eq(103),'EPA']=50000
                y=analyze(snap,'SCHOOL_A','SCHOOL_B',runs=180,min_games=1,now=RUN_AT)
                self.assertEqual(x['simulated_margin_home'],y['simulated_margin_home'])
                self.assertEqual(x['advanced_research_challenger'],y['advanced_research_challenger'])
                self.assertEqual(first_source,y['source_manifest_sha256'])

    def test_true_early_archive_unchanged_vs_old_games(self):
        with tempfile.TemporaryDirectory() as td:
            from cfb.test_cfb_build6 import G,K
            p=Path(td);a=p/'pre';frozen(a)
            pack_forecasts(a,p/'archives')
            schedule=__import__('pandas').DataFrame([{'game_id':int(G),'completed':True,'start_date':K,
                'home_id':333,'away_id':245,'home_team':'Alabama','away_team':'Texas A&M',
                'home_points':40,'away_points':20}])
            out=grade(p/'archives',p/'no_odds',schedule,'2026-10-25T17:00:00Z','https://example.com/final','a'*64)
            self.assertEqual(out['forward_margin']['mae'],14)
            self.assertEqual(out['advanced_challenger_forward']['games_with_frozen_challenger'],0)
            self.assertNotIn('advanced_challenger',out['scored_games'][0])

    def test_new_challenger_is_graded_from_frozen_record_only(self):
        with tempfile.TemporaryDirectory() as td:
            import pandas as pd
            from cfb.test_cfb_build6 import G,K
            p=Path(td);a=p/'pre';frozen(a)
            file=a/(G+'.json'); data=json.loads(file.read_text())
            data['advanced_research_challenger']={'status':'AVAILABLE_UNCALIBRATED_RESEARCH_ONLY',
                'version':'cfb-advanced-scoring-research-v1','simulated_margin_home':12.}
            file.write_text(json.dumps(data)+'\n')
            index=json.loads((a/'index.json').read_text())
            from cfb.archive import digest
            index['predictions'][0]['sha256']=digest(file.read_bytes())
            (a/'index.json').write_text(json.dumps(index)+'\n')
            pack_forecasts(a,p/'archives')
            sched=pd.DataFrame([{'game_id':int(G),'completed':True,'start_date':K,
                'home_id':333,'away_id':245,'home_team':'Alabama','away_team':'Texas A&M',
                'home_points':40,'away_points':20}])
            out=grade(p/'archives',p/'no_odds',sched,'2026-10-25T17:00:00Z','https://example.com/final','a'*64)
            self.assertEqual(out['advanced_challenger_forward']['games_with_frozen_challenger'],1)
            self.assertEqual(out['advanced_challenger_forward']['on_same_games_advanced_mae'],8.)
            self.assertEqual(out['advanced_challenger_forward']['on_same_games_baseline_mae'],14.)
            self.assertFalse(out['scored_games'][0]['advanced_challenger']['not_deployed']==False)

    def test_postgame_uses_original_challenger_without_running_a_new_model(self):
        import hashlib
        from cfb.fresh_matchup import review,markdown
        from cfb.test_cfb_fresh_build9 import AFTER
        with tempfile.TemporaryDirectory() as temp:
            snap=Path(temp)/'snapshot';tables,manifest=data_fixture(snap)
            def read(path):return tables[Path(path).name].copy()
            with patch('cfb.fresh_matchup.verify',return_value=manifest),\
                 patch('cfb.fresh_matchup.pd.read_parquet',side_effect=read):
                frozen=analyze(snap,'SCHOOL_A','SCHOOL_B',runs=150,min_games=1,now=RUN_AT)
                manifest['captured_at_utc']=AFTER
                schedule=tables['schedule.parquet'].copy()
                schedule.loc[schedule.game_id.eq(103),'completed']=True
                schedule.loc[schedule.game_id.eq(103),'home_points']=14
                schedule.loc[schedule.game_id.eq(103),'away_points']=35
                tables['schedule.parquet']=schedule
                with patch('cfb.fresh_matchup.simulate_challenger',side_effect=AssertionError('Postgame reran challenger')):
                    scored=review(frozen,'a'*64,'777.json',snap)
            self.assertTrue(scored['postgame_does_not_rerun_the_original_model'])
            c=scored.get('original_advanced_challenger_comparison')
            if frozen['advanced_research_challenger']['simulation'] is not None:
                self.assertEqual(c['status'],'LOCKED_RESEARCH_CHALLENGER_EVALUATED')
                self.assertEqual(c['frozen_projected_home_margin'],frozen['advanced_research_challenger']['simulated_margin_home'])
                self.assertEqual(c['baseline_absolute_error'],round(abs(frozen['simulated_margin_home']+21),5))
                self.assertIn('Original research challenger comparison',markdown(scored))

    def test_learning_does_not_promote_candidate_from_one_game(self):
        rows=[{'game_id':str(i),'kickoff_utc':f'2026-10-{i+1:02}T12:00:00+00:00',
               'projected_margin':3.,'advanced_projected_margin':7.,'actual_margin':7.}
              for i in range(5)]
        z=investigate(rows)
        self.assertEqual(z['status'],'INSUFFICIENT_GENUINE_FORWARD_GAMES')
        self.assertEqual(z['advanced_challenger_paired_games'],5)
        self.assertFalse(z['deploy_automatically'])

    def test_future_scores_do_not_affect_research_challenger_selection(self):
        rows=[{'game_id':str(i),'kickoff_utc':f'2026-05-{1+i//10:02}T12:00:00+00:00',
               'projected_margin':3.,'advanced_projected_margin':7.,'actual_margin':7.} for i in range(150)]
        original=investigate(copy.deepcopy(rows))
        changed=copy.deepcopy(rows)
        # Test block is last calendar days; leave training + validation untouched.
        for r in changed[-25:]:r['actual_margin']=-100.
        updated=investigate(changed)
        self.assertEqual(original['advanced_challenger_validation']['passes_gate'],
                         updated['advanced_challenger_validation']['passes_gate'])
        self.assertEqual(original['advanced_challenger_validation']['advanced_validation_mae'],
                         updated['advanced_challenger_validation']['advanced_validation_mae'])
        self.assertNotEqual(original['advanced_challenger_validation']['holdout_advanced_mae'],
                            updated['advanced_challenger_validation']['holdout_advanced_mae'])
        self.assertFalse(updated['deploy_automatically'])


if __name__=='__main__': unittest.main()
