"""Failure-1 source/score-state and single-challenger opt-in research tests."""
from __future__ import annotations
import copy,math,unittest
import pandas as pd
from cfb.test_cfb_integrated_build15 import histories
from cfb.rush_severity_source import extract,append_training_record
from cfb.rush_severity_research import experiment,MODES
from cfb.mechanism_build16 import simulate_build16
from cfb.game_script import simulate_script
from cfb.research_backtest import project
from cfb.engine import simulate

def fixture():
    rows=[]
    seq=0
    for team in (1,2):
        for i in range(20):
            seq+=1
            rows.append({'game_play_number':seq,'pos_team_id':team,
                'down':1 if i%2 else 2,'distance':10.,
                'rush':True,'pass':False,
                'yards_gained':-3 if i%10==0 else (8 if i%3 else 2),
                'pos_score_diff_start':-12 if i%3==0 else 0})
    return pd.DataFrame(rows)

def decorated():
    h=histories()
    for tid,rows in h.items():
        for r in rows:
            pos=tid==1
            r.update({'mismatch_off_early_rush_success_yes':27 if pos else 14,
              'mismatch_off_early_rush_n':35,
              'mismatch_off_early_negative_yes':1 if pos else 6,
              'mismatch_off_early_nonpositive_yes':3 if pos else 11,
              'mismatch_def_early_rush_success_yes':18 if pos else 29,
              'mismatch_def_early_rush_n':35,
              'mismatch_def_early_negative_yes':5 if pos else 2,
              'mismatch_def_early_nonpositive_yes':9 if pos else 4,
              'mismatch_off_trailing_rush_good':9 if pos else 4,
              'mismatch_off_trailing_rush_n':12,
              'mismatch_def_trailing_rush_good':7 if pos else 9,
              'mismatch_def_trailing_rush_n':12,
              'mismatch_off_within_8_rush_good':7 if pos else 6,
              'mismatch_off_within_8_rush_n':12})
    return h

class SevereRushTest(unittest.TestCase):
    def test_source_only_uses_prep_play_score_and_designed_rushing(self):
        d=fixture()
        x=extract(d,1,2)
        self.assertEqual(x['status'],'OBSERVED_VALID')
        self.assertEqual(x['features']['off_early_rush_n'],20)
        self.assertEqual(x['features']['off_early_negative_yes'],2)
        self.assertEqual(x['features']['off_trailing_rush_n'],7)
        rec={}
        append_training_record(rec,x)
        self.assertEqual(rec['mismatch_off_early_rush_n'],20)
        d=d.drop(columns=['pos_score_diff_start'])
        self.assertEqual(extract(d,1,2)['status'],'UNAVAILABLE_REQUIRED_PBP_FIELDS')

    def test_mismatch_is_asymmetric_and_severity_nonlinear(self):
        h=decorated()
        x=experiment(1,2,h,'combined')
        self.assertEqual(x['status'],'UNFITTED_SCENARIO_READY')
        self.assertGreater(x['home']['joint_severity_signal'],0)
        self.assertEqual(x['away']['joint_severity_signal'],0.)
        self.assertGreater(x['home']['severity_ppd_unfitted'],0)
        self.assertTrue(x['not_deployed'])
        self.assertTrue(x['avoids_raw_rush_EPA_duplication'])
        self.assertLessEqual(abs(x['home']['baseline_ppd_delta']),.45)

    def test_single_challenger_all_modes_determinism_and_locked_baseline(self):
        h=decorated();prior=copy.deepcopy(h)
        base1=project(h[1],h[2],False,'home',False)
        base2=project(h[2],h[1],False,'away',False)
        base=simulate(base1,base2,runs=250,seed=25)
        active=simulate_build16(1,2,h,False,runs=250,seed=25)
        for mode in MODES:
            a=simulate_build16(1,2,h,False,runs=250,seed=25,mismatch_research=mode)
            b=simulate_build16(1,2,h,False,runs=250,seed=25,mismatch_research=mode)
            self.assertEqual(a,b)
            self.assertEqual(a['simulation']['simulations'],250)
            self.assertEqual(a['mismatch_research']['mode'],mode)
            self.assertTrue(a['not_deployed'])
            self.assertTrue(math.isfinite(a['mean_margin_home']))
        self.assertEqual(h,prior)
        self.assertEqual(simulate(base1,base2,runs=250,seed=25),base)
        self.assertEqual(simulate_build16(1,2,h,False,runs=250,seed=25),active)
        self.assertNotIn('mismatch_research',active)

    def test_no_stored_pregame_score_state_cannot_create_trailing_effect(self):
        h=decorated()
        for row in h[1]:
            row.pop('mismatch_off_trailing_rush_n')
        x=experiment(1,2,h,'resilience')
        self.assertEqual(x['status'],'UNAVAILABLE_REQUIRED_PREGAME_FEATURES')
        y=simulate_build16(1,2,h,False,runs=250,seed=25,mismatch_research='resilience')
        self.assertIsNone(y['simulation'])
        with self.assertRaises(ValueError):
            experiment(1,2,h,'calibrated_on_postgame')
        with self.assertRaises(ValueError):
            simulate_build16(1,2,decorated(),False,runs=250,seed=25,
                  mismatch_research='combined',chain_research='combined')

    def test_score_state_resilience_changes_only_trailing_scripts(self):
        h=decorated();hp=project(h[1],h[2],False,'home',False)
        ap=project(h[2],h[1],False,'away',False)
        base={'home':{'trailing_rush_resilience_ppd':0.},
              'away':{'trailing_rush_resilience_ppd':0.},'turnover_field':{}}
        new=copy.deepcopy(base);new['home']['trailing_rush_resilience_ppd']=.2
        original=simulate_script(hp,ap,h[1],h[2],runs=500,seed=5,mechanisms=base)
        adjusted=simulate_script(hp,ap,h[1],h[2],runs=500,seed=5,mechanisms=new)
        self.assertNotEqual(adjusted['mean_score_home'],original['mean_score_home'])
        self.assertEqual(original['simulations'],500)

if __name__=='__main__':unittest.main()
