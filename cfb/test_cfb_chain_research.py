"""CFB third-down and rushing chain feature tests (no baseline modifications)."""
import copy
import unittest
import math
import pandas as pd
from cfb.chain_source import observed_chain,append_historical_chain
from cfb.chain_interaction_research import experiment
from cfb.mechanism_build16 import simulate_build16
from cfb.engine import simulate
from cfb.research_backtest import project
from cfb.test_cfb_integrated_build15 import histories

def row(team,drive,num,down,dist,rush,gain,**kwargs):
    return {'pos_team_id':team,'def_pos_team_id':2 if team==1 else 1,
            'game_play_number':num,'drive_id':drive,
            'down':down,'distance':dist,'rush':rush,'pass':not rush,
            'yards_gained':gain,'firstD_by_yards':kwargs.get('first',False),
            'firstD_by_penalty':False,
            'rush_td':False,'pass_td':False,'int':False,
            'pos_fumble_lost':False}

def fake_history():
    h=histories()
    for team,records in h.items():
        for x in records:
            good=(team==1)
            x.update({
                'chain_off_early_rush_success_yes':17 if good else 6,
                'chain_off_early_rush_success_n':25,
                'chain_off_early_rush_plays':25,
                'chain_off_early_offense_plays':42,
                'chain_off_third_manageable_yes':12 if good else 6,
                'chain_off_third_attempts':24,
                'chain_off_sustained_drives':8 if good else 4,
                'chain_off_observed_drives':12,
                'chain_def_early_rush_success_yes':10 if good else 18,
                'chain_def_early_rush_success_n':25,
                'chain_def_third_manageable_yes':7 if good else 14,
                'chain_def_third_attempts':24,
                'chain_def_sustained_drives':5 if good else 9,
                'chain_def_observed_drives':12})
    return h

class ChainTests(unittest.TestCase):
    def test_third_conversion_is_next_first_down_not_false_provider_flag(self):
        a=[row(1,'d1',1,1,10,True,8),
           row(1,'d1',2,2,2,True,0),
           row(1,'d1',3,3,2,False,5), # provider firstD false
           row(1,'d1',4,1,10,True,9),
           row(1,'d1',5,2,1,True,1),
           row(2,'d2',6,1,10,True,2)]
        x=observed_chain(pd.DataFrame(a),1,2)
        self.assertNotEqual(x['status'],'OBSERVED_QUALIFIED') # insufficient drives
        # Add legitimate additional observed drives: extractor never infers
        # conversion merely from false provider firstD marker.
        more=[]
        for n in range(3,9):
            team=1 if n%2 else 2
            more.extend([row(team,'d'+str(n),20+n*3,1,10,True,4),
                         row(team,'d'+str(n),21+n*3,2,6,True,6),
                         row(team,'d'+str(n),22+n*3,3,4,False,2)])
        z=observed_chain(pd.DataFrame(a+more),1,2)
        self.assertEqual(z['status'],'OBSERVED_QUALIFIED')
        self.assertEqual(z['offense']['third_conversion_yes'],1)
        self.assertEqual(z['offense']['third_resolved_n'],1)
        self.assertGreater(z['offense']['third_unresolved_n'],0)

    def test_unverified_and_no_future_fill(self):
        self.assertEqual(observed_chain(pd.DataFrame(),1,2)['status'],'UNAVAILABLE_SOURCE_FIELDS')
        rec={};append_historical_chain(rec,{'status':'UNAVAILABLE'},'offense')
        self.assertEqual(rec,{})

    def test_bounded_scenario_no_double_count_and_fail_closed(self):
        h=fake_history();before=copy.deepcopy(h)
        for mode in ('distance','sustain','combined'):
            z=experiment(1,2,h,mode)
            self.assertEqual(z['status'],'UNFITTED_SCENARIO_READY')
            self.assertTrue(z['not_deployed'])
            self.assertTrue(z['does_not_double_count_rush_epa_or_third_conversions'])
            self.assertLessEqual(abs(z['ppd_home_delta']),.1)
            a=simulate_build16(1,2,h,False,runs=350,seed=22,chain_research=mode)
            b=simulate_build16(1,2,h,False,runs=350,seed=22,chain_research=mode)
            self.assertEqual(a,b)
            self.assertEqual(a['simulation']['simulations'],350)
            self.assertIn('chain_research',a)
            self.assertTrue(math.isfinite(a['mean_margin_home']))
        self.assertEqual(h,before)
        frozen=simulate_build16(1,2,h,False,runs=350,seed=22)
        self.assertNotIn('chain_research',frozen)
        self.assertEqual(frozen,simulate_build16(1,2,h,False,runs=350,seed=22))
        hp=project(h[1],h[2],False,'home',False)
        ap=project(h[2],h[1],False,'away',False)
        self.assertEqual(simulate(hp,ap,runs=350,seed=22),
                         simulate(project(h[1],h[2],False,'home',False),
                                  project(h[2],h[1],False,'away',False),
                                  runs=350,seed=22))
        for r in h[1]:del r['chain_off_sustained_drives']
        missing=simulate_build16(1,2,h,False,runs=350,seed=22,chain_research='combined')
        self.assertEqual(missing['status'],'UNAVAILABLE_CHAIN_RESEARCH_EVIDENCE')
        self.assertIsNone(missing['simulation'])
        with self.assertRaises(ValueError):experiment(1,2,h,'fit_on_actual_margin')
        with self.assertRaises(ValueError):
            simulate_build16(1,2,h,False,runs=350,seed=22,
                             chain_research='combined',rush_research='mix')

    def test_exact_5000_contract_of_study(self):
        from cfb.chain_study import study,paired_score
        with self.assertRaises(ValueError):study('/missing','/missing',runs=4500)
        self.assertEqual(paired_score([]),{'paired_games':0})

if __name__=='__main__':unittest.main()
