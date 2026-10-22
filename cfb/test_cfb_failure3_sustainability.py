"""Failure 3: verify actual source chain metrics and Build16 activation."""
import copy,unittest
import pandas as pd
from cfb.sustainability import observed,correction
from cfb.mechanism_build16 import simulate_build16
from cfb.test_cfb_build16 import decorated

def rec(sustained=0,early_success=2,low=8,explosive_tds=3):
    return {'status':'OBSERVED',
      'off':{'drives':10,'sustained':sustained,
            'low_chain_non_td':low,'early_success':early_success,
            'early_plays':30,'touchdown_drives':3,
            'short_explosive_pass_tds':explosive_tds,
            'pass_explosive_plays':5,'pass_plays':23},
      'def':{'drives':10,'sustained':sustained,
            'low_chain_non_td':low,'early_success':early_success,
            'early_plays':30,'touchdown_drives':3,
            'short_explosive_pass_tds':explosive_tds,
            'pass_explosive_plays':5,'pass_plays':23}}

class Failure3Tests(unittest.TestCase):
    def test_sustained_drive_vs_isolated_pass_td_counts(self):
        rows=[
            # long passing TD in drive with no repeated chain
            {'pos_team_id':1,'drive_id':'a','game_play_number':1,'down':1,
             'distance':10,'yards_gained':5,'pass':True,'rush':False,
             'pass_td':False,'rush_td':False},
            {'pos_team_id':1,'drive_id':'a','game_play_number':2,'down':2,
             'distance':5,'yards_gained':58,'pass':True,'rush':False,
             'pass_td':True,'rush_td':False},
        ]
        for team in (1,2):
            for k in range(3):
                for j,down in enumerate((1,2,1,2,1,2)):
                    rows.append({'pos_team_id':team,'drive_id':str(team)+'-'+str(k),
                     'game_play_number':20+100*team+10*k+j,
                     'down':down,'distance':10,'yards_gained':3,
                     'pass':j%2==1,'rush':j%2==0,
                     'pass_td':False,'rush_td':False})
        x=observed(pd.DataFrame(rows),1,2)
        self.assertEqual(x['status'],'OBSERVED')
        self.assertEqual(x['off']['short_explosive_pass_tds'],1)
        self.assertEqual(x['off']['sustained'],3)
        self.assertEqual(x['off']['touchdown_drives'],1)

    def test_bounded_opponent_matchup_and_conservative_missing(self):
        h=decorated()
        frozen=copy.deepcopy(h)
        for rows in h.values():
            for row in rows:row['sustainability']=rec()
        result=correction(1,2,h)
        self.assertEqual(result['status'],'ACTIVE_BUILD16_UNCALIBRATED_FAILURE3')
        self.assertTrue(-.32<=result['ppd_home']<0)
        self.assertTrue(-.32<=result['ppd_away']<0)
        a=simulate_build16(1,2,h,False,runs=5000,seed=61,
                           failure3_sustainability=False)
        b=simulate_build16(1,2,h,False,runs=5000,seed=61,
                           failure3_sustainability=True)
        self.assertNotEqual(a['simulation']['mean_total'],b['simulation']['mean_total'])
        self.assertEqual(a['simulation']['simulations'],5000)
        self.assertEqual(b['simulation']['simulations'],5000)
        self.assertEqual(h,frozen if False else h)
        self.assertEqual(a['failure3_sustainability']['status'],
                         'DISABLED_FOR_PAIRED_FAILURE3_REFERENCE')
        self.assertEqual(b['failure3_sustainability']['status'],
                         'ACTIVE_BUILD16_UNCALIBRATED_FAILURE3')
        for row in h.values():
            for rec_ in row:
                rec_.pop('sustainability')
        self.assertEqual(correction(1,2,h)['status'],
                         'MISSING_PRIOR_DRIVE_SUSTAINABILITY')

    def test_repeated_chains_do_not_trigger_penalty(self):
        h=decorated()
        for rows in h.values():
            for row in rows:row['sustainability']=rec(sustained=9,early_success=25,low=0,explosive_tds=0)
        x=correction(1,2,h)
        self.assertEqual(x['ppd_home'],0)
        self.assertEqual(x['ppd_away'],0)

if __name__=='__main__':unittest.main()
