"""Source-backed explosive rushing touchdown routes, with no phantom points."""
import copy
import unittest

import pandas as pd

from cfb.sustainability import observed, correction
from cfb.drive_sustainability_process import parameters
from cfb.empirical_drive_clock import observe as observed_clock
from cfb.test_cfb_empirical_drive_clock import complete_game
from cfb.test_cfb_failure3_sustainability import rec
from cfb.test_cfb_build16 import decorated
from cfb.mechanism_build16 import simulate_build16


def rushing_explosive_plays():
    rows=[]
    k=0
    for team in (1,2):
        for d in range(4):
            rush_td = d==0
            rows.append(dict(pos_team_id=team,drive_id=f'{team}-{d}',
                game_play_number=k+1, down=1,distance=10,
                yards_gained=68 if rush_td else 3,
                rush=True,rush_td=rush_td,pass_td=False,**{'pass':False}))
            k+=1
    return pd.DataFrame(rows)


def build_qualified():
    h=decorated()
    base=rec(explosive_tds=0)
    sample={'status':'SOURCE_CLOCK_VERIFIED',
        'clock_verification':'ESPN_ELAPSED_AND_ENDPOINTS',
        'off':{'explosive_td':[40,53,80],
               'chain_score':[172,194,201],
               'turnover':[65,70,90], 'stalled':[111,137,159]},
        'def':{'explosive_td':[40,53,80],
               'chain_score':[172,194,201],
               'turnover':[65,70,90], 'stalled':[111,137,159]}}
    for history in h.values():
        for row in history:
            row['sustainability']=copy.deepcopy(base)
            row['drive_clock']=copy.deepcopy(sample)
    return h


class ExplosiveRushingTDRegression(unittest.TestCase):
    def test_source_observed_long_rush_td_is_explosive_route(self):
        x=observed(rushing_explosive_plays(),1,2)
        self.assertEqual(x['status'],'OBSERVED')
        self.assertEqual(x['off']['short_explosive_rush_tds'],1)
        self.assertEqual(x['off']['short_explosive_pass_tds'],0)
        self.assertEqual(x['off']['short_explosive_tds'],1)
        self.assertEqual(x['def']['short_explosive_rush_tds'],1)

    def test_source_clock_labels_rushing_td_as_explosive(self):
        raw=complete_game()
        raw.loc[0,'pass']=False
        raw.loc[0,'rush']=True
        raw.loc[0,'pass_td']=False
        raw.loc[0,'rush_td']=True
        clock=observed_clock(raw,raw,10,20,10)
        self.assertEqual(clock['status'],'SOURCE_CLOCK_VERIFIED')
        self.assertEqual(clock['joint_drives']['off'][0]['outcome'],'explosive_td')
        self.assertEqual(clock['joint_drives']['off'][0]['rush_plays'],1)

    def test_new_route_changes_5000_draws_only_if_source_qualified(self):
        h=build_qualified()
        old=simulate_build16(1,2,h,False,runs=5000,seed=309)
        for history in h.values():
            for row in history:
                for side in ('off','def'):
                    values=row['sustainability'][side]
                    values['short_explosive_rush_tds']=2
                    values['short_explosive_tds']=2
        observed_h=correction(1,2,h)
        self.assertEqual(observed_h['status'],'ACTIVE_BUILD16_UNCALIBRATED_FAILURE3')
        self.assertEqual(observed_h['home']['explosive_td_route_type'],'SOURCE_PASS_AND_RUSH')
        self.assertGreater(observed_h['home']['short_explosive_td_share_shrunk'],
                           observed_h['home']['short_explosive_pass_td_share_shrunk'])
        param=parameters(observed_h)
        self.assertAlmostEqual(param['home']['explosive_td_share'],
                               observed_h['home']['short_explosive_td_share_shrunk'])
        new=simulate_build16(1,2,h,False,runs=5000,seed=309)
        self.assertEqual(new,simulate_build16(1,2,h,False,runs=5000,seed=309))
        self.assertEqual(new['simulation']['simulations'],5000)
        self.assertNotEqual(new['simulation']['mean_total'],old['simulation']['mean_total'])
        self.assertTrue(new['not_primary'])
        self.assertFalse(new['ATS_calibrated'])

    def test_legacy_records_fail_closed_to_pass_only_route(self):
        h=build_qualified()
        x=correction(1,2,h)
        self.assertEqual(x['home']['explosive_td_route_type'],
                         'PASS_ONLY_LEGACY_MISSING_SOURCE_RUSH')
        self.assertEqual(x['home']['short_explosive_td_share_shrunk'],
                         x['home']['short_explosive_pass_td_share_shrunk'])


if __name__=='__main__':unittest.main()
