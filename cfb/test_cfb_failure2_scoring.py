"""Failure 2: strict source reconstruction and causal turnover-return simulation."""
import copy
import unittest
import pandas as pd
from cfb.scoring_attribution import attribute
from cfb.mechanism_build16 import _failure2_verified_scoring_histories
from cfb.return_scoring import scoring_rates
from cfb.game_script import simulate_script
from cfb.test_cfb_engine import P,Q
from cfb.test_cfb_priority3 import history

def scoring_sample():
    # 10: offensive touchdown, 20: pick-six, then 10: punt-return TD.
    return pd.DataFrame({'game_play_number':[1,2,3,4],
        'pos_team_id':[10,10,10,10],
        'offense_score_play':[False,True,False,False],
        'pass':[False,True,True,False],
        'rush':[False,False,False,False],
        'int':[False,False,True,False],
        'pos_fumble_lost':[False,False,False,False],
        'home_score':[0,7,7,14],
        'away_score':[0,0,7,7]})

def team_row():
    return {'points':21.,'conceded':14.,
        'offensive_points':7.,'offensive_points_conceded':7.,
        'nonoffensive_points':7.,'defensive_return_points':7.,
        'defensive_return_td_count':1,'defensive_return_td_allowed':0,
        'special_teams_points':0.,'special_teams_td_count':0,
        'opponent_turnovers_observed':2,'interceptions':2,'fumbles_lost':0,
        'scoring_attribution':{'status':'VERIFIED_RAW_SCORE_RECONCILIATION'}}

def rates_fixture():
    return {'mode':'turnover_linked_defensive_returns_and_separate_special_teams',
        'home':[{'defensive_return_tds':1,'defensive_return_tds_allowed':0,
                 'defensive_return_points':7.,'special_teams_tds':0,
                 'special_teams_points':0.,'offense_turnovers':2,
                 'defense_opportunities':2} for _ in range(4)],
        'away':[{'defensive_return_tds':0,'defensive_return_tds_allowed':1,
                 'defensive_return_points':0.,'special_teams_tds':1,
                 'special_teams_points':7.,'offense_turnovers':2,
                 'defense_opportunities':2} for _ in range(4)]}

class Failure2Tests(unittest.TestCase):
    def test_offensive_defensive_special_are_distinct_and_reconciled(self):
        v=attribute(scoring_sample(),10,20,14,7)
        self.assertEqual(v['status'],'VERIFIED_RAW_SCORE_RECONCILIATION')
        self.assertEqual(v['offensive_points']['home'],7)
        self.assertEqual(v['defensive_return_points']['away'],7)
        self.assertEqual(v['defensive_return_tds']['away'],1)
        self.assertEqual(v['special_teams_points']['home'],7)
        self.assertEqual(v['special_teams_tds']['home'],1)
        self.assertEqual(v['nonoffensive_points']['home'],7)

    def test_scoring_misclassification_refused(self):
        bad=scoring_sample()
        bad.loc[2,'int']=False  # scrimmage TD by other team with no observed TO
        self.assertEqual(attribute(bad,10,20,14,7)['status'],'UNVERIFIED_SCORE_ATTRIBUTION')

    def test_bad_final_rejected(self):
        self.assertEqual(attribute(scoring_sample(),10,20,14,8)['status'],
                         'UNVERIFIED_SCORE_ATTRIBUTION')

    def test_missing_raw_score_cannot_fabricate(self):
        self.assertEqual(attribute(pd.DataFrame({'game_play_number':[1]}),10,20,7,0)['reason'],
                         'NO_CUMULATIVE_SCORE_COLUMNS')

    def test_immutable_offense_cleaning_and_verified_categories(self):
        rows={t:[copy.deepcopy(team_row()) for _ in range(4)] for t in (10,20)}
        updated,samples,meta=_failure2_verified_scoring_histories(10,20,rows)
        self.assertEqual(meta['status'],'ACTIVE_LINKED_DEFENSE_AND_SPECIAL_TEAMS_SCORING')
        self.assertEqual(updated[10][0]['points'],7)
        self.assertEqual(rows[10][0]['points'],21)
        self.assertEqual(samples['home'][0]['defensive_return_tds'],1)

    def test_missing_verified_games_keeps_legacy_scoring(self):
        rows={10:[{'points':14}]*3,20:[{'points':7}]*3}
        out,samples,meta=_failure2_verified_scoring_histories(10,20,rows)
        self.assertIs(out,rows)
        self.assertIsNone(samples)
        self.assertEqual(meta['status'],'UNVERIFIED_SCORING_ATTRIBUTION_LEGACY_BUILD16')

    def test_turnover_conditional_td_and_st_are_separate_and_deterministic(self):
        source=rates_fixture()
        rates=scoring_rates(source)
        self.assertGreater(rates['home_turnover_td_allowed_probability'],0)
        self.assertGreater(rates['away_special_td_per_game'],0)
        mech={'home':{},'away':{},'turnover_field':{'status':'UNAVAILABLE'},
              'nonoffensive_scoring':source}
        x=simulate_script(P,Q,history(),history(),runs=5000,seed=18,
            mechanisms=mech,allow_overtime=False)
        y=simulate_script(P,Q,history(),history(),runs=5000,seed=18,
            mechanisms=mech,allow_overtime=False)
        self.assertEqual(x,y)
        self.assertEqual(x['defensive_return_tds_counted'],
                         x['suppressed_shortfield_rewards_counted'])
        self.assertGreater(x['defensive_return_tds_counted'],0)
        self.assertGreater(x['simulated_special_teams_points_away'],0)
        for side in ('home','away'):
            self.assertAlmostEqual(x['simulated_nonoffensive_points_'+side],
                x['simulated_defensive_return_points_'+side]+
                x['simulated_special_teams_points_'+side],delta=.002)
        self.assertLessEqual(x['simulated_defensive_return_tds_away'],
                             x['simulated_turnovers_home']+0.0001)
        self.assertLessEqual(x['simulated_defensive_return_tds_home'],
                             x['simulated_turnovers_away']+0.0001)

if __name__=='__main__': unittest.main()
