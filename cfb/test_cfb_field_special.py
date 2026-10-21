"""Field position / kick-event provenance, regression and scoring safety tests."""
import copy
import unittest

import pandas as pd

from cfb.field_special import observed_special_teams
from cfb.integrated_scoring import project_integrated
from cfb.test_cfb_integrated_build15 import histories


def raw_fixture():
    return pd.DataFrame([
        {'game_id':10,'game_play_number':1,'pos_team_id':1,
         'fg_attempt':True,'fg_made':True,'yds_fg':37,'fg_team':'1',
         'punt':False,'kickoff_play':False},
        {'game_id':10,'game_play_number':2,'pos_team_id':1,
         'fg_attempt':True,'fg_made':False,'yds_fg':51,'fg_team':'1',
         'punt':False,'kickoff_play':False},
        {'game_id':10,'game_play_number':3,'pos_team_id':2,
         'fg_attempt':True,'fg_made':True,'yds_fg':45,'fg_team':'2',
         'punt':False,'kickoff_play':False},
        {'game_id':10,'game_play_number':4,'pos_team_id':1,
         'punt':True,'punt_team':'1','yds_punt_gained':42,
         'fg_attempt':False,'kickoff_play':False},
        {'game_id':10,'game_play_number':5,'pos_team_id':2,
         'kickoff_play':True,'kick_return_team':'1','yds_kickoff_return':23,
         'fg_attempt':False,'punt':False},
        {'game_id':10,'game_play_number':6,'pos_team_id':1,
         'kickoff_play':True,'kick_return_team':'2','yds_kickoff_return':18,
         'fg_attempt':False,'punt':False},
    ])


def augmented():
    h=copy.deepcopy(histories())
    for rows in h.values():
        for r in rows:
            r['start_field_n']=11
            r['start_field_position_allowed']=29.
            r['start_field_allowed_n']=11
            r['special_teams']={
                'fg_made':1,'fg_attempts':2,
                'punt_net_sum':150.,'punt_net_n':4,
                'kick_return_sum':40.,'kick_return_n':2}
    return h


class FieldSpecialTests(unittest.TestCase):
    def test_only_verified_events_and_teams_count(self):
        x=observed_special_teams(raw_fixture(),1,2)
        self.assertEqual((x['fg_made'],x['fg_attempts']),(1,2))
        self.assertEqual((x['fg_short_made'],x['fg_short_attempts']),(1,1))
        self.assertEqual((x['fg_medium_made'],x['fg_medium_attempts']),(0,0))
        self.assertEqual((x['fg_long_made'],x['fg_long_attempts']),(0,1))
        self.assertEqual((x['punt_net_sum'],x['punt_net_n']),(42.,1))
        self.assertEqual((x['kick_return_sum'],x['kick_return_n']),(23.,1))

    def test_unattributed_fg_is_unknown_not_a_zero(self):
        raw=raw_fixture().drop(columns=['fg_team','pos_team_id'])
        x=observed_special_teams(raw,1,2)
        self.assertIsNone(x['fg_made'])
        self.assertIsNone(x['fg_attempts'])

    def test_no_kick_attempts_with_observed_flag_is_zero(self):
        raw=raw_fixture()
        raw=raw.loc[raw.fg_attempt.eq(False)]
        x=observed_special_teams(raw,1,2)
        self.assertEqual(x['fg_attempts'],0)
        self.assertEqual(x['fg_made'],0)

    def test_two_games_must_not_be_mixed(self):
        raw=raw_fixture()
        raw.loc[0,'game_id']=20
        self.assertIsNone(observed_special_teams(raw,1,2)['fg_attempts'])

    def test_opponent_allowed_field_impacts_ppd_with_reliability(self):
        base=augmented()
        low=project_integrated(1,2,base,False,'home')
        high=copy.deepcopy(base)
        for r in high[2]:
            r['start_field_position_allowed']=42.
        result=project_integrated(1,2,high,False,'home')
        self.assertGreater(result['projection']['_ppd'],low['projection']['_ppd'])
        self.assertLessEqual(result['components']['field_position_residual_ppd'],.060001)

    def test_fg_regression_is_not_reward_twice_for_makes(self):
        first=augmented()
        for r in first[1]:
            r['special_teams']['fg_attempts']=8
            r['special_teams']['fg_made']=8
        perfect=project_integrated(1,2,first,False,'home')
        poor=copy.deepcopy(first)
        for r in poor[1]:
            r['special_teams']['fg_made']=1
        missed=project_integrated(1,2,poor,False,'home')
        self.assertLess(perfect['components']['field_goal_regression_residual_ppd'],0)
        self.assertGreater(missed['components']['field_goal_regression_residual_ppd'],0)
        self.assertLessEqual(abs(perfect['components']['field_goal_regression_residual_ppd']),.055001)
        self.assertGreater(missed['projection']['_ppd'],perfect['projection']['_ppd'])

    def test_punting_and_returns_descriptive_not_extra_points(self):
        a=augmented();b=copy.deepcopy(a)
        for r in b[1]:
            r['special_teams']['punt_net_sum']=200.
            r['special_teams']['kick_return_sum']=90.
        p=project_integrated(1,2,a,False,'home')
        q=project_integrated(1,2,b,False,'home')
        self.assertEqual(p['projection']['_ppd'],q['projection']['_ppd'])
        self.assertNotEqual(p['components']['special_teams_observed'],
                            q['components']['special_teams_observed'])

    def test_missing_special_teams_keeps_no_implicit_zero(self):
        x=project_integrated(1,2,histories(),False,'home')
        self.assertIn('field_goals_unverified',x['missing_features'])
        self.assertIsNone(x['components']['special_teams_observed']['field_goals_made'])
        self.assertIn('opponent_matched_defensive_start_field_position',x['missing_features'])


if __name__=='__main__':
    unittest.main()
