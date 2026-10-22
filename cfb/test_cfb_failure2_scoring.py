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
        'away_score':[0,0,7,7],
        'type.text':['Kickoff','Passing Touchdown',
                     'Interception Return Touchdown','Punt Return Touchdown'],
        'text':['Kickoff','Pass for a touchdown',
                'Pass intercepted and returned for a touchdown',
                'Punt returned for a touchdown']})

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

    def test_unknown_kickoff_not_misclassified_as_special_teams(self):
        raw=scoring_sample()
        raw.loc[3,'type.text']='Kickoff'
        raw.loc[3,'text']='Touchback after a kickoff, not a punt return'
        out=attribute(raw,10,20,14,7)
        self.assertEqual(out['status'],'UNVERIFIED_SCORE_ATTRIBUTION')
        self.assertEqual(out['reason'],'SCORING_TYPE_UNVERIFIED_NOT_ASSUMED_SPECIAL_TEAMS')

    def test_authoritative_boise_scorebook_corrects_memphis_false_special_td(self):
        raw=pd.DataFrame({'game_id':[401860881],
                          'game_play_number':[170],'home_score':[38],
                          'away_score':[20], 'pos_team_id':[235],
                          'type.text':['Punt Return Touchdown'],
                          'text':['Cam Bates 72 yd punt return TOUCHDOWN']})
        out=attribute(raw,999,235,38,20)
        self.assertEqual(out['status'],'VERIFIED_RAW_SCORE_RECONCILIATION')
        self.assertEqual(out['verification_method'],
                         'INDEPENDENT_OFFICIAL_SCOREBOOK_EVENT_CROSSCHECK')
        self.assertEqual(out['special_teams_points']['home'],7)
        self.assertEqual(out['special_teams_points']['away'],0)
        self.assertEqual(out['offensive_points']['away'],20)

    def test_authoritative_uab_overtime_delayed_score_not_rejected(self):
        raw=pd.DataFrame({'game_id':[401862800],
                          'game_play_number':[192],'home_score':[30],
                          'away_score':[17],'pos_team_id':[151],
                          'type.text':['Interception Return Touchdown'],
                          'text':['Gulley interception return TOUCHDOWN']})
        out=attribute(raw,5,151,30,17)
        self.assertEqual(out['status'],'VERIFIED_RAW_SCORE_RECONCILIATION')
        self.assertEqual(out['defensive_return_points']['home'],6)
        self.assertEqual(out['offensive_points']['home'],24)
        self.assertEqual(out['offensive_points']['away'],17)
        self.assertEqual(out['nonoffensive_points']['away'],0)
        self.assertEqual(attribute(raw,5,151,31,17)['status'],
                         'UNVERIFIED_SCORE_ATTRIBUTION')

    def test_partial_scoring_history_keeps_verified_and_drops_unknown_ppd(self):
        rows={t:[copy.deepcopy(team_row()) for _ in range(4)] for t in (10,20)}
        rows[10][-1]['scoring_attribution']={
            'status':'UNVERIFIED_SCORE_ATTRIBUTION','reason':'AMBIGUOUS'}
        rows[10][-1]['offensive_points']=None
        original=copy.deepcopy(rows)
        updated,samples,audit=_failure2_verified_scoring_histories(10,20,rows)
        self.assertEqual(audit['status'],'ACTIVE_PARTIAL_VERIFIED_SCORING_ONLY')
        self.assertEqual((audit['verified_home_games'],audit['unverified_home_games']),
                         (3,1))
        self.assertEqual(len(updated[10]),4)
        self.assertFalse(updated[10][-1]['scoring_ppd_eligible'])
        self.assertTrue(updated[10][0]['scoring_ppd_eligible'])
        self.assertEqual(updated[10][-1]['points'],original[10][-1]['points'])
        self.assertEqual(len(samples['home']),3)
        self.assertEqual(rows,original)

    def test_ppd_mask_does_not_count_unverified_high_scoring_game(self):
        from cfb.research_backtest import project
        def row(points,valid):
            return {'points':float(points),'conceded':21.,'drives':10,
                    'faced':10,'interceptions':1,'fumbles_lost':0,
                    'scoring_ppd_eligible':valid}
        a=[row(10,True),row(10,True),row(10,True),row(100,False)]
        b=[row(14,True),row(14,True),row(14,True),row(14,False)]
        before=project(a,b,True,'home',False)
        a[-1]['scoring_ppd_eligible']=True
        after=project(a,b,True,'home',False)
        self.assertLess(before['_ppd'],after['_ppd'])
        self.assertEqual(before['expected_drives'],after['expected_drives'])
        self.assertAlmostEqual(before['_to'],after['_to'])

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
