"""v0.9 invariants: no target leakage; no unearned ATS calibration."""
import json, tempfile, unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from model import MODEL_RULES,InputError
from mechanism_research import extract_mechanisms,matchup_third_down,turnover_following_field_bonus,_passrate_for_state
from evaluate_locked_forecasts import evaluate

CUTOFF='2026-10-18T18:00:00Z'

def fixture(include_target=False):
    dates=pd.date_range('2026-09-01',periods=8,freq='1D',tz='UTC')
    teams=('GB','DAL');schedule=[];rows=[]
    for g,date in enumerate(dates):
        gid=f'2026_{g+1:02d}_DAL_GB'
        schedule.append({'game_id':gid,'home_team':'GB','away_team':'DAL','kickoff_utc':date.isoformat()})
        for idx in range(60):
            possession=idx//5;off=teams[possession%2];opp=teams[1-possession%2]
            state=-12 if idx%4==0 else 0 if idx%4!=1 else 12
            row={'game_id':gid,'play_id':float(idx+1),'posteam':off,'defteam':opp,
             'fixed_drive':float(possession),'fixed_drive_result':'Turnover' if idx%10 in (4,9) else 'Punt', 'play_type':'pass' if idx%3 else 'run',
             'down':3. if idx%5==2 else 1.,'ydstogo':2 if idx%3==0 else 5 if idx%3==1 else 10,
             'third_down_converted':float(idx%4==0),'first_down_penalty':float(idx%17==0),
             'yardline_100':(20. if idx%10==4 else 78. if idx%10==9 else 72. if idx%5==0 else 58.),
             'interception':float(idx%10==4),'fumble_lost':float(idx%10==9),
             'score_differential':state,'qb_dropback':float(idx%3!=0),
             'qb_hit':float(idx%8==1),'sack':float(idx%19==1),'penalty':float(idx%17==0),
             'penalty_team':opp,'penalty_yards':10.,
             'fourth_down_converted':0.,'fourth_down_failed':0.}
            rows.append(row)
    if include_target:
        schedule.append({'game_id':'2026_06_TEST','home_team':'GB','away_team':'DAL','kickoff_utc':'2026-10-18T19:00:00+00:00'})
        for i in range(300):
            r=dict(rows[0]);r.update({'game_id':'2026_06_TEST','play_id':i,'interception':1.,'fumble_lost':0.,'qb_hit':1.,'third_down_converted':1.});rows.append(r)
    return pd.DataFrame(rows),pd.DataFrame(schedule)


def projections():
    def p(team):return {'team':team,'pass_rate':.60,'plays_per_drive':6.,'expected_drives':10.5,
        'interception_rate_per_dropback':.025,'fumble_lost_rate_per_play':.006,
        '_ppd':2.1,'_td':.59,'_to':.11,'third_down_conversion':.40}
    return p('GB'),p('DAL')

class MechanismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pbp,cls.sched=fixture()
        cls.profile=extract_mechanisms(cls.pbp,cls.sched,CUTOFF,('GB','DAL'))

    def test_target_game_never_enters_extracted_mechanisms(self):
        df,sched=fixture(True)
        other=extract_mechanisms(df,sched,CUTOFF,('GB','DAL'))
        self.assertEqual(other['league'],self.profile['league'])
        self.assertEqual(other['eligible_play_count'],self.profile['eligible_play_count'])
        self.assertEqual(other['next_start_own_yardline_after_takeaway_by_prior_offense_zone'],
                         self.profile['next_start_own_yardline_after_takeaway_by_prior_offense_zone'])

    def test_insufficient_target_sample_blocks_not_invents(self):
        df,sched=fixture()
        df=df.loc[df.game_id.isin(sched.game_id[:2])]
        with self.assertRaises(InputError):extract_mechanisms(df,sched,CUTOFF,('GB','DAL'))

    def test_third_down_distance_probabilities_are_valid_and_ordered(self):
        result=matchup_third_down(self.profile,'GB','DAL')
        self.assertTrue(0<result['weighted_conversion_rate']<1)
        self.assertAlmostEqual(sum(result['distance_mix'].values()),1,delta=.001)
        self.assertTrue(0<result['short_1_2']['rate']<1)

    def test_actual_turnover_location_drives_next_field_bonus(self):
        p=self.profile.copy()
        p['next_start_own_yardline_after_takeaway_by_prior_offense_zone']={
            'opponent_40_or_closer':[17.], 'outside_opponent_40':[72.]}
        r=MODEL_RULES['NFL'];rng=np.random.default_rng(3)
        near,startnear=turnover_following_field_bonus(p,15,r,rng)
        far,startfar=turnover_following_field_bonus(p,80,r,rng)
        self.assertEqual(startnear,17)
        self.assertEqual(startfar,72)
        self.assertLess(near,far)

    def test_script_pass_share_not_a_fixed_constant(self):
        p=self.profile
        for state in ('trailing_9_plus','leading_9_plus','within_8'):
            self.assertTrue(0.18<=_passrate_for_state(p,'GB',state,.6)<=.90)

    def test_retired_legacy_runner_is_not_an_active_model(self):
        import mechanism_research
        self.assertFalse(hasattr(mechanism_research,'candidate_simulation'))

    def test_scoring_opportunity_diagnostics_recorded_without_assumed_epa(self):
        self.assertIn('scoring_opportunities',self.profile['offense']['GB'])
        self.assertIn('scoring_opportunity_turnovers',self.profile['league'])
        self.assertNotIn('scoring_opportunity_epa',self.profile['league'])

    def test_penalty_auto_first_down_not_mislabeled_epa(self):
        self.assertIn('penalty_auto_first_down_conceded',self.profile['defense']['DAL'])
        self.assertNotIn('penalty_epa',self.profile['defense']['DAL'])
        self.assertIn('QB hits are NOT PFR pressure or all disruptions',self.profile['limits'])

    def test_future_eval_does_not_backfit_or_claim_ats(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'2026_game.json'
            data={'status':'EXPLORATORY_CORE_PROJECTION_UNCALIBRATED','game_id':'2026_game',
              'home':'GB','away':'DAL','kickoff_utc':'2026-10-18T20:00:00Z',
              'data_as_of_utc':'2026-10-18T17:00:00Z', 'score_home':24,'score_away':20,
              'projected_home_margin':4,'mechanism_research':{'candidate':{
               'mean_margin_home':2,'score_home':25,'score_away':23,'promoted_to_primary':False}}}
            path.write_text(json.dumps(data))
            s=pd.DataFrame([dict(game_id='2026_game',home_team='GB',away_team='DAL',
                                   home_score=23,away_score=13,result=10,gameday='2026-10-18',gametime='16:00')])
            out=evaluate([path],s)
            self.assertEqual(out['completed_games'],1)
            self.assertEqual(out['all_baseline']['baseline_mae_margin'],6)
            self.assertEqual(out['archived_challenger_versions']['0.9.0']['challenger_mae_margin'],8)
            self.assertEqual(out['current_challenger']['challenger_games'],0)
            self.assertEqual(out['ats_record'],'NOT_MEASURED_NO_VALIDATED_PRICED_SIGNED_LINES')
            data['data_as_of_utc']='2026-10-18T19:00:01Z'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(InputError,'After-lock'):evaluate([path],s)

if __name__=='__main__':unittest.main()
