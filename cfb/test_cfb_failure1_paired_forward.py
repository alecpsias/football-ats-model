"""Immutable first-eligible same-snapshot rushing shadow grading tests."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from cfb.archive import _zipwrite,digest
from cfb.failure1_paired_forward import load_first_frozen_pairs,evaluate

GAME='401800001';KICK='2026-10-23T20:00:00+00:00'
SOURCE='a'*64
def projection(run,generated,learned):
    x={'schema':'cfb_prospective_research_forecast_v1','game_id':GAME,
       'home':'Home Team','away':'Away Team','home_id':'1','away_id':'2',
       'kickoff_utc':KICK,'prediction_generated_at_utc':generated,
       'source_snapshot_at_utc':'2026-10-22T11:50:00+00:00',
       'source_snapshot_manifest_sha256':SOURCE,
       'github_run_id':str(run),
       'market_quote':None,'ats_prediction':None,
       'status':'FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE',
       'simulated_margin_home':2.0}
    if learned:
        x['build16_challenger']={
         'status':'AVAILABLE_UNFITTED_BUILD16_RESEARCH_ONLY',
         'mean_margin_home':1.0,'margin_p10_p50_p90':[-16,1,18],
         'simulation':{'simulations':5000,'seed':int(GAME)%2**32,
             'home_win_probability_raw':.54}}
        x['failure1_learned_rush_shadow']={
         'status':'FROZEN_2023_FITTED_RUSH_SHADOW_5000',
         'protocol':'cfb-failure1-learned-rush-shadow-v1',
         'runs':5000,'fitted_year':2023,'seed':int(GAME)%2**32,
         'source_manifest_sha256':SOURCE,
         'frozen_fitted_model_sha256':'b'*64,
         'projected_margin_home':4.,'margin_delta_points':2.,
         'margin_p10_p50_p90':[-15,4,21],
         'win_probability_home_raw':.56}
        x['mandatory_rushing_audit']={'requires_full_rush_stress':True}
    return x

def pack(folder,run,produced,record):
    root=Path(folder);root.mkdir(exist_ok=True,parents=True)
    raw=(json.dumps(record,sort_keys=True)+'\n').encode()
    index={'status':'CFB_PROSPECTIVE_RESEARCH_RECORDS_FROZEN',
       'github_run_id':str(run),
       'generated_at_utc':produced,
       'snapshot_at_utc':'2026-10-22T11:50:00+00:00',
       'game_records':1,
       'predictions':[{'game_id':GAME,'file':GAME+'.json',
           'sha256':digest(raw),'status':record['status']}]}
    with zipfile.ZipFile(root/(str(run)+'.zip'),'x') as z:
        _zipwrite(z,'index.json',(json.dumps(index)+'\n').encode())
        _zipwrite(z,GAME+'.json',raw)

def sched(done=True):
    return pd.DataFrame([{'game_id':int(GAME),'completed':done,
       'start_date':KICK,'home_id':1,'away_id':2,'home_team':'Home Team',
       'away_team':'Away Team','home_points':27 if done else None,
       'away_points':24 if done else None}])

def prices(time='2026-10-23T15:00:00+00:00'):
    return {GAME:{'home':'Home Team','away':'Away Team','kickoff_utc':KICK,
     'observed_at_utc':time,'market_archive_sha256':'c'*64,
     'quote':{'book':'ESPN public','spread_home':-2.5,
              'home_odds_american':-110,'away_odds_american':-110}}}

class PairTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.forecasts=self.root/'forecast'
        self.odds=self.root/'odds'
        self.odds.mkdir()
        pack(self.forecasts,100001,'2026-10-22T12:00:00+00:00',
             projection(100001,'2026-10-22T12:00:00+00:00',False))
        pack(self.forecasts,100002,'2026-10-22T13:00:00+00:00',
             projection(100002,'2026-10-22T13:00:00+00:00',True))
    def tearDown(self):self.tmp.cleanup()

    def report(self,completed=True,quote=True):
        with patch('cfb.failure1_paired_forward.load_quotes',
            return_value=prices() if quote else {}):
            return evaluate(self.forecasts,self.odds,sched(completed),
                '2026-10-24T18:00:00+00:00','https://example.com/free-schedule','d'*64)

    def test_pair_is_first_eligible_not_first_original(self):
        x,counts=load_first_frozen_pairs(self.forecasts)
        self.assertEqual(len(x),1)
        self.assertEqual(x[GAME]['paired_forecast_archive_run_id'],'100002')
        self.assertEqual(counts['archive_game_records'],2)
        self.assertEqual(counts['historical_without_fitted_shadow'],1)
        self.assertEqual(counts['distinct_first_paired_games'],1)

    def test_truth_and_single_quote_comparison_not_hindsight(self):
        r=self.report()
        self.assertEqual(r['qualified_completed_paired_games'],1)
        self.assertEqual(r['first_paired_not_first_ever_forecast'],1)
        x=r['scored_pairs'][0]
        self.assertEqual(x['first_original_forecast_run_id'],'100001')
        self.assertEqual(x['first_eligible_paired_run_id'],'100002')
        self.assertEqual(x['actual_home_margin'],3.)
        self.assertEqual(r['margin']['reference_mae'],2.)
        self.assertEqual(r['margin']['learned_mae'],1.)
        self.assertEqual(r['margin']['mae_gain_reference_minus_learned'],1.)
        self.assertEqual(r['ATS_at_joint_observed_spread']['different_ATS_sides'],1)
        self.assertEqual(r['ATS_at_joint_observed_spread']['reference']['wrong'],1)
        self.assertEqual(r['ATS_at_joint_observed_spread']['learned']['correct'],1)
        self.assertFalse(r['ATS_calibrated'])

    def test_missing_finals_and_missing_odds_are_not_wins(self):
        a=self.report(completed=False)
        self.assertEqual(a['qualified_completed_paired_games'],0)
        self.assertEqual(a['margin']['reference_mae'],None)
        self.assertEqual(a['ATS_at_joint_observed_spread']['fair_paired_decisive_spreads'],0)
        b=self.report(quote=False)
        self.assertEqual(b['qualified_completed_paired_games'],1)
        self.assertEqual(b['ATS_at_joint_observed_spread']['fair_paired_decisive_spreads'],0)

    def test_after_kickoff_quote_fails_eligibility(self):
        with patch('cfb.failure1_paired_forward.load_quotes',
            return_value=prices('2026-10-24T10:00:00+00:00')):
            x=evaluate(self.forecasts,self.odds,sched(),
             '2026-10-24T18:00:00+00:00','https://example.com/schedule','d'*64)
        self.assertEqual(x['ATS_at_joint_observed_spread']['fair_paired_decisive_spreads'],0)

    def test_concurrently_activated_ridge_requires_distinct_uncorrected_reference(self):
        pair=projection(100005,'2026-10-22T15:00:00+00:00',True)
        pair['build16_active_failure1_rushing_correction']={
            'status':'APPLIED_IN_BUILD16_CHALLENGER','margin_delta_points':2.}
        pair['build16_challenger']['mean_margin_home']=4.0
        source=self.root/'activated_ridge_only'
        pack(source,100005,'2026-10-22T15:00:00+00:00',pair)
        invalid,counts=load_first_frozen_pairs(source)
        self.assertEqual(invalid,{})
        self.assertEqual(counts['excluded_active_ridge_without_raw_reference'],1)
        pair['build16_uncorrected_reference']={
            'status':'AVAILABLE_UNFITTED_BUILD16_RESEARCH_ONLY',
            'mean_margin_home':1.0,'margin_p10_p50_p90':[-16,1,18],
            'simulation':{'simulations':5000,'seed':int(GAME)%2**32,
                          'home_win_probability_raw':.54}}
        eligible=self.root/'activated_ridge_with_baseline'
        pack(eligible,100006,'2026-10-22T15:00:00+00:00',pair)
        valid,counts=load_first_frozen_pairs(eligible)
        self.assertEqual(len(valid),1)
        self.assertEqual(valid[GAME]['build16_uncorrected_reference']['mean_margin_home'],1.0)
        self.assertEqual(valid[GAME]['build16_challenger']['mean_margin_home'],4.0)

    def test_mismatched_simulation_seed_is_rejected(self):
        p=projection(100003,'2026-10-22T15:00:00+00:00',True)
        p['failure1_learned_rush_shadow']['seed']=1
        pack(self.forecasts,100003,'2026-10-22T15:00:00+00:00',p)
        with self.assertRaisesRegex(ValueError,'random seed'):
            load_first_frozen_pairs(self.forecasts)

if __name__=='__main__':unittest.main()
