"""Forward grading only uses immutable frozen Build16 and negative-run shadow."""
import unittest
from unittest.mock import patch
import pandas as pd
from cfb.forward_monitor import grade

def future():
    return {'401800001':{
      'game_id':'401800001','home':'Home Team','away':'Away Team',
      'home_id':'1','away_id':'2','kickoff_utc':'2026-10-23T20:00:00+00:00',
      'prediction_generated_at_utc':'2026-10-22T16:00:00+00:00',
      'source_snapshot_manifest_sha256':'a'*64,'simulated_margin_home':3.0,
      'original_forecast_run_id':'111','original_forecast_zip_sha256':'b'*64,
      'build16_challenger':{
          'status':'AVAILABLE_UNFITTED_BUILD16_RESEARCH_ONLY',
          'version':'Build16','mean_margin_home':5.0,
          'margin_p10_p50_p90':[-13,5,23]},
      'failure1_negative_run_shadow':{
          'status':'FROZEN_NEGATIVE_RUN_SHADOW_5000',
          'protocol':'cfb-failure1-frozen-negative-designed-runs-v1',
          'mode':'negative','runs':5000,'projected_margin_home':6.0,
          'source_manifest_sha256':'a'*64,'candidate_source_sha256':'c'*64}
    }}

def schedule():
    return pd.DataFrame([{'game_id':401800001,'completed':True,
      'start_date':'2026-10-23T20:00:00+00:00','home_id':1,'away_id':2,
      'home_team':'Home Team','away_team':'Away Team',
      'home_points':30,'away_points':24}])

def quote():
    return {'401800001':{'home':'Home Team','away':'Away Team',
         'kickoff_utc':'2026-10-23T20:00:00+00:00',
         'observed_at_utc':'2026-10-23T15:00:00+00:00',
         'market_archive_run_id':'112','market_archive_sha256':'d'*64,
         'raw_source_sha256':'e'*64,
         'quote':{'book':'Public ESPN quote',
                 'spread_home':-4.5,'home_odds_american':-110,
                 'away_odds_american':-110}}}

class ForwardTests(unittest.TestCase):
    def rungrade(self,odds):
        with patch('cfb.forward_monitor.load_all_archives',return_value=future()),patch(
             'cfb.forward_monitor.load_quotes',return_value=odds):
            return grade('unused','unused',schedule(),
               '2026-10-24T20:00:00+00:00','https://example.org/final','f'*64)

    def test_same_game_margin_frozen_only(self):
        x=self.rungrade({})
        z=x['failure1_negative_run_shadow_forward']
        self.assertEqual(z['paired_margin_games'],1)
        self.assertEqual(z['mean_build16_absolute_error'],1.0)
        self.assertEqual(z['mean_negative_shadow_absolute_error'],0.0)
        self.assertEqual(z['mae_gain_vs_build16'],1.0)
        self.assertEqual(z['quoted_games'],0)
        self.assertFalse(z['ats_calibrated'])
        self.assertFalse(x['scored_games'][0]['failure1_negative_shadow']['recomputed_after_kickoff'])

    def test_same_pre_kickoff_line_graded_for_shadow_and_build16(self):
        x=self.rungrade(quote())
        z=x['scored_games'][0]
        self.assertEqual(z['failure1_negative_shadow']['observed_signed_spread_home'],-4.5)
        self.assertEqual(z['failure1_negative_shadow']['ats_research_side_at_capture'],'HOME')
        self.assertTrue(z['failure1_negative_shadow']['ats_correct_excluding_push_noedge'])
        self.assertEqual(x['failure1_negative_run_shadow_forward']['quoted_games'],1)

    def test_reject_shadow_source_manifest_mismatch(self):
        f=future()
        f['401800001']['failure1_negative_run_shadow']['source_manifest_sha256']='0'*64
        with patch('cfb.forward_monitor.load_all_archives',return_value=f),patch(
             'cfb.forward_monitor.load_quotes',return_value={}):
            with self.assertRaisesRegex(ValueError,'source manifest'):
                grade('unused','unused',schedule(),
                      '2026-10-24T20:00:00+00:00','https://example.org/final','f'*64)

if __name__=='__main__':unittest.main()
