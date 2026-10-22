"""Frozen 2023 rushing response is one within-Build16 research shadow, never active."""
import copy,unittest
from cfb.test_cfb_severity_research import decorated
from cfb.failure1_learned_shadow import shadow
from cfb.mechanism_build16 import simulate_build16
from cfb.test_cfb_failure1_forward import future,quote,schedule
from cfb.forward_monitor import grade
from unittest.mock import patch

class LearnedForwardTests(unittest.TestCase):
    def test_genuine_5000_run_candidate_and_existing_default_identical(self):
        h=decorated();prior=copy.deepcopy(h)
        x=shadow(1,2,h,False,runs=5000,seed=7,
                 source_manifest_sha256='1'*64)
        self.assertEqual(x['status'],'FROZEN_2023_FITTED_RUSH_SHADOW_5000')
        self.assertEqual(x['runs'],5000)
        self.assertEqual(x['fitted_year'],2023)
        self.assertEqual(len(x['frozen_fitted_model_sha256']),64)
        self.assertEqual(h,prior)
        self.assertEqual(simulate_build16(1,2,h,False,runs=250,seed=7),
                         simulate_build16(1,2,h,False,runs=250,seed=7))
        with self.assertRaises(ValueError):
            shadow(1,2,h,False,runs=4500,seed=7,source_manifest_sha256='1'*64)

    def test_nonexistent_historical_quality_never_adds_rush_margin(self):
        x=shadow(1,2,{},False,runs=5000,seed=8,
                 source_manifest_sha256='2'*64)
        self.assertEqual(x['status'],'UNAVAILABLE_PREKICKOFF_RUSH_EVIDENCE')
        self.assertIsNone(x['margin_delta_points'])
        self.assertNotIn('projected_margin_home',x)

    def test_postgame_grades_original_frozen_record_and_original_quote_only(self):
        p=future()
        p['401800001']['failure1_learned_rush_shadow']={
           'status':'FROZEN_2023_FITTED_RUSH_SHADOW_5000',
           'protocol':'cfb-failure1-learned-rush-shadow-v1',
           'runs':5000,'fitted_year':2023,
           'frozen_fitted_model_sha256':'f'*64,
           'source_manifest_sha256':'a'*64,
           'projected_margin_home':6.5,'margin_delta_points':1.5}
        with patch('cfb.forward_monitor.load_all_archives',return_value=p),patch(
              'cfb.forward_monitor.load_quotes',return_value=quote()):
            d=grade('notread','notread',schedule(),'2026-10-24T20:00:00+00:00',
                     'https://example.com/final','f'*64)
        sh=d['scored_games'][0]['failure1_learned_rush_shadow']
        self.assertEqual(sh['projected_home_margin'],6.5)
        self.assertFalse(sh['resimulated_after_kickoff'])
        self.assertEqual(sh['research_ats_side_at_capture'],'HOME')
        self.assertTrue(sh['cover_correct_no_push_or_no_edge'])
        self.assertEqual(d['failure1_learned_rush_shadow_forward']['paired_margin_games'],1)
        self.assertFalse(d['failure1_learned_rush_shadow_forward']['ATS_calibrated'])

if __name__=='__main__':unittest.main()
