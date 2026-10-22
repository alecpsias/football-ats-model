"""Failure 1 frozen shadow and score-state missingness regression tests."""
import copy,hashlib,unittest
from pathlib import Path
from cfb.failure1_shadow import shadow,include_prior_game,CONFIG_FILE
from cfb.test_cfb_severity_research import decorated,fixture
from cfb.rush_severity_research import experiment,_rate
from cfb.mechanism_build16 import simulate_build16

class FrozenShadowTests(unittest.TestCase):
    def test_freezes_five_thousand_unretuned_matched_seeds(self):
        h=decorated();saved=copy.deepcopy(h)
        x=shadow(1,2,h,False,runs=5000,seed=401,source_manifest_sha256='f'*64)
        self.assertEqual(x['status'],'FROZEN_NEGATIVE_RUN_SHADOW_5000')
        self.assertEqual(x['runs'],5000)
        self.assertEqual(x['seed'],401)
        self.assertEqual(x['candidate_source_sha256'],hashlib.sha256(CONFIG_FILE.read_bytes()).hexdigest())
        self.assertEqual(x['mode'],'negative')
        self.assertIsNone(x['market_quote'])
        self.assertIsNone(x['bet_recommendation'])
        self.assertEqual(h,saved)
        expected=simulate_build16(1,2,h,False,runs=5000,seed=401,mismatch_research='negative')
        self.assertEqual(x['projected_margin_home'],expected['mean_margin_home'])

    def test_missing_history_does_not_invent_forecast(self):
        data=shadow(1,2,{1:[],2:[]},False,runs=5000,seed=7,source_manifest_sha256='0'*64)
        self.assertEqual(data['status'],'INSUFFICIENT_PREKICKOFF_RUSH_EVIDENCE')
        self.assertNotIn('projected_margin_home',data)
        with self.assertRaises(ValueError):
            shadow(1,2,{},False,runs=2000,seed=7,source_manifest_sha256='0'*64)

    def test_zero_attempt_games_are_legitimate_for_trailing(self):
        h=decorated()
        for team in h:
            for item in h[team][:1]:
                for prefix in ('off','def'):
                    item['mismatch_'+prefix+'_trailing_rush_n']=0
                    item['mismatch_'+prefix+'_trailing_rush_good']=0
        x=experiment(1,2,h,'resilience')
        self.assertEqual(x['status'],'UNFITTED_SCENARIO_READY')
        self.assertTrue(x['home']['script_evidence']['trailing_sample_plays']>=12)
        # 0 numerator but impossible positive rate remains rejected.
        h[1][0]['mismatch_off_trailing_rush_good']=1
        self.assertEqual(experiment(1,2,h,'resilience')['status'],'UNAVAILABLE_REQUIRED_PREGAME_FEATURES')

    def test_source_extract_has_trailing_observations(self):
        z=fixture()
        x=include_prior_game({}, {}, z, 1,2)
        self.assertEqual(x['home'],'OBSERVED_VALID')
        self.assertEqual(x['away'],'OBSERVED_VALID')

if __name__=='__main__':unittest.main()
