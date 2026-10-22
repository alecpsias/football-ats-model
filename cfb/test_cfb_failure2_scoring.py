"""Failure 2 regression tests: strict scoring attribution and safe fallback."""
import unittest
import pandas as pd
from cfb.scoring_attribution import attribute
from cfb.mechanism_build16 import _failure2_verified_scoring_histories

class Failure2Tests(unittest.TestCase):
    def test_clean_offense_and_defense_score(self):
        # A 7-point offensive touchdown and a 7-point pick-six.
        raw=pd.DataFrame({'game_play_number':[1,2,3],
                          'pos_team_id':[10,10,20],
                          'offense_score_play':[False,True,False],
                          'home_score':[0,7,14],
                          'away_score':[0,0,0]})
        v=attribute(raw,10,20,14,0)
        self.assertEqual(v['status'],'VERIFIED_RAW_SCORE_RECONCILIATION')
        self.assertEqual(v['offensive_points']['home'],7)
        self.assertEqual(v['nonoffensive_points']['home'],7)

    def test_bad_final_rejected(self):
        raw=pd.DataFrame({'game_play_number':[1],
                          'pos_team_id':[10],
                          'home_score':[7],'away_score':[0]})
        self.assertEqual(attribute(raw,10,20,10,0)['status'],
                         'UNVERIFIED_SCORE_ATTRIBUTION')

    def test_missing_raw_score_cannot_fabricate(self):
        raw=pd.DataFrame({'game_play_number':[1],'pos_team_id':[10]})
        self.assertEqual(attribute(raw,10,20,7,0)['reason'],
                         'NO_CUMULATIVE_SCORE_COLUMNS')

    def test_verified_history_does_not_mutate_baseline(self):
        rows={}
        for t,opp in ((10,20),(20,10)):
            rows[t]=[{'points':14.,'conceded':7.,'offensive_points':7.,
                       'offensive_points_conceded':7.,'nonoffensive_points':7.,
                       'scoring_attribution':{'status':'VERIFIED_RAW_SCORE_RECONCILIATION'}}
                      for _ in range(3)]
        updated,samples,meta=_failure2_verified_scoring_histories(10,20,rows)
        self.assertEqual(meta['status'],'ACTIVE_OFFENSE_ONLY_PPD_SEPARATE_NONOFFENSIVE_SCORING')
        self.assertEqual(updated[10][0]['points'],7)
        self.assertEqual(rows[10][0]['points'],14)
        self.assertEqual(samples['home'],[7.,7.,7.])

    def test_missing_verified_games_keeps_old_build16(self):
        rows={10:[{'points':14}]*3,20:[{'points':7}]*3}
        out,samples,meta=_failure2_verified_scoring_histories(10,20,rows)
        self.assertIs(out,rows)
        self.assertIsNone(samples)
        self.assertEqual(meta['status'],'UNVERIFIED_SCORING_ATTRIBUTION_LEGACY_BUILD16')

if __name__=='__main__': unittest.main()
