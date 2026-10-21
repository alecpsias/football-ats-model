"""Build16 text-flagged disruption must never masquerade as charted pressure."""
import unittest
import copy
import pandas as pd
from cfb.disruption import observed_disruption,matchup_disruption
from cfb.mechanism_build16 import simulate_build16
from cfb.test_cfb_integrated_build15 import histories

def plays():
    return pd.DataFrame([
       {'pos_team_id':1,'pass':True,'sack':True,'qb_hurry':True},
       {'pos_team_id':1,'pass':True,'sack':False,'qb_hurry':True},
       {'pos_team_id':1,'pass':True,'sack':False,'qb_hurry':False},
       {'pos_team_id':1,'pass':False,'sack':True,'qb_hurry':False},
       {'pos_team_id':1,'pass':False,'sack':False,'qb_hurry':True},
       {'pos_team_id':2,'pass':True,'sack':False,'qb_hurry':False}])
def history(n=5):
    return [{'disruption_off':{'status':'OBSERVED_TEXT_FLAGGED_DISRUPTION_PROXY',
            'annotation_coverage':1,'denominator':40,'disrupted_unique':7,'non_sack_hurries':5},
            'disruption_def':{'status':'OBSERVED_TEXT_FLAGGED_DISRUPTION_PROXY',
            'annotation_coverage':1,'denominator':50,'disrupted_unique':8,'non_sack_hurries':4}}
            for _ in range(n)]

class DisruptionTests(unittest.TestCase):
    def test_union_deduplicates_sack_and_hurry(self):
        z=observed_disruption(plays(),1)
        self.assertEqual((z['denominator'],z['sacks'],z['hurries'],z['disrupted_unique'],z['non_sack_hurries']), (4,2,2,3,1))
    def test_run_not_automatically_dropback(self):
        self.assertEqual(observed_disruption(plays(),1)['denominator'],4)
    def test_missing_column_not_a_zero_pressure_claim(self):
        self.assertEqual(observed_disruption(plays().drop(columns='qb_hurry'),1)['status'],'UNAVAILABLE')
    def test_missing_event_flag_fails_closed(self):
        p=plays().astype({'qb_hurry':'object'});p.loc[1,'qb_hurry']=None
        z=observed_disruption(p,1)
        self.assertEqual(z['status'],'UNAVAILABLE')
        self.assertIsNone(z['disrupted_unique'])
    def test_empty_offense_unavailable(self):
        self.assertEqual(observed_disruption(plays(),999)['status'],'UNAVAILABLE')
    def test_match_is_conservative_and_diagnostic(self):
        z=matchup_disruption(history(),history())
        self.assertEqual(z['status'],'OBSERVED_DESCRIPTIVE_ONLY_NOT_CHARTED_PRESSURE')
        self.assertGreater(z['recorded_disruption_match_rate'],0)
        self.assertEqual(z['scoring_adjustment_ppd'],0)
    def test_insufficient_games_do_not_manufacture_protection(self):
        self.assertEqual(matchup_disruption(history(2),history())['status'],'UNAVAILABLE_INCOMPLETE_PROXY')
    def test_legacy_histories_work_and_baseline_unchanged(self):
        h=histories();before=copy.deepcopy(h)
        z=simulate_build16(1,2,h,False,runs=100,seed=12)
        self.assertEqual(z['mechanism_evidence']['home']['recorded_disruption']['status'],'UNAVAILABLE_INCOMPLETE_PROXY')
        self.assertEqual(h,before)
    def test_matches_only_pregame_observed_history(self):
        h=histories()
        for i,rec in enumerate(h[1]):
            rec.update(history(1)[0])
        for i,rec in enumerate(h[2]):
            rec.update(history(1)[0])
        z=simulate_build16(1,2,h,False,runs=100,seed=12)
        self.assertEqual(z['mechanism_evidence']['home']['recorded_disruption']['scoring_adjustment_ppd'],0)
        self.assertFalse(z['ATS_calibrated'])
if __name__=='__main__':unittest.main()
