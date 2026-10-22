"""Clock provenance, overtime exclusion, missingness and class tests."""
import unittest
import pandas as pd
from cfb.clock_possession_validation import audit_game,clock_reference_difference,_reg_abs_clock
from cfb.clock_relationship_study import _metric,USC_SJSU_OFFICIAL

def plays():
    base=[]
    for game_number,tid,did,passmark,rushmark,st,en,duration in [
        (1,30,'x',False,True,'15:00','12:30','2:30'),
        (2,23,'y',True,False,'12:30','10:00','2:30')]:
        for i in range(2):
            base.append({'game_id':100,'game_play_number':game_number*2+i,
                'drive.id':did,'pos_team_id':tid,
                'pass':passmark,'rush':rushmark,
                'drive.timeElapsed.displayValue':duration,
                'drive.start.period.number':1,'drive.start.clock.displayValue':st,
                'drive.end.period.number':1,'drive.end.clock.displayValue':en})
    return pd.DataFrame(base)

class ClockAuditTests(unittest.TestCase):
    def test_extracts_actual_time_not_play_count(self):
        z=audit_game(plays(),100,30,23)
        self.assertEqual(z['teams']['30']['observed_time_seconds'],150)
        self.assertEqual(z['teams']['30']['rush_plays'],2)
        self.assertEqual(z['teams']['23']['pass_plays'],2)
        self.assertEqual(z['observed_seconds_total'],300)
        self.assertEqual(z['status'],'PARTIAL_CLOCK_COVERAGE')
        self.assertEqual(clock_reference_difference(z,{30:150,23:150})['status'],
                         'PASS_WITHIN_90_SEC')

    def test_period_crossing_and_ot_excluded(self):
        self.assertEqual(_reg_abs_clock(1,'0:30')-_reg_abs_clock(2,'14:00'),90)
        x=plays();x.loc[x['drive.id']=='x','drive.start.clock.displayValue']='0:30'
        x.loc[x['drive.id']=='x','drive.end.period.number']=2
        x.loc[x['drive.id']=='x','drive.end.clock.displayValue']='14:00'
        x.loc[x['drive.id']=='x','drive.timeElapsed.displayValue']='1:30'
        z=audit_game(x,100,30,23)
        self.assertEqual(z['teams']['30']['observed_time_seconds'],90)
        x.loc[x['drive.id']=='x','drive.start.period.number']=5
        z=audit_game(x,100,30,23)
        self.assertEqual(z['excluded']['missing_regulation_clock'],1)

    def test_fail_closed_on_elapsed_clock_discrepancy(self):
        x=plays()
        x.loc[x['drive.id']=='x','drive.timeElapsed.displayValue']='4:59'
        z=audit_game(x,100,30,23)
        self.assertEqual(z['excluded']['source_clock_disagreement'],1)
        self.assertEqual(z['teams']['30']['observed_time_seconds'],0)

    def test_provider_elapsed_only_with_missing_endpoint_is_flagged(self):
        x=plays()
        x.loc[x['drive.id']=='x','drive.start.clock.displayValue']=None
        z=audit_game(x,100,30,23)
        self.assertEqual(z['teams']['30']['observed_time_seconds'],150)
        self.assertEqual(z['source_only_elapsed_drive_count'],1)
        self.assertEqual(z['drives'][0]['clock_verification'],'SOURCE_ELAPSED_ONLY')
        self.assertIsNone(z['drives'][0]['clock_delta_seconds'])

    def test_missing_source_never_uses_play_count_proxy(self):
        x=plays().drop(columns=['drive.timeElapsed.displayValue'])
        z=audit_game(x,100,30,23)
        self.assertEqual(z['status'],'MISSING_DRIVE_CLOCK_SCHEMA')
        self.assertIsNone(z['observed_seconds_total'])

    def test_duplicate_inconsistent_drive_owner_fails_closed(self):
        x=plays()
        x.loc[x.game_play_number==5,'pos_team_id']=30
        z=audit_game(x,100,30,23)
        self.assertEqual(z['excluded']['ambiguous_offense_team'],1)

    def test_independently_published_box_reference_locked(self):
        r=USC_SJSU_OFFICIAL
        self.assertEqual(r['usc_possession_seconds']+r['sjsu_possession_seconds'],3600)
        self.assertEqual((r['usc_rushes'],r['sjsu_rushes']),(40,24))

    def test_prediction_metric_uses_fixed_cohort(self):
        self.assertEqual(_metric([1,2],[1,3])['mae'],.5)


if __name__=='__main__':unittest.main()
