"""Source-clock strictness and no invented game-clock assertions."""
import unittest
import pandas as pd
from cfb.empirical_drive_clock import observe,matchup_clocks,sample_seconds

def complete_game():
    rows=[]
    for i in range(16):
        q=i//4+1;within=i%4
        start=900-within*200
        end=start-200
        team=10 if i%2==0 else 20
        rows.append({'game_id':401,'drive_id':str(i),'game_play_number':i+1,
                     'pos_team_id':team,'pass':True,'rush':False,'sack':False,
                     'down':1,'distance':10,'yards_gained':24 if i%3==0 else 2,
                     'pass_td':i%3==0,'rush_td':False,
                     'int':i==5,'pos_fumble_lost':False,
                     'offense_score_play':i%3==0,
                     'fg_made':False,
                     'drive.timeElapsed.displayValue':'03:20',
                     'drive.start.period.number':q,
                     'drive.end.period.number':q,
                     'drive.start.clock.displayValue':f'{start//60:02d}:{start%60:02d}',
                     'drive.end.clock.displayValue':f'{end//60:02d}:{end%60:02d}'})
    raw=pd.DataFrame(rows)
    return raw

def buckets(values):
    return {'status':'SOURCE_CLOCK_VERIFIED','clock_verification':'ESPN_ELAPSED_AND_ENDPOINTS',
            'off':values,'def':values}

class EmpiricalClockTests(unittest.TestCase):
    def test_source_clock_crosschecked_regulation_coverage(self):
        raw=complete_game()
        h=observe(raw,raw,10,20,10)
        self.assertEqual(h['status'],'SOURCE_CLOCK_VERIFIED')
        self.assertEqual(h['game_coverage'],round(3200/3600,5))
        self.assertTrue(h['off']['explosive_td'])
        self.assertTrue(all(x==200 for k in h['off'] for x in h['off'][k]))
        self.assertFalse(h['official_box_top_verified'])

    def test_bad_end_clock_rejected(self):
        raw=complete_game()
        raw.loc[0,'drive.end.clock.displayValue']='07:00'
        x=observe(raw,raw,10,20,10)
        self.assertNotEqual(x['status'],'SOURCE_CLOCK_VERIFIED')

    def test_matchup_pool_uses_only_verified_inputs(self):
        vals={'explosive_td':[37],'chain_score':[182,201],
              'turnover':[68],'stalled':[122,144]}
        rec=buckets(vals)
        h={10:[{'drive_clock':rec} for _ in range(3)],
           20:[{'drive_clock':rec} for _ in range(3)]}
        a=matchup_clocks(10,20,h)
        self.assertEqual(a['status'],'ACTIVE_VERIFIED_SOURCE_DRIVE_CLOCKS')
        self.assertFalse(a['official_box_top_verified'])
        self.assertEqual(a['home']['values']['chain_score'].count(182),6)
        self.assertEqual(a['home']['category_counts']['explosive_td'],6)
        h[10][0]['drive_clock']['status']='UNVERIFIED'
        self.assertEqual(matchup_clocks(10,20,h)['status'],'ACTIVE_VERIFIED_SOURCE_DRIVE_CLOCKS')
        h[10]=[]
        self.assertEqual(matchup_clocks(10,20,h)['status'],'UNVERIFIED_MATCHED_HISTORICAL_CLOCKS')

    def test_sampling_real_duration_values_not_assumed_ratio(self):
        import numpy as np
        vals={'explosive_td':[28,39,60],'chain_score':[100,160,215],
              'turnover':[57,72,80],'stalled':[110,138,177]}
        pool={'values':vals}
        draws=sample_seconds(np.random.default_rng(4),pool,'explosive_td',
                             np.ones(500,dtype=bool))
        self.assertTrue(set(np.unique(draws)).issubset({28.,39.,60.}))

if __name__=='__main__':unittest.main()
