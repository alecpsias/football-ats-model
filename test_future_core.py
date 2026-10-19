from __future__ import annotations
import json, tempfile, unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import numpy as np
from collect_injury_reports import collect
from forecast_upcoming_core import quote_reference, injuries_for_game
from model import simulate,MODEL_RULES

class InjuryStubs:
    def __init__(self,rows):self.rows=rows
    def load_injuries(self,seasons):return pd.DataFrame(self.rows)

class FutureCoreSafetyTests(unittest.TestCase):
    def test_unpriced_simulation_has_no_fabricated_cover_or_total(self):
        # Test minimal projected possession parameters, not a trained game.
        p={'expected_drives':10.5,'_ppd':2.1,'_td':.6,'_to':.10}
        r=dict(MODEL_RULES['NFL'])
        s=simulate(p,p,r,None,None,1000,123)
        for key in ('home_cover_probability_raw','away_cover_probability_raw','push_probability','over_probability_raw','under_probability_raw','total_push_probability','home_cover_monte_carlo_se'):
            self.assertIsNone(s[key],key)
        self.assertIsNotNone(s['mean_margin_home'])
    def test_historical_paid_market_unsupported_unsigned_nflverse(self):
        row=pd.Series({'spread_line':7.5,'home_spread_odds':-110,'away_spread_odds':-110,'total_line':52.5})
        q=quote_reference(row,'2026-10-18T23:00:00Z')
        self.assertEqual(q['spread_line'],7.5)
        self.assertIsNone(q['signed_home_spread'])
        self.assertFalse(q['executable_quote_verified'])
    def test_injury_capture_source_and_future_records_excluded(self):
        rows=[dict(season=2026,week=6,team='SF',report_status='Out',date_modified='2026-10-18T18:00:00Z'),
              dict(season=2026,week=6,team='WAS',report_status='Questionable',date_modified='2026-10-19T18:00:00Z')]
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)
            m=collect(2026,f,InjuryStubs(rows),now=datetime(2026,10,18,19,0,tzinfo=timezone.utc))
            self.assertEqual(m['status'],'COLLECTED')
            self.assertEqual(m['rows'],1)
            self.assertEqual(m['excluded_undated_or_future_rows'],1)
            x=injuries_for_game(f,'SF','WAS',6,datetime(2026,10,18,19,0,tzinfo=timezone.utc))
            self.assertEqual(x['team_report_rows']['SF'],1)
            self.assertIsNone(x['team_report_rows']['WAS'])
            self.assertIsNone(x['confirmed_inactives'])
            self.assertFalse(x['personnel_point_adjustments_applied'])
    def test_unlabeled_date_timezone_is_not_assumed_utc(self):
        rows=[dict(season=2026,week=6,team='SF',report_status='Out',date_modified='2026-10-18 18:00:00')]
        with tempfile.TemporaryDirectory() as t:
            m=collect(2026,Path(t),InjuryStubs(rows),now=datetime(2026,10,18,19,0,tzinfo=timezone.utc))
            self.assertEqual(m['status'],'CAPTURED_ROWS_UNVERIFIED_ASOF')
            self.assertEqual(m['rows'],0)
    def test_no_injury_feed_does_not_claim_healthy(self):
        with tempfile.TemporaryDirectory() as t:
            x=injuries_for_game(Path(t),'SF','WAS',6,datetime.now(timezone.utc))
            self.assertEqual(x['status'],'NO_INJURY_FEED')
            self.assertIsNone(x['confirmed_inactives'])
    def test_bad_feed_file_hash_hard_fails(self):
        rows=[dict(season=2026,week=6,team='SF',report_status='Out',date_modified='2026-10-18T18:00:00Z')]
        with tempfile.TemporaryDirectory() as t:
            f=Path(t);m=collect(2026,f,InjuryStubs(rows),now=datetime(2026,10,18,19,0,tzinfo=timezone.utc))
            (f/m['file']).write_text('CORRUPTED')
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                injuries_for_game(f,'SF','WAS',6,datetime(2026,10,18,19,0,tzinfo=timezone.utc))
    def test_injury_folder_collected_after_cutoff_rejected(self):
        rows=[dict(season=2026,week=6,team='SF',report_status='Out',date_modified='2026-10-18T18:00:00Z')]
        with tempfile.TemporaryDirectory() as t:
            f=Path(t);collect(2026,f,InjuryStubs(rows),now=datetime(2026,10,18,19,0,tzinfo=timezone.utc))
            x=injuries_for_game(f,'SF','WAS',6,datetime(2026,10,18,18,30,tzinfo=timezone.utc))
            self.assertEqual(x['status'],'POST_CUTOFF_INJURY_FEED_REJECTED')

if __name__=='__main__': unittest.main()
