"""Build 4 CFB-only forward-leakage, market, and matchup feature contracts."""
import ast
import csv
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from cfb.calibration import calibrate,predict_margin
from cfb.market import (FIELDS,CFBMarketError,validate_quote,load_quotes,grade_replay,
                        make_template)
from cfb.matchups import matchup_signals


def sample(n=120):
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    result=[]
    for i in range(n):
        b=float(i%9-4)
        result.append({'game_id':str(401800000+i),'kickoff':(start+timedelta(days=i)).isoformat(),
                       'home':'Alpha','away':'Bravo',
                       'baseline_margin':b,'epa_candidate_margin':b+.25,
                       'actual_margin':2*b+1, 'sos_margin_feature':None,
                       'context_margin_feature':None})
    return result


def quote():
    return {'game_id':'401800110','home':'Alpha','away':'Bravo',
            'kickoff_utc':'2026-04-21T00:00:00+00:00',
            'quoted_at_utc':'2026-04-20T10:00:00+00:00',
            'captured_at_utc':'2026-04-20T10:01:00+00:00',
            'book':'Public free-source archive',
            'source_url':'https://example.org/public/sportsbook/odds',
            'spread_home':'-7.5','home_odds_american':'-110',
            'away_odds_american':'+105',
            'evidence_sha256':'a'*64}


class Build4Tests(unittest.TestCase):
    def test_new_modules_never_import_nfl(self):
        for name in ('matchups.py','calibration.py','market.py','research_backtest.py'):
            tree=ast.parse((Path(__file__).parent/name).read_text())
            for n in ast.walk(tree):
                if isinstance(n,ast.Import):
                    self.assertFalse(any(x.name=='model' or x.name.startswith('nfl') for x in n.names),name)
                if isinstance(n,ast.ImportFrom):
                    self.assertFalse(n.module=='model' or (n.module or '').startswith('nfl'),name)

    def test_calibration_is_deterministic(self):
        self.assertEqual(calibrate(sample()),calibrate(sample()))

    def test_calibration_has_three_strict_time_windows(self):
        o=calibrate(sample())
        self.assertEqual((o['train_games'],o['validation_games'],o['untouched_holdout_games']),(72,24,24))
        self.assertLess(o['train_last_kickoff'],o['validation_first_kickoff'])
        self.assertLess(o['validation_last_kickoff'],o['holdout_first_kickoff'])
        self.assertFalse(o['ats_validated'])
        self.assertEqual(o['deployment_status'],'NOT_APPROVED_UNTIL_INDEPENDENT_ASOF_AND_ATS_TESTS')

    def test_untouched_results_cannot_change_coefficients(self):
        base=sample()
        a=calibrate(base)
        for r in base[-24:]:
            r['actual_margin']=10000
        b=calibrate(base)
        self.assertEqual(a['coefficients'],b['coefficients'])
        self.assertEqual(a['selected_ridge_strength'],b['selected_ridge_strength'])
        self.assertNotEqual(a['holdout_selected'],b['holdout_selected'])

    def test_rejects_duplicate_game(self):
        data=sample();data[20]['game_id']=data[19]['game_id']
        with self.assertRaises(ValueError):calibrate(data)

    def test_refuse_unready_calibration(self):
        x=calibrate(sample(3))
        self.assertEqual(x['status'],'INSUFFICIENT_GAMES_FOR_CFB_TRAIN_VALIDATION_TEST')
        with self.assertRaises(ValueError):predict_margin(sample()[0],x)

    def test_matchup_features_no_fabrication_when_sos_missing(self):
        r={'opponent_id':2,'off_pass_sum':2,'off_pass_n':10,'off_rush_sum':1,'off_rush_n':10,
           'def_pass_sum':3,'def_pass_n':10,'def_rush_sum':1,'def_rush_n':10}
        o=matchup_signals(1,2,{1:[r],2:[dict(r,opponent_id=1)]})
        self.assertIsNone(o['sos_margin_feature'])
        self.assertIsNone(o['context_margin_feature'])

    def test_matchup_observes_prior_other_opponents(self):
        def rec(op,pass_off,pass_def):
            return {'opponent_id':op,'off_pass_sum':pass_off,'off_pass_n':10,
                    'off_rush_sum':0,'off_rush_n':10,'def_pass_sum':pass_def,
                    'def_pass_n':10,'def_rush_sum':0,'def_rush_n':10}
        h={1:[rec(3,5,1)],2:[rec(4,1,2)],
           3:[rec(4,4,3)],4:[rec(3,3,2)]}
        o=matchup_signals(1,2,h)
        self.assertIsNotNone(o['sos_margin_feature'])

    def test_quote_accepts_american_plus_money(self):
        q=validate_quote(quote())
        self.assertEqual(q['away_odds_american'],105)
        self.assertEqual(q['spread_home'],-7.5)

    def test_market_rejects_late_capture(self):
        q=quote();q['captured_at_utc']='2026-04-21T00:00:01Z'
        with self.assertRaises(CFBMarketError):validate_quote(q)

    def test_market_rejects_missing_source_proof(self):
        q=quote();q['evidence_sha256']='not_an_immutable_digest'
        with self.assertRaises(CFBMarketError):validate_quote(q)

    def test_market_rejects_wrong_odds_or_missing_book(self):
        for k,v in [('home_odds_american','-50'),('away_odds_american','0'),('book','')]:
            q=quote();q[k]=v
            with self.assertRaises(CFBMarketError):validate_quote(q)

    def test_missing_quote_archive_never_claims_ats(self):
        with TemporaryDirectory() as d:
            p=Path(d); (p/'preds.json').write_text('[]')
            (p/'calibration.json').write_text('{}')
            result=grade_replay(p/'preds.json',p/'calibration.json',p/'not_here.csv',p/'ats.json')
            self.assertEqual(result['status'],'NO_USER_ATTESTED_PREKICKOFF_SPREADS')
            self.assertFalse(result['ats_validated'])

    def test_market_grade_signed_spread_and_no_validated_claim(self):
        rows=sample()
        cal=calibrate(rows)
        hold=next(x for x in rows if x['game_id']=='401800110')
        # Match fixture kickoff using known constant UTC day count.
        q=quote();q['kickoff_utc']=hold['kickoff']
        dt=datetime.fromisoformat(hold['kickoff'])
        q['quoted_at_utc']=(dt-timedelta(days=1)).isoformat()
        q['captured_at_utc']=(dt-timedelta(hours=23)).isoformat()
        hold['actual_margin']=10
        hold['baseline_margin']=8
        with TemporaryDirectory() as d:
            p=Path(d);(p/'preds.json').write_text(json.dumps(rows));
            (p/'calibration.json').write_text(json.dumps(cal))
            with (p/'odds.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerow(q)
            out=grade_replay(p/'preds.json',p/'calibration.json',p/'odds.csv',p/'ats.json')
            self.assertEqual(out['quoted_holdout_games'],1)
            self.assertEqual(out['ats_directional_accuracy'],1.0)
            self.assertEqual(out['games'][0]['predicted_side'],'home')
            self.assertFalse(out['ats_validated'])

    def test_template_contains_ids_but_no_invented_lines(self):
        with TemporaryDirectory() as d:
            p=Path(d);(p/'index.json').write_text(json.dumps({'upcoming_games':[
                {'game_id':'100','home':'A','away':'B'}]}))
            self.assertEqual(make_template(p/'index.json',p/'template.csv'),1)
            with (p/'template.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]['game_id'],'100')
            self.assertEqual(rows[0]['spread_home'],'')

if __name__=='__main__':unittest.main()
