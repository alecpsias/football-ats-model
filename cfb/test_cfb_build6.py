"""Build 6: actual archived seeds, immutable captures, pregame timing, ATS safeguards."""
from __future__ import annotations
import ast
import hashlib
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from cfb.archive import (_check_index,pack_forecasts,verify_archive,
                         load_all_archives,digest)
from cfb.free_odds import (endpoint,choose_espn_quote,_odds_item,capture,
                           verify_odds_archive,load_quotes)
from cfb.forward_monitor import grade, archive_final_source, verify_final_source

S='2026-10-20T15:10:37+00:00'
K='2026-10-24T19:30:00+00:00'
G='401856727'


def frozen(folder, run='1234567', predicted_margin=6, stamp=S):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rec={'game_id':G,'home':'Alabama','away':'Texas A&M','home_id':'333','away_id':'245',
         'kickoff_utc':K,'prediction_generated_at_utc':stamp,
         'source_snapshot_at_utc':'2026-10-20T15:10:02+00:00',
         'simulated_margin_home':predicted_margin,'status':'FROZEN_CFB_RESEARCH_MARGIN_NO_ATS_GRADE',
         'market_quote':None,'ats_prediction':None,'github_run_id':run}
    raw=(json.dumps(rec,sort_keys=True)+'\n').encode()
    (folder/(G+'.json')).write_bytes(raw)
    idx={'status':'CFB_PROSPECTIVE_RESEARCH_RECORDS_FROZEN','game_records':1,
         'github_run_id':run,'generated_at_utc':stamp,'snapshot_at_utc':'2026-10-20T15:10:02+00:00',
         'predictions':[{'game_id':G,'file':G+'.json','sha256':digest(raw),
                         'status':rec['status']}], 'margin_projections':1}
    (folder/'index.json').write_text(json.dumps(idx)+'\n')
    return rec


def odds(spread=-3.5):
    return {'provider':{'name':'DraftKings','id':'41'},'spread':spread,
            'homeTeamOdds':{'favorite':spread<0,'spreadOdds':-110},
            'awayTeamOdds':{'favorite':spread>0,'spreadOdds':-110}}


class Build6(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.dir=Path(self.t.name)
        self.forecasts=self.dir/'forecasts'
        self.src=self.dir/'source'
        self.a=frozen(self.src)
        pack_forecasts(self.src,self.forecasts)

    def fakefetch(self,url):
        self.assertEqual(url,endpoint(G))
        return json.dumps({'count':1,'items':[odds()]}).encode()

    def test_base_archive_verifiable_and_immutable(self):
        archive=self.forecasts/'1234567.zip'
        idx,data=verify_archive(archive)
        self.assertEqual(idx['game_records'],1)
        self.assertEqual(data[G]['simulated_margin_home'],6)
        with self.assertRaises(FileExistsError):
            pack_forecasts(self.src,self.forecasts)

    def test_seed_build5_original_45_forecasts(self):
        original=Path(__file__).parent/'39445851316.zip'
        if original.is_file():
            idx,data=verify_archive(original)
            self.assertEqual(len(data),45)
            self.assertEqual(idx['generated_at_utc'],'2026-10-20T15:10:37.827417+00:00')
            self.assertTrue(all(p['market_quote'] is None for p in data.values()))

    def test_archive_fails_if_original_json_tampered(self):
        path=self.forecasts/'1234567.zip'
        with zipfile.ZipFile(path) as z:
            entries={n:z.read(n) for n in z.namelist()}
        entries[G+'.json']=entries[G+'.json'].replace(b'"away": "Texas A&M"',b'"away": "TEXAS"')
        with zipfile.ZipFile(path,'w') as z:
            for n,v in entries.items():z.writestr(n,v)
        with self.assertRaises(ValueError):verify_archive(path)

    def test_earliest_forecast_selected_not_retroactively_replaced(self):
        s=self.dir/'source2';frozen(s,'2345678',55,'2026-10-21T16:00:00+00:00')
        pack_forecasts(s,self.forecasts)
        x=load_all_archives(self.forecasts)
        self.assertEqual(x[G]['simulated_margin_home'],6)
        self.assertEqual(x[G]['original_forecast_run_id'],'1234567')

    def test_provider_home_signed_spread_both_odds_required(self):
        self.assertEqual(_odds_item(odds())['spread_home'],-3.5)
        self.assertIsNone(_odds_item(odds(3.5)|{'homeTeamOdds':{'favorite':True,'spreadOdds':-110}}))
        q=odds();q['awayTeamOdds'].pop('spreadOdds')
        self.assertIsNone(_odds_item(q))
        self.assertEqual(_odds_item(odds(7.5))['spread_home'],7.5)

    def test_deterministic_provider_selection(self):
        bad={**odds(),'provider':{'name':'Unknown'}}
        alternative={**odds(),'provider':{'name':'FanDuel'}}
        sample=json.dumps({'items':[bad,alternative,odds()]}).encode()
        selected,raw,ref=choose_espn_quote(sample)
        self.assertEqual(selected['book'],'DraftKings')
        self.assertIsNone(ref)
        self.assertIsNone(raw)

    def test_pointer_source_requires_pinned_espn_host(self):
        bad=json.dumps({'items':[{'$ref':'https://evil.example.com/foo'}]}).encode()
        chosen,_,_=choose_espn_quote(bad,fetcher=lambda url: self.fail('unsafe fetch'))
        self.assertIsNone(chosen)

    def test_capture_complete_source_bytes_and_check_quote(self):
        info=capture(self.forecasts,self.dir/'odds', '3456789',
                     now=datetime(2026,10,22,14,tzinfo=timezone.utc),fetcher=self.fakefetch)
        self.assertEqual(info['quotes'],1)
        idx,recs=verify_odds_archive(self.dir/'odds'/'3456789.zip')
        self.assertEqual(recs[0]['quote']['spread_home'],-3.5)
        quote=load_quotes(self.dir/'odds')[G]
        self.assertEqual(quote['observed_at_utc'],'2026-10-22T14:00:00+00:00')
        self.assertFalse(quote['book_timestamp_verified'])

    def test_capture_zero_quotes_when_free_source_has_no_odds(self):
        info=capture(self.forecasts,self.dir/'odds', '3456789',
                     now=datetime(2026,10,22,14,tzinfo=timezone.utc),
                     fetcher=lambda url: json.dumps({'items':[]}).encode())
        self.assertEqual(info['quotes'],0)
        self.assertFalse(load_quotes(self.dir/'odds'))

    def test_capture_fails_closed_after_kickoff(self):
        info=capture(self.forecasts,self.dir/'odds','3456789',
                     now=datetime(2026,10,25,14,tzinfo=timezone.utc),fetcher=self.fakefetch)
        self.assertEqual(info['candidates'],0)

    def result_schedule(self,completed=True,home_points=40,away_points=20):
        return pd.DataFrame([{'game_id':int(G),'completed':completed,'start_date':K,
             'home_id':333,'away_id':245,'home_team':'Alabama','away_team':'Texas A&M',
             'home_points':home_points,'away_points':away_points}])

    def grade(self, odds_root=None, schedule=None):
        return grade(self.forecasts,odds_root or self.dir/'missing_odds',
                     schedule if schedule is not None else self.result_schedule(),
                     '2026-10-25T17:00:00+00:00','https://github.com/open/schedule',
                     'a'*64)

    def test_postgame_scoring_without_odds_not_ats(self):
        report=self.grade()
        self.assertEqual(report['graded_final_games'],1)
        self.assertEqual(report['forward_margin']['mae'],14.0)
        self.assertEqual(report['ats_provisional']['research_decisions'],0)
        self.assertFalse(report['ats_provisional']['ats_validated'])

    def test_pre_kickoff_market_later_than_forecast_is_valid(self):
        capture(self.forecasts,self.dir/'odds', '3456789',
                now=datetime(2026,10,22,14,tzinfo=timezone.utc),fetcher=self.fakefetch)
        report=self.grade(self.dir/'odds')
        self.assertEqual(report['ats_provisional']['research_decisions'],1)
        self.assertEqual(report['ats_provisional']['wins'],1)
        ats=report['scored_games'][0]['ATS']
        self.assertTrue(ats['market_observed_after_initial_margin_forecast'])
        self.assertEqual(ats['status'],'OBSERVED_PREKICKOFF_ESPN_PROVISIONAL')
        self.assertFalse(ats['original_book_quote_time_verified'])

    def test_no_final_if_schedule_not_complete(self):
        r=self.grade(schedule=self.result_schedule(completed=False))
        self.assertEqual(r['graded_final_games'],0)
        self.assertEqual(r['pending_games'],1)

    def test_team_ids_verified_no_ambiguous_match(self):
        r=self.result_schedule();r.loc[0,'away_id']=233
        with self.assertRaises(ValueError):self.grade(schedule=r)

    def test_failure_if_kickoff_shift_over_hour(self):
        r=self.result_schedule();r.loc[0,'start_date']='2026-10-25T23:00:00+00:00'
        s=self.grade(schedule=r)
        self.assertEqual(s['graded_final_games'],0)

    def test_first_45_seed_can_be_restored_without_overwriting(self):
        from cfb.archive import seed_archive
        seed=Path(__file__).parent/'39445851316.zip'
        new=self.dir/'permanent'
        result=seed_archive(seed,new)
        self.assertTrue(result['seeded'])
        self.assertEqual(len(load_all_archives(new)),45)
        self.assertFalse(seed_archive(seed,new)['seeded'])

    def test_store_exact_raw_final_score_release_and_verify(self):
        raw=b'PAR1'+b'x'*240+b'PAR1'
        archive=self.dir/'finals'/'4567890.zip'
        a=archive_final_source(archive,raw,'2026-10-25T12:00:00+00:00',
                               'https://github.com/sportsdataverse/test/schedule.parquet')
        self.assertEqual(a['schedule_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(verify_final_source(archive),a)
        with self.assertRaises(FileExistsError):
            archive_final_source(archive,raw,'2026-10-25T12:00:00+00:00','https://example.com')
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('evidence.json',json.dumps(a))
            z.writestr('schedule.parquet',raw+b'changed')
        with self.assertRaises(ValueError):verify_final_source(archive)

    def test_no_nfl_import_or_nfl_model_dependencies(self):
        for f in ('archive.py','free_odds.py','forward_monitor.py'):
            tree=ast.parse((Path(__file__).parent/f).read_text())
            imports=[]
            for n in ast.walk(tree):
                if isinstance(n,ast.ImportFrom): imports.append(n.module)
                elif isinstance(n,ast.Import):imports +=[x.name for x in n.names]
            self.assertFalse(any(x=='model' or x.startswith('nfl') for x in imports),f)


if __name__=='__main__':unittest.main()
