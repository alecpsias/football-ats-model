"""Regression tests for ESPN's complete-inline-plus-legacy-http-ref payload."""
from __future__ import annotations
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from cfb.archive import pack_forecasts
from cfb.free_odds import (
    choose_espn_quote, capture, verify_odds_archive, load_quotes, endpoint,
)
from cfb.test_cfb_build6 import frozen, G, K


_REF = (
    'http://sports.core.api.espn.com/v2/sports/football/'
    f'leagues/college-football/events/{G}/competitions/{G}/odds/100?lang=en&region=us'
)


def _inline():
    """Structure and values reproduced from the stored Oct 20 ESPN response.

    Alabama -14 with -110/-110, ESPN provider name DraftKings, an HTTP $ref,
    and complete signed home/away odds embedded in the collection item.
    """
    return {
        '$ref': _REF,
        'provider': {'$ref': 'http://sports.core.api.espn.com/v2/sports/football/leagues/college-football/providers/100',
                     'id': '100', 'name': 'DraftKings', 'priority': 1},
        'details': 'ALA -14', 'spread': -14,
        'homeTeamOdds': {'favorite': True, 'spreadOdds': -110},
        'awayTeamOdds': {'favorite': False, 'spreadOdds': -110},
    }


class OddsInlineRegressionTests(unittest.TestCase):
    def test_actual_espn_inline_with_http_ref_is_accepted_without_network(self):
        collection = json.dumps({'count': 1, 'items': [_inline()]}).encode()

        def network_must_not_be_called(_):
            raise AssertionError('Inline complete odds should not follow optional HTTP reference')

        q, provider_raw, provider_url = choose_espn_quote(
            collection, fetcher=network_must_not_be_called, expected_gid=G)
        self.assertEqual(q['spread_home'], -14)
        self.assertEqual(q['home_odds_american'], -110)
        self.assertEqual(q['away_odds_american'], -110)
        self.assertEqual(q['book'], 'DraftKings')
        self.assertIsNone(provider_raw)
        self.assertIsNone(provider_url)

    def test_2026_live_provider_draft_kings_alias_is_captured_verbatim(self):
        # Source-observed on Oct 22 2026: provider id 100 has name
        # 'Draft Kings'. Previous parser silently discarded every price.
        obj = _inline()
        obj['provider']={**obj['provider'],'name':'Draft Kings'}
        obj['homeTeamOdds']={**obj['homeTeamOdds'],'spreadOdds':-110.0,
          'current':{'pointSpread':{'alternateDisplayValue':'-14.0'}}}
        obj['awayTeamOdds']={**obj['awayTeamOdds'],'spreadOdds':-110.0,
          'current':{'pointSpread':{'alternateDisplayValue':'+14.0'}}}
        original=json.dumps({'count':1,'items':[obj]}).encode()
        q,raw,provider=choose_espn_quote(original,expected_gid=G)
        self.assertEqual(q['book'],'DraftKings')
        self.assertEqual(q['provider_id'],'100')
        self.assertEqual(q['spread_home'],-14.0)
        self.assertIsNone(raw);self.assertIsNone(provider)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'frozen'; frozen(source)
            forecasts=Path(tmp)/'forecasts';pack_forecasts(source,forecasts)
            odds=Path(tmp)/'odds'
            def get(url):
                self.assertEqual(url,endpoint(G))
                return original
            ret=capture(forecasts,odds,'123456789',
                        now=datetime(2026,10,21,12,0,tzinfo=timezone.utc),
                        fetcher=get)
            self.assertEqual(ret['quotes'],1)
            _, rec=verify_odds_archive(odds/'123456789.zip')
            self.assertEqual(rec[0]['quote']['book'],'DraftKings')
            self.assertEqual(load_quotes(odds)[G]['quote']['home_odds_american'],-110)

    def test_conflicting_home_away_current_line_rejected_without_guess(self):
        obj=_inline()
        obj['provider']={**obj['provider'],'name':'Draft Kings'}
        obj['homeTeamOdds']={**obj['homeTeamOdds'],
            'current':{'pointSpread':{'alternateDisplayValue':'-13.5'}}}
        obj['awayTeamOdds']={**obj['awayTeamOdds'],
            'current':{'pointSpread':{'alternateDisplayValue':'+14'}}}
        # Never let a failed inline test follow ESPN's optional $ref and
        # accidentally read a DIFFERENT live provider line.
        obj.pop('$ref',None)
        q,_,_=choose_espn_quote(json.dumps({'items':[obj]}).encode(),expected_gid=G)
        self.assertIsNone(q)

    def test_fractional_or_nonfinite_float_price_rejected(self):
        for wrong in (-110.25,float('nan'),float('inf'),-99.0,0.0):
            with self.subTest(value=wrong):
                obj=_inline()
                obj['provider']={**obj['provider'],'name':'Draft Kings'}
                obj['homeTeamOdds']={**obj['homeTeamOdds'],'spreadOdds':wrong}
                obj.pop('$ref',None)
                q,_,_=choose_espn_quote(json.dumps({'items':[obj]}).encode(),expected_gid=G)
                self.assertIsNone(q)

    def test_unknown_similar_provider_name_not_accepted(self):
        obj=_inline()
        obj['provider']={**obj['provider'],'name':'Draft Kingzz'}
        quote,_,_=choose_espn_quote(json.dumps({'items':[obj]}).encode(),expected_gid=G)
        self.assertIsNone(quote)

    def test_incomplete_inline_can_follow_same_origin_http_ref_as_https(self):
        calls=[]
        def fake_get(url):
            calls.append(url)
            return json.dumps({k:v for k,v in _inline().items() if k != '$ref'}).encode()
        body=json.dumps({'items':[{'$ref':_REF}]}).encode()
        q, raw, url=choose_espn_quote(body,fetcher=fake_get,expected_gid=G)
        self.assertEqual(q['spread_home'],-14)
        self.assertEqual(url,'https://'+_REF[len('http://'):])
        self.assertEqual(calls,[url])
        self.assertTrue(raw)

    def test_other_event_and_external_host_references_remain_blocked(self):
        refs = [_REF.replace('/events/'+G+'/','/events/401999999/'),
                'http://example.org/steal','https://example.org/steal']
        calls=[]
        def fake_get(url):
            calls.append(url)
            raise AssertionError('Cross-event/offsite URL must not be requested')
        for ref in refs:
            with self.subTest(ref=ref):
                q,raw,url=choose_espn_quote(json.dumps({'items':[{'$ref':ref}]}).encode(),
                                              fetcher=fake_get, expected_gid=G)
                self.assertIsNone(q)
        self.assertEqual(calls,[])

    def test_complete_inline_quote_survives_capture_hash_verification_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            source=base/'frozen'
            frozen(source)
            packed=base/'forecasts'
            pack_forecasts(source,packed)
            odds=base/'odds'
            def fake_get(url):
                self.assertEqual(url,endpoint(G))
                return json.dumps({'count':1,'items':[_inline()]}).encode()
            result=capture(packed,odds,'12345678',
                           now=datetime(2026,10,21,12,0,tzinfo=timezone.utc),
                           fetcher=fake_get)
            self.assertEqual(result['quotes'],1)
            idx, recs=verify_odds_archive(odds/'12345678.zip')
            self.assertEqual(idx['complete_spread_prices'],1)
            self.assertEqual(recs[0]['status'],'OBSERVED_COMPLETE_SPREAD_AND_BOTH_PRICES')
            self.assertIsNotNone(recs[0]['raw_source_sha256'])
            quotes=load_quotes(odds)
            self.assertEqual(quotes[G]['quote']['spread_home'],-14)
            self.assertEqual(quotes[G]['quote']['home_odds_american'],-110)


if __name__=='__main__':
    unittest.main()

class SavedEvidenceRecoveryTests(unittest.TestCase):
    def test_original_october_capture_style_recovers_without_overwriting_source(self):
        """Recovers the 2026-10-20 inline DraftKings shape and keeps provenance."""
        from cfb.archive import _zipwrite, digest
        from cfb.odds_recover import recover_archive
        import zipfile
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            original=root/'odds'/'12345678.zip'
            original.parent.mkdir(parents=True)
            raw=json.dumps({'count':1,'items':[_inline()]}).encode()
            rec={'game_id':G,'status':'NO_COMPLETE_VERIFIABLE_SPREAD_ODDS',
                 'home':'Alabama','away':'Texas A&M',
                 'kickoff_utc':K,'observed_at_utc':'2026-10-20T15:50:43+00:00',
                 'source_url':endpoint(G),'quote':None,'book_timestamp_verified':False,
                 'raw_source_file':'raw/'+G+'.json','raw_source_sha256':digest(raw),
                 'raw_provider_file':None,'raw_provider_sha256':None,
                 'provider_ref_url':None,'error':None}
            rec_raw=(json.dumps(rec,sort_keys=True)+'\n').encode()
            ix={'schema':'cfb_espn_observed_odds_archive_v1','run_id':'12345678',
                'captured_at_utc':'2026-10-20T15:50:43+00:00',
                'records':[{'game_id':G,'record_file':G+'.json',
                            'record_sha256':digest(rec_raw),'status':rec['status']}],
                'candidates':1,'complete_spread_prices':0,
                'missing_or_incomplete_prices':1,'fetch_errors':0}
            with zipfile.ZipFile(original,'x') as z:
                _zipwrite(z,'index.json',json.dumps(ix).encode())
                _zipwrite(z,G+'.json',rec_raw)
                _zipwrite(z,'raw/'+G+'.json',raw)
            orig_digest=digest(original.read_bytes())
            repair=root/'odds'/'recovered'/'12345678.zip'
            result=recover_archive(original,repair,now='2026-10-21T12:00:00+00:00')
            self.assertEqual(result['recovered'],1)
            self.assertEqual(digest(original.read_bytes()),orig_digest)
            repaired_idx, rows=verify_odds_archive(repair)
            self.assertEqual(repaired_idx['original_archive_sha256'],orig_digest)
            self.assertEqual(rows[0]['observed_at_utc'],rec['observed_at_utc'])
            self.assertEqual(rows[0]['quote']['spread_home'],-14)
            self.assertEqual(rows[0]['book_timestamp_verified'],False)
            self.assertEqual(load_quotes(root/'odds')[G]['quote']['home_odds_american'],-110)
            again=recover_archive(original,repair)
            self.assertEqual(again['status'],'ALREADY_ARCHIVED_UNCHANGED')
            self.assertEqual(digest(original.read_bytes()),orig_digest)
