"""Owner-only, no-spread, no-replay request routing for GitHub CFB automation."""
from __future__ import annotations
import copy
import json
import unittest
from pathlib import Path
from cfb.issue_request import parse_issue, KIND, TITLE


def ev(mode='pregame',team_a='Alabama',team_b='Tennessee', game_id=''):
    return {'action':'opened','repository':{'owner':{'login':'alecpsias'}},
            'issue':{'title':TITLE,'user':{'login':'alecpsias'},'body':json.dumps({
                'kind':KIND,'mode':mode,'team_a':team_a,'team_b':team_b,'game_id':game_id})}}

class IssueRequestTests(unittest.TestCase):
    def test_owner_pregame_without_spread(self):
        z=parse_issue(ev(),repository_owner='alecpsias')
        self.assertEqual(z['CFB_MODE'],'pregame')
        self.assertEqual(z['CFB_TEAM_A'],'Alabama')
        self.assertNotIn('spread',z)
    def test_owner_postgame(self):
        self.assertEqual(parse_issue(ev(mode='postgame',game_id='401856727'))['CFB_GAME_ID'],'401856727')
    def test_forged_owner_rejected(self):
        e=ev();e['issue']['user']['login']='stranger'
        with self.assertRaisesRegex(ValueError,'owner'):parse_issue(e)
    def test_other_issue_not_processed(self):
        e=ev();e['issue']['title']='Please run arbitrary code'
        with self.assertRaises(ValueError):parse_issue(e)
    def test_bad_json(self):
        e=ev();e['issue']['body']='not json'
        with self.assertRaises(ValueError):parse_issue(e)
    def test_extra_fields(self):
        e=ev();j=json.loads(e['issue']['body']);j['command']='rm -rf /'
        e['issue']['body']=json.dumps(j)
        with self.assertRaises(ValueError):parse_issue(e)
    def test_shell_injection_not_accepted(self):
        for school in ['Alabama\nGITHUB_TOKEN=abc','A;bash /tmp/x','$(whoami)','X' * 100]:
            with self.subTest(school=school):
                with self.assertRaises(ValueError):parse_issue(ev(team_a=school))
    def test_name_edge_cases(self):
        self.assertEqual(parse_issue(ev(team_a='Miami (FL)',team_b="Hawai'i"))['CFB_TEAM_A'],'Miami (FL)')
    def test_same_team_refused(self):
        with self.assertRaises(ValueError):parse_issue(ev(team_b='alabama'))
    def test_bad_mode(self):
        with self.assertRaises(ValueError):parse_issue(ev(mode='batch'))
    def test_bad_game_id(self):
        with self.assertRaises(ValueError):parse_issue(ev(game_id='https://bad'))
    def test_workflow_has_one_file_no_second(self):
        workflow=(Path(__file__).parents[1]/'.github/workflows/cfb-free-data.yml').read_text()
        self.assertIn('  issues:\n    types: [opened]',workflow)
        self.assertIn("github.event_name == 'issues'",workflow)
        self.assertIn('python -m cfb.issue_request',workflow)
        self.assertIn('python -m cfb.fresh_matchup',workflow)
        self.assertIn('--odds cfb/_market_disabled',workflow)
        self.assertIn("if: ${{ vars.CFB_CAPTURE_ODDS == 'true' }}",workflow)
        self.assertIn("github.event_name != 'issues'",workflow)
        self.assertEqual(workflow.count('    runs-on: ubuntu-latest'),2)

if __name__=='__main__':unittest.main()
