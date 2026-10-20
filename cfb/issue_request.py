"""Strict, zero-cost bridge from a repository-owner CFB GitHub Issue to fresh analysis.

An issue is ONLY an input request. It cannot supply a model score, odds, injury
status, future final, or commands. The GitHub job obtains actual statistics
itself and runs `cfb.fresh_matchup` from scratch before kickoff.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

TITLE='CFB Matchup Request'
KIND='cfb_matchup_request_v1'
SCHOOL=re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .&'()\-/]{0,79}$")
GAME=re.compile(r'^[0-9]{8,12}$')


def parse_issue(event, repository_owner=None):
    if not isinstance(event,dict):
        raise ValueError('Missing GitHub event')
    issue=event.get('issue')
    repo=event.get('repository')
    if not isinstance(issue,dict) or not isinstance(repo,dict):
        raise ValueError('Not a GitHub issue event')
    owner=(repo.get('owner') or {}).get('login')
    author=(issue.get('user') or {}).get('login')
    if not owner or not author or author!=owner:
        raise ValueError('Only repository owner may request model execution')
    if repository_owner is not None and owner!=repository_owner:
        raise ValueError('Repository owner mismatch')
    if issue.get('title')!=TITLE:
        raise ValueError('Not a CFB model request')
    body=issue.get('body')
    if not isinstance(body,str) or len(body)>2048:
        raise ValueError('Expected concise JSON issue body')
    try:
        request=json.loads(body)
    except (TypeError,ValueError) as exc:
        raise ValueError('Issue body must be JSON') from exc
    if not isinstance(request,dict) or set(request)!={'kind','mode','team_a','team_b','game_id'}:
        raise ValueError('Unexpected CFB request fields')
    if request['kind']!=KIND or request['mode'] not in ('pregame','postgame'):
        raise ValueError('Unknown CFB model request')
    names=[]
    for key in ('team_a','team_b'):
        value=request[key]
        if not isinstance(value,str) or not SCHOOL.fullmatch(value) or value!=value.strip():
            raise ValueError('Unsafe/ambiguous team name')
        names.append(value)
    if names[0].casefold()==names[1].casefold():
        raise ValueError('Requested teams must differ')
    gid=request['game_id']
    if gid is None:gid=''
    if not isinstance(gid,str) or (gid and not GAME.fullmatch(gid)):
        raise ValueError('Invalid optional ESPN game ID')
    return {'CFB_MODE':request['mode'],'CFB_TEAM_A':names[0],
            'CFB_TEAM_B':names[1],'CFB_GAME_ID':gid}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--event',required=True);p.add_argument('--env-file',required=True)
    a=p.parse_args()
    fields=parse_issue(json.loads(Path(a.event).read_text()))
    # The real GITHUB_ENV path already exists; append safely after validating.

    with Path(a.env_file).open('a',encoding='utf8') as f:
        f.write(''.join(f'{k}={v}\n' for k,v in fields.items()))
    print(json.dumps({'status':'CFB_REQUEST_VALIDATED_NO_MARKET_REQUIRED',
        'mode':fields['CFB_MODE'],'team_a':fields['CFB_TEAM_A'],'team_b':fields['CFB_TEAM_B']}))


if __name__=='__main__':main()
