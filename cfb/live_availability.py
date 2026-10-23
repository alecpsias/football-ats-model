"""Pregame availability capture from a free linked-report mirror.

The mirror reproduces conference reports but is NOT the conference itself.
The report is fetched during the forecast run, with an observed UTC capture
time and SHA256. It is not a game-day inactives list or proof of who starts.
Failures yield explicit non-use, never healthy-by-default.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from cfb.personnel import _key

MAX_BYTES=380000
STATUSES={'OUT':'OUT','DOUBTFUL':'DOUBTFUL','QUESTIONABLE':'QUESTIONABLE',
          'PROBABLE':'PROBABLE','AVAILABLE':'AVAILABLE','GAME-TIME DECISION':'GAME_TIME_DECISION'}

def _norm(s):
    s=unicodedata.normalize('NFKD',str(s))
    s=''.join(x for x in s if not unicodedata.combining(x))
    s=re.sub(r'["“”\']([^"“”\']+)["“”\']',' ',s)
    return re.sub('[^a-z0-9]+',' ',s.lower()).strip()

def _slug(s):
    return _norm(s).replace(' ','-')

class _Rows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.heading=None;self.tagstack=[];self.in_cell=False
        self.cellbuf=[];self.cells=[];self.rows=[];self.current_heading=None
        self.headerbuf=[];self.in_header=False;self.heading_depth=None
    def handle_starttag(self,tag,attrs):
        self.tagstack.append(tag)
        if tag in ('h2','h3','h4'):
            self.in_header=True;self.headerbuf=[];self.heading_depth=tag
        if tag=='tr':
            self.cells=[];self.current_heading=self.heading
        if tag in ('td','th'):
            self.in_cell=True;self.cellbuf=[]
    def handle_data(self,data):
        if self.in_header:self.headerbuf.append(data)
        if self.in_cell:self.cellbuf.append(data)
    def handle_endtag(self,tag):
        if self.in_header and tag==self.heading_depth:
            self.heading=' '.join(' '.join(self.headerbuf).split())
            self.in_header=False
        if self.in_cell and tag in ('td','th'):
            self.cells.append(' '.join(' '.join(self.cellbuf).split()))
            self.in_cell=False
        if tag=='tr' and self.cells:
            self.rows.append((self.current_heading,list(self.cells)))
        if tag in self.tagstack:
            self.tagstack=self.tagstack[:len(self.tagstack)-1-self.tagstack[::-1].index(tag)]

def parse_report(html,away,home):
    parser=_Rows();parser.feed(html)
    records=[]
    for heading,cells in parser.rows:
        if len(cells)<3 or not heading:continue
        heading_normalized=_norm(heading)
        if any(x in heading_normalized for x in
            ('unofficial','injury notes','status changes','other week')):continue
        team=next((x for x in (away,home) if _norm(x) in heading_normalized),None)
        if team is None:continue
        status=STATUSES.get(cells[2].upper().strip())
        if status is None:continue
        m=re.match(r'^(.*?)\s*#\s*(\d+)(?:\s.*)?$',cells[0])
        if not m:continue
        name=m.group(1).strip()
        if not name or len(name)<3:continue
        records.append({'team':team,'name':name,'jersey':m.group(2),
          'position':cells[1].strip().upper(),'status':status,
          'source_row':' | '.join(cells[:3])})
    unique={(x['team'],_norm(x['name']),x['jersey']):x for x in records}
    return list(unique.values())

def _athlete_id(roster,team_id,name,jersey):
    import pandas as pd
    required={'team_id','athlete_id'}
    if roster is None or not required.issubset(roster.columns):return None
    rows=roster.loc[roster.team_id.map(_key)==str(team_id)]
    name_cols=[c for c in ('display_name','athlete_display_name','full_name') if c in rows]
    if not name_cols:return None
    target=_norm(name)
    found=[]
    for _,row in rows.iterrows():
        a=_key(row.athlete_id)
        if not a:continue
        aliases=[_norm(row[c]) for c in name_cols if not pd.isna(row[c])]
        if target in aliases or (len(target.split())>=2 and any(
                len(v.split())>=2 and target.split()[0]==v.split()[0]
                and target.split()[-1]==v.split()[-1] for v in aliases)):
            found.append(a)
    return found[0] if len(set(found))==1 else None

def fetch_matchup(away,home,away_id,home_id,kickoff,*,roster=None,
                  observed_at=None,opener=None):
    """Capture and parse linked-conference transcription strictly before kickoff.

    A secondary reproduction cannot be certified as an official source.
    A match with the live page is therefore a qualified, lower-trust input
    with explicit provenance, not a confirmed starter or official capture.
    """
    from cfb.market import _date
    now=_date(observed_at or datetime.now(timezone.utc).isoformat(),'capture time')
    start=_date(kickoff,'kickoff')
    unavailable={'status':'NO_VERIFIED_LIVE_AVAILABILITY',
       'players':[],'source_verified':False,'effects_applied_to_score':False,
       'source_captured_during_forecast':False,'reported_lineups_confirmed':False}
    if now>=start:
        return {**unavailable,'reason':'KICKOFF_ALREADY_STARTED_NO_RETROACTIVE_CAPTURE'}
    # Mainland-US game day; even late East Coast games cross 00 UTC.
    day=(start-timedelta(hours=10)).date().isoformat()
    url='https://kickoffstatus.com/games/'+day+'-'+_slug(away)+'-at-'+_slug(home)+'/'
    if urlparse(url).hostname!='kickoffstatus.com':raise ValueError('Untrusted report host')
    try:
        request=Request(url,headers={'User-Agent':'CFB-research-pregame/1.0'})
        response=(opener or urlopen)(request,timeout=9)
        with response as fh:
            data=fh.read(MAX_BYTES+1)
    except (HTTPError,URLError,OSError,TimeoutError,ValueError) as e:
        return {**unavailable,'reason':'LIVE_SOURCE_UNAVAILABLE',
                'source_url':url,'exception_class':type(e).__name__}
    # The capture timestamp is the actual read completion in production, not
    # an analysis-start timestamp carried forward by the caller.
    captured=(datetime.now(timezone.utc) if opener is None else now)
    if captured>=start:
        return {**unavailable,'reason':'SOURCE_RETURNED_AT_OR_AFTER_KICKOFF',
                'source_url':url}
    if len(data)>MAX_BYTES:
        return {**unavailable,'reason':'SOURCE_SIZE_LIMIT', 'source_url':url}
    html=data.decode('utf-8','replace')
    if ('theacc.com' not in html.lower() and 'conference' not in html.lower()):
        return {**unavailable,'reason':'NO_CONFERENCE_PROVENANCE_LINK','source_url':url}
    raw=parse_report(html,away,home)
    if not raw:
        return {**unavailable,'reason':'NO_PARSEABLE_REPORT_ROWS','source_url':url}
    idmap={str(away):str(away_id),str(home):str(home_id)}
    players=[]
    for x in raw:
        tid=idmap[x['team']]
        players.append({'team_id':tid,'athlete_id':_athlete_id(
            roster,tid,x['name'],x['jersey']),'name':x['name'],
            'jersey':x['jersey'],'position':x['position'],
            'status':x['status'],'verbatim_source_excerpt':x['source_row'],
            'is_reported_out':x['status']=='OUT','is_confirmed_out':False})
    return {'status':'PREGAME_SECONDARY_CONFERENCE_TRANSCRIPTION_CAPTURED',
        'players':players,'source_url':url,
        'captured_at_utc':captured.isoformat(),
        'source_sha256':hashlib.sha256(data).hexdigest(),
        'source_verified':False,'secondary_report_captured':True,
        'source_captured_during_forecast':True,
        'carries_official_conference_link':True,
        'reported_lineups_confirmed':False,
        'effects_applied_to_score':False,
        'limits':['Secondary reproduction may lag source report',
                  'Unknown starter and backup participation',
                  'No unsupported out=zero efficiency or arbitrary points']}
