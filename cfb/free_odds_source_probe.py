"""Probe real public CFB odds schema without interpreting guesses as bets.

Diagnostic only. Protect evidence and report missing/invalid provider structures.
"""
from __future__ import annotations
import json,urllib.request,datetime,pathlib,argparse,hashlib
from urllib.error import HTTPError,URLError

USER_AGENT="Mozilla/5.0 (compatible; CFB-Free-Data-Audit/1.0; research)"
MAX=2000000
IDS=("401862810","401858498","401871093","401862808","401871055")
def fetch(url):
    try:
        req=urllib.request.Request(url,headers={"User-Agent":USER_AGENT,"Accept":"application/json"})
        with urllib.request.urlopen(req,timeout=12) as r: raw=r.read(MAX+1)
        if len(raw)>MAX:raise ValueError('Oversized response')
        parsed=json.loads(raw)
        return {"status":"HTTP_OK","received_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "url":url,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()},parsed,raw
    except (HTTPError,URLError,TimeoutError,ValueError,OSError) as e:
        return {"status":type(e).__name__,"description":str(e)[:150],"url":url},None,None

def describe(data):
    if not isinstance(data,dict):return {"type":type(data).__name__}
    out={"top_keys":list(data)[:35]}
    if 'items' in data:
        it=data.get('items')
        out['items_count']=len(it) if isinstance(it,list) else None
        if isinstance(it,list):
            out['first_items']=[{"keys":list(z)[:35] if isinstance(z,dict) else [],
              "provider":z.get('provider') if isinstance(z,dict) else None,
              "spread":z.get('spread') if isinstance(z,dict) else None,
              "details":z.get('details') if isinstance(z,dict) else None,
              "homeTeamOdds":z.get('homeTeamOdds') if isinstance(z,dict) else None,
              "awayTeamOdds":z.get('awayTeamOdds') if isinstance(z,dict) else None,
              "ref":z.get('$ref') if isinstance(z,dict) else None} for z in it[:3]]
    if isinstance(data.get('pickcenter'),list):
        out['pickcenter_count']=len(data['pickcenter'])
        out['pickcenter_sample']=data['pickcenter'][:2]
    h=data.get('header')
    if isinstance(h,dict):
        out['header_competition_count']=len(h.get('competitions') or [])
        cs=h.get('competitions') or []
        out['header_competition_sample']=[{"keys":list(c),"odds":c.get('odds'),"competitors":c.get('competitors')} for c in cs[:1]]
    if isinstance(data.get('events'),list):
        out['events_count']=len(data['events'])
        out['events_sample']=[{"id":e.get('id'),"competitions":[{"odds":c.get('odds'),
              "competitors":[{"homeAway":x.get('homeAway'),"id":x.get('id'),"displayName":(x.get('team') or {}).get('displayName')} for x in c.get('competitors',[])]} for c in e.get('competitions',[])[:1]]} for e in data['events'][:3]]
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    records=[]
    for gid in IDS:
        urls=[
            ('core',f'https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/events/{gid}/competitions/{gid}/odds?limit=50'),
            ('core_params',f'https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/events/{gid}/competitions/{gid}/odds?lang=en&region=us'),
            ('site_summary',f'https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={gid}'),
            ('site_web_summary',f'https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={gid}'),
        ]
        for source,url in urls:
            meta,content,raw=fetch(url)
            info={"game_id":gid,"source":source,"response":meta,"structural_sample":describe(content)}
            records.append(info)
            print("AUDIT",json.dumps(info,default=str)[:11000],flush=True)
    url='https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates=20261024&groups=80&limit=100'
    meta,content,_=fetch(url)
    info={"source":"site_scoreboard","response":meta,"structural_sample":describe(content)}
    records.append(info);print("AUDIT",json.dumps(info,default=str)[:9000])
    target=pathlib.Path(a.out);target.parent.mkdir(exist_ok=True,parents=True)
    target.write_text(json.dumps({"schema":"cfb_public_odds_source_structure_probe_v1","results":records},indent=2,default=str)+'\n')
if __name__=='__main__':main()
