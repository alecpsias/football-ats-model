"""Print safe rejection diagnostics for original public odds JSON."""
import json
from cfb.free_odds import fetch,endpoint,_odds_item,choose_espn_quote
from cfb.market import _american

for gid in ['401862810','401858498','401871093','401862808','401871055']:
  body=fetch(endpoint(gid))
  data=json.loads(body)
  rows=data.get('items',[])
  print('DIAGNOSTIC',json.dumps({'game':gid,'count':len(rows),'keys':list(data),
    'first':[{ 'provider':x.get('provider'),'spread':x.get('spread'),
      'h':{'favorite':x.get('homeTeamOdds',{}).get('favorite'),
            'spreadOdds':x.get('homeTeamOdds',{}).get('spreadOdds'),
            'current':x.get('homeTeamOdds',{}).get('current',{}).get('pointSpread')},
      'a':{'favorite':x.get('awayTeamOdds',{}).get('favorite'),
            'spreadOdds':x.get('awayTeamOdds',{}).get('spreadOdds'),
            'current':x.get('awayTeamOdds',{}).get('current',{}).get('pointSpread')},
      'accepted':_odds_item(x)} for x in rows[:2]],
     'chose':choose_espn_quote(body,expected_gid=gid)[0]},default=str),flush=True)
