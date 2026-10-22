"""Print safe rejection diagnostics for original public odds JSON."""
import json,math
from cfb.free_odds import fetch,endpoint,_odds_item,choose_espn_quote
from cfb.market import _american
from cfb.market import _american

for gid in ['401862810','401858498','401871093','401862808','401871055']:
  body=fetch(endpoint(gid))
  data=json.loads(body)
  rows=data.get('items',[])
  for obj in rows[:2]:
    try:
      h=obj['homeTeamOdds'];a=obj['awayTeamOdds']
      spread=float(obj['spread'])
      lineh=float(h['current']['pointSpread']['alternateDisplayValue'])
      linea=float(a['current']['pointSpread']['alternateDisplayValue'])
      probe={'game':gid,'home_price_raw':h['spreadOdds'],
        'away_price_raw':a['spreadOdds'],'home_price_parsed':_american(h['spreadOdds']),
        'away_price_parsed':_american(a['spreadOdds']),'book':obj['provider']['name'],
        'spread':spread,'line_home':lineh,'line_away':linea,
        'hf':h.get('favorite'),'af':a.get('favorite'),
        'spread_sign_consistent':abs(spread-lineh)<1e-8,
        'both_teams_opposite':abs(lineh+linea)<1e-8}
      print('PRICE_STEPS',json.dumps(probe),flush=True)
    except Exception as e:
      print('PRICE_STEPS_ERROR',gid,type(e).__name__,str(e),flush=True)
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
