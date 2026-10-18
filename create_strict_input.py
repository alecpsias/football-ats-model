"""Build an *incomplete* v0.1 strict input from sourced pregame numbers only.
All unavailable statistics are null: NEVER replace them with fabricated numbers.
"""
from pathlib import Path
import json, sys
b=Path(__file__).resolve().parent
root=b.parents[1]
tpl=json.loads((root/'examples/real_matchup_TEMPLATE.json').read_text())
s=json.loads((b/'pregame_verified_partial.json').read_text())
tpl.update(id=s['game_id'], kickoff_utc=s['kickoff_utc'],data_as_of_utc=None, neutral_site=True)
for side,name in [('home','Jacksonville Jaguars'),('away','Houston Texans')]:
   d=tpl[side]; src=s[name]
   d['name']=name
   d['stat_source']=s['source']
   d['stat_as_of_utc']=None # archived source labels pregame but provides no exact timestamp
   mapping={
     'offense': {'games':'games','epa_pass':'off_pass_epa','epa_rush':'off_rush_epa','pass_rate':'pass_rate','explosive_pass_rate':'off_explosive_pass_rate','explosive_rush_rate':'off_explosive_rush_rate','sack_allowed':'off_sack_rate','redzone_td_rate':'off_redzone_td_rate','third_down_rate':'off_third_down_conversion'},
     'defense': {'epa_pass_allowed':'def_pass_epa_allowed','epa_rush_allowed':'def_rush_epa_allowed','explosive_pass_allowed':'def_explosive_pass_allowed','explosive_rush_allowed':'def_explosive_rush_allowed','sack_generated':'def_sack_rate','redzone_td_allowed':'def_redzone_td_allowed','third_down_allowed':'def_third_down_allowed'}
   }
   for sect,items in mapping.items():
     for k,v in items.items():d[sect][k]=src[v]
   d['personnel']['notes']='; '.join(src['pre_kickoff_personnel_notes'])+'; numeric impacts deliberately not estimated'
   d['personnel']['source']=s['source']
   d['personnel']['checked_at_utc']=None
# Document odds from dated Oct14 source, but as-of exact timezone not verified.
tpl['market'].update(spread_home=-2.5,total=42.5,home_odds_american=-112,away_odds_american=-105,source='https://sports.betmgm.com/en/blog/nfl/texans-jaguars-prediction-odds-player-prop-bets-picks-week-6-jaa/ (Oct 14)',as_of_utc=None)
(b/'strict_input_INCOMPLETE.json').write_text(json.dumps(tpl,indent=2)+'\n')
missing=[]
for side in ('home','away'):
  for sect in ('offense','defense','schedule','context','personnel'):
   for field,val in tpl[side][sect].items():
    if val is None:missing.append(f'{side}.{sect}.{field}')
  if tpl[side]['stat_as_of_utc'] is None:missing.append(f'{side}.stat_as_of_utc')
for attr in ('data_as_of_utc',):
 if tpl[attr] is None:missing.append(attr)
if tpl['market']['as_of_utc'] is None:missing.append('market.as_of_utc')
print('Strict-required fields still missing:',len(missing))
print('\n'.join(missing))
sys.path.insert(0,str(root))
from model import run, InputError
try:run(tpl,runs=2000)
except Exception as e: print('Strict-engine run blocked as designed:',str(e))
else: raise AssertionError('Strict engine should have refused incomplete snapshot')
