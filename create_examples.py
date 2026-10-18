"""Developer-only: create explicitly synthetic and blank sample input files."""
import json
from pathlib import Path

HERE=Path(__file__).parent / "examples"
def team(name, off=0.0, de=0.0):
    return {
      "name":name,
      "stat_source":"SYNTHETIC EXAMPLE - NO REAL GAME STATS",
      "stat_as_of_utc":"2026-10-17T23:00:00Z",
      "offense":{
        "games":5,"plays":330,"epa_pass":round(.055+off,3),"epa_rush":round(-.025+off*.5,3),
        "pass_rate":.57,"success_pass":.45+off*.1,"success_rush":.42+off*.08,
        "explosive_pass_rate":.105,"explosive_rush_rate":.11,
        "pressure_allowed":.32,"sack_allowed":.068,"interception_rate":.027,
        "fumble_lost_rate":.006,"redzone_td_rate":.60,"redzone_trips":18,
        "third_down_rate":.40,"drives_per_game":11.0,"plays_per_drive":5.9,
        "qb_rushes_per_game":3.3,"qb_rush_epa":.12
      },
      "defense":{
        "plays":340,"epa_pass_allowed":round(.02+de,3),"epa_rush_allowed":round(-.015+de*.5,3),
        "success_pass_allowed":.445+de*.1,"success_rush_allowed":.44+de*.1,
        "explosive_pass_allowed":.11,"explosive_rush_allowed":.105,
        "pressure_generated":.34,"sack_generated":.073,
        "interception_rate_forced":.025,"fumble_recovery_rate":.007,
        "redzone_td_allowed":.58,"redzone_trips_faced":19,
        "third_down_allowed":.39,"drives_faced_per_game":10.9,"plays_per_drive_allowed":5.8
      },
      "schedule":{
        "opp_def_pass_epa_allowed_avg":-.015,"opp_def_rush_epa_allowed_avg":-.01,
        "opp_off_pass_epa_avg":.015,"opp_off_rush_epa_avg":.004
      },
      "context":{"avg_start_yardline":30,"penalty_epa_per_game":-.6,"special_teams_epa_per_game":.25},
      "personnel":{"pass_epa_shift":0,"rush_epa_shift":0,"pressure_allowed_shift":0,
                   "qb_rush_multiplier":1,"pace_multiplier":1,"notes":"No personnel override applied",
                   "source":"SYNTHETIC: news checked for demo only", "checked_at_utc":"2026-10-18T19:00:00Z"}
    }

game={
    "id":"SYNTHETIC-DEMO-DO-NOT-BET", "league":"NFL",
    "kickoff_utc":"2026-10-18T20:25:00Z","data_as_of_utc":"2026-10-18T19:00:00Z",
    "neutral_site":False,
    "home":team("Synthetic Home",off=.015,de=-.015),
    "away":team("Synthetic Away",off=-.015,de=.015),
    "market":{"spread_home":-3.5,"home_odds_american":-110,"away_odds_american":-110,
              "total":45.5,"as_of_utc":"2026-10-18T18:55:00Z",
              "source":"SYNTHETIC - invented odds for software testing"}
}
(HERE/'synthetic_matchup.json').write_text(json.dumps(game,indent=2)+'\n')
template=json.loads(json.dumps(game))
template['id']="REPLACE-WITH-REAL-GAME-ID"
for side in ('home','away'):
    t=template[side]
    t['name']='REPLACE_WITH_'+side.upper()+'_TEAM'
    t['stat_source']='REQUIRED: actual source citation or URL + data definitions'
    t['stat_as_of_utc']='2026-10-17T23:00:00Z'
    for section in ('offense','defense','schedule','context'):
        for k in t[section]:t[section][k]=None
    t['personnel']['source']='REPLACE: verified up-to-date injury/lineup news, even with neutral shifts'
template['market']['source']='REPLACE: sportsbook and timestamp'
(HERE/'real_matchup_TEMPLATE.json').write_text(json.dumps(template,indent=2)+'\n')
print('Wrote example and blank template')
