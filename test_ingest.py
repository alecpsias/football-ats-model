"""Synthetic ingestion tests: these records are NOT real NFL games."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from ingest import build, normalize_schedule
from model import validate_game, InputError

CUTOFF = "2026-10-18T12:00:00Z"
KICK = "2026-10-18T13:30:00Z"


def fake_data():
    prior = [
        ("p1","HOU","BUF","2026-09-08T17:00:00Z"),
        ("p2","HOU","KC","2026-09-14T17:00:00Z"),
        ("p3","JAX","LAR","2026-09-08T17:00:00Z"),
        ("p4","JAX","DAL","2026-09-14T17:00:00Z"),
        ("p5","BUF","LAR","2026-09-21T17:00:00Z"),
        ("p6","KC","DAL","2026-09-21T17:00:00Z"),
        ("target","JAX","HOU",KICK)
    ]
    schedule=pd.DataFrame([dict(game_id=g,home_team=h,away_team=a,kickoff_utc=k,season_type="REG") for g,h,a,k in prior])
    rec=[]
    part=[]
    for game,h,a,_ in prior:
        for side,t in enumerate((h,a)):
            foe=a if t==h else h
            for driveidx in range(7):
                drive=side*7+driveidx+1
                for ix in range(6):
                    play_id=1000*(side+1)+100*driveidx+ix
                    ispass=ix<3
                    isqb=ix==5
                    rec.append(dict(game_id=game,posteam=t,defteam=foe,drive=drive,play_id=play_id,
                                    epa=(10 if game=='target' else (.05 if ispass else .01)),
                                    **{"pass":int(ispass)},rush=int(not ispass),qb_dropback=int(ispass),
                                    sack=int(ix==0 and driveidx==0),interception=int(ix==1 and driveidx==3),
                                    success=int(ix%2==0),yards_gained=(16 if ix==1 else 4),no_play=0,qb_kneel=0,qb_spike=0,
                                    fumble_lost=int(ix==4 and driveidx==5),
                                    yardline_100=[76,60,45,24,15,8][ix],
                                    pass_touchdown=0,rush_touchdown=int(ix==5),
                                    third_down_converted=int(ix==2 and driveidx%2==0),
                                    third_down_failed=int(ix==2 and driveidx%2==1),
                                    rusher_player_id=(f"{t}_QB" if isqb else f"{t}_RB") if not ispass else None))
                    part.append(dict(game_id=game,play_id=play_id,was_pressure=bool(ispass and ix==0)))
    pbp=pd.DataFrame(rec)
    part=pd.DataFrame(part)
    roster=pd.DataFrame([dict(player_id=f"{t}_QB",position="QB") for t in {k for game,h,a,_ in prior for k in (h,a)}])
    zero_context={k:{"value":0, "source":"SYNTHETIC fixture", "available_at_utc":CUTOFF} for k in ("penalty_epa_per_game","special_teams_epa_per_game")}
    personnel={"pass_epa_shift":0,"rush_epa_shift":0,"pressure_allowed_shift":0,"qb_rush_multiplier":1,"pace_multiplier":1,
               "notes":"Synthetic status check","source":"SYNTHETIC roster","checked_at_utc":CUTOFF}
    cfg={"id":"SYNTHETIC-HOU-JAX","kickoff_utc":KICK,"data_as_of_utc":CUTOFF,
         "home_code":"JAX","away_code":"HOU","home_name":"Jaguars","away_name":"Texans",
         "neutral_site":True,"pbp_source":"SYNTHETIC play-by-play","pbp_source_available_at_utc":CUTOFF,
         "participation_source_available_at_utc":CUTOFF,"roster_source_available_at_utc":CUTOFF,
         "market":{"spread_home":-2.5,"total":42.5,"home_odds_american":-110,"away_odds_american":-110,
                   "as_of_utc":CUTOFF,"source":"SYNTHETIC market"},
         "teams":{"HOU":{"context":zero_context,"personnel":personnel},
                  "JAX":{"context":zero_context,"personnel":personnel}}}
    return pbp,schedule,part,roster,cfg

class TestIngest(unittest.TestCase):
    def test_ready_complete_synthetic_snapshot(self):
        pbp,schedule,part,roster,cfg=fake_data()
        g,a=build(pbp,schedule,cfg,part,roster)
        self.assertEqual(a['status'],'READY_FOR_MODEL_UNCALIBRATED',a)
        self.assertEqual(a['missing_count'],0)
        self.assertEqual(len(a['asof_issues']),0)
        self.assertEqual(g['home']['offense']['games'],2)
        self.assertGreater(g['home']['offense']['qb_rushes_per_game'],0)
        self.assertIsInstance(validate_game(g),dict)
    def test_known_target_game_not_used(self):
        pbp,schedule,part,roster,cfg=fake_data()
        g,a=build(pbp,schedule,cfg,part,roster)
        self.assertLess(g['home']['offense']['epa_pass'],1)
        self.assertAlmostEqual(g['home']['offense']['epa_pass'],.05)
    def test_pressure_not_made_up(self):
        pbp,schedule,part,roster,cfg=fake_data()
        g,a=build(pbp,schedule,cfg,None,roster)
        self.assertTrue(any(s.endswith('pressure_allowed') for s in a['missing_fields']))
        self.assertNotEqual(a['status'],'READY_FOR_MODEL_UNCALIBRATED')
    def test_qb_rush_not_made_up(self):
        pbp,schedule,part,roster,cfg=fake_data()
        g,a=build(pbp,schedule,cfg,part,None)
        self.assertTrue(any(s.endswith('qb_rush_epa') for s in a['missing_fields']))
    def test_no_asof_provenance_blocks(self):
        pbp,schedule,part,roster,cfg=fake_data()
        cfg['pbp_source_available_at_utc']=None
        g,a=build(pbp,schedule,cfg,part,roster)
        self.assertEqual(a['status'],'BLOCKED_MISSING_INPUTS_OR_ASOF')
        self.assertTrue(any('retrospective event filtering' in s for s in a['asof_issues']))
    def test_future_source_blocks(self):
        pbp,schedule,part,roster,cfg=fake_data()
        cfg['pbp_source_available_at_utc']='2026-10-19T12:00:00Z'
        g,a=build(pbp,schedule,cfg,part,roster)
        self.assertEqual(a['status'],'BLOCKED_MISSING_INPUTS_OR_ASOF')
    def test_future_participation_blocks(self):
        pbp,schedule,part,roster,cfg=fake_data()
        cfg['participation_source_available_at_utc']='2027-02-01T12:00:00Z'
        g,a=build(pbp,schedule,cfg,part,roster)
        self.assertEqual(a['status'],'BLOCKED_MISSING_INPUTS_OR_ASOF')
    def test_bad_market_time_blocks(self):
        pbp,schedule,part,roster,cfg=fake_data()
        cfg['market']['as_of_utc']='2026-10-18T13:59:00Z'
        g,a=build(pbp,schedule,cfg,part,roster)
        self.assertEqual(a['status'],'BLOCKED_MISSING_INPUTS_OR_ASOF')
    def test_missing_column_rejected(self):
        pbp,schedule,part,roster,cfg=fake_data()
        with self.assertRaises(InputError):
            build(pbp.drop(columns=['epa']),schedule,cfg,part,roster)
    def test_schedule_time_localization(self):
        s=pd.DataFrame([{"game_id":"t","home_team":"JAX","away_team":"HOU","gameday":"2026-10-18","gametime":"09:30"}])
        self.assertEqual(str(normalize_schedule(s).iloc[0].kickoff_utc),'2026-10-18 13:30:00+00:00')

if __name__=='__main__': unittest.main()

class TestCurrentNflverseSchema(unittest.TestCase):
    def test_canonical_play_type_replaces_absent_no_play_column(self):
        pbp, schedule, part, roster, config = fake_data()
        pbp['play_type'] = pbp['pass'].map({1: 'pass', 0: 'run'})
        pbp = pbp.drop(columns=['no_play'])
        original, audit0 = build(pbp, schedule, config, part, roster)
        self.assertEqual(audit0['status'], 'READY_FOR_MODEL_UNCALIBRATED')
        # An explicit play_type=no_play should be excluded, not counted as a pass.
        row = pbp[(pbp.game_id.eq('p1')) & (pbp.posteam.eq('HOU'))].index[0]
        pbp.loc[row, 'play_type'] = 'no_play'
        _, audit1 = build(pbp, schedule, config, part, roster)
        self.assertEqual(audit1['status'], 'READY_FOR_MODEL_UNCALIBRATED')
        self.assertEqual(audit1['coverage']['eligible_scrimmage_plays'],
                         audit0['coverage']['eligible_scrimmage_plays'] - 1)

    def test_missing_no_play_and_source_classification_rejected(self):
        pbp, schedule, part, roster, config = fake_data()
        pbp = pbp.drop(columns=['no_play'])
        with self.assertRaisesRegex(InputError, 'play_type source classification'):
            build(pbp, schedule, config, part, roster)
