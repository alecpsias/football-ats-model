"""Failure 3: drive sustainability versus isolated explosive scoring.

Only observed earlier-game *scrimmage drives* enter matchups. Explosive TDs
remain possible; a bounded incremental correction recognizes that isolated
explosives do not independently imply repeatable first-down chains.
Coefficients are conservative, explicitly UNFITTED hypotheses.
"""
from __future__ import annotations
import math
import numpy as np

VERSION='cfb-failure3-sustainability-v2-rush-td-route'
# Hypothetical reference levels. Must be independently calibrated before
# promotion to an ATS confidence model.
PRIOR_CHAIN=.42  # Explicit structural benchmark: >=3 first-down series/drive
PRIOR_EARLY_SUCCESS=.45
PRIOR_LOW_CHAIN=.34
PRIOR_EXPLOSIVE_TD=.20
PRIOR_DRIVES=45.
PRIOR_TDS=16.
MAX_PENALTY_PPD=.32


def observed(plays,team,opponent):
    """Compute distinct source-observed offense and defense drive features.

    Scrimmage-only possession IDs are the same ones used by the existing
    engine. Count first-down series by actual down resets, not unreliable
    provider firstD_by_yards FALSE labels. Flag short explosive TD drives
    separately. Missing or ambiguous source fields remain unavailable.
    """
    needed={'pos_team_id','drive_id','game_play_number','down','distance',
            'yards_gained','pass','rush','pass_td','rush_td'}
    if not needed.issubset(plays.columns):
        return {'status':'MISSING_DRIVE_FIELDS'}
    out={}
    for side,tid in (('off',team),('def',opponent)):
        x=plays.loc[plays.pos_team_id.eq(tid)]
        if x.empty or x[list(needed-{'pos_team_id'})].isna().any().any():
            return {'status':'INCOMPLETE_DRIVE_FIELDS'}
        x=x.loc[x['pass'].eq(True)^x['rush'].eq(True)].sort_values('game_play_number')
        if x.empty:return {'status':'NO_SCRIMMAGE_PLAYS'}
        rec={'drives':0,'sustained':0,'low_chain_non_td':0,
             'early_success':0,'early_plays':0,'touchdown_drives':0,
             'short_explosive_pass_tds':0,'short_explosive_rush_tds':0,
             'short_explosive_tds':0,'pass_explosive_plays':0,
             'pass_plays':0}
        for _,d in x.groupby('drive_id',sort=False):
            rec['drives']+=1
            first_series=int(d.down.eq(1).sum())
            td=bool((d.pass_td.eq(True)|d.rush_td.eq(True)).any())
            if first_series>=3:rec['sustained']+=1
            if first_series<=1 and not td:rec['low_chain_non_td']+=1
            rec['touchdown_drives']+=int(td)
            explosive=((d['pass'].eq(True))&
                       (d.yards_gained.astype(float)>=20))
            rec['pass_explosive_plays']+=int(explosive.sum())
            rec['pass_plays']+=int(d['pass'].eq(True).sum())
            # One explosive touchdown *route* per short scoring drive.
            # A rush of >=20 yards can finish a two-play drive just as a
            # long pass can. Do not credit two TDs if both appear on a drive.
            explosive_rush=d['rush'].eq(True)&d.yards_gained.astype(float).ge(20)
            if td and first_series<=2:
                rec['short_explosive_pass_tds']+=int(bool(explosive.any()))
                rec['short_explosive_rush_tds']+=int(bool(explosive_rush.any()))
                rec['short_explosive_tds']+=int(bool(explosive.any()|
                                                     explosive_rush.any()))
            early=d.loc[d.down.isin((1,2))]
            if len(early):
                threshold=np.where(early.down.eq(1),.5,.7)*early.distance.astype(float)
                rec['early_success']+=int((early.yards_gained.astype(float)>=threshold).sum())
                rec['early_plays']+=len(early)
        out[side]=rec
    if min(out['off']['drives'],out['def']['drives'])<3:
        return {'status':'INSUFFICIENT_DRIVES'}
    return {'status':'OBSERVED', 'version':VERSION,**out}


def _pooled(rows,side):
    cols=('drives','sustained','low_chain_non_td','early_success','early_plays',
          'touchdown_drives','short_explosive_pass_tds','pass_explosive_plays','pass_plays')
    totals={x:0. for x in cols}
    # The new combined pass/rush TD route is usable only when historically
    # measured. Older prior-game rows retain the old pass-only behavior;
    # absent rushing labels are never interpreted as observed zero runs.
    totals.update(short_explosive_total_new=0.,short_explosive_tds_den_new=0.,
                  short_explosive_rush_new=0.,short_explosive_new_games=0.)
    supported=0
    for r in rows:
        o=r.get('sustainability')
        if not isinstance(o,dict) or o.get('status')!='OBSERVED' or side not in o:
            continue
        vals=o[side]
        if any(type(vals.get(k)) not in (int,float) or
               not math.isfinite(vals[k]) or vals[k]<0 for k in cols):
            continue
        if (vals['sustained']>vals['drives'] or
            vals['low_chain_non_td']>vals['drives'] or
            vals['early_success']>vals['early_plays'] or
            vals['short_explosive_pass_tds']>vals['touchdown_drives'] or
            vals['pass_explosive_plays']>vals['pass_plays']):
            continue
        for k in cols:totals[k]+=vals[k]
        all_td=vals.get('short_explosive_tds')
        rush_td=vals.get('short_explosive_rush_tds')
        if (type(all_td) is int and type(rush_td) is int and
                0<=all_td<=vals['touchdown_drives'] and
                0<=rush_td<=all_td):
            totals['short_explosive_total_new']+=all_td
            totals['short_explosive_tds_den_new']+=vals['touchdown_drives']
            totals['short_explosive_rush_new']+=rush_td
            totals['short_explosive_new_games']+=1
        supported+=1
    if supported<3 or totals['drives']<24 or totals['early_plays']<45:
        return None
    return totals,supported


def _match(off,opdef):
    od,og=_pooled(off,'off') or (None,None)
    dd,dg=_pooled(opdef,'def') or (None,None)
    if od is None or dd is None:return {'status':'UNAVAILABLE_VERIFIED_SUSTAINABILITY'}
    def shrink(num,den,prior,k):
        return (num+k*prior)/(den+k)
    chain=.5*(shrink(od['sustained'],od['drives'],PRIOR_CHAIN,PRIOR_DRIVES)+
              shrink(dd['sustained'],dd['drives'],PRIOR_CHAIN,PRIOR_DRIVES))
    low=.5*(shrink(od['low_chain_non_td'],od['drives'],PRIOR_LOW_CHAIN,PRIOR_DRIVES)+
            shrink(dd['low_chain_non_td'],dd['drives'],PRIOR_LOW_CHAIN,PRIOR_DRIVES))
    early=.5*(shrink(od['early_success'],od['early_plays'],PRIOR_EARLY_SUCCESS,140.)+
              shrink(dd['early_success'],dd['early_plays'],PRIOR_EARLY_SUCCESS,140.))
    # Fit the fraction of existing scoring TDs taking an isolated
    # explosive route (pass OR run), without adding a new scoring bonus.
    # At least 3 source-complete games and 4 known scoring TDs are needed.
    new_route=(od['short_explosive_new_games']>=3 and
               od['short_explosive_tds_den_new']>=4)
    explosive_td=shrink(
        od['short_explosive_total_new'] if new_route
            else od['short_explosive_pass_tds'],
        od['short_explosive_tds_den_new'] if new_route
            else od['touchdown_drives'],
        PRIOR_EXPLOSIVE_TD,PRIOR_TDS)
    reliable=min(od['drives'],dd['drives'])
    weight=reliable/(reliable+PRIOR_DRIVES)
    # A large penalty requires both limited chain production and explosive
    # touchdown dependence. Early-down weakness / unproductive short drives
    # increase its severity. This is a residual after EPA and scoring PPD,
    # NOT additive yardage nor a second explosive-play bonus.
    chain_weak=max(0.,min(1.,(PRIOR_CHAIN-chain)/PRIOR_CHAIN))
    early_weak=max(0.,min(1.,(PRIOR_EARLY_SUCCESS-early)/PRIOR_EARLY_SUCCESS))
    low_excess=max(0.,min(1.,(low-PRIOR_LOW_CHAIN)/(1-PRIOR_LOW_CHAIN)))
    explosive_excess=max(0.,min(1.,(explosive_td-PRIOR_EXPLOSIVE_TD)/
                                      (1-PRIOR_EXPLOSIVE_TD)))
    interaction=(chain_weak*(.15+.85*explosive_excess)*
                  (.45+.35*early_weak+.20*low_excess))
    correction=-min(MAX_PENALTY_PPD,.65*weight*interaction)
    return {'status':'AVAILABLE_BOUNDED_UNFITTED',
        'chain_share_matched':round(chain,6),
        'early_down_success_matched':round(early,6),
        'low_chain_non_td_matched':round(low,6),
        'short_explosive_pass_td_share_shrunk':round(explosive_td,6)
              if not new_route else round(shrink(
               od['short_explosive_pass_tds'],od['touchdown_drives'],
               PRIOR_EXPLOSIVE_TD,PRIOR_TDS),6),
        'short_explosive_td_share_shrunk':round(explosive_td,6),
        'explosive_td_route_type':('SOURCE_PASS_AND_RUSH'
                if new_route else 'PASS_ONLY_LEGACY_MISSING_SOURCE_RUSH'),
        'observed_short_explosive_rush_tds':int(od['short_explosive_rush_new']),
        'source_complete_explosive_rush_games':int(od['short_explosive_new_games']),
        'explosive_dependence_excess':round(explosive_excess,6),
        'chain_weakness':round(chain_weak,6),
        'ppd_delta':round(float(correction),6),
        'matched_drives':min(od['drives'],dd['drives']),
        'qualified_offense_games':og,'qualified_opponent_defense_games':dg,
        'incremental_not_duplicate_epa':True,
        'fitted':False,'ATS_calibrated':False}


def correction(home_id,away_id,histories):
    home=_match(histories.get(home_id,()),histories.get(away_id,()))
    away=_match(histories.get(away_id,()),histories.get(home_id,()))
    if any(x['status']!='AVAILABLE_BOUNDED_UNFITTED' for x in (home,away)):
        return {'status':'MISSING_PRIOR_DRIVE_SUSTAINABILITY',
                'home':home,'away':away,'ppd_home':0.,'ppd_away':0.,
                'baseline_unchanged':True,'ATS_calibrated':False}
    return {'status':'ACTIVE_BUILD16_UNCALIBRATED_FAILURE3',
            'home':home,'away':away,
            'ppd_home':home['ppd_delta'],'ppd_away':away['ppd_delta'],
            'baseline_unchanged':True,'ATS_calibrated':False,
            'unfitted_hypothesis':True,
            'limitations':['Explosive touchdowns still simulated by Build16',
                           'Observed scrimmage drives may omit non-scrimmage possessions',
                           'Partial drive and clock effects are not modeled here']}
