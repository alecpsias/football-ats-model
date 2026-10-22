"""Failure-1 research: matchup-severity, negative designed runs, adverse-script resilience.

One opt-in laboratory mechanism inside the existing CFB Build16 challenger.
NOT calibrated, no extra raw rushing EPA credit. Asymmetric **joint** severe
offense × defense signals can be nonlinear; stress-test on high mismatch
games versus all others, never fit from NM State/Sam Houston outcome.
"""
from __future__ import annotations
import math

VERSION="cfb-failure1-severity-negative-script-v2-feature-distribution-thresholds"
MODES=("severity","negative","resilience","combined")
PRIOR_PLAYS=160.
MAX_PPD=.45
MAX_SCRIPT_PPD=.25

def _rate(rows,k,n,minimum,minimum_games=3):
    try:
        x=[(r['mismatch_'+k],r['mismatch_'+n]) for r in rows
           if 'mismatch_'+k in r and 'mismatch_'+n in r]
        if len(x)<minimum_games or any(type(a) not in (int,float) or type(b) not in (int,float)
                           or not math.isfinite(a) or not math.isfinite(b)
                           or a<0 or b<=0 or a>b for a,b in x):return None
        num=sum(v for v,_ in x);den=sum(v for _,v in x)
        return (num/den,den) if den>=minimum else None
    except (ValueError,TypeError,KeyError):return None

def _clip(x,a,b):return max(a,min(b,x))

def _side(off,defense,mode):
    own=_rate(off,'off_early_rush_success_yes','off_early_rush_n',35)
    allowed=_rate(defense,'def_early_rush_success_yes','def_early_rush_n',35)
    own_negative=_rate(off,'off_early_negative_yes','off_early_rush_n',35)
    allowed_negative=_rate(defense,'def_early_negative_yes','def_early_rush_n',35)
    own_stuffed=_rate(off,'off_early_nonpositive_yes','off_early_rush_n',35)
    allowed_stuffed=_rate(defense,'def_early_nonpositive_yes','def_early_rush_n',35)
    required=(own,allowed,own_negative,allowed_negative,own_stuffed,allowed_stuffed)
    if any(x is None for x in required):
        return {'status':'UNAVAILABLE_OFFENSE_OR_DEFENSE_DESIGNED_RUSH'}
    own_r,allow_r,neg,allow_neg,stuff,allow_stuff=required
    reliability=min(own_r[1],allow_r[1])/(min(own_r[1],allow_r[1])+PRIOR_PLAYS)
    # Feature-distribution-only, exploratory V2 cutpoints: offensive 42%
    # early-rush success and defense allowed 38%, grounded in observed
    # marginal SOURCE feature medians. The previous 50%/50% threshold
    # activated in only 2/360 team-sides; not a viable severe-cohort test.
    # These thresholds were adjusted AFTER seeing feature data: no claim of
    # independently held-out prediction validation is allowed.
    joint=max(0.,min(own_r[0]-.42,allow_r[0]-.38))
    matchup_delta=.5*(allow_r[0]-own_r[0])
    nonlinear=(joint/.075)**1.6 if joint else 0.
    severity_ppd=_clip((.27*nonlinear+.36*matchup_delta)*reliability,
                       -.24,.35)
    # Negative / stuffed run avoidance: a favorable matchup should reduce
    # expected failed early rushes, improving ability to sustain drives.
    # Penalize adverse mismatch as much as reward favorable one.
    neg_delta=(neg[0]-allow_neg[0])
    stuff_delta=(stuff[0]-allow_stuff[0])
    negative_ppd=_clip((-1.2*neg_delta-.35*stuff_delta)*reliability,
                       -.17,.17)
    resilience_ppd=0.
    script=None
    if mode in ('resilience','combined'):
        # Require ACTUAL pre-play trailing states on offense and opponents'
        # defensive allowed plays. Never interpolate missing trailing games.
        own_tr=_rate(off,'off_trailing_rush_good','off_trailing_rush_n',12,minimum_games=2)
        allow_tr=_rate(defense,'def_trailing_rush_good','def_trailing_rush_n',12,minimum_games=2)
        own_neutral=_rate(off,'off_within_8_rush_good','off_within_8_rush_n',12,minimum_games=2)
        if any(z is None for z in (own_tr,allow_tr,own_neutral)):
            return {'status':'UNAVAILABLE_VERIFIED_TRAILING_AND_NEUTRAL_STATES'}
        evidence=min(own_tr[1],allow_tr[1],own_neutral[1])
        shrink=evidence/(evidence+60)
        # Only estimated trailing-specific surplus relative to OWN neutral
        # historical rush success and opponent's trailing defense. Applied
        # exclusively while losing by 9+ in sequential simulation.
        matched_tr=.5*(own_tr[0]+allow_tr[0])
        resilience_ppd=_clip(.8*(matched_tr-own_neutral[0])*shrink,
                             -MAX_SCRIPT_PPD,MAX_SCRIPT_PPD)
        script={'own_trailing_success':round(own_tr[0],5),
                'opponent_allowed_trailing_success':round(allow_tr[0],5),
                'own_within8_success':round(own_neutral[0],5),
                'trailing_sample_plays':int(evidence),
                'trailing_ppd_shift_when_down9plus':round(resilience_ppd,6)}
    base=(severity_ppd if mode in ('severity','combined') else 0.)
    base+=(negative_ppd if mode in ('negative','combined') else 0.)
    base=_clip(base,-MAX_PPD,MAX_PPD)
    return {'status':'AVAILABLE_UNFITTED',
       'own_early_rush_success':round(own_r[0],5),
       'defense_allowed_early_rush_success':round(allow_r[0],5),
       'own_negative_run_rate':round(neg[0],5),
       'defense_allowed_negative_run_rate':round(allow_neg[0],5),
       'own_stuffed_run_rate':round(stuff[0],5),
       'defense_allowed_stuffed_run_rate':round(allow_stuff[0],5),
       'joint_severity_signal':round(joint,6),
       'relative_matched_success_delta':round(matchup_delta,6),
       'severe_nonlinear_signal':round(nonlinear,6),
       'sample_reliability':round(reliability,6),
       'severity_ppd_unfitted':round(severity_ppd,6),
       'negative_ppd_unfitted':round(negative_ppd,6),
       'baseline_ppd_delta':round(base,6),
       'trailing_only_ppd_delta':round(resilience_ppd,6),
       'script_evidence':script,
       'nonlinear_joint_threshold':{'own_success':.42,'opponent_allowed_success':.38}}

def experiment(home_id,away_id,histories,mode):
    if mode not in MODES:raise ValueError('Unrecognized matchup severity mode')
    if home_id==away_id:raise ValueError('Teams must differ')
    h=_side(histories.get(home_id,[]),histories.get(away_id,[]),mode)
    a=_side(histories.get(away_id,[]),histories.get(home_id,[]),mode)
    if h['status']!='AVAILABLE_UNFITTED' or a['status']!='AVAILABLE_UNFITTED':
        return {'status':'UNAVAILABLE_REQUIRED_PREGAME_FEATURES','mode':mode,
                'home':h,'away':a,'not_deployed':True}
    return {'status':'UNFITTED_SCENARIO_READY','version':VERSION,'mode':mode,
      'home':h,'away':a,'ppd_home_delta':h['baseline_ppd_delta'],
      'ppd_away_delta':a['baseline_ppd_delta'],
      'home_trailing_only_ppd':h['trailing_only_ppd_delta'],
      'away_trailing_only_ppd':a['trailing_only_ppd_delta'],
      'max_joint_severity':max(h['joint_severity_signal'],a['joint_severity_signal']),
      'difference_joint_severity':h['joint_severity_signal']-a['joint_severity_signal'],
      'hypotheses_fitted':False,'ATS_calibrated':False,'not_deployed':True,
      'avoids_raw_rush_EPA_duplication':True,
      'caveat':'Score-script success is observational and game-state-confounded'}
