"""CFB-only, pre-kickoff observed matchup signals. Research features, not fitted effects.

All histories MUST consist solely of earlier games. Matchup features are derived
from recorded observations; missing measurements never silently become facts.
"""
from __future__ import annotations


def _rate(records, numerator, denominator):
    items = [r for r in records if r.get(numerator) is not None and
             r.get(denominator) is not None and r[denominator] > 0]
    if not items:
        return None
    den=sum(float(r[denominator]) for r in items)
    return sum(float(r[numerator]) for r in items)/den if den else None


def _off_opponent_quality(records, history, team, kind):
    """Opponent's defensive EPA against OTHER schools, from prior games only."""
    vals=[]
    for r in records:
        opp=r.get('opponent_id')
        if opp is None:
            continue
        earlier=[x for x in history.get(opp,[]) if x.get('opponent_id')!=team]
        value=_rate(earlier,'def_'+kind+'_sum','def_'+kind+'_n')
        if value is not None and r.get('off_'+kind+'_n',0)>0:
            vals.append((value,r['off_'+kind+'_n']))
    return (sum(v*w for v,w in vals)/sum(w for _,w in vals)) if vals else None


def _def_opponent_quality(records, history, team, kind):
    """Opponent's offensive EPA against OTHER schools, from prior games only."""
    vals=[]
    for r in records:
        opp=r.get('opponent_id')
        if opp is None:
            continue
        earlier=[x for x in history.get(opp,[]) if x.get('opponent_id')!=team]
        value=_rate(earlier,'off_'+kind+'_sum','off_'+kind+'_n')
        if value is not None and r.get('def_'+kind+'_n',0)>0:
            vals.append((value,r['def_'+kind+'_n']))
    return (sum(v*w for v,w in vals)/sum(w for _,w in vals)) if vals else None


def _adjusted_attack(team, enemy, histories, kind):
    own=histories.get(team,[]); opposing=histories.get(enemy,[])
    attack=_rate(own,'off_'+kind+'_sum','off_'+kind+'_n')
    defense=_rate(opposing,'def_'+kind+'_sum','def_'+kind+'_n')
    opp_def=_off_opponent_quality(own,histories,team,kind)
    opp_off=_def_opponent_quality(opposing,histories,enemy,kind)
    if any(x is None for x in (attack,defense,opp_def,opp_off)):
        return None
    return .5*((attack-opp_def)+(defense-opp_off))


def _matched_diff(home, away, kind, offense, defense):
    """Relative matchup rate from observed offensive + allowed defensive rates."""
    hp=_rate(home,offense+'_sum',offense+'_n')
    ha=_rate(away,defense+'_sum',defense+'_n')
    ap=_rate(away,offense+'_sum',offense+'_n')
    ah=_rate(home,defense+'_sum',defense+'_n')
    if any(v is None for v in (hp,ha,ap,ah)):
        return None
    return (hp+ha-ap-ah)/2


def matchup_signals(home_id, away_id, history):
    """Unfitted margin predictors from earlier-game evidence.

    SOS uses opponent EPA observed versus *other* teams as of prediction time.
    Context only uses observed paired rate differentials; field position and
    QB/sack data remain descriptive until explicitly fitted and tested.
    """
    home=history.get(home_id,[]);away=history.get(away_id,[])
    sos={}
    for k in ('pass','rush'):
        h=_adjusted_attack(home_id,away_id,history,k)
        a=_adjusted_attack(away_id,home_id,history,k)
        sos[k]=(h-a) if h is not None and a is not None else None
    # Pass share uses actual counts and is between 0 and 1.
    np=sum(r.get('off_pass_n',0) for r in home)
    nr=sum(r.get('off_rush_n',0) for r in home)
    share=np/(np+nr) if np+nr else None
    sos_per_play=(share*sos['pass']+(1-share)*sos['rush']) if (
        share is not None and all(v is not None for v in sos.values())) else None
    # 64 estimated scrimmage plays per matchup side are only a feature scale,
    # NEVER a claim of independently estimated EPA points or a fixed adjustment.
    sos_points=64*sos_per_play if sos_per_play is not None else None
    def _both_avg(key):
        if any(r.get(key) is None for r in home+away) or not home or not away:
            return None
        return sum(r[key] for r in home)/len(home)-sum(r[key] for r in away)/len(away)

    ctx={
       'pass_explosive':_matched_diff(home,away,'pass','pass_explosive','def_pass_explosive'),
       'rush_explosive':_matched_diff(home,away,'rush','rush_explosive','def_rush_explosive'),
       'third_down':_matched_diff(home,away,'third','third_conversions','def_third_conversions'),
       'qb_hurries_allowed':_matched_diff(home,away,'qb','qb_hurries','def_qb_hurries'),
       'redzone_td_rate':_matched_diff(home,away,'redzone','redzone_tds','def_redzone_tds'),
       'field_position':_both_avg('start_field_position'),
       'qb_rushing_usage':_both_avg('qb_rushes'),
    }
    # Rate difference features are expressed in units of 10 margin-feature
    # points per unit (not presumed causal points). Fit the coefficients later.
    weights={'pass_explosive':10,'rush_explosive':6,'third_down':10,
             'qb_hurries_allowed':-3,'redzone_td_rate':4,
             'field_position':0.2,'qb_rushing_usage':0.25}
    available={k:v for k,v in ctx.items() if v is not None}
    ctx_points=sum(weights[k]*v for k,v in available.items()) if available else None
    return {
       'sos_margin_feature':round(float(sos_points),5) if sos_points is not None else None,
       'context_margin_feature':round(float(ctx_points),5) if ctx_points is not None else None,
       'observed_context':{k:(round(float(v),6) if v is not None else None) for k,v in ctx.items()},
       'sos_pass_epa_differential':sos['pass'],
       'sos_rush_epa_differential':sos['rush'],
       'limitations':['SOS requires opponents with separate prior games; missing SOS remains null',
                      'QB hurries are NOT a full pressure rate',
                      'Context feature scales are uncalibrated until trained'],
    }
