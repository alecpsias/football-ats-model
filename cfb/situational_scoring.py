"""Opponent-matched fourth-down, drive-finishing and nonthird penalty effects.

Unfitted research hypotheses, NEVER isolated causal EPA. Missing data => no shift.
All records are historical pregame evidence supplied by the caller.
"""
VERSION='cfb-situational-scoring-research-v1'
def _clip(x,lo,hi):return max(lo,min(hi,float(x)))
def _counts(records,key,num,den):
    yes=trials=0
    for r in records:
        d=(r.get('mechanism') or {}).get(key) if isinstance(r,dict) else None
        if not isinstance(d,dict):continue
        a=d.get(num);b=d.get(den)
        if type(a) is int and type(b) is int and 0<=a<=b:
            yes+=a;trials+=b
    return yes,trials

def matchup_effects(offense,defense,plays_per_drive=6.0):
    result={'version':VERSION,'status':'NO_SITUATIONAL_EFFECT_AVAILABLE',
      'fourth_down':{'status':'UNAVAILABLE','delta_ppd':0.0},
      'finishing':{'status':'UNAVAILABLE','delta_td_share':0.0},
      'penalty_first_down':{'status':'UNAVAILABLE','delta_ppd':0.0},
      'delta_ppd':0.0,'delta_td_share':0.0,
      'fitted':False,'ATS_calibrated':False,
      'limitations':['Empirical counts, uncalibrated conversion-to-points coefficients',
                     'Situational events overlap EPA/PPD; effects capped',
                     'Finishing changes TD-to-FG composition only',
                     'Only non-third-down first downs from penalties counted']}
    # Per-drive fourth-down value conditional on this particular defense.
    oa,on=_counts(offense,'fourth_off','converted','attempted')
    da,dn=_counts(defense,'fourth_def','converted','attempted')
    if on>=8 and dn>=8:
        prior=(oa+da+12*.5)/(on+dn+12)
        o=(oa+20*prior)/(on+20);d=(da+20*prior)/(dn+20)
        drives=sum(int(x.get('drives') or 0) for x in offense)
        freq=_clip(on/max(1,drives),0,.35)
        delta=_clip((d-prior)*freq*1.0,-.055,.055)
        result['fourth_down']={'status':'OBSERVED_UNFITTED',
            'offense_attempts':on,'defense_attempts_faced':dn,
            'matched_conversion_rate':round(_clip(o+.5*(d-prior),.05,.95),6),
            'fourth_attempts_per_drive':round(freq,6),
            'delta_ppd':round(delta,6)}
    # The existing RZ and PPD parameters already include an offense's normal
    # finishing ability; only opponent-defense finishing residual enters.
    ot,oo=_counts(offense,'scoring_off','td','opportunities')
    dt,do=_counts(defense,'scoring_def','td','opportunities')
    if oo>=8 and do>=8:
        prior=(ot+dt+12*.5)/(oo+do+12)
        o=(ot+20*prior)/(oo+20);d=(dt+20*prior)/(do+20)
        delta=_clip((d-prior)*.18,-.032,.032)
        result['finishing']={'status':'OBSERVED_UNFITTED',
            'offense_opportunities':oo,'defense_opportunities_faced':do,
            'matched_td_rate':round(_clip(o+.5*(d-prior),.05,.95),6),
            'delta_td_share':round(delta,6)}
    # Penalty-created first downs EXCLUDING third down, already modeled there.
    pa,pn=_counts(offense,'penalty_first_down_off','awarded','plays')
    da,dn=_counts(defense,'penalty_first_down_def','allowed','plays')
    if pn>=100 and dn>=100:
        prior=(pa+da+30*.02)/(pn+dn+30)
        d=(da+120*prior)/(dn+120)
        o=(pa+120*prior)/(pn+120)
        delta=_clip((d-prior)*_clip(plays_per_drive,3,10)*.35,-.04,.04)
        result['penalty_first_down']={'status':'OBSERVED_UNFITTED',
            'offense_plays':pn,'defense_plays_faced':dn,
            'matched_awarded_rate':round(_clip(o+.5*(d-prior),0,.2),6),
            'delta_ppd':round(delta,6)}
    result['delta_ppd']=round(_clip(result['fourth_down']['delta_ppd']+
                                   result['penalty_first_down']['delta_ppd'],-.085,.085),6)
    result['delta_td_share']=result['finishing']['delta_td_share']
    if any(result[k]['status']!='UNAVAILABLE' for k in ('fourth_down','finishing','penalty_first_down')):
        result['status']='OBSERVED_BOUNDED_UNFITTED'
    return result
