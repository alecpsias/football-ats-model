"""NFL v0.9 source-backed but UNFITTED fourth/finishing/penalty residuals.

Only observed previously completed plays. Missing sample => no scoring effect.
No isolated penalty EPA and no claim of calibrated ATS probabilities.
"""
from model import clip
VERSION='nfl-v09-situational-scoring-research-v1'
def scored_situations(profile,offense,defense):
    off=profile['offense'][offense];df=profile['defense'][defense];lg=profile['league']
    r={'version':VERSION,'status':'NO_SITUATIONAL_EFFECT_AVAILABLE',
       'fourth':{'status':'UNAVAILABLE','delta_ppd':0.},
       'finishing':{'status':'UNAVAILABLE','delta_td_share':0.},
       'penalty_first_down':{'status':'UNAVAILABLE','delta_ppd':0.},
       'total_delta_ppd':0.,'total_delta_td_share':0.,'fitted':False,'ATS_calibrated':False,
       'limits':['Unfitted coefficients; scoring effects not established causal',
                 'EPA/PPD already include much fourth-down and penalty contribution',
                 'Finishing only changes TD/FG mix; third-down penalties excluded']}
    on=off['fourth_attempts'];dn=df['fourth_attempts'];ln=lg['fourth_attempts']
    if on>=8 and dn>=8 and ln>=20:
        prior=(ln-lg['fourth_failed']+12*.5)/(ln+12)
        oa=(on-off['fourth_failed']+20*prior)/(on+20)
        da=(dn-df['fourth_failed']+20*prior)/(dn+20)
        attempts=clip(on/max(1,off['snaps_taken']/6),0,.35)
        delta=clip((da-prior)*attempts*1.,-.055,.055)
        r['fourth']={'status':'OBSERVED_UNFITTED','offense_attempts':on,'defense_faced':dn,
                     'matched_conversion_rate':round(clip(oa+.5*(da-prior),.05,.95),6),
                     'attempts_per_drive_proxy':round(attempts,6),'delta_ppd':round(delta,6)}
    on=off['scoring_opportunities'];dn=df['scoring_opportunities'];ln=lg['scoring_opportunities']
    if on>=8 and dn>=8 and ln>=30:
        prior=(lg['scoring_opportunity_touchdowns']+12*.5)/(ln+12)
        oa=(off['scoring_opportunity_touchdowns']+20*prior)/(on+20)
        da=(df['scoring_opportunity_touchdowns']+20*prior)/(dn+20)
        delta=clip((da-prior)*.18,-.032,.032)
        r['finishing']={'status':'OBSERVED_UNFITTED','offense_opportunities':on,
                        'defense_opportunities_faced':dn,
                        'matched_td_rate':round(clip(oa+.5*(da-prior),.05,.95),6),
                        'delta_td_share':round(delta,6)}
    n=off['snaps_taken'];d=df['snaps_faced'];lg_n=lg['snaps_taken']
    if n>=100 and d>=100 and lg_n>=250:
        prior=(lg['penalty_nonthird_firsts_earned']+30*.02)/(lg_n+30)
        oa=(off['penalty_nonthird_firsts_earned']+120*prior)/(n+120)
        da=(df['penalty_nonthird_firsts_allowed']+120*prior)/(d+120)
        delta=clip((da-prior)*6*.35,-.04,.04)
        r['penalty_first_down']={'status':'OBSERVED_UNFITTED',
              'offense_snaps':n,'defense_snaps_faced':d,
              'matched_awarded_rate':round(clip(oa+.5*(da-prior),0,.2),6),
              'delta_ppd':round(delta,6)}
    r['total_delta_ppd']=round(clip(r['fourth']['delta_ppd']+
                                    r['penalty_first_down']['delta_ppd'],-.085,.085),6)
    r['total_delta_td_share']=r['finishing']['delta_td_share']
    if any(r[k]['status']!='UNAVAILABLE' for k in ('fourth','finishing','penalty_first_down')):
        r['status']='OBSERVED_BOUNDED_UNFITTED'
    return r
