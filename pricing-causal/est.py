import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
import pyfixest as pf
D=pd.read_parquet('stacked.parquet')
D['lr']=np.log1p(D.revenue)
D['tp']=D.treated*D.post
res={}
def did(y,data,lab):
    m=pf.feols(f'{y} ~ tp | unit_id + cw',data=data,vcov={'CRV1':'sku'})
    c=m.coef()['tp']; se=m.se()['tp']; res[lab]=dict(coef=float(c),se=float(se),ci=[float(c-1.96*se),float(c+1.96*se)],p=float(m.pvalue()['tp']),n=int(m._N))
    return m
did('lu',D,'units_did')
did('lr',D,'revenue_did')
# first stage: realized log price on weeks with sales
Dp=D[D.units>0].copy(); Dp['lp']=np.log(Dp.revenue/Dp.units)
did('lp',Dp,'price_did')
# naive pre-post on treated only
tr=D[D.treated==1]
pre=tr[tr.post==0].groupby('unit_id').lu.mean(); post=tr[tr.post==1].groupby('unit_id').lu.mean()
d=(post-pre); res['units_naive']=dict(coef=float(d.mean()),se=float(d.std()/np.sqrt(len(d))),n=len(d))
prr=tr[tr.post==0].groupby('unit_id').lr.mean(); por=tr[tr.post==1].groupby('unit_id').lr.mean(); dr=por-prr
res['revenue_naive']=dict(coef=float(dr.mean()),se=float(dr.std()/np.sqrt(len(dr))))
# control group change over same window (what naive misses)
ct=D[D.treated==0]; dc=ct[ct.post==1].groupby('unit_id').lu.mean()-ct[ct.post==0].groupby('unit_id').lu.mean()
res['units_control_change']=float(dc.mean())
# event study
es=pf.feols('lu ~ i(rel, treated, ref=-1) | unit_id + cw',data=D,vcov={'CRV1':'sku'})
co=es.coef(); se=es.se()
ev=[]
for k in co.index:
    r=int(k.split('::')[1].split(':')[0].replace('[T.','').replace(']',''))
    ev.append(dict(rel=r,coef=float(co[k]),lo=float(co[k]-1.96*se[k]),hi=float(co[k]+1.96*se[k])))
ev.append(dict(rel=-1,coef=0.0,lo=0.0,hi=0.0)); ev=sorted(ev,key=lambda x:x['rel'])
res['event_study']=ev
# joint pre-trend test (rel -12..-2)
pre_names=[k for k in co.index if int(k.split('::')[1].split(':')[0].replace('[T.','').replace(']',''))<-1]
w=es.wald_test(R=np.eye(len(co))[[list(co.index).index(k) for k in pre_names]]) if hasattr(es,'wald_test') else None
res['pretrend_wald']=None if w is None else {k:float(v) for k,v in dict(w).items()} if not isinstance(w,pd.Series) else {k:float(v) for k,v in w.items()}
# placebo: pre-window only, fake event at rel=-6
P=D[D.rel<0].copy(); P['tp']=P.treated*(P.rel>=-6).astype(int)
did('lu',P,'units_placebo')
# robustness: controls matched on pre-period mean log units (within treated range, same cohort)
pm=D[D.post==0].groupby('unit_id').lu.mean().rename('pre_lu'); D2=D.join(pm,on='unit_id')
lohi=D2[D2.treated==1].groupby('cohort').pre_lu.agg(['min','max'])
D2=D2.join(lohi,on='cohort'); D2=D2[(D2.treated==1)|((D2.pre_lu>=D2['min'])&(D2.pre_lu<=D2['max']))]
did('lu',D2,'units_did_matched'); res['matched_controls']=int(D2[D2.treated==0].unit_id.nunique())
res['elasticity']=res['units_did']['coef']/res['price_did']['coef']
res['n_treated']=int(D[D.treated==1].sku.nunique()); res['n_control_units']=int(D[D.treated==0].unit_id.nunique()); res['n_control_skus']=int(D[D.treated==0].sku.nunique())
json.dump(res,open('did_results.json','w'),indent=1)
for k,v in res.items():
    if k!='event_study': print(k,v)
print([ (e['rel'],round(e['coef'],3)) for e in ev])
