import pandas as pd, numpy as np, json
E=pd.read_parquet('events.parquet'); lp=pd.read_parquet('listprice.parquet'); U=pd.read_parquet('units.parquet')
s=pd.read_parquet('sales.parquet')
weeks=list(U.columns); T=len(weeks)
rev=s.assign(a=s.Quantity*s.Price).groupby(['StockCode','week']).a.sum().unstack().reindex(columns=weeks).fillna(0).reindex(U.index).fillna(0)
lpf=lp.ffill(axis=1,limit=4).reindex(U.index)
L=12  # event window +-12 weeks
treat=E[(E.chg>0.02)&(E.chg<0.25)].copy()
# one event per SKU: keep first qualifying
treat=treat.sort_values('t').drop_duplicates('sku')
treat=treat[(treat.t-L>=0)&(treat.t+L<T)]
cohorts=treat.groupby('t').sku.apply(list)
rows=[]
for t,tsk in cohorts.items():
    win=range(t-L,t+L)
    # controls: list price observed and constant across window (ffill), active in pre period
    P=lpf.iloc[:,t-L:t+L]
    stable=P.notna().sum(axis=1)>=int(0.8*2*L)
    const=(P.max(axis=1).round(2)==P.min(axis=1).round(2))
    act=(U.iloc[:,t-L:t]>0).mean(axis=1)>=0.5
    ctrl=U.index[stable&const&act].difference(tsk)
    tact=[k for k in tsk if (U.loc[k].iloc[t-L:t]>0).mean()>=0.5]
    if not tact: continue
    for k,tr in [(k,1) for k in tact]+[(k,0) for k in ctrl]:
        for j,w in enumerate(range(t-L,t+L)):
            rows.append((t,k,tr,j-L,U.loc[k].iat[w],rev.loc[k].iat[w]))
D=pd.DataFrame(rows,columns=['cohort','sku','treated','rel','units','revenue'])
D['post']=(D.rel>=0).astype(int); D['unit_id']=D.cohort.astype(str)+'_'+D.sku; D['cw']=D.cohort.astype(str)+'_'+D.rel.astype(str)
D['lu']=np.log1p(D.units)
D.to_parquet('stacked.parquet')
info=D.groupby(['cohort','treated']).sku.nunique().unstack()
info.index=[str(pd.Timestamp(weeks[i]).date()) for i in info.index]; print(info); print(len(D))
tr=treat[treat.sku.isin(D[D.treated==1].sku)]
print('treated events used',len(tr),'median chg',tr.chg.median(), 'mean',tr.chg.mean())
tr.to_parquet('treat_used.parquet')
