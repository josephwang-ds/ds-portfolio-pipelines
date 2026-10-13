import pandas as pd, numpy as np, json
import sys
r=pd.read_parquet(sys.argv[1])  # UCI Online Retail II, both sheets combined (parquet)
r.columns=[c.replace(' ','') for c in r.columns]
r=r.drop(columns=['source_sheet']).drop_duplicates()
inv=r.Invoice.astype(str); sc=r.StockCode.astype(str)
nonprod=sc.str.match(r'^[A-Za-z]+$')|sc.str.upper().isin(['BANK CHARGES','C2','DOT','POST','M','ADJUST','ADJUST2','TEST001','TEST002','AMAZONFEE'])
canc=r[inv.str.startswith('C')&~nonprod]
key=lambda x: x.CustomerID.astype(str)+'|'+x.StockCode.astype(str)+'|'+x.Quantity.abs().astype(str)
s=r[(~inv.str.startswith('C'))&(r.Quantity>0)&(r.Price>0)&~nonprod].copy()
s=s[~(key(s).isin(set(key(canc)))&s.CustomerID.notna())]
s['week']=s.InvoiceDate.dt.to_period('W-SUN').dt.start_time
s=s[s.week<s.week.max()]
s.to_parquet('sales.parquet')
weeks=sorted(s.week.unique()); W={w:i for i,w in enumerate(weeks)}
reg=s[s.CustomerID.notna()]
lp=reg.groupby(['StockCode','week']).Price.max().unstack().reindex(columns=weeks)
units=s.groupby(['StockCode','week']).Quantity.sum().unstack().reindex(columns=weeks).fillna(0)
lp.to_parquet('listprice.parquet'); units.to_parquet('units.parquet')
# forward fill list price over gaps up to 4 weeks
lpf=lp.ffill(axis=1,limit=4)
ev=[]
K=8
for sku,row in lpf.iterrows():
    v=row.values
    for t in range(K,len(v)-K):
        pre=v[t-K:t]; post=v[t:t+K]
        if np.isnan(pre).any() or np.isnan(post).any(): continue
        if len(set(np.round(pre,2)))==1 and len(set(np.round(post,2)))==1 and round(pre[0],2)!=round(post[0],2):
            ev.append(dict(sku=sku,t=t,week=str(pd.Timestamp(weeks[t]).date()),p0=pre[0],p1=post[0],chg=post[0]/pre[0]-1,u_pre=units.loc[sku].values[t-K:t].sum()))
E=pd.DataFrame(ev); E.to_parquet('events.parquet')
print(len(E), E.sku.nunique()); print(E.chg.describe()); print((E.chg>0).mean())
print(E.groupby('week').size().sort_values(ascending=False).head(10))
print(E.sort_values('u_pre',ascending=False).head(10).to_string())
