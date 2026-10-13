import pandas as pd, numpy as np, json
import sys
r=pd.read_parquet(sys.argv[1])  # UCI Online Retail II, both sheets combined (parquet)
r.columns=[c.replace(' ','') for c in r.columns]
stats={'raw_rows':len(r)}
r=r.drop(columns=['source_sheet']).drop_duplicates(); stats['after_dedup']=len(r)
inv=r['Invoice'].astype(str); sc=r['StockCode'].astype(str)
nonprod=sc.str.match(r'^[A-Za-z]+$')|sc.str.upper().isin(['BANK CHARGES','C2','DOT','POST','M','ADJUST','ADJUST2','TEST001','TEST002','AMAZONFEE'])
canc=r[inv.str.startswith('C')&~nonprod]
key=lambda x: x.CustomerID.astype(str)+'|'+x.StockCode.astype(str)+'|'+x.Quantity.abs().astype(str)
ck=set(key(canc))
sales=r[(~inv.str.startswith('C'))&(r.Quantity>0)&(r.Price>0)&~nonprod].copy()
matched=key(sales).isin(ck)&sales.CustomerID.notna()
stats['sales_lines']=len(sales); stats['cancelled_matched_lines']=int(matched.sum()); stats['cancelled_matched_units']=int(sales.Quantity[matched].sum())
sales=sales[~matched]
sales['week']=sales.InvoiceDate.dt.to_period('W-SUN').dt.start_time
last=sales.week.max(); sales=sales[sales.week<last]  # drop partial final week
stats['weeks']=int(sales.week.nunique()); stats['first_week']=str(sales.week.min().date()); stats['last_week']=str(sales.week.max().date())
wk=sales.groupby(['StockCode','week']).agg(units=('Quantity','sum'),revenue=('Price',lambda s: 0)).reset_index()
rev=sales.assign(a=sales.Quantity*sales.Price).groupby(['StockCode','week']).a.sum().reset_index()
wk=wk.drop(columns='revenue').merge(rev,on=['StockCode','week']).rename(columns={'a':'revenue'})
weeks=pd.Index(sorted(sales.week.unique()))
cover=wk.groupby('StockCode').week.nunique()/len(weeks)
tot=wk.groupby('StockCode').units.sum()
elig=cover[cover>=0.8].index
top=tot.loc[elig].sort_values(ascending=False).head(100).index
stats['eligible_skus']=len(elig); stats['selected_skus']=len(top)
grid=pd.MultiIndex.from_product([top,weeks],names=['StockCode','week']).to_frame(index=False)
panel=grid.merge(wk,how='left').fillna({'units':0,'revenue':0})
desc=sales[sales.StockCode.isin(top)].groupby('StockCode').Description.agg(lambda s:s.mode().iat[0])
price=sales[sales.StockCode.isin(top)].groupby('StockCode').Price.median()
panel['desc']=panel.StockCode.map(desc); panel['price']=panel.StockCode.map(price)
stats['selected_share_units']=float(panel.units.sum()/sales.Quantity.sum())
panel.to_parquet('panel.parquet'); json.dump(stats,open('prep_stats.json','w'),indent=1,default=str); print(json.dumps(stats,indent=1,default=str))
print(panel.groupby('week').units.sum().describe())
