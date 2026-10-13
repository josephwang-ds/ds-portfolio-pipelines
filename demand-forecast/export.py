import pandas as pd, numpy as np, json
bt=pd.read_parquet('backtest.parquet'); p=pd.read_parquet('panel.parquet'); st=json.load(open('prep_stats.json'))
M=['naive','ma4','snaive','gbm']
weeks=sorted(p.week.unique()); wk=lambda i: str(pd.Timestamp(weeks[i]).date())
wm=lambda d,c: float(np.abs(d[c]-d.y).sum()/d.y.sum()); bi=lambda d,c: float((d[c]-d.y).sum()/d.y.sum())
out={'data':st,'eval':{'origins':[wk(int(bt.origin.min())),wk(int(bt.origin.max()))],'targets':[wk(int(bt.target.min())),wk(int(bt.target.max()))],'n_origins':int(bt.origin.nunique()),'n_forecasts':len(bt)}}
out['sku_level']={c:{'wmape':round(wm(bt,c),4),'bias':round(bi(bt,c),4),'by_h':[round(wm(bt[bt.h==h],c),4) for h in range(1,5)]} for c in M}
agg=bt.groupby(['origin','h','target'])[['y']+M].sum().reset_index()
out['portfolio_level']={c:{'wmape':round(wm(agg,c),4),'bias':round(bi(agg,c),4)} for c in M}
# per-sku comparison
ps=bt.groupby('sku').apply(lambda d: pd.Series({'gbm':wm(d,'gbm'),'ma4':wm(d,'ma4'),'vol':d.y.sum(),'abs_err_gbm':np.abs(d.gbm-d.y).sum()}),include_groups=False)
out['sku_compare']={'gbm_better_share':round(float((ps.gbm<ps.ma4).mean()),3),'n':len(ps),'median_wmape_gbm':round(float(ps.gbm.median()),3),'median_wmape_ma4':round(float(ps.ma4.median()),3)}
desc=p.groupby('StockCode').desc.first()
ps['share']=ps.abs_err_gbm/ps.abs_err_gbm.sum()
top=ps.sort_values('abs_err_gbm',ascending=False).head(8)
out['error_drivers']=[{'sku':k,'desc':desc[k].title(),'units':int(r.vol),'wmape_gbm':round(r.gbm,3),'wmape_ma4':round(r.ma4,3),'err_share':round(r.share,3)} for k,r in top.iterrows()]
out['top10_err_share']=round(float(ps.sort_values('abs_err_gbm',ascending=False).share.head(10).sum()),3)
# portfolio weekly series, h=1
h1=agg[agg.h==1].sort_values('target')
out['portfolio_series']=[{'week':wk(int(r.target)),'actual':int(r.y),'ma4':round(r.ma4),'gbm':round(r.gbm),'snaive':round(r.snaive)} for _,r in h1.iterrows()]
# history for context (full weekly total)
tot=p.groupby('week').units.sum()
out['history']=[{'week':str(pd.Timestamp(w).date()),'units':int(v)} for w,v in tot.items()]
# example SKUs: highest-volume and an error driver
ex=[ps.sort_values('vol',ascending=False).index[0], top.index[0]]
out['examples']=[]
for k in dict.fromkeys(ex):
    d=bt[(bt.sku==k)&(bt.h==1)].sort_values('target')
    out['examples'].append({'sku':k,'desc':desc[k].title(),'series':[{'week':wk(int(r.target)),'actual':int(r.y),'ma4':round(r.ma4,1),'gbm':round(r.gbm,1)} for _,r in d.iterrows()]})
json.dump(out,open('forecast_results.json','w'),indent=1,ensure_ascii=False)
print(json.dumps({k:out[k] for k in ['eval','sku_level','portfolio_level','sku_compare','top10_err_share']},indent=1)); print(out['error_drivers'][:4])
