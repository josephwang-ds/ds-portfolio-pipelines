import pandas as pd, numpy as np, json
bt=pd.read_parquet('backtest.parquet'); bt_all=pd.read_parquet('backtest_all.parquet'); p=pd.read_parquet('panel.parquet')
M=['naive','ma4','snaive','gbm']
price=p.groupby('StockCode').price.first()
# aggregate (portfolio total per target week, per horizon)
agg=bt.groupby(['origin','h','target'])[['y']+M].sum().reset_index()
aggw={c:float(np.abs(agg[c]-agg.y).sum()/agg.y.sum()) for c in M}
print('portfolio-week WMAPE',{k:round(v,3) for k,v in aggw.items()})
# decision: every 4th origin, order covers h1..h4
cyc=sorted(bt.origin.unique())[::4]
res=[]
for c in ['ma4','gbm']:
    tot=bt_all.groupby(['origin','sku']).agg(y4=('y','sum'),f4=(c,'sum')).reset_index()
    tot['err']=tot.y4-tot.f4
    for z_name,z in [('80%',0.8416),('90%',1.2816),('95%',1.6449)]:
        out=[]
        for o in cyc:
            hist=tot[tot.origin<=o-4]   # only errors whose 4 weeks are fully observed by origin o
            sd=hist.groupby('sku').err.std() if len(hist.origin.unique())>=4 else None
            cur=tot[tot.origin==o].set_index('sku')
            if sd is None or len(sd)==0:
                # first cycles: fall back to SKU std of 4-week totals from history up to o
                sd=pd.Series(dtype=float)
            # fallback: pooled relative error
            pooled=(hist.err.std()/hist.f4.mean()) if len(hist)>0 else 0.5
            s=cur.index.map(lambda k: sd.get(k,np.nan)).values
            s=np.where(np.isnan(s), pooled*cur.f4.values, s)
            mu=(hist.err.groupby(hist.sku).mean()).reindex(cur.index).fillna(0).values if len(hist)>0 else 0
            Q=np.maximum(cur.f4.values+mu+z*s,0)
            y=cur.y4.values; pr=price.reindex(cur.index).values
            out.append(pd.DataFrame(dict(origin=o,sku=cur.index,Q=Q,y=y,price=pr)))
        o_=pd.concat(out)
        fill=np.minimum(o_.Q,o_.y).sum()/o_.y.sum()
        excess=np.maximum(o_.Q-o_.y,0); short=np.maximum(o_.y-o_.Q,0)
        stockout_rate=(o_.y>o_.Q).mean()
        res.append(dict(method=c,target=z_name,fill_rate=fill,cycle_service=1-stockout_rate,
            excess_units=excess.sum(),short_units=short.sum(),excess_value=(excess*o_.price).sum(),short_value=(short*o_.price).sum(),
            ordered_units=o_.Q.sum(),demand_units=o_.y.sum()))
R=pd.DataFrame(res); print(R.round(3).to_string()); R.to_json('decision.json',orient='records'); print('cycles',[str(pd.Timestamp(sorted(p.week.unique())[o]).date()) for o in cyc])
