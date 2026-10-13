import pandas as pd, numpy as np, json
ba=pd.read_parquet('backtest_all.parquet')
out=[]
for o in range(74,int(ba.origin.max())+1):
    for h in range(1,5):
        past=ba[(ba.target<=o)&(ba.h==h)]
        k=past.y.sum()/past.gbm.sum()
        cur=ba[(ba.origin==o)&(ba.h==h)].copy(); cur['gbm_bc']=cur.gbm*k; cur['k']=k; out.append(cur)
bc=pd.concat(out); bc.to_parquet('backtest_bc.parquet')
wm=lambda d,c: float(np.abs(d[c]-d.y).sum()/d.y.sum())
agg=bc.groupby(['origin','h','target'])[['y','gbm','gbm_bc','ma4']].sum().reset_index()
print('sku',{c:round(wm(bc,c),4) for c in ['gbm','gbm_bc','ma4']}, 'bias',round((bc.gbm_bc-bc.y).sum()/bc.y.sum(),4))
print('portfolio',{c:round(wm(agg,c),4) for c in ['gbm','gbm_bc','ma4']}, 'k range',bc.k.min().round(3),bc.k.max().round(3))
