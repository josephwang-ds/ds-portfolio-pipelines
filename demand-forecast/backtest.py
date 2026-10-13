import pandas as pd, numpy as np, json, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
p=pd.read_parquet('panel.parquet')
weeks=sorted(p.week.unique()); T=len(weeks); W={w:i for i,w in enumerate(weeks)}
skus=sorted(p.StockCode.unique()); S=len(skus)
Y=p.pivot(index='StockCode',columns='week',values='units').loc[skus,weeks].values.astype(float)
H=4; ORIGINS=list(range(56,T-H))  # 56..98; metrics reported on 74..98 only
def feats(s,h):
    # features using data up to and including week s, for target s+h
    X={}
    for k in range(4): X[f'lag{k}']=Y[:,s-k]
    for n in (4,8,13): X[f'ma{n}']=Y[:,s-n+1:s+1].mean(1)
    X['std8']=Y[:,s-7:s+1].std(1)
    t=s+h-52; X['lag52']=Y[:,t] if t>=0 else np.full(S,np.nan)
    t2=s+h-52; X['ma52win']=Y[:,max(t2-1,0):t2+2].mean(1) if t2-1>=0 else np.full(S,np.nan)
    X['woy']=np.full(S,pd.Timestamp(weeks[s+h]).isocalendar().week if s+h<T else 0)
    X['h']=np.full(S,h); X['sku']=np.arange(S)
    return pd.DataFrame(X)
rows=[]
for o in ORIGINS:
    tr=[]; ty=[]
    for h in range(1,H+1):
        for s in range(13,o-h+1):
            tr.append(feats(s,h)); ty.append(Y[:,s+h])
    Xtr=pd.concat(tr,ignore_index=True); ytr=np.concatenate(ty)
    m=lgb.LGBMRegressor(objective='tweedie',n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,random_state=7,verbose=-1)
    m.fit(Xtr,ytr,categorical_feature=['sku'])
    for h in range(1,H+1):
        Xt=feats(o,h); yh=m.predict(Xt)
        t=o+h
        for i in range(S):
            rows.append(dict(origin=o,h=h,target=t,sku=skus[i],y=Y[i,t],
                naive=Y[i,o], ma4=Y[i,o-3:o+1].mean(), snaive=Y[i,t-52], gbm=max(yh[i],0)))
bt_all=pd.DataFrame(rows); bt_all.to_parquet('backtest_all.parquet'); bt=bt_all[bt_all.origin>=74]; bt.to_parquet('backtest.parquet')
def wm(df,c): return np.abs(df[c]-df.y).sum()/df.y.sum()
def bias(df,c): return (df[c]-df.y).sum()/df.y.sum()
M=['naive','ma4','snaive','gbm']
summ={c:dict(wmape=wm(bt,c),bias=bias(bt,c)) for c in M}
byh={h:{c:wm(bt[bt.h==h],c) for c in M} for h in range(1,H+1)}
print(json.dumps(summ,indent=1)); print(pd.DataFrame(byh).round(3))
