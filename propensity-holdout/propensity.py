import pandas as pd, numpy as np, json, sys, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
r=pd.read_parquet(sys.argv[1]); r.columns=[c.replace(' ','') for c in r.columns]
r=r.drop(columns=['source_sheet']).drop_duplicates()
r=r[r.CustomerID.notna()].copy(); r['CustomerID']=r.CustomerID.astype(int)
inv=r.Invoice.astype(str); sc=r.StockCode.astype(str)
nonprod=sc.str.match(r'^[A-Za-z]+$')|sc.str.upper().isin(['BANK CHARGES','C2','DOT','POST','M','ADJUST','ADJUST2','TEST001','TEST002','AMAZONFEE'])
r=r[~nonprod]
r['is_c']=inv.str.startswith('C'); r['amt']=r.Quantity*r.Price
sales=r[(~r.is_c)&(r.Quantity>0)&(r.Price>0)]
orders=sales.groupby(['CustomerID','Invoice']).agg(dt=('InvoiceDate','min'),amt=('amt','sum'),nsku=('StockCode','nunique')).reset_index()
canc=r[r.is_c].groupby('CustomerID').InvoiceDate.apply(list)
country=sales.groupby('CustomerID').Country.agg(lambda s:s.mode().iat[0])
H=90
def build(cut):
    cut=pd.Timestamp(cut); end=cut+pd.Timedelta(days=H)
    o=orders[orders.dt<cut]; f=orders[(orders.dt>=cut)&(orders.dt<end)]
    last=o.groupby('CustomerID').dt.max()
    elig=last[last>=cut-pd.Timedelta(days=365)].index
    o=o[o.CustomerID.isin(elig)]
    g=o.groupby('CustomerID')
    X=pd.DataFrame(index=elig)
    X['recency']=(cut-g.dt.max()).dt.days
    X['tenure']=(cut-g.dt.min()).dt.days
    for d in (90,180,365):
        od=o[o.dt>=cut-pd.Timedelta(days=d)].groupby('CustomerID')
        X[f'orders_{d}']=od.size().reindex(elig).fillna(0)
        X[f'spend_{d}']=od.amt.sum().reindex(elig).fillna(0)
    X['orders_all']=g.size(); X['aov']=g.amt.mean(); X['skus_per_order']=g.nsku.mean()
    gaps=o.sort_values('dt').groupby('CustomerID').dt.apply(lambda s: s.diff().dt.days.mean())
    X['mean_gap']=gaps.reindex(elig).fillna(X.tenure)
    X['uk']=(country.reindex(elig)=='United Kingdom').astype(int)
    cc=r[(r.is_c)&(r.InvoiceDate<cut)&(r.InvoiceDate>=cut-pd.Timedelta(days=365))].groupby('CustomerID').Invoice.nunique()
    X['cancels_365']=cc.reindex(elig).fillna(0)
    y=pd.Series(elig.isin(f.CustomerID.unique()).astype(int),index=elig)
    X['cutoff']=str(cut.date())
    return X,y
train_cuts=['2010-03-01','2010-06-01','2010-09-01','2010-12-01']; test_cuts=['2011-06-01','2011-09-01']
tr=[build(c) for c in train_cuts]; te={c:build(c) for c in test_cuts}
Xtr=pd.concat([x for x,_ in tr]); ytr=pd.concat([y for _,y in tr])
feats=[c for c in Xtr.columns if c!='cutoff']
lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)); lr.fit(np.log1p(Xtr[feats].clip(lower=0)),ytr)
gbm=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,random_state=7,verbose=-1); gbm.fit(Xtr[feats],ytr)
out={'train':{'cutoffs':train_cuts,'n':int(len(ytr)),'base_rate':float(ytr.mean())},'horizon_days':H,'tests':{}}
for c,(X,y) in te.items():
    p_lr=lr.predict_proba(np.log1p(X[feats].clip(lower=0)))[:,1]; p_g=gbm.predict_proba(X[feats])[:,1]
    rule=-X.recency.values  # RFM rule baseline: most recent first
    rule2=X.orders_365.values+1e-3*-X.recency.values
    res={'n':int(len(y)),'base_rate':float(y.mean())}
    for name,s in [('recency_rule',rule),('frequency_rule',rule2),('logistic',p_lr),('lightgbm',p_g)]:
        res[name]={'auc':float(roc_auc_score(y,s))}
        q=pd.qcut(pd.Series(s).rank(method='first'),10,labels=False)
        dec=pd.DataFrame({'d':9-q.values,'y':y.values})
        top=dec[dec.d==0].y.mean(); res[name]['top_decile_rate']=float(top); res[name]['top_decile_lift']=float(top/y.mean())
        res[name]['top3_capture']=float(dec[dec.d<=2].y.sum()/y.sum())
    for name,p in [('logistic',p_lr),('lightgbm',p_g)]:
        res[name]['brier']=float(brier_score_loss(y,p))
        q=pd.qcut(pd.Series(p).rank(method='first'),10,labels=False)
        cal=pd.DataFrame({'q':q.values,'p':p,'y':y.values}).groupby('q').agg(pred=('p','mean'),actual=('y','mean'),n=('y','size')).reset_index()
        res[name]['calibration']=[{'decile':int(10-a.q),'pred':round(float(a.pred),4),'actual':round(float(a.actual),4),'n':int(a.n)} for a in cal.itertuples()][::-1]
    out['tests'][c]=res
imp=pd.Series(gbm.booster_.feature_importance('gain'),index=feats); imp=imp/imp.sum()
out['importance']=[{'feature':k,'share':round(float(v),4)} for k,v in imp.sort_values(ascending=False).items()]
json.dump(out,open('propensity_results.json','w'),indent=1)
for c,res in out['tests'].items():
    print(c,res['n'],round(res['base_rate'],3),{k:{kk:round(vv,3) for kk,vv in v.items() if not isinstance(vv,list)} for k,v in res.items() if isinstance(v,dict)})
print(out['train'])
