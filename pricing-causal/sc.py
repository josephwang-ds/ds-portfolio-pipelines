import pandas as pd, numpy as np, json
from scipy.optimize import minimize
E=pd.read_parquet('treat_used.parquet'); lp=pd.read_parquet('listprice.parquet'); U=pd.read_parquet('units.parquet')
weeks=list(U.columns); T=len(weeks)
lpf=lp.ffill(axis=1,limit=4).reindex(U.index)
t=int(E.groupby('t').size().idxmax()); PRE=40; POST=12
tsk=E[E.t==t].sku.tolist()
P=lpf.iloc[:,t-PRE:t+POST]
stable=P.notna().sum(axis=1)>=int(0.8*(PRE+POST)); const=P.max(axis=1).round(2)==P.min(axis=1).round(2)
act=(U.iloc[:,t-PRE:t]>0).mean(axis=1)>=0.5
donors=U.index[stable&const&act].difference(tsk)
# index each series to its own pre-period mean (shape matching), treated = mean of indexed treated SKUs
def idx(k): x=U.loc[k].iloc[t-PRE:t+POST].values.astype(float); return x/max(x[:PRE].mean(),1e-9)
Y1=np.mean([idx(k) for k in tsk if U.loc[k].iloc[t-PRE:t].mean()>0],axis=0)
Y0=np.array([idx(k) for k in donors])
# smooth with 4-week rolling mean to reduce noise
sm=lambda a: pd.Series(a).rolling(4,min_periods=1).mean().values
Y1s=sm(Y1); Y0s=np.array([sm(r) for r in Y0])
def fit(y, X):
    n=X.shape[0]; w0=np.full(n,1/n)
    f=lambda w: ((y[:PRE]-w@X[:,:PRE])**2).sum()
    r=minimize(f,w0,method='SLSQP',bounds=[(0,1)]*n,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':500})
    return r.x
# restrict donor pool to 60 donors most correlated in pre-period to keep it tractable & interpretable
corr=np.array([np.corrcoef(Y1s[:PRE],r[:PRE])[0,1] for r in Y0s]); corr=np.nan_to_num(corr,nan=-1)
top=np.argsort(-corr)[:60]; X=Y0s[top]
w=fit(Y1s,X); syn=w@X
gap=Y1s-syn
rmspe=lambda g,a,b: np.sqrt((g[a:b]**2).mean())
ratio=rmspe(gap,PRE,PRE+POST)/rmspe(gap,0,PRE)
post_eff=float((Y1s[PRE:]-syn[PRE:]).mean()/syn[PRE:].mean())
# placebo in space: each donor as fake treated, others as donors
ratios=[]
for i in range(len(top)):
    yi=X[i]; Xi=np.delete(X,i,0); wi=fit(yi,Xi); gi=yi-wi@Xi
    ratios.append(rmspe(gi,PRE,PRE+POST)/max(rmspe(gi,0,PRE),1e-9))
p_rank=float((np.array(ratios)>=ratio).mean()+1/(len(ratios)+1))
out={'cohort_week':str(pd.Timestamp(weeks[t]).date()),'n_treated':len(tsk),'n_donor_pool':int(len(donors)),'n_donors_used':int((w>0.01).sum()),
 'pre_weeks':PRE,'post_weeks':POST,'pre_rmspe':float(rmspe(gap,0,PRE)),'post_effect_pct':post_eff,'rmspe_ratio':float(ratio),'placebo_p':p_rank,
 'series':[{'week':str(pd.Timestamp(weeks[t-PRE+i]).date()),'treated':round(float(Y1s[i]),3),'synthetic':round(float(syn[i]),3)} for i in range(PRE+POST)],
 'treated_skus':tsk,'price_change_median':float(E[E.t==t].chg.median())}
json.dump(out,open('sc_results.json','w'),indent=1)
print({k:v for k,v in out.items() if k not in ('series','treated_skus')})
print('top weights',sorted(zip(w.round(3),[donors[top[i]] for i in range(len(top))]),reverse=True)[:5])
