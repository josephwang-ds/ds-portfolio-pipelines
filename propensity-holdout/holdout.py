import pandas as pd, numpy as np, json, sys, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
d=pd.read_csv(sys.argv[1])
d['treat']=(d.segment!='No E-Mail').astype(int)
X=pd.get_dummies(d[['recency','history','mens','womens','newbie','zip_code','channel']],columns=['zip_code','channel'],dtype=float)
rng=np.random.default_rng(7)
# split: half of each arm builds models, the other half is the evaluation set
d['fold']=rng.random(len(d))<0.5
def fit(mask):
    m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.03,num_leaves=15,min_child_samples=100,random_state=7,verbose=-1)
    m.fit(X[mask],d.visit[mask]); return m
m_ctrl=fit(d.fold&(d.treat==0))        # "would visit anyway" (baseline propensity)
m_resp=fit(d.fold&(d.treat==1))        # "visits when emailed" (typical response model)
ev=d[~d.fold].copy(); Xe=X[~d.fold]
ev['s_base']=m_ctrl.predict_proba(Xe)[:,1]; ev['s_resp']=m_resp.predict_proba(Xe)[:,1]
ev['s_uplift']=ev.s_resp-ev.s_base   # two-model uplift score
out={'n':int(len(d)),'n_eval':int(len(ev)),'treat_share':float(d.treat.mean()),
     'overall':{}, 'scores':{}}
def rate(x): return float(x.mean())
t=ev[ev.treat==1]; c=ev[ev.treat==0]
lift=rate(t.visit)-rate(c.visit); se=np.sqrt(rate(t.visit)*(1-rate(t.visit))/len(t)+rate(c.visit)*(1-rate(c.visit))/len(c))
out['overall']={'treat_visit':rate(t.visit),'ctrl_visit':rate(c.visit),'lift':lift,'ci':[lift-1.96*se,lift+1.96*se],'treat_conv':rate(t.conversion),'ctrl_conv':rate(c.conversion),'n_t':len(t),'n_c':len(c)}
out['auc']={'base_on_ctrl':float(roc_auc_score(c.visit,c.s_base)),'resp_on_treat':float(roc_auc_score(t.visit,t.s_resp))}
for s in ['s_resp','s_base','s_uplift']:
    q=pd.qcut(ev[s].rank(method='first'),10,labels=False); ev['dec']=10-q
    rows=[]
    for k,g in ev.groupby('dec'):
        gt=g[g.treat==1]; gc=g[g.treat==0]
        pt,pc=gt.visit.mean(),gc.visit.mean(); se=np.sqrt(pt*(1-pt)/len(gt)+pc*(1-pc)/len(gc))
        rows.append({'decile':int(k),'n_t':len(gt),'n_c':len(gc),'treat':round(float(pt),4),'ctrl':round(float(pc),4),'lift':round(float(pt-pc),4),'lo':round(float(pt-pc-1.96*se),4),'hi':round(float(pt-pc+1.96*se),4)})
    # cumulative targeting: top k deciles
    cum=[]
    for k in range(1,11):
        g=ev[ev.dec<=k]; gt=g[g.treat==1]; gc=g[g.treat==0]
        cum.append({'top_deciles':k,'share_contacted':k/10,'lift':round(float(gt.visit.mean()-gc.visit.mean()),4),'inc_visits_per_1000_customers':round(float((gt.visit.mean()-gc.visit.mean())*1000*k/10),2)})
    out['scores'][s]={'deciles':rows,'cumulative':cum}
json.dump(out,open('holdout_results.json','w'),indent=1)
print(json.dumps(out['overall'],indent=0)); print(out['auc'])
for s in out['scores']:
    print(s,[(r['decile'],r['ctrl'],r['lift']) for r in out['scores'][s]['deciles']])
    print('  cum',[(c['top_deciles'],c['inc_visits_per_1000_customers']) for c in out['scores'][s]['cumulative']])
