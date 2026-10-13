import json, pandas as pd
d=json.load(open('did_results.json')); s=json.load(open('sc_results.json')); E=pd.read_parquet('treat_used.parquet')
D=pd.read_parquet('stacked.parquet')
weeks=list(pd.read_parquet('units.parquet').columns)
coh=E.groupby('t').agg(n=('sku','size'),chg=('chg','median')).reset_index()
coh['week']=[str(pd.Timestamp(weeks[t]).date()) for t in coh.t]
# raw means for the naive picture: treated vs control mean log units by rel
m=D.groupby(['treated','rel']).lu.mean().unstack(0)
out={'did':{k:d[k] for k in ['units_did','revenue_did','price_did','units_placebo','units_did_matched']},
 'naive':{'units':d['units_naive'],'revenue':d['revenue_naive'],'control_change':d['units_control_change']},
 'pretrend':d['pretrend_wald'],'event_study':d['event_study'],'elasticity':d['elasticity'],
 'mde_units':2.8*d['units_did']['se'],
 'counts':{'treated_skus':d['n_treated'],'control_units':d['n_control_units'],'control_skus':d['n_control_skus'],'matched_controls':d['matched_controls'],
           'price_change_mean':float(E.chg.mean()),'price_change_median':float(E.chg.median()),'n_cohorts':int(len(coh))},
 'cohorts':coh[['week','n','chg']].to_dict('records'),
 'raw_means':[{'rel':int(r),'treated':round(float(v[1]),3),'control':round(float(v[0]),3)} for r,v in m.iterrows()],
 'sc':s}
json.dump(out,open('pricing_did_results.json','w'),indent=1)
print(round(out['mde_units'],3), out['counts'])
