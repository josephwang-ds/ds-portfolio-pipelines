import json
a=json.load(open('propensity_results.json')); b=json.load(open('holdout_results.json'))
for s in b['scores'].values():
    for r in s['deciles']:
        r['incremental_share']=round(r['lift']/r['treat'],4) if r['treat']>0 else None
json.dump({'propensity':a,'holdout':b},open('propensity_holdout_results.json','w'),indent=1)
d=b['scores']['s_resp']['deciles']; print([(r['decile'],r['treat'],r['lift'],r['incremental_share']) for r in d])
