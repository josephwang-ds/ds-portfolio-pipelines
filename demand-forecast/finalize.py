"""Merge decision + bias-correction results into forecast_results.json (run after export.py)."""
import json, pandas as pd, numpy as np
o = json.load(open("forecast_results.json"))
R = json.load(open("decision.json"))
for r in R:
    for k, v in list(r.items()):
        if isinstance(v, float):
            r[k] = round(v, 4) if v < 10 else round(v)
weeks = sorted(pd.read_parquet("panel.parquet").week.unique())
bt = pd.read_parquet("backtest.parquet")
cyc = sorted(bt.origin.unique())[::4]
o["decision"] = {"cycles": [str(pd.Timestamp(weeks[c]).date()) for c in cyc], "rows": R}
bc = pd.read_parquet("backtest_bc.parquet")
wm = lambda d, c: float(np.abs(d[c] - d.y).sum() / d.y.sum())
agg = bc.groupby(["origin", "h", "target"])[["y", "gbm_bc"]].sum().reset_index()
o["bias_corrected"] = {"sku_wmape": round(wm(bc, "gbm_bc"), 4), "portfolio_wmape": round(wm(agg, "gbm_bc"), 4),
    "bias": round(float((bc.gbm_bc - bc.y).sum() / bc.y.sum()), 4),
    "factor_range": [round(float(bc.k.min()), 3), round(float(bc.k.max()), 3)]}
for e in o["error_drivers"]:
    for k in e:
        if hasattr(e[k], "item"):
            e[k] = e[k].item()
json.dump(o, open("forecast_results.json", "w"), ensure_ascii=False, indent=1)
print("ok")
