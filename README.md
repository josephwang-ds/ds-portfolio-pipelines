# Data science pipelines behind josephjwang.com

Reproducible Python pipelines for three demos on [josephjwang.com](https://www.josephjwang.com/method). All three use public data and report results as they came out — no tuning toward a target number, and each README states what the analysis cannot show.

| Folder | Question | Data | Methods | Demo |
|---|---|---|---|---|
| [`demand-forecast`](demand-forecast) | Does a better forecast improve replenishment? | UCI Online Retail II | Rolling-origin backtest, naive / moving average / seasonal naive / LightGBM, WMAPE and bias at SKU vs portfolio level, safety-stock simulation | [/demand-forecast](https://www.josephjwang.com/demand-forecast) |
| [`pricing-causal`](pricing-causal) | Did list-price increases cost volume? | UCI Online Retail II | Stacked difference-in-differences, event study, placebo, synthetic control with in-space placebos | [/pricing-causal](https://www.josephjwang.com/pricing-causal) |
| [`propensity-holdout`](propensity-holdout) | Are high-propensity customers the ones a campaign changes? | UCI Online Retail II, Hillstrom e-mail experiment | Out-of-time propensity model vs simple rules, calibration, lift by score decile on a randomized holdout | [/propensity-holdout](https://www.josephjwang.com/propensity-holdout) |

## Findings in one line each

- **Forecasting:** LightGBM beats a 4-week moving average per SKU (53.6% vs 59.1% WMAPE) but loses once aggregated (18.6% vs 17.0%) because of a −11.6% bias — pick the error measure by the decision the forecast serves.
- **Pricing:** a before/after read says units fell 11% after price increases; with a control group the estimate is +2% (95% CI −19% to +28%), and pre-trends suggest the increases were targeted.
- **Propensity:** a 0.79 out-of-time AUC model barely beats a one-line frequency rule (0.78); in the randomized test, 71% of the top decile's e-mailed visits would have happened anyway.

## Data

Not included in the repo. Download:

- UCI Online Retail II — https://archive.ics.uci.edu/dataset/502/online+retail+ii (combine both sheets into one parquet)
- Hillstrom MineThatData e-mail experiment — https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html

## Run

```bash
pip install -r requirements.txt
# then follow the README in each folder
```

Joseph Wang · [josephjwang.com](https://www.josephjwang.com) · [resume](https://www.josephjwang.com/resume)
