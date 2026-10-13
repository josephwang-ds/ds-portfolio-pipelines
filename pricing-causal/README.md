# Did the list-price increases cost volume? (UCI Online Retail II)

Source pipeline for the [/pricing-causal](https://www.josephjwang.com/pricing-causal) demo — an observational evaluation of real list-price increases, the way a pricing pilot without randomization would have to be evaluated.

## Events (`events.py`)

- Same cleaning as `analysis/demand-forecast` (dedupe, drop non-product lines, drop sales matched to a cancellation).
- Weekly list price per SKU = highest unit price paid by registered customers that week (lower prices are volume tiers); gaps of ≤ 4 weeks forward-filled.
- An event is a week where the list price changes and holds at the old level for the 8 weeks before and the new level for the 8 weeks after.
- Treated: increases of +2% to +25% (catalogue revisions). Large cuts (mostly −60% to −70%) are clearance of discontinued items and are excluded — their volume change is confounded with delisting.

## Design (`stack.py`, `est.py`)

- Stacked event study: for each event week, treated SKUs plus control SKUs whose list price was constant over ±12 weeks and that sold in ≥ 50% of the pre weeks. 51 treated SKUs across 12 event weeks; ~1,000 controls per event.
- Outcomes: log(1 + weekly units), log(1 + weekly revenue), and log realized unit price (first stage, weeks with sales).
- DiD: `y ~ treated×post | cohort×SKU + cohort×week`, SEs clustered by SKU (pyfixest).
- Event study with week −1 as reference; joint Wald test on pre-period coefficients.
- Placebo: pre-period only, fake event 6 weeks before the real one.
- Robustness: controls restricted to the treated SKUs' range of pre-period volume.

## Synthetic control (`sc.py`)

Largest cohort (2011-03-28, 18 SKUs). Each series indexed to its own 40-week pre-period mean and smoothed with a trailing 4-week mean; weights fit on the pre-period (non-negative, sum to 1) from the 60 donors most correlated in the pre-period. Inference by in-space placebos (each donor re-run as if treated, RMSPE post/pre ratio rank).

## Run

```bash
pip install pandas pyarrow numpy scipy pyfixest
python events.py path/to/online_retail_ii_combined.parquet
python stack.py && python est.py && python sc.py && python export.py
```
