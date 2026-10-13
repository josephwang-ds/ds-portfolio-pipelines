# Propensity lists vs incrementality

Source pipeline for the [/propensity-holdout](https://www.josephjwang.com/propensity-holdout) demo. Two public datasets, two halves of one question.

## Part 1 — repurchase propensity with out-of-time validation (`propensity.py`)

UCI Online Retail II, registered customers only (same cleaning as the other analyses).

- Label: does a customer place an order in the 90 days after a cutoff date.
- Eligible list at each cutoff: customers with an order in the prior 365 days.
- Features as of the cutoff: recency, tenure, orders and spend in the last 90/180/365 days, lifetime orders, average order value, SKUs per order, mean days between orders, UK flag, cancellations in the last year.
- Train on cutoffs 2010-03-01, 2010-06-01, 2010-09-01, 2010-12-01 (all label windows end before the test cutoffs); test out of time on 2011-06-01 and 2011-09-01.
- Models: logistic regression, LightGBM; baselines: rank by recency, rank by orders in the last year.
- Metrics: AUC, top-decile response rate and lift, share of buyers captured in the top 30%, Brier score and decile calibration.

## Part 2 — does a high score mean a high incremental effect? (`holdout.py`)

Hillstrom MineThatData e-mail experiment (64,000 customers, randomized: men's e-mail / women's e-mail / no e-mail). Treatment = any e-mail; outcome = website visit within two weeks.

- Random 50/50 split. On the model half: a response model fit on e-mailed customers (the usual "who responds" score), a baseline model fit on the no-e-mail group ("who visits anyway"), and their difference as a two-model uplift score.
- On the held-out half: rank customers by each score into deciles; in every decile compare e-mailed vs no-e-mail visit rates (randomization makes this the causal lift), with 95% Wald CIs; cumulative incremental visits when targeting the top k deciles.

## Run

```bash
pip install pandas pyarrow numpy scikit-learn lightgbm
python propensity.py path/to/online_retail_ii_combined.parquet
python holdout.py path/to/hillstrom_email.csv
python export.py
```
