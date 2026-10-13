# Weekly demand forecasting → replenishment decision (UCI Online Retail II)

Source pipeline for the [/demand-forecast](https://www.josephjwang.com/demand-forecast) demo. Public data, no tuning toward a target number.

## Data

UCI Online Retail II (UK online wholesaler, Dec 2009 – Dec 2011), both sheets combined into one parquet.

Cleaning (`prep.py`):
1. Drop exact duplicate rows (the two sheets overlap in Dec 2010).
2. Keep sales lines with quantity > 0 and price > 0; drop non-product codes (postage, fees, manual adjustments).
3. Drop sales lines that have a matching cancellation (same customer, SKU, quantity) — removes huge orders that were later cancelled.
4. Aggregate to Monday-start weeks; drop the final partial week.
5. Keep SKUs that sold in ≥ 80% of weeks, then take the top 100 by units (≈ 27% of all units). Missing SKU-weeks are zeros.

## Backtest (`backtest.py`)

- Rolling origin, horizons 1–4 weeks. Evaluation window: 25 origins (2011-05-16 … 2011-10-31), targets 2011-05-23 … 2011-11-28, 10,000 SKU-week forecasts.
- Methods: last week (naive), 4-week moving average, seasonal naive (lag 52), LightGBM global model (Tweedie loss, one fixed config, retrained at every origin on data available at that origin only).
- Features use only weeks ≤ origin: lags 0–3, 4/8/13-week means, 8-week std, same-week-last-year values, target week-of-year, horizon, SKU id.
- Metric: WMAPE = Σ|actual − forecast| / Σ actual, at SKU-week level and at portfolio-week level; bias = Σ(forecast − actual) / Σ actual.

## Decision layer (`decide.py`)

Every 4th origin (7 non-overlapping 4-week cycles), order = 4-week forecast + trailing mean error + z × trailing std of 4-week errors per SKU (only errors fully observed before the order date). Reports fill rate, achieved cycle service, excess and short units/value at 80/90/95% targets. Simplification: each cycle starts from zero stock (no carry-over), so absolute excess is overstated; compare methods, not levels.

## Run

```bash
pip install pandas pyarrow lightgbm numpy
python prep.py path/to/online_retail_ii_combined.parquet
python backtest.py && python decide.py && python biascorr.py && python export.py && python finalize.py
```

Deterministic (`random_state=7`).
