# Trend Pillar - ML Operations

This directory contains the ML infrastructure for the Trend Pillar.

## Files

- **`trend_targets.ini`**: Configuration for ML targets (e.g., `trend_ml.target.bull_2pct_4h`).
- **`trend_train.py`**: Trains XGBoost models for defined targets.
  - Loads data from `market.futures_candles`.
  - Computes features via `TrendFeatureEngine`.
  - Saves models to `models/trend/`.
  - Writes in-sample scores to `indicators.ml_pillars`.
- **`trend_mlscore_backfill.py`**: Loads saved models and backfills scores for historical data.
- **`trend_bucket_stats.py`**: Calibrates raw probabilities against realized returns and populates `indicators.trend_calibration_4h`.

## Routine (TRD-24)

To refresh ML models and calibration (e.g., Monthly/Quarterly):

1. **Train Models:**
   ```bash
   python3 Trend/ML/trend_train.py --lookback-days 365
   ```

2. **Generate Calibration Stats:**
   ```bash
   python3 Trend/ML/trend_bucket_stats.py
   ```

3. **(Optional) Backfill History:**
   If you need to re-score the entire history with the new model:
   ```bash
   python3 Trend/ML/trend_mlscore_backfill.py --days 365
   ```
