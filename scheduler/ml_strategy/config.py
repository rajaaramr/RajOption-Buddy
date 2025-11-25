# scheduler/ml_strategy/config.py

"""
Configuration settings for the ML strategy pipeline.
"""

import os

# --- Database Configuration ---
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5432")
DB_NAME = os.getenv("PGDATABASE", "TradeHub18")

CONN_STR = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Table Names ---
TABLES = {
    "spot_candles": "market.spot_candles",
    "futures_candles": "market.futures_candles",
    "spot_frames": "indicators.spot_frames",
    "futures_frames": "indicators.futures_frames",
    "daily_futures": "raw_ingest.daily_futures",
    "daily_options": "raw_ingest.daily_options",
    "symbol_meta": "reference.symbol_meta",
    "intraday_chart_dump": "raw_ingest.intraday_chart_dump",
    "ml_signals": "indicators.ml_signals_2pct_4h",
}

# --- Feature Engineering ---
TARGET_PERIODS_AHEAD = 16  # 4 hours (16 * 15 min)
TARGET_THRESHOLD = 0.02  # 2% move
CALIBRATION_FRAC = 0.7

# Bins for probability-style features
FEATURE_BINS = {
    "rsi_SPOT_15m": [0, 20, 30, 40, 50, 60, 70, 80, 100],
    "adx_SPOT_30m": [0, 10, 20, 25, 30, 40, 60, 100],
    "macd_hist_FUT_15m": [-2.0, -1.0, -0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5, 1.0, 2.0],
    "pcr_oi": [0, 0.6, 0.8, 1.0, 1.2, 1.5, 10],
    "dist_call_wall": [-1.0, -0.05, 0.0, 0.02, 0.05, 0.10, 1.0],
    "dist_put_wall": [-1.0, -0.10, -0.05, -0.02, 0.0, 0.02, 1.0],
    "avg_iv": [0, 10, 15, 20, 25, 30, 40, 60, 200],
}

# --- Model Training ---
TEST_SET_FRAC = 0.20
XGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 7,
    "n_estimators": 300,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

# Evaluation thresholds for binary classifiers
EVALUATION_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
FINAL_THRESHOLD_UP = 0.80
FINAL_THRESHOLD_DOWN = 0.80

# --- Output ---
MODEL_UP_PATH = "xgb_direction_up_2pct_4h.json"
MODEL_DOWN_PATH = "xgb_direction_down_2pct_4h.json"
FEATURE_NAMES_PATH = "model_feature_names_2pct_4h.npy"
SIGNALS_CSV_PATH = "high_precision_signals_backtest_up_down.csv"

# --- Model Versioning ---
MODEL_VERSION_UP = "xgb_up_2pct_4h_v1"
MODEL_VERSION_DOWN = "xgb_down_2pct_4h_v1"
