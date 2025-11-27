# scheduler/ml_strategy_automated/config.py

"""
Configuration settings for the Automated ML strategy pipeline.
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
    "daily_futures": "raw_ingest.daily_futures",
    "daily_options": "raw_ingest.daily_options",
    "intraday_chart_dump": "raw_ingest.intraday_chart_dump",
    "fo_daily_unified": "raw_ingest.fo_daily_unified",
    "ml_signals_automated": "indicators.ml_signals_automated",
}

# --- Feature Engineering ---
TARGET_PERIODS_AHEAD = 16  # 4 hours (16 * 15 min)
TARGET_THRESHOLD = 0.02  # 2% move

# --- Model Training ---
TEST_SET_FRAC = 0.20
XGB_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 7,
    "n_estimators": 300,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "objective": "multi:softprob",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}
