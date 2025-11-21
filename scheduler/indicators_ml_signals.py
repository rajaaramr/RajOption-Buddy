"""
ML: ±2% move in 4 Hours (XGBoost) with nRSI / nADX / nMACD / nF&O

- Loads data from Postgres
- Builds consolidated intraday feature set (spot, futures, frames, daily_option)
- Creates probability-style features (UP side calibration):
    nRSI_prob_2pct_up
    nADX_prob_2pct_up
    nMACD_prob_2pct_up
    nFO_pcr_prob_2pct_up
    nFO_callwall_prob_2pct_up
    nFO_putwall_prob_2pct_up
    nFO_iv_prob_2pct_up
- Defines targets:
    Direction_2pct_up   (future_ret_4h >= +2%)
    Direction_2pct_down (future_ret_4h <= -2%)
- Trains two XGB classifiers (UP / DOWN)
- Saves models + feature names
- Writes test-set signals back into Postgres (indicators.ml_signals_2pct_4h)
"""

import os
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sqlalchemy import create_engine

# =====================================================================
# 0. DB ENGINE
# =====================================================================

def get_engine():
    """
    Build a SQLAlchemy engine for PostgreSQL.
    Use env vars in your .env / system where possible.
    """
    DB_USER = os.getenv("PGUSER", "postgres")
    DB_PASSWORD = os.getenv("PGPASSWORD", "Ajantha18")
    DB_HOST = os.getenv("PGHOST", "localhost")
    DB_PORT = os.getenv("PGPORT", "5432")
    DB_NAME = os.getenv("PGDATABASE", "TradeHub18")

    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(conn_str)

# =====================================================================
# 1. RAW LOAD FROM POSTGRES
# =====================================================================

def load_raw_from_postgres():
    """
    Loads all required tables from Postgres.

    ⚠️ Adjust schema-qualified table names if needed:
        market.spot_candles, market.futures_candles,
        indicators.spot_frames, indicators.futures_frames,
        options.daily_option, reference.symbol_meta
    """
    engine = get_engine()

    print("[INFO] Loading core tables from Postgres...")

    # Intraday spot candles (15m)
    df_spot_candles = pd.read_sql(
        """
        SELECT ts, symbol, interval, close, volume
        FROM market.spot_candles
        WHERE interval = '15m'
        """,
        engine,
        parse_dates=["ts"],
    )

    # Intraday futures candles (15m)
    df_fut_candles = pd.read_sql(
        """
        SELECT ts, symbol, interval, close, oi
        FROM market.futures_candles
        WHERE interval = '15m'
        """,
        engine,
        parse_dates=["ts"],
    )

    # Indicator frames (spot & futures, all TFs)
    df_spot_frames = pd.read_sql(
        "SELECT * FROM indicators.spot_frames",
        engine,
        parse_dates=["ts"],
    )

    df_fut_frames = pd.read_sql(
        "SELECT * FROM indicators.futures_frames",
        engine,
        parse_dates=["ts"],
    )

    # Daily futures (optional, reserved for later features)
    df_daily_fut = pd.read_sql(
        "SELECT * FROM raw_ingest.daily_futures",
        engine,
        parse_dates=["trade_date"],
    )

    # Daily option chain snapshot (your table)
    df_daily_opt = pd.read_sql(
        "SELECT * FROM raw_ingest.daily_options",   # ← table name from your description
        engine,
    )

    # Symbol metadata (industry, sector)
    df_meta = pd.read_sql(
        "SELECT symbol, industry_name, sector_name FROM reference.symbol_meta",
        engine,
    )

    print("[INFO] Raw data load complete.")
    return (
        df_spot_candles,
        df_fut_candles,
        df_spot_frames,
        df_fut_frames,
        df_daily_fut,
        df_daily_opt,
        df_meta,
        engine,
    )


# =====================================================================
# 2. DAILY OPTION → SYMBOL+DAY F&O FEATURES
# =====================================================================

def build_daily_option_features(df_daily_opt: pd.DataFrame) -> pd.DataFrame:
    """
    Input: raw_ingest.daily_options
       columns: symbol, option_type(CE/PE), strike_price, spot,
                oi, oi_change, iv, ... , trade_date (or ts to derive date)

    Output: one row per symbol + trade_date with:
      - spot_ref
      - total_call_oi, total_put_oi, pcr_oi
      - call_wall_strike, put_wall_strike
      - dist_call_wall, dist_put_wall
      - avg_iv, net_oi_change, etc.
    """

    df = df_daily_opt.copy()

    # Ensure trade_date column exists
    if "trade_date" not in df.columns:
        if "ts" in df.columns:
            df["trade_date"] = pd.to_datetime(df["ts"]).dt.date
        else:
            raise ValueError("Need a trade_date or ts column in options.daily_option")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = df["symbol"].astype(str)

    # Split CE / PE
    ce = df[df["option_type"] == "CE"].copy()
    pe = df[df["option_type"] == "PE"].copy()

    group_cols = ["trade_date", "symbol"]

    # Spot per symbol/day (assume same across strikes)
    spot_ref = (
        df.groupby(group_cols)["spot"]
        .first()
        .rename("spot_ref")
        .reset_index()
    )

    # --- CE aggregates ---
    ce_agg = (
        ce.groupby(group_cols)
        .agg(
            total_call_oi=("oi", "sum"),
            total_call_oi_change=("oi_change", "sum"),
            avg_call_iv=("iv", "mean"),
        )
        .reset_index()
    )

    # Call wall: strike with max OI
    ce_wall = (
        ce.sort_values(["trade_date", "symbol", "oi"])
        .groupby(group_cols)
        .tail(1)[["trade_date", "symbol", "strike_price", "oi"]]
        .rename(
            columns={
                "strike_price": "call_wall_strike",
                "oi": "call_wall_oi",
            }
        )
    )

    # --- PE aggregates ---
    pe_agg = (
        pe.groupby(group_cols)
        .agg(
            total_put_oi=("oi", "sum"),
            total_put_oi_change=("oi_change", "sum"),
            avg_put_iv=("iv", "mean"),
        )
        .reset_index()
    )

    pe_wall = (
        pe.sort_values(["trade_date", "symbol", "oi"])
        .groupby(group_cols)
        .tail(1)[["trade_date", "symbol", "strike_price", "oi"]]
        .rename(
            columns={
                "strike_price": "put_wall_strike",
                "oi": "put_wall_oi",
            }
        )
    )

    # Combine CE + PE + spot
    df_sym_day = (
        spot_ref.merge(ce_agg, on=group_cols, how="left")
        .merge(pe_agg, on=group_cols, how="left")
        .merge(ce_wall, on=group_cols, how="left")
        .merge(pe_wall, on=group_cols, how="left")
    )

    # Derived features
    df_sym_day["pcr_oi"] = df_sym_day["total_put_oi"] / df_sym_day["total_call_oi"]

    df_sym_day["dist_call_wall"] = (
        (df_sym_day["call_wall_strike"] - df_sym_day["spot_ref"])
        / df_sym_day["spot_ref"]
    )
    df_sym_day["dist_put_wall"] = (
        (df_sym_day["spot_ref"] - df_sym_day["put_wall_strike"])
        / df_sym_day["spot_ref"]
    )

    df_sym_day["net_oi_change"] = (
        df_sym_day["total_call_oi_change"].fillna(0)
        + df_sym_day["total_put_oi_change"].fillna(0)
    )

    df_sym_day["avg_iv"] = (
        df_sym_day[["avg_call_iv", "avg_put_iv"]]
        .mean(axis=1, skipna=True)
    )

    return df_sym_day


# =====================================================================
# 3. CONSOLIDATE INTRADAY SPOT/FUT + FRAMES + DAILY OPTION
# =====================================================================

def build_consolidated_features(
    df_spot_candles,
    df_fut_candles,
    df_spot_frames,
    df_fut_frames,
    df_daily_fut,
    df_daily_opt,
    df_meta,
):
    """
    Build df_consolidated with:

      index: (ts, symbol)
      columns: close_SPOT, close_FUT, volume_SPOT, oi_FUT,
               stoch_k_FUT_15m, rsi_SPOT_15m,
               macd_hist_FUT_15m, bb_score_FUT_15m,
               adx_SPOT_30m, mfi_14_SPOT_60m,
               plus daily F&O features (pcr_oi, dist_call_wall, dist_put_wall, avg_iv, ...)
    """

    # ---- Base: 15m spot ----
    spot_15 = df_spot_candles.copy()
    spot_15["ts"] = pd.to_datetime(spot_15["ts"])
    spot_15["symbol"] = spot_15["symbol"].astype(str)

    spot_15 = spot_15.rename(
        columns={
            "close": "close_SPOT",
            "volume": "volume_SPOT",
        }
    )

    # ---- 15m futures ----
    fut_15 = df_fut_candles.copy()
    fut_15["ts"] = pd.to_datetime(fut_15["ts"])
    fut_15["symbol"] = fut_15["symbol"].astype(str)

    fut_15 = fut_15.rename(
        columns={
            "close": "close_FUT",
            "oi": "oi_FUT",
        }
    )

    # Merge spot + futures
    df = spot_15.merge(
        fut_15[["ts", "symbol", "close_FUT", "oi_FUT"]],
        on=["ts", "symbol"],
        how="inner",
    )

    # ---- Futures indicators 15m ----
    fut_15_ind = df_fut_frames[df_fut_frames["interval"] == "15m"].copy()
    fut_15_ind["ts"] = pd.to_datetime(fut_15_ind["ts"])
    fut_15_ind["symbol"] = fut_15_ind["symbol"].astype(str)

    fut_15_ind = fut_15_ind.rename(
        columns={
            "stoch_k": "stoch_k_FUT_15m",
            "macd_hist": "macd_hist_FUT_15m",
            "bb_score": "bb_score_FUT_15m",  # adjust if your BB column is differently named
        }
    )

    df = df.merge(
        fut_15_ind[["ts", "symbol", "stoch_k_FUT_15m", "macd_hist_FUT_15m", "bb_score_FUT_15m"]],
        on=["ts", "symbol"],
        how="left",
    )

    # ---- Spot indicators 15m RSI ----
    spot_15_ind = df_spot_frames[df_spot_frames["interval"] == "15m"].copy()
    spot_15_ind["ts"] = pd.to_datetime(spot_15_ind["ts"])
    spot_15_ind["symbol"] = spot_15_ind["symbol"].astype(str)

    spot_15_ind = spot_15_ind.rename(
        columns={
            "rsi": "rsi_SPOT_15m",
        }
    )

    df = df.merge(
        spot_15_ind[["ts", "symbol", "rsi_SPOT_15m"]],
        on=["ts", "symbol"],
        how="left",
    )

    # ---- Spot ADX 30m (forward-fill within symbol) ----
    spot_30_ind = df_spot_frames[df_spot_frames["interval"] == "30m"].copy()
    spot_30_ind["ts"] = pd.to_datetime(spot_30_ind["ts"])
    spot_30_ind["symbol"] = spot_30_ind["symbol"].astype(str)

    spot_30_ind = spot_30_ind.rename(columns={"adx": "adx_SPOT_30m"})
    spot_30_ind = spot_30_ind.set_index(["symbol", "ts"]).sort_index()
    spot_30_ind = spot_30_ind.groupby(level="symbol").ffill().reset_index()

    df = df.merge(
        spot_30_ind[["ts", "symbol", "adx_SPOT_30m"]],
        on=["ts", "symbol"],
        how="left",
    )

    # ---- Spot MFI 60m (forward-fill within symbol) ----
    spot_60_ind = df_spot_frames[df_spot_frames["interval"] == "60m"].copy()
    spot_60_ind["ts"] = pd.to_datetime(spot_60_ind["ts"])
    spot_60_ind["symbol"] = spot_60_ind["symbol"].astype(str)

    spot_60_ind = spot_60_ind.rename(columns={"mfi_14": "mfi_14_SPOT_60m"})
    spot_60_ind = spot_60_ind.set_index(["symbol", "ts"]).sort_index()
    spot_60_ind = spot_60_ind.groupby(level="symbol").ffill().reset_index()

    df = df.merge(
        spot_60_ind[["ts", "symbol", "mfi_14_SPOT_60m"]],
        on=["ts", "symbol"],
        how="left",
    )

    # ---- Attach daily option features by trade_date ----
    df_opt_sym_day = build_daily_option_features(df_daily_opt)

    df = df.sort_values(["ts", "symbol"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["symbol"] = df["symbol"].astype(str)

    tmp = df.copy()
    # trade_date as UTC-normalized date
    tmp["trade_date"] = tmp["ts"].dt.normalize()

    df_opt_sym_day = df_opt_sym_day.rename(columns={"spot_ref": "spot_daily"})

    if not df_opt_sym_day.empty:
        df_opt_sym_day["trade_date"] = pd.to_datetime(df_opt_sym_day["trade_date"])
        # localize to UTC only if it's naive
        if df_opt_sym_day["trade_date"].dt.tz is None:
            df_opt_sym_day["trade_date"] = df_opt_sym_day["trade_date"].dt.tz_localize("UTC")

        tmp = tmp.merge(
            df_opt_sym_day,
            left_on=["trade_date", "symbol"],
            right_on=["trade_date", "symbol"],
            how="left",
        )
        tmp = tmp.drop(columns=["trade_date"])
    else:
        # just drop helper column if no options data
        tmp = tmp.drop(columns=["trade_date"])

    df_final = tmp.sort_values(["ts", "symbol"]).set_index(["ts", "symbol"])

    return df_final, df_meta


# =====================================================================
# 4. PROBABILITY-STYLE FEATURE HELPERS
# =====================================================================

def add_prob_feature_from_indicator(
    df,
    value_col: str,
    target_col: str = "Direction_2pct_up",
    bins=None,
    calib_frac: float = 0.7,
    new_col: str = "nX_prob_2pct_up",
):
    """
    Convert a continuous indicator into a probability-style feature:
      new_col = P(target_col=1 | value_col in bin)
    estimated on an early slice of the history.
    """
    if bins is None:
        raise ValueError("bins required")

    # work on a flat df with ts, symbol in columns
    tmp = df.reset_index().sort_values("ts")

    calib_end = int(len(tmp) * calib_frac)
    calib = tmp.iloc[:calib_end].copy()
    calib = calib.dropna(subset=[value_col, target_col])

    # if we don't have enough calibration data, just return a flat 0.5 series
    if calib.empty:
        out = pd.Series(0.5, index=df.index, name=new_col)
        return out

    # bin in calibration slice
    calib["bin"] = pd.cut(calib[value_col], bins=bins, right=False)
    # explicit observed=False to silence warning & keep old behavior
    prob_by_bin = calib.groupby("bin", observed=False)[target_col].mean()

    # bin full data
    tmp["bin"] = pd.cut(tmp[value_col], bins=bins, right=False)

    # map bin → prob, then coerce to float (avoid Categorical dtype)
    prob_series = tmp["bin"].map(prob_by_bin)
    prob_series = pd.to_numeric(prob_series, errors="coerce")

    # global fallback prob
    global_prob = float(calib[target_col].mean())
    if math.isnan(global_prob):
        global_prob = 0.5

    prob_series = prob_series.fillna(global_prob).astype("float64")

    tmp[new_col] = prob_series

    # return aligned back to original multi-index
    return tmp.set_index(["ts", "symbol"])[new_col]


# =====================================================================
# 5. FEATURE ENGINEERING + TRAIN/TEST SPLIT
# =====================================================================

def build_train_test_from_consolidated(df_consolidated, df_meta):
    df = df_consolidated.copy()

    # 5.1 LAG FEATURES
    lag_cols = [
        "close_SPOT", "close_FUT", "volume_SPOT", "oi_FUT",
        "stoch_k_FUT_15m", "rsi_SPOT_15m",
        "macd_hist_FUT_15m", "bb_score_FUT_15m",
        "adx_SPOT_30m", "mfi_14_SPOT_60m",
    ]
    lags = [1, 2, 4]

    for col in lag_cols:
        for lag in lags:
            df[f"{col}_LAG_{lag}"] = df.groupby(level="symbol")[col].shift(lag)

    # 5.2 TARGETS: ±2% in next 4 hours (16 x 15m)
    PERIODS_AHEAD = 16
    THRESHOLD = 0.00

    df["future_close_SPOT"] = df.groupby(level="symbol")["close_SPOT"].shift(-PERIODS_AHEAD)
    df["future_ret_4h"] = (df["future_close_SPOT"] / df["close_SPOT"]) - 1.0

    # Bullish label
    df["Direction_2pct_up"] = 0
    df.loc[df["future_ret_4h"] > THRESHOLD, "Direction_2pct_up"] = 1

    # Bearish label
    df["Direction_2pct_down"] = 0
    df.loc[df["future_ret_4h"] < -THRESHOLD, "Direction_2pct_down"] = 1

    # 5.3 BASE F&O EXTRA: simple OI zscore (for nFO_oi if needed later)
    df = df.reset_index()
    df["ret_1"] = df.groupby("symbol")["close_SPOT"].pct_change(1)
    df["oi_mean_20"] = (
        df.groupby("symbol")["oi_FUT"].rolling(20).mean().reset_index(level=0, drop=True)
    )
    df["oi_std_20"] = (
        df.groupby("symbol")["oi_FUT"].rolling(20).std().reset_index(level=0, drop=True)
    )
    df["oi_zscore_20"] = (df["oi_FUT"] - df["oi_mean_20"]) / (df["oi_std_20"] + 1e-9)
    df = df.set_index(["ts", "symbol"])

    # 5.4 nRSI / nADX / nMACD / nF&O (PCR / walls / IV) — all calibrated vs UP label

    # nRSI
    rsi_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    df["nRSI_prob_2pct_up"] = add_prob_feature_from_indicator(
        df,
        value_col="rsi_SPOT_15m",
        target_col="Direction_2pct_up",
        bins=rsi_bins,
        calib_frac=0.7,
        new_col="nRSI_prob_2pct_up",
    )

    # nADX
    adx_bins = [0, 10, 20, 25, 30, 40, 60, 100]
    df["nADX_prob_2pct_up"] = add_prob_feature_from_indicator(
        df,
        value_col="adx_SPOT_30m",
        target_col="Direction_2pct_up",
        bins=adx_bins,
        calib_frac=0.7,
        new_col="nADX_prob_2pct_up",
    )

    # nMACD
    macd_bins = [-2.0, -1.0, -0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5, 1.0, 2.0]
    df["nMACD_prob_2pct_up"] = add_prob_feature_from_indicator(
        df,
        value_col="macd_hist_FUT_15m",
        target_col="Direction_2pct_up",
        bins=macd_bins,
        calib_frac=0.7,
        new_col="nMACD_prob_2pct_up",
    )

    # nF&O : PCR
    pcr_bins = [0, 0.6, 0.8, 1.0, 1.2, 1.5, 10]
    df["nFO_pcr_prob_2pct_up"] = add_prob_feature_from_indicator(
        df,
        value_col="pcr_oi",
        target_col="Direction_2pct_up",
        bins=pcr_bins,
        calib_frac=0.7,
        new_col="nFO_pcr_prob_2pct_up",
    )

    # nF&O : distance to call wall
    call_bins = [-1.0, -0.05, 0.0, 0.02, 0.05, 0.10, 1.0]
    df["nFO_callwall_prob_2pct_up"] = add_prob_feature_from_indicator(
        df,
        value_col="dist_call_wall",
        target_col="Direction_2pct_up",
        bins=call_bins,
        calib_frac=0.7,
        new_col="nFO_callwall_prob_2pct_up",
    )

    # nF&O : distance to put wall
    put_bins = [-1.0, -0.10, -0.05, -0.02, 0.0, 0.02, 1.0]
    df["nFO_putwall_prob_2pct_up"] = add_prob_feature_from_indicator(
        df,
        value_col="dist_put_wall",
        target_col="Direction_2pct_up",
        bins=put_bins,
        calib_frac=0.7,
        new_col="nFO_putwall_prob_2pct_up",
    )

    # nF&O : IV regime
    iv_bins = [0, 10, 15, 20, 25, 30, 40, 60, 200]
    df["nFO_iv_prob_2pct_up"] = add_prob_feature_from_indicator(
        df,
        value_col="avg_iv",
        target_col="Direction_2pct_up",
        bins=iv_bins,
        calib_frac=0.7,
        new_col="nFO_iv_prob_2pct_up",
    )

    # 5.5 META + OHE
    df = df.reset_index()
    df_meta = df_meta[["symbol", "industry_name", "sector_name"]].drop_duplicates("symbol")
    df = df.merge(df_meta, on="symbol", how="left").set_index(["ts", "symbol"])
    df = pd.get_dummies(df, columns=["industry_name", "sector_name"], drop_first=True)

    # 5.6 X / y, time-based split (one X, two y's)

    y_up = df["Direction_2pct_up"]
    y_down = df["Direction_2pct_down"]

    drop_cols = [
        "Direction_2pct_up",
        "Direction_2pct_down",
        "close_SPOT",
        "future_close_SPOT",
        "future_ret_4h",
    ] + lag_cols
    drop_cols += ["trade_date", "Direction", "interval"]

    # First drop known non-feature columns
    X = df.drop(columns=drop_cols, errors="ignore")

    # 🔐 Keep only numeric + bool dtypes for XGBoost
    X = X.select_dtypes(include=["number", "bool"]).copy()

    # Clean NaNs — keep all rows, just neutralize bad values
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_up_clean = y_up.loc[X_clean.index]
    y_down_clean = y_down.loc[X_clean.index]

    df_all = X_clean.copy()
    df_all["target_up"] = y_up_clean
    df_all["target_down"] = y_down_clean
    df_all = df_all.reset_index().sort_values("ts")

    split_ts = df_all["ts"].quantile(0.80)
    train_df = df_all[df_all["ts"] <= split_ts]
    test_df = df_all[df_all["ts"] > split_ts]

    X_train = train_df.drop(columns=["target_up", "target_down", "ts", "symbol"])
    y_up_train = train_df["target_up"]
    y_down_train = train_df["target_down"]

    X_test = test_df.drop(columns=["target_up", "target_down", "ts", "symbol"])
    y_up_test = test_df["target_up"]
    y_down_test = test_df["target_down"]

    # Also return the index for writing signals later
    test_index = test_df.set_index(["ts", "symbol"]).index

    return X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test, test_index


def load_data_from_postgres():
    try:
        (
            df_spot_candles,
            df_fut_candles,
            df_spot_frames,
            df_fut_frames,
            df_daily_fut,
            df_daily_opt,
            df_meta,
            engine,
        ) = load_raw_from_postgres()
    except Exception as e:
        print(f"[ERROR] Failed to load raw data from Postgres: {e}")
        return None, None, None, None, None, None, None, None

    df_consolidated, df_meta = build_consolidated_features(
        df_spot_candles,
        df_fut_candles,
        df_spot_frames,
        df_fut_frames,
        df_daily_fut,
        df_daily_opt,
        df_meta,
    )

    (
        X_train,
        X_test,
        y_up_train,
        y_up_test,
        y_down_train,
        y_down_test,
        test_index,
    ) = build_train_test_from_consolidated(df_consolidated, df_meta)

    return X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test, test_index, engine


# =====================================================================
# 6. TRAIN + EVAL (GENERIC BINARY) + WRITE SIGNALS
# =====================================================================

def train_xgb_binary(X_train, X_test, y_train, y_test, label_name: str):
    # ---- sanity check on labels ----
    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()
    print(f"[INFO] {label_name} train label counts: {train_counts}")
    print(f"[INFO] {label_name} test  label counts: {test_counts}")

    # If only one class in train, XGBoost logistic will blow up (base_score=0 or 1)
    if y_train.nunique() < 2:
        print(f"[WARN] {label_name}: y_train has a single class "
              f"{list(train_counts.keys())}. Skipping XGBoost training.")

        # Fallback: constant probability = base rate (or 0 if even that fails)
        base_p = float(y_train.mean()) if len(y_train) > 0 else 0.0
        y_proba = np.full(shape=len(y_test), fill_value=base_p, dtype="float64")

        results = {
            "NO_MODEL": {
                "Precision": "N/A",
                "Total_Trades": 0,
                "BaseRate": f"{base_p * 100:.2f}%",
            }
        }
        return None, y_proba, results

    best_params = {
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

    model = xgb.XGBClassifier(**best_params)

    print(f"\n[INFO] Training XGBoost model for {label_name} ...")
    model.fit(X_train, y_train)
    print(f"[INFO] Training complete for {label_name}.")

    y_proba = model.predict_proba(X_test)[:, 1]

    df_thr = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Proba": y_proba,
        }
    )

    thresholds_to_check = [0.50, 0.60, 0.70, 0.80, 0.90]
    results = {}

    for threshold in thresholds_to_check:
        df_thr["Signal"] = (df_thr["Proba"] > threshold).astype(int)
        trades = df_thr[df_thr["Signal"] == 1]

        if len(trades) > 0:
            correct_trades = trades[trades["Actual"] == 1]
            precision = len(correct_trades) / len(trades)
            results[threshold] = {
                "Precision": f"{precision * 100:.2f}%",
                "Total_Trades": len(trades),
            }
        else:
            results[threshold] = {"Precision": "N/A", "Total_Trades": 0}

    base_rate = y_test.mean() * 100
    print(f"\n[STATS] {label_name}: base positive rate in test set = {base_rate:.2f}%")
    print(f"[STATS] {label_name}: precision summary by threshold:")
    print(pd.DataFrame(results).T)

    thr_eval = 0.80
    y_pred_thr = (df_thr["Proba"] > thr_eval).astype(int)
    print(f"\n=== Detailed metrics for {label_name} @ threshold {thr_eval:.2f} ===")
    print(confusion_matrix(df_thr["Actual"], y_pred_thr))
    print(classification_report(df_thr["Actual"], y_pred_thr, digits=3))

    return model, y_proba, results


def main():
    (
        X_train,
        X_test,
        y_up_train,
        y_up_test,
        y_down_train,
        y_down_test,
        test_index,
        engine,
    ) = load_data_from_postgres()

    if X_train is None:
        print("[FATAL] Aborting due to data load failure.")
        return

    # -----------------------------------------------------
    # Train UP model
    # -----------------------------------------------------
    model_up, y_proba_up, results_up = train_xgb_binary(
        X_train, X_test, y_up_train, y_up_test, label_name="UP_2pct"
    )

    # -----------------------------------------------------
    # Train DOWN model
    # -----------------------------------------------------
    model_down, y_proba_down, results_down = train_xgb_binary(
        X_train, X_test, y_down_train, y_down_test, label_name="DOWN_2pct"
    )

    # -----------------------------------------------------
    # Save models + feature names
    # -----------------------------------------------------
    feature_names = X_train.columns.tolist()
    np.save("model_feature_names_2pct_4h.npy", np.array(feature_names, dtype=object))

    if model_up is not None:
        model_up.save_model("xgb_direction_up_2pct_4h.json")
    else:
        print("[WARN] UP_2pct: no model trained (single-class labels), skipping save.")

    if model_down is not None:
        model_down.save_model("xgb_direction_down_2pct_4h.json")
    else:
        print("[WARN] DOWN_2pct: no model trained (single-class labels), skipping save.")

    print("\n[MODEL] Saved:")
    print("  - xgb_direction_up_2pct_4h.json")
    print("  - xgb_direction_down_2pct_4h.json")
    print("  - model_feature_names_2pct_4h.npy")

    # -----------------------------------------------------
    # Build combined test-set signals dataframe
    # -----------------------------------------------------
    df_signals = pd.DataFrame(
        {
            "prob_up_2pct_4h": y_proba_up,
            "prob_down_2pct_4h": y_proba_down,
        },
        index=test_index,
    )
    df_signals.index.names = ["ts", "symbol"]

    # For debug: thresholds for direction
    thr_up = 0.80
    thr_down = 0.80

    direction = np.zeros(len(df_signals), dtype=int)

    # Short if strong down edge and >= up edge
    short_mask = (df_signals["prob_down_2pct_4h"] > thr_down) & (
        df_signals["prob_down_2pct_4h"] >= df_signals["prob_up_2pct_4h"]
    )
    direction[short_mask] = -1

    # Long if strong up edge and > down edge
    long_mask = (df_signals["prob_up_2pct_4h"] > thr_up) & (
        df_signals["prob_up_2pct_4h"] > df_signals["prob_down_2pct_4h"]
    )
    direction[long_mask] = 1

    df_signals["ml_direction_2pct_4h"] = direction
    df_signals["model_version_up"] = "xgb_up_2pct_4h_v1"
    df_signals["model_version_down"] = "xgb_down_2pct_4h_v1"

    # Save CSV for offline analysis
    df_signals.reset_index().to_csv("high_precision_signals_backtest_up_down.csv", index=False)
    print("\n[FILE] Saved high_precision_signals_backtest_up_down.csv")

    # -----------------------------------------------------
    # Write signals into Postgres: indicators.ml_signals_2pct_4h
    # -----------------------------------------------------
    signals_db = df_signals.reset_index()[[
        "ts", "symbol",
        "prob_up_2pct_4h", "prob_down_2pct_4h",
        "ml_direction_2pct_4h",
        "model_version_up", "model_version_down",
    ]]

    signals_db.to_sql(
        "ml_signals_2pct_4h",
        engine,
        schema="indicators",
        if_exists="append",   # for backtest; in prod you'll upsert
        index=False,
    )

    print("[DB] Wrote test-set ML signals into indicators.ml_signals_2pct_4h")


if __name__ == "__main__":
    main()
