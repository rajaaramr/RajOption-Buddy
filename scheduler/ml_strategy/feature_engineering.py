# scheduler/ml_strategy/feature_engineering.py

"""
Functions for feature engineering.
"""

import math
import pandas as pd
import numpy as np

from . import config

def _calculate_atr(df, period=14):
    """Helper to calculate ATR."""
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    tr = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    return tr.rolling(period).mean()

def add_prob_feature(df, value_col, target_col, bins, new_col_name):
    """
    Converts a continuous indicator into a probability-style feature.
    """
    tmp = df.reset_index().sort_values("ts")
    calib_end = int(len(tmp) * config.CALIBRATION_FRAC)
    calib = tmp.iloc[:calib_end].copy()
    calib = calib.dropna(subset=[value_col, target_col])

    if calib.empty:
        out = pd.Series(0.5, index=df.index, name=new_col_name)
        return out

    calib["bin"] = pd.cut(calib[value_col], bins=bins, right=False, labels=False, include_lowest=True)
    prob_by_bin = calib.groupby("bin")[target_col].mean()

    tmp["bin"] = pd.cut(tmp[value_col], bins=bins, right=False, labels=False, include_lowest=True)
    prob_series = tmp["bin"].map(prob_by_bin)

    global_prob = float(calib[target_col].mean())
    if math.isnan(global_prob):
        global_prob = 0.5

    prob_series = prob_series.fillna(global_prob).astype("float64")
    tmp[new_col_name] = prob_series

    return tmp.set_index(["ts", "symbol"])[new_col_name]

def build_features(data_dict):
    """
    Main function to build all features.
    """
    print("[INFO] Starting feature engineering...")

    # --- Initial merge ---
    spot_15 = data_dict["spot_candles"].rename(columns={"close": "close_SPOT", "volume": "volume_SPOT"})
    fut_15 = data_dict["futures_candles"].rename(columns={"close": "close_FUT", "oi": "oi_FUT"})
    # Drop the interval column from futures to avoid the merge conflict
    df = pd.merge(spot_15, fut_15.drop(columns=['interval']), on=["ts", "symbol"], how="inner")

    # --- Merge indicators ---
    def merge_indicator(df, indicator_df, interval, cols_to_rename):
        ind = indicator_df[indicator_df["interval"] == interval].copy()
        ind = ind.rename(columns=cols_to_rename)

        # Columns to merge: the keys of cols_to_rename + the join keys
        merge_cols = ["ts", "symbol"] + list(cols_to_rename.values())

        # Forward-fill for larger timeframes
        if interval != '15m':
            ind = ind.set_index(["symbol", "ts"]).sort_index()
            ind = ind.groupby(level="symbol").ffill().reset_index()

        return pd.merge(df, ind[merge_cols], on=["ts", "symbol"], how="left")

    df = merge_indicator(df, data_dict["futures_frames"], '15m',
                         {"stoch_k": "stoch_k_FUT_15m", "macd_hist": "macd_hist_FUT_15m", "bb_score": "bb_score_FUT_15m"})
    df = merge_indicator(df, data_dict["spot_frames"], '15m', {"rsi": "rsi_SPOT_15m"})
    df = merge_indicator(df, data_dict["spot_frames"], '30m', {"adx": "adx_SPOT_30m"})
    df = merge_indicator(df, data_dict["spot_frames"], '60m', {"mfi_14": "mfi_14_SPOT_60m"})

    # --- Add daily option and TradingView features ---
    df["trade_date"] = pd.to_datetime(df["ts"]).dt.normalize()

    df_opt_sym_day = _build_daily_option_features(data_dict["daily_options"])
    if not df_opt_sym_day.empty:
        df = pd.merge(df, df_opt_sym_day, on=["trade_date", "symbol"], how="left")

    df_tv_features = _build_tradingview_features(data_dict["intraday_chart_dump"])
    if not df_tv_features.empty:
        df = pd.merge(df, df_tv_features, on=["trade_date", "symbol"], how="left")

    df = df.drop(columns=["trade_date"])
    df = df.sort_values(["ts", "symbol"]).set_index(["ts", "symbol"])

    # --- Lag Features & Targets ---
    lag_cols = [c for c in df.columns if "_SPOT_" in c or "_FUT_" in c]
    for col in lag_cols:
        for lag in [1, 2, 4]:
            df[f"{col}_LAG_{lag}"] = df.groupby(level="symbol")[col].shift(lag)

    df["future_close_SPOT"] = df.groupby(level="symbol")["close_SPOT"].shift(-config.TARGET_PERIODS_AHEAD)
    df["future_ret"] = (df["future_close_SPOT"] / df["close_SPOT"]) - 1.0

    # --- New ATR Feature ---
    df_spot_for_atr = spot_15.set_index(['ts', 'symbol']).sort_index()
    atr_series = df_spot_for_atr.groupby(level='symbol').apply(
        lambda x: _calculate_atr(x.reset_index(), period=14)
    )
    atr_series.name = 'atr_14_SPOT_15m'
    df = df.join(atr_series, on=['ts', 'symbol'])

    # --- Targets (multi-class) ---
    df["target"] = 0 # Sideways
    df.loc[df["future_ret"] > config.TARGET_THRESHOLD, "target"] = 1 # Up
    df.loc[df["future_ret"] < -config.TARGET_THRESHOLD, "target"] = -1 # Down

    # --- Probability-style features (calibrated against UP and SIDEWAYS) ---
    df["temp_target_up"] = (df["target"] == 1).astype(int)
    df["temp_target_sideways"] = (df["target"] == 0).astype(int)

    for col, bins in config.FEATURE_BINS.items():
        prefix = f"n{col.split('_')[0].upper()}"

        # Calibrate against UP
        prob_up_col = f"{prefix}_prob_up"
        df[prob_up_col] = add_prob_feature(df, col, "temp_target_up", bins, prob_up_col)

        # Calibrate against SIDEWAYS
        prob_sideways_col = f"{prefix}_prob_sideways"
        df[prob_sideways_col] = add_prob_feature(df, col, "temp_target_sideways", bins, prob_sideways_col)

        # Combined feature to reduce bias
        # This feature represents the tendency for the indicator to predict an UP move relative to a SIDEWAYS move.
        df[f"{prefix}_prob_up_vs_sideways"] = df[prob_up_col] / (df[prob_up_col] + df[prob_sideways_col] + 1e-9)

    df = df.drop(columns=["temp_target_up", "temp_target_sideways"])

    # --- Final Touches ---
    df = df.reset_index()
    df_meta = data_dict["symbol_meta"].drop_duplicates("symbol")
    df = pd.merge(df, df_meta, on="symbol", how="left").set_index(["ts", "symbol"])
    df = pd.get_dummies(df, columns=["industry_name", "sector_name"], drop_first=True)

    print("[INFO] Feature engineering complete.")
    return df

def _build_tradingview_features(df_tv):
    """Pivots the TradingView data to create daily features."""
    if df_tv.empty:
        return pd.DataFrame()

    # Define columns to pivot
    indicator_cols = [
        'tech_rating', 'rvol', 'gap_pct', 'vwap', 'vwma', 'ema_10', 'rsi_14', 'mfi_14', 'adx_14'
    ]

    # Pivot the table
    df_pivot = df_tv.pivot_table(
        index=['trade_date', 'symbol'],
        columns='interval',
        values=indicator_cols
    ).reset_index()

    # Flatten the multi-level column index
    df_pivot.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_pivot.columns]

    # Handle categorical 'tech_rating'
    for col in [c for c in df_pivot.columns if 'tech_rating' in c]:
        # Simple mapping: Strong Buy=2, Buy=1, Neutral=0, Sell=-1, Strong Sell=-2
        rating_map = {"Strong buy": 2, "Buy": 1, "Neutral": 0, "Sell": -1, "Strong sell": -2}
        df_pivot[col] = df_pivot[col].map(rating_map).fillna(0)

    return df_pivot

def _build_daily_option_features(df_daily_opt):
    """Helper to build daily option features."""
    if df_daily_opt.empty:
        return pd.DataFrame()

    df = df_daily_opt.copy()
    df["trade_date"] = pd.to_datetime(df.get("trade_date") or df.get("ts")).dt.date
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    ce = df[df["option_type"] == "CE"]
    pe = df[df["option_type"] == "PE"]
    group_cols = ["trade_date", "symbol"]

    spot_ref = df.groupby(group_cols)["spot"].first().rename("spot_ref")
    ce_agg = ce.groupby(group_cols).agg(total_call_oi=("oi", "sum"), avg_call_iv=("iv", "mean"))
    pe_agg = pe.groupby(group_cols).agg(total_put_oi=("oi", "sum"), avg_put_iv=("iv", "mean"))

    ce_wall = ce.loc[ce.groupby(group_cols)["oi"].idxmax()][group_cols + ["strike_price"]].rename(columns={"strike_price": "call_wall_strike"})
    pe_wall = pe.loc[pe.groupby(group_cols)["oi"].idxmax()][group_cols + ["strike_price"]].rename(columns={"strike_price": "put_wall_strike"})

    df_sym_day = pd.concat([spot_ref, ce_agg, pe_agg], axis=1).reset_index()
    df_sym_day = pd.merge(df_sym_day, ce_wall, on=group_cols, how="left")
    df_sym_day = pd.merge(df_sym_day, pe_wall, on=group_cols, how="left")

    df_sym_day["pcr_oi"] = df_sym_day["total_put_oi"] / df_sym_day["total_call_oi"]
    df_sym_day["dist_call_wall"] = (df_sym_day["call_wall_strike"] - df_sym_day["spot_ref"]) / df_sym_day["spot_ref"]
    df_sym_day["dist_put_wall"] = (df_sym_day["spot_ref"] - df_sym_day["put_wall_strike"]) / df_sym_day["spot_ref"]
    df_sym_day["avg_iv"] = df_sym_day[["avg_call_iv", "avg_put_iv"]].mean(axis=1)

    return df_sym_day
