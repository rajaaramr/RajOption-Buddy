# scheduler/ml_strategy_automated/automated_feature_engineering.py

"""
Handles the automated feature engineering using Featuretools.
"""
import pandas as pd
import featuretools as ft

def create_entityset(data_dict):
    """
    Creates a Featuretools EntitySet from the loaded dataframes.
    The target for feature generation will be daily data, not intraday.
    """
    es = ft.EntitySet(id="trading_data")

    # Base dataframe: daily unified data, represents each symbol's state for a given day
    es.add_dataframe(
        dataframe_name="daily_unified",
        dataframe=data_dict["fo_daily_unified"].rename(columns={'nse_code': 'symbol'}),
        index="symbol",
        time_index="trade_date",
    )

    # Add other daily dataframes
    es.add_dataframe(dataframe_name="daily_futures", dataframe=data_dict["daily_futures"], time_index="trade_date", make_index=True, index="daily_futures_id")
    es.add_dataframe(dataframe_name="daily_options", dataframe=data_dict["daily_options"], time_index="trade_date", make_index=True, index="daily_options_id")
    es.add_dataframe(dataframe_name="intraday_chart_dump", dataframe=data_dict["intraday_chart_dump"], time_index="trade_date", make_index=True, index="chart_dump_id")

    # Add intraday dataframes
    df_spot = data_dict["spot_candles"].copy()
    df_spot["trade_date"] = df_spot["ts"].dt.normalize()
    es.add_dataframe(dataframe_name="spot_candles", dataframe=df_spot, make_index=True, index="spot_candle_id", time_index="ts")

    df_fut = data_dict["futures_candles"].copy()
    df_fut["trade_date"] = df_fut["ts"].dt.normalize()
    es.add_dataframe(dataframe_name="futures_candles", dataframe=df_fut, make_index=True, index="futures_candle_id", time_index="ts")

    # --- Define Relationships ---
    es.add_relationship("daily_unified", "symbol", "daily_futures", "symbol")
    es.add_relationship("daily_unified", "symbol", "daily_options", "symbol")
    es.add_relationship("daily_unified", "symbol", "intraday_chart_dump", "symbol")
    es.add_relationship("daily_unified", ["symbol", "trade_date"], "spot_candles", ["symbol", "trade_date"])
    es.add_relationship("daily_unified", ["symbol", "trade_date"], "futures_candles", ["symbol", "trade_date"])

    return es

def generate_automated_features(es):
    """
    Runs Deep Feature Synthesis on the EntitySet to create daily contextual features.
    """
    print("[INFO] Starting automated feature generation with DFS...")

    # We generate features for each symbol for each day
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="daily_unified",
        agg_primitives=["sum", "std", "max", "min", "mean", "count", "skew", "num_unique", "mode"],
        trans_primitives=["day", "year", "month", "weekday", "is_month_start", "is_month_end"],
        max_depth=2,
        verbose=1,
    )

    print("[INFO] Automated feature generation complete.")
    return feature_matrix, feature_defs
