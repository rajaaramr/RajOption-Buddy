# scheduler/ml_strategy_automated/data_loader.py

"""
Handles data loading from PostgreSQL for the automated pipeline.
"""

import pandas as pd
from sqlalchemy import create_engine
# Use the local config for the automated pipeline
from . import config

def get_engine():
    """Builds and returns a SQLAlchemy engine for PostgreSQL."""
    if config.DB_PASSWORD is None:
        raise ValueError("Database password is not set. Please set the PGPASSWORD environment variable.")
    return create_engine(config.CONN_STR)

def load_all_tables_for_featuretools(engine):
    """
    Loads all required tables from Postgres for Featuretools.

    Returns a dictionary of DataFrames.
    """
    print("[INFO] Loading all source tables for Featuretools...")

    tables = {}

    # Intraday spot candles (15m)
    tables["spot_candles"] = pd.read_sql(
        f"""
        SELECT ts, symbol, close, volume, high, low
        FROM {config.TABLES['spot_candles']}
        WHERE interval = '15m'
        AND ts::time BETWEEN '09:15:00' AND '15:30:00'
        """,
        engine, parse_dates=["ts"],
    )

    # Intraday futures candles (15m)
    tables["futures_candles"] = pd.read_sql(
        f"""
        SELECT ts, symbol, close as fut_close, oi
        FROM {config.TABLES['futures_candles']}
        WHERE interval = '15m'
        AND ts::time BETWEEN '09:15:00' AND '15:30:00'
        """,
        engine, parse_dates=["ts"],
    )

    # Daily futures
    tables["daily_futures"] = pd.read_sql(
        f"SELECT * FROM {config.TABLES['daily_futures']}",
        engine, parse_dates=["trade_date"],
    )

    # Daily option chain snapshot
    tables["daily_options"] = pd.read_sql(
        f"SELECT * FROM {config.TABLES['daily_options']}",
        engine, parse_dates=["trade_date"],
    )

    # TradingView intraday chart dump
    tables["intraday_chart_dump"] = pd.read_sql(
        f"SELECT * FROM {config.TABLES['intraday_chart_dump']}",
        engine, parse_dates=["trade_date"],
    )

    # F&O daily unified data
    tables["fo_daily_unified"] = pd.read_sql(
        f"SELECT * FROM {config.TABLES['fo_daily_unified']}",
        engine, parse_dates=["trade_date"],
    )

    print("[INFO] All tables for Featuretools loaded successfully.")
    return tables
