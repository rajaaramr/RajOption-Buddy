# scheduler/ml_strategy/data_loader.py

"""
Handles data loading from PostgreSQL.
"""

import pandas as pd
from sqlalchemy import create_engine

from . import config

def get_engine():
    """Builds and returns a SQLAlchemy engine for PostgreSQL."""
    if config.DB_PASSWORD is None:
        raise ValueError("Database password is not set. Please set the PGPASSWORD environment variable.")
    return create_engine(config.CONN_STR)

def load_all_tables(engine):
    """
    Loads all required tables from Postgres.

    Returns a dictionary of DataFrames.
    """
    print("[INFO] Loading all source tables from Postgres...")

    tables = {}

    # Intraday spot candles (15m)
    tables["spot_candles"] = pd.read_sql(
        f"""
        SELECT ts, symbol, interval, close, volume, high, low
        FROM {config.TABLES['spot_candles']}
        WHERE interval = '15m'
        """,
        engine,
        parse_dates=["ts"],
    )

    # Intraday futures candles (15m)
    tables["futures_candles"] = pd.read_sql(
        f"""
        SELECT ts, symbol, interval, close, oi
        FROM {config.TABLES['futures_candles']}
        WHERE interval = '15m'
        """,
        engine,
        parse_dates=["ts"],
    )

    # Indicator frames (spot & futures, all TFs)
    tables["spot_frames"] = pd.read_sql(
        f"SELECT * FROM {config.TABLES['spot_frames']}",
        engine,
        parse_dates=["ts"],
    )

    tables["futures_frames"] = pd.read_sql(
        f"SELECT * FROM {config.TABLES['futures_frames']}",
        engine,
        parse_dates=["ts"],
    )

    # Daily futures (optional)
    tables["daily_futures"] = pd.read_sql(
        f"SELECT * FROM {config.TABLES['daily_futures']}",
        engine,
        parse_dates=["trade_date"],
    )

    # Daily option chain snapshot
    tables["daily_options"] = pd.read_sql(
        f"SELECT * FROM {config.TABLES['daily_options']}",
        engine,
    )

    # Symbol metadata
    tables["symbol_meta"] = pd.read_sql(
        f"SELECT symbol, industry_name, sector_name FROM {config.TABLES['symbol_meta']}",
        engine,
    )

    # TradingView intraday chart dump
    tables["intraday_chart_dump"] = pd.read_sql(
        f"SELECT * FROM {config.TABLES['intraday_chart_dump']}",
        engine,
        parse_dates=["trade_date"],
    )

    print("[INFO] All tables loaded successfully.")
    return tables
