#!/usr/bin/env python3
# backfill_charts_csv.py — Bulk ingest Intraday Chart CSVs into Postgres
# Optimized for 30m, 60m, 120m TradingView Exports

import os, re, sys, glob, argparse
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

from configparser import ConfigParser, ParsingError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# ---------------------- CONFIG / DB ----------------------

def load_ini():
    path = os.getenv("ALPHA_CFG", os.path.join(os.path.dirname(__file__), "config.ini"))
    cp = ConfigParser(inline_comment_prefixes=("#",";"))
    if not os.path.exists(path):
        # Fallback to looking one dir up if running from scheduler folder
        up_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.ini")
        if os.path.exists(up_path):
            path = up_path
            
    if not os.path.exists(path):
        print(f"[cfg] INI not found at {path}")
        return cp

    for enc in ("utf-8","utf-8-sig","cp1252"):
        try:
            cp.read(path, encoding=enc)
            return cp
        except (UnicodeDecodeError, ParsingError):
            continue
    return cp

CFG = load_ini()

def make_engine():
    if CFG.has_section("postgres"):
        pg = {k:v for k,v in CFG.items("postgres")}
        url = URL.create(
            "postgresql+psycopg2",
            username=pg.get("user","postgres"),
            password=pg.get("password",""),
            host=pg.get("host","127.0.0.1"),
            port=int(pg.get("port","5432")),
            database=pg.get("dbname","TradeHub18"),
        )
        sp = pg.get("search_path","alpha")
    else:
        url = URL.create(
            "postgresql+psycopg2",
            username=os.getenv("PGUSER","postgres"),
            password=os.getenv("PGPASSWORD",""),
            host=os.getenv("PGHOST","127.0.0.1"),
            port=int(os.getenv("PGPORT","5432")),
            database=os.getenv("PGDATABASE","TradeHub18"),
        )
        sp = os.getenv("PGSEARCH_PATH","raw_ingest")

    eng = create_engine(url, future=True, pool_pre_ping=True, connect_args={"options": f"-c search_path={sp}"})
    with eng.begin() as cx:
        cx.execute(text(f"CREATE SCHEMA IF NOT EXISTS {sp}"))
        cx.execute(text(f"SET search_path TO {sp}"))
    print(f"[db] connected to {url.host}:{url.port}/{url.database} (search_path={sp})")
    return eng, sp

ENG, SCHEMA = make_engine()

# ---------------------- SCHEMA ----------------------

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.intraday_chart_dump (
  trade_date date NOT NULL,
  symbol text NOT NULL,
  interval text NOT NULL,
  
  price numeric,
  description text,
  tech_rating text,
  
  -- Volatility
  donchian_upper numeric,
  atr numeric,
  bb_basis numeric,
  bb_hourly_basis numeric,
  
  -- Volume/Structure
  rvol numeric,
  gap_pct numeric,
  adr_pct numeric,
  vwap numeric,
  vwma numeric,
  
  -- Averages
  ema_10 numeric,
  ema_hourly_10 numeric,
  
  -- Oscillators
  rsi_5 numeric,
  rsi_hourly_5 numeric,
  rsi_14 numeric,
  rsi_hourly_14 numeric,
  mfi_14 numeric,
  mfi_hourly_14 numeric,
  adx_14 numeric,
  
  -- MACD
  macd_signal numeric,
  macd_level numeric,
  
  -- Candlestick
  candle_pattern text,
  
  -- Daily Context
  day_change_pct numeric,
  ma_rating_day text,
  osc_rating_day text,
  
  day_high numeric,
  day_low numeric,
  day_open numeric,
  day_vol numeric,
  avg_vol_10d numeric,
  
  updated_at timestamptz DEFAULT now(),
  
  PRIMARY KEY (trade_date, symbol, interval)
);
"""

with ENG.begin() as cx:
    cx.execute(text(CREATE_TABLE_SQL))

# ---------------------- MAPPING LOGIC ----------------------

# Strict mapping of keywords to DB columns.
# Note: We strip timeframe strings before checking these.
BASE_COL_MAP = {
    "symbol": ["symbol"],
    "description": ["description"],
    "tech_rating": ["technical rating"],
    "price": ["price"], # Be careful not to match 'price change'
    
    "donchian_upper": ["donchian channels (20)"], 
    "rvol": ["relative volume at time"],
    "gap_pct": ["gap %"],
    "adr_pct": ["average daily range %"],
    
    "bb_basis": ["bollinger bands (20)", "basis"], 
    
    "vwap": ["volume weighted average price"],
    "vwma": ["volume weighted moving average (20)"],
    "ema_10": ["exponential moving average (10)"],
    
    "rsi_5": ["relative strength index (5)"],
    "rsi_14": ["relative strength index (14)"],
    "mfi_14": ["money flow index (14)"],
    "adx_14": ["average directional index (14)"],
    "atr": ["average true range (14)"],
    
    "macd_signal": ["convergence divergence", "signal"], # Matches "Moving Average Convergence Divergence ... Signal"
    "macd_level": ["convergence divergence", "level"],   # Matches "Moving Average Convergence Divergence ... Level"
    
    "day_change_pct": ["price change %"],
    "candle_pattern": ["candlestick pattern"],
    
    "ma_rating_day": ["moving averages rating 1 day"],
    "osc_rating_day": ["oscillators rating 1 day"],
    
    "day_high": ["high 1 day"],
    "day_low": ["low 1 day"],
    "day_open": ["open 1 day"],
    "day_vol": ["volume 1 day"],
    "avg_vol_10d": ["average volume 10 days"]
}

def normalize_frame(df: pd.DataFrame, file_interval: str) -> pd.DataFrame:
    """
    Maps CSV headers to DB columns handling Timeframe logic.
    file_interval: '30m', '60m', '120m'
    """
    out = pd.DataFrame()
    if df.empty: return out

    # Symbol is always first
    out["symbol"] = df.iloc[:, 0]

    for raw_col in df.columns:
        clean_raw = str(raw_col).lower().strip()
        
        # Skip currency columns (duplicates of price/high/low)
        if "- currency" in clean_raw:
            continue

        # 1. Determine if this column is an "Hourly Anchor"
        # Logic: 
        # - If file is 60m, then "1 hour" cols are just the current TF.
        # - If file is 30m/120m, then "1 hour" cols are Hourly Anchors.
        is_hourly_anchor = False
        if "1 hour" in clean_raw and file_interval != "60m":
            is_hourly_anchor = True

        # 2. Create a "Base Name" by stripping all timeframes
        # Removes: " 30 minutes", " 1 hour", " 2 hours", " 1 day"
        base_name = re.sub(r"\s+\d+\s+(minutes|minute|hour|hours|day|days)", "", clean_raw)
        base_name = base_name.replace("basis", "").strip() # BB Basis cleanup

        mapped_db_col = None
        
        # 3. Match against Base Map
        for db_col, keywords in BASE_COL_MAP.items():
            # ALL keywords must be present in the cleaned name to match
            # e.g. MACD Signal needs "convergence" AND "signal"
            if all(k in clean_raw for k in keywords):
                mapped_db_col = db_col
                break
        
        # Special Fixes for ambiguous matches
        if "price change" in clean_raw: mapped_db_col = "day_change_pct"
        if "average daily range" in clean_raw: mapped_db_col = "adr_pct"

        # 4. Apply Mapping
        if mapped_db_col:
            # If it's an hourly anchor, verify if we have a slot for it
            # Schema supports: bb_hourly_basis, ema_hourly_10, rsi_hourly_5, rsi_hourly_14, mfi_hourly_14
            if is_hourly_anchor:
                if mapped_db_col == "bb_basis": mapped_db_col = "bb_hourly_basis"
                elif mapped_db_col == "ema_10": mapped_db_col = "ema_hourly_10"
                elif mapped_db_col == "rsi_5":  mapped_db_col = "rsi_hourly_5"
                elif mapped_db_col == "rsi_14": mapped_db_col = "rsi_hourly_14"
                elif mapped_db_col == "mfi_14": mapped_db_col = "mfi_hourly_14"
                else:
                    # Matches an hourly col we don't store specifically (like ADX hourly), skip or store in base?
                    # For now, if we don't have a specific hourly column, we skip to avoid pollution.
                    continue 

            out[mapped_db_col] = df[raw_col]

    # 5. Data Cleanup
    for c in out.columns:
        if c in ["symbol", "description", "tech_rating", "ma_rating_day", "osc_rating_day", "candle_pattern"]:
            out[c] = out[c].astype(str).str.strip().replace('nan', None)
            if c == "symbol": out[c] = out[c].str.upper()
        else:
            # Numeric Cleanup
            out[c] = (out[c].astype(str)
                      .str.replace("INR","", regex=False)
                      .str.replace("%","", regex=False)
                      .str.replace(",","", regex=False)
                      .apply(pd.to_numeric, errors='coerce'))
            
    return out

def parse_filename_meta(filename: str) -> tuple[str, dt.date]:
    name = Path(filename).name
    
    # Regex for Interval (30, 60, 120)
    tf_match = re.search(r"^(\d+)-Minute", name, re.IGNORECASE)
    interval = f"{tf_match.group(1)}m" if tf_match else "0m"
    
    # Regex for Date
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if date_match:
        trade_date = dt.datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
    else:
        trade_date = dt.date.today()
        
    return interval, trade_date

def upsert_data(df: pd.DataFrame, trade_date: dt.date, interval: str):
    if df.empty: return 0
    
    df = df.copy()
    df["trade_date"] = trade_date
    df["interval"] = interval
    
    # Get valid columns from DB
    with ENG.begin() as cx:
        db_cols = [r[0] for r in cx.execute(text(f"""
            SELECT column_name FROM information_schema.columns 
            WHERE table_schema='{SCHEMA}' AND table_name='intraday_chart_dump'
        """)).fetchall()]
    
    final_cols = [c for c in df.columns if c in db_cols]
    df = df[final_cols]
    
    tmp_table = "tmp_chart_ingest"
    col_str = ", ".join([f'"{c}"' for c in final_cols])
    
    with ENG.begin() as cx:
        cx.execute(text(f"DROP TABLE IF EXISTS {tmp_table}"))
        cx.execute(text(f"CREATE TEMP TABLE {tmp_table} AS SELECT {col_str} FROM {SCHEMA}.intraday_chart_dump WHERE 1=0"))
    
    print(f"   > Inserting {len(df)} rows into temp table...")
    df.to_sql(tmp_table, ENG, if_exists="append", index=False, method="multi", chunksize=2000)
    
    pk = ["trade_date", "symbol", "interval"]
    update_cols = [c for c in final_cols if c not in pk]
    
    update_clause = "DO NOTHING"
    if update_cols:
        updates = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in update_cols])
        update_clause = f"DO UPDATE SET {updates}, updated_at = now()"
        
    sql_merge = f"""
    INSERT INTO {SCHEMA}.intraday_chart_dump ({col_str})
    SELECT {col_str} FROM {tmp_table}
    ON CONFLICT (trade_date, symbol, interval)
    {update_clause};
    """
    
    with ENG.begin() as cx:
        cx.execute(text(sql_merge))
        cx.execute(text(f"DROP TABLE IF EXISTS {tmp_table}"))
        
    return len(df)

# ---------------------- MAIN ----------------------

def main():
    ap = argparse.ArgumentParser(description="Backfill Intraday Charts CSVs")
    ap.add_argument("--inbox", default=r"D:\Trading\RajOptionBuddy\Inbox", help="Folder with csv files")
    ap.add_argument("--pattern", default="*Minute Chart*.csv", help="Glob pattern") 
    args = ap.parse_args()
    
    inbox = Path(args.inbox)
    files = sorted(list(inbox.glob(args.pattern)))
    
    if not files:
        print(f"No files found in {inbox} matching {args.pattern}")
        return

    print(f"Found {len(files)} files to process.")
    
    total = 0
    for f in files:
        try:
            print(f"Processing: {f.name}")
            interval, trade_date = parse_filename_meta(f.name)
            print(f"   > Detected: Date={trade_date}, Interval={interval}")
            
            df_raw = pd.read_csv(f)
            
            # Pass file interval to normalization logic to handle '1 hour' ambiguity
            df_clean = normalize_frame(df_raw, interval)
            
            rows = upsert_data(df_clean, trade_date, interval)
            print(f"   > Success: Upserted {rows} rows.")
            total += rows
            
        except Exception as e:
            print(f"   > ERROR processing {f.name}: {e}")
            # import traceback; traceback.print_exc()

    print(f"\nAll done. Total rows processed: {total}")

if __name__ == "__main__":
    main()