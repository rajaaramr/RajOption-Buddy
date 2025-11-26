"""
Backfill Session VWAP for the entire history.
"""
import os
import sys
import argparse
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import psycopg2.extras as pgx

# Add parent dir to sys.path to allow importing utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.db import get_db_connection

TZ = timezone.utc

_PIVOT_TF_TO_OFFSET = {
    "15m":  "15min",
    "30m":  "30min",
    "60m":  "60min",
    "90m":  "90min",
    "120m": "120min",
    "240m": "240min",
}

def _table_name(kind: str) -> str:
    return "market.spot_candles" if kind == "spot" else "market.futures_candles"

def _frames_table(kind: str) -> str:
    return "indicators.spot_frames" if kind == "spot" else "indicators.futures_frames"

def _load_full_history(symbol: str, kind: str) -> pd.DataFrame:
    tbl = _table_name(kind)
    print(f"[LOAD] Fetching FULL history for {kind}:{symbol}...")
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ts, high::float8, low::float8, close::float8, COALESCE(volume,0)::float8
              FROM {tbl}
             WHERE symbol=%s AND interval='15m'
             ORDER BY ts ASC
            """,
            (symbol,)
        )
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return df

def _compute_session_vwap_series(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized Session VWAP computation.
    Resets VWAP at the start of each IST day.
    """
    if df.empty:
        return pd.Series(dtype="float64")

    # Localize to IST for day grouping
    df_ist = df.copy()
    if df_ist.index.tz is None:
        df_ist.index = df_ist.index.tz_localize("UTC")
    df_ist.index = df_ist.index.tz_convert("Asia/Kolkata")

    grouper = df_ist.index.date
    tp = (df_ist["high"] + df_ist["low"] + df_ist["close"]) / 3.0
    vol = df_ist["volume"].fillna(0.0)
    pv = tp * vol

    # Group by date and cumsum
    cum_pv = pv.groupby(grouper).cumsum()
    cum_v = vol.groupby(grouper).cumsum()

    vwap = cum_pv / cum_v

    # Replace infs and return with original index
    vwap = vwap.replace([float('inf'), float('-inf')], float('nan'))
    vwap.index = df.index
    return vwap

def backfill_vwap_for_symbol(symbol: str, kind: str) -> int:
    df = _load_full_history(symbol, kind)
    if df.empty:
        print(f"[WARN] No data for {kind}:{symbol}")
        return 0

    vwap_series = _compute_session_vwap_series(df)
    if vwap_series.empty:
        return 0

    table = _frames_table(kind)
    run_id = f"bk_vwap_{datetime.now(TZ).strftime('%Y%m%d')}"
    source = "backfill_vwap"

    total_upserted = 0

    # Loop through all target timeframes to write resampled VWAP
    for tf, offset in _PIVOT_TF_TO_OFFSET.items():
        if tf == "15m":
            # Use the 15m series directly
            target_series = vwap_series
        else:
            # Resample to target TF, taking the last value in the bin (right-labeled)
            # Since vwap_session is cumulative intraday, the end-of-bar value is correct.
            target_series = vwap_series.resample(offset, label='right', closed='right').last().dropna()

        if target_series.empty:
            continue

        payload = []
        for ts, val in target_series.items():
            if pd.isna(val):
                continue
            payload.append(
                (
                    symbol,
                    tf,
                    ts.to_pydatetime(),
                    float(val),
                    run_id,
                    source,
                )
            )

        if not payload:
            continue

        print(f"[WRITE] Upserting {len(payload)} rows for {kind}:{symbol} [{tf}] into {table}...")

        with get_db_connection() as conn, conn.cursor() as cur:
            pgx.execute_values(
                cur,
                f"""
                INSERT INTO {table} (symbol, interval, ts, vwap_session, run_id, source)
                VALUES %s
                ON CONFLICT (symbol, interval, ts)
                DO UPDATE SET
                  vwap_session = EXCLUDED.vwap_session,
                  run_id       = EXCLUDED.run_id,
                  source       = EXCLUDED.source,
                  updated_at   = NOW()
                """,
                payload,
                page_size=2000,
            )
            conn.commit()

        total_upserted += len(payload)

    print(f"[DONE] {kind}:{symbol} -> {total_upserted} total rows across TFs")
    return total_upserted

def fetch_universe_symbols() -> List[str]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM reference.symbol_universe ORDER BY symbol")
        return [r[0] for r in cur.fetchall()]

def main():
    parser = argparse.ArgumentParser(description="Backfill Session VWAP for full history")
    parser.add_argument("--symbol", type=str, help="Specific symbol to backfill")
    parser.add_argument("--kinds", type=str, default="spot,futures", help="Comma-separated kinds (spot,futures)")
    args = parser.parse_args()

    kinds = [k.strip().lower() for k in args.kinds.split(",") if k.strip()]

    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = fetch_universe_symbols()
        print(f"Found {len(symbols)} symbols in universe.")

    total_rows = 0
    for sym in symbols:
        for kind in kinds:
            try:
                total_rows += backfill_vwap_for_symbol(sym, kind)
            except Exception as e:
                print(f"[ERROR] Failed {kind}:{sym} -> {e}")

    print(f"\nTotal rows upserted: {total_rows}")

if __name__ == "__main__":
    main()
