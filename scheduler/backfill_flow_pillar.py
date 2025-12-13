from __future__ import annotations

import argparse
from datetime import datetime
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from utils.db import get_db_connection
from pillars.common import TZ, BaseCfg, write_values, maybe_trim_last_bar

from flow.pillar.flow_pillar_optimized import process_symbol_vectorized
from flow.pillar.flow_features_optimized import load_daily_futures, load_external_metrics

UNIVERSE_TABLE = "reference.symbol_universe"
DEFAULT_INI = Path(__file__).resolve().parent / "flow_scenarios.ini"

# ============================================================
# Universe helpers
# ============================================================

def get_symbols(extra_where: str = "enabled = TRUE", limit: int = 0) -> List[str]:
    sql = f"""
        SELECT symbol
        FROM {UNIVERSE_TABLE}
        WHERE {extra_where}
        ORDER BY 1
    """
    if limit and limit > 0:
        sql += f" LIMIT {int(limit)}"

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return [r[0] for r in cur.fetchall()]

def is_symbol_enabled(symbol: str) -> bool:
    sql = f"SELECT enabled FROM {UNIVERSE_TABLE} WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol,))
        r = cur.fetchone()
        return bool(r and r[0])

def set_flow_error(symbol: str, kind: str, msg: str) -> None:
    col = "fut_last_error" if kind == "futures" else "spot_last_error"
    sql = f"UPDATE {UNIVERSE_TABLE} SET {col}=%s WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (msg[:500], symbol))
        conn.commit()

# ============================================================
# Candle loader
# ============================================================

def load_15m(symbol: str, kind: str) -> pd.DataFrame:
    if kind == "futures":
        candidates = [
            ("market.futures_candles", True),
            ("public.futures_ohlcv_15m", False),
            ("market.futures_ohlcv_15m", False),
        ]
    else:
        candidates = [
            ("market.spot_candles", True),
            ("public.spot_ohlcv_15m", False),
            ("market.spot_ohlcv_15m", False),
        ]

    with get_db_connection() as conn:
        for table, has_interval in candidates:
            try:
                interval_clause = "AND interval='15m'" if has_interval else ""
                sql = f"""
                    SELECT ts, open, high, low, close, volume
                           {", oi" if kind=="futures" else ""}
                    FROM {table}
                    WHERE symbol=%s
                    {interval_clause}
                    ORDER BY ts
                """
                df = pd.read_sql(sql, conn, params=(symbol,))
                if df is None or df.empty:
                    continue

                df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                df = df.dropna(subset=["ts"]).set_index("ts").sort_index()

                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                if "oi" in df.columns:
                    df["oi"] = pd.to_numeric(df["oi"], errors="coerce")

                return df
            except Exception:
                continue

    # Return empty if nothing found instead of crash, let caller handle
    return pd.DataFrame()

# ============================================================
# Backfill core
# ============================================================

def backfill_symbol_vectorized(
    symbol: str,
    kind: str,
    tfs: List[str],
    ini_path: str,
    run_id: str,
    flush_every: int = 5000,
) -> None:
    symbol = symbol.strip().upper()

    print(f"[{symbol}] Starting vectorized backfill...")

    # 1. Load 15m Candles
    df15 = load_15m(symbol, kind)
    if df15.empty:
        print(f"[SKIP] {symbol} {kind}: no 15m data")
        return

    # 2. Load Daily Futures
    daily_df = load_daily_futures(symbol)
    if daily_df.empty:
        print(f"[WARN] {symbol}: No data in raw_ingest.daily_futures. 'daily_*' scenarios will fail.")
        # We continue, but warn.

    # 3. Load Metrics (Bulk)
    start_ts = df15.index[0]
    end_ts = df15.index[-1]
    metrics_df = load_external_metrics(symbol, start_ts, end_ts)

    base_cfg = BaseCfg(tfs=tfs, lookback_days=0, run_id=run_id, source="FLOW_BACKFILL_VEC")

    total_rows = 0

    # 4. Process each TF
    for tf in tfs:
        print(f"[{symbol}] Processing {tf}...")

        # Call the vectorized processor
        rows = process_symbol_vectorized(
            symbol=symbol,
            kind=kind,
            tf=tf,
            df15=df15,
            daily_df=daily_df,
            metrics_df=metrics_df,
            base_cfg=base_cfg,
            ini_path=Path(ini_path)
        )

        if not rows:
            print(f"[{symbol}] {tf}: No scores generated.")
            continue

        # 5. Write to DB
        # Use simple chunking
        batch_size = flush_every
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i+batch_size]
            write_values(chunk)

        total_rows += len(rows)
        print(f"[{symbol}] {tf}: Wrote {len(rows)} rows.")

    print(f"[{symbol}] Done. Total rows written: {total_rows}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="", help="Single symbol")
    ap.add_argument("--all-symbols", action="store_true")
    ap.add_argument("--kind", default="futures", choices=["spot", "futures"])
    ap.add_argument("--tfs", default="15m,30m,60m,120m,240m")
    ap.add_argument("--ini", default=str(DEFAULT_INI))
    ap.add_argument("--run-id", default=f"FLOW_BACKFILL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--flush-every", type=int, default=5000)

    args = ap.parse_args()

    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]

    if args.all_symbols:
        symbols = get_symbols()
    else:
        symbols = [args.symbol.strip().upper()] if args.symbol else get_symbols()

    if not symbols:
        print("No symbols selected.")
        return

    print(f"Starting Vectorized Backfill for {len(symbols)} symbols. Kind={args.kind}")

    # Pre-flight check prompt (Automated)
    # In a real user session, we might ask input. Here we assume yes based on plan.
    # But let's check one symbol for daily futures existence as a heuristic.
    check_sym = symbols[0]
    check_df = load_daily_futures(check_sym)
    if check_df.empty:
        print("!"*60)
        print(f"WARNING: raw_ingest.daily_futures appears empty for {check_sym}.")
        print("Ensure you have run the daily futures ingestion/backfill first!")
        print("!"*60)

    for s in symbols:
        try:
            backfill_symbol_vectorized(
                symbol=s,
                kind=args.kind,
                tfs=tfs,
                ini_path=args.ini,
                run_id=args.run_id,
                flush_every=args.flush_every
            )
        except Exception as e:
            print(f"[ERROR] {s}: {e}")
            import traceback
            traceback.print_exc()
            set_flow_error(s, args.kind, str(e))

if __name__ == "__main__":
    main()
