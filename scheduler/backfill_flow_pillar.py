from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2.extras

from utils.db import get_db_connection
from pillars.common import TZ, BaseCfg, maybe_trim_last_bar

from flow.pillar.flow_pillar_optimized import process_symbol_vectorized
from flow.pillar.flow_features_optimized import load_daily_futures, load_external_metrics

UNIVERSE_TABLE = "reference.symbol_universe"
DEFAULT_INI = Path(__file__).resolve().parent / "flow_scenarios.ini"

# ============================================================
# Universe helpers (Preserved from Original)
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

def _resume_col(kind: str, mode: str) -> str:
    if mode == "ml":
        return "flow_ml_last_fut_ts" if kind == "futures" else "flow_ml_last_spot_ts"
    return "flow_rules_last_fut_ts" if kind == "futures" else "flow_rules_last_spot_ts"

def _audit_cols(mode: str) -> tuple[str, str]:
    if mode == "ml":
        return ("flow_ml_last_run_at", "flow_ml_last_run_id")
    return ("flow_rules_last_run_at", "flow_rules_last_run_id")

def update_flow_run_audit(symbol: str, run_id: str, mode: str) -> None:
    col_at, col_run = _audit_cols(mode)
    sql = f"""
        UPDATE {UNIVERSE_TABLE}
           SET {col_at} = NOW(),
               {col_run} = %s
         WHERE symbol = %s
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (run_id, symbol))
        conn.commit()

def update_flow_last_ts(symbol: str, kind: str, mode: str, last_ts: datetime) -> None:
    col = _resume_col(kind, mode)
    # Ensure last_ts is timezone aware or properly formatted for Postgres
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=TZ)

    sql = f"UPDATE {UNIVERSE_TABLE} SET {col}=%s WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (last_ts, symbol))
        conn.commit()

def set_flow_error(symbol: str, kind: str, msg: str) -> None:
    col = "fut_last_error" if kind == "futures" else "spot_last_error"
    sql = f"UPDATE {UNIVERSE_TABLE} SET {col}=%s WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (msg[:500], symbol))
        conn.commit()

def get_resume_ts(symbol: str, kind: str, mode: str) -> Optional[datetime]:
    col = _resume_col(kind, mode)
    sql = f"SELECT {col} FROM {UNIVERSE_TABLE} WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol,))
        r = cur.fetchone()
        return r[0] if r and r[0] else None

def parse_dt(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    s = s.replace(" ", "T")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt

def write_values_direct(rows: List[tuple]) -> None:
    """
    Writes rows directly to indicators.values, bypassing pillars.common.
    Expected row format: (symbol, kind, tf, ts, metric, val, ctx, run_id, source)
    """
    if not rows:
        return

    # Dedupe locally
    dedup = {}
    for r in rows:
        # Key: (symbol, kind, tf, ts, metric)
        key = (r[0], r[1], r[2], r[3], r[4])
        dedup[key] = r
    rows_to_write = list(dedup.values())

    sql = """
        INSERT INTO indicators.values
        (symbol, market_type, interval, ts, metric, val, context, run_id, source)
        VALUES %s
        ON CONFLICT (symbol, market_type, interval, ts, metric)
        DO UPDATE SET
            val = EXCLUDED.val,
            context = EXCLUDED.context,
            run_id = EXCLUDED.run_id,
            source = EXCLUDED.source,
            updated_at = NOW()
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, sql, rows_to_write, page_size=1000
                )
            conn.commit()
            # print(f"[DB] Committed {len(rows_to_write)} rows to indicators.values")
    except Exception as e:
        print(f"[DB ERROR] Failed to write rows: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# Candle loader
# ============================================================

def load_15m(symbol: str, kind: str, lookback_days: int = 0, from_ts: Optional[datetime] = None) -> pd.DataFrame:
    # Determine start time
    start_dt = None
    if from_ts:
        start_dt = from_ts
    elif lookback_days > 0:
        start_dt = datetime.now(TZ) - timedelta(days=lookback_days)

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

                # Add time filter
                time_clause = ""
                params = [symbol]
                if start_dt:
                    time_clause = "AND ts >= %s"
                    params.append(start_dt)

                sql = f"""
                    SELECT ts, open, high, low, close, volume
                           {", oi" if kind=="futures" else ""}
                    FROM {table}
                    WHERE symbol=%s
                    {interval_clause}
                    {time_clause}
                    ORDER BY ts
                """
                df = pd.read_sql(sql, conn, params=tuple(params))
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

    return pd.DataFrame()

# ============================================================
# Backfill core
# ============================================================

def backfill_symbol_vectorized(
    symbol: str,
    kind: str,
    mode: str,
    tfs: List[str],
    ini_path: str,
    run_id: str,
    lookback_days: int = 180,
    from_ts: Optional[datetime] = None,
    flush_every: int = 5000,
    resume: bool = False,
) -> None:
    symbol = symbol.strip().upper()

    # Check resume pointer if requested
    if resume and not from_ts:
        resume_ts = get_resume_ts(symbol, kind, mode)
        if resume_ts:
            print(f"[{symbol}] Resuming from {resume_ts}...")
            from_ts = resume_ts

    update_flow_run_audit(symbol, run_id, mode)

    print(f"[{symbol}] Starting vectorized backfill...")

    # 1. Load 15m Candles
    df15 = load_15m(symbol, kind, lookback_days, from_ts)
    if df15.empty:
        print(f"[SKIP] {symbol} {kind}: no 15m data")
        return

    # 2. Load Daily Futures
    daily_df = load_daily_futures(symbol)
    if daily_df.empty:
        print(f"[WARN] {symbol}: No data in raw_ingest.daily_futures. 'daily_*' scenarios will fail.")

    # 3. Load Metrics (Bulk)
    start_ts = df15.index[0]
    end_ts = df15.index[-1]
    metrics_df = load_external_metrics(symbol, start_ts, end_ts)

    base_cfg = BaseCfg(tfs=tfs, lookback_days=lookback_days, run_id=run_id, source="FLOW_BACKFILL_VEC")

    total_rows = 0
    max_ts_processed = None

    # 4. Process each TF
    for tf in tfs:
        print(f"[{symbol}] Processing {tf}...")

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

        # Update max_ts for resume pointer (only from 15m to be consistent with logic)
        if tf == "15m" and rows:
            # rows are tuples: (symbol, kind, tf, ts, ...)
            # Extract max ts from the batch
            batch_max_ts = max(r[3] for r in rows)
            if max_ts_processed is None or batch_max_ts > max_ts_processed:
                max_ts_processed = batch_max_ts

        # 5. Write to DB
        batch_size = flush_every
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i+batch_size]
            # Use direct writer
            write_values_direct(chunk)

        total_rows += len(rows)
        print(f"[{symbol}] {tf}: Wrote {len(rows)} rows.")

    # Update resume pointer
    if max_ts_processed:
        update_flow_last_ts(symbol, kind, mode, max_ts_processed)
        print(f"[{symbol}] Updated resume pointer to {max_ts_processed}")

    print(f"[{symbol}] Done. Total rows written: {total_rows}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="", help="Single symbol")
    ap.add_argument("--all-symbols", action="store_true")
    ap.add_argument("--universe-where", default="enabled = TRUE", help="SQL WHERE for reference.symbol_universe")
    ap.add_argument("--limit", type=int, default=0, help="limit universe symbols")

    ap.add_argument("--kind", default="futures", choices=["spot", "futures"])
    ap.add_argument("--mode", choices=["rules", "ml", "fused"], default="rules")
    ap.add_argument("--resume", action="store_true", help="Resume from universe flow_*_last_*_ts columns")

    ap.add_argument("--tfs", default="15m,30m,60m,120m,240m")
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--ini", default=str(DEFAULT_INI))
    ap.add_argument("--run-id", default=f"FLOW_BACKFILL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--from-ts", default="")
    ap.add_argument("--max-bars", type=int, default=0) # Kept for compat, ignored in vec
    ap.add_argument("--flush-every", type=int, default=5000)
    ap.add_argument("--use-target-until", action="store_true") # Kept for compat

    args = ap.parse_args()

    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]
    from_ts = parse_dt(args.from_ts)

    if args.all_symbols:
        symbols = get_symbols(args.universe_where, args.limit)
    else:
        symbols = [args.symbol.strip().upper()] if args.symbol else get_symbols(args.universe_where, args.limit)

    if not symbols:
        print("No symbols selected.")
        return

    print(f"Starting Vectorized Backfill for {len(symbols)} symbols. Kind={args.kind} Mode={args.mode}")

    # Pre-flight check prompt (Automated)
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
                mode=args.mode,
                tfs=tfs,
                ini_path=args.ini,
                run_id=args.run_id,
                lookback_days=args.lookback_days,
                from_ts=from_ts,
                flush_every=args.flush_every,
                resume=args.resume
            )
        except Exception as e:
            print(f"[ERROR] {s}: {e}")
            import traceback
            traceback.print_exc()
            set_flow_error(s, args.kind, str(e))

if __name__ == "__main__":
    main()
