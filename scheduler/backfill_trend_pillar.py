from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2.extras

from utils.db import get_db_connection
from pillars.common import TZ, BaseCfg

from Trend.Pillar.trend_pillar import process_symbol_vectorized
# Reuse loaders from Flow (or common if moved)
from flow.pillar.flow_features_optimized import load_daily_futures, load_external_metrics

UNIVERSE_TABLE = "reference.symbol_universe"
DEFAULT_INI = Path(__file__).resolve().parent / "../Trend/Pillar/trend_scenarios.ini"

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

def _resume_col(kind: str, mode: str) -> str:
    # Assuming trend_rules_last_fut_ts pattern exists or will exist
    if mode == "ml":
        return "trend_ml_last_fut_ts" if kind == "futures" else "trend_ml_last_spot_ts"
    return "trend_rules_last_fut_ts" if kind == "futures" else "trend_rules_last_spot_ts"

def _audit_cols(mode: str) -> tuple[str, str]:
    if mode == "ml":
        return ("trend_ml_last_run_at", "trend_ml_last_run_id")
    return ("trend_rules_last_run_at", "trend_rules_last_run_id")

def update_trend_run_audit(symbol: str, run_id: str, mode: str) -> None:
    col_at, col_run = _audit_cols(mode)
    sql = f"""
        UPDATE {UNIVERSE_TABLE}
           SET {col_at} = NOW(),
               {col_run} = %s
         WHERE symbol = %s
    """
    # Fail silently if columns don't exist (migration pending)
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (run_id, symbol))
            conn.commit()
    except Exception as e:
        print(f"[WARN] Failed to update audit cols {col_at}/{col_run}: {e}")

def update_trend_last_ts(symbol: str, kind: str, mode: str, last_ts: datetime) -> None:
    col = _resume_col(kind, mode)
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=TZ)

    sql = f"UPDATE {UNIVERSE_TABLE} SET {col}=%s WHERE symbol=%s"
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (last_ts, symbol))
            conn.commit()
    except Exception as e:
        print(f"[WARN] Failed to update resume col {col}: {e}")

def set_trend_error(symbol: str, kind: str, msg: str) -> None:
    col = "fut_last_error" if kind == "futures" else "spot_last_error"
    sql = f"UPDATE {UNIVERSE_TABLE} SET {col}=%s WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (f"[TREND] {msg}"[:500], symbol))
        conn.commit()

def get_resume_ts(symbol: str, kind: str, mode: str) -> Optional[datetime]:
    col = _resume_col(kind, mode)
    sql = f"SELECT {col} FROM {UNIVERSE_TABLE} WHERE symbol=%s"
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (symbol,))
            r = cur.fetchone()
            return r[0] if r and r[0] else None
    except Exception:
        return None

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
    Writes rows directly to indicators.values.
    Expected row format: (symbol, kind, tf, ts, metric, val, ctx, run_id, source)
    """
    if not rows:
        return

    # Dedupe locally
    dedup = {}
    for r in rows:
        key = (r[0], r[1], r[2], r[3], r[4])
        dedup[key] = r
    rows_to_write = list(dedup.values())

    # print(f"[DB] Writing to table: indicators.values for {len(rows_to_write)} rows")

    sql = """
        INSERT INTO indicators.values
        (symbol, market_type, interval, ts, metric, val, context, run_id, source)
        VALUES %s
        ON CONFLICT (symbol, market_type, interval, ts, metric)
        DO UPDATE SET
            val = EXCLUDED.val,
            context = EXCLUDED.context,
            run_id = EXCLUDED.run_id,
            source = EXCLUDED.source
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, sql, rows_to_write, page_size=1000
                )
            conn.commit()
    except Exception as e:
        print(f"[DB ERROR] Failed to write rows: {e}")

# ============================================================
# Candle loader
# ============================================================

def load_15m(symbol: str, kind: str, lookback_days: int = 0, from_ts: Optional[datetime] = None) -> pd.DataFrame:
    start_dt = None
    if from_ts:
        start_dt = from_ts
    elif lookback_days > 0:
        start_dt = datetime.now(TZ) - timedelta(days=lookback_days)

    if kind == "futures":
        candidates = [("market.futures_candles", True)]
    else:
        candidates = [("market.spot_candles", True)]

    with get_db_connection() as conn:
        for table, has_interval in candidates:
            try:
                interval_clause = "AND interval='15m'" if has_interval else ""
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

    if resume and not from_ts:
        resume_ts = get_resume_ts(symbol, kind, mode)
        if resume_ts:
            print(f"[{symbol}] Resuming from {resume_ts}...")
            from_ts = resume_ts

    update_trend_run_audit(symbol, run_id, mode)

    print(f"[{symbol}] Starting Trend vectorized backfill...")

    # 1. Load 15m Candles
    df15 = load_15m(symbol, kind, lookback_days, from_ts)
    if df15.empty:
        print(f"[SKIP] {symbol} {kind}: no 15m data")
        return

    # 2. Load Daily Futures
    daily_df = load_daily_futures(symbol)

    # 3. Load Metrics
    start_ts = df15.index[0]
    end_ts = df15.index[-1]
    metrics_df = load_external_metrics(symbol, start_ts, end_ts)

    base_cfg = BaseCfg(tfs=tfs, lookback_days=lookback_days, run_id=run_id, source="TREND_BACKFILL_VEC")

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

        if tf == "15m" and rows:
            batch_max_ts = max(r[3] for r in rows)
            if max_ts_processed is None or batch_max_ts > max_ts_processed:
                max_ts_processed = batch_max_ts

        # 5. Write to DB
        batch_size = flush_every
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i+batch_size]
            write_values_direct(chunk)

        total_rows += len(rows)
        print(f"[{symbol}] {tf}: Wrote {len(rows)} rows.")

    if max_ts_processed:
        update_trend_last_ts(symbol, kind, mode, max_ts_processed)
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
    ap.add_argument("--resume", action="store_true", help="Resume from trend_*_last_*_ts columns")

    ap.add_argument("--tfs", default="15m,30m,60m,120m,240m")
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--ini", default=str(DEFAULT_INI))
    ap.add_argument("--run-id", default=f"TREND_BACKFILL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--from-ts", default="")
    ap.add_argument("--flush-every", type=int, default=5000)

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

    print(f"Starting Trend Vectorized Backfill for {len(symbols)} symbols. Kind={args.kind} Mode={args.mode}")

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
            set_trend_error(s, args.kind, str(e))

if __name__ == "__main__":
    main()
