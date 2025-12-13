from __future__ import annotations

import argparse
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.db import get_db_connection
from pillars.common import TZ, BaseCfg, write_values, resample, maybe_trim_last_bar

from flow.pillar.flow_pillar import score_flow


UNIVERSE_TABLE = "reference.symbol_universe"


# ============================================================
# Universe helpers (YOUR schema)
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
    """
    Which column holds the resume pointer.
    rules: flow_rules_last_{fut|spot}_ts
    ml:    flow_ml_last_{fut|spot}_ts
    fused: resume from rules pointer (safe)
    """
    if mode == "ml":
        return "flow_ml_last_fut_ts" if kind == "futures" else "flow_ml_last_spot_ts"
    # rules or fused
    return "flow_rules_last_fut_ts" if kind == "futures" else "flow_rules_last_spot_ts"


def _audit_cols(mode: str) -> tuple[str, str]:
    """
    Which audit columns to update for run_at/run_id.
    """
    if mode == "ml":
        return ("flow_ml_last_run_at", "flow_ml_last_run_id")
    return ("flow_rules_last_run_at", "flow_rules_last_run_id")


def get_resume_ts(symbol: str, kind: str, mode: str) -> Optional[datetime]:
    col = _resume_col(kind, mode)
    sql = f"SELECT {col} FROM {UNIVERSE_TABLE} WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol,))
        r = cur.fetchone()
        return r[0] if r and r[0] else None


def get_target_until_ts(symbol: str, kind: str) -> Optional[datetime]:
    col = "fut_15m_target_until_ts" if kind == "futures" else "spot_15m_target_until_ts"
    sql = f"SELECT {col} FROM {UNIVERSE_TABLE} WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol,))
        r = cur.fetchone()
        return r[0] if r and r[0] else None


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


# ============================================================
# Candle loader (adjust here ONLY if table names differ)
# ============================================================

def load_15m(symbol: str, kind: str) -> pd.DataFrame:
    if kind == "futures":
        candidates = [
            ("market.futures_candles", True),      # has interval col
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

    raise RuntimeError(
        f"[LOAD] No candle source worked for {symbol} kind={kind}. Update load_15m() candidates."
    )


# ============================================================
# Utils
# ============================================================

def parse_dt(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    s = s.replace(" ", "T")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt


def _to_dt_ist(ts: pd.Timestamp) -> datetime:
    return ts.to_pydatetime().replace(tzinfo=TZ)

def dedupe_rows(rows):
    seen = set()
    out = []
    for r in rows:
        key = (r[0], r[1], r[2], r[3], r[4])  # symbol, kind, tf, ts, metric
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
# ============================================================
# Backfill core
# ============================================================


def backfill_symbol(
    symbol: str,
    kind: str,
    mode: str,
    tfs: List[str],
    lookback_days: int,
    ini_path: str,
    run_id: str,
    from_ts: Optional[datetime],
    max_bars: int,
    flush_every: int,
    use_target_until: bool,
    resume: bool,
) -> None:
    symbol = symbol.strip().upper()

    if not is_symbol_enabled(symbol):
        print(f"[SKIP] {symbol} disabled")
        return

    update_flow_run_audit(symbol, run_id, mode)

    df15 = load_15m(symbol, kind)
    if df15 is None or df15.empty:
        print(f"[SKIP] {symbol} {kind}: no 15m data")
        return

    df15 = maybe_trim_last_bar(df15)
    if len(df15) < 5:
        print(f"[SKIP] {symbol} {kind}: too few bars")
        return

    base = BaseCfg(tfs=tfs, lookback_days=lookback_days, run_id=run_id, source="FLOW_BACKFILL")

    # Optional stop boundary
    target_until = get_target_until_ts(symbol, kind) if use_target_until else None
    target_until_utc = None
    if target_until:
        try:
            target_until_utc = pd.Timestamp(target_until).tz_convert("UTC")
        except Exception:
            target_until_utc = pd.Timestamp(target_until).tz_localize("Asia/Kolkata").tz_convert("UTC")

    idx15 = df15.index.values  # numpy datetime64[ns, UTC]

    # mode -> force_ml
    force_ml = None
    if mode == "rules":
        force_ml = False
    elif mode in ("ml", "fused"):
        force_ml = True

    # ✅ Freeze resume pointer ONCE per symbol run
    resume_ts0: Optional[datetime] = None
    if resume:
        resume_ts0 = get_resume_ts(symbol, kind, mode)

    eff_from0 = resume_ts0
    if from_ts and (eff_from0 is None or from_ts > eff_from0):
        eff_from0 = from_ts

    # Convert eff_from0 to UTC once (fast filtering)
    eff_from0_utc = None
    if eff_from0:
        try:
            eff_from0_utc = pd.Timestamp(eff_from0).tz_convert("UTC")
        except Exception:
            eff_from0_utc = pd.Timestamp(eff_from0).tz_localize("Asia/Kolkata").tz_convert("UTC")

    # ✅ Final pointer must track ONLY 15m progress
    final_last_ts_15m: Optional[datetime] = None

    for tf in tfs:
        if not is_symbol_enabled(symbol):
            print(f"[PAUSE] {symbol} disabled mid-run. stopping.")
            return

        dftf = df15 if tf == "15m" else maybe_trim_last_bar(resample(df15, tf))
        if dftf is None or dftf.empty:
            continue

        if target_until_utc is not None:
            dftf = dftf.loc[:target_until_utc]
            if dftf.empty:
                continue

        ts_tf = dftf.index
        if eff_from0_utc is not None:
            ts_tf = ts_tf[ts_tf > eff_from0_utc]

        if len(ts_tf) == 0:
            print(f"[UPTODATE] {symbol} {kind} tf={tf}")
            continue

        if max_bars and max_bars > 0:
            ts_tf = ts_tf[-max_bars:]

        end_pos = np.searchsorted(idx15, ts_tf.values, side="right")

        batch_rows = []
        last_written_dt: Optional[datetime] = None
        scored = 0

        for i, pos in enumerate(end_pos):
            if (i % 200) == 0 and (not is_symbol_enabled(symbol)):
                print(f"[PAUSE] {symbol} disabled. stopping tf={tf}")
                return

            if pos <= 0:
                continue

            df_slice = df15.iloc[:pos]

            try:
                res = score_flow(
                    symbol=symbol,
                    kind=kind,
                    tf=tf,
                    df5=df_slice,
                    base=base,
                    ini_path=ini_path,
                    context=None,
                    force_ml=force_ml,
                    write_to_db=False,
                )
            except TypeError:
                res = score_flow(
                    symbol=symbol,
                    kind=kind,
                    tf=tf,
                    df5=df_slice,
                    base=base,
                    ini_path=ini_path,
                    context=None,
                    force_ml=force_ml,
                )

            if not res:
                continue

            ts_scored = res[0]
            rows = res[6] if (len(res) >= 7) else None

            if rows:
                batch_rows.extend(rows)

            last_written_dt = ts_scored
            scored += 1

            # ✅ No universe pointer update mid-run
            if rows and flush_every and len(batch_rows) >= flush_every:
                if "dedupe_rows" in globals():
                    batch_rows = dedupe_rows(batch_rows)
                write_values(batch_rows)
                batch_rows.clear()

        if batch_rows:
            if "dedupe_rows" in globals():
                batch_rows = dedupe_rows(batch_rows)
            write_values(batch_rows)
            batch_rows.clear()

        # ✅ Capture pointer ONLY from 15m
        if tf == "15m" and last_written_dt:
            final_last_ts_15m = last_written_dt

        print(f"[DONE] {symbol} {kind} mode={mode} tf={tf} scored={scored} last={last_written_dt}")

    # ✅ Update universe pointer ONCE per symbol run (15m only)
    if final_last_ts_15m:
        update_flow_last_ts(symbol, kind, mode, final_last_ts_15m)
        print(f"[RESUME_PTR] {symbol} {kind} mode={mode} set={final_last_ts_15m} (from tf=15m)")



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--symbol", default="", help="Single symbol (if not using --all-symbols)")
    ap.add_argument("--all-symbols", action="store_true", help="Run for all symbols in universe filter")
    ap.add_argument("--universe-where", default="enabled = TRUE", help="SQL WHERE for reference.symbol_universe")

    ap.add_argument("--limit", type=int, default=0, help="limit universe symbols")
    ap.add_argument("--kind", default="futures", choices=["spot", "futures"])
    ap.add_argument("--mode", choices=["rules", "ml", "fused"], default="rules")

    ap.add_argument("--resume", action="store_true", help="Resume from universe flow_*_last_*_ts columns")

    ap.add_argument("--tfs", default="15m,30m,60m,120m,240m")
    ap.add_argument("--lookback-days", type=int, default=180)

    ap.add_argument("--ini", required=True)
    ap.add_argument("--run-id", default=f"FLOW_BACKFILL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--from-ts", default="")
    ap.add_argument("--max-bars", type=int, default=0)
    ap.add_argument("--flush-every", type=int, default=1500)
    ap.add_argument("--use-target-until", action="store_true")

    args = ap.parse_args()

    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]
    from_ts = parse_dt(args.from_ts)

    if args.all_symbols:
        symbols = get_symbols(args.universe_where, args.limit)
    else:
        symbols = [args.symbol.strip().upper()] if args.symbol else get_symbols(args.universe_where, args.limit)

    if not symbols:
        raise SystemExit("No symbols selected. Use --symbol or --all-symbols or relax --universe-where.")

    print(f"[FLOW_BACKFILL] symbols={len(symbols)} kind={args.kind} mode={args.mode} tfs={tfs} run_id={args.run_id} resume={args.resume}")

    for s in symbols:
        try:
            backfill_symbol(
                symbol=s,
                kind=args.kind,
                mode=args.mode,
                tfs=tfs,
                lookback_days=args.lookback_days,
                ini_path=args.ini,
                run_id=args.run_id,
                from_ts=from_ts,
                max_bars=args.max_bars,
                flush_every=args.flush_every,
                use_target_until=args.use_target_until,
                resume=args.resume,
            )
        except Exception as e:
            print(f"[ERROR] {s} {args.kind}: {e}")
            set_flow_error(s, args.kind, str(e))


if __name__ == "__main__":
    main()
