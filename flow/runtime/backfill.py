from __future__ import annotations

import argparse
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.db import get_db_connection
from pillars.common import TZ, BaseCfg, write_values, resample, maybe_trim_last_bar

# IMPORTANT:
# Your flow_pillar.score_flow must support write_to_db=False and return rows
# (or you can keep write_to_db=True and skip batching; but batching is faster)
from flow.pillar.flow_pillar import score_flow


UNIVERSE_TABLE = "reference.symbol_universe"


# -----------------------------
# Universe helpers (YOUR schema)
# -----------------------------
def get_enabled_symbols(limit: int = 0) -> List[str]:
    sql = f"""
        SELECT symbol
        FROM {UNIVERSE_TABLE}
        WHERE enabled = TRUE
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


def get_resume_ts(symbol: str, kind: str) -> Optional[datetime]:
    col = "flow_rules_last_fut_ts" if kind == "futures" else "flow_rules_last_spot_ts"
    sql = f"SELECT {col} FROM {UNIVERSE_TABLE} WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol,))
        r = cur.fetchone()
        return r[0] if r and r[0] else None


def get_target_until_ts(symbol: str, kind: str) -> Optional[datetime]:
    # Optional. If NULL, we backfill till end of available candles.
    col = "fut_15m_target_until_ts" if kind == "futures" else "spot_15m_target_until_ts"
    sql = f"SELECT {col} FROM {UNIVERSE_TABLE} WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol,))
        r = cur.fetchone()
        return r[0] if r and r[0] else None


def update_flow_run_audit(symbol: str, run_id: str) -> None:
    sql = f"""
        UPDATE {UNIVERSE_TABLE}
        SET flow_rules_last_run_at = NOW(),
            flow_rules_last_run_id = %s
        WHERE symbol = %s
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (run_id, symbol))
        conn.commit()


def update_flow_last_ts(symbol: str, kind: str, last_ts: datetime) -> None:
    col = "flow_rules_last_fut_ts" if kind == "futures" else "flow_rules_last_spot_ts"
    sql = f"UPDATE {UNIVERSE_TABLE} SET {col}=%s WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (last_ts, symbol))
        conn.commit()


def set_flow_error(symbol: str, kind: str, msg: str) -> None:
    # You have generic last_error fields per market-type.
    col = "fut_last_error" if kind == "futures" else "spot_last_error"
    sql = f"UPDATE {UNIVERSE_TABLE} SET {col}=%s WHERE symbol=%s"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (msg[:500], symbol))
        conn.commit()


# -----------------------------
# Candle loader (adjusted to your likely tables)
# -----------------------------
def load_15m(symbol: str, kind: str, lookback_days: int) -> pd.DataFrame:
    """
    Uses your typical candle tables.
    If your actual table names differ, change them HERE (only one place).
    """
    if kind == "futures":
        candidates = [
            ("market.futures_candles", True),      # (table, has_interval_col)
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

    raise RuntimeError(f"[LOAD] No candle source worked for {symbol} kind={kind}. Update load_15m() table list.")


# -----------------------------
# Backfill engine
# -----------------------------
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
    # keep everything consistent with TZ from pillars.common
    return ts.to_pydatetime().replace(tzinfo=TZ)


def backfill_symbol(
    symbol: str,
    kind: str,
    tfs: List[str],
    lookback_days: int,
    ini_path: str,
    run_id: str,
    from_ts: Optional[datetime],
    max_bars: int,
    flush_every: int,
    use_target_until: bool,
) -> None:
    if not is_symbol_enabled(symbol):
        print(f"[SKIP] {symbol} disabled")
        return

    update_flow_run_audit(symbol, run_id)

    df15 = load_15m(symbol, kind, lookback_days)
    if df15 is None or df15.empty:
        print(f"[SKIP] {symbol} {kind}: no 15m data")
        return

    df15 = maybe_trim_last_bar(df15)

    # BaseCfg is needed for write_values rows
    base = BaseCfg(tfs=tfs, lookback_days=lookback_days, run_id=run_id, source="FLOW_BACKFILL")

    # Optional target stop
    target_until = get_target_until_ts(symbol, kind) if use_target_until else None

    idx15 = df15.index.values  # numpy datetime64[ns, UTC]
    if len(idx15) < 5:
        print(f"[SKIP] {symbol} {kind}: too few bars")
        return

    for tf in tfs:
        if not is_symbol_enabled(symbol):
            print(f"[PAUSE] {symbol} disabled mid-run. stopping.")
            return

        # resume pointer from universe table
        resume_ts = get_resume_ts(symbol, kind)
        eff_from = resume_ts
        if from_ts and (eff_from is None or from_ts > eff_from):
            eff_from = from_ts

        # build tf dataframe ONCE
        dftf = df15 if tf == "15m" else maybe_trim_last_bar(resample(df15, tf))
        if dftf is None or dftf.empty:
            continue

        # apply target stop if requested
        if target_until:
            dftf = dftf.loc[:pd.Timestamp(target_until).tz_convert("UTC") if hasattr(pd.Timestamp(target_until), "tz_convert") else pd.Timestamp(target_until)]
            if dftf.empty:
                continue

        ts_tf = dftf.index
        # filter by resume/from
        if eff_from:
            ts_tf = ts_tf[ts_tf.to_series().apply(lambda x: _to_dt_ist(x) > eff_from).values]

        if len(ts_tf) == 0:
            print(f"[UPTODATE] {symbol} {kind} tf={tf}")
            continue

        if max_bars and max_bars > 0:
            ts_tf = ts_tf[-max_bars:]

        # PERF: map each tf timestamp to a df15 end position via searchsorted
        # We need slice df15 up to ts (15m index), using iloc[:pos]
        # Convert ts_tf to numpy datetime64[ns, UTC] array compatible with idx15
        ts_tf_np = ts_tf.values
        end_pos = np.searchsorted(idx15, ts_tf_np, side="right")  # gives index in df15

        batch_rows = []
        last_written_dt: Optional[datetime] = resume_ts
        scored = 0

        for i, pos in enumerate(end_pos):
            # pause check every ~200 bars (cheap)
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
                    write_to_db=False,   # <<< must be supported in score_flow
                )
            except TypeError:
                # fallback if you haven't added write_to_db flag yet
                res = score_flow(
                    symbol=symbol,
                    kind=kind,
                    tf=tf,
                    df5=df_slice,
                    base=base,
                    ini_path=ini_path,
                    context=None,
                )

            if not res:
                continue

            ts_scored = res[0]
            # If you added "rows" return, take it; else assume it already wrote
            rows = res[6] if len(res) >= 7 else None

            if rows:
                batch_rows.extend(rows)

            last_written_dt = ts_scored
            scored += 1

            if rows and flush_every and len(batch_rows) >= flush_every:
                write_values(batch_rows)
                batch_rows.clear()
                update_flow_last_ts(symbol, kind, last_written_dt)

        # final flush
        if batch_rows:
            write_values(batch_rows)
            batch_rows.clear()

        if last_written_dt:
            update_flow_last_ts(symbol, kind, last_written_dt)

        print(f"[DONE] {symbol} {kind} tf={tf} scored={scored} last={last_written_dt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="")
    ap.add_argument("--limit", type=int, default=0, help="limit universe symbols")
    ap.add_argument("--kind", default="futures", choices=["spot", "futures"])
    ap.add_argument("--tfs", default="15m,30m,60m,120m,240m")
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--ini", required=True)
    ap.add_argument("--run-id", default=f"FLOW_BACKFILL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--from-ts", default="")
    ap.add_argument("--max-bars", type=int, default=0)
    ap.add_argument("--flush-every", type=int, default=1500, help="rows buffer flush threshold")
    ap.add_argument("--use-target-until", action="store_true", help="stop at *_target_until_ts if set")
    args = ap.parse_args()

    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]
    from_ts = parse_dt(args.from_ts)

    symbols = [args.symbol.strip().upper()] if args.symbol else get_enabled_symbols(args.limit)
    print(f"[FLOW_BACKFILL] symbols={len(symbols)} kind={args.kind} tfs={tfs} run_id={args.run_id}")

    for s in symbols:
        try:
            backfill_symbol(
                symbol=s,
                kind=args.kind,
                tfs=tfs,
                lookback_days=args.lookback_days,
                ini_path=args.ini,
                run_id=args.run_id,
                from_ts=from_ts,
                max_bars=args.max_bars,
                flush_every=args.flush_every,
                use_target_until=args.use_target_until,
            )
        except Exception as e:
            print(f"[ERROR] {s} {args.kind}: {e}")
            set_flow_error(s, args.kind, str(e))


if __name__ == "__main__":
    main()
