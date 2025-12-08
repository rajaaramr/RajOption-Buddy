# scheduler/backfill_flow_pillar.py
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone, time
from typing import List, Tuple, Optional

import pandas as pd

from flow.pillar.flow_pillar import score_flow
from flow.runtime.utils import (
    get_flow_last_ts,
    update_flow_status,
    update_flow_ml_backfill_status,
)
from utils.db import get_db_connection
from pillars.common import (
    TZ,
    BaseCfg,
    resample,
    ensure_min_bars,
    maybe_trim_last_bar,
)

# ---------------------------------------------------------------------
# Load base 15m candles
# ---------------------------------------------------------------------
def load_base_15m(symbol: str, kind: str, lookback_days: int) -> pd.DataFrame:
    """
    Load 15m OHLCV(+OI) from Postgres for a single symbol.

    kind:
      - "futures" -> market.futures_candles
      - "spot"    -> market.spot_candles

    Assumes these tables store 15m candles.
    """
    cutoff = datetime.now(TZ) - timedelta(days=lookback_days)

    if kind == "futures":
        table = "market.futures_candles"
        cols = "ts, open, high, low, close, volume, oi"
    else:
        table = "market.spot_candles"
        cols = "ts, open, high, low, close, volume, 0::bigint AS oi"

    sql = f"""
        SELECT {cols}
          FROM {table}
         WHERE symbol = %s
           AND ts >= %s
           AND EXTRACT(ISODOW FROM ts) BETWEEN 1 AND 5   -- Mon–Fri only
           AND ts::time >= TIME '09:15:00'               -- after market open
           AND ts::time <= TIME '15:30:00'               -- before close
         ORDER BY ts
    """

    with get_db_connection() as conn:
        df = pd.read_sql(sql, conn, params=(symbol, cutoff))

    if df.empty:
        raise RuntimeError(f"No 15m data for {symbol} kind={kind} after cutoff={cutoff}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    # Extra safety: filter in Python with IST-aware checks
    ts_ist = df.index.tz_convert("Asia/Kolkata")

    # 1. Monday–Friday
    mask_weekday = ts_ist.dayofweek < 5  # 0=Mon, 4=Fri

    # 2. Time filter 09:15–15:30 IST
    t_open = time(9, 15)
    t_close = time(15, 30)
    mask_time = (ts_ist.time >= t_open) & (ts_ist.time <= t_close)

    df = df.loc[mask_weekday & mask_time]

    # numeric cleanup
    for col in ["open", "high", "low", "close", "volume", "oi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df


# ---------------------------------------------------------------------
# Min bars helper
# ---------------------------------------------------------------------
def min_bars_for_tf_rules(tf: str) -> int:
    """
    Return minimum bars required for rules-based flow scoring for a given TF.
    """
    min_bars_map = {
        "15m": 5,
        "30m": 3,
        "60m": 2,
        "120m": 2,
        "240m": 2,
    }
    return min_bars_map.get(tf, 5)


# ---------------------------------------------------------------------
# TF selection helper
# ---------------------------------------------------------------------
def get_flow_tfs(args_tf: Optional[str]) -> List[str]:
    """
    Decide which TFs to backfill.

    For now:
      - if --tf provided -> use only that
      - else -> default: 15m,30m,60m,120m,240m
    """
    if args_tf:
        return [args_tf]

    return ["15m", "30m", "60m", "120m", "240m"]


# ---------------------------------------------------------------------
# Universe loader
# ---------------------------------------------------------------------
def load_universe_symbols() -> List[str]:
    """
    Load universe of symbols from reference.symbol_universe.
    Only enabled symbols.
    """
    sql = """
        SELECT DISTINCT symbol
          FROM reference.symbol_universe
         WHERE enabled = TRUE
         ORDER BY symbol
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    return [r[0] for r in rows] if rows else []


# ---------------------------------------------------------------------
# from_ts helper for delta logic
# ---------------------------------------------------------------------
def compute_from_ts(
    now_utc: datetime,
    lookback_days: int,
    last_ts_status: Optional[datetime],
    from_date_str: Optional[str],
) -> datetime:
    """
    Decide from_ts for delta backfill.

    Priority:
      1) If --from-date provided → use that (IST 00:00 converted to UTC),
         but never later than now_utc.
      2) Else if last_ts_status → max(last_ts_status, now_utc - lookback_days)
      3) Else → now_utc - lookback_days
    """
    lb_default = now_utc - timedelta(days=lookback_days)

    if from_date_str:
        dt = datetime.fromisoformat(from_date_str.strip())  # naive local date
        dt_ist = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        dt_utc = dt_ist - timedelta(hours=5, minutes=30)
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        return min(now_utc, dt_utc)

    if last_ts_status is not None:
        if last_ts_status.tzinfo is None:
            last_ts_status = last_ts_status.replace(tzinfo=timezone.utc)
        return max(last_ts_status, lb_default)

    return lb_default


# ---------------------------------------------------------------------
# Core backfill logic for ONE symbol (all TFs)
# ---------------------------------------------------------------------
def backfill_flow_for_symbol(
    symbol: str,
    kind: str,
    tfs: List[str],
    lookback_days: int,
    max_bars: int,
    flow_ini_path: str,
    run_id: str,
    source: str,
    mode: str,
    from_ts: Optional[datetime],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    For a given symbol+kind, backfill FLOW for the given TFs.

    - Loads 15m candles once.
    - For each TF:
        * Resample 15m → TF.
        * Build candidate TS list (respecting from_ts + max_bars).
        * For each TS: slice DF up to TS and call score_flow(...).

    Returns:
        (rules_last_ts, ml_last_ts) based on what score_flow() actually produced.
    """
    mode = (mode or "rules").lower()
    print(
        f"[FLOW_BACKFILL] symbol={symbol} kind={kind} "
        f"tfs={tfs} lookback_days={lookback_days} max_bars={max_bars} mode={mode} from_ts={from_ts}"
    )

    # 1) Load base 15m history once
    try:
        df15 = load_base_15m(symbol, kind, lookback_days)
    except Exception as e:
        print(f"[FLOW_BACKFILL] {symbol} {kind}: ERROR loading 15m data -> {e}")
        return (None, None)

    if df15.empty:
        print(f"[FLOW_BACKFILL] {symbol} {kind}: no 15m data after load, skipping")
        return (None, None)

    base = BaseCfg(
        tfs=tfs,
        lookback_days=lookback_days,
        run_id=run_id,
        source=source,
    )

    total_scored = 0
    rules_last_ts: Optional[datetime] = None
    ml_last_ts: Optional[datetime] = None

    for tf in tfs:
        # --- Build TF-level series to get timestamps we want to score ---
        if tf == "15m":
            dftf = df15.copy()
        else:
            dftf = resample(df15, tf)

        dftf = maybe_trim_last_bar(dftf)

        # ---- use *rules* min-bars, not ML min-bars ----
        n = len(dftf)
        rules_min = min_bars_for_tf_rules(tf)
        if n < rules_min:
            print(
                f"[FLOW_BACKFILL] {symbol} {kind} tf={tf}: "
                f"only {n} bars, need {rules_min} for rules → skipping TF"
            )
            continue

        # Candidate TF timestamps in ascending order
        ts_list = list(dftf.index)

        # Apply from_ts filter for delta runs
        if from_ts is not None:
            ts_list = [ts for ts in ts_list if ts >= from_ts]

        if not ts_list:
            print(f"[FLOW_BACKFILL] {symbol} {kind} tf={tf}: no TF bars after from_ts={from_ts}, skipping")
            continue

        # Apply max_bars cap from the end
        if max_bars is not None and max_bars > 0:
            ts_list = ts_list[-max_bars:]

        print(f"[FLOW_BACKFILL] {symbol} {kind} tf={tf}: {len(ts_list)} TF bars to score")

        scored_tf = 0

        for ts in ts_list:
            df_slice = df15.loc[:ts]
            if df_slice.empty:
                continue

            try:
                res = score_flow(
                    symbol=symbol,
                    kind=kind,
                    tf=tf,
                    df5=df_slice,
                    base=base,
                    ini_path=flow_ini_path,
                    context=None,
                )
            except Exception as e:
                print(f"[FLOW_BACKFILL] ERROR symbol={symbol} kind={kind} tf={tf} ts={ts}: {e}")
                continue

            if res is None:
                # not enough bars / calc skipped upstream
                continue

            # -------------------------------------------------
            # Tolerant to different score_flow() signatures
            # -------------------------------------------------
            if isinstance(res, tuple):
                if len(res) == 4:
                    ts_scored, fused_score, fused_veto, extras = res
                    rules_score = extras.get("rules_score")
                    ml_score = extras.get("ml_score")
                elif len(res) == 6:
                    ts_scored, rules_score, rules_veto, ml_score, fused_score, fused_veto = res
                    extras = {}
                elif len(res) == 7:
                    ts_scored, rules_score, rules_veto, ml_score, fused_score, fused_veto, extras = res
                else:
                    print(
                        f"[FLOW_BACKFILL] Unexpected score_flow() return len={len(res)} "
                        f"for {symbol} {kind} tf={tf} ts={ts}"
                    )
                    continue
            else:
                print(
                    f"[FLOW_BACKFILL] Unexpected score_flow() return type={type(res)} "
                    f"for {symbol} {kind} tf={tf} ts={ts}"
                )
                continue

            # -------------------------
            # Track last rules / ML ts
            # -------------------------
            if rules_score is not None:
                rules_last_ts = ts_scored if rules_last_ts is None else max(rules_last_ts, ts_scored)

            if ml_score is not None:
                ml_last_ts = ts_scored if ml_last_ts is None else max(ml_last_ts, ts_scored)

            scored_tf += 1

        print(f"[FLOW_BACKFILL] {symbol} {kind} tf={tf}: scored {scored_tf} bars")
        total_scored += scored_tf

    print(
        f"[FLOW_BACKFILL] DONE {symbol} {kind}: "
        f"total scored bars across TFs = {total_scored}"
    )

    # Respect mode for what we *report* upwards:
    if mode == "rules":
        return (rules_last_ts, None)
    elif mode == "ml":
        return (None, ml_last_ts)
    else:  # "both"
        return (rules_last_ts, ml_last_ts)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()

    # --- symbol selection ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--symbol",
        help="Single root symbol (e.g. ICICIBANK, INDIGO, etc.)",
    )
    group.add_argument(
        "--all-symbols",
        action="store_true",
        help="Run for all symbols from reference.symbol_universe",
    )

    # --- core options ---
    parser.add_argument("--kind", default="futures", choices=["futures", "spot"])
    parser.add_argument(
        "--tf",
        default=None,
        help="Single TF (e.g. 15m,30m,60m,120m,240m). If omitted, runs all.",
    )
    parser.add_argument("--lookback-days", type=int, default=40)
    parser.add_argument("--max-bars", type=int, default=20)

    # --- mode + from-date ---
    parser.add_argument(
        "--mode",
        choices=["rules", "ml", "both"],
        default="rules",
        help="What to compute: rules-only, ml-only, or both",
    )
    parser.add_argument(
        "--from-date",
        help="Optional manual from-date (YYYY-MM-DD). If given, overrides status-based from_ts.",
    )

    # --- config + run-id ---
    parser.add_argument(
        "--flow-ini",
        default="flow_scenarios.ini",
        help="Path to flow_scenarios.ini (rules & [flow_ml])",
    )
    parser.add_argument(
        "--run-id",
        default="FLOW_BACKFILL",
        help="run_id to stamp in indicators.values",
    )
    parser.add_argument(
        "--source",
        default="flow_backfill",
        help="source to stamp in indicators.values",
    )

    parser.add_argument(
        "--use-expiry-selector",
        action="store_true",
        help="(placeholder) Reserved for futures expiry selection; currently ignored.",
    )

    args = parser.parse_args()

    # -------------------------
    # 1) Resolve TFs + symbols
    # -------------------------
    tfs = get_flow_tfs(args.tf)

    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = load_universe_symbols()
        if not symbols:
            print("[FLOW_BACKFILL] No symbols found in reference.symbol_universe")
            return
        print(f"[FLOW_BACKFILL] Loaded {len(symbols)} symbols from reference.symbol_universe")

    if args.use_expiry_selector and args.kind == "futures":
        print(
            "[FLOW_BACKFILL] WARNING: --use-expiry-selector requested, "
            "but expiry selection is not yet wired to your metadata schema. "
            "Using symbol as-is for futures backfill."
        )

    # -------------------------
    # 2) Mode / timing context
    # -------------------------
    now_utc = datetime.now(timezone.utc)
    kind = args.kind.lower()
    mode = args.mode.lower()
    lookback_days = args.lookback_days
    run_id = args.run_id
    source = args.source

    # -------------------------
    # 3) Per-symbol delta logic
    # -------------------------
    with get_db_connection() as conn:
        for symbol in symbols:
            print(f"[FLOW_BACKFILL] symbol={symbol} kind={kind} mode={mode}")

            # ---- 3a) Read last status timestamps from symbol_universe ----
            last_rules_ts, last_ml_ts = get_flow_last_ts(conn, symbol, kind, mode="both")

            if mode == "rules":
                last_ts_status = last_rules_ts
            elif mode == "ml":
                last_ts_status = last_ml_ts
            else:  # "both"
                if last_rules_ts and last_ml_ts:
                    last_ts_status = min(last_rules_ts, last_ml_ts)
                else:
                    last_ts_status = last_rules_ts or last_ml_ts

            # ---- 3b) Decide from_ts using helper ----
            from_ts = compute_from_ts(
                now_utc=now_utc,
                lookback_days=lookback_days,
                last_ts_status=last_ts_status,
                from_date_str=args.from_date,
            )

            # -----------------------------
            # 4) Delegate work for symbol
            # -----------------------------
            rules_last_scored_ts, ml_last_scored_ts = backfill_flow_for_symbol(
                symbol=symbol,
                kind=kind,
                tfs=tfs,
                lookback_days=lookback_days,
                max_bars=args.max_bars,
                flow_ini_path=args.flow_ini,
                run_id=run_id,
                source=source,
                mode=mode,
                from_ts=from_ts,
            )

            # -----------------------------
            # 5) Map timestamps → futures/spot RULES columns
            # -----------------------------
            rules_ts_fut: Optional[datetime] = None
            rules_ts_spot: Optional[datetime] = None

            if mode in ("rules", "both") and rules_last_scored_ts is not None:
                if kind == "futures":
                    rules_ts_fut = rules_last_scored_ts
                else:
                    rules_ts_spot = rules_last_scored_ts

            # ---- 5a) Update RULES status (only rules_* columns) ----
            if mode in ("rules", "both") and (rules_ts_fut or rules_ts_spot):
                update_flow_status(
                    conn,
                    symbol=symbol,
                    run_id=run_id,
                    rules_ts_fut=rules_ts_fut,
                    rules_ts_spot=rules_ts_spot,
                    mode="rules",   # 🔒 rules-only
                )

            # ---- 5b) Update ML status (per-symbol, monotonic) ----
            if mode in ("ml", "both") and ml_last_scored_ts is not None:
                update_flow_ml_backfill_status(
                    conn=conn,
                    symbol=symbol,
                    kind=kind,
                    last_ts=ml_last_scored_ts,
                    run_id=run_id,
                )


if __name__ == "__main__":
    main()
