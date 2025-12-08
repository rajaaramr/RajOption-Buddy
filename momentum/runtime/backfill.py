# momentum/runtime/backfill.py
from __future__ import annotations

import argparse
import datetime as dt
import sys
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

from utils.db import get_db_connection
from pillars.common import resample, BaseCfg
from momentum.pillar.momentum_pillar import score_momentum

# ---------------------------------
# CONFIG
# ---------------------------------
BASE_TF = "15m"
TF_LIST = ["15m", "30m", "60m", "120m", "240m"]
WARMUP_DAYS = 10  # extra history before from_ts for indicators


# ---------------------------------
# CLI
# ---------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Momentum pillar backfill")

    g_sym = p.add_mutually_exclusive_group(required=True)
    g_sym.add_argument("--symbol", help="Single symbol")
    g_sym.add_argument("--all-symbols", action="store_true", help="All enabled symbols")

    p.add_argument("--kind", choices=["futures", "spot"], required=True)
    p.add_argument("--mode", choices=["rules", "ml", "both"], required=True)

    p.add_argument(
        "--from-date",
        help="Override start date (YYYY-MM-DD). If set, ignores last_ts + lookback-days.",
        default=None,
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help=(
            "If set, score last N days BACK from now (full history in that window). "
            "Does NOT use mom_*_last_*_ts for resume – it rewrites that window."
        ),
    )

    p.add_argument(
        "--momentum-ini",
        help="Path to momentum_scenarios.ini (optional). If not set, default is used.",
        default=None,
    )
    p.add_argument("--run-id", help="Run identifier", default=None)
    p.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Limit number of symbols for testing",
    )

    return p.parse_args()


# ---------------------------------
# Universe + status
# ---------------------------------
def _load_universe(
    kind: str,
    symbol: Optional[str] = None,
    max_symbols: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load symbols + existing momentum status from reference.symbol_universe.
    We keep mom_*_last_*_ts ONLY for writing back, not for resume.
    """
    if kind not in ("futures", "spot"):
        raise ValueError(f"Unknown kind={kind}")

    cols = [
        "symbol",
        "mom_rules_last_spot_ts",
        "mom_rules_last_fut_ts",
        "mom_ml_last_spot_ts",
        "mom_ml_last_fut_ts",
        "enabled",
    ]

    sql = f"""
        SELECT {", ".join(cols)}
          FROM reference.symbol_universe
         WHERE enabled = TRUE
    """

    params: List[Any] = []
    if symbol:
        sql += " AND symbol = %s"
        params.append(symbol)

    sql += " ORDER BY symbol"
    if max_symbols:
        sql += f" LIMIT {int(max_symbols)}"

    with get_db_connection() as conn:
        df = pd.read_sql(sql, conn, params=params)

    return df


def _get_last_ts_for_kind(row: pd.Series, kind: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return (rules_ts, ml_ts) for given kind."""
    if kind == "futures":
        rules_ts = row["mom_rules_last_fut_ts"]
        ml_ts = row["mom_ml_last_fut_ts"]
    else:
        rules_ts = row["mom_rules_last_spot_ts"]
        ml_ts = row["mom_ml_last_spot_ts"]

    rules_ts = pd.to_datetime(rules_ts) if rules_ts is not None else None
    ml_ts = pd.to_datetime(ml_ts) if ml_ts is not None else None
    return rules_ts, ml_ts


def _compute_from_ts(
    now: dt.datetime,
    mode: str,
    from_date_str: Optional[str],
    lookback_days: Optional[int],
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    IMPORTANT BEHAVIOUR:

    - If from-date is given:
          from_ts = that date (start of day, UTC)
    - Else if lookback-days is given:
          from_ts = now - lookback_days (UTC, tz-aware)
    - Else:
          from_ts = None → score FULL history available.
    """
    base_from: Optional[pd.Timestamp] = None

    if from_date_str:
        # from_date_str is naive → localize to UTC
        base_from = pd.Timestamp(from_date_str).tz_localize("UTC")
    elif lookback_days is not None:
        # now is already tz-aware (UTC), so just subtract days
        dt_from = now - dt.timedelta(days=lookback_days)
        base_from = pd.Timestamp(dt_from)  # keep tzinfo from dt_from

    rules_from: Optional[pd.Timestamp] = None
    ml_from: Optional[pd.Timestamp] = None

    if mode in ("rules", "both"):
        rules_from = base_from
    if mode in ("ml", "both"):
        ml_from = base_from

    return rules_from, ml_from


# ---------------------------------
# Candle loader – 15m BASE
# ---------------------------------
def _load_15m_candles(
    symbol: str,
    kind: str,
    from_ts: Optional[pd.Timestamp],
    now: dt.datetime,
) -> pd.DataFrame:
    """
    Load 15m candles from:
      - market.futures_candles
      - market.spot_candles

    No tf column – table itself is 15m.
    """
    if kind == "futures":
        table = "market.futures_candles"
    else:
        table = "market.spot_candles"

    if from_ts is not None:
        candle_from = (from_ts - dt.timedelta(days=WARMUP_DAYS)).to_pydatetime()
    else:
        candle_from = (now - dt.timedelta(days=WARMUP_DAYS + 365)).replace(tzinfo=None)

    sql = f"""
        SELECT ts, open, high, low, close, volume
          FROM {table}
         WHERE symbol = %s
           AND ts >= %s
         ORDER BY ts
    """

    with get_db_connection() as conn:
        df = pd.read_sql(sql, conn, params=(symbol, candle_from))

    if df.empty:
        return pd.DataFrame()

    # Normalize to tz-aware UTC index
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    return df


# ---------------------------------
# Core per-symbol processing
# ---------------------------------
def _process_symbol(
    row: pd.Series,
    kind: str,
    mode: str,
    args: argparse.Namespace,
    now: dt.datetime,
) -> Dict[str, Optional[pd.Timestamp]]:
    """
    For a single symbol:
      - Decide from_ts window based on lookback / from-date.
      - Load 15m candles for that window + warmup.
      - For each TF (15m/30m/60m/120m), and for each bar in the window,
        call score_momentum() so we get FULL HISTORY in that window.
    """
    symbol = row["symbol"]

    # Existing last_ts (only for status write-back)
    rules_last_ts_old, ml_last_ts_old = _get_last_ts_for_kind(row, kind)

    # Compute from_ts purely from CLI args (no resume).
    rules_from, ml_from = _compute_from_ts(
        now=now,
        mode=mode,
        from_date_str=args.from_date,
        lookback_days=args.lookback_days,
    )

    # Effective window start = earliest of rules_from / ml_from
    effective_from = min(
        [t for t in (rules_from, ml_from) if t is not None],
        default=None,
    )

    print(
        f"[MOM] {symbol} {kind} mode={mode} "
        f"rules_from={rules_from} ml_from={ml_from} effective_from={effective_from}"
    )

    df15 = _load_15m_candles(symbol, kind, effective_from, now)
    if df15.empty or len(df15) < 20:
        print(f"[MOM] {symbol} {kind}: no 15m candles or too few rows, skipping.")
        return {"rules_last_ts": rules_last_ts_old, "ml_last_ts": ml_last_ts_old}

    # If no from_ts given, use full history window from candles
    if effective_from is None:
        effective_from = df15.index[0]

    # Base config for writes
    base = BaseCfg(
        run_id=args.run_id or f"MOM_BACKFILL_{now.strftime('%Y%m%dT%H%M%SZ')}",
        source="mom_backfill",
        tfs=TF_LIST,
        lookback_days=args.lookback_days or 0,
    )

    new_rules_last_ts = rules_last_ts_old
    new_ml_last_ts = ml_last_ts_old

    ini_path = args.momentum_ini  # can be None → default inside _cfg()

    # Loop timeframes
    for tf in TF_LIST:
        # Build TF index (just to know which timestamps to score)
        if tf == BASE_TF:
            tf_frame = df15
        else:
            tf_frame = resample(df15, tf)

        if tf_frame is None or tf_frame.empty:
            continue

        # Score only bars >= effective_from
        for ts_tf in tf_frame.index:
            if ts_tf < effective_from:
                continue

            # Slice base 15m up to this TF timestamp
            df_slice = df15[df15.index <= ts_tf]
            if df_slice.empty:
                continue

            try:
                res = score_momentum(
                    symbol=symbol,
                    kind=kind,
                    tf=tf,
                    df5=df_slice,        # 15m base frame
                    base=base,
                    ini_path=ini_path,
                    mode=mode,
                )
            except Exception as e:
                print(f"[MOM] ERROR scoring {symbol} {kind} {tf} @ {ts_tf}: {e}", file=sys.stderr)
                continue

            if res is None:
                continue

            ts_scored, fused_score, fused_veto = res
            print(
                f"[MOM] {symbol} {kind} {tf} @ {ts_scored}: "
                f"fused={fused_score:.2f}, veto={fused_veto}"
            )

            # Track last_ts for rules / ml
            if mode in ("rules", "both"):
                if new_rules_last_ts is None or ts_scored > new_rules_last_ts:
                    new_rules_last_ts = ts_scored

            if mode in ("ml", "both"):
                if new_ml_last_ts is None or ts_scored > new_ml_last_ts:
                    new_ml_last_ts = ts_scored

    return {"rules_last_ts": new_rules_last_ts, "ml_last_ts": new_ml_last_ts}

from utils.db import get_db_connection  # already imported at top

def _sync_rules_status_from_metrics(kind: str):
    """
    Sync mom_rules_last_*_ts in reference.symbol_universe
    from indicators.metric_values (MOM.score).
    Run after a rules backfill.
    """
    if kind not in ("futures", "spot"):
        return

    col = "mom_rules_last_fut_ts" if kind == "futures" else "mom_rules_last_spot_ts"

    sql = f"""
        UPDATE reference.symbol_universe su
        SET
            {col} = mv.max_ts,
            mom_rules_last_run_at = NOW()
        FROM (
            SELECT
                symbol,
                MAX(ts) AS max_ts
            FROM indicators.metric_values
            WHERE
                kind = %s
                AND name = 'MOM.score'
            GROUP BY symbol
        ) mv
        WHERE
            su.symbol = mv.symbol
            AND su.enabled = TRUE;
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (kind,))
        conn.commit()

# ---------------------------------
# Status writer
# ---------------------------------
def _update_symbol_universe_status(
    kind: str,
    mode: str,
    updates: Dict[str, Dict[str, Optional[pd.Timestamp]]],
    run_finished_at: dt.datetime,
):
    """
    updates: { symbol: { rules_last_ts: ts, ml_last_ts: ts } }
    """
    if not updates:
        return

    with get_db_connection() as conn, conn.cursor() as cur:
        for symbol, vals in updates.items():
            rules_ts = vals.get("rules_last_ts")
            ml_ts = vals.get("ml_last_ts")

            if kind == "futures":
                if mode in ("rules", "both") and rules_ts is not None:
                    cur.execute(
                        """
                        UPDATE reference.symbol_universe
                           SET mom_rules_last_fut_ts = %s,
                               mom_rules_last_run_at = %s
                         WHERE symbol = %s
                        """,
                        (rules_ts, run_finished_at, symbol),
                    )
                if mode in ("ml", "both") and ml_ts is not None:
                    cur.execute(
                        """
                        UPDATE reference.symbol_universe
                           SET mom_ml_last_fut_ts = %s,
                               mom_ml_last_run_at = %s
                         WHERE symbol = %s
                        """,
                        (ml_ts, run_finished_at, symbol),
                    )
            else:
                if mode in ("rules", "both") and rules_ts is not None:
                    cur.execute(
                        """
                        UPDATE reference.symbol_universe
                           SET mom_rules_last_spot_ts = %s,
                               mom_rules_last_run_at = %s
                         WHERE symbol = %s
                        """,
                        (rules_ts, run_finished_at, symbol),
                    )
                if mode in ("ml", "both") and ml_ts is not None:
                    cur.execute(
                        """
                        UPDATE reference.symbol_universe
                           SET mom_ml_last_spot_ts = %s,
                               mom_ml_last_run_at = %s
                         WHERE symbol = %s
                        """,
                        (ml_ts, run_finished_at, symbol),
                    )
        conn.commit()


# ---------------------------------
# MAIN
# ---------------------------------
def main():
    args = _parse_args()
    now = dt.datetime.now(dt.timezone.utc)

    if args.symbol:
        df_univ = _load_universe(args.kind, symbol=args.symbol, max_symbols=args.max_symbols)
    else:
        df_univ = _load_universe(args.kind, symbol=None, max_symbols=args.max_symbols)

    if df_univ.empty:
        print("[MOM] No symbols in reference.symbol_universe (enabled = TRUE).")
        return

    updates: Dict[str, Dict[str, Optional[pd.Timestamp]]] = {}

    for _, row in df_univ.iterrows():
        symbol = row["symbol"]
        try:
            res = _process_symbol(
                row=row,
                kind=args.kind,
                mode=args.mode,
                args=args,
                now=now,
            )
            updates[symbol] = res
        except Exception as e:
            print(f"[MOM] ERROR processing {symbol}: {e}", file=sys.stderr)

    run_finished_at = dt.datetime.now(dt.timezone.utc)
    _update_symbol_universe_status(
        kind=args.kind,
        mode=args.mode,
        updates=updates,
        run_finished_at=run_finished_at,
    )

     # 🔁 Extra safety: sync from metric_values for rules runs
    if args.mode in ("rules", "both"):
        print(f"[MOM] Syncing mom_rules_last_*_ts from indicators.metric_values for kind={args.kind}...")
        _sync_rules_status_from_metrics(args.kind)

    print("[MOM] Backfill complete.")

if __name__ == "__main__":
    main()
