# scheduler/indicators_worker.py
"""
Indicator Calculation Worker (snapshot-first).
- Writes directly to indicators.{spot,futures}_indicators
- 15d lookback by default
- Uses classic.update_indicators_multi_tf for calculations
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Optional, Iterable, List, Tuple, Dict, TypedDict
from datetime import datetime, timezone, timedelta

import pandas as pd
import psycopg2.extras as pgx

from scheduler import (
    nonlinear_features as nlf,
    update_vp_bb as vpbb,
    track_zone_breaks as tzb,
    update_indicators_multi_tf as classic,
)
from utils.db import get_db_connection

# =========================
# Config
# =========================
TZ = timezone.utc

# Safe config loader
try:
    from utils.configs import get_config_parser  # preferred
except Exception:
    import configparser
    def get_config_parser():
        cp = configparser.ConfigParser()
        if os.path.exists("configs/data.ini"):
            cp.read("configs/data.ini")
        return cp

CFG = get_config_parser()

SOURCE         = os.getenv("IND_SOURCE", "universe").lower()  # universe | webhooks
UNIVERSE_NAME  = os.getenv("UNIVERSE_NAME", CFG.get("universe", "name", fallback="largecaps_v1"))
BASE_INTERVAL  = os.getenv("BASE_INTERVAL", CFG.get("live", "interval", fallback="15m"))  # "15m"
WORKER_RUN_ID  = os.getenv("RUN_ID", datetime.now(TZ).strftime("%Y%m%dT%H%M%SZ_ind"))
# 15 days default (fixed extra parenthesis)

IND_LOOKBACK_DAYS = int(os.getenv("IND_LOOKBACK_DAYS", "15"))

# =========================
# Small helpers
# =========================
def _floor_15m(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    m = (dt.minute // 15) * 15
    return dt.replace(minute=m, second=0, microsecond=0)

def _table_name(kind: str) -> str:
    return "market.spot_candles" if kind == "spot" else "market.futures_candles"

def _snap_table(kind: str) -> str:
    return "indicators.spot_indicators" if kind == "spot" else "indicators.futures_indicators"

def _d(x: Optional[datetime]) -> str:
    try:
        if isinstance(x, datetime):
            return x.strftime("%Y-%m-%d %H:%M")
        return str(x) if x is not None else "NA"
    except Exception:
        return "NA"

# =========================
# Typed rows
# =========================
class WorkItem(TypedDict):
    unique_id: Optional[str]
    symbol: str

# =========================
# DB helpers & status
# =========================
def _exec(sql: str, params: Iterable[Any] | None = None) -> None:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        conn.commit()

def set_status(unique_id: Optional[str], *, status: Optional[str] = None,
               sub_status: Optional[str] = None, error: Optional[str] = None) -> None:
    if SOURCE != "webhooks" or not unique_id:
        return
    sets, vals = [], []
    if status is not None:     sets.append("status=%s");     vals.append(status)
    if sub_status is not None: sets.append("sub_status=%s"); vals.append(sub_status)
    if error is not None:      sets.append("last_error=%s"); vals.append(error)
    if not sets: return
    vals.append(unique_id)
    _exec(f"""
        UPDATE webhooks.webhook_alerts
           SET {', '.join(sets)}, last_checked_at=NOW()
         WHERE unique_id=%s
    """, vals)

def set_error_with_retry(unique_id: Optional[str], sub_status: str, msg: str) -> None:
    if SOURCE != "webhooks" or not unique_id:
        return
    _exec("""
        UPDATE webhooks.webhook_alerts
           SET sub_status=%s,
               last_error=%s,
               retry_count = COALESCE(retry_count,0)+1,
               next_retry_at = NOW()+INTERVAL '10 minutes',
               last_checked_at = NOW()
         WHERE unique_id=%s
    """, (sub_status, msg[:500], unique_id))

def handoff_to_signal(unique_id: Optional[str]) -> None:
    if SOURCE != "webhooks" or not unique_id:
        return
    _exec("""
        UPDATE webhooks.webhook_alerts
           SET status='SIGNAL_PROCESS',
               sub_status='SIG_PENDING',
               last_checked_at=NOW()
         WHERE unique_id=%s
    """, (unique_id,))

# =========================
# Batch pickers
# =========================
def fetch_batch_universe(limit: int = 200) -> List[WorkItem]:
    """
    Drive from reference.symbol_universe for BASE_INTERVAL (15m by default).
    Pick symbols that have newer data than last_ind_* gates.
    """
    sql = """
        WITH s AS (
          SELECT u.symbol,
                 COALESCE((SELECT max(ts) FROM market.spot_candles    sc WHERE sc.symbol=u.symbol AND sc.interval IN ('15m',%s)), 'epoch'::timestamptz) AS newest_spot_ts,
                 COALESCE((SELECT max(ts) FROM market.futures_candles fc WHERE fc.symbol=u.symbol AND fc.interval IN ('15m',%s)), 'epoch'::timestamptz) AS newest_fut_ts
            FROM reference.symbol_universe u
           WHERE u.universe_name = %s
        )
        SELECT NULL::text AS unique_id, s.symbol
          FROM s
          JOIN reference.symbol_universe u USING(symbol)
         WHERE (u.last_ind_spot_at IS NULL OR u.last_ind_spot_at < s.newest_spot_ts)
            OR (u.last_ind_fut_at  IS NULL OR u.last_ind_fut_at  < s.newest_fut_ts)
         ORDER BY s.symbol
         LIMIT %s
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (BASE_INTERVAL, BASE_INTERVAL, UNIVERSE_NAME, limit))
        return [{"unique_id": None, "symbol": r[1]} for r in cur.fetchall()]

def fetch_batch_webhooks(limit: int = 50) -> List[WorkItem]:
    sql = """
        SELECT unique_id, symbol
          FROM webhooks.webhook_alerts
         WHERE status='INDICATOR_PROCESS'
           AND COALESCE(sub_status,'IND_PENDING')='IND_PENDING'
         ORDER BY received_at ASC
         FOR UPDATE SKIP LOCKED
         LIMIT %s
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (limit,))
        return [{"unique_id": r[0], "symbol": r[1]} for r in cur.fetchall()]

# =========================
# Data I/O
# =========================
def load_intra_from_db(symbol: str, kind: str) -> pd.DataFrame:
    """
    Return intraday OHLCV (prefer BASE_INTERVAL, fallback 5m), bounded by TF lookback.
    """
    tbl = _table_name(kind)
    intervals_to_try = [BASE_INTERVAL] + ([] if BASE_INTERVAL == "5m" else ["5m"])

    cutoff = datetime.now(timezone.utc) - timedelta(days=IND_LOOKBACK_DAYS)
    with get_db_connection() as conn, conn.cursor() as cur:
        for iv in intervals_to_try:
            cur.execute(
                f"""
                SELECT ts,
                       (open)::float8  AS open,
                       (high)::float8  AS high,
                       (low)::float8   AS low,
                       (close)::float8 AS close,
                       COALESCE(volume,0)::float8 AS volume
                  FROM {tbl}
                 WHERE symbol=%s AND interval=%s AND ts >= %s
                 ORDER BY ts ASC
                """,
                (symbol, iv, cutoff)
            )
            rows = cur.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
                df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
                for c in ("open","high","low","close","volume"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["volume"] = df["volume"].fillna(0.0)
                df = df.dropna(subset=["open","high","low","close"]).astype(
                    {"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"}
                )
                return df
    return pd.DataFrame(columns=["open","high","low","close","volume"])

def get_last_ts_from_db(symbol: str, kind: str, tf: str, metric: str) -> Optional[pd.Timestamp]:
    """
    Classic asks: "what's the latest ts we stored for this metric?"
    We answer from the snapshot table now.
    """
    tbl = _snap_table(kind)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT ts FROM {tbl}
                 WHERE symbol=%s AND interval=%s AND metric=%s
                 ORDER BY ts DESC LIMIT 1""",
            (symbol, tf, metric),
        )
        row = cur.fetchone()
    return pd.to_datetime(row[0], utc=True) if row else None
def upsert_snapshot_rows(kind: str, rows: List[tuple]) -> int:
    """
    Accept rows in classic format:
      (symbol, market_type, interval, ts, metric, val, context, run_id, source)
    Write them into indicators.{spot,futures}_indicators:
      (symbol, interval, metric, ts, val, context, run_id, source, updated_at)
    Conflict = (symbol, interval, metric) → keep the newer ts.
    """
    if not rows:
        return 0

    tgt = _snap_table(kind)

    # Map classic -> snapshot (8 values; updated_at is DEFAULT now())
    snap_rows = [
        (r[0], r[2], r[4], r[3], r[5], r[6], r[7], r[8])  # sym, interval, metric, ts, val, ctx, run_id, source
        for r in rows
    ]

    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(
            cur,
            f"""
            INSERT INTO {tgt} (symbol, interval, metric, ts, val, context, run_id, source)
            VALUES %s
            ON CONFLICT (symbol, interval, metric)
            DO UPDATE SET
              ts        = EXCLUDED.ts,
              val       = EXCLUDED.val,
              context   = EXCLUDED.context,
              run_id    = EXCLUDED.run_id,
              source    = EXCLUDED.source,
              updated_at= NOW()
            WHERE EXCLUDED.ts >= {tgt}.ts
            """,
            snap_rows,
            page_size=1000
        )
        conn.commit()
        return len(snap_rows)

# =========================
# Session VWAP for futures → write to futures snapshot
# =========================
def _today_ist_clause() -> str:
    return "(ts AT TIME ZONE 'Asia/Kolkata')::date = (NOW() AT TIME ZONE 'Asia/Kolkata')::date"

def _load_fut_intra_today(symbol: str) -> List[Dict[str, Any]]:
    with get_db_connection() as conn, conn.cursor() as cur:
        for iv in [BASE_INTERVAL] + ([] if BASE_INTERVAL == "5m" else ["5m"]):
            cur.execute(f"""
                SELECT ts, high::float8, low::float8, close::float8, COALESCE(volume,0)::float8
                  FROM market.futures_candles
                 WHERE symbol=%s AND interval=%s AND {_today_ist_clause()}
                 ORDER BY ts ASC
            """, (symbol, iv))
            rows = cur.fetchall()
            if rows:
                return [{"ts": r[0], "high": r[1], "low": r[2], "close": r[3], "volume": r[4]} for r in rows]
    return []

def _compute_session_vwap(rows: List[Dict[str, Any]]) -> List[Tuple[datetime, Optional[float]]]:
    out: List[Tuple[datetime, Optional[float]]] = []
    cum_pv = 0.0
    cum_v  = 0.0
    for r in rows:
        tp = (r["high"] + r["low"] + r["close"]) / 3.0
        v  = max(0.0, float(r["volume"]))
        cum_pv += tp * v
        cum_v  += v
        vwap = (cum_pv / cum_v) if cum_v > 0 else None
        out.append((r["ts"], vwap))
    return out

def write_futures_vwap_session(symbol: str, *, run_id: Optional[str] = None, source: str = "session_vwap") -> int:
    rows = _load_fut_intra_today(symbol)
    if not rows:
        return 0
    series = _compute_session_vwap(rows)
    run_id = run_id or datetime.now(TZ).strftime("ind_%Y%m%d")
    bulk = []
    for ts, vwap in series:
        if vwap is None:
            continue
        # classic-format row → futures snapshot writer will translate
        bulk.append((symbol, "futures", BASE_INTERVAL, ts, "VWAP", float(vwap), None, run_id, source))
    return upsert_snapshot_rows("futures", bulk)

# =========================
# Calls to classic / VP+BB / NL
# =========================
def _call_classic(symbol: str, *, kind: str) -> Tuple[int, int]:
    P = classic.load_cfg()
    out = classic.update_indicators_multi_tf(
        symbols=[symbol],
        kinds=(kind,),
        load_5m=load_intra_from_db,
        get_last_ts=get_last_ts_from_db,
        P=P
    )
    rows: List[tuple] = out.get("rows", [])
    attempted: int = out.get("attempted", len(rows))
    inserted = upsert_snapshot_rows(kind, rows)  # <— THIS one
    print(f"[WRITE] classic batch → tried {attempted} rows")
    return attempted, inserted


def _status_cb_for(uid: Optional[str]):
    def _cb(step: str, ts: Optional[Any] = None):
        if step == "ZON_OK":
            zon_ok(uid, ts)
        else:
            zon_step(uid, step)
    return _cb

def zon_step(uid: Optional[str], step: str):
    set_status(uid, sub_status=step)

def zon_ok(uid: Optional[str], last_zone_ts=None):
    set_status(uid, sub_status="ZON_OK")

def zon_error(uid: Optional[str], step: str, msg: str):
    set_error_with_retry(uid, step, msg)

def _call_vpbb(symbol: str, *, kind: str, uid: Optional[str]) -> Optional[Any]:
    if hasattr(vpbb, "run"):
        try:
            return vpbb.run(symbols=[symbol], kind=kind, uid=uid, status_cb=_status_cb_for(uid))  # type: ignore
        except TypeError:
            # fallback to older signature
            zon_step(uid, "ZON_LOADING_DATA")
            zon_step(uid, "ZON_RESAMPLING_25_65_125")
            zon_step(uid, "ZON_COMPUTING_PROFILE")
            zon_step(uid, "ZON_COMPUTING_BB")
            zon_step(uid, "ZON_WRITING")
            out = vpbb.run(symbols=[symbol])  # type: ignore
            zon_ok(uid, None)
            return out
    if hasattr(vpbb, "process_symbol"):
        zon_step(uid, "ZON_LOADING_DATA")
        zon_step(uid, "ZON_RESAMPLING_25_65_125")
        zon_step(uid, "ZON_COMPUTING_PROFILE")
        zon_step(uid, "ZON_COMPUTING_BB")
        zon_step(uid, "ZON_WRITING")
        out = vpbb.process_symbol(symbol=symbol)  # type: ignore
        zon_ok(uid, None)
        return out
    raise RuntimeError("No VP+BB entry point found in update_vp_bb.*")

def _call_nonlinear(symbol: str, kind: str, uid: Optional[str]):
    set_status(uid, sub_status=f"IND_NL_{kind}")
    nlf.process_symbol(symbol, kind=kind)  # type: ignore
    set_status(uid, sub_status=f"IND_NL_OK_{kind}")

# =========================
# Orchestrator
# =========================
def run_once(limit: int = 50, kinds: Iterable[str] = ("spot", "futures")):
    rows = fetch_batch_webhooks(limit) if SOURCE == "webhooks" else fetch_batch_universe(limit)
    print(f"\n⏱️ Indicators batch: {len(rows)} row(s) [source={SOURCE}, base={BASE_INTERVAL}, lookback={IND_LOOKBACK_DAYS}d, universe={UNIVERSE_NAME}]")

    for r in rows:
        uid: Optional[str] = r.get("unique_id")
        sym: str = r["symbol"]
        print(f"\n🧮 Indicators: {sym} ({uid or 'no-uid'})")

        for kind in kinds:
            try:
                with get_db_connection() as _conn:
                    start, target, should_run = _plan_indicator_window(_conn, sym, kind)

                if not should_run:
                    with get_db_connection() as _conn:
                        _set_universe_run_status(_conn, sym, kind, "IND_DONE")
                    set_status(uid, sub_status=f"IND_OK_{kind}")
                    print(f"[IND] {sym}:{kind} up-to-date (start={_d(start)}, target={_d(target)})")
                    continue

                with get_db_connection() as _conn:
                    _set_universe_run_status(_conn, sym, kind, "IND_RUNNING")

                set_status(uid, sub_status=f"IND_LOADING_DATA_{kind}")
                _call_classic(sym, kind=kind)

                if kind == "futures":
                    wrote_vwap = write_futures_vwap_session(sym)
                    print(f"[WRITE] futures:{sym} → VWAP({BASE_INTERVAL}) rows={wrote_vwap}")

                with get_db_connection() as _conn:
                    _advance_universe_gate(_conn, sym, kind, target, WORKER_RUN_ID)
                    _set_universe_run_status(_conn, sym, kind, "IND_DONE")
                set_status(uid, sub_status=f"IND_SNAP_OK_{kind}")

            except Exception as e:
                set_error_with_retry(uid, f"IND_ERROR_{kind}", str(e))
                print(f"[IND ERROR] {sym} [{kind}] → {e}\n{traceback.format_exc()}")

        handoff_to_signal(uid)
        if SOURCE == "webhooks":
            print(f"[OK] {sym} → SIGNAL_PROCESS/SIG_PENDING)")

# =========================
# Indicator window planning (simple)
# =========================
def _get_universe_cursor(conn, symbol: str, kind: str) -> Tuple[Optional[datetime], Optional[datetime], str]:
    col_prefix = "last_ind_spot" if kind == "spot" else "last_ind_fut"
    sql = f"""
        SELECT {col_prefix}_at, {col_prefix}_target_until_ts, COALESCE({col_prefix}_status,'NOT_STARTED')
          FROM reference.symbol_universe
         WHERE symbol = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (symbol,))
        row = cur.fetchone()
    return (row[0], row[1], row[2]) if row else (None, None, "NOT_STARTED")

def _set_universe_run_status(conn, symbol: str, kind: str, status: str,
                             gate_ts: Optional[datetime] = None,
                             last_ingested_ts: Optional[datetime] = None,
                             error: Optional[str] = None,
                             run_id: Optional[str] = None) -> None:
    status_col  = "ind_status_spot" if kind == "spot" else "ind_status_fut"
    gate_col    = "last_ind_spot_at" if kind == "spot" else "last_ind_fut_at"

    sets: List[str] = [f"{status_col}=%s", "ind_status_ts=NOW()"]
    params: List[Any] = [status]

    if gate_ts is not None:
        sets.append(f"{gate_col}=%s")
        params.append(gate_ts)
    if run_id is not None:
        sets.append("last_ind_run_id=%s")
        params.append(run_id)

    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE reference.symbol_universe SET {', '.join(sets)} WHERE symbol=%s",
            params + [symbol],
        )
    conn.commit()

def _advance_universe_gate(conn, symbol: str, kind: str, new_last_ind_at: datetime, run_id: str) -> None:
    col_name = "last_ind_spot_at" if kind == "spot" else "last_ind_fut_at"
    run_id_col = "last_ind_run_id"
    sql = f"""
        UPDATE reference.symbol_universe
           SET {col_name} = GREATEST(COALESCE({col_name}, 'epoch'::timestamptz), %s),
               {run_id_col} = %s
         WHERE symbol = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (new_last_ind_at, run_id, symbol))
    conn.commit()

def _plan_indicator_window(conn, symbol: str, kind: str) -> Tuple[datetime, datetime, bool]:
    return datetime.now(TZ) - timedelta(days=IND_LOOKBACK_DAYS), datetime.now(TZ), True

# =========================
# CLI
# =========================
def _parse_flags(argv: list[str]) -> dict:
    out = {}
    it = iter(argv)
    for tok in it:
        if tok == "--source":          out["source"] = next(it, SOURCE).lower()
        elif tok == "--universe":      out["universe"] = next(it, UNIVERSE_NAME)
        elif tok == "--base-interval": out["base_interval"] = next(it, BASE_INTERVAL)
        elif tok == "--kinds":         out["kinds"] = next(it, "spot,futures")
        elif tok == "--limit":         out["limit"] = int(next(it, "50"))
    return out

if __name__ == "__main__":
    flags = _parse_flags(sys.argv[1:])
    if "source" in flags:        SOURCE = flags["source"]
    if "universe" in flags:      UNIVERSE_NAME = flags["universe"]
    if "base_interval" in flags: BASE_INTERVAL = flags["base_interval"]
    kinds_arg = tuple(k.strip().lower() for k in flags.get("kinds", "spot,futures").split(",") if k.strip())
    limit = flags.get("limit", 50)
    run_once(limit=limit, kinds=kinds_arg)
