# scheduler/indicators_worker.py
from __future__ import annotations

import os
import traceback
from typing import Any, Optional, Iterable, List, Tuple, Dict
from datetime import datetime, timezone, timedelta

import pandas as pd
import psycopg2.extras as pgx

import scheduler.nonlinear_features as nlf
import scheduler.update_vp_bb as vpbb
import scheduler.track_zone_breaks as tzb

from utils.db import get_db_connection
from typing import Optional, TypedDict, List
from datetime import datetime, timezone
from utils.configs import load_ini

# =========================
# Globals / Config
# =========================
TZ = timezone.utc
BASE_INTERVAL = os.getenv("BASE_INTERVAL", "15m")  # preferred base (fallback to 5m)
WORKER_RUN_ID = os.getenv("RUN_ID", datetime.now(TZ).strftime("%Y%m%dT%H%M%SZ_ind"))

# ---- runtime switches ----
SOURCE = os.getenv("IND_SOURCE", "universe").lower()  # "universe" | "webhooks"
UNIVERSE_NAME = os.getenv("UNIVERSE_NAME", "largecaps_v1")

CFG = load_ini()

BASE_INTERVAL = os.getenv("BASE_INTERVAL", CFG.get("live", "interval", fallback="15m"))
IND_LOOKBACK_DAYS = int(
    os.getenv(
        "IND_LOOKBACK_DAYS",
        CFG.get("indicators", "lookback_days_5m" if BASE_INTERVAL == "5m" else "lookback_days_15m", fallback="5"),
    )
)

# allow CLI flags too
def _parse_flags(argv: list[str]) -> None:
    global SOURCE, UNIVERSE_NAME, BASE_INTERVAL
    it = iter(argv)
    for tok in it:
        if tok == "--source":
            SOURCE = next(it, SOURCE).lower()
        elif tok == "--universe":
            UNIVERSE_NAME = next(it, UNIVERSE_NAME)
        elif tok == "--base-interval":
            BASE_INTERVAL = next(it, BASE_INTERVAL)

class WorkItem(TypedDict):
    unique_id: Optional[str]
    symbol: str

# =========================
# DB helpers & status
# =========================
def _today_ist_clause() -> str:
    # IST day for WHERE; avoids tz headaches in SQL
    return "(ts AT TIME ZONE 'Asia/Kolkata')::date = (NOW() AT TIME ZONE 'Asia/Kolkata')::date"

def _exec(sql: str, params):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
    finally:
        conn.close()

def set_status(unique_id, *, status=None, sub_status=None, error=None):
    if SOURCE != "webhooks" or not unique_id:
        return
    sets, vals = [], []
    if status is not None:     sets.append("status=%s");       vals.append(status)
    if sub_status is not None: sets.append("sub_status=%s");   vals.append(sub_status)
    if error is not None:      sets.append("last_error=%s");   vals.append(error)
    if not sets: return
    vals.append(unique_id)
    _exec(f"""
        UPDATE webhooks.webhook_alerts
           SET {', '.join(sets)}, last_checked_at=NOW()
         WHERE unique_id=%s
    """, vals)

def set_error_with_retry(unique_id, sub_status: str, msg: str):
    if SOURCE != "webhooks" or not unique_id:
        return
    _exec("""
        UPDATE webhooks.webhook_alerts
           SET sub_status = %s,
               last_error = %s,
               retry_count = COALESCE(retry_count,0) + 1,
               next_retry_at = NOW() + INTERVAL '15 minutes',
               last_checked_at = NOW()
         WHERE unique_id = %s
    """, (sub_status, msg[:500], unique_id))

def handoff_to_signal(unique_id):
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
# Batch picker (gated by newest candles vs last indicator timestamps)
# =========================
def fetch_batch_webhooks(limit: int = 50) -> List[WorkItem]:
    sql = f"""
        SELECT w.unique_id, w.symbol
          FROM webhooks.webhook_alerts w
          JOIN reference.symbol_universe u USING(symbol)
          LEFT JOIN LATERAL (
              SELECT
                GREATEST(
                   COALESCE((SELECT max(ts) FROM market.spot_candles    sc WHERE sc.symbol=w.symbol AND sc.interval IN ('15m','{BASE_INTERVAL}')), 'epoch'::timestamptz),
                   COALESCE((SELECT max(ts) FROM market.futures_candles fc WHERE fc.symbol=w.symbol AND fc.interval IN ('15m','{BASE_INTERVAL}')), 'epoch'::timestamptz)
                ) AS newest_any_ts,
                COALESCE((SELECT max(ts) FROM market.spot_candles    WHERE symbol=w.symbol AND interval IN ('15m','{BASE_INTERVAL}')), 'epoch'::timestamptz) AS newest_spot_ts,
                COALESCE((SELECT max(ts) FROM market.futures_candles WHERE symbol=w.symbol AND interval IN ('15m','{BASE_INTERVAL}')), 'epoch'::timestamptz) AS newest_fut_ts
          ) s ON TRUE
         WHERE w.status='INDICATOR_PROCESS'
           AND COALESCE(w.sub_status,'IND_PENDING')='IND_PENDING'
           AND (
                u.last_ind_spot_at IS NULL OR u.last_ind_spot_at < s.newest_spot_ts
               OR
                u.last_ind_fut_at  IS NULL OR u.last_ind_fut_at  < s.newest_fut_ts
           )
         ORDER BY w.received_at ASC
         FOR UPDATE SKIP LOCKED
         LIMIT %s
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            return [{"unique_id": r[0], "symbol": r[1]} for r in cur.fetchall()]  # webhooks
    finally:
        conn.close()


def fetch_batch_universe(limit: int = 200) -> List[WorkItem]:
    """
    Drive from reference.symbol_universe for BASE_INTERVAL (15m by default).
    Pick symbols that have newer data than last_ind_* gates.
    """
    sql = f"""
        WITH s AS (
          SELECT u.symbol,
                 COALESCE((SELECT max(ts) FROM market.spot_candles    sc WHERE sc.symbol=u.symbol AND sc.interval IN ('15m','{BASE_INTERVAL}')), 'epoch'::timestamptz) AS newest_spot_ts,
                 COALESCE((SELECT max(ts) FROM market.futures_candles fc WHERE fc.symbol=u.symbol AND fc.interval IN ('15m','{BASE_INTERVAL}')), 'epoch'::timestamptz) AS newest_fut_ts
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
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (UNIVERSE_NAME, limit))
            return [{"unique_id": None, "symbol": r[1]} for r in cur.fetchall()]  # universe
    finally:
        conn.close()

# =========================
# Zone helpers
# =========================
def _coerce_ts(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        dt = v.to_pydatetime()
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, (int, float)):
        if v <= 0:
            return None
        return datetime.fromtimestamp(v, tz=timezone.utc)
    if isinstance(v, str):
        try:
            return pd.to_datetime(v, utc=True).to_pydatetime()
        except Exception:
            return None
    return None

def zon_step(uid, step: str):
    if SOURCE != "webhooks" or not uid:
        return
    _exec("""
        UPDATE webhooks.webhook_alerts
           SET sub_status=%s, last_checked_at=NOW()
         WHERE unique_id=%s
    """, (step, uid))

def zon_error(uid, step: str, msg: str):
    if SOURCE != "webhooks" or not uid:
        return
    _exec("""
        UPDATE webhooks.webhook_alerts
           SET sub_status=%s,
               last_error=%s,
               retry_count=COALESCE(retry_count,0)+1,
               next_retry_at=NOW()+INTERVAL '15 minutes',
               last_checked_at=NOW()
         WHERE unique_id=%s
    """, (step, msg[:500], uid))

def zon_ok(uid, last_zone_ts=None):
    if SOURCE != "webhooks" or not uid:
        return
    ts = _coerce_ts(last_zone_ts)
    _exec("""
        UPDATE webhooks.webhook_alerts
           SET sub_status='ZON_OK',
               last_zone_ts=%s,
               last_checked_at=NOW()
         WHERE unique_id=%s
    """, (ts, uid))

def _call_zone_breaks(symbol: str, *, kind: str, uid: str, interval: str = "60m"):
    step = f"ZONBREAK_{kind}"
    try:
        set_status(uid, sub_status=step)
        tzb.track_zone_break(symbol, interval=interval)
        set_status(uid, sub_status=f"{step}_OK")
    except Exception as e:
        set_error_with_retry(uid, f"{step}_ERROR", str(e))
        print(f"[ZONEBREAK ERROR] {symbol} [{kind}] → {e}")

def _call_nonlinear(symbol: str, kind: str, uid: Optional[str]):
    try:
        set_status(uid, sub_status=f"IND_NL_{kind}")
        nlf.process_symbol(symbol, kind=kind)
        set_status(uid, sub_status=f"IND_NL_OK_{kind}")
    except Exception as e:
        set_error_with_retry(uid, f"IND_NL_ERROR_{kind}", str(e))
        print(f"[NL ERROR] {symbol} [{kind}] → {e}")

# =========================
# I/O callbacks for classic module
# =========================
def _table_name(kind: str) -> str:
    return "market.spot_candles" if kind == "spot" else "market.futures_candles"

def load_intra_from_db(symbol: str, kind: str) -> pd.DataFrame:
    tbl = _table_name(kind)
    intervals_to_try = [BASE_INTERVAL] + ([] if BASE_INTERVAL == "15m" else ["15m"])

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
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT ts FROM indicators.values
             WHERE symbol=%s AND market_type=%s AND interval=%s AND metric=%s
             ORDER BY ts DESC LIMIT 1
        """, (symbol, kind, tf, metric))
        row = cur.fetchone()
    return pd.to_datetime(row[0], utc=True) if row else None

def upsert_indicator_rows(rows: List[tuple]) -> int:
    if not rows:
        return 0
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, """
            INSERT INTO indicators.values
                (symbol, market_type, interval, ts, metric, val, context, run_id, source)
            VALUES %s
            ON CONFLICT (symbol, market_type, interval, ts, metric) DO NOTHING
        """, rows, page_size=1000)
        conn.commit()
        return len(rows)

# =========================
# After-classic bookkeeping (3a/3b)
# =========================
def _get_latest_metric_ts(symbol: str, kind: str) -> Optional[datetime]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT max(ts) FROM indicators.values
             WHERE symbol=%s AND market_type=%s
        """, (symbol, kind))
        row = cur.fetchone()
    return (pd.to_datetime(row[0], utc=True).to_pydatetime() if row and row[0] else None)

def _update_universe_last_ind(symbol: str, kind: str, run_id: str = WORKER_RUN_ID) -> None:
    latest = _get_latest_metric_ts(symbol, kind)
    if latest is None:
        return
    with get_db_connection() as conn, conn.cursor() as cur:
        if kind == "spot":
            cur.execute("""
                UPDATE reference.symbol_universe u
                   SET last_ind_spot_at = GREATEST(COALESCE(u.last_ind_spot_at,'epoch'::timestamptz), %s),
                       last_ind_run_id   = %s
                 WHERE u.symbol = %s
            """, (latest, run_id, symbol))
        else:
            cur.execute("""
                UPDATE reference.symbol_universe u
                   SET last_ind_fut_at = GREATEST(COALESCE(u.last_ind_fut_at,'epoch'::timestamptz), %s),
                       last_ind_run_id  = %s
                 WHERE u.symbol = %s
            """, (latest, run_id, symbol))
        conn.commit()


def _upsert_latest_snapshot(symbol: str, kind: str) -> int:
    tgt = "indicators.spot_indicators" if kind == "spot" else "indicators.futures_indicators"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            WITH latest AS (
              SELECT DISTINCT ON (symbol, interval, metric)
                     symbol, interval, metric, ts, val, context, run_id, source
                FROM indicators.values
               WHERE symbol = %s AND market_type = %s
               ORDER BY symbol, interval, metric, ts DESC
            )
            INSERT INTO {tgt} (symbol, interval, metric, ts, val, context, run_id, source, updated_at)
            SELECT symbol, interval, metric, ts, val, context, run_id, source, now()
              FROM latest
            ON CONFLICT (symbol, interval, metric) DO UPDATE
              SET ts         = EXCLUDED.ts,
                  val        = EXCLUDED.val,
                  context    = EXCLUDED.context,
                  run_id     = EXCLUDED.run_id,
                  source     = EXCLUDED.source,
                  updated_at = now()
        """, (symbol, kind))
        conn.commit()
        return cur.rowcount

# =========================
# Futures session VWAP (base interval)
# =========================
def _load_fut_intra_today(symbol: str) -> List[Dict[str, Any]]:
    with get_db_connection() as conn, conn.cursor() as cur:
        for iv in [BASE_INTERVAL] + ([] if BASE_INTERVAL == "15m" else ["15m"]):
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
        bulk.append((symbol, "futures", BASE_INTERVAL, ts, "VWAP", float(vwap), None, run_id, source))
    return upsert_indicator_rows(bulk)

# =========================
# Import computation modules (classic + VP/BB)
# =========================
from scheduler import update_indicators_multi_tf as classic

def _status_cb_for(uid: Any):
    def _cb(step: str, ts: Optional[Any] = None):
        if step == "ZON_OK":
            zon_ok(uid, ts)
        else:
            zon_step(uid, step)
    return _cb

def _call_vpbb(symbol: str, *, kind: str, uid: Any) -> Optional[Any]:
    if hasattr(vpbb, "run"):
        try:
            return vpbb.run(symbols=[symbol], kind=kind, uid=uid, status_cb=_status_cb_for(uid))  # type: ignore
        except TypeError:
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

def _coerce_last_ts(ret: Any) -> Optional[Any]:
    if ret is None:
        return None
    if isinstance(ret, dict) and "last_ts" in ret:
        return ret["last_ts"]
    if isinstance(ret, (list, tuple)) and len(ret) > 0:
        return ret[-1]
    return ret

def _call_classic(symbol: str, *, kind: str) -> Tuple[int, int]:
    """
    Build indicator rows via the DB-free classic module and write them.
    Returns (attempted, inserted).
    """
    P = classic.load_cfg()
    out = classic.update_indicators_multi_tf(
        symbols=[symbol],
        kinds=(kind,),
        load_5m=load_intra_from_db,      # accepts BASE_INTERVAL or 5m, classic now handles resampling
        get_last_ts=get_last_ts_from_db,
        P=P
    )
    rows: List[tuple] = out.get("rows", [])
    attempted: int = out.get("attempted", 0)
    inserted = upsert_indicator_rows(rows)
    print(f"[WRITE] {kind}:{symbol} → tried {attempted} / inserted {inserted}")
    return attempted, inserted

# =========================
# Orchestrator
# =========================
def run_once(limit: int = 50, kinds: Iterable[str] = ("spot", "futures")):
    rows = fetch_batch_webhooks(limit) if SOURCE == "webhooks" else fetch_batch_universe(limit)
    print(f"\n⏱️ Indicators batch: {len(rows)} row(s) [source={SOURCE}, base={BASE_INTERVAL}, universe={UNIVERSE_NAME}]")


    for r in rows:
        uid, sym = r["unique_id"], r["symbol"]
        print(f"\n🧮 Indicators: {sym} ({uid})")

        for kind in kinds:
            # ----- classic indicators -----
            try:
                set_status(uid, sub_status=f"IND_LOADING_DATA_{kind}")
                _call_classic(sym, kind=kind)

                # Ensure session VWAP exists for futures (stored at BASE_INTERVAL)
                if kind == "futures":
                    wrote_vwap = write_futures_vwap_session(sym)
                    print(f"[WRITE] futures:{sym} → VWAP({BASE_INTERVAL}) upserted rows: {wrote_vwap}")

                # --- 3a + 3b: update universe gate + latest snapshot tables
                try:
                    _update_universe_last_ind(sym, kind, WORKER_RUN_ID)   # 3a
                    _ = _upsert_latest_snapshot(sym, kind)                # 3b
                    set_status(uid, sub_status=f"IND_SNAP_OK_{kind}")
                except Exception as e2:
                    set_error_with_retry(uid, f"IND_SNAP_ERROR_{kind}", str(e2))
                    print(f"[SNAP ERROR] {sym} [{kind}] → {e2}")

                set_status(uid, sub_status=f"IND_OK_{kind}")
            except Exception as e:
                set_error_with_retry(uid, f"IND_ERROR_{kind}", str(e))
                print(f"[IND ERROR] {sym} [{kind}] → {e}\n{traceback.format_exc()}")
                continue

            # ----- zones (VP+BB) -----
            try:
                ret = _call_vpbb(sym, kind=kind, uid=uid)
                last_ts = _coerce_last_ts(ret)
                if isinstance(last_ts, (int, float)) and last_ts <= 0:
                    last_ts = None
                zon_ok(uid, last_ts)

                # Optional: record that zones were refreshed for this symbol
                _exec("""
                    UPDATE reference.symbol_universe
                       SET last_zone_at = GREATEST(COALESCE(last_zone_at,'epoch'::timestamptz), NOW())
                     WHERE symbol=%s
                """, (sym,))
            except Exception as e:
                zon_error(uid, f"ZON_ERROR_{kind}", str(e))
                print(f"[ZON ERROR] {sym} [{kind}] → {e}\n{traceback.format_exc()}")

            # ----- Nonlinear features -----
            try:
                _call_nonlinear(sym, kind=kind, uid=uid)
            except Exception as e:
                zon_error(uid, f"Non_Linear_{kind}", str(e))
                print(f"[Non_Linear] {sym} [{kind}] → {e}\n{traceback.format_exc()}")

        # ----- handoff after both kinds complete -----
        handoff_to_signal(uid)
        print(f"[OK] {sym} → SIGNAL_PROCESS/SIG_PENDING")

if __name__ == "__main__":
    import sys
    _parse_flags(sys.argv[1:])
    run_once()
