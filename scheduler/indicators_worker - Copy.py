# scheduler/indicators_worker.py
from __future__ import annotations

import traceback
from typing import Any, Optional, Iterable, List, Tuple, Dict
from datetime import datetime, timezone
import pandas as pd
import psycopg2.extras as pgx
import scheduler.nonlinear_features as nlf
import scheduler.update_vp_bb as vpbb
import scheduler.track_zone_breaks as tzb   # ← add this


from utils.db import get_db_connection

# =========================
# DB + status helpers
# =========================
TZ = timezone.utc

def _today_ist_clause():
    # IST day for WHERE; avoids tz headaches in SQL
    return "(ts AT TIME ZONE 'Asia/Kolkata')::date = (NOW() AT TIME ZONE 'Asia/Kolkata')::date"

def _exec(sql: str, params):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    conn.close()

def fetch_batch(limit: int = 50):
    """
    Claim INDICATOR_PROCESS + IND_PENDING rows for processing (SKIP LOCKED).
    """
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT unique_id, symbol
          FROM webhooks.webhook_alerts
         WHERE status='INDICATOR_PROCESS'
           AND COALESCE(sub_status,'IND_PENDING')='IND_PENDING'
         ORDER BY received_at ASC
         FOR UPDATE SKIP LOCKED
         LIMIT %s
    """, (limit,))
    rows = [{"unique_id": r[0], "symbol": r[1]} for r in cur.fetchall()]
    conn.commit(); conn.close()
    return rows

def _load_fut_5m_today(symbol: str):
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT ts, high::float8, low::float8, close::float8, COALESCE(volume,0)::float8
            FROM market.futures_candles
            WHERE symbol=%s AND interval='5m' AND {_today_ist_clause()}
            ORDER BY ts ASC
        """, (symbol,))
        rows = cur.fetchall()
    return [{"ts": r[0], "high": r[1], "low": r[2], "close": r[3], "volume": r[4]} for r in rows]

def _compute_session_vwap_5m(rows):
    out = []
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

def write_futures_vwap_5m(symbol: str, *, run_id: str = None, source: str = "session_vwap") -> int:
    rows = _load_fut_5m_today(symbol)
    if not rows:
        return 0
    series = _compute_session_vwap_5m(rows)

    # Build bulk upsert tuples to match upsert_indicator_rows() schema:
    # (symbol, market_type, interval, ts, metric, val, context, run_id, source)
    run_id = run_id or datetime.now(TZ).strftime("ind_%Y%m%d")
    bulk = []
    for ts, vwap in series:
        if vwap is None:
            continue
        bulk.append((symbol, "futures", "5m", ts, "VWAP", float(vwap), None, run_id, source))
    return upsert_indicator_rows(bulk)

def set_status(unique_id, *, status=None, sub_status=None, error=None):
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
    _exec("""
        UPDATE webhooks.webhook_alerts
           SET sub_status = %s,
               last_error = %s,
               retry_count = COALESCE(retry_count,0) + 1,
               next_retry_at = NOW() + INTERVAL '5 minutes',
               last_checked_at = NOW()
         WHERE unique_id = %s
    """, (sub_status, msg[:500], unique_id))

def handoff_to_signal(unique_id):
    _exec("""
        UPDATE webhooks.webhook_alerts
           SET status='SIGNAL_PROCESS',
               sub_status='SIG_PENDING',
               last_checked_at=NOW()
         WHERE unique_id=%s
    """, (unique_id,))

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
    _exec("""
        UPDATE webhooks.webhook_alerts
           SET sub_status=%s, last_checked_at=NOW()
         WHERE unique_id=%s
    """, (step, uid))

def zon_error(uid, step: str, msg: str):
    _exec("""
        UPDATE webhooks.webhook_alerts
           SET sub_status=%s,
               last_error=%s,
               retry_count=COALESCE(retry_count,0)+1,
               next_retry_at=NOW()+INTERVAL '5 minutes',
               last_checked_at=NOW()
         WHERE unique_id=%s
    """, (step, msg[:500], uid))

def zon_ok(uid, last_zone_ts=None):
    ts = _coerce_ts(last_zone_ts)
    _exec("""
        UPDATE webhooks.webhook_alerts
           SET sub_status='ZON_OK',
               last_zone_ts=%s,
               last_checked_at=NOW()
         WHERE unique_id=%s
    """, (ts, uid))

def _call_nonlinear(symbol: str, kind: str, uid: str):
    try:
        set_status(uid, sub_status=f"IND_NL_{kind}")
        nlf.process_symbol(symbol, kind=kind)
        set_status(uid, sub_status=f"IND_NL_OK_{kind}")
    except Exception as e:
        set_error_with_retry(uid, f"IND_NL_ERROR_{kind}", str(e))
        print(f"[NL ERROR] {symbol} [{kind}] → {e}")

def _call_zone_breaks(symbol: str, *, kind: str, uid: str, interval: str = "60m"):
    """
    Update support/resistance break flags using latest candle + zone_levels.
    Writes:
      - market.zone_levels.support_break_flag / resistance_break_flag
      - market.zone_breakouts (rows)
    """
    try:
        step = f"ZONBREAK_{kind}"
        set_status(uid, sub_status=step)
        tzb.track_zone_break(symbol, interval=interval)
        set_status(uid, sub_status=f"{step}_OK")
    except Exception as e:
        set_error_with_retry(uid, f"{step}_ERROR", str(e))
        print(f"[ZONEBREAK ERROR] {symbol} [{kind}] → {e}")

# =========================
# I/O callbacks for the pure classic module
# =========================

def _table_name(kind: str) -> str:
    return "market.spot_candles" if kind == "spot" else "market.futures_candles"

def load_5m_from_db(symbol: str, kind: str) -> pd.DataFrame:
    """
    Return a 5m OHLCV dataframe indexed by UTC ts for (symbol, kind).
    """
    tbl = _table_name(kind)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ts,
                   (open)::float8  AS open,
                   (high)::float8  AS high,
                   (low)::float8   AS low,
                   (close)::float8 AS close,
                   COALESCE(volume,0)::float8 AS volume
              FROM {tbl}
             WHERE symbol=%s AND interval='5m'
             ORDER BY ts ASC
            """,
            (symbol,)
        )
        rows = cur.fetchall()

    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

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
# Import computation modules
# =========================
from scheduler import update_indicators_multi_tf as classic
import scheduler.update_vp_bb as vpbb

# -------- classic indicators (RSI/RMI/MFI/MACD/ADX/ROC/ATR/EMA) --------
def _call_classic(symbol: str, *, kind: str) -> Tuple[int, int]:
    """
    Build indicator rows via the DB-free classic module and write them.
    Returns (attempted, inserted).
    """
    P = classic.load_cfg()

    out = classic.update_indicators_multi_tf(
        symbols=[symbol],
        kinds=(kind,),
        load_5m=load_5m_from_db,
        get_last_ts=get_last_ts_from_db,
        P=P
    )
    rows: List[tuple] = out.get("rows", [])
    attempted: int = out.get("attempted", 0)

    inserted = upsert_indicator_rows(rows)
    print(f"[WRITE] {kind}:{symbol} → tried {attempted} / inserted {inserted}")
    return attempted, inserted

# -------- VP + BB hybrid --------
def _status_cb_for(uid: Any):
    def _cb(step: str, ts: Optional[Any] = None):
        if step == "ZON_OK":
            zon_ok(uid, ts)
        else:
            zon_step(uid, step)
    return _cb

def _call_vpbb(symbol: str, *, kind: str, uid: Any) -> Optional[Any]:
    """
    Call update_vp_bb.run with callback. If older signature, bracket with steps.
    Returns whatever vpbb.run returns (we try to coerce last_ts later).
    """
    if hasattr(vpbb, "run"):
        try:
            return vpbb.run(symbols=[symbol], kind=kind, uid=uid, status_cb=_status_cb_for(uid))  # type: ignore
        except TypeError:
            # older signature that doesn't accept kind/callback
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

# =========================
# Orchestrator
# =========================
def run_once(limit: int = 50, kinds: Iterable[str] = ("spot", "futures")):
    rows = fetch_batch(limit)
    print(f"\n⏱️ Indicators batch: {len(rows)} row(s)")

    for r in rows:
        uid, sym = r["unique_id"], r["symbol"]
        print(f"\n🧮 Indicators: {sym} ({uid})")

        for kind in kinds:
            # ----- classic indicators -----
            try:
                set_status(uid, sub_status=f"IND_LOADING_DATA_{kind}")
                _call_classic(sym, kind=kind)

                # NEW: ensure VWAP(5m) exists for futures (session VWAP)
                if kind == "futures":
                    wrote_vwap = write_futures_vwap_5m(sym)
                    print(f"[WRITE] futures:{sym} → VWAP(5m) upserted rows: {wrote_vwap}")

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
    run_once()
