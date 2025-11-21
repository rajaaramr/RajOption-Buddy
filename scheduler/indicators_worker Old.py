# scheduler/indicators_worker.py
"""
Indicator Calculation Worker.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Optional, Iterable, List, Tuple, Dict, TypedDict
from datetime import datetime, timezone, timedelta
import json, re

import os, sys, traceback

import pandas as pd
import psycopg2.extras as pgx
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
# Globals / Config
# =========================
TZ = timezone.utc
# --- TOP OF FILE: replace your configs import with this safe block ---
try:
    from utils.configs import get_config_parser  # preferred
except Exception:
    # Fallback: simple INI loader (configs/data.ini)
    import configparser, os
    def get_config_parser():
        cp = configparser.ConfigParser()
        if os.path.exists("configs/data.ini"):
            cp.read("configs/data.ini")
        return cp
# --------------------------------------------------------------------

_MFI_RE = re.compile(r'^(?:MFI)(?:[._]?(\d+))?$', re.I)
_ATR_RE = re.compile(r'^(?:ATR)(?:[._]?(\d+))?$', re.I)
_MACD_RE = re.compile(r'^(?:MACD)(?:[._]?((?:macd|signal|hist|line)))?$', re.I)
_EMA_RE = re.compile(r'^(?:EMA)[._]?(\d+)$', re.I)
CFG = get_config_parser()

SOURCE = os.getenv("IND_SOURCE", "universe").lower()  # universe | webhooks
UNIVERSE_NAME = os.getenv("UNIVERSE_NAME", CFG.get("universe", "name", fallback="largecaps_v1"))
BASE_INTERVAL = os.getenv("BASE_INTERVAL", CFG.get("live", "interval", fallback="15m"))  # "15m" default
WORKER_RUN_ID = os.getenv("RUN_ID", datetime.now(TZ).strftime("%Y%m%dT%H%M%SZ_ind"))

IND_LOOKBACK_DAYS = int(os.getenv("IND_LOOKBACK_DAYS", "365"))

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

def _safe_float_or_none(x):
    """Safe float converter that tolerates pd.NA, NaN, None."""
    import math
    import pandas as pd

    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v

# --- frames helpers ---
ALLOWED_INTERVALS = {"15m", "30m", "60m", "90m", "120m", "240m"}

def _frames_table(kind: str) -> str:
    return "indicators.spot_frames" if kind == "spot" else "indicators.futures_frames"

def _map_metric_to_col(name: str, ctx_json: Optional[str]) -> Optional[str]:
    """
    Map metric names from classic/nl/vpbb into frames columns.
    Accepts variants like MFI, MFI.14, MFI14; ATR, ATR_14, ATR14; MACD.signal/hist/line; EMA.20, etc.
    """
    if not name:
        return None
    n = name.strip()
    up = n.upper()

    # --- EMA.N -> ema_N
    m = _EMA_RE.match(n)
    if m:
        return f"ema_{int(m.group(1))}"

    # --- MACD family
    m = _MACD_RE.match(n)
    if m:
        part = (m.group(1) or "").lower()
        if part in ("", "macd", "line"):
            return "macd"          # MACD or MACD.line
        if part in ("signal", "sig"):
            return "macd_signal"   # MACD.signal, MACD.SIG
        if part in ("hist", "diff"):
            return "macd_hist"     # MACD.hist or MACD.diff
        return None

    # --- Simple one-to-one / aliases
    if up in ("RSI", "RSI14", "RSI_14"):
        return "rsi"
    if up in ("ROC", "ROC14", "ROC_14"):
        return "roc"
    if up in ("ADX", "ADX14", "ADX_14"):
        return "adx"

    if up in ("DI+", "+DI", "PLUS_DI"):
        return "plus_di"
    if up in ("DI-", "-DI", "MINUS_DI"):
        return "minus_di"

    if up.startswith("CCI"):
        return "cci"

    if up in ("STOCH.K", "STOCH_K", "STOCH%K"):
        return "stoch_k"
    if up in ("STOCH.D", "STOCH_D", "STOCH%D"):
        return "stoch_d"

    if up == "BB.WIDTH":
        return "bb_width"
    if up == "BB.BANDTOP":
        return "bb_bandtop"
    if up == "BB.BANDBOT":
        return "bb_bandbot"
    if up == "PSAR":
        return "psar"

    # --- RMI
    if up.startswith("RMI"):
        return "rmi"

    # --- OBV family
    if up == "OBV":                     return "obv"
    if up in ("OBV.EMA","OBV_EMA"):     return "obv_ema"
    if up in ("OBV.DELTA","OBV_DELTA"): return "obv_delta"
    if up in ("OBV.ZS","OBV_ZS"):       return "obv_zs"
    if up in ("OBV.ZL","OBV_ZL"):       return "obv_zl"
    if up in ("OBV.ZH","OBV_ZH"):       return "obv_zh"
    if up in ("OBV.SIG","OBV_SIG"):     return "obv_sig"
    if up in ("OBV.HIST","OBV_HIST"):   return "obv_hist"
    if up in ("OBV.SIG.EMA","OBV_SIG_EMA"):
        return "obv_sig_ema"
    if up in ("OBV.HIST.EMA","OBV_HIST_EMA"):
        return "obv_hist_ema"

    # --- ATR length (default 14 if not present)
    m = _ATR_RE.match(n)
    if m:
        try:
            if m.group(1):
                L = int(m.group(1))
            else:
                L = int(json.loads(ctx_json or "{}").get("length", 14))
        except Exception:
            L = 14
        return f"atr_{L}"

    # --- MFI length (default 14 if not present)
    m = _MFI_RE.match(n)
    if m:
        try:
            if m.group(1):
                L = int(m.group(1))
            else:
                L = int(json.loads(ctx_json or "{}").get("length", 14))
        except Exception:
            L = 14
        return f"mfi_{L}"

        # --- VWAP family (rolling / cumulative / session)
    up = n.upper()
    if up in ("VWAP.ROLL20", "VWAP.ROLL_20", "VWAP.ROLLING20", "VWAP.ROLLING_20"):
        return "vwap_rolling_20"
    if up in ("VWAP.CUM", "VWAP.CUMULATIVE"):
        return "vwap_cumulative"
    if up in ("VWAP.SESSION", "VWAP_SESSION"):
        return "vwap_session"

    # --- VP/BB metrics (from vpbb writer)
    if n == "VP.VAL":              return "vp_val"
    if n == "VP.VAH":              return "vp_vah"
    if n == "VP.POC":              return "vp_poc"
    if n == "BB.zone_top":         return "bb_zone_top"
    if n == "BB.zone_bot":         return "bb_zone_bot"
    if n == "BB.score":            return "bb_score"
    if n == "BB.width_pct":        return "bb_width_pct"
    if n == "TZB.breaks":          return "tzb_breaks"
    if n == "TZB.score":           return "tzb_score"
    if n == "TZB.last_break_ts":   return "tzb_last_break_ts"

    # --- Nonlinear (if you route them here later)
    if n == "CONF_NL.prob.mtf":    return "nl_prob"
    if n == "CONF_NL.score.mtf":   return "nl_score"
    if n.startswith("NL.RSIxADX"): return "nli_rsixadx"
    if n.startswith("NL.MACDxATR"):return "nli_macdxatr"
    if n.startswith("NL.MFIxROC"): return "nli_mfixroc"

    return None


def _bulk_upsert_frames_from_metric_rows(kind: str, metric_rows: List[tuple]) -> int:
    """
    Adapt classic output rows into wide-frame UPSERTs.

    Input rows are tuples:
      (symbol, market_type, interval, ts, metric, val, context, run_id, source)

    We pivot to one row per (symbol, interval, ts) with many columns, then
    bulk UPSERT into indicators.{spot,futures}_frames.
    """
    if not metric_rows:
        return 0

    # Filter TFs and normalize
    metric_rows = [r for r in metric_rows if r[2] in ALLOWED_INTERVALS]
    if not metric_rows:
        return 0

    # Group by (sym, tf, ts)
    from collections import defaultdict
    grouped: Dict[tuple, Dict[str, Any]] = defaultdict(dict)
    meta: Dict[tuple, Dict[str, Any]] = {}

    for sym, _kind, tf, ts, metric, val, ctx, run_id, src in metric_rows:
        key = (sym, tf, pd.to_datetime(ts, utc=True).to_pydatetime())
        col = _map_metric_to_col(metric, ctx)
        if not col:
            continue
        if col not in grouped[key]:
            grouped[key][col] = float(val) if val is not None else None
        meta[key] = {"run_id": run_id, "source": src}

    if not grouped:
        # Debug: what metric names did we skip?
        unk = sorted(set(str(r[4]) for r in metric_rows if _map_metric_to_col(r[4], r[6]) is None))
        if unk:
            print(f"[frames] skipped metrics (no column mapping): {', '.join(unk[:20])}" +
                  ("" if len(unk) <= 20 else f" (+{len(unk)-20} more)"))
        return 0

    # Union of all dynamic columns for one bulk INSERT
    all_cols = sorted({c for row in grouped.values() for c in row.keys()})
    table = _frames_table(kind)

    base_cols = ["symbol", "interval", "ts"]
    audit_cols = ["run_id", "source"]
    cols = base_cols + all_cols + audit_cols

    # Build rows (fill missing with None)
    values: List[tuple] = []
    for key, cols_dict in grouped.items():
        sym, tf, ts = key
        row = [sym, tf, ts] + [cols_dict.get(c) for c in all_cols] + [
            meta[key].get("run_id"),
            meta[key].get("source"),
        ]
        values.append(tuple(row))

    if not values:
        return 0

    # Build ON CONFLICT SET list for all dynamic + audit cols
    set_list = ", ".join([f"{c}=EXCLUDED.{c}" for c in (all_cols + audit_cols)] + ["updated_at=NOW()"])

    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(
            cur,
            f"""
            INSERT INTO {table} ({", ".join(cols)})
            VALUES %s
            ON CONFLICT (symbol, interval, ts)
            DO UPDATE SET {set_list}
            """,
            values,
            page_size=1000
        )
        conn.commit()
    return len(values)


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

# =========================
# Batch pickers
# =========================
def fetch_batch_universe(limit: int = 200) -> List[WorkItem]:
    """
    Drive from reference.symbol_universe for BASE_INTERVAL (15m by default).
    Pick symbols that have newer data than last_ind_* gates.
    NOTE: No universe_name filter (column not present in this schema).
    """
    sql = """
    WITH s AS (
      SELECT u.symbol,
             COALESCE((SELECT max(ts) FROM market.spot_candles    sc WHERE sc.symbol=u.symbol AND sc.interval IN ('15m',%s)), 'epoch'::timestamptz) AS newest_spot_ts,
             COALESCE((SELECT max(ts) FROM market.futures_candles fc WHERE fc.symbol=u.symbol AND fc.interval IN ('15m',%s)), 'epoch'::timestamptz) AS newest_fut_ts
        FROM reference.symbol_universe u
    )
    SELECT NULL::text AS unique_id, s.symbol
      FROM s
      JOIN reference.symbol_universe u USING(symbol)
        WHERE (u.ind_last_spot_run_at IS NULL OR u.ind_last_spot_run_at < s.newest_spot_ts)
         OR (u.ind_last_fut_run_at  IS NULL OR u.ind_last_fut_run_at  < s.newest_fut_ts)
        ORDER BY s.symbol
        LIMIT %s
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        # note: only 3 params now
        cur.execute(sql, (BASE_INTERVAL, BASE_INTERVAL, limit))
        return [{"unique_id": None, "symbol": r[1]} for r in cur.fetchall()]


# =========================
# Data I/O for classic module
# =========================
def load_intra_from_db(symbol: str, kind: str) -> pd.DataFrame:
    """
    Strictly load BASE_INTERVAL (no 5m fallback).
    """
    tbl = _table_name(kind)
    cutoff = datetime.now(timezone.utc) - timedelta(days=IND_LOOKBACK_DAYS)

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
             WHERE symbol=%s
               AND interval=%s
               AND ts >= %s
             ORDER BY ts ASC
            """,
            (symbol, BASE_INTERVAL, cutoff),
        )
        rows = cur.fetchall()

    if not rows:
        # Useful debug line—delete if noisy
        print(f"[WARN] no {BASE_INTERVAL} data for {kind}:{symbol} since {cutoff.isoformat()}")
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)
    return df.dropna(subset=["open","high","low","close"]).astype(
        {"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"}
    )

def get_last_ts_from_db(symbol: str, kind: str, tf: str, metric: str) -> Optional[pd.Timestamp]:
    """
    For de-dup when calling classic builder: just return the latest ts we
    have in the frames table for this (symbol, tf). We don't track per-metric
    ts anymore (wide-row schema).
    """
    tbl = _frames_table(kind)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT max(ts) FROM {tbl}
             WHERE symbol=%s AND interval=%s
            """,
            (symbol, tf),
        )
        row = cur.fetchone()
    return pd.to_datetime(row[0], utc=True) if row and row[0] else None

def _get_latest_metric_ts(symbol: str, kind: str) -> Optional[datetime]:
    tbl = _frames_table(kind)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT max(ts) FROM {tbl} WHERE symbol=%s", (symbol,))
        row = cur.fetchone()
    return (pd.to_datetime(row[0], utc=True).to_pydatetime() if row and row[0] else None)

# =========================
# Session VWAP for futures → write to futures snapshot
# =========================

def _tp(df: pd.DataFrame) -> pd.Series:
    # typical price
    return (df["high"] + df["low"] + df["close"]) / 3.0

def _load_intra_df(symbol: str, kind: str) -> pd.DataFrame:
    # reuse your existing DB loader + ensure it’s 15m base
    df = load_intra_from_db(symbol, kind)
    return df.sort_index() if df is not None else pd.DataFrame()

def write_vwap_roll_20(symbol: str, *, kind: str, run_id: Optional[str] = None, source: str = "vwap_roll_20") -> int:
    """
    Rolling VWAP over last 20 bars (5h on 15m). Writes to vwap_roll_20 in frames.
    """
    df = _load_intra_df(symbol, kind)
    if df.empty:
        return 0

    tp = _tp(df)
    pv = tp * df["volume"].clip(lower=0.0)
    num = pv.rolling(20, min_periods=5).sum()
    den = df["volume"].rolling(20, min_periods=5).sum().replace(0, pd.NA)
    roll = (num / den).dropna()

    if roll.empty:
        return 0

    table = _frames_table(kind)
    run_id = run_id or datetime.now(TZ).strftime("ind_%Y%m%d")
    payload = [(symbol, BASE_INTERVAL, ts.to_pydatetime(), float(val), run_id, source)
               for ts, val in roll.items()]

    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(
            cur,
            f"""
            INSERT INTO {table} (symbol, interval, ts, vwap_rolling_20, run_id, source)
            VALUES %s
            ON CONFLICT (symbol, interval, ts)
            DO UPDATE SET
              vwap_rolling_20 = EXCLUDED.vwap_rolling_20,
              run_id       = EXCLUDED.run_id,
              source       = EXCLUDED.source,
              updated_at   = NOW()
            """,
            payload, page_size=1000
        )
        conn.commit()
    return len(payload)

def write_vwap_cumulative(symbol: str, *, kind: str, run_id: Optional[str] = None, source: str = "vwap_cumulative") -> int:
    """
    Cumulative VWAP from the start of your lookback window. Writes to vwap_cum in frames.
    """
    df = _load_intra_df(symbol, kind)
    if df.empty:
        return 0

    tp = _tp(df)
    pv = (tp * df["volume"].clip(lower=0.0)).cumsum()
    v  = df["volume"].clip(lower=0.0).cumsum().replace(0, pd.NA)
    cum = (pv / v).dropna()

    if cum.empty:
        return 0

    table = _frames_table(kind)
    run_id = run_id or datetime.now(TZ).strftime("ind_%Y%m%d")
    payload = [(symbol, BASE_INTERVAL, ts.to_pydatetime(), float(val), run_id, source)
               for ts, val in cum.items()]

    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(
            cur,
            f"""
            INSERT INTO {table} (symbol, interval, ts, vwap_cumulative, run_id, source)
            VALUES %s
            ON CONFLICT (symbol, interval, ts)
            DO UPDATE SET
              vwap_cumulative    = EXCLUDED.vwap_cumulative,
              run_id      = EXCLUDED.run_id,
              source      = EXCLUDED.source,
              updated_at  = NOW()
            """,
            payload, page_size=1000
        )
        conn.commit()
    return len(payload)

def upsert_indicator_rows(rows: List[tuple]) -> int:
    """
    rows: (symbol, market_type, interval, ts, metric, val, context, run_id, source)
    """
    if not rows:
        return 0

    # ✅ Fix C: only allow canonical TFs
    ALLOWED_INTERVALS = {"15m", "30m", "60m", "90m", "120m"}
    rows = [r for r in rows if r[2] in ALLOWED_INTERVALS]
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
# Latest snapshot & gates
# =========================

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


# =========================
# Session VWAP for futures
# =========================
def _today_ist_clause() -> str:
    return "(ts AT TIME ZONE 'Asia/Kolkata')::date = (NOW() AT TIME ZONE 'Asia/Kolkata')::date"

def _load_spot_intra_today(symbol: str) -> List[Dict[str, Any]]:
    with get_db_connection() as conn, conn.cursor() as cur:
        for iv in [BASE_INTERVAL] + ([] if BASE_INTERVAL == "5m" else ["5m"]):
            cur.execute(f"""
                SELECT ts, high::float8, low::float8, close::float8, COALESCE(volume,0)::float8
                  FROM market.spot_candles
                 WHERE symbol=%s AND interval=%s
                   AND (ts AT TIME ZONE 'Asia/Kolkata')::date = (NOW() AT TIME ZONE 'Asia/Kolkata')::date
                 ORDER BY ts ASC
            """, (symbol, iv))
            rows = cur.fetchall()
            if rows:
                return [{"ts": r[0], "high": r[1], "low": r[2], "close": r[3], "volume": r[4]} for r in rows]
    return []

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
    """
    Compute intraday session VWAP (IST day) from futures candles and upsert into
    indicators.futures_frames as column 'vwap_session' per (symbol, interval=BASE_INTERVAL, ts).
    """
    rows = _load_fut_intra_today(symbol)
    if not rows:
        return 0

    series = _compute_session_vwap(rows)
    run_id = run_id or datetime.now(TZ).strftime("ind_%Y%m%d")
    table = _frames_table("futures")  # indicators.futures_frames

    # Build bulk rows for UPSERT
    payload = []
    for ts, vwap in series:
        if vwap is None:
            continue
        payload.append((
            symbol, BASE_INTERVAL, pd.to_datetime(ts, utc=True).to_pydatetime(),
            float(vwap), run_id, source
        ))
    if not payload:
        return 0

    # Upsert on (symbol, interval, ts): set vwap_session, run_id, source, updated_at
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(
            cur,
            f"""
            INSERT INTO {table} (symbol, interval, ts, vwap_session, run_id, source)
            VALUES %s
            ON CONFLICT (symbol, interval, ts)
            DO UPDATE SET
              vwap_session = EXCLUDED.vwap_session,
              run_id       = EXCLUDED.run_id,
              source       = EXCLUDED.source,
              updated_at   = NOW()
            """,
            payload,
            page_size=1000
        )
        conn.commit()
    return len(payload)

def write_spot_vwap_session(symbol: str, *, run_id: Optional[str] = None, source: str = "session_vwap") -> int:
    rows = _load_spot_intra_today(symbol)
    if not rows:
        return 0
    # same math as futures
    series = _compute_session_vwap(rows)
    run_id = run_id or datetime.now(TZ).strftime("ind_%Y%m%d")
    payload = []
    for ts, vwap in series:
        if vwap is None:
            continue
        payload.append((symbol, BASE_INTERVAL, pd.to_datetime(ts, utc=True).to_pydatetime(),
                        float(vwap), run_id, source))
    if not payload:
        return 0

    table = "indicators.spot_frames"
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(
            cur,
            f"""
            INSERT INTO {table} (symbol, interval, ts, vwap_session, run_id, source)
            VALUES %s
            ON CONFLICT (symbol, interval, ts)
            DO UPDATE SET
              vwap_session = EXCLUDED.vwap_session,
              run_id       = EXCLUDED.run_id,
              source       = EXCLUDED.source,
              updated_at   = NOW()
            """,
            payload,
            page_size=1000
        )
        conn.commit()
    return len(payload)

def _load_recent_15m(symbol: str, kind: str, *, bars: int = 160) -> pd.DataFrame:
    """Load the last ~N 15m bars for spot/futures (enough for 20-bar rolling)."""
    tbl = _table_name(kind)
    with get_db_connection() as conn, conn.cursor() as cur:
        # pull a few days to be safe; ORDER BY ts DESC LIMIT, then re-sort ASC
        cur.execute(
            f"""
            SELECT ts,
                   (high)::float8 AS high,
                   (low)::float8  AS low,
                   (close)::float8 AS close,
                   COALESCE(volume,0)::float8 AS volume
              FROM {tbl}
             WHERE symbol=%s AND interval=%s
             ORDER BY ts DESC
             LIMIT %s
            """,
            (symbol, "15m", max(60, int(bars)))
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["ts","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    for c in ("high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)
    return df.dropna(subset=["high","low","close"]).astype(
        {"high":"float64","low":"float64","close":"float64","volume":"float64"}
    )

def write_vwap_extensions(symbol: str, *, kind: str, run_id: Optional[str] = None, source: str = "vwap_ext") -> int:
    """
    Compute and upsert:
      - vwap_rolling_20  (rolling over the last 20 bars)
      - vwap_cumulative  (cumulative since earliest bar we loaded)
    into indicators.{spot,futures}_frames keyed by (symbol, interval=BASE_INTERVAL, ts).
    """
    df = _load_recent_15m(symbol, kind, bars=200)
    if df is None or df.empty:
        return 0

    # typical price
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]

    # rolling 20 over the last 20 bars
    roll_v   = df["volume"].rolling(20, min_periods=1).sum().replace(0, pd.NA)
    roll_pv  = pv.rolling(20, min_periods=1).sum()
    vwap_roll_20 = (roll_pv / roll_v).astype("float64")

    # cumulative since earliest loaded bar (stable across the window)
    cum_v   = df["volume"].cumsum().replace(0, pd.NA)
    cum_pv  = pv.cumsum()
    vwap_cum = (cum_pv / cum_v).astype("float64")

    # Build payload for UPSERT into frames (one row per ts)
    table = _frames_table(kind)
    run_id = run_id or datetime.now(TZ).strftime("ind_%Y%m%d")
    values = []
    for ts in df.index:
        vr20 = _safe_float_or_none(vwap_roll_20.loc[ts])
        vcum = _safe_float_or_none(vwap_cum.loc[ts])
        if vr20 is None and vcum is None:
            continue
        values.append((
            symbol, BASE_INTERVAL, ts.to_pydatetime(),  # key
            vr20, vcum, run_id, source
        ))
    if not values:
        return 0

    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(
            cur,
            f"""
            INSERT INTO {table} (symbol, interval, ts, vwap_rolling_20, vwap_cumulative, run_id, source)
            VALUES %s
            ON CONFLICT (symbol, interval, ts)
            DO UPDATE SET
              vwap_rolling_20 = COALESCE(EXCLUDED.vwap_rolling_20, {table}.vwap_rolling_20),
              vwap_cumulative = COALESCE(EXCLUDED.vwap_cumulative, {table}.vwap_cumulative),
              run_id          = EXCLUDED.run_id,
              source          = EXCLUDED.source,
              updated_at      = NOW()
            """,
            values,
            page_size=1000
        )
        conn.commit()
    return len(values)

# ========= VWAP on ALL TFs (exact Σ(tp·vol)/Σ(vol) math) =========

def _bucket_floor(ts: pd.Series, bucket_mins: int) -> pd.Series:
    """Floor timestamps to TF bucket (UTC)."""
    s = ts.astype("int64") // 10**9
    size = bucket_mins * 60
    floored = (s // size) * size
    return pd.to_datetime(floored, utc=True, unit="s")

def _compute_tf_vwap_from_15m(df15: pd.DataFrame, bucket_mins: int) -> pd.DataFrame:
    if df15.empty:
        return pd.DataFrame(columns=["ts", "vwap_rolling_20_calc", "vwap_cumulative_calc"])
    tp = (df15["high"] + df15["low"] + df15["close"]) / 3.0
    vol = df15["volume"].clip(lower=0.0)
    tpvol = tp * vol

    bucket_ts = _bucket_floor(df15.index.to_series(), bucket_mins)
    tf = pd.DataFrame({
        "ts": bucket_ts.values,
        "tpvol": tpvol.values,
        "vol": vol.values
    }).groupby("ts", as_index=False).sum()

    if tf.empty:
        return pd.DataFrame(columns=["ts", "vwap_rolling_20_calc", "vwap_cumulative_calc"])

    tf["r20_tpvol"] = tf["tpvol"].rolling(20, min_periods=1).sum()
    tf["r20_vol"]   = tf["vol"].rolling(20, min_periods=1).sum()
    tf["vwap_rolling_20_calc"] = (tf["r20_tpvol"] / tf["r20_vol"].replace(0, pd.NA)).astype("float64")

    tf["cum_tpvol"] = tf["tpvol"].cumsum()
    tf["cum_vol"]   = tf["vol"].cumsum()
    tf["vwap_cumulative_calc"] = (tf["cum_tpvol"] / tf["cum_vol"].replace(0, pd.NA)).astype("float64")

    return tf[["ts", "vwap_rolling_20_calc", "vwap_cumulative_calc"]].dropna(
        how="all", subset=["vwap_rolling_20_calc","vwap_cumulative_calc"]
    )

def _write_vwap_for_tf(symbol: str, *, kind: str, bucket_mins: int, tf_label: str,
                       run_id: Optional[str]=None, source: str="vwap_tf_exact") -> int:
    df15 = _load_intra_df(symbol, kind)
    if df15 is None or df15.empty:
        return 0
    calc = _compute_tf_vwap_from_15m(df15, bucket_mins)
    if calc.empty:
        return 0

    table = _frames_table(kind)
    run_id = run_id or datetime.now(TZ).strftime("ind_%Y%m%d")
    values = []
    for _, r in calc.iterrows():
        ts = pd.to_datetime(r["ts"], utc=True).to_pydatetime()
        vr20 = _safe_float_or_none(r["vwap_rolling_20_calc"])
        vcum = _safe_float_or_none(r["vwap_cumulative_calc"])
        if vr20 is None and vcum is None:
            continue
        values.append((symbol, tf_label, ts, vr20, vcum, run_id, source))
    if not values:
        return 0

    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(
            cur,
            f"""
            INSERT INTO {table} (symbol, interval, ts, vwap_rolling_20, vwap_cumulative, run_id, source)
            VALUES %s
            ON CONFLICT (symbol, interval, ts)
            DO UPDATE SET
              vwap_rolling_20 = COALESCE(EXCLUDED.vwap_rolling_20, {table}.vwap_rolling_20),
              vwap_cumulative = COALESCE(EXCLUDED.vwap_cumulative, {table}.vwap_cumulative),
              run_id          = EXCLUDED.run_id,
              source          = EXCLUDED.source,
              updated_at      = NOW()
            """,
            values, page_size=1000
        )
        conn.commit()
    return len(values)

def write_vwap_all_tfs(symbol: str, *, kind: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    out["15m"]  = write_vwap_extensions(symbol, kind=kind, source="vwap_tf_exact_15m") or 0
    out["30m"]  = _write_vwap_for_tf(symbol, kind=kind, bucket_mins=30,  tf_label="30m")
    out["60m"]  = _write_vwap_for_tf(symbol, kind=kind, bucket_mins=60,  tf_label="60m")
    out["90m"]  = _write_vwap_for_tf(symbol, kind=kind, bucket_mins=90,  tf_label="90m")
    out["120m"] = _write_vwap_for_tf(symbol, kind=kind, bucket_mins=120, tf_label="120m")
    out["240m"] = _write_vwap_for_tf(symbol, kind=kind, bucket_mins=240, tf_label="240m")   
    return out



# =========================
# Calls to classic / VP+BB / NL
# =========================
def _call_classic(symbol: str, *, kind: str) -> Tuple[int, int]:
    P = classic.load_cfg()

    # Only these 5 TFs
    P["TF_LIST"] = ["15m", "30m", "60m", "90m", "120m" , "240m"]

    # <<< IMPORTANT: tell classic to build higher TFs from the 15m stream >>>
    P["EXTRA_RESAMPLES"] = {
        "30m":  "15m",
        "60m":  "15m",
        "90m": "15m",
        "120m": "15m",
        "240m": "15m",
    }

    # Turn on the metrics you want written into frames
    P["METRICS"] = P.get("METRICS", {})
    for k in ["RSI", "MACD", "ADX", "ROC", "ATR", "MFI", "EMA",
              "RMI", "CCI", "STOCH", "PSAR"]:
        P["METRICS"][k] = True

    out = classic.update_indicators_multi_tf(
        symbols=[symbol],
        kinds=(kind,),
        # correct: we ONLY have 15m in DB
        load_15m=load_intra_from_db,
        get_last_ts=get_last_ts_from_db,
        P=P
    )

    rows: List[tuple] = out.get("rows", [])
    fixed: List[tuple] = []
    for sym, knd, tf, ts, metric, val, ctx, rid, src in rows:
        if metric == "MACD":              # normalize legacy
            metric = "MACD.signal"
        fixed.append((sym, knd, tf, ts, metric, val, ctx, rid, src))

    upserted = _bulk_upsert_frames_from_metric_rows(kind, fixed)
    return (len(rows), upserted)


def _coerce_last_ts(ret: Any) -> Optional[Any]:
    if ret is None:
        return None
    if isinstance(ret, dict) and "last_ts" in ret:
        return ret["last_ts"]
    if isinstance(ret, (list, tuple)) and len(ret) > 0:
        return ret[-1]
    return ret

# --- config flag (top of file, near other CFG/env) ---
NL_ENABLE = (os.getenv("NL_ENABLE", CFG.get("nl", "enable", fallback="1")) != "0")

# --- replace the old function with this ---
def _call_nonlinear(symbol: str, kind: str) -> None:
    """
    Optional nonlinear features. No webhook status, no return value required.
    Safe to no-op if module shape differs.
    """
    if not NL_ENABLE:
        print("[NL] disabled by config/env")
        return
    try:
        # prefer the explicit API if present
        if hasattr(nlf, "process_symbol"):
            nlf.process_symbol(symbol, kind=kind)  # type: ignore
            return
        # fall back to a more generic entry point
        if hasattr(nlf, "run"):
            # support either signature
            try:
                nlf.run(symbol=symbol, kind=kind)  # type: ignore
            except TypeError:
                nlf.run([symbol], kind=kind)  # type: ignore
            return
        # nothing to do if the module has no entry point
        print(f"[NL] skipped: no entry point for nonlinear_features (symbol={symbol}, kind={kind})")
    except Exception as e:
        # let caller decide whether to swallow or log; we rethrow for visibility
        raise RuntimeError(f"nonlinear failed for {symbol}:{kind} → {e}") from e

TF_LIST = ["15m", "30m", "60m", "90m", "120m"]

def _latest_ts_by_tf(symbol: str, kind: str) -> Dict[str, Optional[datetime]]:
    tbl = _frames_table(kind)
    out: Dict[str, Optional[datetime]] = {}
    with get_db_connection() as conn, conn.cursor() as cur:
        for tf in TF_LIST:
            cur.execute(f"SELECT max(ts) FROM {tbl} WHERE symbol=%s AND interval=%s", (symbol, tf))
            row = cur.fetchone()
            out[tf] = (pd.to_datetime(row[0], utc=True).to_pydatetime() if row and row[0] else None)
    return out

def _update_tf_statuses(symbol: str, kind: str, tf_max: Dict[str, Optional[datetime]], run_id: str) -> None:
    """
    Best-effort TF status writer.
    - Only updates <prefix>_<tf>_status (e.g., spot_15min_status).
    - Does NOT touch any *_last_ts columns (they don't exist in DB right now).
    - If the status columns also don't exist, it logs a warning and moves on.
    """
    prefix = "spot" if kind == "spot" else "fut"
    sets, params = [], []

    for tf in TF_LIST:
        col_status = f"{prefix}_{tf}_status".replace("m", "min")
        last_ts    = tf_max.get(tf)
        status_val = "DONE" if last_ts else "NOT_STARTED"
        sets.append(f"{col_status}=%s")
        params.append(status_val)

    sets.append("last_ind_run_id=%s")
    params.append(run_id)

    sql = f"""
        UPDATE reference.symbol_universe
           SET {', '.join(sets)}
         WHERE symbol = %s
    """

    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params + [symbol])
            conn.commit()
    except Exception as e:
        print(f"[STATUS WARN] {symbol} [{kind}] → skipping TF status update: {e}")


# =========================
# Orchestrator
# =========================
# --- ENSURE run_once is defined like this (loop INSIDE the function) ---
def run_once(limit: int = 50, kinds: Iterable[str] = ("spot", "futures")):
    """
    Orchestrator (universe-driven):
    - Picks symbols from reference.symbol_universe needing work
    - Classic → frames (wide) with dedupe via get_last_ts_from_db
    - Session VWAP for BOTH spot and futures → vwap_session in frames
    - Updates per-TF status and gates in symbol_universe
    - Optional: VP+BB and Nonlinear (kept but no webhook statuses)
    """
    rows = fetch_batch_universe(limit)
    print(f"\n⏱️ Indicators batch: {len(rows)} row(s) [universe={UNIVERSE_NAME}, base={BASE_INTERVAL}]")

    for r in rows:
        sym: str = r["symbol"]
        print(f"\n🧮 Indicators: {sym}")

        for kind in kinds:
            try:
                with get_db_connection() as _conn:
                    start, target, should_run = _plan_indicator_window(_conn, sym, kind)

                if not should_run:
                    with get_db_connection() as _conn:
                        _set_universe_run_status(_conn, sym, kind, "IND_DONE")
                    print(f"[IND] {sym}:{kind} up-to-date (start={_d(start)}, target={_d(target)})")
                    continue

                with get_db_connection() as _conn:
                    _set_universe_run_status(_conn, sym, kind, "IND_RUNNING")

                # Classic → frames
                attempted, upserted = _call_classic(sym, kind=kind)
                print(f"[WRITE] {kind}:{sym} classic → tried {attempted} metric rows, upserted {upserted} frame rows")

                # Session VWAP → frames for BOTH kinds
                if kind == "futures":
                    wrote_vwap = write_futures_vwap_session(sym)
                    print(f"[WRITE] futures:{sym} → vwap_session rows={wrote_vwap}")
                else:
                    wrote_vwap = write_spot_vwap_session(sym)
                    print(f"[WRITE] spot:{sym} → vwap_session rows={wrote_vwap}")
                # Rolling & Cumulative VWAP (frames)
                # Extended VWAPs → frames (rolling-20 + cumulative)
                # VWAPs → frames for ALL TFs (exact math)
                try:
                    vwap = write_vwap_all_tfs(sym, kind=kind)
                    print(f"[WRITE] {kind}:{sym} → vwap_all_tfs rows={vwap}")
                except Exception as e:
                    print(f"[VWAP ERROR] {sym} [{kind}] → {e}")
                
                
                # Advance gates + per-TF statuses
                try:
                    with get_db_connection() as _conn:
                        _advance_universe_gate(_conn, sym, kind, target, WORKER_RUN_ID)
                        _set_universe_run_status(_conn, sym, kind, "IND_DONE")
                    tf_max = _latest_ts_by_tf(sym, kind)
                    _update_tf_statuses(sym, kind, tf_max, WORKER_RUN_ID)
                except Exception as e2:
                    print(f"[STATUS ERROR] {sym} [{kind}] → {e2}")

            except Exception as e:
                print(f"[IND ERROR] {sym} [{kind}] → {e}\n{traceback.format_exc()}")
                continue

            # Optional extras (no webhook statuses)
            # --- replace the whole dynamic vpbb_func block with this ---
            try:
                if hasattr(vpbb, "run"):
                    # correct entrypoint: handles kind='spot' | 'futures' | None (ini) safely
                    vpbb.run(symbols=[sym], kind=kind)
                else:
                    # fallback: call process_symbol with an explicit cfg.market_kind
                    if hasattr(vpbb, "load_vpbb_cfg") and hasattr(vpbb, "process_symbol"):
                        _cfg = vpbb.load_vpbb_cfg()
                        _cfg.market_kind = kind  # force the side we're looping
                        vpbb.process_symbol(sym, cfg=_cfg)
                    else:
                        print("[VPBB] no callable entry point found")
            except Exception as e:
                print(f"[VPBB ERROR] {sym} [{kind}] → {e}\n{traceback.format_exc()}")
            try:
                _call_nonlinear(sym, kind=kind)
            except Exception as e:
                print(f"[Non_Linear ERROR] {sym} [{kind}] → {e}\n{traceback.format_exc()}")


# --------------------------------------------------------------------

# =========================
# Indicator window planning
# =========================
def _get_universe_cursor(conn, symbol: str, kind: str) -> Tuple[Optional[datetime], Optional[datetime], str]:
    """Fetches the indicator cursor for a given symbol and market kind."""
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

from typing import Optional, Any, List
from datetime import datetime

def _set_universe_run_status(
    conn,
    symbol: str,
    kind: str,                 # "spot" | "futures"
    status: str,               # e.g. "IND_RUNNING" | "IND_DONE" | "IND_ERROR"
    gate_ts: Optional[datetime] = None,        # advance last_ind_*_at to this if provided
    last_ingested_ts: Optional[datetime] = None,  # legacy (best-effort)
    error: Optional[str] = None,                  # legacy (best-effort)
    run_id: Optional[str] = None,                 # optional: write last_ind_run_id
) -> None:
    """
    Canonical updater for indicator run status + gate.
    Writes to standardized columns; tries legacy columns if present (best-effort).
    """

    # Canonical column names
    status_col = "ind_status_spot" if kind == "spot" else "ind_status_fut"
    gate_col   = "ind_last_spot_run_at" if kind == "spot" else "ind_last_fut_run_at"


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

        # ---- Legacy columns: try to keep old codepaths happy (ignore if not present)
        try:
            legacy_sets: List[str] = []
            legacy_params: List[Any] = []

            legacy_status_col = "last_ind_spot_status" if kind == "spot" else "last_ind_fut_status"
            legacy_sets.append(f"{legacy_status_col}=%s")
            legacy_params.append(status)

            if last_ingested_ts is not None:
                legacy_ing_col = (
                    "last_ind_spot_last_ingested_ts" if kind == "spot" else "last_ind_fut_last_ingested_ts"
                )
                legacy_sets.append(f"{legacy_ing_col}=%s")
                legacy_params.append(last_ingested_ts)

            if error is not None:
                legacy_err_col = (
                    "last_ind_spot_last_error" if kind == "spot" else "last_ind_fut_last_error"
                )
                legacy_sets.append(f"{legacy_err_col}=%s")
                legacy_params.append(error)

            if legacy_sets:
                cur.execute(
                    f"UPDATE reference.symbol_universe SET {', '.join(legacy_sets)} WHERE symbol=%s",
                    legacy_params + [symbol],
                )
        except Exception:
            # Column might not exist; ignore to stay resilient
            pass

    conn.commit()


def _advance_universe_gate(conn, symbol: str, kind: str, new_last_ind_at: datetime, run_id: str) -> None:
    col_name   = "ind_last_spot_run_at" if kind == "spot" else "ind_last_fut_run_at"
    run_id_col = "last_ind_run_id"
    sql = f"""
        UPDATE reference.symbol_universe
           SET {col_name}   = GREATEST(COALESCE({col_name}, 'epoch'::timestamptz), %s),
               {run_id_col} = %s
         WHERE symbol = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (new_last_ind_at, run_id, symbol))
    conn.commit()


def _plan_indicator_window(conn, symbol: str, kind: str) -> Tuple[datetime, datetime, bool]:
    """Determines the start, target, and if indicators should run for a symbol/kind."""
    # For now, simply return a window that always runs. This needs to be integrated with actual cursor logic.
    return datetime.now(TZ) - timedelta(days=IND_LOOKBACK_DAYS), datetime.now(TZ), True

# =========================
# CLI
# =========================
# --- MAIN: add a tiny kinds flag and call run_once with a default ---
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
# --------------------------------------------------------------------
