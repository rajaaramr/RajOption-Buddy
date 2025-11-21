# scheduler/fetch_prices.py — ORCHESTRATOR (spot+fut + option/PCR snapshots, status handoff)
from __future__ import annotations

import os, sys
from datetime import datetime, timezone
from typing import Optional, Callable, Dict, Any

from utils.db import get_db_connection
from utils.db_ops import log_run_status

from scheduler.backfill_history import backfill_symbols, sync_today
from configparser import ConfigParser
from datetime import timezone
TZ = timezone.utc

import os, configparser

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_PATH  = os.path.join(PROJ_ROOT, "configs", "data.ini")

def _load_cfg() -> dict:
    cp = configparser.ConfigParser()
    cp.read(INI_PATH)
    out = {}
    for sec in cp.sections():
        out[sec.lower()] = {k.lower(): v for k, v in cp.items(sec)}
    return out


CFG = _load_cfg()
MODULES = CFG.get("modules", {})        # dict
OPTIONS = CFG.get("options", {})        # dict

live_cfg = CFG.get("live", {})
interval = str(live_cfg.get("interval", "15m")).lower()
CFG["live_interval"] = interval if interval in ("1m","3m","5m","10m","15m","30m","60m","120m","240m","1d") else "15m"
def _as_bool(x):
    return str(x).strip().lower() in ("1", "true", "yes", "on")

FETCH_OPTION_SNAPSHOT = _as_bool(MODULES.get("options_snapshot", False))
FETCH_PCR_SNAPSHOT    = _as_bool(MODULES.get("pcr_snapshot", False))
OPTION_STRIKE_RANGE   = int(OPTIONS.get("strike_range", 3))
print(f"[CFG] options_snapshot={FETCH_OPTION_SNAPSHOT} pcr_snapshot={FETCH_PCR_SNAPSHOT} strike_range={OPTION_STRIKE_RANGE}")

# --- Lazy import shims (so ingestion never dies if optional deps fail) ---
_opt_fn: Optional[Callable[..., int]] = None
_opt_err: Optional[BaseException] = None

_pcr_fn: Optional[Callable[..., Dict[str, Any]]] = None
_pcr_err: Optional[BaseException] = None

def _lazy_import_option_snapshot() -> Optional[Callable[..., int]]:
    global _opt_fn, _opt_err
    if _opt_fn is not None or _opt_err is not None:
        return _opt_fn
    try:
        # IMPORTANT: your module must export fetch_and_store_option_chain(symbol, kite=None, strike_range=int) -> int
        from scheduler.fetch_and_store_option_chain import fetch_and_store_option_chain  # type: ignore
        _opt_fn = fetch_and_store_option_chain
    except Exception as e:
        _opt_err = e
    return _opt_fn

def _lazy_import_pcr_snapshot() -> Optional[Callable[..., Dict[str, Any]]]:
    global _pcr_fn, _pcr_err
    if _pcr_fn is not None or _pcr_err is not None:
        return _pcr_fn
    try:
        # From scheduler/fetch_pcr_option_chain.py provided earlier
        from scheduler.fetch_pcr_option_chain import run_pcr_snapshot  # type: ignore
        _pcr_fn = run_pcr_snapshot
    except Exception as e:
        _pcr_err = e
    return _pcr_fn

# --- Small DB helpers ---
def _exec(sql: str, params: list | tuple):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
    finally:
        conn.close()

def set_status(unique_id: str, *, status: Optional[str] = None, sub_status: Optional[str] = None, error: Optional[str] = None):
    sets, vals = [], []
    if status is not None:     sets.append("status=%s");     vals.append(status)
    if sub_status is not None: sets.append("sub_status=%s"); vals.append(sub_status)
    if error is not None:      sets.append("last_error=%s"); vals.append(error)
    if not sets:
        return
    vals.append(unique_id)
    _exec(f"UPDATE webhooks.webhook_alerts SET {', '.join(sets)} WHERE unique_id=%s", vals)

def _get_last_ts(symbol: str, interval: str, table: str) -> Optional[datetime]:
    sql = f"SELECT max(ts) FROM {table} WHERE symbol=%s AND interval=%s"
    conn = get_db_connection()
    try:
        cur = conn.cursor(); cur.execute(sql, (symbol, interval)); row = cur.fetchone()
        cur.close()
        return row[0] if row and row[0] else None
    finally:
        conn.close()

def _needs_backfill(symbol: str, interval: str, gap_minutes: int) -> bool:
    fut_last = _get_last_ts(symbol, interval, "market.futures_candles")
    spot_last = _get_last_ts(symbol, interval, "market.spot_candles")
    last = max([x for x in (fut_last, spot_last) if x is not None], default=None)
    if last is None:
        return True
    delta_min = (datetime.now(TZ) - last).total_seconds() / 60.0
    return delta_min > gap_minutes



# --- Core orchestration for a single row ---
def orchestrate_fetch(row: Dict[str, Any]):
    run_id = datetime.now(TZ).strftime("%Y%m%dT%H%M%SZ")
    from utils.db_ops import log_run_status
    
    unique_id = row["unique_id"]
    symbol = str(row["symbol"]).upper()
    run_id = datetime.now(TZ).strftime("ING-%Y%m%dT%H%M%S")
    print(f"\n🛠️ Ingestion: {symbol} ({unique_id})")

    try:
        # 1) History seeds / checks
        set_status(unique_id, sub_status="HISTORY_BACKFILLING")
        log_run_status(run_id=run_id, job="INGEST", symbol=symbol, phase="START", status="OK")
        lookback_days = int(CFG.get("lookback_days", 60))
        backfill_symbols([symbol], kinds=("spot", "futures")) 
        # 2) Live ingestion (spot + futures)
        set_status(unique_id, sub_status="LIVE_INGESTING")
        # 2) Live ingestion (spot + futures OHLCV 5m etc.)
        log_run_status(run_id=run_id, job="INGEST", symbol=symbol, phase="LIVE", status="OK")
        out = sync_today([symbol], kinds=("spot", "futures"), interval="15m", lookback_days_today=2)
        rows_written = sum((v[0] + v[1]) for v in out.values())
        if rows_written <= 0:
            # Not fatal. Symbol might already be up to date. Continue to snapshots.
            set_status(unique_id, sub_status="LIVE_INGESTING")
            print(f"ℹ️ {symbol}: engine wrote 0 rows (already synced?)")


        # 3) Option Chain snapshot
        if FETCH_OPTION_SNAPSHOT:
            fn = _lazy_import_option_snapshot()
            if not fn:
                if _opt_err:
                    print(f"⚠️ Option snapshot disabled (import fail): {_opt_err}")
            else:
                try:
                    nrows = fn(symbol, kite=None, strike_range=OPTION_STRIKE_RANGE)
                    print(f"🧩 {symbol}: option snapshot rows={nrows}")
                except Exception as e:
                    print(f"⚠️ {symbol}: option snapshot failed: {e}")

        # 4) PCR snapshot
        if FETCH_PCR_SNAPSHOT:
            pcr = _lazy_import_pcr_snapshot()
            if not pcr:
                if _pcr_err:
                    print(f"⚠️ PCR snapshot disabled (import fail): {_pcr_err}")
            else:
                try:
                    run_id = datetime.now(TZ).strftime("%Y%m%dT%H%M%SZ")
                    out = pcr(symbol, None, run_id=run_id)
                    print(f"🧪 {symbol}: PCR={out.get('pcr')} (rows={out.get('rows')}, exp={out.get('expiry')})")
                except Exception as e:
                    print(f"⚠️ {symbol}: PCR snapshot failed: {e}")

        # 5) Handoff to indicators
        set_status(unique_id, sub_status="INGESTED_OK")
        set_status(unique_id, status="INDICATOR_PROCESS", sub_status="IND_PENDING")
        log_run_status(run_id=run_id, job="INGEST", symbol=symbol, phase="DONE", status="OK")
        print(f"✅ {symbol}: INGESTED_OK → INDICATOR_PROCESS")

    except Exception as e:
        log_run_status(run_id=run_id, job="INGEST", symbol=symbol, phase="ERROR", status="ERROR",
               error_code=type(e).__name__, info={"msg": str(e)})
        set_status(unique_id, sub_status="INGESTION_ERROR_CRASH", error=str(e))
        print(f"❌ {symbol}: crash → {e}")

# --- Batch runner: pulls rows from webhook table and orchestrates each ---
def run():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT unique_id, symbol
              FROM webhooks.webhook_alerts
             WHERE status='DATA_PROCESSING'
               AND (sub_status IS NULL OR sub_status NOT IN ('INGESTED_OK','INGESTION_ERROR_NO_DATA','INGESTION_ERROR_CRASH'))
             ORDER BY received_at ASC
             FOR UPDATE SKIP LOCKED
        """)
        rows = [{"unique_id": r[0], "symbol": r[1]} for r in cur.fetchall()]
        cur.close()
    finally:
        conn.close()

    print(f"\n⏱️ Ingestion batch: {len(rows)} row(s)")
    if not rows:
        if _opt_err:
            print(f"ℹ️ Note: option snapshot import initially failed: {_opt_err}")
        if _pcr_err:
            print(f"ℹ️ Note: PCR snapshot import initially failed: {_pcr_err}")
        return

    for r in rows:
        orchestrate_fetch(r)

if __name__ == "__main__":
    run()
