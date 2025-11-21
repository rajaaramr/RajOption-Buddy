# Single engine for spot+fut ingestion (+optional zones).
# Backfill is idempotent and skipped if 5d present (unless mode='backfill').

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from utils.kite_utils import load_kite
from utils.db import get_db_connection
from utils.db_ops import (
    insert_spot_bar,
    insert_futures_bar,
    upsert_volume_zone,
)
from utils.symbol_utils import get_futures_tradingsymbol, get_spot_tradingsymbol
from scheduler.backfill_history import ensure_history_timescale

TZ = timezone.utc

@dataclass
class FetchParams:
    symbols: Optional[List[str]] = None
    interval: str = "5m"
    mode: str = "live"             # 'live' | 'backfill' | 'zones'
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    include_zones: bool = False
    source: str = "manual"         # 'manual' | 'webhooks'
    webhooks_status: str = "DATA_PROCESSING"

# ---------------- Helpers ----------------

_VALID_INTERVALS = {"1m","3m","5m","10m","15m","30m","60m","1d"}
_TF_MAP = {"1m":"minute","3m":"3minute","5m":"5minute","10m":"10minute",
           "15m":"15minute","30m":"30minute","60m":"60minute","1d":"day"}

def _symbols_from_webhooks(status: str = "DATA_PROCESSING") -> List[str]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM webhooks.webhook_alerts WHERE status=%s", (status,))
        rows = cur.fetchall()
    return [r[0] for r in rows] if rows else []

def _max_ts(table: str, symbol: str, interval: str) -> Optional[datetime]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT max(ts) FROM {table} WHERE symbol=%s AND interval=%s", (symbol, interval))
        row = cur.fetchone()
    return row[0] if row and row[0] else None

def _min_ts(table: str, symbol: str, interval: str) -> Optional[datetime]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT min(ts) FROM {table} WHERE symbol=%s AND interval=%s", (symbol, interval))
        row = cur.fetchone()
    return row[0] if row and row[0] else None

def _has_recent_futures(symbol: str, interval: str, max_age_minutes: int = 120) -> bool:
    last = _max_ts("market.futures_candles", symbol, interval)
    if not last:
        return False
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
    return age_min <= max_age_minutes

def _has_5d_coverage(symbol: str, interval: str) -> bool:
    """
    True if spot has >=5d coverage. Futures is best-effort:
    - If futures exists, require >=5d too.
    - If futures absent (no min ts), do NOT block on it.
    """
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=5)

    spot_min = _min_ts("market.spot_candles", symbol, interval)
    spot_ok = bool(spot_min and (spot_min <= cutoff))

    fut_min = _min_ts("market.futures_candles", symbol, interval)
    fut_ok = (fut_min is None) or (fut_min <= cutoff)

    return bool(spot_ok and fut_ok)

def _ist_to_utc(ts_naive_ist: datetime) -> datetime:
    """
    Kite historical timestamps are IST-naive. Localize to IST then convert to UTC.
    """
    # IST = UTC+05:30
    ist = timezone(timedelta(hours=5, minutes=30))
    if ts_naive_ist.tzinfo is None:
        ts_local = ts_naive_ist.replace(tzinfo=ist)
    else:
        ts_local = ts_naive_ist.astimezone(ist)
    return ts_local.astimezone(TZ)

# ---------------- Source fetchers ----------------

def _fetch_from_source(
    *, kite, symbol: str, interval: str,
    since: Optional[datetime], until: Optional[datetime],
    live: bool
) -> List[Dict[str, Any]]:
    """
    LIVE: synth bar from LTP for spot and (if resolvable) futures (O=H=L=C=ltp).
    BACKFILL: historical_data for spot and futures (best-effort).
    Returns list of bars:
      {"ts","open","high","low","close","volume","oi?","type":"spot"|"futures"}
    """
    bars: List[Dict[str, Any]] = []
    now = datetime.now(tz=TZ)

    if interval not in _VALID_INTERVALS:
        raise ValueError(f"Unsupported interval={interval}. Allowed: {_VALID_INTERVALS}")

    if live:
        # ---- LIVE via ltp() ----
        nse_key = f"NSE:{(get_spot_tradingsymbol(symbol) or symbol).upper()}"
        fut_tsym = get_futures_tradingsymbol(symbol)
        req = [nse_key] + ([f"NFO:{fut_tsym}"] if fut_tsym else [])
        try:
            ltp_map = kite.ltp(req) or {}
        except Exception as e:
            print(f"❌ LTP failed for {symbol}: {e}")
            return bars

        # spot synth
        spot_px = (ltp_map.get(nse_key) or {}).get("last_price")
        if spot_px is not None:
            px = float(spot_px)
            bars.append({"ts": now, "open": px, "high": px, "low": px, "close": px, "volume": 0, "type": "spot"})

        # futures synth
        if fut_tsym:
            fut_key = f"NFO:{fut_tsym}"
            fut_px = (ltp_map.get(fut_key) or {}).get("last_price")
            if fut_px is not None:
                px = float(fut_px)
                bars.append({"ts": now, "open": px, "high": px, "low": px, "close": px, "volume": 0, "oi": 0, "type": "futures"})

        return bars

    # ---- BACKFILL via historical_data ----
    tf = _TF_MAP.get(interval, "5minute")
    now = datetime.now(tz=TZ)
    if since is None:
        since = now - timedelta(days=10)
    # Make `until` slightly ahead to include the latest day’s bucket
    if until is None:
        until = now + timedelta(minutes=2)

    # Spot token
    spot_token: Optional[int] = None
    try:
        spot_tsym = get_spot_tradingsymbol(symbol) or symbol
        for ins in kite.instruments("NSE"):
            if ins.get("tradingsymbol", "").upper() == spot_tsym.upper():
                spot_token = int(ins["instrument_token"]); break
    except Exception as e:
        print(f"⚠️ instruments(NSE) failed: {e}")

    if spot_token:
        try:
            data = kite.historical_data(spot_token, since, until, interval=tf, continuous=False, oi=False) or []
            for d in data:
                ts = d["date"]
                # FIX: localize IST, convert to UTC
                ts_utc = _ist_to_utc(ts)
                bars.append({
                    "ts": ts_utc,
                    "open": d["open"], "high": d["high"], "low": d["low"], "close": d["close"],
                    "volume": d.get("volume", 0),
                    "type": "spot"
                })
        except Exception as e:
            print(f"⚠️ backfill SPOT failed for {symbol}: {e}")

    # Futures token (best-effort)
    fut_token: Optional[int] = None
    try:
        from utils.kite_utils import get_futures_instrument
        finfo = get_futures_instrument(symbol, kite) or {}
        fut_token = finfo.get("instrument_token")
        if not fut_token:
            fut_tsym = get_futures_tradingsymbol(symbol)
            if fut_tsym:
                for ins in kite.instruments("NFO"):
                    if ins.get("tradingsymbol", "").upper() == fut_tsym.upper():
                        fut_token = int(ins["instrument_token"]); break
    except Exception:
        pass

    if fut_token:
        try:
            data = kite.historical_data(fut_token, since, until, interval=tf, continuous=False, oi=True) or []
            for d in data:
                ts = d["date"]
                ts_utc = _ist_to_utc(ts)  # FIX: IST->UTC
                bars.append({
                    "ts": ts_utc,
                    "open": d["open"], "high": d["high"], "low": d["low"], "close": d["close"],
                    "volume": d.get("volume", 0),
                    "oi": d.get("oi", 0),
                    "type": "futures"
                })
        except Exception as e:
            print(f"⚠️ backfill FUTURES failed for {symbol}: {e}")

    return bars

# ---------------- Public API ----------------

def fetch_prices(params: FetchParams, kite=None) -> int:
    """
    Unified engine. Returns approx rows written.
      - live:     backfill only if 5d history is NOT present; then write LTP-synth bar(s)
      - backfill: bulk history pull between since..until
      - zones:    compute/update zones only (fast path), or after inserts if include_zones=True
    """
    if kite is None:
        kite = load_kite()

    if params.mode not in {"live", "backfill", "zones"}:
        raise ValueError(f"Unsupported mode: {params.mode}")

    symbols = params.symbols
    if (not symbols) and params.source == "webhooks":
        symbols = _symbols_from_webhooks(status=params.webhooks_status)

    if not symbols:
        return 0

    written = 0

    if params.mode in {"live", "backfill"}:
        for sym in symbols:
            if params.mode == "live":
                try:
                    cov_ok = _has_5d_coverage(sym, params.interval)
                    if not cov_ok:
                        s, f = ensure_history_timescale(sym, interval=params.interval, lookback_days=5)
                        print(f"📦 Backfilled {sym} {params.interval}: spot={s}, fut={f}")
                except Exception as e:
                    print(f"⚠️ Backfill check failed for {sym}: {e}")

            bars = _fetch_from_source(
                kite=kite, symbol=sym, interval=params.interval,
                since=params.since, until=params.until, live=(params.mode == "live"),
            )

            for bar in bars:
                if bar.get("type") == "futures" or sym.endswith("FUT"):
                    insert_futures_bar(
                        symbol=sym[:-3] if sym.endswith("FUT") else sym,
                        interval=params.interval,
                        ts=bar["ts"],
                        open_price=bar["open"], high_price=bar["high"],
                        low_price=bar["low"], close_price=bar["close"],
                        volume=bar.get("volume", 0), oi=bar.get("oi", 0),
                    )
                else:
                    insert_spot_bar(
                        symbol=sym,
                        interval=params.interval,
                        ts=bar["ts"],
                        open_price=bar["open"], high_price=bar["high"],
                        low_price=bar["low"], close_price=bar["close"],
                        volume=bar.get("volume", 0),
                    )
                written += 1

    if params.mode == "zones" or params.include_zones:
        for sym in symbols:
            _update_zones_for_symbol(sym, params.interval)

    return written

def _update_zones_for_symbol(symbol: str, interval: str) -> None:
    # Hook your zone computation here by calling upsert_volume_zone(...)
    return

if __name__ == "__main__":
    syms = ["INFY", "TCS"]
    params = FetchParams(symbols=syms, interval="5m", mode="live", include_zones=False, source="manual")
    wrote = fetch_prices(params)
    print(f"✅ market_data wrote {wrote} bar(s) for {len(syms)} symbol(s).")
