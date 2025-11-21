# scheduler/backfill_history.py
# Seed & maintain 5m spot/futures history in Timescale (Zerodha Kite).
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict, Any, Iterable, Literal
from utils.db import get_db_connection  # same helper you use elsewhere
from utils.kite_utils import load_kite
from utils.db_ops import insert_spot_price, insert_futures_price
import psycopg2.extras as pgx


TZ = timezone.utc

# Map our interval strings to Kite's
_KITE_TF = {
    "1m": "minute", "3m": "3minute", "5m": "5minute", "10m": "10minute",
    "15m": "15minute", "30m": "30minute", "60m": "60minute", "1d": "day",
}


# ---- Discovery helpers (add to backfill_history.py) ----


def discover_symbols(source: str = "webhooks",
                     lookback_days: int = 14,
                     universe_name: str | None = None) -> list[str]:
    """
    Find symbols to backfill.
    source:
      - "webhooks": DISTINCT symbols seen recently in webhooks.webhook_alerts
      - "universe": read from reference.symbol_universe (requires universe_name)
    """
    syms: list[str] = []
    with get_db_connection() as conn, conn.cursor() as cur:
        if source == "webhooks":
            cur.execute("""
                SELECT DISTINCT symbol
                FROM webhooks.webhook_alerts
                WHERE received_at >= now() - INTERVAL %s
                ORDER BY 1
            """, (f"{lookback_days} days",))
            syms = [r[0] for r in cur.fetchall() or []]
        elif source == "universe":
            if not universe_name:
                return []
            cur.execute("""
                SELECT symbol
                FROM reference.symbol_universe
                WHERE universe_name=%s
                ORDER BY 1
            """, (universe_name,))
            syms = [r[0] for r in cur.fetchall() or []]
    return [s.upper() for s in syms if s]

def snapshot_universe(symbols: list[str],
                      universe_name: str = "largecaps_v1",
                      source: str = "webhooks") -> int:
    """
    Save a stable list into reference.symbol_universe. Idempotent.
    """
    if not symbols:
        return 0
    symbols = sorted(set(s.upper() for s in symbols))
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, """
            INSERT INTO reference.symbol_universe (universe_name, symbol, source)
            VALUES %s
            ON CONFLICT (universe_name, symbol) DO NOTHING
        """, [(universe_name, s, source) for s in symbols])
        conn.commit()
    return len(symbols)


# -------- Zerodha helpers --------

def _historical_data_with_retry(kite, token: int, start: datetime, end: datetime,
                                tf: str, with_oi: bool, *, max_retries=4) -> List[Dict[str, Any]]:
    """
    Call Kite historical_data with simple exponential backoff on 429/5xx.
    """
    attempt = 0
    pause = 0.75
    while True:
        try:
            data = kite.historical_data(token, start, end, interval=tf,
                                        continuous=False, oi=with_oi) or []
            out: List[Dict[str, Any]] = []
            for c in data:
                ts = c["date"]
                if getattr(ts, "tzinfo", None) is None:
                    ts = ts.replace(tzinfo=TZ)
                else:
                    ts = ts.astimezone(TZ)
                out.append({
                    "ts": ts,
                    "open": float(c["open"]), "high": float(c["high"]),
                    "low": float(c["low"]),  "close": float(c["close"]),
                    "volume": int(c.get("volume") or 0),
                    "oi": int(c.get("oi") or 0) if with_oi else 0
                })
            return out
        except Exception as e:
            msg = str(e)
            attempt += 1
            # crude detection for throttling; back off
            if attempt <= max_retries and ("429" in msg or "Too many" in msg or "timeout" in msg or "500" in msg):
                time.sleep(pause)
                pause *= 2.0
                continue
            raise

def _chunk_ranges(start: datetime, end: datetime, *, days_per_chunk: int) -> List[Tuple[datetime, datetime]]:
    out = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days_per_chunk), end)
        out.append((cur, nxt))
        cur = nxt
    return out

def _instrument_maps(kite) -> Tuple[Dict[str,int], Dict[str,int]]:
    """
    Build tradingsymbol -> instrument_token maps for NSE (spot) and NFO (futures).
    We do this once per run to avoid calling kite.instruments() repeatedly.
    """
    spot_map: Dict[str,int] = {}
    fut_map: Dict[str,int]  = {}
    for ins in (kite.instruments("NSE") or []):
        ts = str(ins.get("tradingsymbol","")).upper()
        tok = ins.get("instrument_token")
        if ts and tok:
            spot_map[ts] = int(tok)
    for ins in (kite.instruments("NFO") or []):
        ts = str(ins.get("tradingsymbol","")).upper()
        tok = ins.get("instrument_token")
        if ts and tok:
            fut_map[ts] = int(tok)
    return spot_map, fut_map

def _guess_spot_tsym(symbol: str) -> str:
    # If you already maintain a proper mapping, swap this with your util.
    # Here we assume the DB symbol equals NSE tradingsymbol.
    return symbol.upper()

def _guess_frontmonth_future_ts(spot_tsym: str, *, month_hint: Optional[str] = None) -> Optional[str]:
    """
    Very simple guesser: tries current month/month+1 weekly/monthly futures code patterns.
    If you have a robust mapping util already, use that instead.
    """
    # Prefer your existing utils if available:
    # from utils.symbol_utils import get_futures_tradingsymbol
    # return get_futures_tradingsymbol(spot_tsym)
    return None  # leave None to rely on instrument map filtering below

# -------- Core writers --------

def _write_spot_rows(symbol: str, rows: List[Dict[str, Any]], *, interval: str) -> int:
    wrote = 0
    for r in rows:
        insert_spot_price(
            ts=r["ts"], symbol=symbol,
            open=r["open"], high=r["high"], low=r["low"], close=r["close"],
            volume=r["volume"], interval=interval
        )
        wrote += 1
    return wrote

def _write_fut_rows(symbol: str, rows: List[Dict[str, Any]], *, interval: str) -> int:
    wrote = 0
    for r in rows:
        insert_futures_price(
            ts=r["ts"], symbol=symbol,
            open=r["open"], high=r["high"], low=r["low"], close=r["close"],
            volume=r["volume"], oi=r["oi"], interval=interval
        )
        wrote += 1
    return wrote

def _get_fut_cursor(conn, symbol: str) -> tuple[datetime | None, datetime | None, str]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT fut_15m_next_from_ts, fut_15m_target_until_ts, COALESCE(fut_15m_status,'NOT_STARTED')
            FROM reference.symbol_universe
            WHERE symbol = %s
        """, (symbol,))
        row = cur.fetchone()
        return (row[0], row[1], row[2]) if row else (None, None, 'NOT_STARTED')

def _update_fut_cursor(conn, symbol: str, *,
                       status: str | None = None,
                       next_from_ts: datetime | None = None,
                       last_ingested_ts: datetime | None = None,
                       error: str | None = None) -> None:
    sets = []
    params = []
    if status is not None:
        sets.append("fut_15m_status = %s"); params.append(status)
    if next_from_ts is not None:
        sets.append("fut_15m_next_from_ts = %s"); params.append(next_from_ts)
    if last_ingested_ts is not None:
        sets.append("fut_15m_last_ingested_ts = %s"); params.append(last_ingested_ts)
    if error is not None:
        sets.append("fut_last_error = %s"); params.append(error)
    if not sets:
        return
    sql = f"UPDATE reference.symbol_universe SET {', '.join(sets)} WHERE symbol = %s"
    params.append(symbol)
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()



# -------- Public API --------

def backfill_symbols(symbols: Iterable[str], *,
                     interval: str = "15m",              # forced to 15m
                     lookback_days: int = 60,            # used only if DB cursor missing
                     chunk_days: int = 1,                # small chunks to avoid throttling
                     pace_sleep: float = 0.4) -> Dict[str, Tuple[int,int]]:
    """
    Futures-only, 15m, resume-safe backfill:
    - Reads fut_15m_next_from_ts / fut_15m_target_until_ts from reference.symbol_universe
    - Fetches in chunks [start, start+chunk_days)
    - UPSERTS into futures.candles via insert_futures_price()
    - Updates cursor after each successful chunk
    - Marks SEED_DONE when next_from_ts >= target_until_ts
    Returns {symbol: (spot_rows_written=0, fut_rows_written)} for compatibility.
    """
    kite = load_kite()
    tf = _KITE_TF.get("15m", "15minute")  # hard lock to 15m
    now_utc = datetime.now(tz=TZ)

    # Build NFO instrument token map once
    _, fut_map = _instrument_maps(kite)

    results: Dict[str, Tuple[int,int]] = {}
    with get_db_connection() as conn:
        for sym in (s.upper() for s in symbols):
            start_ts, target_ts, status = _get_fut_cursor(conn, sym)
            # If not initialized (shouldn't happen if you ran the init SQL), fallback to lookback
            if start_ts is None:
                start_ts = now_utc - timedelta(days=lookback_days)
            if target_ts is None:
                target_ts = now_utc

            fut_token = None
            for ts, tok in fut_map.items():
                if ts.startswith(sym) and "FUT" in ts:
                    fut_token = tok
                    break

            wrote_fut = 0
            cur_start = start_ts

            while cur_start < target_ts:
                cur_end = min(cur_start + timedelta(days=chunk_days), target_ts)
                try:
                    if fut_token:
                        data = _historical_data_with_retry(kite, fut_token, cur_start, cur_end, tf, with_oi=True)
                        # write rows
                        for r in data:
                            insert_futures_price(
                                ts=r["ts"], symbol=sym,
                                open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                                volume=r["volume"], oi=r["oi"], interval="15m"
                            )
                        wrote_fut += len(data)

                    # advance cursor to cur_end
                    _update_fut_cursor(
                        conn, sym,
                        status="SEED_RUNNING",
                        next_from_ts=cur_end,
                        last_ingested_ts=(cur_end - timedelta(minutes=15))
                    )
                    time.sleep(pace_sleep)

                except Exception as e:
                    # record error, keep going to next symbol
                    _update_fut_cursor(conn, sym, status="ERROR", error=str(e))
                    print(f"⚠️ FUT backfill {sym} {cur_start:%Y-%m-%d}->{cur_end:%Y-%m-%d}: {e}")
                    break  # move to next symbol

                cur_start = cur_end

            # If finished window, mark done
            # Re-read to avoid race
            new_next, new_target, _ = _get_fut_cursor(conn, sym)
            if new_next and new_target and new_next >= new_target:
                _update_fut_cursor(conn, sym, status="SEED_DONE")

            results[sym] = (0, wrote_fut)
            print(f"✅ {sym}: fut_15m={wrote_fut} rows")

    return results


def sync_today(symbols: Iterable[str], *,
               interval: str = "5m",
               lookback_days_today: int = 2,
               chunk_days: int = 1,
               pace_sleep: float = 0.4) -> Dict[str, Tuple[int,int]]:
    """
    Lightweight daily/nightly sync (e.g., schedule at 18:30 IST).
    Only fetches the last 1–2 days to catch up today’s bars.
    """
    return backfill_symbols(
        symbols, interval=interval,
        lookback_days=lookback_days_today,
        chunk_days=chunk_days, pace_sleep=pace_sleep
    )

# -------- CLI --------
if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=",1)[0]: a.split("=",1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}

    mode = opts.get("mode","seed")  # seed or sync
    interval = opts.get("interval","5m")  # keep 5m for fidelity
    lookback = int(opts.get("lookback_days", "60" if mode=="seed" else "2"))
    chunk = int(opts.get("chunk_days", "7" if mode=="seed" else "1"))
    discover = opts.get("discover","webhooks")  # webhooks | universe | none
    lookback_discovery = int(opts.get("discover_lookback_days","14"))
    universe_name = opts.get("universe","largecaps_v1")
    snapshot = opts.get("snapshot","false").lower() in {"1","true","yes","on"}

    # Symbols: explicit args > discovery > default sample
    symbols = [s.upper() for s in args]
    if not symbols:
        if discover == "webhooks":
            symbols = discover_symbols("webhooks", lookback_days=lookback_discovery)
        elif discover == "universe":
            symbols = discover_symbols("universe", universe_name=universe_name)
        else:
            symbols = ["RELIANCE","HDFCBANK","INFY","TCS","ICICIBANK"]

    if snapshot:
        n = snapshot_universe(symbols, universe_name=universe_name, source=discover)
        print(f"📌 Universe snapshot '{universe_name}': {n} symbols recorded")

    if mode == "seed":
        out = backfill_symbols(symbols, interval=interval, lookback_days=lookback, chunk_days=chunk)
    else:
        out = sync_today(symbols, interval=interval, lookback_days_today=lookback, chunk_days=chunk)

    total_s = sum(v[0] for v in out.values())
    total_f = sum(v[1] for v in out.values())
    print(f"TOTAL: spot={total_s}, fut={total_f} across {len(symbols)} symbols")
