from __future__ import annotations
import os, sys, time
from datetime import datetime, timezone, timedelta, date
from typing import Optional, Dict, Any, List, Tuple

# --- make imports bulletproof whether run as module or script ---
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# ✅ ABSOLUTE imports only; DO NOT import from scheduler.fetch_prices here (avoid circulars)
from utils.kite_utils import load_kite
from utils.db import get_db_connection

__all__ = ["fetch_and_store_option_chain"]  # <-- export exactly this name

TZ_UTC = timezone.utc
TZ_IST = timezone(timedelta(hours=5, minutes=30))

# ---------------- helpers ----------------
def _as_date(x) -> Optional[date]:
    from datetime import datetime as _dt, date as _d
    if x is None: return None
    if isinstance(x, _d) and not isinstance(x, _dt): return x
    if isinstance(x, _dt): return x.date()
    try:
        return _dt.fromisoformat(str(x)).date()
    except Exception:
        return None

def _underlying_matches_symbol(ins: Dict[str, Any], symbol: str) -> bool:
    sym = symbol.upper()
    name = (ins.get("name") or "").upper()
    tsym = (ins.get("tradingsymbol") or "").upper()
    return (name == sym) or tsym.startswith(sym)

def _nearest_expiry(instruments: List[Dict[str, Any]], symbol: str, on_or_after: Optional[date]=None) -> Optional[date]:
    today = on_or_after or datetime.now(TZ_UTC).date()
    exps: List[date] = []
    for ins in instruments:
        try:
            if ins.get("segment") != "NFO-OPT": continue
            if not _underlying_matches_symbol(ins, symbol): continue
            exp = _as_date(ins.get("expiry"))
            if exp and exp >= today: exps.append(exp)
        except Exception:
            continue
    return sorted(set(exps))[0] if exps else None

def _filter_chain_for_expiry(instruments: List[Dict[str, Any]], symbol: str, expiry: date) -> List[Dict[str, Any]]:
    chain = []
    for ins in instruments:
        try:
            if ins.get("segment") != "NFO-OPT": continue
            if not _underlying_matches_symbol(ins, symbol): continue
            exp = _as_date(ins.get("expiry"))
            if exp != expiry: continue
            strike = float(ins.get("strike", 0) or 0)
            itype  = (ins.get("instrument_type") or "").upper()  # CE/PE
            tsym   = ins.get("tradingsymbol")
            if strike > 0 and itype in ("CE","PE") and tsym:
                chain.append({"tradingsymbol": tsym, "strike": strike, "type": itype})
        except Exception:
            continue
    chain.sort(key=lambda r: (r["strike"], r["type"]))
    return chain

def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _ensure_chain_table():
    sql = """
    CREATE TABLE IF NOT EXISTS market.options_chain_snapshots (
        symbol        TEXT NOT NULL,
        expiry_date   DATE NOT NULL,
        tradingsymbol TEXT NOT NULL,
        strike        DOUBLE PRECISION NOT NULL,
        type          TEXT NOT NULL,           -- 'CE'|'PE'
        oi            BIGINT NOT NULL,
        ltp           DOUBLE PRECISION,
        volume        BIGINT,
        snapshot_ts   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        snapshot_date DATE GENERATED ALWAYS AS ((snapshot_ts AT TIME ZONE 'Asia/Kolkata')::date) STORED
    );
    CREATE INDEX IF NOT EXISTS idx_opt_chain_day
      ON market.options_chain_snapshots(symbol, expiry_date, snapshot_date);
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql); conn.commit()

# ---------------- the function your orchestrator imports ----------------
def fetch_and_store_option_chain(symbol: str, kite=None, strike_range: int = 5) -> int:
    """
    Fetch nearest-expiry option chain via quote(), optionally narrow to ATM±strike_range,
    and store rows in market.options_chain_snapshots. Returns rows written (int).
    """
    # Keep this import-time safe. Do not raise at module import.
    if kite is None:
        kite = load_kite()

    instruments = kite.instruments("NFO") or []
    expiry = _nearest_expiry(instruments, symbol)
    if not expiry:
        return 0

    chain = _filter_chain_for_expiry(instruments, symbol, expiry)
    if not chain:
        return 0

    # If you want ATM band, we do a cheap strike-step band around median as a fallback
    if strike_range and strike_range > 0 and len({c["strike"] for c in chain}) > 3:
        strikes = sorted({c["strike"] for c in chain})
        # crude ATM: mid strike; optional: pull spot and pick nearest
        mid = strikes[len(strikes)//2]
        step = min((abs(strikes[i+1]-strikes[i]) for i in range(len(strikes)-1)), default=100.0)
        lo = mid - step*strike_range
        hi = mid + step*strike_range
        chain = [c for c in chain if lo <= c["strike"] <= hi]

    tokens = [f"NFO:{r['tradingsymbol']}" for r in chain]
    rows: List[Dict[str, Any]] = []
    for batch in _chunks(tokens, 200):
        q = kite.quote(batch)  # has 'oi' for F&O
        for k in batch:
            info = q.get(k, {})
            ts = k.split(":",1)[1]
            base = next((r for r in chain if r["tradingsymbol"] == ts), None)
            if not base: continue
            oi  = int(info.get("oi") or 0)
            ltp = float(info.get("last_price") or 0.0)
            vol = int(info.get("volume") or 0)
            if oi <= 0 and vol <= 0:
                continue
            rows.append({"tradingsymbol": ts, "strike": base["strike"], "type": base["type"], "oi": oi, "ltp": ltp, "volume": vol})
        time.sleep(0.15)

    if not rows:
        return 0

    _ensure_chain_table()
    written = 0
    from datetime import date as _date
    snap_date = datetime.now(TZ_IST).date()

    with get_db_connection() as conn, conn.cursor() as cur:
        for r in rows:
            # app-layer idempotency (safe with compressed hypertables)
            cur.execute("""
                SELECT 1
                  FROM market.options_chain_snapshots
                 WHERE symbol=%s AND expiry_date=%s AND strike=%s AND type=%s AND snapshot_date=%s
                 LIMIT 1
            """, (symbol.upper(), expiry, float(r["strike"]), r["type"], snap_date))
            exists = cur.fetchone() is not None

            if exists:
                cur.execute("""
                    UPDATE market.options_chain_snapshots
                       SET oi     = GREATEST(%s, oi),
                           ltp    = %s,
                           volume = GREATEST(%s, volume),
                           snapshot_ts = NOW()
                     WHERE symbol=%s AND expiry_date=%s AND strike=%s AND type=%s AND snapshot_date=%s
                """, (r["oi"], r["ltp"], r["volume"],
                      symbol.upper(), expiry, float(r["strike"]), r["type"], snap_date))
            else:
                cur.execute("""
                    INSERT INTO market.options_chain_snapshots
                      (symbol, expiry_date, tradingsymbol, strike, type, oi, ltp, volume)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (symbol.upper(), expiry, r["tradingsymbol"], float(r["strike"]), r["type"], r["oi"], r["ltp"], r["volume"]))
            written += 1
        conn.commit()

    return written

if __name__ == "__main__":
    # sanity run
    n = fetch_and_store_option_chain("BHARTIARTL", strike_range=3)
    print({"rows": n})
