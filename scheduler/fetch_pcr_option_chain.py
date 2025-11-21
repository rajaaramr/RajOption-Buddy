# scheduler/fetch_pcr_option_chain.py
from __future__ import annotations
import os, sys, time
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, List, Tuple

# make imports bulletproof whether run as module or script
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from utils.kite_utils import load_kite
from utils.db import get_db_connection  # must return psycopg2-like connection

TZ = timezone.utc
__all__ = ["fetch_option_chain_snapshot", "compute_pcr_from_rows", "run_pcr_snapshot"]

# ---------- helpers ----------
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
    today = on_or_after or datetime.now(TZ).date()
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
            itype  = (ins.get("instrument_type") or "").upper()
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

# ---------- main snapshot ----------
def fetch_option_chain_snapshot(symbol: str, expiry: Optional[date]=None, skip_dead: bool=True) -> List[Dict[str, Any]]:
    """
    Returns rows: {tradingsymbol, strike, type, oi, ltp, volume}
    Uses Kite.quote() which carries 'oi' for derivatives.
    """
    kite = load_kite()
    instruments = kite.instruments("NFO") or []
    if expiry is None:
        expiry = _nearest_expiry(instruments, symbol)
        if not expiry:
            return []

    chain = _filter_chain_for_expiry(instruments, symbol, expiry)
    if not chain:
        return []

    tokens = [f"NFO:{r['tradingsymbol']}" for r in chain]
    rows: List[Dict[str, Any]] = []

    for batch in _chunks(tokens, 200):
        q = kite.quote(batch)  # richer than ltp(); contains 'oi' for F&O
        for k in batch:
            info = q.get(k, {})
            ts = k.split(":",1)[1]
            base = next((r for r in chain if r["tradingsymbol"] == ts), None)
            if not base: continue
            oi  = int(info.get("oi") or 0)
            ltp = float(info.get("last_price") or 0.0)
            vol = int(info.get("volume") or 0)
            if skip_dead and oi <= 0 and vol <= 0:
                continue
            rows.append({"tradingsymbol": ts, "strike": base["strike"], "type": base["type"], "oi": oi, "ltp": ltp, "volume": vol})
        time.sleep(0.15)  # be gentle

    return rows

def compute_pcr_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    cum_call_oi = sum(r["oi"] for r in rows if r["type"] == "CE")
    cum_put_oi  = sum(r["oi"] for r in rows if r["type"] == "PE")
    pcr = (cum_put_oi / cum_call_oi) if cum_call_oi > 0 else None
    return {"cum_call_oi": int(cum_call_oi), "cum_put_oi": int(cum_put_oi), "pcr": (float(pcr) if pcr is not None else None)}

# ---------- DB ----------
def ensure_tables():
    sql = """
    CREATE TABLE IF NOT EXISTS options_pcr (
        id BIGSERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        expiry_date DATE NOT NULL,
        cum_call_oi BIGINT NOT NULL,
        cum_put_oi BIGINT NOT NULL,
        pcr DOUBLE PRECISION,
        computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        run_id TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS uq_options_pcr ON options_pcr(symbol, expiry_date, run_id);
    CREATE INDEX IF NOT EXISTS idx_options_pcr_symbol_date ON options_pcr(symbol, expiry_date);
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)

def upsert_pcr(symbol: str, expiry: date, stats: Dict[str, Any], run_id: Optional[str]=None):
    ensure_tables()
    sql = """
    INSERT INTO options_pcr (symbol, expiry_date, cum_call_oi, cum_put_oi, pcr, run_id)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (symbol, expiry_date, run_id)
    DO UPDATE SET
      cum_call_oi = EXCLUDED.cum_call_oi,
      cum_put_oi  = EXCLUDED.cum_put_oi,
      pcr         = EXCLUDED.pcr,
      computed_at = NOW();
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol.upper(), expiry, stats["cum_call_oi"], stats["cum_put_oi"], stats["pcr"], run_id))

# ---------- public runner ----------
def run_pcr_snapshot(symbol: str, expiry: Optional[str]=None, run_id: Optional[str]=None) -> Dict[str, Any]:
    kite = load_kite()
    instruments = kite.instruments("NFO") or []
    exp_date = _as_date(expiry) if expiry else _nearest_expiry(instruments, symbol)
    if not exp_date:
        raise RuntimeError(f"No valid expiry for {symbol}")
    rows = fetch_option_chain_snapshot(symbol, exp_date)
    stats = compute_pcr_from_rows(rows)
    upsert_pcr(symbol, exp_date, stats, run_id=run_id)
    return {"symbol": symbol.upper(), "expiry": str(exp_date), **stats, "rows": len(rows)}

if __name__ == "__main__":
    rid = datetime.now(TZ).strftime("%Y%m%dT%H%M%SZ")
    print(run_pcr_snapshot("BHARTIARTL", None, run_id=rid))
