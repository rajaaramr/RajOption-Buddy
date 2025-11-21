# load_spot_logger15m.py
# Parse "OHLCV_Logger_15m_*_NSE_<SYMBOL>_YYYY-MM-DD" files and upsert into spot.candles.
import os, re, glob, sys
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+, pip install tzdata if needed on Windows
import psycopg2
import psycopg2.extras as pgx
import openpyxl  # pip install openpyxl
from utils.db_ops import insert_spot_price

# --- CONFIG ---
DATA_DIR = r"E:\RajOptionBuddy\data\spot"
GLOB_PATTERN = "OHLCV_Logger_15m*"
PG_DSN = "host=localhost port=5432 dbname=TradeHub18 user=postgres password=Ajantha@18"
IST = ZoneInfo("Asia/Kolkata")

# Regex to pull SYMBOL from file name: ..._NSE_<SYMBOL>_YYYY...
SYM_RE = re.compile(r"_NSE_([A-Za-z0-9]+)_")

# Regex to extract key=val pairs from the 4th column (O=..., H=..., etc.)
KV_RE = re.compile(r"\b([A-Z_]+)=([0-9NaN\.\-]+)")

# put near your other imports
import re
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")

# robust regex: find the timestamp anywhere in the line
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)")
# key/value pairs like O=6134.5 H=6155.5 ... V=27596 (NaN allowed)
KV_RE = re.compile(r"\b([A-Z_]+)=([0-9]+(?:\.[0-9]+)?|NaN)\b", re.IGNORECASE)

def parse_line(line: str):
    """
    Extract ts (IST -> UTC) and O,H,L,C,V from a noisy 'Logger_15m' line.
    Returns (ts_utc, o,h,l,c,v) or None to skip the line.
    """
    if not line or not line.strip():
        return None

    # 1) timestamp anywhere in the line
    m = TS_RE.search(line)
    if not m:
        return None
    ts_str = m.group(1)

    # parse with or without seconds
    fmt = "%Y-%m-%d %H:%M:%S" if len(ts_str) == 19 else "%Y-%m-%d %H:%M"
    try:
        ts_local = datetime.strptime(ts_str, fmt).replace(tzinfo=IST)
    except Exception:
        return None
    ts_utc = ts_local.astimezone(ZoneInfo("UTC"))

    # 2) pull key/vals from the whole line
    kv = {k.upper(): v for k, v in KV_RE.findall(line)}

    def fnum(key, cast=float, default=None):
        raw = kv.get(key)
        if raw is None or str(raw).lower() == "nan":
            return default
        try:
            return cast(raw)
        except Exception:
            return default

    o = fnum("O", float)
    h = fnum("H", float)
    l = fnum("L", float)
    c = fnum("C", float)
    v = fnum("V", int, 0)

    # need at least OHLC
    if None in (o, h, l, c):
        return None

    # optional: enforce true 15-minute boundaries (00/15/30/45)
    if ts_utc.minute % 15 != 0:
        return None

    return ts_utc, o, h, l, c, v


def upsert_rows(conn, symbol: str, rows):
    """
    rows: list of tuples (symbol, ts_utc, open, high, low, close, volume, source)
    Uses centralized insert_spot_price (commits internally). Returns rows written.
    """
    if not rows:
        return 0

    wrote = 0
    for (_sym, ts_utc, o, h, l, c, v, src) in rows:
        insert_spot_price(
            ts=ts_utc,
            symbol=symbol,
            open=float(o), high=float(h), low=float(l), close=float(c),
            volume=float(v),
            interval="15m",
            source=str(src) if src else "logger15m",
        )
        wrote += 1
    return wrote

def iter_excel_lines(path: str):
    """
    Yield 'synthetic lines' from an xlsx/xls so parse_line() can stay unchanged.
    We join non-empty cell values with a space, e.g.,
      "1 Exit long 2025-06-06 10:00 EXIT O=... H=... L=... C=... V=..."
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    for row in ws.iter_rows(values_only=True):
        # convert row tuple to a single string
        parts = [str(c).strip() for c in row if c is not None and str(c).strip() != ""]
        if parts:
            yield " ".join(parts)

def load_file(conn, path: str):
    base = os.path.basename(path)

    # symbol from filename ..._NSE_<SYMBOL>_YYYY...
    m = SYM_RE.search(base)
    if not m:
        print(f"⚠️  Skip (symbol not found in name): {base}")
        return 0, None
    symbol = m.group(1).upper()

    # choose reader based on extension
    ext = os.path.splitext(base)[1].lower()
    if ext in (".xlsx", ".xls"):
        line_iter = iter_excel_lines(path)
    else:
        # text mode
        def _iter_text():
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    yield line
        line_iter = _iter_text()

    # last-wins de-dup per timestamp
    bucket = {}  # ts_utc -> tuple for insert
    bad = 0
    for line in line_iter:
        parsed = parse_line(line)
        if not parsed:
            bad += 1
            continue
        ts_utc, o, h, l, c, v = parsed
        bucket[ts_utc] = (symbol, ts_utc, o, h, l, c, v, "logger15m")

    rows = [bucket[k] for k in sorted(bucket.keys())]
    if not rows:
        print(f"⚠️  No parseable rows in {base} (skipped {bad})")
        return 0, symbol

    wrote = upsert_rows(conn, symbol, rows)
    print(f"✅ {symbol}: wrote {wrote} rows from {base} (skipped {bad})")
    return wrote, symbol

def main():
    only_symbol = None
    if len(sys.argv) > 1:
        only_symbol = sys.argv[1].upper()

    files = sorted(glob.glob(os.path.join(DATA_DIR, GLOB_PATTERN)))
    if not files:
        print(f"No files matched: {DATA_DIR}\\{GLOB_PATTERN}")
        return

    conn = psycopg2.connect(PG_DSN)
    total = 0
    touched = set()
    try:
        for p in files:
            if only_symbol:
                # only load files whose extracted symbol matches
                base = os.path.basename(p)
                m = SYM_RE.search(base)
                if not (m and m.group(1).upper() == only_symbol):
                    continue
            wrote, sym = load_file(conn, p)
            total += wrote
            if sym:
                touched.add(sym)
    finally:
        conn.close()

    print(f"\nTOTAL rows: {total} across {len(touched)} symbols: {', '.join(sorted(touched))}")

if __name__ == "__main__":
    main()
