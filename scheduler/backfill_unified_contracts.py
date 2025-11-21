#!/usr/bin/env python3
"""
backfill_unified_contracts.py

Read 'contracts_YYYY_MM_DD*.xlsx' files that have the columns:

SYMBOL, OPTION TYPE, STRIKE PRICE, PRICE, SPOT, EXPIRY, LAST UPDATED, BUILD UP, LOT SIZE,
DAY CHANGE, %DAY CHANGE, OPEN PRICE, HIGH PRICE, LOW PRICE, PREV CLOSE PRICE,
OI, %OI CHANGE, OI CHANGE, PREV DAY OI, TRADED CONTRACTS, TRADED CONTRACTS CHANGE%,
SHARES TRADED, %VOLUME SHARES CHANGE, PREV DAY VOL, BASIS, COST OF CARRY (CoC),
IV, PREV DAY IV, %IV CHANGE, DELTA, VEGA, GAMMA, THETA, RHO

Futures rows → raw_ingest.daily_futures
Options  rows → raw_ingest.daily_options

Environment:
  ALPHA_CFG (optional) – points to a config.ini with [postgres] section
"""

import os, re, glob, argparse, datetime as dt
import pandas as pd
from configparser import ConfigParser, ParsingError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from pathlib import Path

# ---------------------- CONFIG / DB ----------------------

def load_ini():
    path = os.getenv("ALPHA_CFG", os.path.join(os.path.dirname(__file__), "config.ini"))
    cp = ConfigParser(inline_comment_prefixes=("#",";"))
    if not os.path.exists(path):
        return cp
    for enc in ("utf-8","utf-8-sig","cp1252","latin-1"):
        try:
            cp.read(path, encoding=enc)
            return cp
        except (UnicodeDecodeError, ParsingError):
            continue
    with open(path, "rb") as f:
        raw = f.read()
    cp.read_string(raw.decode("utf-8", errors="replace"))
    return cp

CFG = load_ini()

def make_engine():
    if CFG.has_section("postgres"):
        pg = dict(CFG.items("postgres"))
        url = URL.create(
            "postgresql+psycopg2",
            username=pg.get("user","postgres"),
            password=pg.get("password",""),
            host=pg.get("host","127.0.0.1"),
            port=int(pg.get("port","5432")),
            database=pg.get("dbname","TradeHub18"),
        )
        sp = pg.get("search_path","raw_ingest")
    else:
        url = URL.create(
            "postgresql+psycopg2",
            username=os.getenv("PGUSER","postgres"),
            password=os.getenv("PGPASSWORD",""),
            host=os.getenv("PGHOST","127.0.0.1"),
            port=int(os.getenv("PGPORT","5432")),
            database=os.getenv("PGDATABASE","TradeHub18"),
        )
        sp = os.getenv("PGSEARCH_PATH","raw_ingest")

    eng = create_engine(url, future=True, pool_pre_ping=True,
                        connect_args={"options": f"-c search_path={sp}"})
    with eng.begin() as cx:
        cx.execute(text(f"CREATE SCHEMA IF NOT EXISTS {sp}"))
        cx.execute(text(f"SET search_path TO {sp}"))
    print(f"[db] connected → {url.host}:{url.port}/{url.database} (search_path={sp})")
    return eng, sp

ENG, SCHEMA = make_engine()

# ---------------------- SCHEMA ----------------------

CREATE_FUTURES = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.daily_futures (
  trade_date      date        NOT NULL,
  symbol          text        NOT NULL,
  expiry          date,
  last_updated    timestamptz,
  lot_size        numeric,
  price           numeric,
  spot            numeric,
  open_price      numeric,
  high_price      numeric,
  low_price       numeric,
  prev_close      numeric,
  day_change      numeric,
  day_change_pct  numeric,
  basis           numeric,
  coc             numeric,           -- cost of carry
  buildup         text,
  oi              numeric,
  oi_change       numeric,
  oi_change_pct   numeric,
  prev_day_oi     numeric,
  traded_contracts        numeric,
  traded_contracts_chg_pct numeric,
  shares_traded          numeric,
  volume_shares_chg_pct  numeric,
  prev_day_vol          numeric,
  iv              numeric,
  iv_prev         numeric,
  iv_change_pct   numeric,
  delta           numeric,
  vega            numeric,
  gamma           numeric,
  theta           numeric,
  rho             numeric,
  PRIMARY KEY (trade_date, symbol, expiry)
);
"""

CREATE_OPTIONS = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.daily_options (
  trade_date      date        NOT NULL,
  symbol          text        NOT NULL,
  option_type     text        NOT NULL,  -- 'CE' or 'PE'
  strike_price    numeric     NOT NULL,
  expiry          date,
  last_updated    timestamptz,
  lot_size        numeric,
  price           numeric,
  spot            numeric,
  open_price      numeric,
  high_price      numeric,
  low_price       numeric,
  prev_close      numeric,
  day_change      numeric,
  day_change_pct  numeric,
  buildup         text,
  oi              numeric,
  oi_change       numeric,
  oi_change_pct   numeric,
  prev_day_oi     numeric,
  traded_contracts        numeric,
  traded_contracts_chg_pct numeric,
  shares_traded          numeric,
  volume_shares_chg_pct  numeric,
  prev_day_vol          numeric,
  iv              numeric,
  iv_prev         numeric,
  iv_change_pct   numeric,
  delta           numeric,
  vega            numeric,
  gamma           numeric,
  theta           numeric,
  rho             numeric,
  PRIMARY KEY (trade_date, symbol, option_type, strike_price, expiry)
);
"""

with ENG.begin() as cx:
    cx.execute(text(CREATE_FUTURES))
    cx.execute(text(CREATE_OPTIONS))
print("[db] target tables ready.")

# ---------------------- HELPERS ----------------------

def infer_trade_date_from_name(path: str) -> dt.date | None:
    name = Path(path).name
    m = re.search(r'(\d{4})[_-](\d{2})[_-](\d{2})', name)  # contracts_2025_10_16
    if m:
        y, mm, dd = map(int, m.groups())
        try: return dt.date(y, mm, dd)
        except: pass
    return None

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def strip_pct(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace("%","", regex=False)
              .str.replace(",","", regex=False)
              .str.strip()
              .pipe(pd.to_numeric, errors="coerce"))

def parse_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def read_any_excel(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, sheet_name=0)

# ---------------------- NORMALIZE ----------------------

COLMAP = {
    "SYMBOL": "symbol",
    "OPTION TYPE": "option_type",
    "STRIKE PRICE": "strike_price",
    "PRICE": "price",
    "SPOT": "spot",
    "EXPIRY": "expiry",
    "LAST UPDATED": "last_updated",
    "BUILD UP": "buildup",
    "LOT SIZE": "lot_size",
    "DAY CHANGE": "day_change",
    "%DAY CHANGE": "day_change_pct",
    "OPEN PRICE": "open_price",
    "HIGH PRICE": "high_price",
    "LOW PRICE": "low_price",
    "PREV CLOSE PRICE": "prev_close",
    "OI": "oi",
    "%OI CHANGE": "oi_change_pct",
    "OI CHANGE": "oi_change",
    "PREV DAY OI": "prev_day_oi",
    "TRADED CONTRACTS": "traded_contracts",
    "TRADED CONTRACTS CHANGE%": "traded_contracts_chg_pct",
    "SHARES TRADED": "shares_traded",
    "%VOLUME SHARES CHANGE": "volume_shares_chg_pct",
    "PREV DAY VOL": "prev_day_vol",
    "BASIS": "basis",
    "COST OF CARRY (CoC)": "coc",
    "IV": "iv",
    "PREV DAY IV": "iv_prev",
    "%IV CHANGE": "iv_change_pct",
    "DELTA": "delta",
    "VEGA": "vega",
    "GAMMA": "gamma",
    "THETA": "theta",
    "RHO": "rho",
}

def normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # standardize headers
    df.columns = [c.strip() for c in df.columns]
    rename = {c: COLMAP.get(c, c) for c in df.columns}
    df = df.rename(columns=rename)

    # normalize symbol / option_type
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    if "option_type" in df.columns:
        df["option_type"] = (df["option_type"].astype(str).str.upper()
                               .str.replace("FUTURES","FUT").str.replace("FUTURE","FUT"))

    # numerics
    pct_cols = {"day_change_pct","oi_change_pct","traded_contracts_chg_pct",
                "volume_shares_chg_pct","iv_change_pct"}
    for c in df.columns:
        if c in {"symbol","option_type","buildup"}:
            continue
        if c == "last_updated":
            df[c] = parse_ts(df[c])
        elif c == "expiry":
            df[c] = parse_date(df[c])
        elif c in pct_cols:
            df[c] = strip_pct(df[c])
        else:
            df[c] = to_num(df[c])

    return df

# ---------------------- UPSERTS ----------------------

def upsert_futures(df: pd.DataFrame, trade_date: dt.date) -> int:
    if df.empty: return 0
    df = df.copy()
    df["trade_date"] = trade_date
    cols = [
        "trade_date","symbol","expiry","last_updated","lot_size","price","spot",
        "open_price","high_price","low_price","prev_close",
        "day_change","day_change_pct","basis","coc","buildup",
        "oi","oi_change","oi_change_pct","prev_day_oi",
        "traded_contracts","traded_contracts_chg_pct","shares_traded","volume_shares_chg_pct",
        "prev_day_vol","iv","iv_prev","iv_change_pct","delta","vega","gamma","theta","rho"
    ]
    df = df[cols].dropna(subset=["symbol"]).drop_duplicates(subset=["trade_date","symbol","expiry"], keep="last")
    tmp = "tmp_daily_fut"
    with ENG.begin() as cx:
        cx.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
        cx.execute(text(f"CREATE TEMP TABLE {tmp} AS SELECT * FROM {SCHEMA}.daily_futures WHERE 1=0"))
    df.to_sql(tmp, ENG, if_exists="append", index=False, method="multi", chunksize=2000)
    merge = f"""
        INSERT INTO {SCHEMA}.daily_futures
        SELECT * FROM {tmp}
        ON CONFLICT (trade_date, symbol, expiry) DO UPDATE SET
          last_updated=EXCLUDED.last_updated,
          lot_size=EXCLUDED.lot_size,
          price=EXCLUDED.price,
          spot=EXCLUDED.spot,
          open_price=EXCLUDED.open_price,
          high_price=EXCLUDED.high_price,
          low_price=EXCLUDED.low_price,
          prev_close=EXCLUDED.prev_close,
          day_change=EXCLUDED.day_change,
          day_change_pct=EXCLUDED.day_change_pct,
          basis=EXCLUDED.basis,
          coc=EXCLUDED.coc,
          buildup=EXCLUDED.buildup,
          oi=EXCLUDED.oi,
          oi_change=EXCLUDED.oi_change,
          oi_change_pct=EXCLUDED.oi_change_pct,
          prev_day_oi=EXCLUDED.prev_day_oi,
          traded_contracts=EXCLUDED.traded_contracts,
          traded_contracts_chg_pct=EXCLUDED.traded_contracts_chg_pct,
          shares_traded=EXCLUDED.shares_traded,
          volume_shares_chg_pct=EXCLUDED.volume_shares_chg_pct,
          prev_day_vol=EXCLUDED.prev_day_vol,
          iv=EXCLUDED.iv,
          iv_prev=EXCLUDED.iv_prev,
          iv_change_pct=EXCLUDED.iv_change_pct,
          delta=EXCLUDED.delta,
          vega=EXCLUDED.vega,
          gamma=EXCLUDED.gamma,
          theta=EXCLUDED.theta,
          rho=EXCLUDED.rho;
    """
    with ENG.begin() as cx:
        cx.execute(text(merge))
        n = cx.execute(text(f"SELECT COUNT(*) FROM {tmp}")).scalar()
        cx.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
    return int(n or 0)

def upsert_options(df: pd.DataFrame, trade_date: dt.date) -> int:
    if df.empty: return 0
    df = df.copy()
    df["trade_date"] = trade_date
    cols = [
        "trade_date","symbol","option_type","strike_price","expiry","last_updated","lot_size",
        "price","spot","open_price","high_price","low_price","prev_close",
        "day_change","day_change_pct","buildup",
        "oi","oi_change","oi_change_pct","prev_day_oi",
        "traded_contracts","traded_contracts_chg_pct","shares_traded","volume_shares_chg_pct",
        "prev_day_vol","iv","iv_prev","iv_change_pct","delta","vega","gamma","theta","rho"
    ]
    df = df[cols].dropna(subset=["symbol","option_type","strike_price"]).drop_duplicates(
        subset=["trade_date","symbol","option_type","strike_price","expiry"], keep="last")
    tmp = "tmp_daily_opt"
    with ENG.begin() as cx:
        cx.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
        cx.execute(text(f"CREATE TEMP TABLE {tmp} AS SELECT * FROM {SCHEMA}.daily_options WHERE 1=0"))
    df.to_sql(tmp, ENG, if_exists="append", index=False, method="multi", chunksize=2000)
    merge = f"""
        INSERT INTO {SCHEMA}.daily_options
        SELECT * FROM {tmp}
        ON CONFLICT (trade_date, symbol, option_type, strike_price, expiry) DO UPDATE SET
          last_updated=EXCLUDED.last_updated,
          lot_size=EXCLUDED.lot_size,
          price=EXCLUDED.price,
          spot=EXCLUDED.spot,
          open_price=EXCLUDED.open_price,
          high_price=EXCLUDED.high_price,
          low_price=EXCLUDED.low_price,
          prev_close=EXCLUDED.prev_close,
          day_change=EXCLUDED.day_change,
          day_change_pct=EXCLUDED.day_change_pct,
          buildup=EXCLUDED.buildup,
          oi=EXCLUDED.oi,
          oi_change=EXCLUDED.oi_change,
          oi_change_pct=EXCLUDED.oi_change_pct,
          prev_day_oi=EXCLUDED.prev_day_oi,
          traded_contracts=EXCLUDED.traded_contracts,
          traded_contracts_chg_pct=EXCLUDED.traded_contracts_chg_pct,
          shares_traded=EXCLUDED.shares_traded,
          volume_shares_chg_pct=EXCLUDED.volume_shares_chg_pct,
          prev_day_vol=EXCLUDED.prev_day_vol,
          iv=EXCLUDED.iv,
          iv_prev=EXCLUDED.iv_prev,
          iv_change_pct=EXCLUDED.iv_change_pct,
          delta=EXCLUDED.delta,
          vega=EXCLUDED.vega,
          gamma=EXCLUDED.gamma,
          theta=EXCLUDED.theta,
          rho=EXCLUDED.rho;
    """
    with ENG.begin() as cx:
        cx.execute(text(merge))
        n = cx.execute(text(f"SELECT COUNT(*) FROM {tmp}")).scalar()
        cx.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
    return int(n or 0)

# ---------------------- MAIN ----------------------

def main():
    ap = argparse.ArgumentParser(description="Backfill raw_ingest daily_futures / daily_options from contracts files")
    ap.add_argument("--inbox", default="./inbox", help="Folder containing Excel files")
    ap.add_argument("--pattern", default="contracts_*.xlsx", help="Glob to match (e.g. contracts_2025_10_16*.xlsx)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.inbox) / args.pattern)))
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        print(f"[run] no files for pattern {args.pattern}")
        return

    total_fut = total_opt = 0
    for f in files:
        try:
            td = infer_trade_date_from_name(f) or dt.date.today()
            df_raw = read_any_excel(f)
            df = normalize(df_raw)

            fut_mask = df["option_type"].fillna("").str.contains(r"^FUT", na=False)
            opt_mask = df["option_type"].fillna("").isin(["CE","PE"])

            fut = df.loc[fut_mask].copy()
            opt = df.loc[opt_mask].copy()

            n1 = upsert_futures(fut, td) if not fut.empty else 0
            n2 = upsert_options(opt, td) if not opt.empty else 0
            total_fut += n1; total_opt += n2
            print(f"[ok] {Path(f).name} @ {td}: futures={n1}, options={n2}")
        except Exception as e:
            print(f"[warn] {Path(f).name} failed → {e}")

    print(f"[done] inserted/merged: futures={total_fut}, options={total_opt}")

if __name__ == "__main__":
    main()
