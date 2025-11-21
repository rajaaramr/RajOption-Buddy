#!/usr/bin/env python3
# ingest_option_chain.py — load Futures + nearest CE/PE (File B) and publish simple rollups

import os, re, sys, glob, argparse, datetime as dt
from pathlib import Path
import pandas as pd

from configparser import ConfigParser, ParsingError
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# ---------------------- CONFIG / DB ----------------------

def load_ini():
    """
    Loads config.ini (UTF-8 / cp1252 tolerant). Env var ALPHA_CFG can override path.
    Expected INI (same as your other ETL):
      [postgres]
      host=127.0.0.1
      port=5432
      dbname=TradeHub18
      user=postgres
      password=xxxx
      search_path=raw_ingest
    """
    path = os.getenv("ALPHA_CFG", os.path.join(os.path.dirname(__file__), "config.ini"))
    cp = ConfigParser(inline_comment_prefixes=("#",";"))
    if not os.path.exists(path):
        print(f"[cfg] INI not found at {path}; falling back to env or defaults.")
        return cp
    for enc in ("utf-8","utf-8-sig","cp1252"):
        try:
            cp.read(path, encoding=enc)
            return cp
        except (UnicodeDecodeError, ParsingError):
            continue
    with open(path, "rb") as f:
        raw = f.read()
    txt = raw.decode("utf-8", errors="replace")
    cp.read_string(txt)
    print("[cfg] Loaded INI with errors='replace'. Consider resaving as UTF-8.")
    return cp

CFG = load_ini()

def make_engine():
    if CFG.has_section("postgres"):
        pg = {k:v for k,v in CFG.items("postgres")}
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
        cx.execute(text("CREATE SCHEMA IF NOT EXISTS derivatives"))
        cx.execute(text("""
            CREATE TABLE IF NOT EXISTS derivatives.option_chain (
              trade_date      date        NOT NULL,
              symbol          text        NOT NULL,
              instrument      text        NOT NULL,            -- FUT | OPT
              option_type     text        NULL,                -- CE | PE | NULL for FUT
              strike          numeric     NULL,
              expiry          date        NULL,

              price           numeric     NULL,
              spot            numeric     NULL,
              last_updated    timestamptz NULL,
              build_up        text        NULL,
              lot_size        integer     NULL,

              day_change      numeric     NULL,
              pct_day_change  numeric     NULL,
              open_price      numeric     NULL,
              high_price      numeric     NULL,
              low_price       numeric     NULL,
              prev_close      numeric     NULL,

              oi              numeric     NULL,
              pct_oi_change   numeric     NULL,
              oi_change       numeric     NULL,
              prev_day_oi     numeric     NULL,

              traded_contracts           numeric NULL,
              traded_contracts_chg_pct   numeric NULL,
              shares_traded              numeric NULL,
              pct_volume_shares_change   numeric NULL,
              prev_day_vol               numeric NULL,

              basis           numeric     NULL,
              coc             numeric     NULL,                -- Cost of Carry
              iv              numeric     NULL,
              prev_day_iv     numeric     NULL,
              pct_iv_change   numeric     NULL,

              delta           numeric     NULL,
              vega            numeric     NULL,
              gamma           numeric     NULL,
              theta           numeric     NULL,
              rho             numeric     NULL,

              source_file     text        NULL,

              PRIMARY KEY (trade_date, symbol, instrument, option_type, expiry, strike)
            );
        """))
        # indicators.values (if not already present in your system)
        cx.execute(text("""
        CREATE SCHEMA IF NOT EXISTS indicators;
        CREATE TABLE IF NOT EXISTS indicators.values (
          symbol      text        NOT NULL,
          market_type text        NOT NULL,
          interval    text        NOT NULL,
          ts          timestamptz NOT NULL,
          metric      text        NOT NULL,
          val         numeric,
          context     jsonb DEFAULT '{}'::jsonb,
          run_id      text DEFAULT 'etl-opt',
          source      text DEFAULT 'engine',
          created_at  timestamptz DEFAULT now(),
          PRIMARY KEY (symbol, market_type, interval, ts, metric)
        );
        """))
    print(f"[db] connected to {url.host}:{url.port}/{url.database} (search_path={sp})")
    return eng, sp

ENG, SCHEMA = make_engine()

# ---------------------- HELPERS ----------------------

def infer_trade_date_from_name(path: str):
    name = Path(path).name.replace("_","-")
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', name)
    if m:
        y, mth, d = map(int, m.groups())
        try: return dt.date(y, mth, d)
        except: pass
    m = re.search(r'(\d{1,2})-([A-Za-z]{3,4})-(\d{4})', name)
    mon = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Sept":9,"Oct":10,"Nov":11,"Dec":12}
    if m and m.group(2) in mon:
        try: return dt.date(int(m.group(3)), mon[m.group(2)], int(m.group(1)))
        except: pass
    return None

def read_any_excel(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, sheet_name=0)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+"," ", c).strip() for c in df.columns]
    return df

def to_num(v):
    return pd.to_numeric(v, errors="coerce")

def coerce_series_percent(s: pd.Series) -> pd.Series:
    if s is None: return pd.Series(dtype=float)
    return (s.astype(str)
              .str.replace("%","", regex=False)
              .str.replace(",","", regex=False)
              .str.strip()
              .replace({"": None})
              .pipe(pd.to_numeric, errors="coerce"))

def parse_expiry(s):
    if pd.isna(s): return None
    if isinstance(s, dt.date): return s
    if isinstance(s, dt.datetime): return s.date()
    txt = str(s).strip()
    # Try common formats
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(txt, fmt).date()
        except Exception:
            continue
    return None

# Column mapping: canonical -> list of possible header names
MAP = {
  "symbol": ["SYMBOL","Symbol","NSE Code","NSE code"],
  "option_type": ["OPTION TYPE","Option Type","TYPE"],
  "strike": ["STRIKE PRICE","Strike","STRIKE"],
  "price": ["PRICE","Price","LTP","Last Price","Close"],
  "spot": ["SPOT","Spot"],
  "expiry": ["EXPIRY","Expiry"],
  "last_updated": ["LAST UPDATED","Last Updated","Timestamp"],
  "build_up": ["BUILD UP","BUILD-UP","Build Up","Build-Up"],
  "lot_size": ["LOT SIZE","Lot Size","Lot"],
  "day_change": ["DAY CHANGE","Day Change"],
  "pct_day_change": ["%DAY CHANGE","% Day Change"],
  "open_price": ["OPEN PRICE","Open"],
  "high_price": ["HIGH PRICE","High"],
  "low_price": ["LOW PRICE","Low"],
  "prev_close": ["PREV CLOSE PRICE","Prev Close","Previous Close"],
  "oi": ["OI","Open Interest"],
  "pct_oi_change": ["%OI CHANGE","% OI CHANGE","OI % Change"],
  "oi_change": ["OI CHANGE","Change in OI"],
  "prev_day_oi": ["PREV DAY OI","Prev OI"],
  "traded_contracts": ["TRADED CONTRACTS","Contracts Traded"],
  "traded_contracts_chg_pct": ["TRADED CONTRACTS CHANGE%","Contracts Change %"],
  "shares_traded": ["SHARES TRADED","Shares"],
  "pct_volume_shares_change": ["%VOLUME SHARES CHANGE","% Shares Change"],
  "prev_day_vol": ["PREV DAY VOL","Prev Volume"],
  "basis": ["BASIS"],
  "coc": ["COST OF CARRY (CoC)","CoC","COST OF CARRY"],
  "iv": ["IV","Implied Volatility"],
  "prev_day_iv": ["PREV DAY IV","Prev IV"],
  "pct_iv_change": ["%IV CHANGE","IV % Change"],
  "delta": ["DELTA"],
  "vega": ["VEGA"],
  "gamma": ["GAMMA"],
  "theta": ["THETA"],
  "rho": ["RHO"],
}

def pick_column(df, names):
    for want in names:
        for c in df.columns:
            if c.strip().lower() == want.strip().lower():
                return c
        for c in df.columns:
            if want.strip().lower() in c.strip().lower():
                return c
    return None

def normalize_frame(df_raw: pd.DataFrame, trade_date: dt.date, source_file: str) -> pd.DataFrame:
    df = clean_columns(df_raw)
    out = {}
    for canon, choices in MAP.items():
        col = pick_column(df, choices)
        if col is not None:
            out[canon] = df[col]
    out_df = pd.DataFrame(out)

    # Coerce types
    # symbol/option/build_up textual
    if "symbol" in out_df: out_df["symbol"] = out_df["symbol"].astype(str).str.strip().str.upper()
    # option_type text -> normalize
    if "option_type" in out_df:
        out_df["option_type"] = (
            out_df["option_type"].astype(str).str.strip().str.upper()
            .replace({
                "CALL": "CE", "CE": "CE",
                "PUT": "PE",  "PE": "PE",
                "FUT": "FUT", "FUTURE": "FUT", "FUTURES": "FUT", "FUT IDX": "FUT", "FUTSTK": "FUT"
            })
        )
    
# instrument detection
    out_df["instrument"] = out_df["option_type"].apply(lambda x: "FUT" if str(x).upper()=="FUT" else "OPT")

    # FUT rows: clear option_type and strike
    out_df.loc[out_df["instrument"]=="FUT", "option_type"] = None
    if "strike" in out_df:
        out_df["strike"] = pd.to_numeric(out_df["strike"])
    out_df.loc[out_df["instrument"]=="FUT", "strike"] = None


    # Numerics
    for c in ["price","spot","lot_size","day_change","open_price","high_price","low_price","prev_close",
              "oi","oi_change","prev_day_oi","traded_contracts","shares_traded","prev_day_vol",
              "basis","coc","iv","prev_day_iv","delta","vega","gamma","theta","rho"]:
        if c in out_df.columns: out_df[c] = to_num(out_df[c])

    # Percents
    for c in ["pct_day_change","pct_oi_change","traded_contracts_chg_pct",
              "pct_volume_shares_change","pct_iv_change"]:
        if c in out_df.columns: out_df[c] = coerce_series_percent(out_df[c])

    # Date-like
    if "expiry" in out_df.columns:
        out_df["expiry"] = out_df["expiry"].apply(parse_expiry)
    else:
        out_df["expiry"] = None

    if "last_updated" in out_df.columns:
        # Allow passthrough; DB column is timestamptz
        pass
    else:
        out_df["last_updated"] = None

    # FUT normalization: strike should be NULL
    out_df.loc[out_df["instrument"]=="FUT", "strike"] = None

    # Ensure mandatory columns exist
    out_df["trade_date"] = trade_date
    out_df["source_file"] = source_file

    # Minimal subset for table compatibility
    return out_df[[
        "trade_date","symbol","instrument","option_type","strike","expiry",
        "price","spot","last_updated","build_up","lot_size",
        "day_change","pct_day_change","open_price","high_price","low_price","prev_close",
        "oi","pct_oi_change","oi_change","prev_day_oi",
        "traded_contracts","traded_contracts_chg_pct","shares_traded","pct_volume_shares_change","prev_day_vol",
        "basis","coc","iv","prev_day_iv","pct_iv_change",
        "delta","vega","gamma","theta","rho","source_file"
    ]]

def upsert_option_chain(df: pd.DataFrame):
    if df is None or df.empty: return 0
    # Remove clearly empty symbol rows
    df = df.dropna(subset=["symbol"]).copy()
    tmp = "tmp_option_chain_stage"
    with ENG.begin() as cx:
        cx.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
        cx.execute(text(f"""
            CREATE TEMP TABLE {tmp}
            AS SELECT * FROM derivatives.option_chain WHERE 1=0
        """))
    df.to_sql(tmp, ENG, if_exists="append", index=False, method="multi", chunksize=5000)
    merge_sql = """
    INSERT INTO derivatives.option_chain (
      trade_date, symbol, instrument, option_type, expiry, strike,
      price, spot, last_updated, build_up, lot_size,
      day_change, pct_day_change, open_price, high_price, low_price, prev_close,
      oi, pct_oi_change, oi_change, prev_day_oi,
      traded_contracts, traded_contracts_chg_pct, shares_traded, pct_volume_shares_change, prev_day_vol,
      basis, coc, iv, prev_day_iv, pct_iv_change,
      delta, vega, gamma, theta, rho, source_file
    )
    SELECT
      trade_date, symbol, instrument, option_type, expiry, strike,
      price, spot, last_updated, build_up, lot_size,
      day_change, pct_day_change, open_price, high_price, low_price, prev_close,
      oi, pct_oi_change, oi_change, prev_day_oi,
      traded_contracts, traded_contracts_chg_pct, shares_traded, pct_volume_shares_change, prev_day_vol,
      basis, coc, iv, prev_day_iv, pct_iv_change,
      delta, vega, gamma, theta, rho, source_file
    FROM """ + tmp + """
    ON CONFLICT (trade_date, symbol, instrument, expiry, optype_key, strike_key)
    DO UPDATE SET
      price=EXCLUDED.price, spot=EXCLUDED.spot, last_updated=EXCLUDED.last_updated, build_up=EXCLUDED.build_up, lot_size=EXCLUDED.lot_size,
      day_change=EXCLUDED.day_change, pct_day_change=EXCLUDED.pct_day_change, open_price=EXCLUDED.open_price, high_price=EXCLUDED.high_price, low_price=EXCLUDED.low_price, prev_close=EXCLUDED.prev_close,
      oi=EXCLUDED.oi, pct_oi_change=EXCLUDED.pct_oi_change, oi_change=EXCLUDED.oi_change, prev_day_oi=EXCLUDED.prev_day_oi,
      traded_contracts=EXCLUDED.traded_contracts, traded_contracts_chg_pct=EXCLUDED.traded_contracts_chg_pct, shares_traded=EXCLUDED.shares_traded,
      pct_volume_shares_change=EXCLUDED.pct_volume_shares_change, prev_day_vol=EXCLUDED.prev_day_vol,
      basis=EXCLUDED.basis, coc=EXCLUDED.coc, iv=EXCLUDED.iv, prev_day_iv=EXCLUDED.prev_day_iv, pct_iv_change=EXCLUDED.pct_iv_change,
      delta=EXCLUDED.delta, vega=EXCLUDED.vega, gamma=EXCLUDED.gamma, theta=EXCLUDED.theta, rho=EXCLUDED.rho,
      source_file=EXCLUDED.source_file;
    """
    with ENG.begin() as cx:
        cx.execute(text(merge_sql))
        cnt = cx.execute(text(f"SELECT COUNT(*) FROM {tmp}")).scalar()
        cx.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
    return cnt

# ---------------------- ROLLUPS → indicators.values (simple starters) ----------------------

def write_symbol_rollups_for_date(trade_date: dt.date):
    sql = """
    WITH base AS (
      SELECT *
      FROM derivatives.option_chain
      WHERE trade_date = %(d)s
    ),
    fut AS (
      SELECT symbol,
             MAX(basis) AS basis,
             MAX(coc)   AS coc,
             MAX(oi)    AS oi
      FROM base
      WHERE instrument='FUT'
      GROUP BY symbol
    ),
    ce AS (
      SELECT symbol,
             MAX(iv)  AS iv, MAX(delta) AS delta,
             MAX(oi)  AS oi, MAX(traded_contracts) AS contracts
      FROM base
      WHERE instrument='OPT' AND option_type='CE'
      GROUP BY symbol
    ),
    pe AS (
      SELECT symbol,
             MAX(iv)  AS iv, MAX(delta) AS delta,
             MAX(oi)  AS oi, MAX(traded_contracts) AS contracts
      FROM base
      WHERE instrument='OPT' AND option_type='PE'
      GROUP BY symbol
    ),
    -- crude tradeability: scale contracts & OI within-day (0..40 each) + delta band bonus
    ce_scaled AS (
      SELECT b.symbol,
             ce.iv, ce.delta, ce.oi, ce.contracts,
             width_bucket(ce.contracts, 0, NULLIF(MAX(ce.contracts) OVER (),0), 40) AS liq_score,
             width_bucket(ce.oi,        0, NULLIF(MAX(ce.oi)        OVER (),0), 40) AS oi_score
      FROM base b
      LEFT JOIN ce ON ce.symbol=b.symbol
      GROUP BY b.symbol, ce.iv, ce.delta, ce.oi, ce.contracts
    ),
    pe_scaled AS (
      SELECT b.symbol,
             pe.iv, pe.delta, pe.oi, pe.contracts,
             width_bucket(pe.contracts, 0, NULLIF(MAX(pe.contracts) OVER (),0), 40) AS liq_score,
             width_bucket(pe.oi,        0, NULLIF(MAX(pe.oi)        OVER (),0), 40) AS oi_score
      FROM base b
      LEFT JOIN pe ON pe.symbol=b.symbol
      GROUP BY b.symbol, pe.iv, pe.delta, pe.oi, pe.contracts
    ),
    ce_tradeability AS (
      SELECT symbol,
             COALESCE(liq_score,0) + COALESCE(oi_score,0)
             + CASE WHEN ABS(delta) BETWEEN 0.30 AND 0.40 THEN 10 ELSE 0 END AS score
      FROM ce_scaled
    ),
    pe_tradeability AS (
      SELECT symbol,
             COALESCE(liq_score,0) + COALESCE(oi_score,0)
             + CASE WHEN ABS(delta) BETWEEN 0.30 AND 0.40 THEN 10 ELSE 0 END AS score
      FROM pe_scaled
    ),
    rows AS (
      SELECT
        jsonb_build_object(
          'symbol', f.symbol, 'market_type','futures','interval','1d',
          'ts', (%(d)s::timestamptz), 'metric','FUT.basis', 'val', f.basis
        ) AS j
      FROM fut f
      UNION ALL
      SELECT jsonb_build_object('symbol', f.symbol,'market_type','futures','interval','1d','ts',(%(d)s::timestamptz),'metric','FUT.coc','val',f.coc) FROM fut f
      UNION ALL
      SELECT jsonb_build_object('symbol', f.symbol,'market_type','futures','interval','1d','ts',(%(d)s::timestamptz),'metric','FUT.oi','val',f.oi) FROM fut f

      UNION ALL
      SELECT jsonb_build_object('symbol', c.symbol,'market_type','options','interval','1d','ts',(%(d)s::timestamptz),'metric','OPT.near.ce.iv','val',c.iv) FROM ce c
      UNION ALL
      SELECT jsonb_build_object('symbol', c.symbol,'market_type','options','interval','1d','ts',(%(d)s::timestamptz),'metric','OPT.near.ce.delta','val',c.delta) FROM ce c
      UNION ALL
      SELECT jsonb_build_object('symbol', c.symbol,'market_type','options','interval','1d','ts',(%(d)s::timestamptz),'metric','OPT.near.ce.oi','val',c.oi) FROM ce c
      UNION ALL
      SELECT jsonb_build_object('symbol', c.symbol,'market_type','options','interval','1d','ts',(%(d)s::timestamptz),'metric','OPT.near.ce.contracts','val',c.contracts) FROM ce c

      UNION ALL
      SELECT jsonb_build_object('symbol', p.symbol,'market_type','options','interval','1d','ts',(%(d)s::timestamptz),'metric','OPT.near.pe.iv','val',p.iv) FROM pe p
      UNION ALL
      SELECT jsonb_build_object('symbol', p.symbol,'market_type','options','interval','1d','ts',(%(d)s::timestamptz),'metric','OPT.near.pe.delta','val',p.delta) FROM pe p
      UNION ALL
      SELECT jsonb_build_object('symbol', p.symbol,'market_type','options','interval','1d','ts',(%(d)s::timestamptz),'metric','OPT.near.pe.oi','val',p.oi) FROM pe p
      UNION ALL
      SELECT jsonb_build_object('symbol', p.symbol,'market_type','options','interval','1d','ts',(%(d)s::timestamptz),'metric','OPT.near.pe.contracts','val',p.contracts) FROM pe p

      UNION ALL
      SELECT jsonb_build_object('symbol', t.symbol,'market_type','options','interval','1d','ts',(%(d)s::timestamptz),'metric','OPT.tradeability.ce','val',t.score) FROM ce_tradeability t
      UNION ALL
      SELECT jsonb_build_object('symbol', t.symbol,'market_type','options','interval','1d','ts',(%(d)s::timestamptz),'metric','OPT.tradeability.pe','val',t.score) FROM pe_tradeability t
    )
    INSERT INTO indicators.values (symbol, market_type, interval, ts, metric, val, context)
    SELECT
      (j->>'symbol')::text,
      (j->>'market_type')::text,
      (j->>'interval')::text,
      (j->>'ts')::timestamptz,
      (j->>'metric')::text,
      NULLIF((j->>'val'),'')::numeric,
      jsonb_build_object('source','opt_rollup')
    FROM rows
    ON CONFLICT (symbol, market_type, interval, ts, metric)
    DO UPDATE SET
      val = EXCLUDED.val,
      context = indicators.values.context || EXCLUDED.context;
    """
    with ENG.begin() as cx:
        cx.execute(text(sql), {"d": trade_date})
    return 1


# ---------------------- MAIN ----------------------

def main():
    ap = argparse.ArgumentParser(description="Ingest Option Chain (FUT + nearest CE/PE) files")
    ap.add_argument("--inbox", default="./inbox", help="Folder containing files")
    ap.add_argument("--pattern", default="*option*.xlsx", help="Glob to match files")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of files")
    ap.add_argument("--no_rollups", action="store_true", help="Skip writing indicators rollups")
    args = ap.parse_args()

    inbox = Path(args.inbox)
    files = sorted(glob.glob(str(inbox / args.pattern)))
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    if not files:
        print(f"[run] No files matched: {args.pattern} under {inbox}")
        sys.exit(0)

    total_rows = 0
    for f in files:
        try:
            trade_date = infer_trade_date_from_name(f) or dt.date.today()
            df_raw = read_any_excel(f)
            norm = normalize_frame(df_raw, trade_date, Path(f).name)
            merged = upsert_option_chain(norm)
            print(f"[ok] {Path(f).name}: upserted {merged} rows for {trade_date}")
            total_rows += (merged or 0)
            if not args.no_rollups and merged:
                write_symbol_rollups_for_date(trade_date)
                print(f"[ok] indicators.values rollups written for {trade_date}")
        except Exception as e:
            print(f"[warn] failed on {Path(f).name}: {e}")

    print(f"[done] total rows processed: {total_rows}")

if __name__ == "__main__":
    main()
