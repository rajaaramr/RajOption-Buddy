#!/usr/bin/env python3
# backfill_unified.py  — merge all historical daily FO files into alpha.fo_daily_unified

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
    Expected INI:
      [postgres]
      host=127.0.0.1
      port=5432
      dbname=TradeHub18
      user=postgres
      password=Ajantha@18
      search_path=raw_ingest
    """
    path = os.getenv("ALPHA_CFG", os.path.join(os.path.dirname(__file__), "config.ini"))
    cp = ConfigParser(inline_comment_prefixes=("#",";"))
    if not os.path.exists(path):
        print(f"[cfg] INI not found at {path}; falling back to env or defaults.")
        return cp
    # try common encodings
    for enc in ("utf-8","utf-8-sig","cp1252"):
        try:
            cp.read(path, encoding=enc)
            return cp
        except (UnicodeDecodeError, ParsingError):
            continue
    # lossy fallback
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
        sp = pg.get("search_path","alpha")
    else:
        # env fallbacks
        url = URL.create(
            "postgresql+psycopg2",
            username=os.getenv("PGUSER","postgres"),
            password=os.getenv("PGPASSWORD",""),
            host=os.getenv("PGHOST","127.0.0.1"),
            port=int(os.getenv("PGPORT","5432")),
            database=os.getenv("PGDATABASE","TradeHub18"),
        )
        sp = os.getenv("PGSEARCH_PATH","raw_ingest")

    eng = create_engine(url, future=True, pool_pre_ping=True, connect_args={"options": f"-c search_path={sp}"})
    with eng.begin() as cx:
        cx.execute(text(f"CREATE SCHEMA IF NOT EXISTS {sp}"))
        cx.execute(text(f"SET search_path TO {sp}"))
    print(f"[db] connected to {url.host}:{url.port}/{url.database} (search_path={sp})")
    return eng, sp

ENG, SCHEMA = make_engine()

# ---------------------- SCHEMA (CREATE IF NEEDED) ----------------------

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.fo_daily_unified (
  trade_date date NOT NULL,
  nse_code text NOT NULL,
  stock_name text,
  bse_code text,
  stock_code text,
  isin text,
  industry_name text,
  sector_name text,

  current_price numeric,
  market_cap numeric,
  vwap_day numeric,

  -- highs/lows & vols
  price_1y_h numeric,
  price_1y_l numeric,
  chg_pct_d numeric,
  chg_pct_m numeric,
  chg_pct_q numeric,
  chg_pct_1y numeric,
  chg_pct_2y numeric,
  chg_pct_3y numeric,
  chg_pct_5y numeric,
  chg_pct_10y numeric,
  vol_day numeric,
  vol_wk_avg numeric,
  vol_mo_avg numeric,
  day_h numeric,
  month_h numeric,
  qtr_h numeric,
  year_5_h numeric,
  year_10_h numeric,
  day_l numeric,
  month_l numeric,
  qtr_l numeric,
  year_5_l numeric,
  year_10_l numeric,

  -- F&O core
  fno_total_oi numeric,
  fno_prev_total_oi numeric,
  fno_put_vol_total numeric,
  fno_call_vol_total numeric,
  fno_prev_put_vol numeric,
  fno_prev_call_vol numeric,
  fno_put_oi_total numeric,
  fno_call_oi_total numeric,
  fno_prev_put_oi numeric,
  fno_prev_call_oi numeric,
  mwpl_abs numeric,
  mwpl_pct numeric,
  mwpl_prev_pct numeric,

  pcr_vol numeric,
  pcr_vol_prev numeric,
  pcr_vol_chg_pct numeric,
  pcr_oi numeric,
  pcr_oi_prev numeric,
  pcr_oi_chg_pct numeric,
  oi_chg_pct numeric,
  put_oi_chg_pct numeric,
  call_oi_chg_pct numeric,
  put_vol_chg_pct numeric,
  call_vol_chg_pct numeric,
  fno_rollover_cost numeric,
  fno_rollover_cost_pct numeric,
  fno_rollover_pct numeric,

  -- Trendlyne
  tl_dur numeric,
  tl_val numeric,
  tl_mom numeric,
  tl_dur_prev_d numeric,
  tl_val_prev_d numeric,
  tl_mom_prev_d numeric,
  tl_dur_prev_w numeric,
  tl_val_prev_w numeric,
  tl_mom_prev_w numeric,
  tl_dur_prev_m numeric,
  tl_val_prev_m numeric,
  tl_mom_prev_m numeric,
  norm_mom numeric,
  dvm_class text,

  -- Indicators (daily)
  mfi numeric,
  rsi numeric,
  macd numeric,
  macd_sig numeric,
  atr numeric,
  adx numeric,
  roc21 numeric,
  roc125 numeric,

  -- Averages
  sma5 numeric,
  sma30 numeric,
  sma50 numeric,
  sma100 numeric,
  sma200 numeric,
  ema12 numeric,
  ema20 numeric,
  ema50 numeric,
  ema100 numeric,

  -- Beta
  beta numeric,
  beta_1m numeric,
  beta_3m numeric,
  beta_1y numeric,
  beta_3y numeric,

  -- Pivots
  pivot numeric,
  r1 numeric,
  r1_diff_pct numeric,
  r2 numeric,
  r2_diff_pct numeric,
  r3 numeric,
  r3_diff_pct numeric,
  s1 numeric,
  s1_diff_pct numeric,
  s2 numeric,
  s2_diff_pct numeric,
  s3 numeric,
  s3_diff_pct numeric,

  PRIMARY KEY (trade_date, nse_code)
);
"""

with ENG.begin() as cx:
    cx.execute(text(CREATE_TABLE_SQL))
print("[db] fo_daily_unified ready.")

# --- Ensure target schema/table for indicator rows (1d features pillars can read) ---
CREATE_INDICATORS_SQL = """
CREATE SCHEMA IF NOT EXISTS indicators;
CREATE TABLE IF NOT EXISTS indicators.values (
  symbol      text        NOT NULL,
  market_type text        NOT NULL,
  interval    text        NOT NULL,
  ts          timestamptz NOT NULL,
  metric      text        NOT NULL,
  val         numeric,
  context     jsonb DEFAULT '{}'::jsonb,
  PRIMARY KEY (symbol, market_type, interval, ts, metric)
);
CREATE INDEX IF NOT EXISTS idx_ind_vals_recent
  ON indicators.values(symbol, metric, ts DESC);
"""
with ENG.begin() as cx:
    cx.execute(text(CREATE_INDICATORS_SQL))
print("[db] indicators.values ready.")

def upsert_indicator_values_for_date(trade_date: dt.date) -> int:
    sql = f"""
    WITH eod AS (
      SELECT
        d.trade_date::timestamptz AS ts_eod,
        d.nse_code                AS symbol,
        d.current_price,
        d.market_cap,

        -- F&O aggregates / PCR / MWPL
        d.fno_total_oi, d.fno_prev_total_oi,
        d.oi_chg_pct,
        d.pcr_vol, d.pcr_vol_prev, d.pcr_vol_chg_pct,
        d.pcr_oi,  d.pcr_oi_prev,  d.pcr_oi_chg_pct,
        d.mwpl_abs, d.mwpl_pct, d.mwpl_prev_pct,

        -- Pivots & distances
        d.pivot, d.r1, d.r2, d.r3, d.s1, d.s2, d.s3,
        d.r1_diff_pct, d.r2_diff_pct, d.r3_diff_pct,
        d.s1_diff_pct, d.s2_diff_pct, d.s3_diff_pct,

        -- Trendlyne vendor scores
        d.tl_mom, d.tl_mom_prev_d, d.tl_mom_prev_w, d.tl_mom_prev_m,
        d.tl_dur, d.tl_val
      FROM {SCHEMA}.fo_daily_unified d
      WHERE d.trade_date = :d
    ),
    metrics AS (
      SELECT
        e.symbol,
        'spot'::text   AS market_type,
        '1d'::text     AS interval,
        e.ts_eod       AS ts,
        m.metric,
        m.val,
        m.ctx
      FROM eod e
      CROSS JOIN LATERAL (VALUES
        ('PX.last',   e.current_price,  jsonb_build_object('source','raw_ingest')),
        ('MKT.cap',   e.market_cap,     jsonb_build_object('source','raw_ingest')),
        ('FNO.oi.total',           e.fno_total_oi,         jsonb_build_object('source','raw_ingest')),
        ('FNO.oi.total.prev',      e.fno_prev_total_oi,    jsonb_build_object('source','raw_ingest')),
        ('FNO.oi.total.chg_pct',   e.oi_chg_pct,           jsonb_build_object('source','raw_ingest')),
        ('FNO.pcr.vol',            e.pcr_vol,              jsonb_build_object('source','raw_ingest')),
        ('FNO.pcr.vol.prev',       e.pcr_vol_prev,         jsonb_build_object('source','raw_ingest')),
        ('FNO.pcr.vol.chg_pct',    e.pcr_vol_chg_pct,      jsonb_build_object('source','raw_ingest')),
        ('FNO.pcr.oi',             e.pcr_oi,               jsonb_build_object('source','raw_ingest')),
        ('FNO.pcr.oi.prev',        e.pcr_oi_prev,          jsonb_build_object('source','raw_ingest')),
        ('FNO.pcr.oi.chg_pct',     e.pcr_oi_chg_pct,       jsonb_build_object('source','raw_ingest')),
        ('FNO.mwpl.abs',           e.mwpl_abs,             jsonb_build_object('source','raw_ingest')),
        ('FNO.mwpl.pct',           e.mwpl_pct,             jsonb_build_object('source','raw_ingest')),
        ('FNO.mwpl.prev_pct',      e.mwpl_prev_pct,        jsonb_build_object('source','raw_ingest')),
        ('PIVOT.pp',               e.pivot,                 jsonb_build_object('source','raw_ingest')),
        ('PIVOT.r1',               e.r1,                    jsonb_build_object('source','raw_ingest')),
        ('PIVOT.r2',               e.r2,                    jsonb_build_object('source','raw_ingest')),
        ('PIVOT.r3',               e.r3,                    jsonb_build_object('source','raw_ingest')),
        ('PIVOT.s1',               e.s1,                    jsonb_build_object('source','raw_ingest')),
        ('PIVOT.s2',               e.s2,                    jsonb_build_object('source','raw_ingest')),
        ('PIVOT.s3',               e.s3,                    jsonb_build_object('source','raw_ingest')),
        ('PIVOT.r1.dist_pct',      e.r1_diff_pct,           jsonb_build_object('source','raw_ingest')),
        ('PIVOT.r2.dist_pct',      e.r2_diff_pct,           jsonb_build_object('source','raw_ingest')),
        ('PIVOT.r3.dist_pct',      e.r3_diff_pct,           jsonb_build_object('source','raw_ingest')),
        ('PIVOT.s1.dist_pct',      e.s1_diff_pct,           jsonb_build_object('source','raw_ingest')),
        ('PIVOT.s2.dist_pct',      e.s2_diff_pct,           jsonb_build_object('source','raw_ingest')),
        ('PIVOT.s3.dist_pct',      e.s3_diff_pct,           jsonb_build_object('source','raw_ingest')),
        ('TL.momentum',            e.tl_mom,                jsonb_build_object('source','vendor','vendor','trendlyne')),
        ('TL.momentum.prev_d',     e.tl_mom_prev_d,         jsonb_build_object('source','vendor','vendor','trendlyne')),
        ('TL.momentum.prev_w',     e.tl_mom_prev_w,         jsonb_build_object('source','vendor','vendor','trendlyne')),
        ('TL.momentum.prev_m',     e.tl_mom_prev_m,         jsonb_build_object('source','vendor','vendor','trendlyne')),
        ('TL.durability',          e.tl_dur,                jsonb_build_object('source','vendor','vendor','trendlyne')),
        ('TL.valuation',           e.tl_val,                jsonb_build_object('source','vendor','vendor','trendlyne'))
      ) AS m(metric, val, ctx)
    ),
    final_rows AS (
      SELECT
        symbol,
        CASE WHEN metric LIKE 'FNO.%' THEN 'futures' ELSE market_type END AS market_type,
        interval,
        ts,
        metric,
        val,
        ctx
      FROM metrics
      WHERE val IS NOT NULL
    )
    INSERT INTO indicators.values (symbol, market_type, interval, ts, metric, val, context)
    SELECT symbol, market_type, interval, ts, metric, val, ctx
    FROM final_rows
    ON CONFLICT (symbol, market_type, interval, ts, metric)
    DO UPDATE SET
      val     = EXCLUDED.val,
      context = indicators.values.context || EXCLUDED.context;
    """
    with ENG.begin() as cx:
        cx.execute(text(sql), {"d": trade_date})
    return 1

# ---------------------- HELPERS ----------------------

PCT_COL_KEYS = {
    # any column in this set will be cleaned as percent if its raw value looks like "12.3%" or has commas
    "chg_pct_d","chg_pct_m","chg_pct_q","chg_pct_1y","chg_pct_2y","chg_pct_3y","chg_pct_5y","chg_pct_10y",
    "pcr_vol_chg_pct","pcr_oi_chg_pct","oi_chg_pct","put_oi_chg_pct","call_oi_chg_pct","put_vol_chg_pct","call_vol_chg_pct",
    "fno_rollover_cost_pct","fno_rollover_pct",
    "r1_diff_pct","r2_diff_pct","r3_diff_pct","s1_diff_pct","s2_diff_pct","s3_diff_pct"
}

def to_num(v):
    return pd.to_numeric(v, errors="coerce")

def coerce_series_percent(s: pd.Series) -> pd.Series:
    if s is None: return s
    return (s.astype(str)
              .str.replace("%","", regex=False)
              .str.replace(",","", regex=False)
              .str.strip()
              .replace({"": None})
              .pipe(pd.to_numeric, errors="coerce"))

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
    # best-effort read first sheet if named sheet missing
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, sheet_name=0)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+"," ", c).strip() for c in df.columns]
    return df

# Column mapping: canonical -> list of possible header names
MAP = {
  # identity
  "stock_name": ["Stock Name"],
  "nse_code": ["NSE Code","NSE code","SYMBOL","Symbol"],
  "bse_code": ["BSE Code"],
  "stock_code": ["Stock Code"],
  "isin": ["ISIN"],
  "industry_name": ["Industry Name","industry_name"],
  "sector_name": ["sector_name","Sector","Sector Name"],

  # price+cap
  "current_price": ["Current Price","LTP","Last Price","Price","Close"],
  "market_cap": ["Market Capitalization","MktCap"],
  "vwap_day": ["VWAP Day","Current Price VWAP Day"],

  # highs/lows/vols
  "price_1y_h": ["1Yr High"],
  "price_1y_l": ["1Yr Low"],
  "chg_pct_d": ["Day change %"],
  "chg_pct_m": ["Month Change %"],
  "chg_pct_q": ["Qtr Change %"],
  "chg_pct_1y": ["1Yr change %"],
  "chg_pct_2y": ["2Yr price change %"],
  "chg_pct_3y": ["3Yr price change %"],
  "chg_pct_5y": ["5Yr price change %"],
  "chg_pct_10y": ["10Yr price change %"],
  "vol_day": ["Day Volume"],
  "vol_wk_avg": ["Week Volume Avg"],
  "vol_mo_avg": ["Month Volume Avg"],
  "day_h": ["Day High"],
  "month_h": ["Month High"],
  "qtr_h": ["Qtr High"],
  "year_5_h": ["5Yr High"],
  "year_10_h": ["10Yr High"],
  "day_l": ["Day Low"],
  "month_l": ["Month Low"],
  "qtr_l": ["Qtr Low"],
  "year_5_l": ["5Yr Low"],
  "year_10_l": ["10Yr Low"],

  # F&O
  "fno_total_oi": ["FnO Total Open Interest"],
  "fno_prev_total_oi": ["FnO Previous Day Total Open Interest"],
  "fno_put_vol_total": ["FnO Total Put Volume"],
  "fno_call_vol_total": ["FnO Total Call Volume"],
  "fno_prev_put_vol": ["FnO Previous Day Total Put Volume"],
  "fno_prev_call_vol": ["FnO Previous Day Total Call Volume"],
  "fno_put_oi_total": ["FnO Total Put Open Interest"],
  "fno_call_oi_total": ["FnO Total Call Open Interest"],
  "fno_prev_put_oi": ["FnO Previous Day Total Put Open Interest"],
  "fno_prev_call_oi": ["FnO Previous Day Total Call Open Interest"],
  "mwpl_abs": ["FnO Marketwide position limit","FnO Marketwide Position Limit"],
  "mwpl_pct": ["FnO Marketwide Position Limit %","FnO Marketwide Position Limit % "],
  "mwpl_prev_pct": ["FnO previous day Marketwide Position Limit %"],
  "pcr_vol": ["FnO PCR Put to call Volume ratio"],
  "pcr_vol_prev": ["FnO PCR Put to call Volume ratio previous day"],
  "pcr_vol_chg_pct": ["FnO PCR Volume change %"],
  "pcr_oi": ["FnO PCR OI Put to call open interest ratio"],
  "pcr_oi_prev": ["FnO PCR OI Put to call open interest ratio previous day"],
  "pcr_oi_chg_pct": ["FnO PCR OI change %"],
  "oi_chg_pct": ["FnO Total Open Interest change %"],
  "put_oi_chg_pct": ["FnO Total Put Open Interest change %"],
  "call_oi_chg_pct": ["FnO Total Call Open Interest change %"],
  "put_vol_chg_pct": ["FnO Total Put Volume change %"],
  "call_vol_chg_pct": ["FnO Total Call Volume change %"],
  "fno_rollover_cost": ["FnO Rollover Cost"],
  "fno_rollover_cost_pct": ["FnO Rollover Cost %"],
  "fno_rollover_pct": ["FnO Rollover %"],

  # Trendlyne block
  "tl_dur": ["Trendlyne Durability Score"],
  "tl_val": ["Trendlyne Valuation Score"],
  "tl_mom": ["Trendlyne Momentum Score"],
  "tl_dur_prev_d": ["Prev Day Trendlyne Durability Score"],
  "tl_val_prev_d": ["Prev Day Trendlyne Valuation Score"],
  "tl_mom_prev_d": ["Prev Day Trendlyne Momentum Score"],
  "tl_dur_prev_w": ["Prev Week Trendlyne Durability Score"],
  "tl_val_prev_w": ["Prev Week Trendlyne Valuation Score"],
  "tl_mom_prev_w": ["Prev Week Trendlyne Momentum Score"],
  "tl_dur_prev_m": ["Prev Month Trendlyne Durability Score"],
  "tl_val_prev_m": ["Prev Month Trendlyne Valuation Score"],
  "tl_mom_prev_m": ["Prev Month Trendlyne Momentum Score"],
  "norm_mom": ["Normalized Momentum Score"],
  "dvm_class": ["DVM_classification_text"],

  # indicators
  "mfi": ["Day MFI","MFI"],
  "rsi": ["Day RSI","RSI"],
  "macd": ["Day MACD","MACD"],
  "macd_sig": ["Day MACD Signal Line","MACD Signal"],
  "atr": ["Day ATR","ATR"],
  "adx": ["Day ADX","ADX"],
  "roc21": ["Day ROC21","ROC21"],
  "roc125": ["Day ROC125","ROC125"],

  # averages
  "sma5": ["5Day SMA","SMA 5"],
  "sma30": ["30Day SMA","SMA 30"],
  "sma50": ["50Day SMA","SMA 50"],
  "sma100": ["100Day SMA","SMA 100"],
  "sma200": ["200Day SMA","SMA 200"],
  "ema12": ["12Day EMA","EMA 12"],
  "ema20": ["20Day EMA","EMA 20"],
  "ema50": ["50Day EMA","EMA 50"],
  "ema100": ["100Day EMA","EMA 100"],

  # beta
  "beta": ["Beta"],
  "beta_1m": ["Beta 1Month","1Month Beta"],
  "beta_3m": ["Beta 3Month","3Month Beta"],
  "beta_1y": ["Beta 1Year","1Year Beta"],
  "beta_3y": ["Beta 3Year","3Year Beta"],

  # pivots
  "pivot": ["Pivot point","Pivot"],
  "r1": ["First resistance R1"],
  "r1_diff_pct": ["First resistance R1 to price diff %"],
  "r2": ["Second resistance R2"],
  "r2_diff_pct": ["Second resistance R2 to price diff %"],
  "r3": ["Third resistance R3"],
  "r3_diff_pct": ["Third resistance R3 to price diff %"],
  "s1": ["First support S1"],
  "s1_diff_pct": ["First support S1 to price diff %"],
  "s2": ["Second support S2"],
  "s2_diff_pct": ["Second support S2 to price diff %"],
  "s3": ["Third support S3"],
  "s3_diff_pct": ["Third support S3 to price diff %"],
}

CANON_ORDER = [c for c in CREATE_TABLE_SQL.splitlines() if c.strip().startswith(("",))]  # not used, but kept for clarity

def pick_column(df, names):
    """Return the first matching column in df for the list of candidate names."""
    for want in names:
        for c in df.columns:
            if c.strip().lower() == want.strip().lower():
                return c
        for c in df.columns:
            if want.strip().lower() in c.strip().lower():
                return c
    return None

def normalize_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df_raw)
    out = {}
    # identity first (nse_code is mandatory)
    for canon, choices in MAP.items():
        col = pick_column(df, choices)
        if col is not None:
            out[canon] = df[col]
    out_df = pd.DataFrame(out)
    # coerce numerics
    for col in out_df.columns:
        if col in {"stock_name","nse_code","bse_code","stock_code","isin","industry_name","sector_name","dvm_class"}:
            continue
        if col in PCT_COL_KEYS:
            out_df[col] = coerce_series_percent(out_df[col])
        else:
            out_df[col] = to_num(out_df[col])
    # uppercase NSE codes; trim strings
    if "nse_code" in out_df.columns:
        out_df["nse_code"] = out_df["nse_code"].astype(str).str.strip().str.upper()
    for c in ["stock_name","bse_code","stock_code","isin","industry_name","sector_name","dvm_class"]:
        if c in out_df.columns:
            out_df[c] = out_df[c].astype(str).str.strip()
    return out_df

def upsert_unified(df: pd.DataFrame, trade_date: dt.date):
    if df is None or df.empty:
        return 0
    df = df.copy()
    df["trade_date"] = trade_date
    if "nse_code" not in df.columns:
        print("[skip] file lacks NSE Code; nothing to upsert.")
        return 0
    # remove dupes on PK within this batch
    df = df.dropna(subset=["nse_code"]).drop_duplicates(subset=["trade_date","nse_code"], keep="last")

    # limit to known columns in table
    with ENG.begin() as cx:
        cols = [r[0] for r in cx.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema=:s AND table_name='fo_daily_unified'
        """), {"s": SCHEMA}).fetchall()]
    use_cols = [c for c in df.columns if c in cols]
    df = df[use_cols]
    if df.empty:
        return 0

    # create temp table with identical columns
    tmp = "tmp_fo_unified_stage"
    col_list = ", ".join([f'"{c}"' for c in use_cols])
    with ENG.begin() as cx:
        cx.execute(text(f'DROP TABLE IF EXISTS {tmp}'))
        cx.execute(text(f'CREATE TEMP TABLE {tmp} AS SELECT {col_list} FROM {SCHEMA}.fo_daily_unified WHERE 1=0'))

    # bulk insert into temp
    df.to_sql(tmp, ENG, if_exists="append", index=False, method="multi", chunksize=5000)

    # build upsert
    pk = ['trade_date','nse_code']
    non_pk = [c for c in use_cols if c not in pk]
    set_sql = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in non_pk]) if non_pk else ""
    merge_sql = f"""
    INSERT INTO {SCHEMA}.fo_daily_unified ({col_list})
    SELECT {col_list} FROM {tmp}
    ON CONFLICT (trade_date, nse_code) DO UPDATE
      SET {set_sql};
    """
    with ENG.begin() as cx:
        cx.execute(text(merge_sql))
        # rows merged count is not returned reliably across drivers; we can estimate:
        cnt = cx.execute(text(f"SELECT COUNT(*) FROM {tmp}")).scalar()
        cx.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
    return cnt

# ---------------------- MAIN ----------------------

def main():
    ap = argparse.ArgumentParser(description="Backfill fo_daily_unified from historical daily FO files")
    ap.add_argument("--inbox", default="./inbox", help="Folder containing files")
    ap.add_argument("--pattern", default="2025-*.xlsx", help="Glob to match files (e.g. '*v2*.xlsx')")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of files")
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
            norm = normalize_frame(df_raw)
            merged = upsert_unified(norm, trade_date)
            print(f"[ok] {Path(f).name}: upserted {merged} rows for {trade_date}")
            if merged is not None and merged > 0:
                upsert_indicator_values_for_date(trade_date)
                print(f"[ok] indicators.values updated for {trade_date}")
            total_rows += (merged or 0)
        except Exception as e:
            print(f"[warn] failed on {Path(f).name}: {e}")

    print(f"[done] total rows processed: {total_rows}")

if __name__ == "__main__":
    main()
