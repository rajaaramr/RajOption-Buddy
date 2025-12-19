# scheduler/update_vp_bb.py
from __future__ import annotations

import os, math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import configparser
from pandas.tseries.frequencies import to_offset

import psycopg2.extras as pgx  # NEW: for bulk inserts

from utils.db import get_db_connection

# ---------------------------------------------------------------------
# Optional indicator libs (robust imports)
# ---------------------------------------------------------------------
PTA_OK = False
TA_OK  = False
try:
    import pandas_ta as pta  # type: ignore
    PTA_OK = True
except Exception:
    pta = None
    PTA_OK = False

try:
    from ta.momentum import RSIIndicator
    from ta.trend import EMAIndicator, MACD as TA_MACD
    from ta.volatility import AverageTrueRange
    TA_OK = True
except Exception:
    RSIIndicator = EMAIndicator = TA_MACD = AverageTrueRange = None
    TA_OK = False

# ---------------------------------------------------------------------
# Globals / Defaults
# ---------------------------------------------------------------------
TZ = timezone.utc
DEFAULT_INI = os.getenv("INDICATORS_INI", "indicators.ini")

# map TF names -> pandas offset rules
TF_TO_OFFSET = {
    "15m":  "15min",
    "30m":  "30min",
    "60m":  "60min",
    "90m":  "90min",
    "120m": "120min",
    "240m": "240min",
}

# ---------------------------------------------------------------------
# Config loading (INI + env overrides)
# ---------------------------------------------------------------------
def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default

def _as_bool(x: Optional[str], default: bool) -> bool:
    if x is None: return default
    return str(x).strip().lower() in {"1","true","yes","on","y"}

def _as_int(x: Optional[str], default: int) -> int:
    try: return int(str(x).strip())
    except Exception: return default

def _as_float(x: Optional[str], default: float) -> float:
    try: return float(str(x).strip())
    except Exception: return default

def _as_list_csv(x: Optional[str], default: List[str]) -> List[str]:
    if not x: return default
    return [p.strip() for p in x.split(",") if p.strip()]

@dataclass
class VPBBConfig:
    lookback_days: int
    tf_list: List[str]
    market_kind: str
    run_id: str
    source: str
    vp_bins: int
    vp_value_pct: float
    # BB thresholds (base)
    vol_ma_len: int
    vol_pct_thr: int
    csv_z_thr: float
    obv_delta_min: float
    min_blocks: int
    score_hi: float
    # Extras
    write_diagnostics: bool
    use_typical_price: bool
    pick_best_run: bool
    auto_scale: bool
    # New knobs
    regime_lookback_days: int
    std_lookback_days: int
    rsi_len: int
    macd_fast: int
    macd_slow: int
    macd_sig: int
    ema_refs: List[int]
    dyn_volpct_low: int
    dyn_volpct_high: int
    vwap_anchor_mode: str     # "none" | "swing" | "days"
    vwap_anchor_days: int
    vp_decay_alpha: float
    # Backfill tail (how many most-recent bars per TF to write)
    backfill_bars: int

def load_vpbb_cfg(ini_path: str = DEFAULT_INI) -> VPBBConfig:
    dflt = {
        "lookback_days": 90,
        "tf_list": "15m,30m,60m,90m,120m,240m",
        "market_kind": "futures",
        "run_id": "vpbb_run",
        "source": "vp_bb",
        "vp_bins": 50,
        "vp_value_pct": 0.70,
        # relaxed defaults
        "vol_ma_len": 30,
        "vol_pct_thr": 55,
        "csv_z_thr": 0.4,
        "obv_delta_min": 30000.0,
        "min_blocks": 2,
        "score_hi": 6.5,
        # extras
        "write_diagnostics": True,
        "use_typical_price": True,
        "pick_best_run": True,
        "auto_scale": True,
        # new defaults
        "regime_lookback_days": 60,
        "std_lookback_days": 60,
        "rsi_len": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_sig": 9,
        "ema_refs": "20,50,200",
        "dyn_volpct_low": 40,
        "dyn_volpct_high": 65,
        "vwap_anchor_mode": "none",
        "vwap_anchor_days": 20,
        "vp_decay_alpha": 0.0,
    }

    cfgp = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cfgp.read(ini_path)

    tflist_ini = cfgp.get("timeframes", "list", fallback=None) if cfgp.has_section("timeframes") else None
    vb = "vpbb"
    def g(sec, key, fb=None): return cfgp.get(sec, key, fallback=fb) if cfgp.has_section(sec) else None

    lookback_days = _as_int(_get_env("VPBB_LOOKBACK_DAYS", g(vb, "lookback_days")), dflt["lookback_days"])
    tf_list       = _as_list_csv(_get_env("VPBB_TF_LIST", tflist_ini or dflt["tf_list"]), dflt["tf_list"].split(","))
    market_kind   = (_get_env("VPBB_KIND", g(vb, "market_kind") or dflt["market_kind"]) or dflt["market_kind"]).lower()
    run_id        = _get_env("RUN_ID", dflt["run_id"]) or dflt["run_id"]
    source        = _get_env("SRC", dflt["source"]) or dflt["source"]

    vp_bins      = _as_int(_get_env("VP_BINS", g(vb, "vp_bins")), dflt["vp_bins"])
    vp_value_pct = _as_float(_get_env("VP_VALUE_PCT", g(vb, "vp_value_pct")), dflt["vp_value_pct"])

    vol_ma_len    = _as_int(_get_env("BB_VOL_MA_LEN", g(vb, "vol_ma_len")), dflt["vol_ma_len"])
    vol_pct_thr   = _as_int(_get_env("BB_VOL_PCT_THR", g(vb, "vol_pct_thr")), dflt["vol_pct_thr"])
    csv_z_thr     = _as_float(_get_env("BB_CSV_Z_THR", g(vb, "csv_z_thr")), dflt["csv_z_thr"])
    obv_delta_min = _as_float(_get_env("BB_OBV_DELTA_MIN", g(vb, "obv_delta_min")), dflt["obv_delta_min"])
    min_blocks    = _as_int(_get_env("BB_MIN_BLOCKS", g(vb, "min_blocks")), dflt["min_blocks"])
    score_hi      = _as_float(_get_env("BB_SCORE_HI", g(vb, "score_hi")), dflt["score_hi"])

    write_diagnostics = _as_bool(_get_env("BB_WRITE_DIAGNOSTICS", g(vb, "write_diagnostics")), dflt["write_diagnostics"])
    use_typical_price = _as_bool(_get_env("VP_USE_TYPICAL", g(vb, "use_typical_price")), dflt["use_typical_price"])
    pick_best_run     = _as_bool(_get_env("BB_PICK_BEST", g(vb, "pick_best_run")), dflt["pick_best_run"])
    auto_scale        = _as_bool(_get_env("BB_AUTO_SCALE", g(vb, "auto_scale")), dflt["auto_scale"])

    regime_lookback_days = _as_int(_get_env("VPBB_REGIME_LOOKBACK_DAYS", g(vb,"regime_lookback_days")), dflt["regime_lookback_days"])
    std_lookback_days    = _as_int(_get_env("VPBB_STD_LOOKBACK_DAYS",    g(vb,"std_lookback_days")),  dflt["std_lookback_days"])
    rsi_len    = _as_int(_get_env("BB_RSI_LEN",   g(vb,"rsi_len")),   dflt["rsi_len"])
    macd_fast  = _as_int(_get_env("BB_MACD_FAST", g(vb,"macd_fast")), dflt["macd_fast"])
    macd_slow  = _as_int(_get_env("BB_MACD_SLOW", g(vb,"macd_slow")), dflt["macd_slow"])
    macd_sig   = _as_int(_get_env("BB_MACD_SIG",  g(vb,"macd_sig")),  dflt["macd_sig"])
    ema_refs_raw = _as_list_csv(_get_env("BB_EMA_REFS", g(vb,"ema_refs")), dflt["ema_refs"].split(","))
    ema_refs   = [int(x) for x in ema_refs_raw if str(x).isdigit()]
    dyn_volpct_low  = _as_int(_get_env("BB_DYN_VOLPCT_LOW",  g(vb,"dyn_volpct_low")),  dflt["dyn_volpct_low"])
    dyn_volpct_high = _as_int(_get_env("BB_DYN_VOLPCT_HIGH", g(vb,"dyn_volpct_high")), dflt["dyn_volpct_high"])
    vwap_anchor_mode = (_get_env("BB_VWAP_ANCHOR_MODE", g(vb,"vwap_anchor_mode")) or dflt["vwap_anchor_mode"]).lower()
    vwap_anchor_days = _as_int(_get_env("BB_VWAP_ANCHOR_DAYS", g(vb,"vwap_anchor_days")), dflt["vwap_anchor_days"])
    vp_decay_alpha   = _as_float(_get_env("VP_DECAY_ALPHA", g(vb,"vp_decay_alpha")), dflt["vp_decay_alpha"])

    backfill_bars = _as_int(_get_env("VPBB_BACKFILL_BARS", g(vb, "BACKFILL_BARS")), 2000)

    return VPBBConfig(
        lookback_days=lookback_days, tf_list=tf_list, market_kind=market_kind,
        run_id=run_id, source=source, vp_bins=vp_bins, vp_value_pct=vp_value_pct,
        vol_ma_len=vol_ma_len, vol_pct_thr=vol_pct_thr, csv_z_thr=csv_z_thr,
        obv_delta_min=obv_delta_min, min_blocks=min_blocks, score_hi=score_hi,
        write_diagnostics=write_diagnostics, use_typical_price=use_typical_price,
        pick_best_run=pick_best_run, auto_scale=auto_scale,
        regime_lookback_days=regime_lookback_days, std_lookback_days=std_lookback_days,
        rsi_len=rsi_len, macd_fast=macd_fast, macd_slow=macd_slow, macd_sig=macd_sig,
        ema_refs=ema_refs, dyn_volpct_low=dyn_volpct_low, dyn_volpct_high=dyn_volpct_high,
        vwap_anchor_mode=vwap_anchor_mode, vwap_anchor_days=vwap_anchor_days,
        vp_decay_alpha=vp_decay_alpha, backfill_bars=backfill_bars
    )

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _table_name(kind: str) -> str:
    return "market.futures_candles" if kind == "futures" else "market.spot_candles"

def _frames_table(kind: str) -> str:
    return "indicators.futures_frames" if kind == "futures" else "indicators.spot_frames"

def _get_last_vpbb_ts(symbol: str, kind: str, tf: str) -> Optional[datetime]:
    """Fetch the latest timestamp that has valid VP/BB data."""
    tbl = _frames_table(kind)
    # check bb_score to see if we ran there
    sql = f"SELECT max(ts) FROM {tbl} WHERE symbol=%s AND interval=%s AND bb_score IS NOT NULL"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (symbol, tf))
        row = cur.fetchone()
    if row and row[0]:
        # return as UTC datetime
        return pd.to_datetime(row[0], utc=True).to_pydatetime()
    return None

def _now_utc() -> datetime: return datetime.now(TZ)
def _cutoff_days(n: int) -> datetime: return _now_utc() - timedelta(days=n)

def _safe_iloc(s: Optional[pd.Series], idx: int, default: float) -> float:
    try:
        v = float(s.iloc[idx])
        return v if np.isfinite(v) else default
    except Exception:
        return default

# ---------------------------------------------------------------------
# Data prep / resampling
# ---------------------------------------------------------------------
def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    d = df.copy()

    if "ts" in d.columns:
        d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
        d = d.dropna(subset=["ts"]).set_index("ts")
    else:
        d.index = pd.to_datetime(d.index, utc=True, errors="coerce")
        d = d[~d.index.isna()]
    d = d.sort_index()

    d.columns = [c.lower() for c in d.columns]
    alias = {"open_price":"open","high_price":"high","low_price":"low","close_price":"close",
             "o":"open","h":"high","l":"low","c":"close","v":"volume"}
    for old, new in alias.items():
        if old in d.columns and new not in d.columns:
            d[new] = d[old]
    for c in ("open","high","low","close","volume"):
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["volume"] = d["volume"].fillna(0.0)
    d = d.dropna(subset=["open","high","low","close"])
    return d.astype({"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"})

def _infer_rule(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    d = df.index.to_series().diff().dropna()
    if d.empty:
        return None
    sec = int(d.mode().iloc[0].total_seconds())
    return f"{int(sec // 60)}min" if sec % 60 == 0 else None

def _secs(rule: str | None) -> int | None:
    if not rule:
        return None
    off = to_offset(rule)
    # robust: works for DateOffset/Timedelta now and in future pandas
    try:
        return int(pd.Timedelta(off).total_seconds())
    except Exception:
        # very old pandas fallback
        return int(getattr(off, "delta", pd.Timedelta(0)).total_seconds())

def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    d = sanitize_ohlcv(df)
    rule = TF_TO_OFFSET.get(tf)
    if d.empty or not rule:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    base_rule = _infer_rule(d)

    # same granularity → no-op
    if base_rule and _secs(base_rule) == _secs(rule):
        return d
    # don’t upsample (e.g., 30m base → 15m target)
    if base_rule and _secs(base_rule) and _secs(rule) and _secs(rule) < _secs(base_rule):
        return d

    out = pd.DataFrame({
        "open":   d["open"]  .resample(rule, label="right", closed="right").first(),
        "high":   d["high"]  .resample(rule, label="right", closed="right").max(),
        "low":    d["low"]   .resample(rule, label="right", closed="right").min(),
        "close":  d["close"] .resample(rule, label="right", closed="right").last(),
        "volume": d["volume"].resample(rule, label="right", closed="right").sum(),
    }).dropna(how="any").astype(
        {"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"}
    )
    return out

# ---------------------------------------------------------------------
# IO: load 15m intraday candles
# ---------------------------------------------------------------------
def load_intra(symbol: str, kind: str, *, lookback_days: int) -> pd.DataFrame:
    tbl = _table_name(kind)
    cutoff = _cutoff_days(lookback_days)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ts, open, high, low, close, volume
              FROM {tbl}
             WHERE symbol=%s AND interval=%s AND ts >= %s
             ORDER BY ts ASC
            """,
            (symbol, "15m", cutoff)
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)
    return df.dropna(subset=["open","high","low","close"]).astype(
        {"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"}
    )

# ---------------------------------------------------------------------
# Technical helpers (pandas-ta / ta if available)
# ---------------------------------------------------------------------
def _ema(s: pd.Series, span: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    # 1) pandas-ta functional
    if PTA_OK:
        try:
            out = pta.ema(s, length=span)
            if out is not None and isinstance(out, pd.Series):
                return out
        except Exception:
            pass
    # 2) "ta"
    if TA_OK and EMAIndicator is not None:
        try:
            ind = EMAIndicator(close=s, window=span, fillna=False)
            out = ind.ema_indicator()
            if out is not None and isinstance(out, pd.Series):
                return out
        except Exception:
            pass
    # 3) fallback
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, win: int) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    # 1) pandas-ta
    if PTA_OK:
        try:
            out = pta.rsi(c, length=win)
            if out is not None and isinstance(out, pd.Series):
                return out.bfill().fillna(50.0)  # deprecation-safe
        except Exception:
            pass
    # 2) "ta"
    if TA_OK and RSIIndicator is not None:
        try:
            ind = RSIIndicator(close=c, window=win, fillna=False)
            out = ind.rsi()
            if out is not None and isinstance(out, pd.Series):
                return out.bfill().fillna(50.0)
        except Exception:
            pass
    # 3) fallback (Wilder-ish)
    d = c.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    rs = (up.rolling(win, min_periods=max(2, win//2)).mean()) / (
         dn.rolling(win, min_periods=max(2, win//2)).mean().replace(0, np.nan))
    out = 100.0 - 100.0/(1.0 + rs)
    return out.bfill().fillna(50.0)

def _macd(close: pd.Series, fast: int, slow: int, sig: int):
    c = pd.to_numeric(close, errors="coerce")
    # 1) pandas-ta
    if PTA_OK:
        try:
            macd_line  = pta.ema(c, length=fast) - pta.ema(c, length=slow)
            macd_sig   = macd_line.ewm(span=sig, adjust=False).mean()
            macd_hist  = macd_line - macd_sig
            return macd_line, macd_sig, macd_hist
        except Exception:
            pass
    # 2) "ta"
    if TA_OK and TA_MACD is not None:
        try:
            ind = TA_MACD(close=c, window_slow=slow, window_fast=fast, window_sign=sig, fillna=False)
            macd_line = ind.macd()
            macd_sig  = ind.macd_signal()
            macd_hist = ind.macd_diff()
            return macd_line, macd_sig, macd_hist
        except Exception:
            pass
    # 3) fallback
    ema_f = _ema(c, fast)
    ema_s = _ema(c, slow)
    macd_line = ema_f - ema_s
    macd_sig  = macd_line.ewm(span=sig, adjust=False).mean()
    macd_hist = macd_line - macd_sig
    return macd_line, macd_sig, macd_hist

def _atr(close: pd.Series, high: pd.Series, low: pd.Series, span: int = 14) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    # 1) pandas-ta
    if PTA_OK:
        try:
            out = pta.atr(h, l, c, length=span)
            if out is not None and isinstance(out, pd.Series):
                return out
        except Exception:
            pass
    # 2) "ta"
    if TA_OK and AverageTrueRange is not None:
        try:
            ind = AverageTrueRange(high=h, low=l, close=c, window=span, fillna=False)
            out = ind.average_true_range()
            if out is not None and isinstance(out, pd.Series):
                return out
        except Exception:
            pass
    # 3) fallback
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/span, adjust=False).mean()

def _obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum()

def _vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"]+df["low"]+df["close"])/3.0
    pv = tp * df["volume"]
    cum_pv = pv.cumsum()
    cum_v  = df["volume"].cumsum().replace(0, np.nan)
    return cum_pv / cum_v

def _anchor_index_by_swing(close: pd.Series, lookback: int = 200) -> int:
    window = min(len(close), lookback)
    if window < 5: return 0
    seg = close.iloc[-window:]
    i_lo = int(seg.values.argmin()); i_hi = int(seg.values.argmax())
    pick = i_lo if (window-1 - i_lo) <= (window-1 - i_hi) else i_hi
    return (len(close)-window) + pick

def _anchored_vwap(df: pd.DataFrame, anchor_pos: int) -> pd.Series:
    tp = (df["high"]+df["low"]+df["close"])/3.0
    pv = tp * df["volume"]
    out = pd.Series(index=df.index, dtype=float)
    if anchor_pos < 0 or anchor_pos >= len(df):
        return out
    pv_a = pv.iloc[anchor_pos:].cumsum()
    v_a  = df["volume"].iloc[anchor_pos:].cumsum().replace(0, np.nan)
    out.iloc[anchor_pos:] = pv_a / v_a
    return out

# ---------------------------------------------------------------------
# Volume Profile (VAL/VAH/POC) with recent-volume weighting
# ---------------------------------------------------------------------
@dataclass
class VPLevels:
    poc: Optional[float]; val: Optional[float]; vah: Optional[float]

def _exp_weights(n: int, alpha: float) -> np.ndarray:
    if alpha <= 0 or n <= 0: return np.ones(n)
    w = np.power(1.0 - alpha, np.arange(n-1, -1, -1))
    m = w.mean()
    return w if m == 0 else (w / m)

def compute_vp_levels(dftf: pd.DataFrame, *, bins: int, value_pct: float,
                      use_typical: bool, decay_alpha: float = 0.0) -> VPLevels:
    if dftf.empty: return VPLevels(None, None, None)
    price_series = ((dftf["high"] + dftf["low"] + dftf["close"]) / 3.0) if use_typical else dftf["close"]
    prices = price_series.values.astype(float)
    vols   = dftf["volume"].values.astype(float)
    if len(prices) < 3 or np.nansum(vols) <= 0: return VPLevels(None, None, None)

    w_time = _exp_weights(len(vols), decay_alpha)
    vol_w  = vols * w_time

    lo, hi = np.nanmin(prices), np.nanmax(prices)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo: return VPLevels(None, None, None)

    hist, edges = np.histogram(prices, bins=bins, range=(lo, hi), weights=vol_w)
    if hist.sum() <= 0: return VPLevels(None, None, None)

    poc_idx   = int(np.argmax(hist))
    poc_price = float((edges[poc_idx] + edges[poc_idx+1]) / 2.0)

    order = np.argsort(-hist); cum=0.0; total=float(hist.sum()); chosen=set()
    for idx in order:
        chosen.add(idx); cum += float(hist[idx])
        if cum >= value_pct * total:
            break

    min_edge = min(edges[i]   for i in chosen)
    max_edge = max(edges[i+1] for i in chosen)
    return VPLevels(poc=poc_price, val=float(min_edge), vah=float(max_edge))

def vp_null_reason_for(dftf: pd.DataFrame, *, use_typical: bool) -> Optional[str]:
    if dftf.empty or len(dftf) < 3:
        return "no_data_or_too_few_bars"
    price_series = ((dftf["high"] + dftf["low"] + dftf["close"]) / 3.0) if use_typical else dftf["close"]
    if not np.isfinite(price_series.max()) or not np.isfinite(price_series.min()):
        return "invalid_prices"
    if price_series.max() <= price_series.min():
        return "flat_price_range"
    if float(np.nansum(dftf["volume"].values)) <= 0:
        return "zero_volume"
    return None

def bb_null_reason_for(cond_row_ok: bool, *, obv_delta_v: float, vol_pct_v: float, vol_pct_thr: int,
                       csv_z_v: float, csv_z_thr: float, obv_thr: float,
                       have_any_run: bool, have_enough_bars: bool) -> str:
    if not have_enough_bars:
        return "insufficient_bars"
    if have_any_run:
        return "no_pickable_run"
    fails = []
    if abs(obv_delta_v) < obv_thr: fails.append(f"obv_delta<{obv_thr:.2f} (got {obv_delta_v:.2f})")
    if vol_pct_v < vol_pct_thr:     fails.append(f"vol_pct<{vol_pct_thr} (got {vol_pct_v:.1f})")
    if csv_z_v < csv_z_thr:         fails.append(f"csv_z<{csv_z_thr:.2f} (got {csv_z_v:.2f})")
    return "gate_fail: " + "; ".join(fails or ["unknown"])

# ---------------------------------------------------------------------
# Hybrid Black Block (OBVΔ + CSV z + Volume percentile) with momentum & location
# ---------------------------------------------------------------------
@dataclass
class BBMetrics:
    zone_top: Optional[float]
    zone_bot: Optional[float]
    block_len: int
    vol_pct: Optional[float]
    csv_z: Optional[float]
    obv_delta: Optional[float]
    score: Optional[float]

@dataclass
class BBResult:
    metrics: BBMetrics
    meets_all: bool
    end_ts: Optional[pd.Timestamp]
    reason: Optional[str] = None

def _rolling_std(s: pd.Series, win: int) -> float:
    if win <= 2: return float(s.std(ddof=1) or 0.0)
    return float(s.rolling(win, min_periods=max(5, win//5)).std(ddof=1).iloc[-1] or 0.0)

def _dynamic_volpct_threshold(dftf: pd.DataFrame, base_low: int, base_high: int) -> int:
    close = dftf["close"]; high=dftf["high"]; low=dftf["low"]
    atr = _atr(close, high, low, span=14)
    atr_pct = (atr / close).clip(lower=0).fillna(0.0)
    mu = atr_pct.rolling(100).mean()
    sd = atr_pct.rolling(100).std(ddof=1).replace(0, np.nan)
    z = float(((atr_pct - mu) / sd).iloc[-1] if len(sd) else 0.0)
    if not np.isfinite(z): z = 0.0
    w = 1.0 / (1.0 + np.exp(-z))
    thr = int(round(base_low * w + base_high * (1.0 - w)))
    return min(max(thr, 20), 95)

def _auto_thresholds(dftf: pd.DataFrame, base: dict, *, cfg: VPBBConfig) -> dict:
    cl, vol = dftf["close"], dftf["volume"]
    win_long = min(len(vol), cfg.std_lookback_days * max(1, int(24*60/5)))
    obv = _obv_series(cl, vol)
    obv_delta = obv.diff().fillna(0.0)
    _ = _rolling_std(vol, win=win_long)
    obv_std = _rolling_std(obv_delta, win=win_long)

    obv_min_scaled = max(float(base["obv_delta_min"]), 1.5 * (obv_std or 1.0))
    pct_len_scaled = max(int(base["vol_ma_len"]), 30)
    vol_pct_thr_dyn = _dynamic_volpct_threshold(dftf, base_low=cfg.dyn_volpct_low, base_high=cfg.dyn_volpct_high)

    return {
        **base,
        "obv_delta_min": obv_min_scaled,
        "vol_ma_len": pct_len_scaled,
        "vol_pct_thr": vol_pct_thr_dyn
    }

def compute_bb_metrics(
    dftf: pd.DataFrame, *,
    vol_ma_len: int, vol_pct_thr: int, csv_z_thr: float,
    obv_delta_min: float, min_blocks: int, pick_best_run: bool, cfg: VPBBConfig
) -> BBResult:
    if dftf.empty or len(dftf) < max(vol_ma_len+2, 10):
        end_ts = dftf.index[-1] if len(dftf) else None
        return BBResult(BBMetrics(None,None,0,None,None,None,None), False, end_ts, reason="insufficient_bars")

    close = dftf["close"]; open_ = dftf["open"]
    high  = dftf["high"];  low   = dftf["low"];   vol = dftf["volume"]

    obv = _obv_series(close, vol); obv_delta = obv.diff()
    csv = (close - open_) * vol

    vol_ma = vol.rolling(vol_ma_len, min_periods=5).mean()

    vol_pct_series = vol.rolling(vol_ma_len, min_periods=5).apply(
        lambda a: 100.0 * (np.sum(a <= a[-1]) / max(1, len(a))), raw=True
    )

    csv_mean = csv.rolling(vol_ma_len, min_periods=5).mean()
    csv_std  = csv.rolling(vol_ma_len, min_periods=5).std(ddof=1).replace(0, np.nan)
    csv_z_series = (csv - csv_mean) / csv_std

    cond = (obv_delta.abs() >= obv_delta_min) & (vol_pct_series >= vol_pct_thr) & (csv_z_series >= csv_z_thr)

    runs: List[Tuple[float,int,float,float,int]] = []  # (score, len, top, bot, endpos)
    in_run = False; block_len = 0; run_top = -math.inf; run_bot = math.inf

    def _score_at(i_end: int, blk_len: int, top: float, bot: float) -> float:
        v = float(vol.iloc[i_end]); v_ma_v = float(vol_ma.iloc[i_end] or 0.0)
        csv_z_v = float(csv_z_series.iloc[i_end] or 0.0)
        obv_d_v = float(obv_delta.iloc[i_end] or 0.0)

        vol_score      = min(5.0, (v / (v_ma_v or 1.0)) * 5.0) if v_ma_v > 0 else 0.0
        delta_score    = min(3.0, abs(obv_d_v) / (obv_delta_min or 1.0) * 3.0)
        duration_score = min(2.0, blk_len / (min_blocks or 1) * 2.0)

        dir_sign = 1.0 if csv_z_v >= 0 else -1.0
        dir_bonus = 1.0 if (np.sign(obv_d_v) == dir_sign and abs(obv_d_v) >= obv_delta_min) else 0.0

        # momentum (RSI + MACD diff)
        rsi_series = _rsi(close, cfg.rsi_len)
        _, _, macd_diff = _macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_sig)

        rsi_v = _safe_iloc(rsi_series, i_end, 50.0)
        macd_diff_v = _safe_iloc(macd_diff,  i_end, 0.0)

        if dir_sign > 0:
            mom_bonus = (1.0 if rsi_v > 55 else 0.0) + (1.0 if macd_diff_v > 0 else 0.0)
        else:
            mom_bonus = (1.0 if rsi_v < 45 else 0.0) + (1.0 if macd_diff_v < 0 else 0.0)
        mom_bonus = min(2.0, mom_bonus)

        loc_bonus = 0.0
        for span in (cfg.ema_refs or []):
            ema_v = float(_ema(close, span).iloc[i_end] or close.iloc[i_end])
            dist = (close.iloc[i_end] - ema_v) / (ema_v or 1.0)
            near = 1.0 if abs(dist) < 0.01 else 0.0
            aligned = 1.0 if (dir_sign > 0 and close.iloc[i_end] >= ema_v) or (dir_sign < 0 and close.iloc[i_end] <= ema_v) else 0.0
            loc_bonus += 0.5 * near + 0.5 * aligned
        loc_bonus = min(2.0, loc_bonus)

        return round(vol_score + delta_score + duration_score + dir_bonus + mom_bonus + loc_bonus, 2)

    for i in range(len(dftf)):
        if bool(cond.iloc[i]):
            in_run = True
            block_len += 1
            run_top = max(run_top, float(high.iloc[i]))
            run_bot = min(run_bot, float(low.iloc[i]))
        else:
            if in_run and block_len >= min_blocks:
                s = _score_at(i-1, block_len, run_top, run_bot)
                runs.append((s, block_len, run_top, run_bot, i-1))
            in_run = False; block_len = 0; run_top = -math.inf; run_bot = math.inf

    endpos_diag = len(dftf) - 1
    have_enough_bars = True
    obv_d_last   = float(obv_delta.iloc[endpos_diag] or 0.0)
    vol_pct_last = float(vol_pct_series.iloc[endpos_diag] or 0.0)
    csv_z_last   = float(csv_z_series.iloc[endpos_diag] or 0.0)
    cond_ok_last = bool(cond.iloc[endpos_diag])

    diag = BBMetrics(
        zone_top=None, zone_bot=None, block_len=0,
        vol_pct=vol_pct_last, csv_z=csv_z_last, obv_delta=obv_d_last,
        score=0.0
    )

    if not runs:
        why = bb_null_reason_for(
            cond_row_ok=cond_ok_last,
            obv_delta_v=obv_d_last, vol_pct_v=vol_pct_last, vol_pct_thr=vol_pct_thr,
            csv_z_v=csv_z_last, csv_z_thr=csv_z_thr,
            obv_thr=obv_delta_min, have_any_run=False, have_enough_bars=have_enough_bars
        )
        return BBResult(diag, False, dftf.index[endpos_diag], reason=why)

    pick = max(runs, key=lambda t: t[0]) if pick_best_run else runs[-1]
    score, blk_len, top, bot, endpos = pick
    diag.block_len = blk_len
    diag.score = score

    return BBResult(
        BBMetrics(zone_top=float(top), zone_bot=float(bot), block_len=int(blk_len),
                  vol_pct=diag.vol_pct, csv_z=diag.csv_z, obv_delta=diag.obv_delta, score=score),
        True,
        dftf.index[endpos],
        reason=None
    )

# ---------------------------------------------------------------------
# Upsert into frames (single-row helper)
# ---------------------------------------------------------------------
def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        # handle pd.NA or other NA types
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def _upsert_vpbb_frames_row(
    symbol: str, kind: str, tf: str, ts: datetime,
    *,
    vp_val: Optional[float], vp_vah: Optional[float], vp_poc: Optional[float],
    bb_zone_top: Optional[float], bb_zone_bot: Optional[float], bb_score: Optional[float],
    diag: Optional[Dict[str, float]],
    run_id: str, source: str,
    vp_reason: Optional[str] = None,
    bb_reason: Optional[str] = None,
) -> int:
    tbl = _frames_table(kind)

    cols = [
        "symbol",
        "interval",
        "ts",
        "run_id",
        "source",
        "vp_val",
        "vp_vah",
        "vp_poc",
        "bb_zone_top",
        "bb_zone_bot",
        "bb_score",
        "bb_diag_vol_pct",
        "bb_diag_csv_z",
        "bb_diag_obv_delta",
        "bb_diag_block_len",
        "bb_diag_vwap",
        "vp_reason",
        "bb_reason",
    ]

    vals = [
        symbol,
        tf,
        ts,
        run_id,
        source,
        _safe_float(vp_val),
        _safe_float(vp_vah),
        _safe_float(vp_poc),
        _safe_float(bb_zone_top),
        _safe_float(bb_zone_bot),
        _safe_float(bb_score),
        _safe_float((diag or {}).get("vol_pct")),
        _safe_float((diag or {}).get("csv_z")),
        _safe_float((diag or {}).get("obv_delta")),
        _safe_float((diag or {}).get("block_len")),
        _safe_float((diag or {}).get("vwap")),
        vp_reason,
        bb_reason,
    ]

    set_list = """
        run_id             = EXCLUDED.run_id,
        source             = EXCLUDED.source,
        vp_val             = EXCLUDED.vp_val,
        vp_vah             = EXCLUDED.vp_vah,
        vp_poc             = EXCLUDED.vp_poc,
        bb_zone_top        = EXCLUDED.bb_zone_top,
        bb_zone_bot        = EXCLUDED.bb_zone_bot,
        bb_score           = EXCLUDED.bb_score,
        bb_diag_vol_pct    = EXCLUDED.bb_diag_vol_pct,
        bb_diag_csv_z      = EXCLUDED.bb_diag_csv_z,
        bb_diag_obv_delta  = EXCLUDED.bb_diag_obv_delta,
        bb_diag_block_len  = EXCLUDED.bb_diag_block_len,
        bb_diag_vwap       = EXCLUDED.bb_diag_vwap,
        vp_reason          = EXCLUDED.vp_reason,
        bb_reason          = EXCLUDED.bb_reason,
        updated_at         = NOW()
    """

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {tbl} ({", ".join(cols)})
            VALUES ({", ".join(["%s"] * len(cols))})
            ON CONFLICT (symbol, interval, ts)
            DO UPDATE SET {set_list}
            """,
            vals,
        )
        conn.commit()
        return 1

# ---------------------------------------------------------------------
# NEW: Bulk upsert helper using same columns/order
# ---------------------------------------------------------------------
def _bulk_upsert_vpbb_frames(kind: str, rows: List[tuple]) -> int:
    """
    Bulk version of _upsert_vpbb_frames_row.
    Expects tuples in the same column order as _upsert_vpbb_frames_row.
    """
    if not rows:
        return 0

    tbl = _frames_table(kind)

    cols = [
        "symbol",
        "interval",
        "ts",
        "run_id",
        "source",
        "vp_val",
        "vp_vah",
        "vp_poc",
        "bb_zone_top",
        "bb_zone_bot",
        "bb_score",
        "bb_diag_vol_pct",
        "bb_diag_csv_z",
        "bb_diag_obv_delta",
        "bb_diag_block_len",
        "bb_diag_vwap",
        "vp_reason",
        "bb_reason",
    ]

    set_list = """
        run_id             = EXCLUDED.run_id,
        source             = EXCLUDED.source,
        vp_val             = EXCLUDED.vp_val,
        vp_vah             = EXCLUDED.vp_vah,
        vp_poc             = EXCLUDED.vp_poc,
        bb_zone_top        = EXCLUDED.bb_zone_top,
        bb_zone_bot        = EXCLUDED.bb_zone_bot,
        bb_score           = EXCLUDED.bb_score,
        bb_diag_vol_pct    = EXCLUDED.bb_diag_vol_pct,
        bb_diag_csv_z      = EXCLUDED.bb_diag_csv_z,
        bb_diag_obv_delta  = EXCLUDED.bb_diag_obv_delta,
        bb_diag_block_len  = EXCLUDED.bb_diag_block_len,
        bb_diag_vwap       = EXCLUDED.bb_diag_vwap,
        vp_reason          = EXCLUDED.vp_reason,
        bb_reason          = EXCLUDED.bb_reason,
        updated_at         = NOW()
    """

    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(
            cur,
            f"""
            INSERT INTO {tbl} ({", ".join(cols)})
            VALUES %s
            ON CONFLICT (symbol, interval, ts)
            DO UPDATE SET {set_list}
            """,
            rows,
            page_size=1000,
        )
        conn.commit()

    return len(rows)

# ---------------------------------------------------------------------
# Old writer (kept for compatibility, not used in new process_symbol)
# ---------------------------------------------------------------------
def _write_vp_bb(
    symbol: str, kind: str, tf: str, ts: datetime,
    vp: VPLevels, bbres: BBResult, cfg: VPBBConfig,
    diag_extras: Optional[Dict[str, float]] = None,
    vp_reason: Optional[str] = None,
    bb_reason: Optional[str] = None,
) -> int:
    bb = bbres.metrics
    diag = {
        "vol_pct":   float(bb.vol_pct or 0.0) if bb.vol_pct is not None else None,
        "csv_z":     float(bb.csv_z  or 0.0) if bb.csv_z  is not None else None,
        "obv_delta": float(bb.obv_delta or 0.0) if bb.obv_delta is not None else None,
        "block_len": float(bb.block_len or 0.0),
    }
    if diag_extras:
        if "vwap" in diag_extras: diag["vwap"] = diag_extras["vwap"]
        if "avwap" in diag_extras: diag["avwap"] = diag_extras["avwap"]

    return _upsert_vpbb_frames_row(
        symbol, kind, tf, ts,
        vp_val=_safe_float(vp.val),
        vp_vah=_safe_float(vp.vah),
        vp_poc=_safe_float(vp.poc),
        bb_zone_top=_safe_float(bb.zone_top),
        bb_zone_bot=_safe_float(bb.zone_bot),
        bb_score=_safe_float(bb.score),
        diag=diag,
        run_id=cfg.run_id,
        source=cfg.source,
        vp_reason=vp_reason,
        bb_reason=bb_reason
    )

# ---------------------------------------------------------------------
# Driver (per symbol) — writes TAIL of bars per TF with reasons
# ---------------------------------------------------------------------
def process_symbol(symbol: str, *, cfg: Optional[VPBBConfig] = None, df: Optional[pd.DataFrame] = None, start_dt: Optional[datetime] = None) -> int:
    """
    HYBRID VERSION:
    - Keeps original VP/BB math, gating, diagnostics and reasons.
    - Changes ONLY the I/O pattern: per-TF per-symbol bulk upserts instead of per-row.
    - If 'df' is provided (15m OHLCV), it uses that instead of reloading from DB.
    - If 'start_dt' is provided, processes data from that point (incremental).
    """
    cfg = cfg or load_vpbb_cfg()

    # single-kind or both
    kinds_to_run = [cfg.market_kind] if cfg.market_kind in ("spot","futures") else ["spot","futures"]

    total = 0
    for kind in kinds_to_run:
        if df is not None and not df.empty:
            df15 = df
        else:
            df15 = load_intra(symbol, kind, lookback_days=cfg.lookback_days)

        if df15.empty:
            print(f"⚠️ {kind}:{symbol} no intraday data in last {cfg.lookback_days}d")
            continue

        for tf in cfg.tf_list:
            if tf not in TF_TO_OFFSET:
                continue

            dftf = resample(df15, tf)
            if dftf.empty:
                continue

            base = {
                "vol_ma_len": cfg.vol_ma_len,
                "vol_pct_thr": cfg.vol_pct_thr,
                "csv_z_thr": cfg.csv_z_thr,
                "obv_delta_min": cfg.obv_delta_min,
                "min_blocks": cfg.min_blocks
            }

            # --- Incremental / Tail Logic ---
            # Check specifically for VPBB last run to detect if we need a full backfill
            # even if the global 'start_dt' (from indicators) says we are up to date.
            last_internal_ts = _get_last_vpbb_ts(symbol, kind, tf)

            if last_internal_ts is None:
                # Never ran VPBB for this TF? Force full backfill.
                start_ts = None
            elif start_dt:
                # We have history, so respect the incremental start_dt (global gate)
                last_ts = start_dt
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=TZ)
                start_ts = last_ts - timedelta(hours=4)
            else:
                # Fallback to internal TS if no start_dt provided
                last_ts = last_internal_ts
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=TZ)
                start_ts = last_ts - timedelta(hours=4)

            # 3. Select indices to process
            tail_n = max(1, int(cfg.backfill_bars))
            if start_ts:
                # Process from start_ts onwards
                idxs = dftf.index[dftf.index >= start_ts]
                # FALLBACK: If index selection returns empty (e.g. data is slightly behind start_ts),
                # force at least the last few bars to ensure continuity if dftf is not empty.
                if len(idxs) == 0 and not dftf.empty:
                    # e.g. check if start_ts is ahead of last data point
                    # Or if start_ts is valid but dftf just doesn't have new bars yet
                    # We'll process the very last bar just in case to keep 'last_updated' fresh.
                    idxs = dftf.index[-1:]
            else:
                # No start_ts (never run before) -> Process ALL available data, not just tail_n.
                # tail_n is only for "live" optimization. First run must backfill.
                idxs = dftf.index

            if len(idxs) == 0:
                continue

            # Safety: For large TFs like 240m, ensure we have enough history for the first calculation in this batch
            if tf in ("120m", "240m") and len(dftf) < 10:
                # Not enough bars to do meaningful BB/VP
                # But we should write NULLs or status if possible, or just skip.
                # Currently we skip, which leaves NULLs in the database if rows were created by 'classic' worker.
                # Let's try to process what we have if it's at least minimal.
                pass

            wrote_tf = 0

            # NEW: buffers for bulk upsert
            zone_rows: List[tuple] = []
            main_rows: List[tuple] = []

            for ts_i in idxs:
                # compute on history up to ts_i
                hist = dftf.loc[:ts_i]
                if len(hist) < 10:
                    continue
                tuned = _auto_thresholds(hist, base, cfg=cfg) if cfg.auto_scale else base

                # VP + BB on that window
                vp = compute_vp_levels(
                    hist, bins=cfg.vp_bins, value_pct=cfg.vp_value_pct,
                    use_typical=cfg.use_typical_price, decay_alpha=cfg.vp_decay_alpha
                )
                bbres = compute_bb_metrics(
                    hist,
                    vol_ma_len=tuned["vol_ma_len"],
                    vol_pct_thr=tuned["vol_pct_thr"],
                    csv_z_thr=tuned["csv_z_thr"],
                    obv_delta_min=tuned["obv_delta_min"],
                    min_blocks=tuned["min_blocks"],
                    pick_best_run=cfg.pick_best_run,
                    cfg=cfg
                )

                # Build reasons for this timestamp (same logic as old)
                vp_reason = None
                if vp.poc is None and vp.val is None and vp.vah is None:
                    vp_reason = vp_null_reason_for(hist, use_typical=cfg.use_typical_price)
                bb_reason = None if bbres.meets_all else (bbres.reason or "no_block")

                # Diagnostics: VWAP & Anchored VWAP at ts_i (same as old)
                diag_extras: Dict[str, float] = {}
                try:
                    vwap = _vwap(hist)
                    diag_extras["vwap"] = float(vwap.iloc[-1] or np.nan)
                except Exception:
                    pass
                try:
                    avwap_last = None
                    if cfg.vwap_anchor_mode and cfg.vwap_anchor_mode != "none":
                        if cfg.vwap_anchor_mode == "days":
                            anchor_pos = max(0, len(hist)-1 - max(1, cfg.vwap_anchor_days))
                        else:  # "swing"
                            anchor_pos = _anchor_index_by_swing(hist["close"])
                        av = _anchored_vwap(hist, anchor_pos)
                        if len(av) and np.isfinite(av.iloc[-1] or np.nan):
                            avwap_last = float(av.iloc[-1])
                    diag_extras["avwap"] = avwap_last
                except Exception:
                    pass

                ts_write = pd.to_datetime(ts_i, utc=True).to_pydatetime()
                bb = bbres.metrics

                # --- Zone-level mirror row (same as old _upsert_vpbb_frames_row call with diag=None, source override) ---
                zone_rows.append((
                    symbol,
                    tf,
                    ts_write,
                    os.getenv("RUN_ID", cfg.run_id),
                    "vp_bb_zone_levels",
                    _safe_float(vp.val),
                    _safe_float(vp.vah),
                    _safe_float(vp.poc),
                    _safe_float(bb.zone_top),
                    _safe_float(bb.zone_bot),
                    _safe_float(bb.score),
                    None,  # bb_diag_vol_pct
                    None,  # bb_diag_csv_z
                    None,  # bb_diag_obv_delta
                    None,  # bb_diag_block_len
                    None,  # bb_diag_vwap
                    vp_reason,
                    bb_reason,
                ))

                # --- Main VPBB row with diagnostics (mirrors _write_vp_bb) ---
                diag = {
                    "vol_pct":   float(bb.vol_pct or 0.0) if bb.vol_pct is not None else None,
                    "csv_z":     float(bb.csv_z  or 0.0) if bb.csv_z  is not None else None,
                    "obv_delta": float(bb.obv_delta or 0.0) if bb.obv_delta is not None else None,
                    "block_len": float(bb.block_len or 0.0),
                }
                if diag_extras:
                    if "vwap" in diag_extras: diag["vwap"] = diag_extras["vwap"]
                    if "avwap" in diag_extras: diag["avwap"] = diag_extras["avwap"]

                main_rows.append((
                    symbol,
                    tf,
                    ts_write,
                    cfg.run_id,
                    cfg.source,
                    _safe_float(vp.val),
                    _safe_float(vp.vah),
                    _safe_float(vp.poc),
                    _safe_float(bb.zone_top),
                    _safe_float(bb.zone_bot),
                    _safe_float(bb.score),
                    _safe_float(diag.get("vol_pct")),
                    _safe_float(diag.get("csv_z")),
                    _safe_float(diag.get("obv_delta")),
                    _safe_float(diag.get("block_len")),
                    _safe_float(diag.get("vwap")),
                    vp_reason,
                    bb_reason,
                ))

            # --- Bulk write for this TF ---
            try:
                _bulk_upsert_vpbb_frames(kind, zone_rows)   # zone mirror rows (not counted in total)
            except Exception as e:
                print(f"⚠️ zone_levels bulk upsert failed for {symbol} {kind} {tf}: {e}")

            main_written = _bulk_upsert_vpbb_frames(kind, main_rows)
            wrote_tf += main_written
            total += main_written

            print(f"✅ {kind}:{symbol} {tf} → wrote {wrote_tf} row(s) (tail={tail_n})")

    if total == 0:
        print(f"ℹ️ {symbol} up-to-date / no VPBB metrics to write")
    return total

# ---------------------------------------------------------------------
# Batch entry
# ---------------------------------------------------------------------
def run(symbols: Optional[List[str]] = None, *, kind: Optional[str] = None,
        uid: Optional[str] = None, status_cb=None, df: Optional[pd.DataFrame] = None, start_dt: Optional[datetime] = None) -> Dict[str, object]:
    """
    Batch VP+BB. Caller (indicators_worker) provides the symbol list.
    - kind: override market_kind from cfg ('spot' or 'futures')
    - df: Optional pre-loaded 15m dataframe. ONLY valid if len(symbols)==1.
    - start_dt: Optional timestamp to start incremental processing from.
    Returns {"rows": <int>, "last_ts": <datetime|None>}
    """
    cfg = load_vpbb_cfg()
    if kind:
        cfg.market_kind = kind

    if not symbols:
        return {"rows": 0, "last_ts": None}

    if df is not None and len(symbols) > 1:
        print("⚠️ [VPBB] 'df' argument provided but multiple symbols passed. Ignoring df to avoid data mismatch.")
        df = None

    if status_cb: status_cb("ZON_RESAMPLING")

    total = 0
    last_ts = None
    for s in symbols:
        try:
            if status_cb: status_cb("ZON_COMPUTING_PROFILE")
            if status_cb: status_cb("ZON_COMPUTING_BB")
            # Pass df only if it's the correct context (though we already checked len=1)
            n = process_symbol(s, cfg=cfg, df=df, start_dt=start_dt)
            total += n
            if status_cb: status_cb("ZON_WRITING")
            last_ts = last_ts or datetime.now(TZ)
        except Exception as e:
            print(f"❌ {cfg.market_kind}:{s} error → {e}")

    if status_cb: status_cb("ZON_OK", last_ts)
    return {"rows": total, "last_ts": last_ts}
