# scheduler/update_vp_bb.py
from __future__ import annotations

import os, math, json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras as pgx
import configparser

from utils.db import get_db_connection

TZ = timezone.utc
DEFAULT_INI = os.getenv("INDICATORS_INI", "indicators.ini")

# ------------------------
# Config loading (INI + env overrides)
# ------------------------

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
    auto_scale: bool  # adaptive thresholds gate

    # --- New knobs (backwards compatible defaults) ---
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
    vp_decay_alpha: float     # 0 disables recent-volume overweight

def load_vpbb_cfg(ini_path: str = DEFAULT_INI) -> VPBBConfig:
    dflt = {
        "lookback_days": 7,
        "tf_list": "25m,65m,125m",
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
        "regime_lookback_days": 30,
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
    market_kind   = _get_env("VPBB_KIND", g(vb, "market_kind") or dflt["market_kind"]) or dflt["market_kind"]
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
    rsi_len = _as_int(_get_env("BB_RSI_LEN", g(vb,"rsi_len")), dflt["rsi_len"])
    macd_fast = _as_int(_get_env("BB_MACD_FAST", g(vb,"macd_fast")), dflt["macd_fast"])
    macd_slow = _as_int(_get_env("BB_MACD_SLOW", g(vb,"macd_slow")), dflt["macd_slow"])
    macd_sig  = _as_int(_get_env("BB_MACD_SIG",  g(vb,"macd_sig")),  dflt["macd_sig"])
    ema_refs_raw = _as_list_csv(_get_env("BB_EMA_REFS", g(vb,"ema_refs")), dflt["ema_refs"].split(","))
    ema_refs = [int(x) for x in ema_refs_raw if str(x).isdigit()]
    dyn_volpct_low  = _as_int(_get_env("BB_DYN_VOLPCT_LOW",  g(vb,"dyn_volpct_low")),  dflt["dyn_volpct_low"])
    dyn_volpct_high = _as_int(_get_env("BB_DYN_VOLPCT_HIGH", g(vb,"dyn_volpct_high")), dflt["dyn_volpct_high"])
    vwap_anchor_mode = _get_env("BB_VWAP_ANCHOR_MODE", g(vb,"vwap_anchor_mode")) or dflt["vwap_anchor_mode"]
    vwap_anchor_days = _as_int(_get_env("BB_VWAP_ANCHOR_DAYS", g(vb,"vwap_anchor_days")), dflt["vwap_anchor_days"])
    vp_decay_alpha   = _as_float(_get_env("VP_DECAY_ALPHA", g(vb,"vp_decay_alpha")), dflt["vp_decay_alpha"])

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
        vp_decay_alpha=vp_decay_alpha
    )

# ------------------------
# Helpers
# ------------------------

TF_TO_OFFSET = {
    "5m":"5min","15m":"15min","25m":"25min","30m":"30min",
    "65m":"65min","125m":"125min","250m":"250min"
}

def _table_name(kind: str) -> str:
    return "market.futures_candles" if kind == "futures" else "market.spot_candles"

def _now_utc() -> datetime: return datetime.now(TZ)
def _cutoff_days(n: int) -> datetime: return _now_utc() - timedelta(days=n)

def _dbg_head(tag: str, df: pd.DataFrame, n: int = 3):
    if df.empty:
        print(f"[DBG] {tag}: (empty)")
        return
    print(f"\n[DBG] {tag} dtypes:\n{df.dtypes}")
    print(f"[DBG] {tag} head:\n{df.head(n)}\n")

# ------------------------
# IO: load 5m candles
# ------------------------
def load_5m(symbol: str, kind: str, *, lookback_days: int) -> pd.DataFrame:
    tbl = _table_name(kind); cutoff = _cutoff_days(lookback_days)
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ts, open, high, low, close, volume
              FROM {tbl}
             WHERE symbol=%s
               AND interval='5m'
               AND ts >= %s
             ORDER BY ts ASC
            """,
            (symbol, cutoff)
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.set_index("ts", inplace=True)
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df.dropna()

def resample(df5: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "5m": return df5.copy()
    rule = TF_TO_OFFSET.get(tf)
    if rule is None or df5.empty: return pd.DataFrame()
    out = pd.DataFrame({
        "open":   df5["open"]  .resample(rule, label="right", closed="right").first(),
        "high":   df5["high"]  .resample(rule, label="right", closed="right").max(),
        "low":    df5["low"]   .resample(rule, label="right", closed="right").min(),
        "close":  df5["close"] .resample(rule, label="right", closed="right").last(),
        "volume": df5["volume"].resample(rule, label="right", closed="right").sum(),
    }).dropna(how="any")
    return out

# ------------------------
# Volume Profile (VAL/VAH/POC) with recent-volume weighting
# ------------------------
@dataclass
class VPLevels:
    poc: Optional[float]; val: Optional[float]; vah: Optional[float]

def _exp_weights(n: int, alpha: float) -> np.ndarray:
    if alpha <= 0 or n <= 0: return np.ones(n)
    # newer bars have bigger weight; normalize to mean ~1.0
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

# ------------------------
# Hybrid Black Block (OBVΔ + CSV z + Volume percentile) with momentum & location
# ------------------------
@dataclass
class BBMetrics:
    zone_top: Optional[float]
    zone_bot: Optional[float]
    block_len: int
    vol_pct: Optional[float]
    csv_z: Optional[float]
    obv_delta: Optional[float]
    score: Optional[float]  # 0..~14 with bonuses; you can still treat >=score_hi as strong

@dataclass
class BBResult:
    metrics: BBMetrics
    meets_all: bool
    end_ts: Optional[pd.Timestamp]

def _obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum()

def _rolling_tr(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    a = (h - l).abs()
    b = (h - prev_c).abs()
    d = (l - prev_c).abs()
    return pd.concat([a,b,d], axis=1).max(axis=1)

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, win: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = (up.rolling(win, min_periods=max(2, win//2)).mean()) / (
         dn.rolling(win, min_periods=max(2, win//2)).mean().replace(0, np.nan))
    out = 100 - 100/(1+rs)
    return out.fillna(method="bfill").fillna(50.0)

def _macd(close: pd.Series, fast: int, slow: int, sig: int):
    ema_f = _ema(close, fast); ema_s = _ema(close, slow)
    macd = ema_f - ema_s; macd_sig = _ema(macd, sig); macd_diff = macd - macd_sig
    return macd, macd_sig, macd_diff

def _atr(close: pd.Series, high: pd.Series, low: pd.Series, span: int = 14):
    prev_c = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/span, adjust=False).mean()

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
    # high vol => lean towards lower percentile (easier to qualify spikes)
    w = 1.0 / (1.0 + np.exp(-z))  # 0..1 increases in high-vol regimes
    thr = int(round(base_low * w + base_high * (1.0 - w)))
    return min(max(thr, 20), 95)

def _auto_thresholds(dftf: pd.DataFrame, base: dict, *, cfg: VPBBConfig) -> dict:
    # base includes: vol_ma_len, vol_pct_thr, csv_z_thr, obv_delta_min, min_blocks
    cl, hi, lo, vol = dftf["close"], dftf["high"], dftf["low"], dftf["volume"]

    # long-window stds (use approximate mapping: days * (24*60/TFmin))
    # For robustness, clamp to series length.
    win_long = min(len(vol), cfg.std_lookback_days * max(1, int(24*60/5)))  # rough for 5m base
    obv = _obv_series(cl, vol)
    obv_delta = obv.diff().fillna(0.0)

    vol_std = _rolling_std(vol, win=win_long)
    obv_std = _rolling_std(obv_delta, win=win_long)

    # min OBVΔ scales by its long std
    obv_min_scaled = max(float(base["obv_delta_min"]), 1.5 * (obv_std or 1.0))
    pct_len_scaled = max(int(base["vol_ma_len"]), 30)

    # dynamic volume percentile by regime
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
        return BBResult(BBMetrics(None,None,0,None,None,None,None), False, end_ts)

    close = dftf["close"]; open_ = dftf["open"]
    high  = dftf["high"];  low   = dftf["low"];   vol = dftf["volume"]

    obv = _obv_series(close, vol)
    obv_delta = obv.diff()

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

        # base components
        vol_score      = min(5.0, (v / (v_ma_v or 1.0)) * 5.0) if v_ma_v > 0 else 0.0
        delta_score    = min(3.0, abs(obv_d_v) / (obv_delta_min or 1.0) * 3.0)
        duration_score = min(2.0, blk_len / (min_blocks or 1) * 2.0)

        # direction via csv_z sign
        dir_sign = 1.0 if csv_z_v >= 0 else -1.0
        dir_bonus = 1.0 if (np.sign(obv_d_v) == dir_sign and abs(obv_d_v) >= obv_delta_min) else 0.0

        # momentum (RSI + MACD diff)
        rsi_series = _rsi(close, cfg.rsi_len)
        rsi_v = float(rsi_series.iloc[i_end] or 50.0)
        _, _, macd_diff = _macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_sig)
        macd_diff_v = float(macd_diff.iloc[i_end] or 0.0)
        if dir_sign > 0:
            mom_bonus = (1.0 if rsi_v > 55 else 0.0) + (1.0 if macd_diff_v > 0 else 0.0)
        else:
            mom_bonus = (1.0 if rsi_v < 45 else 0.0) + (1.0 if macd_diff_v < 0 else 0.0)
        mom_bonus = min(2.0, mom_bonus)

        # block location bonus: near & aligned with ref EMAs
        loc_bonus = 0.0
        for span in (cfg.ema_refs or []):
            ema_v = float(_ema(close, span).iloc[i_end] or close.iloc[i_end])
            dist = (close.iloc[i_end] - ema_v) / (ema_v or 1.0)
            near = 1.0 if abs(dist) < 0.01 else 0.0   # within ~1%
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
    diag = BBMetrics(
        zone_top=None, zone_bot=None, block_len=0,
        vol_pct=float(vol_pct_series.iloc[endpos_diag] or 0.0),
        csv_z=float(csv_z_series.iloc[endpos_diag] or 0.0),
        obv_delta=float(obv_delta.iloc[endpos_diag] or 0.0),
        score=0.0
    )

    if not runs:
        return BBResult(diag, False, dftf.index[endpos_diag])

    pick = max(runs, key=lambda t: t[0]) if pick_best_run else runs[-1]
    score, blk_len, top, bot, endpos = pick

    diag.block_len = blk_len
    diag.score = score

    return BBResult(
        BBMetrics(zone_top=float(top), zone_bot=float(bot), block_len=int(blk_len),
                  vol_pct=diag.vol_pct, csv_z=diag.csv_z, obv_delta=diag.obv_delta, score=score),
        True,
        dftf.index[endpos]
    )
# --- ADD: zone_levels upsert -----------------------------------------------

def _get_rsi_mfi_at(cur, symbol: str, tf: str, ts: datetime) -> tuple[Optional[float], Optional[float]]:
    cur.execute("""
        SELECT metric, val
          FROM indicators.values
         WHERE symbol=%s AND market_type='futures' AND interval=%s AND ts=%s
           AND metric IN ('RSI','MFI')
    """, (symbol, tf, ts))
    rsi = mfi = None
    for m, v in cur.fetchall() or []:
        if m == 'RSI': rsi = float(v) if v is not None else None
        elif m == 'MFI': mfi = float(v) if v is not None else None
    return rsi, mfi

def _upsert_zone_levels_row(
    symbol: str, tf: str, ts: datetime,
    vp: VPLevels, bb: BBResult,
) -> int:
    sql = """
      INSERT INTO market.zone_levels
        (symbol, interval, ts,
         val, vah, poc,
         rsi_at_val, mfi_at_val,
         rsi_at_vah, mfi_at_vah,
         bb_zone_top, bb_zone_bot, bb_score, bb_meets_all,
         support_break_flag, resistance_break_flag)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,0)
      ON CONFLICT (symbol, interval, ts) DO UPDATE SET
         val = EXCLUDED.val,
         vah = EXCLUDED.vah,
         poc = EXCLUDED.poc,
         rsi_at_val = EXCLUDED.rsi_at_val,
         mfi_at_val = EXCLUDED.mfi_at_val,
         rsi_at_vah = EXCLUDED.rsi_at_vah,
         mfi_at_vah = EXCLUDED.mfi_at_vah,
         bb_zone_top  = EXCLUDED.bb_zone_top,
         bb_zone_bot  = EXCLUDED.bb_zone_bot,
         bb_score     = EXCLUDED.bb_score,
         bb_meets_all = EXCLUDED.bb_meets_all
    """
    with get_db_connection() as conn, conn.cursor() as cur:
        rsi, mfi = _get_rsi_mfi_at(cur, symbol, tf, ts)
        cur.execute(sql, (
            symbol, tf, ts,
            vp.val, vp.vah, vp.poc,
            rsi, mfi,                 # we store same snapshot for *_at_val/_at_vah
            rsi, mfi,
            (bb.metrics.zone_top if bb.meets_all else None),
            (bb.metrics.zone_bot if bb.meets_all else None),
            (bb.metrics.score    if bb.meets_all else None),
            (1 if bb.meets_all else 0),
        ))
        conn.commit()
        return 1

# ------------------------
# VWAP / Anchored VWAP / POC trend diagnostics
# ------------------------
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
    # choose most recent extreme
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

def _poc_trend(prev_poc: Optional[float], poc: Optional[float]) -> Tuple[float, int]:
    if prev_poc is None or poc is None or not np.isfinite(prev_poc) or not np.isfinite(poc):
        return (0.0, 0)
    delta = float(poc - prev_poc)
    sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
    return (delta, sign)

# ------------------------
# Upsert to indicators.values
# ------------------------
def _upsert_metric(cur, rows: List[tuple]) -> int:
    if not rows: return 0
    pgx.execute_values(cur, """
        INSERT INTO indicators.values
            (symbol, market_type, interval, ts, metric, val, context, run_id, source)
        VALUES %s
        ON CONFLICT (symbol, market_type, interval, ts, metric)
        DO UPDATE SET
            val=EXCLUDED.val,
            context=EXCLUDED.context,
            run_id=EXCLUDED.run_id,
            source=EXCLUDED.source
    """, rows, page_size=1000)
    return len(rows)

def _write_vp_bb(
    symbol: str, kind: str, tf: str, ts: datetime,
    vp: VPLevels, bbres: BBResult, cfg: VPBBConfig,
    diag_extras: Optional[Dict[str, float]] = None
) -> int:
    rows: List[tuple] = []
    ctx_base = {"tf": tf}

    # VP rows
    if vp.poc is not None: rows.append((symbol, kind, tf, ts, "VP.POC", float(vp.poc), json.dumps(ctx_base), cfg.run_id, cfg.source))
    if vp.val is not None: rows.append((symbol, kind, tf, ts, "VP.VAL", float(vp.val), json.dumps(ctx_base), cfg.run_id, cfg.source))
    if vp.vah is not None: rows.append((symbol, kind, tf, ts, "VP.VAH", float(vp.vah), json.dumps(ctx_base), cfg.run_id, cfg.source))

    # POC drift diagnostics (needs previous POC)
    if cfg.write_diagnostics and vp.poc is not None:
        prev_poc = None
        try:
            with get_db_connection() as conn2, conn2.cursor() as cur2:
                cur2.execute("""
                    SELECT val FROM indicators.values
                     WHERE symbol=%s AND market_type=%s AND interval=%s AND metric='VP.POC'
                     ORDER BY ts DESC LIMIT 1
                """, (symbol, kind, tf))
                r = cur2.fetchone()
                if r: prev_poc = float(r[0])
        except Exception:
            prev_poc = None
        d, s = _poc_trend(prev_poc, vp.poc)
        rows.append((symbol, kind, tf, ts, "VP.POC.delta", float(d), json.dumps(ctx_base), cfg.run_id, cfg.source))
        rows.append((symbol, kind, tf, ts, "VP.POC.trend", float(s), json.dumps(ctx_base), cfg.run_id, cfg.source))

    # BB rows
    bb = bbres.metrics
    if bbres.meets_all and bb.zone_top is not None:
        rows.append((symbol, kind, tf, ts, "BB.zone_top", float(bb.zone_top), json.dumps(ctx_base), cfg.run_id, cfg.source))
        rows.append((symbol, kind, tf, ts, "BB.zone_bot", float(bb.zone_bot), json.dumps(ctx_base), cfg.run_id, cfg.source))
        rows.append((symbol, kind, tf, ts, "BB.score", float(bb.score or 0.0),
                     json.dumps(ctx_base | {
                         "vol_ma_len": cfg.vol_ma_len, "pct_thr": cfg.vol_pct_thr,
                         "csv_z_thr": cfg.csv_z_thr, "obv_delta_min": cfg.obv_delta_min,
                         "min_blocks": cfg.min_blocks
                     }), cfg.run_id, cfg.source))

    # Diagnostics (always if enabled)
    if cfg.write_diagnostics:
        rows.append((symbol, kind, tf, ts, "BB.diag.vol_pct",   float(bb.vol_pct or 0.0), json.dumps(ctx_base | {"pct_len": cfg.vol_ma_len}), cfg.run_id, cfg.source))
        rows.append((symbol, kind, tf, ts, "BB.diag.csv_z",     float(bb.csv_z  or 0.0), json.dumps(ctx_base | {"z_len": cfg.vol_ma_len}),  cfg.run_id, cfg.source))
        rows.append((symbol, kind, tf, ts, "BB.diag.obv_delta", float(bb.obv_delta or 0.0), json.dumps(ctx_base), cfg.run_id, cfg.source))
        rows.append((symbol, kind, tf, ts, "BB.diag.block_len", float(bb.block_len or 0.0), json.dumps(ctx_base), cfg.run_id, cfg.source))

        # VWAP / Anchored VWAP diagnostics if provided
        if diag_extras is not None:
            if "vwap" in diag_extras and np.isfinite(diag_extras["vwap"]):
                rows.append((symbol, kind, tf, ts, "BB.diag.vwap", float(diag_extras["vwap"]), json.dumps(ctx_base), cfg.run_id, cfg.source))
            if "avwap" in diag_extras and diag_extras["avwap"] is not None and np.isfinite(diag_extras["avwap"]):
                rows.append((symbol, kind, tf, ts, "BB.diag.avwap", float(diag_extras["avwap"]), json.dumps(ctx_base | {"anchor_mode": cfg.vwap_anchor_mode}), cfg.run_id, cfg.source))

        rows.append((symbol, kind, tf, ts, "BB.diag.meets_all", 1.0 if bbres.meets_all else 0.0, json.dumps(ctx_base), cfg.run_id, cfg.source))

    if not rows: return 0
    with get_db_connection() as conn, conn.cursor() as cur:
        n = _upsert_metric(cur, rows)
        conn.commit()
        return n

# ------------------------
# Driver (per symbol)
# ------------------------
def process_symbol(symbol: str, *, cfg: Optional[VPBBConfig] = None) -> int:
    cfg = cfg or load_vpbb_cfg()
    df5 = load_5m(symbol, cfg.market_kind, lookback_days=cfg.lookback_days)
    if df5.empty:
        print(f"⚠️ {cfg.market_kind}:{symbol} no 5m data in last {cfg.lookback_days}d")
        return 0

    total = 0
    for tf in cfg.tf_list:
        if tf not in TF_TO_OFFSET: continue
        dftf = resample(df5, tf)
        if dftf.empty: continue

        # auto-scale thresholds per TF window if enabled
        base = {
            "vol_ma_len": cfg.vol_ma_len,
            "vol_pct_thr": cfg.vol_pct_thr,
            "csv_z_thr": cfg.csv_z_thr,
            "obv_delta_min": cfg.obv_delta_min,
            "min_blocks": cfg.min_blocks
        }
        tuned = _auto_thresholds(dftf, base, cfg=cfg) if cfg.auto_scale else base

        # Volume profile (weighted by recent activity if vp_decay_alpha>0)
        vp = compute_vp_levels(
            dftf, bins=cfg.vp_bins, value_pct=cfg.vp_value_pct,
            use_typical=cfg.use_typical_price, decay_alpha=cfg.vp_decay_alpha
        )

        # Block detection & score
        bbres = compute_bb_metrics(
            dftf,
            vol_ma_len=tuned["vol_ma_len"],
            vol_pct_thr=tuned["vol_pct_thr"],
            csv_z_thr=tuned["csv_z_thr"],
            obv_delta_min=tuned["obv_delta_min"],
            min_blocks=tuned["min_blocks"],
            pick_best_run=cfg.pick_best_run,
            cfg=cfg
        )

        # Diagnostics: VWAP & Anchored VWAP
        diag_extras: Dict[str, float] = {}
        try:
            vwap = _vwap(dftf)
            diag_extras["vwap"] = float(vwap.iloc[-1] or np.nan)
        except Exception:
            pass

        try:
            avwap_last = None
            if cfg.vwap_anchor_mode and cfg.vwap_anchor_mode.lower() != "none":
                if cfg.vwap_anchor_mode.lower() == "days":
                    anchor_pos = max(0, len(dftf)-1 - max(1, cfg.vwap_anchor_days))
                else:  # "swing"
                    anchor_pos = _anchor_index_by_swing(dftf["close"])
                av = _anchored_vwap(dftf, anchor_pos)
                if len(av) and np.isfinite(av.iloc[-1] or np.nan):
                    avwap_last = float(av.iloc[-1])
            diag_extras["avwap"] = avwap_last
        except Exception:
            pass

        ts_last = (bbres.end_ts or dftf.index[-1]).to_pydatetime().replace(tzinfo=TZ)
        try:
            _upsert_zone_levels_row(symbol, tf, ts_last, vp, bbres)
        except Exception as e:
            print(f"⚠️ zone_levels upsert failed for {symbol} {tf}: {e}")

        n = _write_vp_bb(symbol, cfg.market_kind, tf, ts_last, vp, bbres, cfg, diag_extras=diag_extras)
        total += n
        print(f"✅ {cfg.market_kind}:{symbol} {tf} @ {ts_last} → wrote {n} metric(s)")
    if total == 0:
        print(f"ℹ️ {cfg.market_kind}:{symbol} up-to-date / no metrics to write")
    return total

# ------------------------
# Discovery (symbols to run)
# ------------------------
def discover_symbols_from_webhooks() -> List[str]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT symbol
              FROM webhooks.webhook_alerts
             WHERE status='INDICATOR_PROCESS'
               AND COALESCE(sub_status,'IND_PENDING')='IND_PENDING'
        """)
        rows = cur.fetchall()
    return [r[0] for r in rows or []]

# ------------------------
# Batch entry
# ------------------------
def run(symbols: Optional[List[str]] = None, *, kind: Optional[str] = None,
        uid: Optional[str] = None, status_cb=None) -> Dict[str, object]:
    """
    Optional signature for indicators_worker:
      - kind: override market_kind from cfg (e.g., 'spot' or 'futures')
      - uid/status_cb: if provided, emit ZON_* breadcrumbs
    Returns {"rows": <int>, "last_ts": <datetime|None>}
    """
    cfg = load_vpbb_cfg()
    if kind: cfg.market_kind = kind

    if status_cb: status_cb("ZON_LOADING_DATA")

    if symbols is None:
        symbols = discover_symbols_from_webhooks()
        print(f"🔎 discovered {len(symbols)} symbol(s) from webhooks INDICATOR_PROCESS/IND_PENDING)")

    if status_cb: status_cb("ZON_RESAMPLING_25_65_125")

    total = 0
    last_ts = None
    for s in symbols:
        try:
            if status_cb: status_cb("ZON_COMPUTING_PROFILE")
            if status_cb: status_cb("ZON_COMPUTING_BB")
            n = process_symbol(s, cfg=cfg)
            total += n
            if status_cb: status_cb("ZON_WRITING")
            last_ts = last_ts or datetime.now(TZ)
        except Exception as e:
            print(f"❌ {cfg.market_kind}:{s} error → {e}")

    if status_cb: status_cb("ZON_OK", last_ts)
    return {"rows": total, "last_ts": last_ts}

if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    syms = [a for a in args if not a.startswith("--")]
    out = run(syms or None)
    print(f"🎯 VP+BB wrote {out.get('rows', 0)} metric row(s)")
