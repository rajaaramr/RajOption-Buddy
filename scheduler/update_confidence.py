# scheduler/update_confidence.py
from __future__ import annotations

import os, json, math, configparser, pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg2.extras as pgx

from utils.db import get_db_connection

TZ = timezone.utc
DEFAULT_INI = os.getenv("INDICATORS_INI", "indicators.ini")
MODELS_DIR  = os.getenv("CONF_MODELS_DIR", "models")  # where we look for HMM/NB/Calib artifacts

# -------------------- Config --------------------

TF_TO_OFFSET = {
    "5m": "5min", "15m": "15min", "25m": "25min", "30m": "30min",
    "65m": "65min", "125m": "125min", "250m": "250min"
}

@dataclass
class ConfCfg:
    tfs: List[str]
    tf_weights: List[float]
    lookback_days: int
    states: int
    barrier_atr_k: float
    horizon_bars: Dict[str, int]
    laplace_alpha: float
    # thresholds
    oi_z_hi: float
    vp_poc_atr_thr: float
    squeeze_bw_pct: float
    bb_score_hi: float
    # penalties/bonuses on logit
    lambda_poc: float
    lambda_crowd: float
    lambda_squeeze: float
    # persistence
    run_id: str
    source: str
    # new feature knobs
    vol_delta_lookback: int
    atr_pct_lookback: int
    spread_cap_bps: float  # cap spread feature to avoid explosions (bps of price)

def _as_list(x: str) -> List[str]:
    return [s.strip() for s in x.split(",") if s.strip()]

def _parse_horizon(x: str) -> Dict[str, int]:
    out: Dict[str,int] = {}
    for part in _as_list(x):
        if ":" in part:
            tf, n = part.split(":", 1)
            try: out[tf.strip()] = int(n.strip())
            except: pass
    return out

def load_cfg(ini_path: str = DEFAULT_INI) -> ConfCfg:
    dflt = {
        "tfs": "25m,65m,125m",
        "tf_weights": "0.25,0.35,0.40",
        "lookback_days": 120,
        "states": 3,
        "barrier_atr_k": 1.0,
        "horizon_bars": "25m:8,65m:5,125m:3",
        "laplace_alpha": 2.0,
        "oi_z_hi": 1.5,
        "vp_poc_atr_thr": 0.20,
        "squeeze_bw_pct": 20.0,
        "bb_score_hi": 6.5,
        "lambda_poc": 0.25,
        "lambda_crowd": 0.35,
        "lambda_squeeze": 0.20,
        "run_id": os.getenv("RUN_ID", "conf_run"),
        "source": os.getenv("SRC", "conf_hmm_bayes"),
        # new
        "vol_delta_lookback": 60,   # bars for volume-delta anomaly z
        "atr_pct_lookback": 240,    # bars for ATR percentile
        "spread_cap_bps": 50.0,     # cap spread feature to 50 bps
    }

    cfg = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cfg.read(ini_path)

    sect = "confidence"
    tfs = _as_list(cfg.get(sect, "tfs", fallback=dflt["tfs"]))
    wts = [float(x) for x in _as_list(cfg.get(sect, "tf_weights", fallback=dflt["tf_weights"]))] or [0.25,0.35,0.40]
    if len(wts) != len(tfs):
        wts = [1.0/len(tfs)] * len(tfs)

    return ConfCfg(
        tfs=tfs,
        tf_weights=wts,
        lookback_days=int(cfg.get(sect, "lookback_days", fallback=str(dflt["lookback_days"]))),
        states=int(cfg.get(sect, "states", fallback=str(dflt["states"]))),
        barrier_atr_k=float(cfg.get(sect, "barrier_atr_k", fallback=str(dflt["barrier_atr_k"]))),
        horizon_bars=_parse_horizon(cfg.get(sect, "horizon_bars", fallback=dflt["horizon_bars"])),
        laplace_alpha=float(cfg.get(sect, "laplace_alpha", fallback=str(dflt["laplace_alpha"]))),
        oi_z_hi=float(cfg.get(sect, "oi_z_hi", fallback=str(dflt["oi_z_hi"]))),
        vp_poc_atr_thr=float(cfg.get(sect, "vp_poc_atr_thr", fallback=str(dflt["vp_poc_atr_thr"]))),
        squeeze_bw_pct=float(cfg.get(sect, "squeeze_bw_pct", fallback=str(dflt["squeeze_bw_pct"]))),
        bb_score_hi=float(cfg.get(sect, "bb_score_hi", fallback=str(dflt["bb_score_hi"]))),
        lambda_poc=float(cfg.get(sect, "lambda_poc", fallback=str(dflt["lambda_poc"]))),
        lambda_crowd=float(cfg.get(sect, "lambda_crowd", fallback=str(dflt["lambda_crowd"]))),
        lambda_squeeze=float(cfg.get(sect, "lambda_squeeze", fallback=str(dflt["lambda_squeeze"]))),
        run_id=cfg.get(sect, "run_id", fallback=dflt["run_id"]),
        source=cfg.get(sect, "source", fallback=dflt["source"]),
        vol_delta_lookback=int(cfg.get(sect, "vol_delta_lookback", fallback=str(dflt["vol_delta_lookback"]))),
        atr_pct_lookback=int(cfg.get(sect, "atr_pct_lookback", fallback=str(dflt["atr_pct_lookback"]))),
        spread_cap_bps=float(cfg.get(sect, "spread_cap_bps", fallback=str(dflt["spread_cap_bps"]))),
    )

# -------------------- DB helpers --------------------

def _exec_values(sql: str, rows: List[tuple]):
    if not rows: return 0
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
        return len(rows)

def _last_metric(symbol: str, kind: str, tf: str, metric: str) -> Optional[float]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT val FROM indicators.values
             WHERE symbol=%s AND market_type=%s AND interval=%s AND metric=%s
             ORDER BY ts DESC LIMIT 1
        """, (symbol, kind, tf, metric))
        row = cur.fetchone()
    return float(row[0]) if row else None

def _load_5m(symbol: str, kind: str, lookback_days: int) -> pd.DataFrame:
    table = "market.futures_candles" if kind == "futures" else "market.spot_candles"
    cutoff = datetime.now(TZ) - timedelta(days=lookback_days)
    cols = "ts, open, high, low, close, volume"
    if kind == "futures":
        cols += ", COALESCE(oi, NULL) AS oi"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT {cols}
              FROM {table}
             WHERE symbol=%s AND interval='5m' AND ts >= %s
             ORDER BY ts ASC
        """, (symbol, cutoff))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    cols_list = ["ts","open","high","low","close","volume"] + (["oi"] if kind=="futures" else [])
    df = pd.DataFrame(rows, columns=cols_list)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    for c in ["open","high","low","close","volume"] + (["oi"] if "oi" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "oi" in df.columns:
        df["oi"] = df["oi"].astype(float)
    return df.dropna(subset=["open","high","low","close","volume"])

def _resample(df5: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df5.empty: return df5
    if tf == "5m": return df5.copy()
    rule = TF_TO_OFFSET.get(tf)
    if not rule: return pd.DataFrame()
    out = pd.DataFrame({
        "open":   df5["open"]  .resample(rule, label="right", closed="right").first(),
        "high":   df5["high"]  .resample(rule, label="right", closed="right").max(),
        "low":    df5["low"]   .resample(rule, label="right", closed="right").min(),
        "close":  df5["close"] .resample(rule, label="right", closed="right").last(),
        "volume": df5["volume"].resample(rule, label="right", closed="right").sum(),
    })
    if "oi" in df5.columns:
        out["oi"] = df5["oi"].resample(rule, label="right", closed="right").last()
    return out.dropna(how="any")

# -------------------- Technical bits we reuse --------------------

def _ema(s: pd.Series, n:int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()

def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    a = (h - l).values
    b = (h - prev_c).abs().values
    d = (l - prev_c).abs().values
    tr = np.maximum.reduce([a, b, d])
    return pd.Series(tr, index=h.index, dtype="float64")

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n:int=14) -> pd.Series:
    tr = _true_range(h, l, c)
    return tr.ewm(alpha=1.0/n, adjust=False).mean().astype("float64")

def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n:int=14):
    up = h.diff(); dn = -l.diff()
    plus_dm  = ((up > dn) & (up > 0.0)) * up
    minus_dm = ((dn > up) & (dn > 0.0)) * dn
    atr_s = _true_range(h, l, c).ewm(alpha=1.0/n, adjust=False).mean()
    plus_di  = 100.0 * (plus_dm.ewm(alpha=1.0/n, adjust=False).mean()  / atr_s.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0/n, adjust=False).mean() / atr_s.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0
    adx_v = dx.ewm(alpha=1.0/n, adjust=False).mean()
    return adx_v.astype("float64"), plus_di.astype("float64"), minus_di.astype("float64")

def _bb_width_pct(close: pd.Series, n:int=20, k:float=2.0) -> pd.Series:
    ma = close.rolling(n, min_periods=max(5, n//2)).mean()
    sd = close.rolling(n, min_periods=max(5, n//2)).std(ddof=1)
    upper = ma + k*sd; lower = ma - k*sd
    width = (upper - lower) / (ma.replace(0, np.nan).abs())
    return 100.0 * width

def _obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0.0))
    return (sign * volume).cumsum()

# -------------------- Optional microstructure (spread) --------------------

def _latest_spread_bps(symbol: str, price: float, kind: str, cap_bps: float) -> float:
    """
    Try a few tables to get latest bid/ask; if nothing exists, return 0.
    Assumes table has columns (symbol, ts, bid, ask). Safe no-op if missing.
    """
    if price <= 0:
        return 0.0
    tables = ["market.l1_quotes", "market.futures_quotes", "market.spot_quotes"]
    for tbl in tables:
        try:
            with get_db_connection() as conn, conn.cursor() as cur:
                cur.execute(f"""
                    SELECT bid, ask
                      FROM {tbl}
                     WHERE symbol=%s
                     ORDER BY ts DESC
                     LIMIT 1
                """, (symbol,))
                row = cur.fetchone()
            if not row: 
                continue
            bid, ask = float(row[0] or 0.0), float(row[1] or 0.0)
            if bid > 0 and ask > 0 and ask >= bid:
                bps = ((ask - bid) / price) * 1e4  # basis points
                return float(min(cap_bps, max(0.0, bps)))
        except Exception:
            # table might not exist; ignore
            continue
    return 0.0

# -------------------- Artifacts loading (optional) --------------------

def _load_hmm_gmm(tf: str, kind: str) -> Optional[dict]:
    path = pathlib.Path(MODELS_DIR) / f"hmm_{kind}_{tf}.npz"
    if not path.exists(): return None
    d = np.load(path, allow_pickle=False)
    return {"pi": d["pi"], "mu": d["mu"], "Sigma": d["Sigma"]}

def _load_nb_table(tf: str, kind: str) -> Optional[dict]:
    path = pathlib.Path(MODELS_DIR) / f"nb_{kind}_{tf}.json"
    if not path.exists(): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_calib(tf: Optional[str], kind: str) -> Optional[dict]:
    name = f"calib_{kind}_{tf}.json" if tf else f"calib_{kind}_mtf.json"
    path = pathlib.Path(MODELS_DIR) / name
    if not path.exists(): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------- Feature extraction (per TF) --------------------

@dataclass
class Features:
    x: Dict[str, float]        # continuous features for gating
    sig: Dict[str, int]        # binary/ternary predicates for NB
    diag: Dict[str, float]     # extra diagnostics (entropy inputs etc.)

def _zscore(arr: pd.Series, win:int=20) -> pd.Series:
    m = arr.rolling(win, min_periods=max(10, win//2)).mean()
    s = arr.rolling(win, min_periods=max(10, win//2)).std(ddof=1).replace(0,np.nan)
    return (arr - m) / s

def _percentile_rank(x: float, arr: pd.Series) -> float:
    a = arr.dropna().values
    if a.size == 0: return float("nan")
    return float(100.0 * (np.sum(a <= x) / a.size))

def _features_for_tf(dftf: pd.DataFrame, symbol: str, kind: str, tf: str, cfg: ConfCfg) -> Features:
    d = dftf.copy()
    close, high, low, vol = d["close"], d["high"], d["low"], d["volume"]
    atr = _atr(high, low, close, 14)
    adx_v, plus_di, minus_di = _adx(high, low, close, 14)
    r1 = close.pct_change(1)
    r3 = close.pct_change(3)
    vol_std = r1.rolling(20, min_periods=10).std(ddof=1)
    bb_bw = _bb_width_pct(close, 20, 2.0)
    # squeeze flag: latest BW below pct threshold of its 120-bar history
    bw_hist = bb_bw.tail(120)
    squeeze = int(bb_bw.iloc[-1] <= np.nanpercentile(bw_hist.dropna().values, cfg.squeeze_bw_pct)) if len(bw_hist.dropna()) > 10 else 0

    # OI z + OBVdelta
    oi_z = 0.0
    if "oi" in d.columns:
        doi = d["oi"].diff()
        oi_z = float(_zscore(doi, win=20).iloc[-1] if len(doi.dropna()) else 0.0)
    obv = _obv_series(close, vol)
    obv_d = float(obv.diff().iloc[-1] if len(obv) > 1 else 0.0)

    # --- NEW: volume-delta anomaly (signed flow) ---
    vol_delta = (close - d["open"]) * vol  # signed “who won the bar”
    vdz = _zscore(vol_delta, win=max(20, cfg.vol_delta_lookback))
    vol_delta_z = float(vdz.iloc[-1] if len(vdz.dropna()) else 0.0)

    # --- NEW: ATR percentile (volatility regime) ---
    atr_win = max(60, cfg.atr_pct_lookback)
    atr_pct = _percentile_rank(float(atr.iloc[-1] if len(atr) else np.nan),
                               atr.tail(atr_win))

    # VP/BB from DB (latest)
    vp_poc = _last_metric(symbol, kind, tf, "VP.POC")
    vp_val = _last_metric(symbol, kind, tf, "VP.VAL")
    vp_vah = _last_metric(symbol, kind, tf, "VP.VAH")
    bb_score = _last_metric(symbol, kind, tf, "BB.score")
    last_close = float(close.iloc[-1])
    last_atr   = float(atr.iloc[-1] or 1.0)

    vp_dist_atr = abs(last_close - vp_poc) / max(last_atr, 1e-9) if vp_poc is not None else np.nan
    above_vah = int(vp_vah is not None and last_close > vp_vah)
    below_val = int(vp_val is not None and last_close < vp_val)
    in_value  = int(vp_val is not None and vp_vah is not None and (vp_val <= last_close <= vp_vah))
    bb_hi     = int(bb_score is not None and bb_score >= cfg.bb_score_hi)

    # EMA stack + MACD
    ema5, ema20, ema50 = _ema(close,5), _ema(close,20), _ema(close,50)
    ema_stack_up = int(ema5.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1])
    macd_line = _ema(close,12) - _ema(close,26)
    macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_sig
    macd_rise = int(macd_hist.diff().iloc[-1] > 0)

    # RSI zones (momentum)
    delta = close.diff()
    gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    rs = gain.ewm(alpha=1/14, adjust=False).mean() / (loss.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan))
    rsi = 100 - 100/(1+rs)
    rsi60 = int(rsi.iloc[-1] >= 60); rsi40 = int(rsi.iloc[-1] <= 40)

    # OI build-up predicates (fut only)
    oi_bu = int((oi_z if not math.isnan(oi_z) else 0.0) > 0.5)
    price_up = int(r1.iloc[-1] > 0)
    oi_build_bull = int(oi_bu and price_up)
    oi_build_bear = int(oi_bu and (not price_up))

    # --- NEW: microstructure (bid/ask spread in bps, capped) ---
    spread_bps = _latest_spread_bps(symbol, last_close, kind, cfg.spread_cap_bps)

    x = {
        "r1": float(r1.iloc[-1]),
        "r3": float(r3.iloc[-1] if not np.isnan(r3.iloc[-1]) else 0.0),
        "vol": float(vol_std.iloc[-1] if not np.isnan(vol_std.iloc[-1]) else 0.0),
        "atr": last_atr,
        "adx": float(adx_v.iloc[-1] if not np.isnan(adx_v.iloc[-1]) else 0.0),
        "vp_dist_atr": float(vp_dist_atr) if not np.isnan(vp_dist_atr) else np.nan,
        "oi_z": float(oi_z if not np.isnan(oi_z) else 0.0),
        "bb_bw": float(bb_bw.iloc[-1] if not np.isnan(bb_bw.iloc[-1]) else 0.0),
        "obv_d": float(obv_d),
        # new continuous features
        "vol_delta_z": float(vol_delta_z),
        "atr_pct": float(atr_pct if not np.isnan(atr_pct) else 50.0),
        "spread_bps": float(spread_bps),
    }
    sig = {
        "RSI60": rsi60, "RSI40": rsi40,
        "EMA_STACK_UP": ema_stack_up,
        "MACD_RISE": macd_rise,
        "ADX20": int(x["adx"] >= 20.0),
        "SQUEEZE": squeeze,
        "VP_ABOVE_VAH": above_vah,
        "VP_BELOW_VAL": below_val,
        "VP_IN_VALUE": in_value,
        "BB_SCORE_HI": bb_hi,
        "OI_BUILD_BULL": oi_build_bull if kind=="futures" else 0,
        "OI_BUILD_BEAR": oi_build_bear if kind=="futures" else 0,
        "ROC_POS": int(r1.iloc[-1] > 0),
        # optional signal hooks you might add to the NB table later
        "VOL_DELTA_STRONG": int(x["vol_delta_z"] >= 1.0),
        "ATR_HIGH_REGIME": int(x["atr_pct"] >= 70.0),
        "SPREAD_TIGHT": int(x["spread_bps"] <= 10.0),  # <= 10 bps → liquid
    }
    diag = {
        "last_close": last_close,
        "vp_poc": vp_poc if vp_poc is not None else float("nan"),
        "atr_pct": x["atr_pct"],
        "vol_delta_z": x["vol_delta_z"],
        "spread_bps": x["spread_bps"],
    }
    return Features(x=x, sig=sig, diag=diag)

# -------------------- HMM / GMM gate --------------------

def _mvnorm_logpdf(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    d = x.shape[0]
    try:
        L = np.linalg.cholesky(Sigma + 1e-9*np.eye(d))
    except np.linalg.LinAlgError:
        return -1e9
    alpha = np.linalg.solve(L, x - mu)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5*(np.dot(alpha, alpha) + logdet + d*np.log(2*np.pi))

def _gate_posteriors(feat: Features, tf: str, kind: str, cfg: ConfCfg) -> Tuple[np.ndarray, float]:
    """
    Returns (p_state[S], entropy). If artifacts missing, uses ADX/vol heuristic.
    """
    art = _load_hmm_gmm(tf, kind)
    # select continuous vector (extend with new features)
    keys = ["r1","vol","atr","adx","vp_dist_atr","oi_z","bb_bw","obv_d","vol_delta_z","atr_pct","spread_bps"]
    xvec = np.array([feat.x.get(k, 0.0) if not np.isnan(feat.x.get(k, np.nan)) else 0.0 for k in keys], dtype=float)

    if art is not None:
        pi = np.asarray(art["pi"], dtype=float)  # (S,)
        mu = np.asarray(art["mu"], dtype=float)  # (S,d)
        Si = np.asarray(art["Sigma"], dtype=float)  # (S,d,d)
        S = pi.shape[0]
        logps = np.array([math.log(max(pi[s],1e-9)) + _mvnorm_logpdf(xvec, mu[s], Si[s]) for s in range(S)])
        logps -= logps.max()
        p = np.exp(logps); p /= p.sum() if p.sum()>0 else 1.0
    else:
        # Heuristic gate: Trend vs Range vs Shock (S=3)
        adx = feat.x["adx"]; vol = max(feat.x["vol"], 1e-6); r1 = abs(feat.x["r1"])
        p_trend = 1.0 / (1.0 + math.exp(-(adx - 20.0)/5.0))
        p_shock = max(0.0, min(1.0, (r1 / (2.0*vol)) - 0.0))
        p_range = max(0.0, 1.0 - (p_trend + p_shock))
        ssum = max(1e-9, p_trend + p_range + p_shock)
        p = np.array([p_trend/ssum, p_range/ssum, p_shock/ssum], dtype=float)
    ent = -float(np.sum([pi*np.log(pi+1e-12) for pi in p]))
    return p, ent

# -------------------- Naive Bayes ensemble --------------------

_DEFAULT_LR = {
    "trend": {
        "RSI60":1.25, "EMA_STACK_UP":1.35, "MACD_RISE":1.20, "ADX20":1.30, "SQUEEZE":1.00,
        "VP_ABOVE_VAH":1.10, "VP_BELOW_VAL":0.80, "VP_IN_VALUE":0.90, "BB_SCORE_HI":1.20,
        "OI_BUILD_BULL":1.25, "OI_BUILD_BEAR":0.75, "ROC_POS":1.10, "RSI40":0.80,
        "VOL_DELTA_STRONG":1.20, "ATR_HIGH_REGIME":1.10, "SPREAD_TIGHT":1.10
    },
    "range": {
        "RSI60":0.95, "EMA_STACK_UP":0.90, "MACD_RISE":1.00, "ADX20":0.80, "SQUEEZE":1.20,
        "VP_ABOVE_VAH":0.90, "VP_BELOW_VAL":0.90, "VP_IN_VALUE":1.15, "BB_SCORE_HI":1.05,
        "OI_BUILD_BULL":1.00, "OI_BUILD_BEAR":1.00, "ROC_POS":1.00, "RSI40":1.05,
        "VOL_DELTA_STRONG":1.00, "ATR_HIGH_REGIME":0.95, "SPREAD_TIGHT":1.05
    },
    "shock": {
        "RSI60":1.05, "EMA_STACK_UP":1.00, "MACD_RISE":1.05, "ADX20":1.10, "SQUEEZE":0.90,
        "VP_ABOVE_VAH":1.00, "VP_BELOW_VAL":1.00, "VP_IN_VALUE":0.90, "BB_SCORE_HI":1.10,
        "OI_BUILD_BULL":1.10, "OI_BUILD_BEAR":0.90, "ROC_POS":1.05, "RSI40":1.00,
        "VOL_DELTA_STRONG":1.20, "ATR_HIGH_REGIME":1.10, "SPREAD_TIGHT":1.05
    }
}
_DEFAULT_PRIOR = {"trend":0.5, "range":0.4, "shock":0.1}

def _nb_logit(sig: Dict[str,int], regime: str, nb_table: Optional[dict], alpha: float) -> float:
    if nb_table is None:
        prior = max(1e-6, _DEFAULT_PRIOR.get(regime, 0.33))
        logit = math.log(prior/(1-prior))
        for k,v in sig.items():
            lr = _DEFAULT_LR.get(regime, {}).get(k, 1.0)
            if v: logit += math.log(max(lr, 1e-6))
        return logit

    priors = nb_table.get("priors", {})
    prior = float(priors.get(regime, 0.5))
    logit = math.log(max(prior,1e-6)/max(1-prior,1e-6))
    like = nb_table.get("likelihoods", {}).get(regime, {})
    for k,v in sig.items():
        if not v: continue
        cell = like.get(k)
        if cell is None:
            continue
        succ = float(cell.get("succ", 0.0)) + alpha
        fail = float(cell.get("fail", 0.0)) + alpha
        denom = max(1e-6, fail)
        lr = succ/denom
        logit += math.log(max(lr, 1e-6))
    return logit

def _platt_calibrate(prob: float, calib: Optional[dict]) -> float:
    if not calib: return prob
    A = float(calib.get("A", 0.0)); B = float(calib.get("B", 0.0))
    x = math.log(max(prob,1e-9)/max(1-prob,1e-9))
    z = 1.0/(1.0 + math.exp(A*x + B))
    return float(min(1.0, max(0.0, z)))

# -------------------- Per-TF probability --------------------

def _per_tf_probability(feat: Features, tf: str, kind: str, cfg: ConfCfg) -> Tuple[float, Dict[str,float], float, np.ndarray]:
    """
    Returns p_TF, p_by_regime (dict), entropy, p_state (np.ndarray)
    """
    p_state, entropy = _gate_posteriors(feat, tf, kind, cfg)
    regimes = ["trend","range","shock"][:len(p_state)]
    nb = _load_nb_table(tf, kind)
    alpha = cfg.laplace_alpha

    p_tf_list = []
    p_by_regime = {}
    for i, r in enumerate(regimes):
        logit_r = _nb_logit(feat.sig, r, nb, alpha)
        p_r = 1.0/(1.0 + math.exp(-logit_r))
        p_by_regime[r] = p_r
        p_tf_list.append(p_state[i] * p_r)
    p_tf = float(sum(p_tf_list))
    return p_tf, p_by_regime, float(entropy), p_state

# -------------------- Dynamic TF weights by regime --------------------

def _dynamic_tf_weights(base: List[float], regime: str, tfs: List[str]) -> List[float]:
    """
    Reweight base weights given the dominant regime:
      - trend  → tilt to higher TFs (e.g., 125m)
      - shock  → tilt to lower TFs (e.g., 25m)
      - range  → keep as-is (light normalization)
    """
    w = np.array(base, dtype=float)
    if len(w) != len(tfs):  # safety
        w = np.ones(len(tfs), dtype=float) / max(1, len(tfs))

    if regime == "trend":
        # multiply weights by increasing factor with TF index
        scale = np.linspace(0.9, 1.2, len(tfs))
        w = w * scale
    elif regime == "shock":
        # multiply weights by decreasing factor with TF index
        scale = np.linspace(1.2, 0.8, len(tfs))
        w = w * scale
    else:
        # range → gentle flatten
        w = 0.8*w + 0.2*(np.ones_like(w)/len(w))

    w = np.clip(w, 1e-6, None)
    w = w / w.sum()
    return w.tolist()

# -------------------- Blend TFs, penalties/bonuses --------------------

def _blend_probabilities(p_tfs: Dict[str,float], ent_tfs: Dict[str,float],
                         cfg: ConfCfg, dynamic_w: Optional[List[float]]=None) -> Tuple[float,float,Dict[str,float]]:
    # entropy-adjusted weights
    eps = 1e-9
    # base weights (possibly dynamic regime-tilted) mapped by tf
    base_map = {tf: (dynamic_w[i] if dynamic_w is not None else cfg.tf_weights[i])
                for i, tf in enumerate(cfg.tfs) if tf in p_tfs}

    adj = {}
    for tf, p in p_tfs.items():
        base = base_map.get(tf, 1.0/len(p_tfs))
        H = ent_tfs.get(tf, 0.0)
        conf = 1.0 - (H / max(eps, math.log(3)))  # normalize by log(3) regimes
        adj[tf] = base * max(0.0, conf)

    wsum = sum(adj.values()) or 1.0
    logits = {}
    for tf,p in p_tfs.items():
        p = min(1-1e-6, max(1e-6, p))
        logits[tf] = math.log(p/(1-p))
    z = sum((adj[tf]/wsum) * logits[tf] for tf in p_tfs.keys())
    p_blend = 1.0/(1.0 + math.exp(-z))
    return float(p_blend), float(z), adj

def _apply_penalties(z_logit: float, feat_by_tf: Dict[str,Features], cfg: ConfCfg) -> float:
    z = z_logit
    # POC magnet penalty (if any TF very close to POC)
    for tf, F in feat_by_tf.items():
        vp_d = F.x.get("vp_dist_atr", np.nan)
        if not np.isnan(vp_d) and vp_d < cfg.vp_poc_atr_thr:
            z -= cfg.lambda_poc
            break
    # Crowding penalty
    for tf, F in feat_by_tf.items():
        if F.x.get("oi_z", 0.0) > cfg.oi_z_hi and (F.sig.get("VP_IN_VALUE",0)==0):
            z -= cfg.lambda_crowd
            break
    # Squeeze bonus: any TF in squeeze with ADX>20 and MACD rising
    for tf, F in feat_by_tf.items():
        if F.sig.get("SQUEEZE",0)==1 and F.sig.get("ADX20",0)==1 and F.sig.get("MACD_RISE",0)==1:
            z += cfg.lambda_squeeze
            break
    return z

# -------------------- Write metrics --------------------

def _write_metrics(symbol: str, kind: str, tf_stats: Dict[str, dict],
                   p_mtf: float, z_mtf: float, cfg: ConfCfg,
                   dyn_weights: Optional[Dict[str,float]]=None, dom_regime: Optional[str]=None):
    rows: List[tuple] = []
    nowts = datetime.now(TZ)

    for tf, st in tf_stats.items():
        ts = st["ts"]
        p_state = st["p_state"]
        entropy = st["entropy"]
        p_tf    = st["p_tf"]
        logit_tf= st["logit_tf"]
        argmax  = int(np.argmax(p_state))
        rows += [
            (symbol, kind, tf, ts, f"HMM.state.{tf}", float(argmax), json.dumps({}), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"HMM.p_state.{tf}.0", float(p_state[0]), json.dumps({}), cfg.run_id, cfg.source),
        ]
        if len(p_state)>1:
            rows.append((symbol, kind, tf, ts, f"HMM.p_state.{tf}.1", float(p_state[1]), json.dumps({}), cfg.run_id, cfg.source))
        if len(p_state)>2:
            rows.append((symbol, kind, tf, ts, f"HMM.p_state.{tf}.2", float(p_state[2]), json.dumps({}), cfg.run_id, cfg.source))
        rows.append((symbol, kind, tf, ts, f"HMM.entropy.{tf}", float(entropy), json.dumps({}), cfg.run_id, cfg.source))
        rows.append((symbol, kind, tf, ts, f"ENS.logit.{tf}", float(logit_tf), json.dumps({}), cfg.run_id, cfg.source))
        rows.append((symbol, kind, tf, ts, f"CONF.prob.{tf}", float(p_tf), json.dumps({}), cfg.run_id, cfg.source))
        rows.append((symbol, kind, tf, ts, f"CONF.score.{tf}", float(round(100.0*p_tf,2)), json.dumps({}), cfg.run_id, cfg.source))
        # record dynamic weight actually used
        if dyn_weights and tf in dyn_weights:
            rows.append((symbol, kind, tf, ts, f"CONF.wt.{tf}", float(dyn_weights[tf]), json.dumps({}), cfg.run_id, cfg.source))

    ts_latest = max([st["ts"] for st in tf_stats.values()]) if tf_stats else nowts
    ctx = {"dom_regime": dom_regime} if dom_regime is not None else {}
    rows += [
        (symbol, kind, "MTF", ts_latest, "CONF.logit.mtf", float(z_mtf), json.dumps(ctx), cfg.run_id, cfg.source),
        (symbol, kind, "MTF", ts_latest, "CONF.prob.mtf",  float(p_mtf), json.dumps(ctx), cfg.run_id, cfg.source),
        (symbol, kind, "MTF", ts_latest, "CONF.score.mtf", float(round(100.0*p_mtf,2)), json.dumps(ctx), cfg.run_id, cfg.source),
    ]

    sql = """
        INSERT INTO indicators.values
            (symbol, market_type, interval, ts, metric, val, context, run_id, source)
        VALUES %s
        ON CONFLICT (symbol, market_type, interval, ts, metric) DO UPDATE
           SET val=EXCLUDED.val, context=EXCLUDED.context, run_id=EXCLUDED.run_id, source=EXCLUDED.source
    """
    _exec_values(sql, rows)

# -------------------- Public driver --------------------

def process_symbol(symbol: str, *, kind: str, cfg: Optional[ConfCfg] = None) -> int:
    cfg = cfg or load_cfg()
    df5 = _load_5m(symbol, kind, lookback_days=cfg.lookback_days)
    if df5.empty:
        print(f"⚠️ {kind}:{symbol} no 5m data")
        return 0

    tf_stats: Dict[str, dict] = {}
    p_tfs: Dict[str,float] = {}
    ent_tfs: Dict[str,float] = {}
    feats: Dict[str,Features] = {}
    p_states: Dict[str,np.ndarray] = {}

    for tf in cfg.tfs:
        if tf not in TF_TO_OFFSET: continue
        dftf = _resample(df5, tf)
        if dftf.empty or len(dftf) < 30:
            continue

        F = _features_for_tf(dftf, symbol, kind, tf, cfg)
        feats[tf] = F
        p_tf, p_by_regime, entropy, p_state = _per_tf_probability(F, tf, kind, cfg)
        p_tfs[tf] = p_tf
        ent_tfs[tf] = entropy
        p_states[tf] = p_state
        logit_tf = math.log(max(p_tf,1e-6)/max(1-p_tf,1e-6))
        tf_stats[tf] = {
            "ts": dftf.index[-1].to_pydatetime().replace(tzinfo=TZ),
            "p_state": p_state,
            "entropy": entropy,
            "p_tf": p_tf,
            "logit_tf": logit_tf
        }

    if not tf_stats:
        print(f"ℹ️ {kind}:{symbol} confidence: no TF frames")
        return 0

    # Determine dominant regime from most confident TF (lowest entropy)
    tf_best = min(ent_tfs.keys(), key=lambda t: ent_tfs[t])
    dom_reg_idx = int(np.argmax(p_states[tf_best]))
    regimes = ["trend","range","shock"]
    dom_regime = regimes[dom_reg_idx] if dom_reg_idx < len(regimes) else "range"

    # Dynamic regime-tilted weights
    dyn_w_list = _dynamic_tf_weights(cfg.tf_weights, dom_regime, cfg.tfs)
    dyn_w_map: Dict[str,float] = {tf: dyn_w_list[i] for i,tf in enumerate(cfg.tfs) if tf in p_tfs}

    # blend
    p_blend, z_blend, used_w = _blend_probabilities(p_tfs, ent_tfs, cfg, dynamic_w=dyn_w_list)
    z_blend = _apply_penalties(z_blend, feats, cfg)
    p_blend = 1.0/(1.0 + math.exp(-z_blend))

    # calibration (optional)
    calib = _load_calib(None, kind)
    p_blend = _platt_calibrate(p_blend, calib)

    _write_metrics(symbol, kind, tf_stats, p_blend, z_blend, cfg, dyn_weights=used_w, dom_regime=dom_regime)
    print(f"✅ {kind}:{symbol} CONF → prob.mtf={p_blend:.3f} score={round(100*p_blend):d} (regime={dom_regime})")
    return 1

def run(symbols: Optional[List[str]] = None, kinds: Iterable[str] = ("spot","futures")) -> int:
    if symbols is None:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT symbol
                  FROM webhooks.webhook_alerts
                 WHERE status='INDICATOR_PROCESS'
            """)
            rows = cur.fetchall()
        symbols = [r[0] for r in rows or []]

    cfg = load_cfg()
    total = 0
    for s in symbols:
        for k in kinds:
            try:
                total += process_symbol(s, kind=k, cfg=cfg)
            except Exception as e:
                print(f"❌ {k}:{s} CONF error → {e}")
    print(f"🎯 CONF wrote metrics for {total} run(s) across {len(symbols)} symbol(s)")
    return total

if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    syms = [a for a in args if not a.startswith("--")]
    ks = [a.split("=",1)[1] for a in args if a.startswith("--kinds=")]
    kinds = tuple(ks[0].split(",")) if ks else ("spot","futures")
    run(syms or None, kinds=kinds)
