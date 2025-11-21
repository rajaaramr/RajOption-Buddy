from __future__ import annotations
import os, json, math, configparser
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Iterable
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg2.extras as pgx

from utils.db import get_db_connection

TZ = timezone.utc
# NEW: separate config file/env
DEFAULT_INI = os.getenv("OI_CONF_INI", "configs/oi_confidence.ini")

TF_TO_OFFSET = {
    "5m": "5min", "15m": "15min", "25m": "25min", "30m": "30min",
    "60m": "60min", "65m": "65min", "120m": "120min", "125m": "125min",
    "250m": "250min"
}

# ========= Config =========

@dataclass
class ConfCfg:
    # confidence (blend) settings
    tfs: List[str]
    tf_weights: List[float]
    lookback_days: int
    integration: str
    bayes_k: float
    bayes_m: float
    mtf_oi_weights: Dict[str, float]
    mtf_align_gate: float
    mtf_conflict_cap: float
    use_baseline_conf: bool  # NEW
    run_id: str
    source: str
    # OI scoring knobs
    oi_window: int
    persistence_min: float
    voi_eff_min: float
    turnover_low: float
    turnover_high: float
    roll_dampen: float
    basis_confirm: int
    squeeze_bw_pct: float
    # roll detection
    roll_pdrop: float
    roll_z: float
    # dynamic weighting
    dyn_w_enable: bool
    dyn_w_fast_bias: float
    dyn_w_slow_bias: float
    dyn_ref_tf: str
    # ini path
    ini_path: str = DEFAULT_INI

def _as_list_csv(x: str) -> List[str]:
    return [s.strip() for s in x.split(",") if s.strip()]

def _as_float_list(x: str) -> List[float]:
    return [float(s.strip()) for s in x.split(",") if s.strip()]

def _parse_weights_map(x: str) -> Dict[str, float]:
    out = {}
    for p in _as_list_csv(x):
        if ":" in p:
            k, v = p.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except:
                pass
    return out

def _normalize_map(m: Dict[str, float]) -> Dict[str, float]:
    s = sum(m.values()) or 1.0
    return {k: (v/s) for k, v in m.items()}

def load_cfg(ini_path: str = DEFAULT_INI) -> ConfCfg:
    dflt = {
        "tfs": "15m,30m,60m,120m",
        "tf_weights": "0.20,0.30,0.25,0.25",
        "lookback_days": 150,
        "integration": "bayes",
        "bayes_k": 0.50,
        "bayes_m": 0.75,
        "mtf_oi_weights": "15m:0.20,30m:0.30,60m:0.25,120m:0.25",
        "mtf_align_gate": 0.60,
        "mtf_conflict_cap": 0.60,
        "use_baseline_conf": True,
        "run_id": os.getenv("RUN_ID", "oi_conf_run"),
        "source": os.getenv("SRC", "oi_confidence_v2"),
        # OI knobs
        "oi_window": 20,
        "persistence_min": 0.60,
        "voi_eff_min": 0.05,
        "turnover_low": 0.10,
        "turnover_high": 1.20,
        "roll_dampen": 0.50,
        "basis_confirm": 1,
        "squeeze_bw_pct": 20.0,
        # roll
        "roll_pdrop": 0.30,
        "roll_z": 2.0,
        # dynamic weighting
        "dyn_w_enable": True,
        "dyn_w_fast_bias": 0.12,
        "dyn_w_slow_bias": 0.10,
        "dyn_ref_tf": "60m",
    }

    cfg = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cfg.read(ini_path)

    sectC = "confidence"
    sectO = "oi"

    tfs = _as_list_csv(cfg.get(sectC, "tfs", fallback=dflt["tfs"]))
    wts = _as_float_list(cfg.get(sectC, "tf_weights", fallback=dflt["tf_weights"]))
    if len(wts) != len(tfs):
        wts = [1.0/len(tfs)]*len(tfs)

    mtf_map = _parse_weights_map(cfg.get(sectC, "mtf_oi_weights", fallback=dflt["mtf_oi_weights"]))

    return ConfCfg(
        tfs=tfs,
        tf_weights=wts,
        lookback_days=int(cfg.get(sectC, "lookback_days", fallback=str(dflt["lookback_days"]))),
        integration=cfg.get(sectC, "oi_integration", fallback=dflt["integration"]).strip().lower(),
        bayes_k=float(cfg.get(sectC, "bayes_k", fallback=str(dflt["bayes_k"]))),
        bayes_m=float(cfg.get(sectC, "bayes_m", fallback=str(dflt["bayes_m"]))),
        mtf_oi_weights=mtf_map or _parse_weights_map(dflt["mtf_oi_weights"]),
        mtf_align_gate=float(cfg.get(sectC, "mtf_align_gate", fallback=str(dflt["mtf_align_gate"]))),
        mtf_conflict_cap=float(cfg.get(sectC, "mtf_conflict_cap", fallback=str(dflt["mtf_conflict_cap"]))),
        use_baseline_conf=str(cfg.get(sectC, "use_baseline_conf", fallback=str(dflt["use_baseline_conf"]))).lower() in {"1","true","yes","on"},
        run_id=cfg.get(sectC, "run_id", fallback=dflt["run_id"]),
        source=cfg.get(sectC, "source", fallback=dflt["source"]),

        oi_window=int(cfg.get(sectO, "window", fallback=str(dflt["oi_window"]))),
        persistence_min=float(cfg.get(sectO, "persistence_min", fallback=str(dflt["persistence_min"]))),
        voi_eff_min=float(cfg.get(sectO, "voi_eff_min", fallback=str(dflt["voi_eff_min"]))),
        turnover_low=float(cfg.get(sectO, "oi_turnover_low", fallback=str(dflt["turnover_low"]))),
        turnover_high=float(cfg.get(sectO, "oi_turnover_high", fallback=str(dflt["turnover_high"]))),
        roll_dampen=float(cfg.get(sectO, "roll_dampen", fallback=str(dflt["roll_dampen"]))),
        basis_confirm=int(cfg.get(sectO, "basis_confirm", fallback=str(dflt["basis_confirm"]))),
        squeeze_bw_pct=float(cfg.get(sectO, "squeeze_bw_pct", fallback=str(dflt["squeeze_bw_pct"]))),
        roll_pdrop=float(cfg.get(sectO, "roll_pdrop", fallback=str(dflt["roll_pdrop"]))),
        roll_z=float(cfg.get(sectO, "roll_z", fallback=str(dflt["roll_z"]))),

        dyn_w_enable=str(cfg.get(sectC, "dyn_w_enable", fallback=str(dflt["dyn_w_enable"]))).lower() in {"1","true","yes","on"},
        dyn_w_fast_bias=float(cfg.get(sectC, "dyn_w_fast_bias", fallback=str(dflt["dyn_w_fast_bias"]))),
        dyn_w_slow_bias=float(cfg.get(sectC, "dyn_w_slow_bias", fallback=str(dflt["dyn_w_slow_bias"]))),
        dyn_ref_tf=cfg.get(sectC, "dyn_ref_tf", fallback=dflt["dyn_ref_tf"]),
        ini_path=ini_path
    )

# -------- per-symbol overrides (INI) --------
def _apply_symbol_overrides(cfg: ConfCfg, symbol: str) -> ConfCfg:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cp.read(cfg.ini_path)
    base = cfg

    osec = f"oi.symbol:{symbol}"
    if cp.has_section(osec):
        def g(k, fb=None): return cp.get(osec, k, fallback=fb)
        base = replace(
            base,
            turnover_low=float(g("oi_turnover_low",  base.turnover_low)),
            turnover_high=float(g("oi_turnover_high", base.turnover_high)),
            bayes_k=float(g("bayes_k", base.bayes_k)),
            bayes_m=float(g("bayes_m", base.bayes_m)),
            basis_confirm=int(g("basis_confirm", str(base.basis_confirm)))
        )

    csec = f"confidence.symbol:{symbol}"
    if cp.has_section(csec):
        def g2(k, fb=None): return cp.get(csec, k, fallback=fb)
        tfs = _as_list_csv(g2("tfs", ",".join(base.tfs)))
        tfw_s = g2("tf_weights", None)
        if tfw_s:
            tfw = _as_float_list(tfw_s)
            if len(tfw) != len(tfs): tfw = [1.0/len(tfs)]*len(tfs)
        else:
            tfw = base.tf_weights
        mtf_map = _parse_weights_map(g2("mtf_oi_weights", ",".join([f"{k}:{v}" for k,v in base.mtf_oi_weights.items()])))
        base = replace(base, tfs=tfs, tf_weights=tfw, mtf_oi_weights=mtf_map)
    return base

# ========= DB helpers =========

def _exec_values(sql: str, rows: List[tuple]) -> int:
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
    cols = ["ts","open","high","low","close","volume"]
    add_oi = (kind == "futures")
    if add_oi: cols.append("oi")
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT {", ".join(cols)}
              FROM {table}
             WHERE symbol=%s AND interval='5m' AND ts >= %s
             ORDER BY ts ASC
        """, (symbol, cutoff))
        rows = cur.fetchall()
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    for c in ["open","high","low","close","volume"] + (["oi"] if add_oi else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close","volume"])

def _resample(df5: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df5.empty: return df5
    if tf == "5m": return df5.copy()
    rule = TF_TO_OFFSET.get(tf)
    if not rule: return pd.DataFrame()
    out = pd.DataFrame({
        "open":   df5["open"].resample(rule, label="right", closed="right").first(),
        "high":   df5["high"].resample(rule, label="right", closed="right").max(),
        "low":    df5["low"].resample(rule, label="right", closed="right").min(),
        "close":  df5["close"].resample(rule, label="right", closed="right").last(),
        "volume": df5["volume"].resample(rule, label="right", closed="right").sum(),
    })
    if "oi" in df5.columns:
        out["oi"] = df5["oi"].resample(rule, label="right", closed="right").last()
    return out.dropna(how="any")

# ========= Utilities =========

def _zscore_series(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window, min_periods=max(5, window//2)).mean()
    s = x.rolling(window, min_periods=max(5, window//2)).std(ddof=1).replace(0,np.nan)
    return (x - m) / s

def _bb_width_pct(close: pd.Series, n:int=20, k:float=2.0) -> pd.Series:
    ma = close.rolling(n, min_periods=max(5, n//2)).mean()
    sd = close.rolling(n, min_periods=max(5, n//2)).std(ddof=1)
    upper = ma + k*sd; lower = ma - k*sd
    width = (upper - lower) / (ma.replace(0, np.nan).abs())
    return 100.0 * width

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n:int=14) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/n, adjust=False).mean()

# ========= OI feature engineering (per TF) =========

@dataclass
class OIFeatures:
    dOI: float
    dOI_pct: float
    OI_z: float
    VOI: float
    VOI_eff: float
    concord: float
    persistence: float
    turnover: float
    roll_flag: int
    basis_shift: float
    tod_adj: float
    squeeze_flag: int
    diag: Dict[str, float]

def _price_oi_concordance(price_ret: float, doi: float) -> float:
    if abs(price_ret) < 1e-9 and abs(doi) < 1e-9: return 0.0
    if price_ret > 0 and doi > 0: return +1.0
    if price_ret < 0 and doi > 0: return -1.0
    if price_ret > 0 and doi < 0: return +0.5
    if price_ret < 0 and doi < 0: return -0.5
    return 0.0

def _roll_window_mask(oi: pd.Series, doi: pd.Series, vol: pd.Series, pdrop: float, zthr: float, win:int=20) -> pd.Series:
    doi_pct = doi / oi.shift(1).replace(0,np.nan)
    doi_z = _zscore_series(doi, max(10, win))
    base = (doi_pct <= -abs(pdrop)) | (doi_z <= -abs(zthr))
    dil = base.rolling(156, min_periods=1).max()
    return (dil > 0).astype(int)

def _time_of_day_adj(ts: pd.Timestamp) -> float:
    hm = ts.hour*60 + ts.minute
    if 12*60 <= hm <= 13*60+45:
        return 0.85
    return 1.0

def _oi_features_for_tf(
    fut_tf: pd.DataFrame, spot_tf: Optional[pd.DataFrame], tf: str, window: int, squeeze_pct: float,
    roll_pdrop: float, roll_z: float
) -> Optional[OIFeatures]:
    if fut_tf.empty or "oi" not in fut_tf.columns: return None
    d = fut_tf.copy()
    close = d["close"]; vol = d["volume"]; oi = d["oi"].replace(0, np.nan)

    doi = oi.diff()
    doi_pct = (oi.diff() / oi.shift(1)).replace([np.inf,-np.inf], np.nan)
    OI_z = _zscore_series(oi, window)
    VOI = (doi / vol.replace(0,np.nan)).replace([np.inf,-np.inf], np.nan)
    VOI_eff = (doi.abs() / vol.replace(0,np.nan)).replace([np.inf,-np.inf], np.nan)

    price_ret = close.pct_change(1)
    concord = price_ret.combine(doi, lambda pr, d: _price_oi_concordance(pr if pd.notna(pr) else 0.0,
                                                                         d if pd.notna(d) else 0.0))

    # persistence
    sign_series = concord.map(lambda x: 1 if x>0 else (-1 if x<0 else 0))
    windowN = max(5, window)
    latest_sign = int(sign_series.replace(0, np.nan).ffill().iloc[-1] or 0)
    if latest_sign == 0:
        persistence = 0.0
    else:
        lastN = sign_series.tail(windowN)
        persistence = float((lastN == latest_sign).sum() / max(1, (lastN != 0).sum()))

    turnover = (vol / oi).replace([np.inf,-np.inf], np.nan)

    # roll filter
    roll_mask = _roll_window_mask(oi, doi, vol, pdrop=roll_pdrop, zthr=roll_z, win=window)

    # basis shift + corr with spot
    basis_shift = np.nan
    corr_spot = np.nan
    if spot_tf is not None and not spot_tf.empty:
        s_aligned = spot_tf.reindex_like(fut_tf)["close"].dropna()
        f_aligned = fut_tf["close"].reindex(s_aligned.index).dropna()
        idx = s_aligned.index.intersection(f_aligned.index)
        if len(idx) >= 5:
            basis = f_aligned.loc[idx] - s_aligned.loc[idx]
            basis_shift = float(basis.diff().iloc[-1] if len(basis) > 1 else 0.0)
            corr = f_aligned.rolling(20).corr(s_aligned)
            corr_spot = float(corr.iloc[-1]) if pd.notna(corr.iloc[-1]) else np.nan

    # squeeze flag
    bb_bw = _bb_width_pct(close, 20, 2.0)
    bw_hist = bb_bw.tail(120).dropna()
    sq_flag = int(len(bw_hist) >= 20 and (bb_bw.iloc[-1] <= np.nanpercentile(bw_hist, squeeze_pct)))

    atr = _atr(d["high"], d["low"], d["close"], 14)
    atr_pctile = float(pd.Series(atr).rank(pct=True).iloc[-1]) if len(atr) else np.nan
    bw_pctile  = float(pd.Series(bb_bw).rank(pct=True).iloc[-1]) if len(bb_bw) else np.nan

    ts = fut_tf.index[-1]
    return OIFeatures(
        dOI=float(doi.iloc[-1] if pd.notna(doi.iloc[-1]) else 0.0),
        dOI_pct=float(doi_pct.iloc[-1] if pd.notna(doi_pct.iloc[-1]) else 0.0),
        OI_z=float(OI_z.iloc[-1] if pd.notna(OI_z.iloc[-1]) else 0.0),
        VOI=float(VOI.iloc[-1] if pd.notna(VOI.iloc[-1]) else 0.0),
        VOI_eff=float(VOI_eff.iloc[-1] if pd.notna(VOI_eff.iloc[-1]) else 0.0),
        concord=float(concord.iloc[-1] if pd.notna(concord.iloc[-1]) else 0.0),
        persistence=float(persistence if pd.notna(persistence) else 0.0),
        turnover=float(turnover.iloc[-1] if pd.notna(turnover.iloc[-1]) else np.nan),
        roll_flag=int(roll_mask.iloc[-1]) if len(roll_mask) else 0,
        basis_shift=float(basis_shift) if not np.isnan(basis_shift) else 0.0,
        tod_adj=float(_time_of_day_adj(ts)),
        squeeze_flag=int(sq_flag),
        diag={
            "ts": ts.value/1e9,
            "corr_fut_spot": float(corr_spot if not np.isnan(corr_spot) else 1.0),
            "bb_bw_pctile": float(bw_pctile if not np.isnan(bw_pctile) else 0.5),
            "atr_pctile": float(atr_pctile if not np.isnan(atr_pctile) else 0.5),
        }
    )

# ========= OI pillar scoring (0..100) =========

@dataclass
class OIPillar:
    flow: float
    quality: float
    structure: float
    pillar: float
    direction: int   # +1 bull, -1 bear, 0 flat

def _clamp100(x: float) -> float: return max(0.0, min(100.0, x))

def _oi_pillar_score(F: OIFeatures, cfg: ConfCfg) -> OIPillar:
    # FLOW (60%)
    sign = 0
    if F.concord > 0: sign = +1
    elif F.concord < 0: sign = -1

    flow = 50.0
    flow += min(25.0, 100.0*max(0.0, F.VOI_eff)) * (1 if sign>0 else (-1 if sign<0 else 0))
    if F.persistence >= cfg.persistence_min and sign != 0:
        flow += 15.0 * (1 if sign>0 else -1)
    if abs(F.OI_z) >= 1.0 and sign != 0:
        flow += 10.0 * (1 if sign>0 else -1)
    flow = _clamp100(flow)

    # QUALITY (25%)
    quality = 50.0
    if not math.isnan(F.turnover):
        if 0.3 <= F.turnover <= 0.8:
            quality += 10.0
        if F.turnover < cfg.turnover_low or F.turnover > cfg.turnover_high:
            quality -= 10.0
    if F.roll_flag == 1:
        quality -= 10.0
    if float(F.diag.get("corr_fut_spot", 1.0)) < 0.3:
        quality -= 5.0
    quality *= F.tod_adj
    quality = _clamp100(quality)

    # STRUCTURE (15%)
    structure = 50.0
    if cfg.basis_confirm == 1:
        if sign>0 and F.basis_shift > 0: structure += 10.0
        elif sign<0 and F.basis_shift < 0: structure += 10.0
        elif sign != 0: structure -= 10.0
    structure = _clamp100(structure)

    pillar = 0.6*flow + 0.25*quality + 0.15*structure
    return OIPillar(flow=flow, quality=quality, structure=structure,
                    pillar=_clamp100(pillar), direction=sign)

# ========= Integration per TF =========

def _fetch_baseline_conf(symbol: str, kind: str, tf: str, use_baseline: bool) -> float:
    if not use_baseline:
        return 0.5
    v = _last_metric(symbol, kind, tf, f"CONF.prob.{tf}")
    return float(v) if v is not None else 0.5

def _integrate_additive(p_base: float, oi_pillar: OIPillar) -> float:
    z = math.log(max(p_base,1e-6)/max(1-p_base,1e-6))
    tilt = (oi_pillar.pillar - 50.0)/50.0
    z += 0.75 * tilt
    return 1.0/(1.0 + math.exp(-z))

def _integrate_bayes(p_base: float, oi: OIFeatures, oi_pillar: OIPillar, cfg: ConfCfg) -> float:
    z = math.log(max(p_base,1e-6)/max(1-p_base,1e-6))
    z += cfg.bayes_k * (1 if oi_pillar.direction>0 else (-1 if oi_pillar.direction<0 else 0))
    z += cfg.bayes_m * (oi.persistence if oi_pillar.direction!=0 else 0.0) * (1 if oi_pillar.direction>0 else -1)
    z += 0.5 * ((oi_pillar.pillar - 50.0)/50.0)
    return 1.0/(1.0 + math.exp(-z))

# ========= Dynamic weighting =========

def _dynamic_mtf_weights(symbol: str, tf_rows_raw: Dict[str, pd.DataFrame], cfg: ConfCfg) -> Dict[str, float]:
    if not cfg.dyn_w_enable:
        return _normalize_map({tf: cfg.mtf_oi_weights.get(tf, 0.0) for tf in cfg.tfs})

    regime: Dict[str, str] = {}
    for tf, dftf in tf_rows_raw.items():
        if dftf.empty or len(dftf) < 30:
            regime[tf] = "neutral"; continue
        close = dftf["close"]
        bb_bw = _bb_width_pct(close, 20, 2.0)
        atr = _atr(dftf["high"], dftf["low"], dftf["close"], 14)
        bw_pct = float(pd.Series(bb_bw).rank(pct=True).iloc[-1]) if len(bb_bw) else 0.5
        atr_pct= float(pd.Series(atr).rank(pct=True).iloc[-1]) if len(atr) else 0.5
        if max(bw_pct, atr_pct) >= 0.70: regime[tf] = "shock"
        elif max(bw_pct, atr_pct) <= 0.40: regime[tf] = "calm"
        else: regime[tf] = "trend"

    weights = dict(cfg.mtf_oi_weights)
    fast_set  = {"15m","30m","25m"}
    slow_set  = {"60m","120m","125m","250m"}
    ref_tf = cfg.dyn_ref_tf if cfg.dyn_ref_tf in tf_rows_raw else next(iter(tf_rows_raw.keys()))
    ref_reg = regime.get(ref_tf, "trend")

    if ref_reg == "shock":
        for tf in list(weights.keys()):
            if tf in fast_set: weights[tf] = weights.get(tf, 0.0) + cfg.dyn_w_fast_bias
            if tf in slow_set: weights[tf] = max(0.0, weights.get(tf, 0.0) - cfg.dyn_w_fast_bias)
    elif ref_reg in {"calm","trend"}:
        for tf in list(weights.keys()):
            if tf in slow_set: weights[tf] = weights.get(tf, 0.0) + cfg.dyn_w_slow_bias
            if tf in fast_set: weights[tf] = max(0.0, weights.get(tf, 0.0) - cfg.dyn_w_slow_bias)

    weights = {k: max(0.0, v) for k,v in weights.items() if k in tf_rows_raw}
    return _normalize_map(weights)

# ========= MTF logic for OI piece =========
def _buildup_from_features(F: OIFeatures) -> tuple[str, float]:
    if F.concord > 0: label = "LONG_BUILDUP"
    elif F.concord < 0: label = "SHORT_BUILDUP"
    else: label = "SHORT_COVERING" if F.concord == 0.5 else ("LONG_UNWINDING" if F.concord == -0.5 else "FLAT")
    strength = max(0.0, min(100.0, 100.0*(0.5*abs(F.VOI_eff) + 0.5*F.persistence)))
    return label, strength

def _mtf_oi_probability(p_ois: Dict[str,float], dir_pers: Dict[str,Tuple[int,float]],
                        weights: Dict[str,float], cfg: ConfCfg) -> float:
    logits, ws = [], []
    aligned = False
    for tf, p in p_ois.items():
        w = weights.get(tf, 0.0)
        ws.append(w)
        p = min(1-1e-6, max(1e-6, p))
        logits.append(math.log(p/(1-p)))
        d, pers = dir_pers.get(tf, (0,0.0))
        if tf in ("60m","120m","125m") and d!=0 and pers >= cfg.mtf_align_gate:
            aligned = True
    wsum = sum(ws) or 1.0
    z = sum((w/wsum)*lz for w,lz in zip(ws, logits))
    p = 1.0/(1.0 + math.exp(-z))
    if aligned: p = min(1.0, p + 0.05)
    return p

# ========= Write metrics =========

def _write_metrics(symbol: str, kind: str, tf_rows: Dict[str, dict],
                   p_mtf_oi: float, p_mtf_final: float, weights_used: Dict[str,float], cfg: ConfCfg):
    rows: List[tuple] = []
    nowts = datetime.now(TZ)

    for tf, st in tf_rows.items():
        ts = st["ts"]
        oi: OIFeatures = st["oi"]
        sc: OIPillar   = st["score"]
        p_base         = st["p_base"]
        p_final        = st["p_final"]

        label, strength = _buildup_from_features(oi)
        rows += [
            (symbol, kind, tf, ts, "OI.buildup_label", 0.0, json.dumps({"label": label}), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.buildup_strength", float(strength), json.dumps({}), cfg.run_id, cfg.source),
        ]

        ctx = {"tf": tf}
        rows += [
            (symbol, kind, tf, ts, "OI.dOI", float(oi.dOI), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.dOI_pct", float(oi.dOI_pct), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.z", float(oi.OI_z), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.VOI", float(oi.VOI), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.VOI_eff", float(oi.VOI_eff), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.concord", float(oi.concord), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.persistence", float(oi.persistence), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.turnover", float(oi.turnover if not math.isnan(oi.turnover) else 0.0), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.roll_flag", float(oi.roll_flag), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.basis_shift", float(oi.basis_shift), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.squeeze", float(oi.squeeze_flag), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.corr_spot", float(oi.diag.get("corr_fut_spot", 1.0)), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.bw_pctile", float(oi.diag.get("bb_bw_pctile", 0.5)), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.atr_pctile", float(oi.diag.get("atr_pctile", 0.5)), json.dumps(ctx), cfg.run_id, cfg.source),
        ]
        rows += [
            (symbol, kind, tf, ts, "OI.flow", float(sc.flow), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.quality", float(sc.quality), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.structure", float(sc.structure), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.pillar", float(sc.pillar), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, "OI.direction", float(sc.direction), json.dumps(ctx), cfg.run_id, cfg.source),
        ]
        rows += [
            (symbol, kind, tf, ts, f"CONF.base.{tf}", float(p_base if p_base is not None else 0.5), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"CONF_OI.prob.{tf}", float(p_final), json.dumps(ctx), cfg.run_id, cfg.source),
            (symbol, kind, tf, ts, f"CONF_OI.score.{tf}", float(round(100.0*p_final,2)), json.dumps(ctx), cfg.run_id, cfg.source),
        ]
        if tf in weights_used:
            rows.append((symbol, kind, tf, ts, f"CONF_OI.mtf_weight.{tf}", float(weights_used[tf]), json.dumps(ctx), cfg.run_id, cfg.source))

    ts_latest = max([st["ts"] for st in tf_rows.values()]) if tf_rows else nowts
    rows += [
        (symbol, kind, "MTF", ts_latest, "CONF_OI.prob.mtf_only", float(p_mtf_oi), json.dumps({}), cfg.run_id, cfg.source),
        (symbol, kind, "MTF", ts_latest, "CONF_OI.prob.mtf", float(p_mtf_final), json.dumps({}), cfg.run_id, cfg.source),
        (symbol, kind, "MTF", ts_latest, "CONF_OI.score.mtf", float(round(100.0*p_mtf_final,2)), json.dumps({}), cfg.run_id, cfg.source),
    ]

    sql = """
        INSERT INTO indicators.values
            (symbol, market_type, interval, ts, metric, val, context, run_id, source)
        VALUES %s
        ON CONFLICT (symbol, market_type, interval, ts, metric) DO UPDATE
           SET val=EXCLUDED.val, context=EXCLUDED.context, run_id=EXCLUDED.run_id, source=EXCLUDED.source
    """
    _exec_values(sql, rows)

# ========= Public driver =========

def process_symbol(symbol: str, *, kind: str, cfg: Optional[ConfCfg] = None) -> int:
    cfg = cfg or load_cfg()
    cfg = _apply_symbol_overrides(cfg, symbol)

    fut_df = _load_5m(symbol, "futures" if kind=="futures" else kind, cfg.lookback_days)
    if fut_df.empty:
        print(f"⚠️ {kind}:{symbol} no 5m data"); return 0

    oi_source_df = fut_df if (kind=="futures" and "oi" in fut_df.columns) else _load_5m(symbol, "futures", cfg.lookback_days)
    if "oi" not in fut_df.columns and "oi" in oi_source_df.columns:
        fut_df = fut_df.join(oi_source_df["oi"], how="left")

    spot_df = _load_5m(symbol, "spot", cfg.lookback_days) if kind == "futures" else None

    tf_rows: Dict[str, dict] = {}
    p_oi_tfs: Dict[str, float] = {}
    dir_pers: Dict[str, Tuple[int,float]] = {}
    raw_by_tf: Dict[str, pd.DataFrame] = {}

    for tf in cfg.tfs:
        if tf not in TF_TO_OFFSET: continue
        dftf = _resample(fut_df, tf)
        if dftf.empty or len(dftf) < 30: continue
        raw_by_tf[tf] = dftf
        sptf = _resample(spot_df, tf) if spot_df is not None and not spot_df.empty else None

        OI = _oi_features_for_tf(dftf, sptf, tf, cfg.oi_window, cfg.squeeze_bw_pct, cfg.roll_pdrop, cfg.roll_z)
        if OI is None: continue
        pillar = _oi_pillar_score(OI, cfg)

        p_base = _fetch_baseline_conf(symbol, kind, tf, cfg.use_baseline_conf)
        p_final = _integrate_additive(p_base, pillar) if cfg.integration == "additive" else _integrate_bayes(p_base, OI, pillar, cfg)

        tf_rows[tf] = {
            "ts": dftf.index[-1].to_pydatetime().replace(tzinfo=TZ),
            "oi": OI, "score": pillar, "p_base": p_base, "p_final": p_final
        }
        p_oi_tfs[tf] = p_final
        dir_pers[tf] = (pillar.direction, OI.persistence)

    if not tf_rows:
        print(f"ℹ️ {kind}:{symbol} OI: no TF rows"); return 0

    weights = _dynamic_mtf_weights(symbol, raw_by_tf, cfg)
    p_mtf_oi = _mtf_oi_probability(p_oi_tfs, dir_pers, weights, cfg)

    cap = 1.0
    d_slow = [dir_pers.get(tf, (0,0.0))[0] for tf in ("120m","125m","60m")]
    if any(d<0 for d in d_slow) and any(d>0 for d,_ in dir_pers.values()):
        cap = cfg.mtf_conflict_cap
    p_mtf_final = min(cap, p_mtf_oi)

    _write_metrics(symbol, kind, tf_rows, p_mtf_oi, p_mtf_final, weights, cfg)
    print(f"✅ {kind}:{symbol} OI-CONF → mtf={p_mtf_final:.3f} ({round(100*p_mtf_final)}), "
          f"tf={ {k:round(100*v,1) for k,v in p_oi_tfs.items()} }, "
          f"w={ {k:round(100*w,1) for k,w in weights.items()} }")
    return 1

def run(symbols: Optional[List[str]] = None, kinds: Iterable[str] = ("futures",)) -> int:
    if symbols is None:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT symbol
                  FROM webhooks.webhook_alerts
                 WHERE status IN ('INDICATOR_PROCESS','SIGNAL_PROCESS')
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
                print(f"❌ {k}:{s} OI-CONF error → {e}")
    print(f"🎯 OI-CONF wrote metrics for {total} run(s) across {len(symbols)} symbol(s)")
    return total

if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:]]
    syms = [a for a in args if not a.startswith("--")]
    ks = [a.split("=",1)[1] for a in args if a.startswith("--kinds=")]
    kinds = tuple(ks[0].split(",")) if ks else ("futures",)
    run(syms or None, kinds=kinds)
