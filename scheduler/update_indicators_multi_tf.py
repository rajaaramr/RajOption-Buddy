# scheduler/update_indicators_multi_tf.py
from __future__ import annotations
import os, json, configparser
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

TZ = timezone.utc
RUN_ID  = os.getenv("RUN_ID",  datetime.now(TZ).strftime("%Y%m%dT%H%M%SZ_ind"))
SOURCE  = os.getenv("SRC", "engine")
INI_PATH = os.getenv("INDICATORS_INI", "indicators.ini")
DEFAULT_INI = "indicators.ini"

# ---------------------------------------------------------------------
# Resampling rules
# ---------------------------------------------------------------------
TF_TO_OFFSET = {
    "15m":"15min",
    "30m":"30min",
    "60m":"60min",
    "90m":"90min",
    "120m":"120min",
    "240m":"240min"
}

def _tf_list_from_ini(cp: configparser.ConfigParser) -> list[str]:
    raw = cp.get("timeframes", "list", fallback="15m,30m,60m,90m,120m")
    return [t.strip() for t in raw.split(",") if t.strip()]

def _obv_family(
    close: pd.Series,
    volume: pd.Series,
    ema_len: int = 20,
    z_short: int = 20,
    z_long: int = 100,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_sig:  int = 9,
    zh_win:    int = 100,
) -> pd.DataFrame:
    """
    Pure computation: build OBV core + momentum-on-OBV set.
    No DB access, no row emission.
    """
    # Base OBV and helpers
    obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()
    obv_ema   = obv.ewm(span=ema_len, adjust=False).mean()
    obv_delta = obv.diff()
    zs = (obv - obv.rolling(z_short).mean()) / (obv.rolling(z_short).std(ddof=0) + 1e-9)
    zl = (obv - obv.rolling(z_long).mean())  / (obv.rolling(z_long).std(ddof=0)  + 1e-9)

    # Momentum (MACD-on-OBV)
    obv_fast = obv.ewm(span=macd_fast, adjust=False).mean()
    obv_slow = obv.ewm(span=macd_slow, adjust=False).mean()
    obv_mom  = obv_fast - obv_slow
    obv_sig  = obv_mom.ewm(span=macd_sig, adjust=False).mean()
    obv_hist = obv_mom - obv_sig
    obv_sig_ema  = obv_sig.ewm(span=macd_sig, adjust=False).mean()
    obv_hist_ema = obv_hist.ewm(span=macd_sig, adjust=False).mean()
    obv_zh = (obv_hist - obv_hist.rolling(zh_win).mean()) / (obv_hist.rolling(zh_win).std(ddof=0) + 1e-9)

    return pd.DataFrame({
        "OBV": obv,
        "OBV.ema": obv_ema,
        "OBV.delta": obv_delta,
        "OBV.zs": zs,
        "OBV.zl": zl,
        "OBV.SIG": obv_sig,
        "OBV.HIST": obv_hist,
        "OBV.SIG.EMA": obv_sig_ema,
        "OBV.HIST.EMA": obv_hist_ema,
        "OBV.ZH": obv_zh,
    })

def _dbg_head(tag: str, df: pd.DataFrame, n: int = 3):
    try:
        print(f"\n[DBG] {tag} dtypes:\n{df.dtypes}")
        print(f"[DBG] {tag} head:\n{df.head(n)}\n")
    except Exception:
        pass

def cci(h: pd.Series, l: pd.Series, c: pd.Series, n:int=20) -> pd.Series:
    tp  = (h + l + c) / 3.0
    sma = tp.rolling(n, min_periods=n).mean()
    md  = (tp - sma).abs().rolling(n, min_periods=n).mean()
    denom = 0.015 * md.replace(0, pd.NA)
    return ((tp - sma) / denom).astype("float64")

def stoch_kd(
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
    k_len: int = 14,
    d_len: int = 3,
    smooth_k: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Robust, smoothed Stochastic %K / %D.

    Uses:
      - %K_raw = (C - LL(k)) / (HH(k) - LL(k)) * 100
      - %K = SMA(%K_raw, smooth_k)
      - %D = SMA(%K, d_len)
    """
    h = pd.to_numeric(h, errors="coerce")
    l = pd.to_numeric(l, errors="coerce")
    c = pd.to_numeric(c, errors="coerce")

    if h.dropna().empty or l.dropna().empty or c.dropna().empty:
        idx = c.index
        return (
            pd.Series(index=idx, dtype="float64"),
            pd.Series(index=idx, dtype="float64"),
        )

    ll = l.rolling(k_len, min_periods=k_len).min()
    hh = h.rolling(k_len, min_periods=k_len).max()
    rng = (hh - ll).replace(0, pd.NA)

    k_raw = ((c - ll) / rng) * 100.0
    k = k_raw.rolling(smooth_k, min_periods=1).mean()
    d = k.rolling(d_len, min_periods=1).mean()

    return k.astype("float64"), d.astype("float64")

def psar(
    h: pd.Series,
    l: pd.Series,
    af_step: float = 0.02,
    af_max: float = 0.2,
    # Backward-compat keywords used in older code / libs
    step: float | None = None,
    max_step: float | None = None,
    af: float | None = None,
    max_af: float | None = None,
    **kwargs,
) -> pd.Series:
    """
    PSAR with backward-compatible kwargs:
    - Accepts af_step / af_max (new)
    - Also accepts step / max_step / af / max_af (old call styles)
    - Ignores any extra kwargs via **kwargs so we never explode on unknown keys
    """

    # 🔁 Normalize all alias params into af_step / af_max
    # Priority: explicit aliases override defaults
    if af is not None:
        af_step = float(af)
    if step is not None:
        af_step = float(step)

    if max_af is not None:
        af_max = float(max_af)
    if max_step is not None:
        af_max = float(max_step)

    idx = h.index
    sar = pd.Series(index=idx, dtype="float64")
    if len(h) < 2:
        return sar

    # Start assuming uptrend from first bar
    uptrend = True
    ep = h.iloc[0]          # extreme point
    sar.iloc[0] = l.iloc[0]
    af_val = af_step

    prev_sar = sar.iloc[0]
    prev_ep = ep
    prev_up = uptrend

    for i in range(1, len(h)):
        hi, lo = h.iloc[i], l.iloc[i]

        # 1) Move SAR
        next_sar = prev_sar + af_val * (prev_ep - prev_sar)

        # 2) Clamp SAR inside previous two bars range
        if prev_up:
            next_sar = min(
                next_sar,
                l.iloc[i - 1],
                l.iloc[i - 2] if i >= 2 else l.iloc[i - 1],
            )
        else:
            next_sar = max(
                next_sar,
                h.iloc[i - 1],
                h.iloc[i - 2] if i >= 2 else h.iloc[i - 1],
            )

        # 3) Trend logic
        if prev_up:
            # still in uptrend?
            if lo > next_sar:
                # continue uptrend
                if hi > prev_ep:
                    prev_ep = hi
                    af_val = min(af_val + af_step, af_max)
                prev_sar = next_sar
                prev_up = True
            else:
                # reverse to downtrend
                prev_up = False
                prev_sar = prev_ep
                prev_ep = lo
                af_val = af_step
        else:
            # in downtrend
            if hi < next_sar:
                # continue downtrend
                if lo < prev_ep:
                    prev_ep = lo
                    af_val = min(af_val + af_step, af_max)
                prev_sar = next_sar
                prev_up = False
            else:
                # reverse to uptrend
                prev_up = True
                prev_sar = prev_ep
                prev_ep = hi
                af_val = af_step

        sar.iloc[i] = prev_sar

    return sar

def _load_ini_any_encoding(path: str | Path) -> configparser.ConfigParser:
    txt = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                txt = f.read()
            break
        except UnicodeDecodeError:
            continue
    if txt is None:
        raise UnicodeDecodeError("ini", b"", 0, 1, "could not decode with common encodings")

    # normalize curly quotes to straight quotes to avoid parser surprises
    trans = {0x2018: 39, 0x2019: 39, 0x201C: 34, 0x201D: 34}
    txt = txt.translate(trans)

    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cfg.read_string(txt)
    return cfg

# ---------------------------------------------------------------------
# INI parsing
# ---------------------------------------------------------------------
def load_cfg(ini: str = INI_PATH) -> Dict[str, object]:
    cfg = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#"),
        interpolation=None,
        strict=False
    )
    read_files = cfg.read(ini)
    if not read_files:
        raise FileNotFoundError(f"Could not read INI at {ini}")

    tf_csv = cfg.get("timeframes", "list", fallback="15m,30m,60m,120m,240m")
    tf_list = [t.strip() for t in tf_csv.split(",") if t.strip()]

    metrics = {
        "RSI":   cfg.getboolean("metrics", "RSI",   fallback=True),
        "RMI":   cfg.getboolean("metrics", "RMI",   fallback=True),
        "MACD":  cfg.getboolean("metrics", "MACD",  fallback=True),
        "ADX":   cfg.getboolean("metrics", "ADX",   fallback=True),
        "ROC":   cfg.getboolean("metrics", "ROC",   fallback=True),
        "ATR":   cfg.getboolean("metrics", "ATR",   fallback=True),
        "MFI":   cfg.getboolean("metrics", "MFI",   fallback=True),
        "CCI":   cfg.getboolean("metrics", "CCI",   fallback=True),
        "STOCH": cfg.getboolean("metrics", "STOCH", fallback=True),
        "PSAR":  cfg.getboolean("metrics", "PSAR",  fallback=True),
    }

    ema_csv = cfg.get("metrics", "EMA", fallback="5,10,15,20,25") or ""
    ema_list = [int(x.strip()) for x in ema_csv.split(",") if x.strip().isdigit()]
    if not ema_list:
        ema_list = [5, 10, 15, 20, 25]

    P: Dict[str, object] = {
        "TF_LIST": tf_list,
        "METRICS": metrics,
        "EMA_LIST": ema_list,
        "RSI_LEN":   cfg.getint("params","RSI.length",   fallback=14),
        "RMI_LEN":   cfg.getint("params","RMI.length",   fallback=14),
        "RMI_MOM":   cfg.getint("params","RMI.momentum", fallback=5),
        "MACD_FAST": cfg.getint("params","MACD.fast",    fallback=12),
        "MACD_SLOW": cfg.getint("params","MACD.slow",    fallback=26),
        "MACD_SIG":  cfg.getint("params","MACD.signal",  fallback=9),
        "ADX_LEN":   cfg.getint("params","ADX.length",   fallback=14),
        "ROC_LEN":   cfg.getint("params","ROC.length",   fallback=14),
        "ATR_LEN":   cfg.getint("params","ATR.length",   fallback=14),
        "MFI_LEN":   cfg.getint("params","MFI.length",   fallback=14),
        "CCI_LEN":   cfg.getint("params","CCI.length",   fallback=20),
        "STOCH_K":   cfg.getint("params","STOCH.k",      fallback=14),
        "STOCH_D":   cfg.getint("params","STOCH.d",      fallback=3),
        # smoothing for %K – not in INI, but we default.
        "STOCH_SMOOTH": cfg.getint("params","STOCH.smooth_k", fallback=3),
        "PSAR_STEP": cfg.getfloat("params","PSAR.af_step", fallback=0.02),
        "PSAR_MAX":  cfg.getfloat("params","PSAR.af_max",  fallback=0.20),
    }

    P.update({
        "ML_EXPORT":  cfg.getboolean("indicators", "ml_export_enabled", fallback=False),
        "FEATURESET": cfg.get("ml", "featureset", fallback="v1"),
        "OBV_EMA":    cfg.getint("params","OBV.ema",     fallback=20),
        "OBV_ZS":     cfg.getint("params","OBV.z_short", fallback=20),
        "OBV_ZL":     cfg.getint("params","OBV.z_long",  fallback=100),
    })

    print(f"[CFG] TF_LIST={P['TF_LIST']} EMA_LIST={P['EMA_LIST']} METRICS={P['METRICS']}")
    return P

# ---------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------
def sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    d = df.copy()
    # index must be datetime (tz-aware or naive; we’ll treat as UTC)
    if "ts" in d.columns:
        d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
        d = d.dropna(subset=["ts"]).set_index("ts")
    d.index = pd.to_datetime(d.index, utc=True, errors="coerce")
    d = d[~d.index.isna()]
    d = d.sort_index()

    d.columns = [c.lower() for c in d.columns]
    alias = {
        "open_price":"open","high_price":"high","low_price":"low","close_price":"close",
        "o":"open","h":"high","l":"low","c":"close","v":"volume"
    }
    for old,new in alias.items():
        if old in d.columns and new not in d.columns:
            d[new] = d[old]
    for c in ["open","high","low","close","volume"]:
        if c not in d.columns:
            d[c] = pd.NA
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["volume"] = d["volume"].fillna(0.0)
    d = d.dropna(subset=["open","high","low","close"])
    return d.astype({"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"})

def resample(df5: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df5 is None or df5.empty or tf == "15m":
        return sanitize_ohlcv(df5)
    d = sanitize_ohlcv(df5)
    rule = TF_TO_OFFSET.get(tf)
    if not rule:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    out = pd.DataFrame({
        "open":   d["open"].resample(rule, label="right", closed="right").first(),
        "high":   d["high"].resample(rule, label="right", closed="right").max(),
        "low":    d["low"].resample(rule, label="right", closed="right").min(),
        "close":  d["close"].resample(rule, label="right", closed="right").last(),
        "volume": d["volume"].resample(rule, label="right", closed="right").sum(),
    }).dropna(how="any").astype(
        {"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"}
    )
    return out

# ---------------------------------------------------------------------
# Indicator math
# ---------------------------------------------------------------------
def ema(s: pd.Series, n:int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n:int) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = (-d).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0/n, adjust=False).mean()
    al = loss.ewm(alpha=1.0/n, adjust=False).mean().replace(0, pd.NA)
    rs = ag/al
    return 100 - (100/(1+rs))

def rmi(close: pd.Series, n:int, mom:int) -> pd.Series:
    m = close.diff(mom)
    gain = m.clip(lower=0.0)
    loss = (-m).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0/n, adjust=False).mean()
    al = loss.ewm(alpha=1.0/n, adjust=False).mean().replace(0, pd.NA)
    rs = ag/al
    return 100 - (100/(1+rs))

def macd(close: pd.Series, fast:int, slow:int, sig:int):
    line = ema(close, fast) - ema(close, slow)
    signal = line.ewm(span=sig, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    a = (h - l).values
    b = (h - prev_c).abs().values
    d = (l - prev_c).abs().values
    tr = np.maximum.reduce([a, b, d])
    return pd.Series(tr, index=h.index, dtype="float64")

def atr(h: pd.Series, l: pd.Series, c: pd.Series, n:int) -> pd.Series:
    tr = true_range(h, l, c)
    return tr.ewm(alpha=1.0/n, adjust=False).mean().astype("float64")

def adx(h: pd.Series, l: pd.Series, c: pd.Series, n:int):
    up = h.diff()
    dn = -l.diff()
    plus_dm  = ((up > dn) & (up > 0.0)) * up
    minus_dm = ((dn > up) & (dn > 0.0)) * dn
    atr_s = true_range(h, l, c).ewm(alpha=1.0/n, adjust=False).mean()
    plus_di  = 100.0 * (plus_dm.ewm(alpha=1.0/n, adjust=False).mean()  / atr_s.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0/n, adjust=False).mean() / atr_s.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0
    adx_v = dx.ewm(alpha=1.0/n, adjust=False).mean()
    return adx_v.astype("float64"), plus_di.astype("float64"), minus_di.astype("float64")

def roc(close: pd.Series, n:int) -> pd.Series:
    return (close/close.shift(n)-1.0)*100.0

def mfi(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, n:int) -> pd.Series:
    tp = (h+l+c)/3.0
    mf = tp*v
    pos = (tp>tp.shift(1))*mf
    neg = (tp<tp.shift(1))*mf
    mr = pos.rolling(n, min_periods=n).sum() / (neg.rolling(n, min_periods=n).sum().replace(0,pd.NA))
    return 100 - (100/(1+mr))

# ---------------------------------------------------------------------
# PUBLIC (pure): build rows for indicators.values
# ---------------------------------------------------------------------
# get_last_ts: (symbol, kind, tf, metric) -> Optional[pd.Timestamp]
GetLastTs = Callable[[str, str, str, str], Optional[pd.Timestamp]]
# load_15m: (symbol, kind) -> pd.DataFrame (indexed by datetime in UTC)
Load15m = Callable[[str, str], pd.DataFrame]

def build_indicator_rows(
    df5: pd.DataFrame,
    *,
    symbol: str,
    kind: str,
    P: Dict[str, object],
    get_last_ts: GetLastTs,
    feature_snapshot_cb: Optional[Callable[[pd.Timestamp, str, str, Dict[str, float], str], None]] = None,
) -> Tuple[List[tuple], Dict[str, int]]:
    """
    Convert a 5m/15m OHLCV dataframe into INSERT rows for indicators.values.
    Returns (rows, stats) where rows is List[tuple] matching:
      (symbol, market_type, interval, ts, metric, val, context, run_id, source)
    """
    rows: List[tuple] = []
    stats = {"attempted": 0, "frames": 0}

    if df5 is None or df5.empty:
        return rows, stats

    # Helper for safe float conversion
    def _safe_floats(s: pd.Series) -> pd.Series:
        return s.replace({pd.NA: np.nan}).astype("float64")

    for tf in P["TF_LIST"]:  # type: ignore
        if tf not in TF_TO_OFFSET:
            continue
        dftf = df5 if tf == "15m" else resample(df5, tf)
        if dftf.empty:
            continue

        stats["frames"] += 1

        # --- RSI ---
        if P["METRICS"].get("RSI", True):
            try:
                s = _safe_floats(rsi(dftf["close"], P["RSI_LEN"])).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "RSI")
                if cutoff is not None:
                    s = s[s.index > cutoff]
                rows += [
                    (symbol, kind, tf, ts.to_pydatetime(), "RSI", float(val),
                     json.dumps({"length":P["RSI_LEN"],"src":"close"}), RUN_ID, SOURCE)
                    for ts, val in s.items()
                ]
            except Exception as e:
                print(f"[SKIP RSI] {kind}:{symbol} {tf} → {e}")

        # --- RMI ---
        if P["METRICS"].get("RMI", True):
            try:
                s = _safe_floats(rmi(dftf["close"], P["RMI_LEN"], P["RMI_MOM"])).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "RMI")
                if cutoff is not None:
                    s = s[s.index > cutoff]
                rows += [
                    (symbol, kind, tf, ts.to_pydatetime(), "RMI", float(val),
                     json.dumps({"length":P["RMI_LEN"],"momentum":P["RMI_MOM"],"src":"close"}), RUN_ID, SOURCE)
                    for ts, val in s.items()
                ]
            except Exception as e:
                print(f"[SKIP RMI] {kind}:{symbol} {tf} → {e}")

        # --- MACD trio ---
        if P["METRICS"].get("MACD", True):
            try:
                line, sig, hist = macd(dftf["close"], P["MACD_FAST"], P["MACD_SLOW"], P["MACD_SIG"])
                for name, series in [("MACD.macd", line), ("MACD.signal", sig), ("MACD.hist", hist)]:
                    s = _safe_floats(series).dropna()
                    cutoff = get_last_ts(symbol, kind, tf, name)
                    if cutoff is not None:
                        s = s[s.index > cutoff]
                    rows += [
                        (symbol, kind, tf, ts.to_pydatetime(), name, float(val),
                         json.dumps({"fast":P["MACD_FAST"],"slow":P["MACD_SLOW"],
                                     "signal":P["MACD_SIG"],"src":"close"}), RUN_ID, SOURCE)
                        for ts, val in s.items()
                    ]
            except Exception as e:
                print(f"[SKIP MACD] {kind}:{symbol} {tf} → {e}")

        # --- ADX/DI ---
        if P["METRICS"].get("ADX", True):
            try:
                adx_v, plus_di, minus_di = adx(dftf["high"], dftf["low"], dftf["close"], P["ADX_LEN"])
                for name, series in [("ADX", adx_v), ("DI+", plus_di), ("DI-", minus_di)]:
                    s = _safe_floats(series).dropna()
                    cutoff = get_last_ts(symbol, kind, tf, name)
                    if cutoff is not None:
                        s = s[s.index > cutoff]
                    rows += [
                        (symbol, kind, tf, ts.to_pydatetime(), name, float(val),
                         json.dumps({"length":P["ADX_LEN"]}), RUN_ID, SOURCE)
                        for ts, val in s.items()
                    ]
            except Exception as e:
                print(f"[SKIP ADX] {kind}:{symbol} {tf} → {e}")

        # --- ROC ---
        if P["METRICS"].get("ROC", True):
            try:
                s = _safe_floats(roc(dftf["close"], P["ROC_LEN"])).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "ROC")
                if cutoff is not None:
                    s = s[s.index > cutoff]
                rows += [
                    (symbol, kind, tf, ts.to_pydatetime(), "ROC", float(val),
                     json.dumps({"length":P["ROC_LEN"],"src":"close"}), RUN_ID, SOURCE)
                    for ts, val in s.items()
                ]
            except Exception as e:
                print(f"[SKIP ROC] {kind}:{symbol} {tf} → {e}")

        # --- ATR ---
        if P["METRICS"].get("ATR", True):
            try:
                s = _safe_floats(atr(dftf["high"], dftf["low"], dftf["close"], P["ATR_LEN"])).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "ATR")
                if cutoff is not None:
                    s = s[s.index > cutoff]
                rows += [
                    (symbol, kind, tf, ts.to_pydatetime(), "ATR", float(val),
                     json.dumps({"length":P["ATR_LEN"]}), RUN_ID, SOURCE)
                    for ts, val in s.items()
                ]
            except Exception as e:
                print(f"[SKIP ATR] {kind}:{symbol} {tf} → {e}")

        # --- MFI ---
        if P["METRICS"].get("MFI", True):
            try:
                s = _safe_floats(mfi(dftf["high"], dftf["low"], dftf["close"], dftf["volume"], P["MFI_LEN"])).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "MFI")
                if cutoff is not None:
                    s = s[s.index > cutoff]
                rows += [
                    (symbol, kind, tf, ts.to_pydatetime(), "MFI", float(val),
                     json.dumps({"length":P["MFI_LEN"]}), RUN_ID, SOURCE)
                    for ts, val in s.items()
                ]
            except Exception as e:
                print(f"[SKIP MFI] {kind}:{symbol} {tf} → {e}")

        # --- EMAs ---
        ema_list: List[int] = P.get("EMA_LIST", [])
        if ema_list:
            try:
                for L in ema_list:
                    series = _safe_floats(ema(dftf["close"], int(L))).dropna()
                    metric = f"EMA.{int(L)}"
                    cutoff = get_last_ts(symbol, kind, tf, metric)
                    if cutoff is not None:
                        series = series[series.index > cutoff]
                    rows += [
                        (symbol, kind, tf, ts.to_pydatetime(), metric, float(val),
                         json.dumps({"length": int(L), "src": "close"}), RUN_ID, SOURCE)
                        for ts, val in series.items()
                    ]
            except Exception as e:
                print(f"[SKIP EMA] {kind}:{symbol} {tf} → {e}")

        # --- OBV family (core + momentum on OBV) ---
        try:
            obv_df = _obv_family(
                dftf["close"], dftf["volume"],
                ema_len=int(P["OBV_EMA"]),
                z_short=int(P["OBV_ZS"]),
                z_long=int(P["OBV_ZL"]),
                macd_fast=int(P.get("MACD_FAST", 12)),
                macd_slow=int(P.get("MACD_SLOW", 26)),
                macd_sig=int(P.get("MACD_SIG", 9)),
                zh_win=int(P.get("OBV_ZL", 100)),
            ).dropna(how="all")

            for metric_col, series in obv_df.items():
                s = _safe_floats(series).dropna()
                cutoff = get_last_ts(symbol, kind, tf, metric_col)
                if cutoff is not None:
                    s = s[s.index > cutoff]
                rows += [
                    (symbol, kind, tf, ts.to_pydatetime(), metric_col, float(val),
                     json.dumps({
                         "ema": int(P["OBV_EMA"]),
                         "z_short": int(P["OBV_ZS"]),
                         "z_long": int(P["OBV_ZL"]),
                         "fast": int(P.get("MACD_FAST", 12)),
                         "slow": int(P.get("MACD_SLOW", 26)),
                         "signal": int(P.get("MACD_SIG", 9)),
                     }),
                     RUN_ID, SOURCE)
                    for ts, val in s.items()
                ]

            if P.get("ML_EXPORT") and feature_snapshot_cb is not None and not dftf.empty:
                last_ts = dftf.index[-1]
                feats = {k: float(obv_df[k].iloc[-1]) for k in obv_df.columns if pd.notna(obv_df[k].iloc[-1])}
                if feats:
                    feature_snapshot_cb(last_ts, symbol, tf, feats, str(P.get("FEATURESET","v1")))
        except Exception as e:
            print(f"[SKIP OBV] {kind}:{symbol} {tf} → {e}")

        # --- CCI ---
        if P["METRICS"].get("CCI", True):
            try:
                s = _safe_floats(cci(dftf["high"], dftf["low"], dftf["close"], int(P["CCI_LEN"]))).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "CCI")
                if cutoff is not None:
                    s = s[s.index > cutoff]
                rows += [
                    (symbol, kind, tf, ts.to_pydatetime(), "CCI", float(val),
                     json.dumps({"length": int(P["CCI_LEN"])}), RUN_ID, SOURCE)
                    for ts, val in s.items()
                ]
            except Exception as e:
                print(f"[SKIP CCI] {kind}:{symbol} {tf} → {e}")

        # --- STOCH %K / %D ---
        if P["METRICS"].get("STOCH", True):
            try:
                k, d = stoch_kd(
                    dftf["high"],
                    dftf["low"],
                    dftf["close"],
                    int(P["STOCH_K"]),
                    int(P["STOCH_D"]),
                    int(P.get("STOCH_SMOOTH", 3)),
                )
                for name, series in [("STOCH.K", k), ("STOCH.D", d)]:
                    s = _safe_floats(series).dropna()
                    cutoff = get_last_ts(symbol, kind, tf, name)
                    if cutoff is not None:
                        s = s[s.index > cutoff]
                    rows += [
                        (symbol, kind, tf, ts.to_pydatetime(), name, float(val),
                         json.dumps({
                             "k": int(P["STOCH_K"]),
                             "d": int(P["STOCH_D"]),
                             "smooth_k": int(P.get("STOCH_SMOOTH", 3)),
                         }),
                         RUN_ID, SOURCE)
                        for ts, val in s.items()
                    ]
            except Exception as e:
                print(f"[SKIP STOCH] {kind}:{symbol} {tf} → {e}")

        # --- PSAR ---
        if P["METRICS"].get("PSAR", True):
            try:
                s = _safe_floats(psar(
                    dftf["high"],
                    dftf["low"],
                    af_step=float(P["PSAR_STEP"]),
                    af_max=float(P["PSAR_MAX"]),
                )).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "PSAR")
                if cutoff is not None:
                    s = s[s.index > cutoff]
                rows += [
                    (symbol, kind, tf, ts.to_pydatetime(), "PSAR", float(val),
                     json.dumps({"af_step": float(P["PSAR_STEP"]), "af_max": float(P["PSAR_MAX"])}),
                     RUN_ID, SOURCE)
                    for ts, val in s.items()
                ]
            except Exception as e:
                print(f"[SKIP PSAR] {kind}:{symbol} {tf} → {e}")

    stats["attempted"] = len(rows)
    return rows, stats

# ---------------------------------------------------------------------
# Batch helpers (pure – no DB). The caller provides I/O callbacks.
# ---------------------------------------------------------------------
def update_indicators_multi_tf(
    *,
    symbols: List[str],
    kinds: Tuple[str, ...] = ("spot","futures"),
    load_15m: Load15m,
    get_last_ts: GetLastTs,
    P: Optional[Dict[str, object]] = None,
    feature_snapshot_cb: Optional[Callable[[pd.Timestamp, str, str, Dict[str, float], str], None]] = None,
) -> Dict[str, object]:
    """
    Pure batch builder.
    - load_15m(symbol, kind) -> pd.DataFrame (15m OHLCV)
    - get_last_ts(symbol, kind, tf, metric) -> Optional[pd.Timestamp]
    Returns:
      {"rows": List[tuple], "attempted": int, "frames": int}
    """
    P = P or load_cfg()
    all_rows: List[tuple] = []
    attempted = frames = 0

    for s in symbols:
        for k in kinds:
            df5 = load_15m(s, k)
            df5 = sanitize_ohlcv(df5)
            _dbg_head(f"load_15m {k}:{s}", df5)
            rows, st = build_indicator_rows(
                df5,
                symbol=s,
                kind=k,
                P=P,
                get_last_ts=get_last_ts,
                feature_snapshot_cb=feature_snapshot_cb,
            )
            all_rows.extend(rows)
            attempted += st.get("attempted", 0)
            frames    += st.get("frames", 0)

    print(f"[WRITE] classic batch → tried {attempted} rows across {frames} TF frames")
    return {"rows": all_rows, "attempted": attempted, "frames": frames}

def run(
    *,
    symbols: Optional[List[str]] = None,
    kinds: Tuple[str, ...] = ("spot","futures"),
    load_15m: Optional[Load15m] = None,
    get_last_ts: Optional[GetLastTs] = None,
    P: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """
    Compatibility alias. You MUST provide load_15m and get_last_ts when
    calling this module directly, since this file is DB-free by design.
    """
    if symbols is None:
        symbols = []
    if load_15m is None or get_last_ts is None:
        raise ValueError("run(...) requires load_15m and get_last_ts callbacks (DB-free module).")
    return update_indicators_multi_tf(
        symbols=symbols,
        kinds=kinds,
        load_15m=load_15m,
        get_last_ts=get_last_ts,
        P=P,
    )
