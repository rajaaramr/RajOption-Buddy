# scheduler/update_indicators_multi_tf.py
from __future__ import annotations
import os, json, configparser
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime, timezone
import configparser
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
    "5m":"5min","15m":"15min","25m":"25min","30m":"30min",
    "65m":"65min","125m":"125min","250m":"250min","1h":"1h","2h":"2h"
}

def _dbg_head(tag: str, df: pd.DataFrame, n: int = 3):
    try:
        print(f"\n[DBG] {tag} dtypes:\n{df.dtypes}")
        print(f"[DBG] {tag} head:\n{df.head(n)}\n")
    except Exception:
        pass

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

    tf_csv = cfg.get("timeframes", "list", fallback="5m,15m,25m,30m,65m,125m,250m,1h,2h")
    tf_list = [t.strip() for t in tf_csv.split(",") if t.strip()]

    metrics = {
        "RSI": cfg.getboolean("metrics", "RSI", fallback=True),
        "RMI": cfg.getboolean("metrics", "RMI", fallback=True),
        "MACD": cfg.getboolean("metrics", "MACD", fallback=True),
        "ADX": cfg.getboolean("metrics", "ADX", fallback=True),
        "ROC": cfg.getboolean("metrics", "ROC", fallback=True),
        "ATR": cfg.getboolean("metrics", "ATR", fallback=True),
        "MFI": cfg.getboolean("metrics", "MFI", fallback=True),
    }
    ema_csv = cfg.get("metrics", "EMA", fallback="5,10,15,20,25") or ""
    ema_list = [int(x.strip()) for x in ema_csv.split(",") if x.strip().isdigit()]
    if not ema_list:
        ema_list = [5, 10, 15, 20, 25]

    P = {
        "TF_LIST": tf_list,
        "METRICS": metrics,
        "EMA_LIST": ema_list,
        "RSI_LEN":   cfg.getint("params","RSI.length",fallback=14),
        "RMI_LEN":   cfg.getint("params","RMI.length",fallback=14),
        "RMI_MOM":   cfg.getint("params","RMI.momentum",fallback=5),
        "MACD_FAST": cfg.getint("params","MACD.fast",fallback=12),
        "MACD_SLOW": cfg.getint("params","MACD.slow",fallback=26),
        "MACD_SIG":  cfg.getint("params","MACD.signal",fallback=9),
        "ADX_LEN":   cfg.getint("params","ADX.length",fallback=14),
        "ROC_LEN":   cfg.getint("params","ROC.length",fallback=14),
        "ATR_LEN":   cfg.getint("params","ATR.length",fallback=14),
        "MFI_LEN":   cfg.getint("params","MFI.length",fallback=14),
    }

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
        if c not in d.columns: d[c] = pd.NA
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["volume"] = d["volume"].fillna(0.0)
    d = d.dropna(subset=["open","high","low","close"])
    return d.astype({"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"})

def resample(df5: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df5 is None or df5.empty or tf == "5m":
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
    gain = d.clip(lower=0.0); loss = (-d).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0/n, adjust=False).mean()
    al = loss.ewm(alpha=1.0/n, adjust=False).mean().replace(0, pd.NA)
    rs = ag/al
    return 100 - (100/(1+rs))

def rmi(close: pd.Series, n:int, mom:int) -> pd.Series:
    m = close.diff(mom)
    gain = m.clip(lower=0.0); loss = (-m).clip(lower=0.0)
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
    up = h.diff(); dn = -l.diff()
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
    tp = (h+l+c)/3.0; mf = tp*v
    pos = (tp>tp.shift(1))*mf; neg=(tp<tp.shift(1))*mf
    mr = pos.rolling(n, min_periods=n).sum() / (neg.rolling(n, min_periods=n).sum().replace(0,pd.NA))
    return 100 - (100/(1+mr))

# ---------------------------------------------------------------------
# PUBLIC (pure): build rows for indicators.values
# ---------------------------------------------------------------------
# get_last_ts: (symbol, kind, tf, metric) -> Optional[pd.Timestamp]
GetLastTs = Callable[[str, str, str, str], Optional[pd.Timestamp]]
# load_5m: (symbol, kind) -> pd.DataFrame (indexed by datetime in UTC)
Load5m = Callable[[str, str], pd.DataFrame]

def build_indicator_rows(
    df5: pd.DataFrame,
    *,
    symbol: str,
    kind: str,
    P: Dict[str, object],
    get_last_ts: GetLastTs
) -> Tuple[List[tuple], Dict[str, int]]:
    """
    Convert a 5m OHLCV dataframe into INSERT rows for indicators.values.
    Returns (rows, stats) where rows is List[tuple] matching:
      (symbol, market_type, interval, ts, metric, val, context, run_id, source)
    """
    rows: List[tuple] = []
    stats = {"attempted": 0, "frames": 0}

    if df5 is None or df5.empty:
        return rows, stats

    for tf in P["TF_LIST"]:  # type: ignore
        if tf not in TF_TO_OFFSET:
            continue
        dftf = df5 if tf == "5m" else resample(df5, tf)
        if dftf.empty:
            continue

        stats["frames"] += 1

        # --- RSI ---
        if P["METRICS"].get("RSI", True):
            try:
                s = rsi(dftf["close"], P["RSI_LEN"]).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "RSI")
                if cutoff is not None: s = s[s.index > cutoff]
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
                s = rmi(dftf["close"], P["RMI_LEN"], P["RMI_MOM"]).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "RMI")
                if cutoff is not None: s = s[s.index > cutoff]
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
                for name, series in [("MACD.line", line), ("MACD.signal", sig), ("MACD.hist", hist)]:
                    s = series.dropna()
                    cutoff = get_last_ts(symbol, kind, tf, name)
                    if cutoff is not None: s = s[s.index > cutoff]
                    rows += [
                        (symbol, kind, tf, ts.to_pydatetime(), name, float(val),
                         json.dumps({"fast":P["MACD_FAST"],"slow":P["MACD_SLOW"],"signal":P["MACD_SIG"],"src":"close"}), RUN_ID, SOURCE)
                        for ts, val in s.items()
                    ]
            except Exception as e:
                print(f"[SKIP MACD] {kind}:{symbol} {tf} → {e}")

        # --- ADX/DI ---
        if P["METRICS"].get("ADX", True):
            try:
                adx_v, plus_di, minus_di = adx(dftf["high"], dftf["low"], dftf["close"], P["ADX_LEN"])
                for name, series in [("ADX", adx_v), ("DI+", plus_di), ("DI-", minus_di)]:
                    s = series.dropna()
                    cutoff = get_last_ts(symbol, kind, tf, name)
                    if cutoff is not None: s = s[s.index > cutoff]
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
                s = roc(dftf["close"], P["ROC_LEN"]).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "ROC")
                if cutoff is not None: s = s[s.index > cutoff]
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
                s = atr(dftf["high"], dftf["low"], dftf["close"], P["ATR_LEN"]).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "ATR")
                if cutoff is not None: s = s[s.index > cutoff]
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
                s = mfi(dftf["high"], dftf["low"], dftf["close"], dftf["volume"], P["MFI_LEN"]).dropna()
                cutoff = get_last_ts(symbol, kind, tf, "MFI")
                if cutoff is not None: s = s[s.index > cutoff]
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
                    series = ema(dftf["close"], int(L)).dropna()
                    metric = f"EMA.{int(L)}"
                    cutoff = get_last_ts(symbol, kind, tf, metric)
                    if cutoff is not None:
                        series = series[series.index > cutoff]
                    rows += [
                        (symbol, kind, tf, ts.to_pydatetime(), metric, float(val),
                         json.dumps({"length":int(L),"src":"close"}), RUN_ID, SOURCE)
                        for ts, val in series.items()
                    ]
            except Exception as e:
                print(f"[SKIP EMA] {kind}:{symbol} {tf} → {e}")

    stats["attempted"] = len(rows)
    return rows, stats

# ---------------------------------------------------------------------
# Batch helpers (pure – no DB). The caller provides I/O callbacks.
# ---------------------------------------------------------------------
def update_indicators_multi_tf(
    *,
    symbols: List[str],
    kinds: Tuple[str, ...] = ("spot","futures"),
    load_5m: Load5m,
    get_last_ts: GetLastTs,
    P: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """
    Pure batch builder.
    - load_5m(symbol, kind) -> pd.DataFrame (5m OHLCV)
    - get_last_ts(symbol, kind, tf, metric) -> Optional[pd.Timestamp]
    Returns:
      {"rows": List[tuple], "attempted": int, "frames": int}
    """
    P = P or load_cfg()
    all_rows: List[tuple] = []
    attempted = frames = 0

    for s in symbols:
        for k in kinds:
            df5 = load_5m(s, k)
            df5 = sanitize_ohlcv(df5)
            _dbg_head(f"load_5m {k}:{s}", df5)
            rows, st = build_indicator_rows(df5, symbol=s, kind=k, P=P, get_last_ts=get_last_ts)
            all_rows.extend(rows)
            attempted += st.get("attempted", 0)
            frames    += st.get("frames", 0)

    print(f"[WRITE] classic batch → tried {attempted} rows across {frames} TF frames")
    return {"rows": all_rows, "attempted": attempted, "frames": frames}

def run(
    *,
    symbols: Optional[List[str]] = None,
    kinds: Tuple[str, ...] = ("spot","futures"),
    load_5m: Optional[Load5m] = None,
    get_last_ts: Optional[GetLastTs] = None,
    P: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """
    Compatibility alias. You MUST provide load_5m and get_last_ts when
    calling this module directly, since this file is DB-free by design.
    """
    if symbols is None:
        symbols = []
    if load_5m is None or get_last_ts is None:
        raise ValueError("run(...) requires load_5m and get_last_ts callbacks (DB-free module).")
    return update_indicators_multi_tf(symbols=symbols, kinds=kinds, load_5m=load_5m, get_last_ts=get_last_ts, P=P)
