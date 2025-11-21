# utils/indicators.py
# Purpose: Config-driven indicators + DB upserts (spot_indicators / futures_indicators)

from __future__ import annotations

import configparser
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----- Optional TA helpers (robust fallbacks) -----
try:
    from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, ROCIndicator
    from ta.volume import MFIIndicator
    from ta.volatility import AverageTrueRange
except Exception:
    EMAIndicator = SMAIndicator = MACD = ADXIndicator = None
    RSIIndicator = ROCIndicator = None
    MFIIndicator = None
    AverageTrueRange = None

# ----- DB connection -----
try:
    from utils.db import get_db_connection
except Exception as e:
    raise RuntimeError("utils.db.get_db_connection import failed. Ensure utils/db.py exists.") from e

DEFAULT_INI_PATH = "indicators.ini"

# =========================
# Config parsing
# =========================

def _parse_bool(x: str, default: bool = True) -> bool:
    if x is None:
        return default
    return str(x).strip().lower() in {"1", "true", "yes", "on"}

def _parse_int_list(x: str, default: List[int]) -> List[int]:
    try:
        parts = [p.strip() for p in str(x).split(",") if p.strip()]
        vals = [int(p) for p in parts]
        return vals or default
    except Exception:
        return default

def _parse_int(x: str, default: int) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default

def _parse_float(x: str, default: float) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return default

def _parse_macd(x: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    try:
        parts = [int(p.strip()) for p in str(x).split(",")]
        if len(parts) == 3:
            return tuple(parts)  # (fast, slow, signal)
        return default
    except Exception:
        return default

def _parse_rmi(x: str, default=(14, 5)) -> Tuple[int, int]:
    try:
        a, b = [int(p.strip()) for p in str(x).split(",")]
        return a, b
    except Exception:
        return default

@dataclass
class IndicatorConfig:
    enabled: Dict[str, bool]
    timeframes: List[str]                 # ["25","65","125","250"] etc (minutes)
    periods: Dict[str, object]            # rsi:int, rmi:(len,mom), ema:list[int], ...
    tf_overrides: Dict[str, List[str]]    # per-indicator TFs
    weights: Dict[str, float]             # for final confidence

def load_indicator_config(ini_path: str = DEFAULT_INI_PATH) -> IndicatorConfig:
    cfg = configparser.ConfigParser()
    cfg.read(ini_path)

    enabled = {
        "rsi": _parse_bool(cfg.get("indicators", "rsi", fallback="on")),
        "rmi": _parse_bool(cfg.get("indicators", "rmi", fallback="off")),
        "ema": _parse_bool(cfg.get("indicators", "ema", fallback="on")),
        "macd": _parse_bool(cfg.get("indicators", "macd", fallback="on")),
        "adx": _parse_bool(cfg.get("indicators", "adx", fallback="on")),
        "roc": _parse_bool(cfg.get("indicators", "roc", fallback="on")),
        "atr": _parse_bool(cfg.get("indicators", "atr", fallback="on")),
        "mfi": _parse_bool(cfg.get("indicators", "mfi", fallback="on")),
        "sma": _parse_bool(cfg.get("indicators", "sma", fallback="off")),
    }

    tflist = cfg.get("timeframes", "list", fallback="25,65,125,250")
    timeframes = [t.strip() for t in tflist.split(",") if t.strip()]

    periods = {
        "rsi": _parse_int(cfg.get("periods", "rsi", fallback="14"), 14),
        "rmi": _parse_rmi(cfg.get("periods", "rmi", fallback="14,5")),
        "ema": _parse_int_list(cfg.get("periods", "ema", fallback="5,10,20,50"), [5,10,20,50]),
        "macd": _parse_macd(cfg.get("periods", "macd", fallback="12,26,9"), (12, 26, 9)),
        "adx": _parse_int(cfg.get("periods", "adx", fallback="14"), 14),
        "roc": _parse_int(cfg.get("periods", "roc", fallback="14"), 14),
        "atr": _parse_int(cfg.get("periods", "atr", fallback="14"), 14),
        "mfi": _parse_int(cfg.get("periods", "mfi", fallback="14"), 14),
        "sma": _parse_int_list(cfg.get("periods", "sma", fallback="5,10,20,50"), [5,10,20,50]),
    }

    tf_overrides: Dict[str, List[str]] = {}
    if cfg.has_section("timeframes_by_indicator"):
        for k, v in cfg.items("timeframes_by_indicator"):
            tf_overrides[k.strip()] = [t.strip() for t in v.split(",") if t.strip()]

    weights: Dict[str, float] = {}
    if cfg.has_section("weights"):
        for k, v in cfg.items("weights"):
            weights[k.strip()] = _parse_float(v, 0.0)

    return IndicatorConfig(
        enabled=enabled,
        timeframes=timeframes,
        periods=periods,
        tf_overrides=tf_overrides,
        weights=weights,
    )

# =========================
# Helpers (math + DF)
# =========================

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def _safe(x, default=0.0):
    try:
        v = float(x)
        return default if not np.isfinite(v) else v
    except Exception:
        return default

def _last(series, default=0.0):
    try:
        return _safe(series.iloc[-1], default)
    except Exception:
        return default

def _normalize_ohlcv(src: pd.DataFrame) -> pd.DataFrame:
    df = src.copy()
    out = pd.DataFrame(index=df.index)

    def pick(cols):
        for c in cols:
            if c in df.columns:
                return _num(df[c])
        return pd.Series(np.nan, index=df.index)

    out["open"]   = pick(["open", "open_price"])
    out["high"]   = pick(["high", "high_price"])
    out["low"]    = pick(["low",  "low_price"])
    out["close"]  = pick(["close","close_price"])
    out["volume"] = pick(["volume"]).fillna(0.0)

    if len(out) > 0:
        last = out.iloc[-1]
        for c in ("open","high","low"):
            if pd.isna(last[c]):
                out.loc[out.index[-1], c] = _safe(last["close"], 0.0)

    return out

# OBV for spot table if requested
def _obv(close: pd.Series, volume: pd.Series) -> float:
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    return _safe(obv.iloc[-1])

# =========================
# Core calculator (single TF snapshot)
# =========================

def _rsi_series(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=length, min_periods=1).mean()
    avg_loss = loss.rolling(window=length, min_periods=1).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _rmi_series(series: pd.Series, length: int, momentum: int) -> pd.Series:
    mom = series - series.shift(momentum)
    return _rsi_series(mom, length)

def compute_indicators(df: pd.DataFrame, cfg: Optional[IndicatorConfig] = None) -> dict:
    """
    Compute indicator snapshot for a single timeframe.
    Returns keys only for enabled indicators (names like rsi, ema_20, sma_50, macd, macd_signal, macd_diff, mfi_14, adx, plus_di, minus_di, roc, atr_14, rmi).
    """
    if cfg is None:
        cfg = load_indicator_config()

    res: Dict[str, float] = {}

    try:
        d = _normalize_ohlcv(df)
        close, high, low, vol = d["close"], d["high"], d["low"], d["volume"]

        # ----- EMA (multi)
        if cfg.enabled.get("ema", False):
            ema_periods: List[int] = cfg.periods.get("ema", [5,10,20,50])  # type: ignore
            for p in ema_periods:
                try:
                    if EMAIndicator is not None:
                        v = EMAIndicator(close, window=int(p)).ema_indicator().iloc[-1]
                        if pd.isna(v):
                            v = close.ewm(span=int(p), adjust=False).mean().iloc[-1]
                    else:
                        v = close.ewm(span=int(p), adjust=False).mean().iloc[-1]
                except Exception:
                    v = close.ewm(span=int(p), adjust=False).mean().iloc[-1]
                res[f"ema_{int(p)}"] = _safe(v)
            if ema_periods:
                res["ema"] = res.get(f"ema_{int(ema_periods[0])}", _last(close))

        # ----- SMA (optional)
        if cfg.enabled.get("sma", False):
            sma_periods: List[int] = cfg.periods.get("sma", [5,10,20,50])  # type: ignore
            for p in sma_periods:
                try:
                    if SMAIndicator is not None:
                        v = SMAIndicator(close, window=int(p)).sma_indicator().iloc[-1]
                        if pd.isna(v):
                            v = close.rolling(window=int(p), min_periods=1).mean().iloc[-1]
                    else:
                        v = close.rolling(window=int(p), min_periods=1).mean().iloc[-1]
                except Exception:
                    v = close.rolling(window=int(p), min_periods=1).mean().iloc[-1]
                res[f"sma_{int(p)}"] = _safe(v)
            if sma_periods:
                res["sma"] = res.get(f"sma_{int(sma_periods[0])}", _last(close))

        # ----- RSI
        if cfg.enabled.get("rsi", False):
            rsi_period: int = int(cfg.periods.get("rsi", 14))  # type: ignore
            try:
                if RSIIndicator is not None:
                    rsi = RSIIndicator(close=close, window=rsi_period).rsi().iloc[-1]
                else:
                    raise RuntimeError()
            except Exception:
                rsi = _last(_rsi_series(close, rsi_period))
            res["rsi"] = _safe(rsi)

        # ----- RMI
        if cfg.enabled.get("rmi", False):
            rmi_len, rmi_mom = cfg.periods.get("rmi", (14, 5))  # type: ignore
            try:
                rmi_val = _rmi_series(close, int(rmi_len), int(rmi_mom)).iloc[-1]
            except Exception:
                rmi_val = 50.0
            res["rmi"] = _safe(rmi_val)

        # ----- ROC
        if cfg.enabled.get("roc", False):
            roc_period: int = int(cfg.periods.get("roc", 14))  # type: ignore
            try:
                if ROCIndicator is not None:
                    roc = ROCIndicator(close=close, window=roc_period).roc().iloc[-1]
                else:
                    raise RuntimeError()
            except Exception:
                roc = (close.pct_change(roc_period) * 100).iloc[-1] if len(close) > 1 else 0.0
            res["roc"] = _safe(roc)

        # ----- MFI
        if cfg.enabled.get("mfi", False):
            mfi_period: int = int(cfg.periods.get("mfi", 14))  # type: ignore
            try:
                if MFIIndicator is not None:
                    mfi = MFIIndicator(high=high, low=low, close=close, volume=vol, window=mfi_period)\
                          .money_flow_index().iloc[-1]
                else:
                    raise RuntimeError()
            except Exception:
                tp = (high + low + close) / 3.0
                rmf = tp * vol
                pos = rmf.where(tp > tp.shift(), 0.0)
                neg = rmf.where(tp < tp.shift(), 0.0)
                mfr = (pos.rolling(mfi_period).sum()) / (neg.rolling(mfi_period).sum().replace(0, np.nan))
                mfi = (100 - (100 / (1 + mfr))).iloc[-1]
            res[f"mfi_{mfi_period}"] = _safe(mfi)

        # ----- MACD
        if cfg.enabled.get("macd", False):
            fast, slow, signal = cfg.periods.get("macd", (12,26,9))  # type: ignore
            try:
                if MACD is not None:
                    macd_obj = MACD(close, window_fast=int(fast), window_slow=int(slow), window_sign=int(signal))
                    res["macd"] = _safe(macd_obj.macd().iloc[-1])
                    res["macd_signal"] = _safe(macd_obj.macd_signal().iloc[-1])
                    res["macd_diff"] = _safe(macd_obj.macd_diff().iloc[-1])
                else:
                    raise RuntimeError()
            except Exception:
                ema_fast = close.ewm(span=int(fast), adjust=False).mean()
                ema_slow = close.ewm(span=int(slow), adjust=False).mean()
                macd = ema_fast - ema_slow
                sig  = macd.ewm(span=int(signal), adjust=False).mean()
                res["macd"] = _last(macd)
                res["macd_signal"] = _last(sig)
                res["macd_diff"] = _last(macd - sig)

        # ----- ATR
        if cfg.enabled.get("atr", False):
            atr_period: int = int(cfg.periods.get("atr", 14))  # type: ignore
            try:
                if AverageTrueRange is not None:
                    atr = AverageTrueRange(high=high, low=low, close=close, window=atr_period)\
                          .average_true_range().iloc[-1]
                else:
                    raise RuntimeError()
            except Exception:
                tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(atr_period, min_periods=1).mean().iloc[-1]
            res[f"atr_{atr_period}"] = _safe(atr)

        # ----- ADX / DI
        if cfg.enabled.get("adx", False):
            adx_period: int = int(cfg.periods.get("adx", 14))  # type: ignore
            try:
                if ADXIndicator is not None:
                    adx_obj = ADXIndicator(high=high, low=low, close=close, window=adx_period)
                    res["adx"] = _safe(adx_obj.adx().iloc[-1])
                    res["plus_di"] = _safe(adx_obj.adx_pos().iloc[-1])
                    res["minus_di"] = _safe(adx_obj.adx_neg().iloc[-1])
                else:
                    raise RuntimeError()
            except Exception:
                tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(adx_period, min_periods=1).mean().replace(0, np.nan)
                plus_dm = (high.diff()).clip(lower=0)
                minus_dm = (-low.diff()).clip(lower=0)
                plus_di = 100 * (plus_dm.rolling(adx_period).sum() / atr)
                minus_di = 100 * (minus_dm.rolling(adx_period).sum() / atr)
                dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
                adx = dx.rolling(adx_period, min_periods=1).mean()
                res["adx"] = _last(adx.fillna(0))
                res["plus_di"] = _last(plus_di.fillna(0))
                res["minus_di"] = _last(minus_di.fillna(0))

    except Exception as e:
        print(f"[❌ indicators.py] Error computing indicators: {e}")
        return {}

    return res

# =========================
# DB I/O + Upserts
# =========================

def _table_columns(table: str, schema: str = "public") -> List[str]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
              FROM information_schema.columns
             WHERE table_schema=%s AND table_name=%s
            """,
            (schema, table)
        )
        return [r[0] for r in cur.fetchall()]

def _fetch_ohlcv_from_market(symbol: str, interval: str, kind: str) -> pd.DataFrame:
    """
    kind: 'spot' | 'futures'
    Reads the last ~300 bars for indicator calc.
    """
    table = "market.spot_candles" if kind == "spot" else "market.futures_candles"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ts, open_price, high_price, low_price, close_price, volume
              FROM {table}
             WHERE symbol=%s AND interval=%s
             ORDER BY ts DESC
             LIMIT 300
            """,
            (symbol, interval)
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","open_price","high_price","low_price","close_price","volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.sort_values("ts", inplace=True)
    df.set_index("ts", inplace=True)
    return df

def _upsert_indicators(table: str, symbol: str, interval: str, ts: pd.Timestamp, values: Dict[str, float], schema: str = "public") -> None:
    """
    Upsert one row into public.{table} with dynamic column subset.
    Requires a unique key on (symbol, interval, ts).
    """
    cols_available = set(_table_columns(table, schema))
    # Always include identity columns
    base_cols = ["symbol", "interval", "ts"]
    payload = {k: v for k, v in values.items() if k in cols_available and k not in base_cols}

    if not payload:
        # Nothing to write (no overlap); still ensure row exists
        payload = {}

    cols = base_cols + list(payload.keys())
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join([f"{c}=EXCLUDED.{c}" for c in payload.keys()])  # skip base cols

    sql = f"""
        INSERT INTO {schema}.{table} ({', '.join(cols)})
        VALUES ({placeholders})
        ON CONFLICT (symbol, interval, ts)
        DO UPDATE SET {updates if updates else 'ts=EXCLUDED.ts'};
    """

    vals = [symbol, interval, ts.to_pydatetime()] + [payload[c] for c in payload.keys()]
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, vals)
        conn.commit()

# Public: compute & write a snapshot for one symbol/kind/interval
def update_indicators_for_symbol(symbol: str, interval: str = "5m", kind: str = "both") -> bool:
    """
    kind: 'spot' | 'futures' | 'both'
    Pulls OHLCV from market.*_candles, computes snapshot, and upserts into
    public.spot_indicators / public.futures_indicators for the latest bar time.
    Returns True if something was written.
    """
    wrote = False
    cfg = load_indicator_config()

    for k in (["spot","futures"] if kind == "both" else [kind]):
        df = _fetch_ohlcv_from_market(symbol, interval, kind=k)
        if df.empty:
            continue

        ind = compute_indicators(df, cfg)
        # Compute optional OBV if the target table has 'obv'
        target_table = "spot_indicators" if k == "spot" else "futures_indicators"
        cols = set(_table_columns(target_table))
        if "obv" in cols and "obv" not in ind:
            ind["obv"] = _obv(df["close_price"], df["volume"])

        # Provide legacy alias keys if tables expect them
        # mfi_14 already emitted; rmi may be exposed as rmi_14 in some schemas
        if "rmi_14" in cols and "rmi" in ind:
            ind["rmi_14"] = ind["rmi"]

        last_ts = df.index[-1]
        _upsert_indicators(target_table, symbol, interval, last_ts, ind)
        wrote = True

    return wrote

# =========================
# Multi-timeframe runners (kept for completeness)
# =========================

def compute_for_timeframes(dfs_by_timeframe: Dict[str, pd.DataFrame], ini_path: str = DEFAULT_INI_PATH) -> Dict[str, Dict[str, float]]:
    cfg = load_indicator_config(ini_path)
    out: Dict[str, Dict[str, float]] = {}
    allowed = {str(x) for x in cfg.timeframes}
    for tf, df in dfs_by_timeframe.items():
        if str(tf) not in allowed:
            continue
        out[str(tf)] = compute_indicators(df, cfg)
    return out

def compute_weighted_scores(dfs_by_timeframe: Dict[str, pd.DataFrame], ini_path: str = DEFAULT_INI_PATH) -> Dict[str, object]:
    cfg = load_indicator_config(ini_path)
    out: Dict[str, object] = {}

    allowed_tfs = {str(x) for x in cfg.timeframes}
    dfs = {str(k): v for k, v in dfs_by_timeframe.items() if str(k) in allowed_tfs}

    def _collect(name: str) -> Dict[str, float]:
        vals: Dict[str, float] = {}
        tfs = cfg.tf_overrides.get(name, cfg.timeframes)
        for tf in [str(t) for t in tfs]:
            df = dfs.get(tf)
            if df is None or df.empty:
                continue
            ind = compute_indicators(df, cfg)
            if name == "rsi" and "rsi" in ind:
                vals[tf] = float(ind["rsi"])
            elif name == "rmi" and "rmi" in ind:
                vals[tf] = float(ind["rmi"])
        if vals:
            vals["score"] = round(sum(vals.values()) / len(vals), 3)
        return vals

    if cfg.enabled.get("rsi", False):
        out["rsi"] = _collect("rsi")
    if cfg.enabled.get("rmi", False):
        out["rmi"] = _collect("rmi")

    # Composite confidence (only weights >0 are included)
    num = den = 0.0
    for name in ("rsi", "rmi"):
        m = out.get(name, {})
        s = m.get("score") if isinstance(m, dict) else None
        w = cfg.weights.get(name, 0.0)
        if s is not None and w > 0:
            num += w * float(s)
            den += w
    out["score"] = round(num / den, 2) if den > 0 else None
    return out

# ---------------- Script utility (manual test) ----------------
if __name__ == "__main__":
    # Example quick test:
    # update_indicators_for_symbol("TECHM", interval="5m", kind="both")
    pass
