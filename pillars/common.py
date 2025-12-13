# pillars/common.py
from __future__ import annotations
import os, math, json, configparser
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import psycopg2.extras as pgx

from utils.db import get_db_connection

# ---------- Core config ----------
TZ = timezone.utc
DEFAULT_INI = os.getenv("PILLARS_V2_INI", "pillars_v2.ini")

TF_TO_OFFSET = {
    "5m":   "5min",
    "15m":  "15min",
    "25m":  "25min",
    "30m":  "30min",
    "60m":  "60min",   # 🔹 add
    "65m":  "65min",
    "120m": "120min",  # 🔹 add
    "125m": "125min",
    "240m": "240min",
}


# Sane defaults so higher TFs can actually run
_DEFAULT_MIN_BARS: Dict[str,int] = {
    "5m":  120,  # 2 sessions
    "15m": 100,
    "25m": 60,
    "30m": 60,
    "60m":  20,   # 🔹 add
    "65m": 40,   # ~1.5–2 weeks
    "120m": 15,   # 🔹 add
    "125m": 30,  # ~2 weeks
    "240m": 12,  # ~3 weeks
    "250m": 24,  # ~3–4 weeks
}

@dataclass
class BaseCfg:
    tfs: List[str]
    lookback_days: int
    run_id: str
    source: str

def _as_list_csv(x:str)->List[str]:
    return [s.strip() for s in (x or "").split(",") if s.strip()]

def load_base_cfg(path: str = DEFAULT_INI) -> BaseCfg:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cp.read(path)
    tfs = _as_list_csv(cp.get("core","tfs",fallback="25m,65m,125m"))
    return BaseCfg(
        tfs=tfs,
        lookback_days=int(cp.get("core","lookback_days",fallback="120")),
        run_id=os.getenv("RUN_ID","pillars_v2_run"),
        source=os.getenv("SRC","pillars_v2")
    )

# ---------- min_bars: INI + env merge ----------
def _parse_min_bars_ini(s: str) -> Dict[str,int]:
    """
    Parse '25m:60, 65m:40, 125m:30' into {'25m':60, '65m':40, ...}
    """
    out: Dict[str,int] = {}
    for part in _as_list_csv(s):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[k.strip()] = int(float(v.strip()))
            except:
                pass
    return out

def get_min_bars_map(ini_path: str = DEFAULT_INI) -> Dict[str,int]:
    # start with defaults
    m = dict(_DEFAULT_MIN_BARS)
    # merge INI
    cp = configparser.ConfigParser(inline_comment_prefixes=(";","#"), interpolation=None, strict=False)
    cp.read(ini_path)
    ini_str = cp.get("core", "min_bars", fallback="").strip()
    if ini_str:
        m.update(_parse_min_bars_ini(ini_str))
    # optional env override: PILLARS_MIN_BARS_JSON='{"25m":50,"65m":30,...}'
    try:
        env_json = os.getenv("PILLARS_MIN_BARS_JSON")
        if env_json:
            m.update(json.loads(env_json))
    except Exception:
        pass
    return m

def load_tf_weights(pillar: str, ini_path: str = DEFAULT_INI) -> Tuple[List[str], Dict[str, float]]:
    """
    Load timeframe weights for a given pillar from INI.

    Expected shape in pillars_v2.ini:

        [weights.flow]
        tfs = 15m,30m,60m,120m,240m
        tf_weights = 15m:0.30, 30m:0.30, 60m:0.20, 120m:0.10, 240m:0.10
        mode = normalize   ; normalize | raw

    Fallbacks:
        - if section missing → use [core].tfs with equal weights
        - if tf_weights missing/empty → equal weights across tfs
        - if mode=normalize → we normalize to sum to 1.0
    """
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(ini_path)

    sec = f"weights.{pillar}"

    # 1) Determine tfs list
    if cp.has_section(sec):
        tfs_str = cp.get(sec, "tfs", fallback="").strip()
    else:
        tfs_str = ""

    if not tfs_str:
        # Fallback to global core.tfs
        tfs_str = cp.get("core", "tfs", fallback="15m,30m,60m")

    tfs = _as_list_csv(tfs_str)
    if not tfs:
        # Extreme fallback: something sane
        tfs = ["15m", "30m", "60m"]

    # 2) Load raw weights, if any
    raw_weights_str = ""
    if cp.has_section(sec):
        raw_weights_str = cp.get(sec, "tf_weights", fallback="").strip()

    raw = _parse_weights(raw_weights_str) if raw_weights_str else {}

    # If no explicit weights → equal weights
    if not raw:
        raw = {tf: 1.0 for tf in tfs}

    # Ensure all tfs exist in weight map
    for tf in tfs:
        raw.setdefault(tf, 0.0)

    mode = "normalize"
    if cp.has_section(sec):
        mode = cp.get(sec, "mode", fallback="normalize").strip().lower()

    # 3) Normalize or leave as raw
    if mode == "normalize":
        total = sum(raw.get(tf, 0.0) for tf in tfs)
        if total <= 0:
            # fallback to equal if something is weird
            w = {tf: 1.0 / len(tfs) for tf in tfs}
        else:
            w = {tf: raw.get(tf, 0.0) / total for tf in tfs}
    else:
        # raw mode: take as-is
        w = {tf: float(raw.get(tf, 0.0)) for tf in tfs}

    return tfs, w

def min_bars_for_tf(tf: str) -> int:
    """
    Very relaxed thresholds so we can run 60m / 120m even with limited history.
    Tune later once full backfill is done.
    """
    tf = tf.lower()
    mapping = {
        "5m":   120,   # ~2 days of 5m
        "15m":   60,   # ~3 days
        "30m":   40,   # ~3–4 days
        "60m":   20,   # super relaxed; you have >200 anyway
        "120m":  15,   # relaxed; you have >100 anyway
    }
    # default to something small if unknown TF
    return mapping.get(tf, 20)

def aggregate_tf_scores(
    pillar: str,
    per_tf_scores: Dict[str, Optional[float]],
    ini_path: str = DEFAULT_INI
) -> Optional[float]:
    """
    Aggregate multi-TF pillar scores into a single number.

    - Reads TF list + weights from [weights.<pillar>]
    - Filters out TFs with None scores
    - Re-normalizes weights over available TFs
    - Returns weighted average score or None if nothing is available
    """
    tfs, base_weights = load_tf_weights(pillar, ini_path=ini_path)

    # 1) Keep only TFs that both:
    #    - are in configured tfs list
    #    - have a non-None score
    available: Dict[str, float] = {}
    for tf, val in per_tf_scores.items():
        if tf not in tfs:
            continue
        if val is None:
            continue
        try:
            available[tf] = float(val)
        except Exception:
            continue

    if not available:
        return None

    # 2) Build weight subset for available TFs
    w_sub: Dict[str, float] = {tf: float(base_weights.get(tf, 0.0)) for tf in available.keys()}
    total_w = sum(w_sub.values())

    # 3) If all weights zero → equal weights across available TFs
    if total_w <= 0:
        n = len(available)
        w_sub = {tf: 1.0 / n for tf in available.keys()}
    else:
        w_sub = {tf: w / total_w for tf, w in w_sub.items()}

    # 4) Weighted average
    agg = 0.0
    for tf, val in available.items():
        agg += val * w_sub.get(tf, 0.0)

    return float(agg)

def ensure_min_bars(df, tf: str) -> bool:
    """
    Returns True if we have enough candles to do any meaningful calc.
    """
    need = min_bars_for_tf(tf)
    have = len(df)
    if have < need:
        print(f"[FLOW] only {have} bars for tf={tf}, need {need} → skipping")
        return False
    return True

def maybe_trim_last_bar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the last row if it's clearly non-tradable (zero volume OR OHLC all equal).
    Helps avoid NSE 'dead' placeholder bars.
    """
    if df is None or df.empty or len(df) < 2:
        return df
    last = df.iloc[-1]
    vol_zero = ("volume" in df.columns) and (float(last["volume"]) == 0.0)
    flat_ohlc = all(k in df.columns for k in ("open","high","low","close")) and \
                (float(last["open"]) == float(last["high"]) == float(last["low"]) == float(last["close"]))
    return df.iloc[:-1] if (vol_zero or flat_ohlc) else df

def aggregate_flow_veto(per_tf_veto: Dict[str, bool]) -> bool:
    """
    Flow-specific veto aggregation.

    Rule:
      - If any HTF (>=60m) is veto=True → final veto=True
      - Else if *all* LTFs (15m,30m) are veto=True → final veto=True
      - Else → final veto=False

    This matches the idea:
      - HTF regime has the authority to block trades
      - Multiple LTF red flags together can also block
    """
    # Normalize keys
    veto_map = {str(tf).lower(): bool(v) for tf, v in per_tf_veto.items()}

    # 1) HTF dominance: 60m, 120m, 240m
    htf_set = {"60m", "120m", "240m"}
    for tf in htf_set:
        if veto_map.get(tf, False):
            return True

    # 2) LTF joint veto: if both 15m and 30m exist and both veto → block
    ltf_keys = ["15m", "30m"]
    ltf_present = [tf for tf in ltf_keys if tf in veto_map]
    if ltf_present and all(veto_map[tf] for tf in ltf_present):
        return True

    # 3) Otherwise → no composite veto
    return False

# ---------- DB helpers ----------
def exec_values(sql: str, rows: List[tuple]) -> int:
    if not rows: return 0
    with get_db_connection() as conn, conn.cursor() as cur:
        pgx.execute_values(cur, sql, rows, page_size=1000)
        conn.commit()
        return len(rows)

def last_metric(symbol: str, kind: str, tf: str, metric: str) -> Optional[float]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT val FROM indicators.values
             WHERE symbol=%s AND market_type=%s AND interval=%s AND metric=%s
             ORDER BY ts DESC LIMIT 1
        """,(symbol,kind,tf,metric))
        r = cur.fetchone()
    return float(r[0]) if r else None

def last_metrics_batch(symbol: str, kind: str, tf_list: List[str], metrics: List[str]) -> Dict[Tuple[str,str], Optional[float]]:
    if not tf_list or not metrics: return {}
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (interval, metric) interval, metric, val
              FROM indicators.values
             WHERE symbol=%s AND market_type=%s
               AND interval = ANY(%s)
               AND metric   = ANY(%s)
             ORDER BY interval, metric, ts DESC
        """, (symbol, kind, tf_list, metrics))
        rows = cur.fetchall()
    out: Dict[Tuple[str,str], Optional[float]] = {}
    for tf, m, v in rows or []:
        out[(tf, m)] = float(v) if v is not None else None
    for tf in tf_list:
        for m in metrics:
            out.setdefault((tf,m), None)
    return out

def write_values(rows):
    if not rows:
        return

    # Dedupe to avoid Postgres "ON CONFLICT cannot affect row a second time"
    # Unique key assumed: (symbol, kind, tf, ts, metric)
    dedup = {}
    for r in rows:
        # r = (symbol, kind, tf, ts, metric, value, ctx_json, run_id, source)
        key = (r[0], r[1], r[2], r[3], r[4])
        dedup[key] = r  # last write wins

    rows2 = list(dedup.values())

    # Optional debug (leave ON for now)
    if len(rows2) != len(rows):
        print(f"[write_values] deduped {len(rows)} -> {len(rows2)} rows")

    # ---- existing insert/upsert logic uses rows2 ----
    rows = rows2

    # ... keep your existing DB execute_values / INSERT ON CONFLICT code below ...


# ---------- Candle IO ----------
def load_5m(symbol: str, kind: str, lookback_days: int) -> pd.DataFrame:
    table = "market.futures_candles" if kind=="futures" else "market.spot_candles"
    cutoff = datetime.now(TZ) - timedelta(days=lookback_days)
    cols = "ts, open, high, low, close, volume"
    if kind=="futures": cols += ", COALESCE(oi,NULL) AS oi"
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT {cols}
              FROM {table}
             WHERE symbol=%s AND interval='5m' AND ts >= %s
             ORDER BY ts ASC
        """,(symbol,cutoff))
        rows = cur.fetchall()
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]+(["oi"] if kind=="futures" else []))
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    for c in ["open","high","low","close","volume"] + (["oi"] if "oi" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close","volume"])

def resample(df5: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df5.empty:
        return df5
    if tf == "15m":   # <-- IMPORTANT: your base is 15m here
        return df5.copy()

    rule = TF_TO_OFFSET.get(tf)
    if not rule:
        return pd.DataFrame()

    # expected bars per bucket (base=15m)
    TF_TO_MIN = {"15m": 15, "30m": 30, "60m": 60, "120m": 120, "240m": 240}
    base_min = 15
    tf_min = TF_TO_MIN.get(tf)
    expected = int(tf_min / base_min) if tf_min else None

    rs = df5.resample(rule, label="right", closed="right")

    out = pd.DataFrame({
        "open":   rs["open"].first(),
        "high":   rs["high"].max(),
        "low":    rs["low"].min(),
        "close":  rs["close"].last(),
        "volume": rs["volume"].sum(),
    })

    if "oi" in df5.columns:
        out["oi"] = rs["oi"].last()

    # ✅ DROP partial last bucket (and any incomplete bucket)
    if expected:
        cnt = rs["close"].count()
        out = out[cnt >= expected]     # strict; change to >= expected*0.9 if you want tolerant

    out = out.dropna(how="any")
    return out

def prepare_tf_frame(df5: pd.DataFrame, tf: str, ini_path: str = DEFAULT_INI) -> pd.DataFrame:
    """
    One true prep: resample → trim dead last bar → enforce min_bars.
    Return empty DataFrame if not enough bars.
    """
    dftf = resample(df5, tf)
    if not ensure_min_bars(dftf, tf):
        return pd.DataFrame()
    return dftf


# ---------- TA utils ----------
def ema(s: pd.Series, n:int)->pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def zscore(s: pd.Series, win:int)->pd.Series:
    m = s.rolling(win, min_periods=max(5,win//2)).mean()
    sd= s.rolling(win, min_periods=max(5,win//2)).std(ddof=1).replace(0,np.nan)
    return (s - m)/sd

def true_range(h,l,c):
    prev=c.shift(1)
    tr = pd.concat([(h-l).abs(),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    return tr

def atr(h,l,c,n:int=14)->pd.Series:
    return true_range(h,l,c).ewm(alpha=1.0/n, adjust=False).mean()

def adx(h,l,c,n:int=14):
    up=h.diff(); dn=-l.diff()
    plus_dm = ((up>dn)&(up>0))*up
    minus_dm= ((dn>up)&(dn>0))*dn
    tr = true_range(h,l,c).ewm(alpha=1.0/n, adjust=False).mean()
    plus_di  = 100*(plus_dm.ewm(alpha=1.0/n,adjust=False).mean() / tr.replace(0,np.nan))
    minus_di = 100*(minus_dm.ewm(alpha=1.0/n,adjust=False).mean()/ tr.replace(0,np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,np.nan))*100
    return dx.ewm(alpha=1.0/n,adjust=False).mean(), plus_di, minus_di

def bb_width_pct(close,n=20,k=2.0):
    ma=close.rolling(n, min_periods=max(5,n//2)).mean()
    sd=close.rolling(n, min_periods=max(5,n//2)).std(ddof=1)
    upper=ma+k*sd; lower=ma-k*sd
    return 100.0*((upper-lower)/ma.replace(0,np.nan).abs())

def obv_series(close, volume):
    sign = np.sign(close.diff().fillna(0.0))
    return (sign*volume).cumsum()

# ---------- Small utils ----------
def clamp(x,a,b): return max(a, min(b, x))
def now_ts(): return datetime.now(TZ)

def _parse_weights(s: str) -> Dict[str, float]:
    """
    Parse '15m:0.3, 30m:0.3, 60m:0.2' into {'15m':0.3, '30m':0.3, '60m':0.2}.
    No normalization here – caller decides.
    """
    out: Dict[str, float] = {}
    for part in (s or "").split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            continue
        try:
            out[k] = float(v)
        except Exception:
            # ignore bad pieces
            continue
    return out


def weighted_avg(values: Dict[str, float], weights: Dict[str, float]) -> float:
    s = w = 0.0
    for tf, val in values.items():
        if val is None: 
            continue
        wt = float(weights.get(tf, 0.0))
        s += wt * float(val); w += wt
    return float(s / w) if w > 0 else float(np.mean([float(v) for v in values.values() if v is not None])) if values else 0.0

# ---------- Session + ranks ----------
def is_ist_session(ts: pd.Timestamp) -> bool:
    try:
        ts_ist = ts.tz_convert("Asia/Kolkata")
    except Exception:
        ts_ist = ts.tz_localize("UTC").tz_convert("Asia/Kolkata")
    minutes = ts_ist.hour * 60 + ts_ist.minute
    open_min, close_min = 9*60 + 15, 15*60 + 30
    return (open_min <= minutes <= close_min)

def last_bar_rank_pct(s: pd.Series, window:int=60) -> float:
    try:
        if s is None or len(s) < 2: return 0.5
        w = s.tail(window)
        if len(w) < 3: return 0.5
        return float(w.rank(pct=True).iloc[-1])
    except Exception:
        return 0.5
