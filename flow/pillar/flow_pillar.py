from __future__ import annotations

import ast
import configparser
import functools
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.db import get_db_connection
from pillars.common import (
    obv_series,
    resample,  # keep existing common resample (but we use resample_nse for TF>15m)
    last_metric,
    write_values,
    clamp,
    TZ,
    BaseCfg,
    maybe_trim_last_bar,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
FLOW_DIR = Path(__file__).resolve().parent
DEFAULT_INI = FLOW_DIR / "flow_scenarios.ini"

print("🔥 FLOW_PILLAR DEFAULT_INI =", DEFAULT_INI)
print("🔥 Exists?", DEFAULT_INI.exists())


# ============================================================
# Config (CACHED)  ✅ speed win
# ============================================================

@functools.lru_cache(maxsize=8)
def _cfg_cached(path_str: str, mtime_ns: int) -> dict:
    """
    Cached config loader.
    Cache key includes file mtime so changes auto-refresh.
    """
    cp = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#"),
        interpolation=None,
        strict=False,
    )
    cp.read(path_str)

    spread_bps_veto = cp.getfloat(
        "flow", "spread_bps_veto",
        fallback=cp.getfloat("liquidity", "max_spread_bps", fallback=40.0),
    )
    min_turnover = cp.getfloat(
        "flow", "min_turnover",
        fallback=cp.getfloat("liquidity", "min_turnover", fallback=0.0),
    )

    scenarios_list = [
        s.strip()
        for s in cp.get("flow_scenarios", "list", fallback="")
                 .replace("\n", " ")
                 .split(",")
        if s.strip()
    ]

    return {
        "mfi_len": cp.getint("flow", "mfi_len", fallback=14),
        "rvol_strong": cp.getfloat("flow", "rvol_strong", fallback=1.5),
        "rvol_extreme": cp.getfloat("flow", "rvol_extreme", fallback=2.0),
        "vol_cv_good": cp.getfloat("flow", "vol_cv_good", fallback=0.50),
        "vol_cv_bad": cp.getfloat("flow", "vol_cv_bad", fallback=1.50),
        "voi_scale": cp.getfloat("flow", "voi_scale", fallback=10.0),
        "roll_look": cp.getint("flow", "roll_look", fallback=5),
        "roll_drop_pct": cp.getfloat("flow", "roll_drop_pct", fallback=0.35),
        "spread_bps_veto": spread_bps_veto,
        "min_rvol_veto": cp.getfloat("flow", "min_rvol_veto", fallback=0.50),
        "vol_rank_floor": cp.getfloat("flow", "vol_rank_floor", fallback=0.20),
        "min_turnover": min_turnover,

        "rules_mode": cp.get("flow", "rules_mode", fallback="additive").strip().lower(),
        "clamp_low": cp.getfloat("flow", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("flow", "clamp_high", fallback=100.0),
        "write_scenarios_debug": cp.getboolean("flow", "write_scenarios_debug", fallback=False),

        # ML (ini-driven)
        "cp": cp,
        "scenarios_list": scenarios_list,
    }


def _cfg(path: str | Path = DEFAULT_INI) -> dict:
    p = Path(path)
    mtime_ns = p.stat().st_mtime_ns if p.exists() else 0
    return _cfg_cached(str(p), mtime_ns)


# ============================================================
# NSE-aligned resample ✅ fixes half-bar TF issue
# ============================================================

def resample_nse(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample to NSE-aligned bars anchored at 09:15 IST.
    Returns UTC indexed output.
    """
    if tf == "15m" or df.empty:
        return df.copy()

    rule = tf.replace("m", "min")

    x = df.copy()
    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    x = x.tz_convert("Asia/Kolkata")

    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "oi" in x.columns:
        ohlc["oi"] = "last"

    out = (
        x.resample(
            rule,
            closed="right",
            label="right",
            origin="start_day",
            offset="9h15min",
        )
        .agg(ohlc)
        .dropna(subset=["open", "high", "low", "close"])
    )

    return out.tz_convert("UTC")


# ============================================================
# Flow-specific min bars (rules-only) ✅ relaxed
# ============================================================

def min_bars_for_tf_rules(tf: str) -> int:
    tf = str(tf).lower()
    if tf in ("15m", "30m"):
        return 2
    if tf in ("60m", "120m", "240m"):
        return 1
    return 1


# ============================================================
# Calibration helpers (Flow ML)
# ============================================================

@functools.lru_cache(maxsize=1)
def _load_flow_calibration() -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    sql = """
        SELECT bucket, p_min, p_max, realized_up_rate
          FROM indicators.flow_calibration_4h
         ORDER BY p_min
    """
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql)
            for bucket, p_min, p_max, realized_up_rate in cur.fetchall():
                if realized_up_rate is None:
                    continue
                rows.append(
                    {
                        "bucket": int(bucket),
                        "p_min": float(p_min),
                        "p_max": float(p_max),
                        "realized_up_rate": float(realized_up_rate),
                    }
                )
    except Exception:
        rows = []
    return rows


def _calibrate_prob(p: Optional[float]) -> float:
    if p is None:
        return 0.0
    try:
        p = float(p)
        if not np.isfinite(p):
            p = 0.0
    except Exception:
        p = 0.0
    p = max(0.0, min(1.0, p))

    buckets = _load_flow_calibration()
    if not buckets:
        return p

    for row in buckets:
        if row["p_min"] <= p < row["p_max"]:
            cal = row.get("realized_up_rate", p)
            if cal is None or not np.isfinite(cal):
                return p
            return max(0.0, min(1.0, float(cal)))

    # p==1.0 → last
    last = buckets[-1].get("realized_up_rate", p)
    if last is None or not np.isfinite(last):
        return p
    return max(0.0, min(1.0, float(last)))


def _find_prob_bucket(p: Optional[float]) -> Optional[int]:
    if p is None:
        return None
    try:
        p = float(p)
        if not np.isfinite(p):
            return None
    except Exception:
        return None

    p = max(0.0, min(1.0, p))
    buckets = _load_flow_calibration()
    if not buckets:
        return None

    for row in buckets:
        if row["p_min"] <= p < row["p_max"]:
            return int(row["bucket"])
    return int(buckets[-1]["bucket"])


# ============================================================
# Safe AST evaluator
# ============================================================

_ALLOWED_BOOL_OPS = {ast.And, ast.Or}
_ALLOWED_UNARY_BOOL = {ast.Not}
_ALLOWED_NUM_OPS = {
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.FloorDiv, ast.Pow, ast.USub, ast.UAdd
}

@functools.lru_cache(maxsize=2048)
def _compile_rule(expr: str) -> Optional[ast.AST]:
    if not expr or not expr.strip():
        return None
    try:
        return ast.parse(expr, mode="eval")
    except Exception:
        return None


def _safe_eval(expr_or_tree: str | ast.AST, scope: Dict[str, Any]) -> bool:
    if not expr_or_tree:
        return False

    tree = _compile_rule(expr_or_tree) if isinstance(expr_or_tree, str) else expr_or_tree
    if not tree:
        return False

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.BoolOp) and type(node.op) in _ALLOWED_BOOL_OPS:
            vals = [bool(_eval(v)) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)

        if isinstance(node, ast.UnaryOp):
            if type(node.op) in _ALLOWED_UNARY_BOOL:
                return not bool(_eval(node.operand))
            if type(node.op) in _ALLOWED_NUM_OPS:
                v = float(_eval(node.operand))
                return -v if isinstance(node.op, ast.USub) else +v

        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_NUM_OPS:
            a = float(_eval(node.left))
            b = float(_eval(node.right))
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b if b != 0 else 0.0
            if isinstance(node.op, ast.FloorDiv): return a // b if b != 0 else 0.0
            if isinstance(node.op, ast.Mod): return a % b if b != 0 else 0.0
            if isinstance(node.op, ast.Pow): return a ** b

        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            a = _eval(node.left)
            b = _eval(node.comparators[0])
            op = node.ops[0]
            try:
                if isinstance(op, ast.Lt): return a < b
                if isinstance(op, ast.LtE): return a <= b
                if isinstance(op, ast.Gt): return a > b
                if isinstance(op, ast.GtE): return a >= b
                if isinstance(op, ast.Eq): return a == b
                if isinstance(op, ast.NotEq): return a != b
            except Exception:
                return False

        if isinstance(node, ast.Name):
            return scope.get(node.id, False)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool, str)):
            return node.value

        raise ValueError("Disallowed token in scenario expression")

    try:
        return bool(_eval(tree))
    except Exception:
        return False


# ============================================================
# Helpers
# ============================================================

def _ist(ts: pd.Timestamp) -> pd.Timestamp:
    try:
        return ts.tz_convert("Asia/Kolkata")
    except Exception:
        return ts.tz_localize("UTC").tz_convert("Asia/Kolkata")


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, n: int = 14) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    vol = pd.to_numeric(vol, errors="coerce").fillna(0.0)

    tp = (high + low + close) / 3.0
    rmf = tp * vol
    diff = tp.diff()

    pos = pd.Series(np.where(diff > 0, rmf, 0.0), index=tp.index)
    neg = pd.Series(np.where(diff < 0, rmf, 0.0), index=tp.index)

    win = max(5, n // 2)
    pos_n = pos.rolling(n, min_periods=win).sum()
    neg_n = neg.rolling(n, min_periods=win).sum()

    mr = pos_n / neg_n.replace(0, np.nan)
    out = 100 - (100 / (1 + mr))
    return out.clip(0, 100).ffill().fillna(50.0)


def _clean_json_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for k, v in d.items():
        if isinstance(v, float) and not np.isfinite(v):
            clean[k] = None
        else:
            clean[k] = v
    return clean


# ============================================================
# Feature builder (kept compatible)
# ============================================================

def _build_features(
    dtf: pd.DataFrame,
    symbol: str,
    kind: str,
    tf: str,
    cfg: dict,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    ts_ist = _ist(dtf.index[-1])
    minutes = ts_ist.hour * 60 + ts_ist.minute
    open_min, close_min = 9 * 60 + 15, 15 * 60 + 30
    in_session = (open_min <= minutes <= close_min)

    last_vol_zero = (float(dtf["volume"].iloc[-1]) == 0.0)
    use = dtf.iloc[:-1] if ((not in_session) and last_vol_zero and len(dtf) > 1) else dtf

    o = use["open"]
    c = use["close"]
    h = use["high"]
    l = use["low"]
    v = use["volume"].fillna(0.0)

    mfi = _mfi(h, l, c, v, n=cfg["mfi_len"])
    mfi_val = float(mfi.iloc[-1]) if len(mfi) else 50.0
    mfi_slope = float(mfi.diff().iloc[-1]) if len(mfi) > 1 else 0.0

    obv = obv_series(c, v) if len(c) else pd.Series([], dtype=float)
    obv_ema = obv.ewm(span=20, adjust=False).mean() if len(obv) else obv
    obv_above_ema = bool(len(obv) and float(obv.iloc[-1]) > float(obv_ema.iloc[-1]))

    v_avg = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean() or 1.0)
    rvol_now = float(v.iloc[-1] / max(1e-9, v_avg))

    vol_cv20 = float((v.rolling(20).std(ddof=1) / v.rolling(20).mean()).iloc[-1] or 0.0) if len(v) >= 20 else 1.0
    vol_rank60 = float(v.tail(60).rank(pct=True).iloc[-1]) if len(v) >= 60 else 0.5

    # context-first VAH/VAL
    vah = context.get(f"VP.VAH|{tf}") if context else None
    val = context.get(f"VP.VAL|{tf}") if context else None
    if vah is None:
        vah = last_metric(symbol, kind, tf, "VP.VAH")
    if val is None:
        val = last_metric(symbol, kind, tf, "VP.VAL")

    # liquidity (context-first)
    spread_bps = None
    if context:
        for m in ("MKT.spread_bps", "LIQ.spread_bps", "MKT.spread_bps.mean"):
            k = f"{m}|{tf}"
            if k in context and context[k] is not None:
                spread_bps = float(context[k])
                break
    if spread_bps is None:
        for m in ("MKT.spread_bps", "LIQ.spread_bps", "MKT.spread_bps.mean"):
            x = last_metric(symbol, kind, tf, m)
            if x is not None:
                try:
                    spread_bps = float(x)
                    break
                except Exception:
                    pass

    close_last = float(c.iloc[-1])
    vol_last = float(v.iloc[-1] or 0.0)
    turnover_est = close_last * vol_last

    illiquid = False
    if (spread_bps is not None) and np.isfinite(spread_bps) and (spread_bps >= cfg["spread_bps_veto"]):
        illiquid = True
    elif in_session and (not last_vol_zero) and (rvol_now < cfg["min_rvol_veto"]) and (vol_rank60 <= cfg["vol_rank_floor"]):
        illiquid = True
    elif (cfg["min_turnover"] > 0) and in_session and (not last_vol_zero) and (turnover_est < cfg["min_turnover"]):
        illiquid = True

    # roll trap (OI drop) futures only
    roll_trap = False
    if kind == "futures" and "oi" in use.columns:
        look = cfg["roll_look"]
        if len(use) > look + 1:
            oi0 = float(use["oi"].iloc[-look-1])
            oin = float(use["oi"].iloc[-1])
            if oi0 > 0:
                drop = (oi0 - oin) / oi0
                if drop >= cfg["roll_drop_pct"] and (rvol_now >= cfg["rvol_strong"]):
                    roll_trap = True

    feats = {
        "mfi_val": mfi_val,
        "mfi_slope": mfi_slope,
        "mfi_up": bool(mfi_slope > 0),

        "obv_above_ema": obv_above_ema,

        "rvol_now": rvol_now,
        "rvol_strong": bool(rvol_now >= cfg["rvol_strong"]),
        "rvol_extreme": bool(rvol_now >= cfg["rvol_extreme"]),
        "vol_cv20": vol_cv20,
        "vol_rank60": vol_rank60,

        "spread_bps": float(spread_bps) if spread_bps is not None and np.isfinite(spread_bps) else None,
        "turnover_est": float(turnover_est),
        "illiquid_flag": bool(illiquid),
        "roll_trap": bool(roll_trap),

        "in_session": bool(in_session),
    }

    return feats


# ============================================================
# Scenario engine (INI-driven)
# ============================================================

def _run_scenarios(features: Dict[str, Any], cfg: dict) -> Tuple[float, bool, Dict[str, float], List[tuple]]:
    cp = cfg["cp"]
    total = 0.0
    veto = False
    parts: Dict[str, float] = {}
    fired: List[tuple[str, float]] = []

    for name in cfg["scenarios_list"]:
        sec = f"flow_scenario.{name}"
        if not cp.has_section(sec):
            continue

        when_str = cp.get(sec, "when", fallback="").strip()
        if not when_str:
            continue

        when_tree = _compile_rule(when_str)
        if not when_tree:
            continue

        if not _safe_eval(when_tree, features):
            continue

        sc = float(cp.get(sec, "score", fallback="0") or 0.0)

        if cfg["rules_mode"] == "override":
            total = sc
        else:
            total += sc

        parts[name] = sc
        fired.append((name, sc))

        if cp.has_option(sec, "bonus_when"):
            bonus_str = cp.get(sec, "bonus_when", fallback="").strip()
            if bonus_str:
                bonus_tree = _compile_rule(bonus_str)
                if bonus_tree and _safe_eval(bonus_tree, features):
                    bonus_val = float(cp.get(sec, "bonus", fallback="0") or 0.0)
                    total = bonus_val if cfg["rules_mode"] == "override" else (total + bonus_val)
                    parts[name + ".bonus"] = bonus_val
                    fired.append((name + ".bonus", bonus_val))

        if cp.get(sec, "set_veto", fallback="false").strip().lower() in {"1", "true", "yes", "on"}:
            veto = True

    return total, veto, parts, fired


# ============================================================
# ML fetch (ml_pillars)
# ============================================================

def _load_ml_prob_from_pillars(
    symbol: str,
    kind: str,
    tf: str,
    ts,
    cfg: dict,
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    cp = cfg["cp"]
    pillar = cp.get("flow_ml", "pillar", fallback="flow").strip()
    target_name = cp.get("flow_ml", "main_target", fallback="flow_ml.target.bull_2pct_4h").strip()
    version = cp.get("flow_ml", "version", fallback="xgb_v1").strip()
    market_type = kind.lower()

    sql = """
        SELECT prob_up, prob_down, ts
          FROM indicators.ml_pillars
         WHERE pillar       = %s
           AND market_type  = %s
           AND tf           = %s
           AND target_name  = %s
           AND version      = %s
           AND symbol       = %s
           AND ts <= %s
         ORDER BY ts DESC
         LIMIT 1
    """
    params = (pillar, market_type, tf, target_name, version, symbol, ts)

    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
    except Exception as e:
        print(f"[FLOW_ML] ERROR loading ml_pillars for {symbol} {kind} {tf} ts={ts}: {e}")
        return None

    if not row:
        return None

    p_up, p_dn, _ = row

    try:
        p_up = float(p_up) if p_up is not None else None
    except Exception:
        p_up = None

    try:
        p_dn = float(p_dn) if p_dn is not None else None
    except Exception:
        p_dn = None

    return (p_up, p_dn)


# ============================================================
# Public API ✅ write_to_db optimization
# ============================================================

def score_flow(
    symbol: str,
    kind: str,
    tf: str,
    df5: pd.DataFrame,
    base: BaseCfg,
    ini_path=DEFAULT_INI,
    context=None, force_ml: Optional[bool] = None, write_to_db: bool = True):
    """
    Returns:
      (ts, rules_score, rules_veto, ml_score, fused_score, fused_veto, rows)

    Optimization:
      - set write_to_db=False in backfill
      - caller batches rows and writes once per chunk
    """

    # --- 0) TF frame ---
    dftf = df5
    if tf != "15m":
        dftf = resample_nse(df5, tf)
    dftf = maybe_trim_last_bar(dftf)

    n = len(dftf)
    need = min_bars_for_tf_rules(tf)
    if n < need:
        # important: backfill calling with df_slice will hit this until enough bars exist
        return None

    cfg = _cfg(ini_path)

    # --- 1) rules ---
    feats = _build_features(dftf, symbol, kind, tf, cfg, context=context)
    total, veto_scen, parts, fired = _run_scenarios(feats, cfg)

    auto_veto = bool(feats.get("illiquid_flag") or feats.get("roll_trap"))
    rules_score = float(clamp(total, cfg["clamp_low"], cfg["clamp_high"]))
    rules_veto = bool(veto_scen or auto_veto)

    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    # --- 2) ML optional ---
    ml_prob_raw = None
    ml_prob_cal = None
    ml_bucket = None
    ml_score = None

    fused_score = rules_score
    fused_veto = rules_veto

    try:
        cp = cfg["cp"]
        ml_enabled_cfg = cp.getboolean("flow_ml", "enabled", fallback=False)
        ml_enabled = ml_enabled_cfg if force_ml is None else bool(force_ml)

        if ml_enabled and n >= 2:
            ml_prob = None
            res_ml = _load_ml_prob_from_pillars(symbol, kind, tf, ts, cfg)
            if res_ml is not None:
                prob_up, prob_down = res_ml
                if prob_up is not None:
                    ml_prob = prob_up
                elif prob_down is not None:
                    ml_prob = 1.0 - prob_down

            # optional callback
            if ml_prob is None:
                cb_path = cp.get("flow_ml", "callback", fallback="").strip()
                if cb_path:
                    mod_name, _, fn_name = cb_path.rpartition(".")
                    if mod_name and fn_name:
                        mod = importlib.import_module(mod_name)
                        fn = getattr(mod, fn_name)
                        ml_prob = float(fn(symbol=symbol, kind=kind, tf=tf, ts=ts, features=feats))
                        if not np.isfinite(ml_prob):
                            ml_prob = 0.0

            if ml_prob is not None:
                ml_prob_raw = float(ml_prob) if np.isfinite(float(ml_prob)) else 0.0
                ml_prob_raw = max(0.0, min(1.0, ml_prob_raw))
                ml_prob_cal = _calibrate_prob(ml_prob_raw)
                ml_bucket = _find_prob_bucket(ml_prob_cal)
                ml_score = round(100.0 * ml_prob_cal, 2)

                w = float(cp.get("flow_ml", "blend_weight", fallback="0.35"))
                w = max(0.0, min(1.0, w))
                base_prob = rules_score / 100.0
                fused_prob = (1.0 - w) * base_prob + w * ml_prob_cal
                fused_prob = max(0.0, min(1.0, fused_prob))
                fused_score = round(100.0 * fused_prob, 2)

                # optional veto threshold
                veto_if_lt_str = cp.get("flow_ml", "veto_if_prob_lt", fallback="").strip()
                if veto_if_lt_str:
                    thr = float(veto_if_lt_str)
                    if thr > 1.0:
                        thr = thr / 100.0
                    if fused_prob < thr:
                        fused_veto = True

    except Exception as e:
        print(f"[FLOW_ML] ERROR during ML blend {symbol} {kind} {tf} @ {ts}: {e}")

    # --- 3) rows ---
    rows: List[tuple] = []

    rows.append((symbol, kind, tf, ts, "FLOW.score", float(rules_score), "{}", base.run_id, base.source))
    rows.append((symbol, kind, tf, ts, "FLOW.veto_flag", 1.0 if rules_veto else 0.0, "{}", base.run_id, base.source))
    rows.append((symbol, kind, tf, ts, "FLOW.fused_score", float(fused_score), "{}", base.run_id, base.source))
    rows.append((symbol, kind, tf, ts, "FLOW.score_final", float(fused_score), "{}", base.run_id, base.source))
    rows.append((symbol, kind, tf, ts, "FLOW.fused_veto", 1.0 if fused_veto else 0.0, "{}", base.run_id, base.source))

    if ml_score is not None and ml_prob_cal is not None:
        rows.append((symbol, kind, tf, ts, "FLOW.ml_score", float(ml_score), "{}", base.run_id, base.source))
        rows.append((symbol, kind, tf, ts, "FLOW.ml_p_up_cal", float(ml_prob_cal), "{}", base.run_id, base.source))
        rows.append((symbol, kind, tf, ts, "FLOW.ml_p_down_cal", float(1.0 - ml_prob_cal), "{}", base.run_id, base.source))
        if ml_bucket is not None:
            rows.append((symbol, kind, tf, ts, "FLOW.ml_bucket", float(ml_bucket), "{}", base.run_id, base.source))

    debug_ctx = _clean_json_dict({
        "rules_score": rules_score,
        "rules_veto": rules_veto,
        "ml_prob_raw": ml_prob_raw,
        "ml_prob_cal": ml_prob_cal,
        "ml_bucket": ml_bucket,
        "ml_score": ml_score,
        "fused_score": fused_score,
        "fused_veto": fused_veto,
        "rvol": feats.get("rvol_now"),
        "vol_cv20": feats.get("vol_cv20"),
        "vol_rank60": feats.get("vol_rank60"),
        "spread_bps": feats.get("spread_bps"),
        "turnover_est": feats.get("turnover_est"),
        "illiquid_flag": feats.get("illiquid_flag"),
        "roll_trap": feats.get("roll_trap"),
        "in_session": feats.get("in_session"),
    })
    rows.append((symbol, kind, tf, ts, "FLOW.debug_ctx", 0.0, json.dumps(debug_ctx), base.run_id, base.source))

    if cfg["write_scenarios_debug"]:
        for nm, sc in fired:
            rows.append((symbol, kind, tf, ts, f"FLOW.scenario.{nm}", float(sc), "{}", base.run_id, base.source))

    # --- 4) write or return ---
    if write_to_db:
        write_values(rows)

    return ts, rules_score, rules_veto, ml_score, fused_score, fused_veto, rows
