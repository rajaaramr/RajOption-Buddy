# pillars/risk_pillar.py
from __future__ import annotations
import json, math, configparser
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from .common import *  # ema, atr, adx, bb_width_pct, resample, write_values, last_metric, clamp, TZ, DEFAULT_INI, BaseCfg
from pillars.common import min_bars_for_tf, ensure_min_bars, maybe_trim_last_bar

# -----------------------------
# Config
# -----------------------------
def _cfg(path=DEFAULT_INI):
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)
    # optional rule-engine controls (safe defaults)
    rules_mode = cp.get("risk", "rules_mode", fallback="additive").strip().lower()  # additive | override
    clamp_low  = cp.getfloat("risk", "clamp_low",  fallback=0.0)
    clamp_high = cp.getfloat("risk", "clamp_high", fallback=100.0)
    write_dbg  = cp.getboolean("risk", "write_scenarios_debug", fallback=False)

    # scenarios (optional)
    scen_list_raw = cp.get("risk_scenarios", "list", fallback="").strip()
    scen_names = [s.strip() for s in scen_list_raw.split(",") if s.strip()]
    scenarios: List[Dict[str, Any]] = []
    for name in scen_names:
        sect = f"risk_scenario.{name}"
        if not cp.has_section(sect):
            continue
        scenarios.append({
            "name":        name,
            "when":        cp.get(sect, "when", fallback=""),
            "score":       cp.get(sect, "score", fallback="0"),
            "set_veto":    cp.getboolean(sect, "set_veto", fallback=False),
            "bonus_when":  cp.get(sect, "bonus_when", fallback=""),
            "bonus":       cp.get(sect, "bonus", fallback="0"),
        })

    return {
        # core knobs (your originals)
        "atr_win": cp.getint("risk", "atr_win", fallback=14),

        # Stop alignment band (mirrors Structure defaults)
        "min_stop_atr": cp.getfloat("risk", "min_stop_atr", fallback=0.5),
        "max_stop_atr": cp.getfloat("risk", "max_stop_atr", fallback=3.0),

        # Squeeze (BB width percentile gate for small bonus)
        "squeeze_pct": cp.getfloat("risk", "squeeze_pct", fallback=25.0),

        # Trigger-at-anchor config
        "near_anchor_atr": cp.getfloat("risk", "near_anchor_atr", fallback=0.25),
        "vol_surge_mult":  cp.getfloat("risk", "vol_surge_mult",  fallback=1.5),

        # Position sizing caps
        "max_pos_frac":  cp.getfloat("risk", "max_pos_frac",  fallback=0.10),   # 10%/trade
        "hard_pos_frac": cp.getfloat("risk", "hard_pos_frac", fallback=0.15),   # hard veto

        # Portfolio context caps
        "max_total_exposure": cp.getfloat("risk", "max_total_exposure", fallback=0.60),
        "max_avg_corr":       cp.getfloat("risk", "max_avg_corr",       fallback=0.65),
        "max_sector_conc":    cp.getfloat("risk", "max_sector_conc",    fallback=0.40),

        # Reward bins — reused from structure if STRUCT.rr is missing
        "rr_bins": tuple(float(x) for x in (cp.get("risk","rr_bins",fallback="1.0,1.5,2.0").split(","))[:3]),

        # rule-engine bits
        "rules_mode": rules_mode,
        "clamp_low": clamp_low, "clamp_high": clamp_high,
        "write_scenarios_debug": write_dbg,
        "scenarios": scenarios,
    }

# -----------------------------
# Helpers
# -----------------------------
def _near_anchor(price: float, level: Optional[float], atr_last: float, near_atr: float) -> bool:
    if level is None or atr_last <= 0: return False
    return abs(price - float(level)) <= (near_atr * atr_last)

def _estimate_stop_atr_from_anchors(close: float, atr_last: float, poc: Optional[float], val: Optional[float], vah: Optional[float]) -> float:
    if atr_last <= 0: return 1.0
    below = [x for x in [val, poc] if x is not None and x < close]
    above = [x for x in [vah, poc] if x is not None and x > close]
    if below:
        stop = max(below); return abs(close - float(stop)) / atr_last
    if above:
        stop = min(above); return abs(float(stop) - close) / atr_last
    return 1.0

def _stop_alignment_points(stop_atr: float, cfg) -> float:
    a, b = cfg["min_stop_atr"], cfg["max_stop_atr"]
    if a <= stop_atr <= b: return 20.0
    if stop_atr < a:       return max(0.0, 20.0 * (stop_atr / max(a, 1e-9)))
    return max(0.0, 20.0 * (b / max(stop_atr, 1e-9)))

def _reward_points(rr: float, rr_bins: Tuple[float,float,float]) -> float:
    a,b,c = rr_bins
    if rr >= c: return 25.0
    if rr >= b: return 18.0
    if rr >= a: return 10.0
    return 0.0

# ---- tiny safe-eval for scenarios ----
_ALLOWED_NAMES = set("""
stop_atr rr trig pos_frac total_expo avg_corr sector_conc
near_vah near_poc vol_ok squeeze_rank
atr_last price rvol_now
""".split())

def _safe_eval(expr: str, vars: Dict[str, Any]) -> bool:
    if not expr: return False
    # only allow our variables + numbers + operators
    # we rely on Python eval with no builtins and our dict
    return bool(eval(expr, {"__builtins__": {}}, vars))

def _eval_number(expr: str, vars: Dict[str, Any]) -> float:
    if not expr: return 0.0
    return float(eval(expr, {"__builtins__": {}}, vars))

def _apply_scenarios(cfg: dict, vars: Dict[str, Any]) -> Tuple[float, bool, list]:
    """Return (delta_points, set_veto, hits[])"""
    total = 0.0; veto = False; hits = []
    for s in cfg.get("scenarios", []):
        try:
            cond = _safe_eval(s.get("when",""), vars)
        except Exception:
            cond = False
        if cond:
            pts = 0.0
            try:
                pts = _eval_number(s.get("score","0"), vars)
            except Exception:
                pts = 0.0
            # optional bonus_when / bonus
            bonus_add = 0.0
            bw = s.get("bonus_when","").strip()
            if bw:
                try:
                    if _safe_eval(bw, vars):
                        bonus_add = _eval_number(s.get("bonus","0"), vars)
                except Exception:
                    bonus_add = 0.0
            total += (pts + bonus_add)
            veto = veto or bool(s.get("set_veto", False))
            hits.append({"name": s.get("name","?"), "points": float(pts + bonus_add)})
    return float(total), bool(veto), hits

# -----------------------------
# Core scorer
# -----------------------------
def _risk_score(dtf: pd.DataFrame, symbol: str, kind: str, tf: str, cfg: dict) -> Tuple[float, bool, Dict[str, float], Dict[str, Any]]:
    c=dtf["close"]; h=dtf["high"]; l=dtf["low"]; v=dtf["volume"]
    ATR = atr(h,l,c,cfg["atr_win"]); atr_last = float(ATR.iloc[-1] or 0.0)
    price = float(c.iloc[-1])

    # Anchors & structure helpers
    poc = last_metric(symbol, kind, tf, "VP.POC")
    vah = last_metric(symbol, kind, tf, "VP.VAH")
    val = last_metric(symbol, kind, tf, "VP.VAL")
    struct_stop_atr = last_metric(symbol, kind, tf, "STRUCT.stop_atr")
    struct_rr       = last_metric(symbol, kind, tf, "STRUCT.rr")
    struct_trigger  = last_metric(symbol, kind, tf, "STRUCT.trigger")

    # ---- Stop-Loss Alignment (0–20) + squeeze bonus up to +3
    stop_atr = float(struct_stop_atr) if struct_stop_atr is not None else _estimate_stop_atr_from_anchors(price, atr_last, poc, val, vah)
    stop_pts = _stop_alignment_points(stop_atr, cfg)

    # squeeze bonus via BB width percentile (lower = tighter)
    bw = bb_width_pct(c, n=20, k=2.0)
    squeeze_rank = None
    squeeze_bonus = 0.0
    if len(bw.dropna()) >= 40:
        squeeze_rank = float(pd.Series(bw).rank(pct=True).iloc[-1]) * 100.0  # 0..100
        if squeeze_rank <= cfg["squeeze_pct"]:
            squeeze_bonus = 3.0
    stop_pts = min(20.0, stop_pts + squeeze_bonus)

    # ---- Reward Map (0–25) + liquidity pocket / FVG +5
    rr = float(struct_rr) if struct_rr is not None else 1.0
    reward_pts = _reward_points(rr, cfg["rr_bins"])

    liq_flag = last_metric(symbol, kind, tf, "MAP.liquidity_pocket")
    fvg_flag = last_metric(symbol, kind, tf, "MAP.fvg_near")
    reward_bonus = 5.0 if ( (liq_flag and float(liq_flag)>0.5) or (fvg_flag and float(fvg_flag)>0.5) ) else 0.0
    reward_pts = min(25.0, reward_pts + reward_bonus)

    # ---- Trigger Quality Context (0–20)
    vol_ok = float(v.iloc[-1]) >= cfg["vol_surge_mult"] * float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else True
    near_vah = _near_anchor(price, vah, atr_last, cfg["near_anchor_atr"])
    near_poc = _near_anchor(price, poc, atr_last, cfg["near_anchor_atr"])
    trig_base = float(struct_trigger) if struct_trigger is not None else 10.0
    trigger_ctx_pts = float(clamp(trig_base, 0, 20))
    if vol_ok and (near_vah or near_poc):
        trigger_ctx_pts = min(20.0, trigger_ctx_pts + 5.0)

    # ---- Position Sizing Sanity (0–10)
    pos_frac = last_metric(symbol, kind, "MTF", "PORT.pos_size_frac")
    pos_frac = float(pos_frac) if pos_frac is not None else 0.05
    if pos_frac <= cfg["max_pos_frac"]:
        pos_pts = 10.0; pos_veto = False
    elif pos_frac <= cfg["hard_pos_frac"]:
        over = (pos_frac - cfg["max_pos_frac"]) / max(1e-9, (cfg["hard_pos_frac"] - cfg["max_pos_frac"]))
        pos_pts = float(max(0.0, 10.0 * (1.0 - over))); pos_veto = False
    else:
        pos_pts = 0.0; pos_veto = True

    # ---- Portfolio Risk Context (0–25)
    total_expo = last_metric(symbol, kind, "MTF", "PORT.total_exposure_frac")
    avg_corr   = last_metric(symbol, kind, "MTF", "PORT.avg_corr")
    sector_conc= last_metric(symbol, kind, "MTF", "PORT.sector_conc_frac")

    total_expo = float(total_expo) if total_expo is not None else 0.30
    avg_corr   = float(avg_corr)   if avg_corr   is not None else 0.35
    sector_conc= float(sector_conc)if sector_conc is not None else 0.25

    port_pts = 25.0; port_veto = False

    if total_expo > cfg["max_total_exposure"]:
        over = (total_expo - cfg["max_total_exposure"]) / max(1e-9, 1.0 - cfg["max_total_exposure"])
        port_pts -= 15.0 * min(1.0, over)
        if total_expo >= min(0.95, cfg["max_total_exposure"] + 0.25):
            port_veto = True

    if avg_corr > cfg["max_avg_corr"]:
        over = (avg_corr - cfg["max_avg_corr"]) / max(1e-9, 1.0 - cfg["max_avg_corr"])
        port_pts -= 7.0 * min(1.0, over)

    if sector_conc > cfg["max_sector_conc"]:
        over = (sector_conc - cfg["max_sector_conc"]) / max(1e-9, 1.0 - cfg["max_sector_conc"])
        port_pts -= 7.0 * min(1.0, over)
        if sector_conc >= min(0.80, cfg["max_sector_conc"] + 0.30):
            port_veto = True

    port_pts = float(clamp(port_pts, 0, 25))

    # ---- Aggregate (base)
    base_total = stop_pts + reward_pts + trigger_ctx_pts + pos_pts + port_pts
    base_score = float(clamp(base_total, 0, 100))
    base_veto  = bool(pos_veto or port_veto)

    # ---- Optional scenario rules
    vars = {
        # primitives used by rules
        "stop_atr": float(stop_atr),
        "rr": float(rr),
        "trig": float(trig_base),
        "pos_frac": float(pos_frac),
        "total_expo": float(total_expo),
        "avg_corr": float(avg_corr),
        "sector_conc": float(sector_conc),
        "near_vah": bool(near_vah),
        "near_poc": bool(near_poc),
        "vol_ok": bool(vol_ok),
        "squeeze_rank": float(squeeze_rank if squeeze_rank is not None else 100.0),  # 0 tight .. 100 loose

        # optional extras if someone writes rules with them
        "atr_last": float(atr_last),
        "price": float(price),
        # a rough RVOL if needed by rules
        "rvol_now": float(v.iloc[-1] / max(1e-9, (v.rolling(20).mean().iloc[-1] if len(v)>=20 else v.iloc[-1]))),
    }

    delta, rule_veto, hits = _apply_scenarios(cfg, vars)
    if cfg["rules_mode"] == "override":
        score = delta
    else:
        score = base_score + delta

    score = float(clamp(score, cfg["clamp_low"], cfg["clamp_high"]))
    veto_flag = bool(base_veto or rule_veto)

    parts = {
        "StopAlign": float(stop_pts),
        "RewardMap": float(reward_pts),
        "TriggerCtx": float(trigger_ctx_pts),
        "PosSize": float(pos_pts),
        "Portfolio": float(port_pts),
        "stop_atr": float(stop_atr),
        "rr": float(rr),
        "pos_frac": float(pos_frac),
        "total_exposure": float(total_expo),
        "avg_corr": float(avg_corr),
        "sector_conc": float(sector_conc),
        "near_vah": 1.0 if near_vah else 0.0,
        "near_poc": 1.0 if near_poc else 0.0,
        "rules_delta": float(delta),
    }
    debug_ctx = {"rules_mode": cfg["rules_mode"], "hits": hits}
    return score, veto_flag, parts, debug_ctx

# -----------------------------
# Public API
# -----------------------------
def score_risk(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg, ini_path=DEFAULT_INI):
    dftf = resample(df5, tf)
    dftf = maybe_trim_last_bar(dftf)
    if not ensure_min_bars(dftf, tf):
        return None

    cfg = _cfg(ini_path)
    score, veto, parts, dbg = _risk_score(dftf, symbol, kind, tf, cfg)
    ts = dftf.index[-1].to_pydatetime().replace(tzinfo=TZ)

    rows = [
        (symbol, kind, tf, ts, "RISK.score", float(score), json.dumps({}), base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.veto_flag", 1.0 if veto else 0.0, "{}", base.run_id, base.source),

        # debug parts
        (symbol, kind, tf, ts, "RISK.StopAlign", float(parts["StopAlign"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.RewardMap", float(parts["RewardMap"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.TriggerCtx", float(parts["TriggerCtx"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.PosSize", float(parts["PosSize"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.Portfolio", float(parts["Portfolio"]), "{}", base.run_id, base.source),

        # raw observability
        (symbol, kind, tf, ts, "RISK.stop_atr", float(parts["stop_atr"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.rr", float(parts["rr"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.pos_frac", float(parts["pos_frac"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.total_exposure", float(parts["total_exposure"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.avg_corr", float(parts["avg_corr"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.sector_conc", float(parts["sector_conc"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.near_vah", float(parts["near_vah"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.near_poc", float(parts["near_poc"]), "{}", base.run_id, base.source),
        (symbol, kind, tf, ts, "RISK.rules_delta", float(parts["rules_delta"]), "{}", base.run_id, base.source),
    ]

    if cfg.get("write_scenarios_debug"):
        rows.append((symbol, kind, tf, ts, "RISK.rules_hits",
                     float(len(dbg.get("hits", []))), json.dumps(dbg), base.run_id, base.source))

    write_values(rows)
    return (ts, score, veto)
