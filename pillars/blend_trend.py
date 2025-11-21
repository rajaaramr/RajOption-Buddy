# pillars/blend_trend.py
import json
from .common import load_base_cfg, last_metric, write_values, now_ts, clamp, DEFAULT_INI

# You can override via the function arg; this is just the fallback.
DEFAULT_WEIGHTS = {"25m": 0.30, "65m": 0.30, "125m": 0.25, "250m": 0.15}

def _blend_pillar_mtf(symbol: str, kind: str, base, pillar: str, score_key: str, veto_key: str|None, weights: dict|None):
    """
    Generic blender for any pillar exposing per-TF `{pillar}.{score_key}` and optional `{pillar}.{veto_key}`.
    Writes interval='MTF' rows back for the same keys.
    """
    W = weights or DEFAULT_WEIGHTS
    num = den = 0.0
    used_tfs = []

    for tf, w in W.items():
        if tf not in base.tfs:
            continue
        s = last_metric(symbol, kind, tf, f"{pillar}.{score_key}")
        if s is None:
            continue
        num += float(w) * float(s)
        den += float(w)
        used_tfs.append(tf)

    if den == 0.0:
        return  # nothing to blend

    score_mtf = clamp(num / den, 0, 100)
    ts = now_ts()

    rows = [
        (symbol, kind, "MTF", ts, f"{pillar}.{score_key}", float(score_mtf),
         json.dumps({"scope": "MTF", "used_tfs": used_tfs}), base.run_id, base.source)
    ]

    if veto_key:
        # OR across participating TFs (fallback to base.tfs if none recorded for some reason)
        tfs_for_veto = used_tfs if used_tfs else base.tfs
        veto = any(((last_metric(symbol, kind, tf, f"{pillar}.{veto_key}") or 0.0) > 0.5) for tf in tfs_for_veto)
        rows.append(
            (symbol, kind, "MTF", ts, f"{pillar}.{veto_key}", 1.0 if veto else 0.0, "{}", base.run_id, base.source)
        )

    write_values(rows)

def blend_trend_mtf(symbol: str, kind: str, ini_path: str|None=None, weights: dict|None=None):
    """
    Backward-compatible entrypoint:
    - Blends TREND across TFs → writes MTF TREND.score & TREND.veto_soft
    - Also blends RISK across TFs → writes MTF RISK.score & RISK.veto_flag
    """
    base = load_base_cfg(ini_path or DEFAULT_INI)

    # 1) TREND (soft veto)
    _blend_pillar_mtf(
        symbol=symbol, kind=kind, base=base,
        pillar="TREND", score_key="score", veto_key="veto_soft",
        weights=weights
    )

    # 2) RISK (hard veto)
    _blend_pillar_mtf(
        symbol=symbol, kind=kind, base=base,
        pillar="RISK", score_key="score", veto_key="veto_flag",
        weights=weights
    )
