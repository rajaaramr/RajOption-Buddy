# File: utils/compose_signals.py
from __future__ import annotations
from typing import Dict

from utils.indicators import compute_weighted_scores, load_indicator_config
from utils.buildups import compute_futures_buildup, compute_optionchain_buildup

def compose_blob(dfs_by_tf: Dict[str, "pd.DataFrame"], symbol: str, ini_path: str = "indicators.ini") -> Dict:
    cfg = load_indicator_config(ini_path)
    blob = compute_weighted_scores(dfs_by_tf, ini_path)  # has rsi/rmi and .score

    fut = compute_futures_buildup(symbol)
    oc  = compute_optionchain_buildup(symbol)

    blob["fut_buildup"] = fut
    blob["oc_buildup"]  = oc

    # recompute final confidence including buildups if weights provided
    num = den = 0.0
    # Start with the rsi/rmi mix already inside blob["score"]
    base_w = (cfg.weights.get("rsi", 0.0) + cfg.weights.get("rmi", 0.0))
    if base_w > 0 and blob.get("score") is not None:
        num += base_w * float(blob["score"])
        den += base_w

    for name in ("fut_buildup", "oc_buildup"):
        w = cfg.weights.get(name, 0.0)
        s = (blob.get(name) or {}).get("score")
        if w > 0 and s is not None:
            num += w * float(s); den += w

    blob["score"] = round(num/den, 2) if den > 0 else blob.get("score")
    return blob
