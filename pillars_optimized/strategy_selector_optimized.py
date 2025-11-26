# pillars_optimized/strategy_selector_optimized.py
from __future__ import annotations
from typing import Dict, Any, List

def select_strategy(symbol: str, pillar_scores: Dict[str, float], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Selects the best strategy for a given symbol based on pillar scores and context.

    Returns a list of suggested strategies.
    """
    # Placeholder logic:
    if pillar_scores.get("final_score", 50.0) > 65:
        return [{"strategy_code": "Full GS (Aggressive)", "fit_score": 0.8}]
    elif pillar_scores.get("final_score", 50.0) < 35:
        return [{"strategy_code": "Short Sell", "fit_score": 0.8}] # Placeholder for bearish strategy
    else:
        return []
