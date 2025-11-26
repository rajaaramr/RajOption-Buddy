# scheduler/strategies_options.py
from typing import Dict, Any, List
from scheduler.strategy_worker import StrategySuggestion

def pick_ce_strike(chain, spot, expected_move_pct, dte, iv_regime, direction="bull"):
    # ... implementation to be added ...
    pass

def pick_pe_strike(chain, spot, expected_move_pct, dte, iv_regime, direction="bear"):
    # ... implementation to be added ...
    pass

def fit_bo_only(ctx: Dict[str, Any]) -> float:
    """
    Calculates a 'fit score' for the 'BO Only' (Breakout Only) options strategy.

    This strategy is ideal for clean breakout days with strong volume and momentum.
    """
    pillars = ctx.get("pillars", {})
    frames = ctx.get("frames", {})

    # Normalize helper
    def norm(val, max_val=100):
        return (val / max_val) if val is not None else 0.5

    score = 0.0

    # 1. High and positive Trend and Momentum are key
    score += 0.4 * norm(pillars.get("trend"))
    score += 0.2 * norm(pillars.get("momentum"))

    # 2. Flow should be strong
    score += 0.2 * norm(pillars.get("flow"))

    # 3. ML should agree with the direction
    ml_edge = ctx.get("ml", {}).get("edge", 0.0)
    if ctx.get("direction_bias") == "bull" and ml_edge > 0:
        score += 0.2 * ml_edge
    elif ctx.get("direction_bias") == "bear" and ml_edge < 0:
        score += 0.2 * abs(ml_edge)

    # 4. Penalize if risk is high or quality is poor
    if pillars.get("risk", 50) < 40: # Lower score is higher risk
        score -= 0.3
    if pillars.get("quality", 50) < 50:
        score -= 0.3

    return max(0.0, score)


def fit_rt_only(ctx: Dict[str, Any]) -> float:
    # ... implementation to be added ...
    return 0.0

def fit_vwap_only(ctx: Dict[str, Any]) -> float:
    # ... implementation to be added ...
    return 0.0

def fit_rt_vwap(ctx: Dict[str, Any]) -> float:
    # ... implementation to be added ...
    return 0.0

# ... other fit functions ...

def suggest_option_strategies(ctx: Dict[str, Any]) -> List[StrategySuggestion]:
    """
    Computes fit scores for all option strategies and returns the top suggestions.
    """
    suggestions = []

    strategy_functions = {
        "OPT_BO_ONLY": fit_bo_only,
        "OPT_RT_ONLY": fit_rt_only,
        "OPT_VWAP_ONLY": fit_vwap_only,
        "OPT_RT_VWAP": fit_rt_vwap,
    }

    for strategy_code, fit_function in strategy_functions.items():
        fit_score = fit_function(ctx)
        if fit_score > 0.6: # Threshold
            suggestions.append(StrategySuggestion(
                symbol=ctx["symbol"],
                direction=ctx["direction_bias"],
                product="options",
                strategy_code=strategy_code,
                fit_score=fit_score,
                tf=ctx["tf"],
                meta={} # Strike info will be added here
            ))

    return suggestions
