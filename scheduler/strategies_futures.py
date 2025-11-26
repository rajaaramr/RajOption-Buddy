# scheduler/strategies_futures.py
from typing import Dict, Any, List
from scheduler.strategy_worker import StrategySuggestion

# Placeholder fit functions for each futures strategy
def fit_f1_bo_only(ctx: Dict[str, Any]) -> float: return 0.0
def fit_f2_retest_only(ctx: Dict[str, Any]) -> float: return 0.0
def fit_f3_vwap_only(ctx: Dict[str, Any]) -> float: return 0.0
def fit_f4_bo_rt(ctx: Dict[str, Any]) -> float: return 0.0
def fit_f5_rt_vwap(ctx: Dict[str, Any]) -> float: return 0.0
def fit_f6_full_gs(ctx: Dict[str, Any]) -> float: return 0.0
def fit_f7_aggressive_gs(ctx: Dict[str, Any]) -> float: return 0.0
def fit_f8_conservative_gs(ctx: Dict[str, Any]) -> float: return 0.0
def fit_f9_vwap_ema_pair(ctx: Dict[str, Any]) -> float: return 0.0
def fit_f10_ml_filtered_gs(ctx: Dict[str, Any]) -> float: return 0.0

def suggest_futures_strategies(ctx: Dict[str, Any]) -> List[StrategySuggestion]:
    """
    Computes fit scores for all futures strategies and returns the top suggestions.
    """
    suggestions = []

    strategy_functions = {
        "F1_BO_ONLY": fit_f1_bo_only,
        "F2_RETEST_ONLY": fit_f2_retest_only,
        "F3_VWAP_ONLY": fit_f3_vwap_only,
        "F4_BO_RT": fit_f4_bo_rt,
        "F5_RT_VWAP": fit_f5_rt_vwap,
        "F6_FULL_GS": fit_f6_full_gs,
        "F7_AGGRESSIVE_GS": fit_f7_aggressive_gs,
        "F8_CONSERVATIVE_GS": fit_f8_conservative_gs,
        "F9_VWAP_EMA_PAIR": fit_f9_vwap_ema_pair,
        "F10_ML_FILTERED_GS": fit_f10_ml_filtered_gs,
    }

    for strategy_code, fit_function in strategy_functions.items():
        fit_score = fit_function(ctx)
        if fit_score > 0.6: # Threshold
            suggestions.append(StrategySuggestion(
                symbol=ctx["symbol"],
                direction=ctx["direction_bias"],
                product="futures",
                strategy_code=strategy_code,
                fit_score=fit_score,
                tf=ctx["tf"],
                meta={}
            ))

    return suggestions
