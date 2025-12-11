# scheduler/strategy_worker.py

from dataclasses import dataclass
from typing import Literal, List, Dict, Any
from utils.db import get_db_connection
import pandas as pd

from scheduler.strategies_options import suggest_option_strategies
from scheduler.strategies_futures import suggest_futures_strategies
from scheduler.strategies_derivatives import suggest_derivative_strategies

Direction = Literal["bull", "bear"]

@dataclass
class StrategySuggestion:
    symbol: str
    direction: Direction
    product: Literal["options", "futures", "derivative"]
    strategy_code: str
    fit_score: float
    tf: str
    meta: Dict[str, Any]

def build_ctx_for_symbol(symbol: str, tf: str) -> dict:
    """
    Gathers all necessary data for a symbol and timeframe into a context dictionary.
    """
    ctx = {"symbol": symbol, "tf": tf}
    with get_db_connection() as conn:
        # 1. Pillar scores from composite_v2
        df_pillars = pd.read_sql("""
            SELECT trend, momentum, quality, flow, structure, risk
            FROM indicators.composite_v2
            WHERE symbol = %s AND interval = 'MTF'
            ORDER BY ts DESC LIMIT 1
        """, conn, params=(symbol,))
        ctx["pillars"] = df_pillars.iloc[0].to_dict() if not df_pillars.empty else {}

        # 2. Latest intraday indicators
        df_frames = pd.read_sql(f"""
            SELECT * FROM indicators.futures_frames
            WHERE symbol = %s AND interval = %s
            ORDER BY ts DESC LIMIT 1
        """, conn, params=(symbol, tf))
        ctx["frames"] = df_frames.iloc[0].to_dict() if not df_frames.empty else {}

        # 3. Daily futures data
        df_daily_fut = pd.read_sql("""
            SELECT * FROM raw_ingest.daily_futures
            WHERE symbol = %s
            ORDER BY trade_date DESC LIMIT 1
        """, conn, params=(symbol,))
        ctx["daily_fut"] = df_daily_fut.iloc[0].to_dict() if not df_daily_fut.empty else {}

        # 4. Daily options chain
        df_daily_opt = pd.read_sql("""
            SELECT * FROM raw_ingest.daily_options
            WHERE symbol = %s
            ORDER BY last_updated DESC
        """, conn, params=(symbol,))
        ctx["daily_opt_chain"] = df_daily_opt.to_dict('records')

        # 5. Unified daily data
        df_unified = pd.read_sql("""
            SELECT * FROM raw_ingest.fo_daily_unified
            WHERE nse_code = %s
            ORDER BY trade_date DESC LIMIT 1
        """, conn, params=(symbol,))
        ctx["unified"] = df_unified.iloc[0].to_dict() if not df_unified.empty else {}

        # 6. ML signals
        df_ml = pd.read_sql("""
            SELECT * FROM public.indicators_ml_signals_multiclass
            WHERE symbol = %s
            ORDER BY ts DESC LIMIT 1
        """, conn, params=(symbol,))
        ctx["ml"] = df_ml.iloc[0].to_dict() if not df_ml.empty else {}

    return ctx


def infer_direction(ctx) -> Direction:
    if not ctx.get("pillars"):
        return "bull" # Default

    trend_score = ctx["pillars"].get("trend", 50.0)
    ml_edge = ctx.get("ml", {}).get("edge", 0.0)

    if trend_score > 60 and ml_edge > 0.1:
        return "bull"
    elif trend_score < 40 and ml_edge < -0.1:
        return "bear"
    else:
        return "bull" # Neutral default

def run_for_universe(symbols: List[str], tf: str = "30m") -> List[StrategySuggestion]:
    all_suggestions: List[StrategySuggestion] = []

    for sym in symbols:
        try:
            ctx = build_ctx_for_symbol(sym, tf)
            direction = infer_direction(ctx)
            ctx["direction_bias"] = direction

            opt_suggestions = suggest_option_strategies(ctx)
            fut_suggestions = suggest_futures_strategies(ctx)
            deriv_suggestions = suggest_derivative_strategies(ctx)

            all_suggestions.extend(opt_suggestions)
            all_suggestions.extend(fut_suggestions)
            all_suggestions.extend(deriv_suggestions)
        except Exception as e:
            print(f"Error processing {sym}: {e}")

    return all_suggestions
