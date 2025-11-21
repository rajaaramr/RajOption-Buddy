# utils/rule_evaluator.py
from __future__ import annotations

import pandas as pd
from typing import Dict, Tuple

from utils.db_ops import fetch_latest_snapshots, fetch_latest_zone_data
from utils.indicators import compute_indicators
from utils.option_utils import advanced_buildup_rules  # kept for compatibility

# ---------------- Rule Registry -------------------
RULES: Dict[str, callable] = {}

RULE_WEIGHTS = {
    "RSI Confirmation": 1.0,
    "ADX Strength Filter": 1.0,
    "ROC Momentum Spike": 0.8,
    "EMA Trend Confirmation": 1.0,
    "SMA Base Strength": 0.8,
    "MFI Strength": 0.8,
    "Trend + Volume Confluence": 1.2,
    "Strong Green Candle": 0.8,
    "Zone Breakout Confirmation": 1.2,
    "RMI Strength": 1.0,
    "MACD Trend Confirmation": 1.0,
}
THRESHOLD_SCORE = 60.0


def rule(name):
    def decorator(func):
        RULES[name] = func
        return func
    return decorator


def _num(x, default=0.0):
    try:
        v = float(x)
        if pd.isna(v):
            return default
        return v
    except Exception:
        return default


def _normalize_ohlcv(market_data: Dict) -> Dict:
    """Ensure we have open/high/low/close/volume floats for indicator calc."""
    o  = market_data.get("open",  market_data.get("open_price"))
    h  = market_data.get("high",  market_data.get("high_price"))
    l  = market_data.get("low",   market_data.get("low_price"))
    c  = market_data.get("close", market_data.get("close_price"))
    v  = market_data.get("volume", 0)

    o = _num(o); h = _num(h); l = _num(l); c = _num(c); v = _num(v)

    # If we only have LTP-like row, backfill O/H/L from close
    if c and (o == 0 and h == 0 and l == 0):
        o = h = l = c

    md = dict(market_data)
    md.update({"open": o, "high": h, "low": l, "close": c, "volume": v})
    return md


# ---------------- Rule Definitions ----------------

@rule("RSI Confirmation")
def rsi_trend_confirmation(market_data, option_data):
    rsi = _num(market_data.get("rsi"))
    if 45 <= rsi <= 70:
        return True, "RSI in bullish trend range"
    return False, f"RSI {rsi} not supportive"

@rule("ADX Strength Filter")
def adx_momentum_check(market_data, option_data):
    adx = _num(market_data.get("adx"))
    if adx >= 20:
        return True, f"ADX strong at {adx}"
    return False, f"ADX weak at {adx}"

@rule("ROC Momentum Spike")
def roc_filter(market_data, option_data):
    roc = _num(market_data.get("roc"))
    if roc > 0.5:
        return True, f"ROC bullish at {roc}"
    return False, f"ROC flat/negative at {roc}"

@rule("EMA Trend Confirmation")
def ema_trend_rule(market_data, option_data):
    ema_20 = _num(market_data.get("ema_20"))
    ema_50 = _num(market_data.get("ema_50"))
    if ema_20 > ema_50:
        return True, f"EMA trend up (20 > 50): {ema_20} > {ema_50}"
    return False, f"EMA trend weak or down: {ema_20} <= {ema_50}"

@rule("SMA Base Strength")
def sma_base_rule(market_data, option_data):
    sma_20 = _num(market_data.get("sma_20"))
    sma_50 = _num(market_data.get("sma_50"))
    if sma_20 > sma_50:
        return True, "SMA short-term strong (20 > 50)"
    return False, "SMA trend not strong"

@rule("MFI Strength")
def mfi_strength_rule(market_data, option_data):
    mfi = _num(market_data.get("mfi_14"))  # ✅ consistent with indicators.py
    if mfi >= 55:
        return True, f"MFI bullish zone ({mfi})"
    return False, f"MFI weak ({mfi})"

@rule("Trend + Volume Confluence")
def trend_volume_confluence(market_data, option_data):
    ema_20 = _num(market_data.get("ema_20"))
    ema_50 = _num(market_data.get("ema_50"))
    mfi    = _num(market_data.get("mfi_14"))
    if ema_20 > ema_50 and mfi >= 55:
        return True, "Trend + Volume confluence confirmed"
    return False, "No trend-volume confluence"

@rule("Strong Green Candle")
def strong_candle_rule(market_data, option_data):
    open_ = _num(market_data.get("open"))
    close = _num(market_data.get("close"))
    high  = _num(market_data.get("high"))
    low   = _num(market_data.get("low"))

    body = abs(close - open_)
    candle_size = max(high - low, 1e-9)
    body_ratio = body / candle_size

    if close > open_ and body_ratio > 0.6:
        return True, "Strong green candle (body > 60% of range)"
    return False, "Candle not strong (body < 60% or red)"

@rule("Zone Breakout Confirmation")
def zone_breakout_confirmation(market_data, option_data):
    symbol = market_data.get("symbol")
    if not symbol:
        return False, "No symbol for zone check"

    zone_data = fetch_latest_zone_data(symbol) or {}
    support_broken    = int(_num(zone_data.get("support_break_flag"))) == 1
    resistance_broken = int(_num(zone_data.get("resistance_break_flag"))) == 1
    buildup_type = (option_data or {}).get("CE_buildup_type", "")

    rsi_val = _num(zone_data.get("rsi_at_val"))
    mfi_val = _num(zone_data.get("mfi_at_val"))
    rsi_vah = _num(zone_data.get("rsi_at_vah"))
    mfi_vah = _num(zone_data.get("mfi_at_vah"))

    if support_broken and buildup_type in ("Long Build Up", "Short Covering") and rsi_val < 40 and mfi_val > 45:
        return True, "Support breakout with RSI < 40 and MFI > 45"

    if resistance_broken and buildup_type in ("Long Build Up", "Short Covering") and rsi_vah > 60 and mfi_vah < 60:
        return True, "Resistance breakout with RSI > 60 and MFI < 60"

    return False, "Zone breakout without RSI/MFI confirmation"

@rule("RMI Strength")
def rmi_strength_rule(market_data, option_data):
    # compute_indicators returns rmi_14 and rmi_14_5 (we mirror your exit evaluator)
    rmi = _num(market_data.get("rmi_14_5", market_data.get("rmi_14")))
    if rmi >= 60:
        return True, f"RMI bullish zone ({rmi})"
    return False, f"RMI not supportive ({rmi})"

@rule("MACD Trend Confirmation")
def macd_trend_rule(market_data, option_data):
    macd        = _num(market_data.get("macd"))
    macd_signal = _num(market_data.get("macd_signal"))
    if macd > macd_signal:
        return True, "MACD line above signal → Bullish trend"
    return False, "MACD below signal → Weak trend"


# ---------------- Evaluation Engine ----------------
def evaluate_alert(symbol: str):
    try:
        market_data, option_data = fetch_latest_snapshots(symbol)
        if not market_data:
            return False, "SnapshotUnavailable", [("snapshot", "Missing market data")], 0, []

        # Normalize OHLCV for indicator computation
        market_data = _normalize_ohlcv(market_data)

        # Compute/refresh indicators on the latest row
        df = pd.DataFrame([{
            "open": market_data["open"],
            "high": market_data["high"],
            "low":  market_data["low"],
            "close":market_data["close"],
            "volume": market_data["volume"],
        }])
        market_data.update(compute_indicators(df) or {})

        passed_rules = []
        failed_rules = []
        score_sum = 0.0
        total_weight = 0.0

        for rule_name, rule_func in RULES.items():
            weight = RULE_WEIGHTS.get(rule_name, 1.0)
            total_weight += weight
            try:
                passed, reason = rule_func(market_data, option_data or {})
                if passed:
                    passed_rules.append(rule_name)
                    score_sum += weight
                else:
                    failed_rules.append((rule_name, reason))
            except Exception as e:
                failed_rules.append((rule_name, f"Exception: {str(e)}"))

        score = round((score_sum / max(total_weight, 1e-9)) * 100, 2)

        if score >= THRESHOLD_SCORE and passed_rules:
            return True, passed_rules[0], failed_rules, score, passed_rules
        else:
            return False, "BelowThreshold", failed_rules, score, passed_rules

    except Exception as e:
        return False, "EvaluationError", [(symbol, str(e))], 0, []
