from __future__ import annotations

import ast
import configparser
import functools
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse common indicators/utils where possible
from pillars.common import (
    ema as _calc_ema,
    atr as _calc_atr,
    adx as _calc_adx,
    bb_width_pct as _calc_bb_width,
    obv_series as _calc_obv,
)

logger = logging.getLogger(__name__)

def _calc_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    if len(close) < 3:
        return pd.Series([50.0] * len(close), index=close.index, dtype=float)
    d = close.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    ema_g = g.ewm(alpha=1 / max(1, n), adjust=False).mean()
    ema_l = l.ewm(alpha=1 / max(1, n), adjust=False).mean().replace(0, np.nan)
    rs = ema_g / ema_l
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INI = str(BASE_DIR / "momentum_scenarios.ini")


class MomentumFeatureEngine:
    """
    Vectorized Feature Engine for Momentum Pillar.
    Replaces the row-by-row `momentum_pillar.py` logic.
    """

    def __init__(self, ini_path: str = None):
        self.ini_path = ini_path or DEFAULT_INI
        self.cfg = self._load_config(self.ini_path)

    def _load_config(self, path: str) -> dict:
        cp = configparser.ConfigParser(
            inline_comment_prefixes=(";", "#"),
            interpolation=None,
            strict=False,
        )
        cp.read(path)

        # Scenarios
        scenarios = []
        if cp.has_section("mom_scenarios"):
            raw_list = cp.get("mom_scenarios", "list", fallback="")
            names = [n.strip() for n in raw_list.replace("\n", " ").split(",") if n.strip()]

            for n in names:
                sec = f"mom_scenario.{n}"
                if cp.has_section(sec):
                    scenarios.append({
                        "name": n,
                        "when": cp.get(sec, "when", fallback=""),
                        "score": cp.getfloat(sec, "score", fallback=0.0),
                        "bonus_when": cp.get(sec, "bonus_when", fallback=""),
                        "bonus": cp.getfloat(sec, "bonus", fallback=0.0),
                    })

        return {
            "rsi_fast": cp.getint("momentum", "rsi_fast", fallback=5),
            "rsi_std": cp.getint("momentum", "rsi_std", fallback=14),
            "rmi_lb": cp.getint("momentum", "rmi_lb", fallback=14),
            "rmi_m": cp.getint("momentum", "rmi_m", fallback=5),
            "atr_win": cp.getint("momentum", "atr_win", fallback=14),
            "low_vol_thr": cp.getfloat("momentum", "low_vol_thr", fallback=3.0),
            "mid_vol_thr": cp.getfloat("momentum", "mid_vol_thr", fallback=6.0),
            "vol_avg_win": cp.getint("momentum", "vol_avg_win", fallback=20),
            "bb_win": cp.getint("momentum", "bb_win", fallback=20),
            "bb_k": cp.getfloat("momentum", "bb_k", fallback=2.0),
            "div_lookback": cp.getint("momentum", "div_lookback", fallback=5),
            "clamp_low": cp.getfloat("momentum", "clamp_low", fallback=0.0),
            "clamp_high": cp.getfloat("momentum", "clamp_high", fallback=100.0),
            "rules_mode": cp.get("momentum", "rules_mode", fallback="additive").lower(),
            "scenarios": scenarios,
        }

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators required for Momentum scenarios.
        Returns a DataFrame with all feature columns added.
        """
        if df.empty:
            return df

        # Copy to avoid SettingWithCopy warnings
        df = df.copy()

        # Ensure numeric
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = df[c].astype(float).fillna(0.0)

        # ---------------------------------------------------------
        # 1. Basic Indicators (Vectorized)
        # ---------------------------------------------------------
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]

        # RSI
        df["rsi_std"] = _calc_rsi(close, self.cfg["rsi_std"])
        df["rsi5"] = _calc_rsi(close, self.cfg["rsi_fast"])  # mapped to 'rsi_fast' in logic, 'rsi5' in scenarios
        df["rsi_fast"] = df["rsi5"]

        # ATR
        # Common ATR returns series
        atr_s = _calc_atr(high, low, close, self.cfg["atr_win"])
        df["atr_val"] = atr_s
        # ATR % (normalized by price)
        # Prevent div/0
        c_safe = close.replace(0, np.nan)
        df["atr_pct"] = (atr_s / c_safe) * 100.0

        # Day ATR: Map to current timeframe ATR as fallback
        df["day_atr"] = df["atr_val"]

        # MACD
        # Standard 12, 26, 9
        ema12 = _calc_ema(close, 12)
        ema26 = _calc_ema(close, 26)
        macd_line = ema12 - ema26
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - macd_sig

        df["macd_line"] = macd_line
        df["macd_sig"] = macd_sig
        df["hist"] = hist
        df["macd_hist"] = hist # Alias

        # Hist EMA (for `hist > hist_ema`)
        df["hist_ema"] = hist.ewm(span=9, adjust=False).mean()
        df["hist_diff"] = hist.diff()

        # ADX / DI
        adx_s, dip, dim = _calc_adx(high, low, close, 14)
        df["adx14"] = adx_s
        df["adx9"] = _calc_adx(high, low, close, 9)[0] # For adx9 > adx14 check
        df["di_plus"] = dip
        df["di_minus"] = dim
        df["di_plus_gt"] = (dip > dim)
        df["adx_rising"] = (adx_s > adx_s.shift(1))

        # Bollinger Bands
        # Width %
        bb_w = _calc_bb_width(close, self.cfg["bb_win"], self.cfg["bb_k"])
        df["bb_width_pct"] = bb_w

        ma = close.rolling(self.cfg["bb_win"]).mean()
        std = close.rolling(self.cfg["bb_win"]).std()
        upper = ma + (self.cfg["bb_k"] * std)
        lower = ma - (self.cfg["bb_k"] * std)

        df["bb_width"] = df["bb_width_pct"]

        # BB Rank (Percentile over past 100 bars approx)
        df["bb_width_pct_rank"] = df["bb_width_pct"].rolling(100).rank(pct=True) * 100.0

        # RMI (Relative Momentum Index)
        lb = self.cfg["rmi_lb"]
        m_rmi = self.cfg["rmi_m"]
        diffm = close.diff(m_rmi)
        up = diffm.clip(lower=0)
        dn = (-diffm).clip(lower=0)
        ema_up = up.ewm(span=max(2, lb), adjust=False).mean()
        ema_dn = dn.ewm(span=max(2, lb), adjust=False).mean().replace(0, np.nan)
        r = ema_up / ema_dn
        df["rmi"] = 100 - 100 / (1 + r)
        df["rmi"] = df["rmi"].fillna(50.0)
        df["rmi_now"] = df["rmi"]

        # OBV & Z-Score
        obv = _calc_obv(close, vol)
        obv_d = obv.diff()
        mu = obv_d.ewm(span=5, adjust=False).mean()
        sd = obv_d.rolling(20).std(ddof=1).replace(0, np.nan)
        df["z_obv"] = (obv_d - mu) / sd
        df["z_obv"] = df["z_obv"].fillna(0.0)

        # RVOL
        v_avg = vol.rolling(self.cfg["vol_avg_win"]).mean().replace(0, np.nan)
        df["rvol_now"] = vol / v_avg
        df["rvol_now"] = df["rvol_now"].fillna(0.0)

        # MFI (Money Flow Index)
        tp = (high + low + close) / 3.0
        rmf = tp * vol
        up_flow = rmf.where(tp > tp.shift(1), 0.0)
        dn_flow = rmf.where(tp < tp.shift(1), 0.0)

        mfi_win = 14
        mfi_up_sum = up_flow.rolling(mfi_win).sum()
        mfi_dn_sum = dn_flow.rolling(mfi_win).sum().replace(0, np.nan)
        mfi_ratio = mfi_up_sum / mfi_dn_sum
        df["mfi"] = 100 - 100 / (1 + mfi_ratio)
        df["mfi"] = df["mfi"].fillna(50.0)
        df["mfi_up"] = (df["mfi"] > df["mfi"].shift(1))

        # Whipsaw Flips
        hist_sign = np.sign(df["hist"])
        flips = (hist_sign.diff().abs() > 0).astype(int)
        df["whipsaw_flips"] = flips.rolling(12).sum()

        # Squeeze Flag
        # Calculate KC (SMA 20, 1.5 ATR)
        sma20 = close.rolling(20).mean()
        atr20 = _calc_atr(high, low, close, 20)
        kc_upper = sma20 + (1.5 * atr20)
        kc_lower = sma20 - (1.5 * atr20)

        sqz_on = (lower > kc_lower) & (upper < kc_upper)
        df["squeeze_flag"] = sqz_on.astype(int)

        # ROC (Rate of Change)
        df["roc21"] = close.pct_change(21) * 100.0
        df["roc_atr_ratio"] = (close.diff() / df["atr_val"]).fillna(0.0)

        # Defaults for missing features
        df["near_r1_break"] = False
        df["tl_mom"] = 0.0
        df["tl_mom_prev"] = 0.0

        # EMA 50
        df["ema50"] = _calc_ema(close, 50)

        # zero_cross_up
        df["zero_cross_up"] = ((df["macd_line"] > 0) & (df["macd_line"].shift(1) <= 0))

        # "prev_n" logic
        n_look = self.cfg["div_lookback"]
        df["close_prev_n"] = close.shift(n_look)
        df["rsi_prev_std_n"] = df["rsi_std"].shift(n_look)
        df["rsi_prev5"] = df["rsi5"].shift(1)
        df["rmi_prev_n"] = df["rmi"].shift(n_look)
        df["bb_width_prev_n"] = df["bb_width_pct"].shift(n_look)
        df["macd_hist_prev_n"] = df["hist"].shift(n_look)
        df["low_prev_n"] = low.shift(n_look)

        # ---------------------------------------------------------
        # 2. Evaluate Scenarios (Vectorized)
        # ---------------------------------------------------------

        # We'll accumulate score in a column
        df["MOM.score"] = 0.0
        df["MOM.veto_flag"] = 0.0

        # ATR logic for RSI threshold
        cond_low = df["atr_pct"] < self.cfg["low_vol_thr"]
        cond_mid = (df["atr_pct"] >= self.cfg["low_vol_thr"]) & (df["atr_pct"] < self.cfg["mid_vol_thr"])

        rsi_thr = pd.Series(70.0, index=df.index)
        rsi_thr[cond_low] = 60.0
        rsi_thr[cond_mid] = 65.0

        # RSI Pts
        rsi_pts = pd.Series(0.0, index=df.index)
        rsi_pts[df["rsi_std"] >= (rsi_thr - 5)] = 8.0
        rsi_pts[df["rsi_std"] >= rsi_thr] = 15.0

        # Hist Pts
        macd_pts = np.where(df["hist"] > 0, 15.0, 0.0)

        # Whipsaw Penalty
        whipsaw_flag = (df["whipsaw_flips"] >= 4)
        whipsaw_pen = np.where(whipsaw_flag, -10.0, 0.0)
        df["MOM.veto_flag"] = whipsaw_flag.astype(float)

        # Vol Pts
        vol_pts = pd.Series(0.0, index=df.index)
        vol_pts[df["z_obv"] >= 1.0] = 10.0
        vol_pts[df["z_obv"] >= 2.0] = 15.0

        # ADX Pts
        adx_pts = np.where((df["adx14"] > 18) & (df["di_plus"] > df["di_minus"]), 15.0, 0.0)

        base_score = rsi_pts + macd_pts + vol_pts + adx_pts + whipsaw_pen
        df["MOM.score"] += base_score

        # Scenarios
        for sc in self.cfg["scenarios"]:
            try:
                name = sc["name"]
                when = sc["when"]
                score = sc["score"]
                bonus_when = sc["bonus_when"]
                bonus = sc["bonus"]

                if when:
                    mask = df.eval(when).fillna(False)
                    if mask.any():
                        df.loc[mask, "MOM.score"] += score

                    if bonus_when:
                        mask_b = df.eval(bonus_when).fillna(False)
                        if mask_b.any():
                            df.loc[mask_b, "MOM.score"] += bonus
            except Exception as e:
                pass

        # Clamp
        df["MOM.score"] = df["MOM.score"].clip(self.cfg["clamp_low"], self.cfg["clamp_high"])

        return df

    def get_feature_columns(self) -> List[str]:
        return [
            "rsi_std", "rsi5", "atr_pct", "macd_line", "macd_sig", "hist",
            "adx14", "di_plus", "di_minus", "bb_width_pct", "bb_width_pct_rank",
            "rmi", "z_obv", "rvol_now", "mfi", "whipsaw_flips", "squeeze_flag",
            "roc21", "roc_atr_ratio", "MOM.score", "MOM.veto_flag"
        ]
