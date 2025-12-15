import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pillars.common import ema, atr, adx

class TrendFeatureEngine:
    """
    Vectorized Feature Engine for Trend Pillar.
    Computes all indicators (EMA, MACD, RSI, etc.) and specific features
    needed for the rules engine.
    """
    def __init__(self, symbol: str, kind: str):
        self.symbol = symbol
        self.kind = kind

    def compute_all_features(self, df: pd.DataFrame, daily_df: pd.DataFrame, metrics_df: pd.DataFrame, cfg: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Computes all trend features and returns the dataframe with added columns.

        Args:
            df: Intraday dataframe (15m usually)
            daily_df: Daily context (optional)
            metrics_df: Metrics for context (optional)
            cfg: Configuration dictionary containing window parameters
        """
        if df.empty:
            return df

        # Ensure we don't modify the original
        df = df.copy()

        # Basic Columns - Ensure OHLCV are preserved and named correctly
        # Assuming input df has them. If missing, we can't create features.
        required_cols = ["open", "high", "low", "close"]
        for c in required_cols:
            if c not in df.columns:
                # If 'low' is missing, rules like 'low < ema20' will fail
                # Assuming standard candle data structure
                pass

        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df.get("volume", pd.Series(0, index=df.index)).fillna(0)

        # ---------------------------------------------------------
        # 1. Configurable Parameters
        # ---------------------------------------------------------
        if cfg is None:
            cfg = {}

        p_ema_fast = int(cfg.get("ema_fast", 10))
        p_ema_mid = int(cfg.get("ema_mid", 20))
        p_ema_slow = int(cfg.get("ema_slow", 50))
        p_adx_main = int(cfg.get("adx_main", 14))
        p_adx_fast = int(cfg.get("adx_fast", 9))
        p_roc_win = int(cfg.get("roc_win", 5))
        p_atr_win = int(cfg.get("atr_win", 14))
        p_bb_win = int(cfg.get("bb_win", 20))
        p_bb_k = float(cfg.get("bb_k", 2.0))
        p_kc_win = int(cfg.get("kc_win", 20))
        p_kc_mult = float(cfg.get("kc_mult", 1.5))
        p_vol_avg = int(cfg.get("vol_avg_win", 20))
        p_div_n = int(cfg.get("div_lookback", 5))

        # ---------------------------------------------------------
        # 2. Indicators
        # ---------------------------------------------------------

        # EMAs
        df["ema10"] = ema(c, p_ema_fast)
        df["ema20"] = ema(c, p_ema_mid)
        df["ema50"] = ema(c, p_ema_slow)

        # Volume Avg
        df["volume_avg_20"] = v.rolling(p_vol_avg).mean()

        # MACD (12, 26, 9)
        macd_line = ema(c, 12) - ema(c, 26)
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_line"] = macd_line
        df["macd_sig"] = macd_sig

        # MACD Histogram & Slope
        hist = macd_line - macd_sig
        df["hist"] = hist
        df["hist_diff"] = hist.diff() # Slope of histogram

        # ATR
        ATR = atr(h, l, c, p_atr_win)
        safe_c = c.replace(0, np.nan)
        df["atr_pct"] = (ATR / safe_c) * 100.0

        # ADX
        adx14, pdi14, mdi14 = adx(h, l, c, p_adx_main)
        df["adx14"] = adx14
        df["pdi"] = pdi14
        df["mdi"] = mdi14

        # Fast ADX
        adx9, _, _ = adx(h, l, c, p_adx_fast)
        df["adx9"] = adx9

        # ROC
        df["roc"] = c.pct_change(p_roc_win)

        # ROC/ATR Ratio
        atr_fraction = ATR / safe_c
        df["roc_abs_over_atr_ratio"] = df["roc"].abs() / atr_fraction.replace(0, np.nan)

        # Bollinger Bands
        bb_mid = c.rolling(p_bb_win).mean()
        bb_std = c.rolling(p_bb_win).std(ddof=0)
        df["bb_up"] = bb_mid + p_bb_k * bb_std
        df["bb_lo"] = bb_mid - p_bb_k * bb_std
        df["bb_width_pct"] = (df["bb_up"] - df["bb_lo"]) / safe_c * 100.0

        # Keltner Channels
        kc_mid = ema(c, p_kc_win)
        df["kc_up"] = kc_mid + p_kc_mult * ATR
        df["kc_lo"] = kc_mid - p_kc_mult * ATR

        # Squeeze Flag
        bb_width = df["bb_up"] - df["bb_lo"]
        kc_width = df["kc_up"] - df["kc_lo"]
        df["squeeze_flag"] = (bb_width < kc_width).astype(int)

        # RSI
        df["rsi_now"] = self._rsi(c, 14)

        # ---------------------------------------------------------
        # 3. Previous / Shifted Values
        # ---------------------------------------------------------
        df["close_prev"] = c.shift(1)
        df["macd_line_prev"] = df["macd_line"].shift(1)
        df["atr_pct_prev"] = df["atr_pct"].shift(1)

        df["adx14_prev"] = df["adx14"].shift(1)
        df["bb_width_pct_prev"] = df["bb_width_pct"].shift(1)

        # Divergence Lookbacks
        df["rsi_prev_n"] = df["rsi_now"].shift(p_div_n)
        df["close_prev_n"] = c.shift(p_div_n)

        # Slopes
        def _calc_slope(s):
            return (s - s.shift(5)) / s.replace(0, np.nan) * 100.0

        df["slope_short"] = _calc_slope(df["ema10"])
        df["slope_mid"] = _calc_slope(df["ema20"])
        df["slope_long"] = _calc_slope(df["ema50"])

        # ---------------------------------------------------------
        # 4. Booleans / Logical Flags
        # ---------------------------------------------------------
        df["dip_gt_dim"] = df["pdi"] > df["mdi"]

        # POC Distance
        if metrics_df is not None and not metrics_df.empty and 'metric' in metrics_df.columns:
            poc_df = metrics_df[metrics_df['metric'] == 'VP.POC'].copy()
            if not poc_df.empty:
                poc_df = poc_df.sort_values('ts')

                df_temp = df.reset_index()
                # Ensure we have 'ts' for merge
                if 'ts' not in df_temp.columns and 'index' in df_temp.columns:
                    df_temp = df_temp.rename(columns={'index': 'ts'})

                # Check if ts exists, otherwise skip
                if 'ts' in df_temp.columns:
                    merged = pd.merge_asof(
                        df_temp,
                        poc_df[['ts', 'val']],
                        on='ts',
                        direction='backward',
                        allow_exact_matches=True
                    )

                    aligned_poc = merged['val'].values
                    dist = np.abs(df['close'].values - aligned_poc)
                    df['poc_dist_atr'] = dist / ATR.replace(0, np.nan)
                else:
                    df['poc_dist_atr'] = 0.0
            else:
                df['poc_dist_atr'] = 0.0
        else:
            df['poc_dist_atr'] = 0.0

        return df

    def _rsi(self, series: pd.Series, win: int = 14) -> pd.Series:
        """Vectorized RSI"""
        delta = series.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.ewm(alpha=1.0 / max(1, win), adjust=False).mean()
        roll_down = down.ewm(alpha=1.0 / max(1, win), adjust=False).mean().replace(0, np.nan)
        rs = roll_up / roll_down
        out = 100.0 - (100.0 / (1.0 + rs))
        return out.fillna(50.0)
