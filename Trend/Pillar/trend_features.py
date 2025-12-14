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

        # Basic Columns
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df.get("volume", pd.Series(0, index=df.index)).fillna(0)

        # ---------------------------------------------------------
        # 1. Configurable Parameters (Matching Trend V2 / INI defaults)
        # ---------------------------------------------------------
        # Use provided cfg or fall back to defaults
        if cfg is None:
            cfg = {}

        p_ema_fast = cfg.get("ema_fast", 10)
        p_ema_mid = cfg.get("ema_mid", 20)
        p_ema_slow = cfg.get("ema_slow", 50)
        p_adx_main = cfg.get("adx_main", 14)
        p_adx_fast = cfg.get("adx_fast", 9)
        p_roc_win = cfg.get("roc_win", 5)
        p_atr_win = cfg.get("atr_win", 14)
        p_bb_win = cfg.get("bb_win", 20)
        p_bb_k = cfg.get("bb_k", 2.0)
        p_kc_win = cfg.get("kc_win", 20)
        p_kc_mult = cfg.get("kc_mult", 1.5)
        p_vol_avg = cfg.get("vol_avg_win", 20)
        p_div_n = cfg.get("div_lookback", 5) # Default changed to 5 per INI

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
        # Note: MACD standard params (12, 26, 9) are usually fixed, but could be configurable if needed.
        # Assuming standard for now as V2 didn't expose them in main config block explicitly?
        # V2 code: ema(c, 12) - ema(c, 26).
        macd_line = ema(c, 12) - ema(c, 26)
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        df["macd_line"] = macd_line
        df["macd_sig"] = macd_sig

        # FIX: hist_diff is delta of histogram, not histogram itself
        hist = macd_line - macd_sig
        df["hist"] = hist # Store raw hist if needed
        df["hist_diff"] = hist.diff() # The change in histogram (rising/falling)

        # ATR
        ATR = atr(h, l, c, p_atr_win)
        # Avoid zero division
        safe_c = c.replace(0, np.nan)
        df["atr_pct"] = (ATR / safe_c) * 100.0

        # ADX
        # Note: adx() returns (adx, pdi, mdi)
        adx14, pdi14, mdi14 = adx(h, l, c, p_adx_main)
        df["adx14"] = adx14
        df["pdi"] = pdi14
        df["mdi"] = mdi14

        # Fast ADX (just for ADX value usually)
        adx9, _, _ = adx(h, l, c, p_adx_fast)
        df["adx9"] = adx9

        # ROC (Rate of Change)
        df["roc"] = c.pct_change(p_roc_win)

        # ROC/ATR Ratio (roc_abs_over_atr_ratio)
        # Ratio = |ROC%| / (ATR% / 100)
        atr_fraction = ATR / safe_c
        df["roc_abs_over_atr_ratio"] = df["roc"].abs() / atr_fraction.replace(0, np.nan)

        # Bollinger Bands
        bb_mid = c.rolling(p_bb_win).mean()
        bb_std = c.rolling(p_bb_win).std(ddof=0)
        df["bb_up"] = bb_mid + p_bb_k * bb_std
        df["bb_lo"] = bb_mid - p_bb_k * bb_std
        df["bb_width_pct"] = (df["bb_up"] - df["bb_lo"]) / safe_c * 100.0

        # Keltner Channels (for Squeeze)
        kc_mid = ema(c, p_kc_win)
        df["kc_up"] = kc_mid + p_kc_mult * ATR
        df["kc_lo"] = kc_mid - p_kc_mult * ATR

        # Squeeze Flag
        # Squeeze ON if BB is inside KC
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

        # Divergence Lookbacks (Using configurable p_div_n)
        df["rsi_prev_n"] = df["rsi_now"].shift(p_div_n)
        df["close_prev_n"] = c.shift(p_div_n)

        # Slopes (Normalized: % change over 5 bars)
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
        # Safer merge logic to avoid index corruption
        if metrics_df is not None and not metrics_df.empty and 'metric' in metrics_df.columns:
            poc_df = metrics_df[metrics_df['metric'] == 'VP.POC'].copy()
            if not poc_df.empty:
                poc_df = poc_df.sort_values('ts')
                # Use merge_asof directly on copies to get aligned values
                # We want to match candle timestamp with most recent POC

                # Reset index to allow merge (merge_asof needs sorted column)
                df_temp = df.reset_index() # assumes 'ts' is index
                if 'ts' not in df_temp.columns and 'index' in df_temp.columns:
                    df_temp = df_temp.rename(columns={'index': 'ts'})

                # Merge
                merged = pd.merge_asof(
                    df_temp,
                    poc_df[['ts', 'val']],
                    on='ts',
                    direction='backward',
                    allow_exact_matches=True
                )

                # Extract aligned POC
                aligned_poc = merged['val'].values # array

                # Assign back to original DF (length matches)
                # Compute distance
                dist = np.abs(df['close'].values - aligned_poc)
                df['poc_dist_atr'] = dist / ATR.replace(0, np.nan)

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
