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

    def compute_all_features(self, df: pd.DataFrame, daily_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all trend features and returns the dataframe with added columns.
        """
        if df.empty:
            return df

        # Ensure we don't modify the original
        df = df.copy()

        # Basic Columns
        c = df["close"]
        h = df["high"]
        l = df["low"]
        o = df["open"]
        v = df.get("volume", pd.Series(0, index=df.index)).fillna(0)

        # ---------------------------------------------------------
        # 1. Configurable Parameters (Matching Trend V2 / INI defaults)
        # ---------------------------------------------------------
        # These should match the INI defaults if not overridden,
        # but for the engine we can hardcode standard defaults or pass config.
        # We will assume standard defaults here which map to the INI 'ema_fast', etc.
        p_ema_fast = 10
        p_ema_mid = 20
        p_ema_slow = 50
        p_adx_main = 14
        p_adx_fast = 9
        p_roc_win = 5
        p_atr_win = 14
        p_bb_win = 20
        p_bb_k = 2.0
        p_kc_win = 20
        p_kc_mult = 1.5
        p_vol_avg = 20
        p_div_n = 4  # Default lookback for divergence (~1h on 15m)

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
        df["hist_diff"] = macd_line - macd_sig

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
        # Ratio = |ROC%| / (ATR% / 100) -> wait.
        # V2 logic: abs(roc) / (atr_val / last_c)
        # roc is pct_change (e.g. 0.01 for 1%). atr_val/last_c is e.g. 0.01 for 1%.
        # So Ratio ~ 1.0 means move is 1 ATR.
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
        # i.e., bb_up < kc_up AND bb_lo > kc_lo
        # Or simply width comparison if centered? Usually checks bandwidth.
        # V2 logic: squeeze = 1 if (bb_up - bb_lo) < (kc_up - kc_lo) else 0
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

        # ---------------------------------------------------------
        # 4. Booleans / Logical Flags (for easier Rules)
        # ---------------------------------------------------------
        df["dip_gt_dim"] = df["pdi"] > df["mdi"]

        # POC Distance (Requires context/metrics, handled via simple column if available, else 0)
        # If 'metrics_df' has POC data, we could merge it.
        # For now, we'll placeholder or extract from metrics if the pattern allows.
        # The V2 passed 'context'. Here we usually pass 'metrics_df' and forward fill.
        # But 'process_symbol_vectorized' passes 'metrics_df'.

        if not metrics_df.empty and 'metric' in metrics_df.columns:
            # Filter for VP.POC
            poc_df = metrics_df[metrics_df['metric'] == 'VP.POC'].copy()
            if not poc_df.empty:
                # Merge POC onto main df by timestamp (asof)
                # Ensure ts is datetime and sorted
                poc_df = poc_df.sort_values('ts')
                df = pd.merge_asof(df.sort_index(), poc_df[['ts', 'val']], left_index=True, right_on='ts', direction='backward')
                df = df.rename(columns={'val': 'poc_price'})
                df.index = df['ts'] # Restore index
                df = df.drop(columns=['ts'])

                # Compute Distance
                dist = (df['close'] - df['poc_price']).abs()
                df['poc_dist_atr'] = dist / ATR.replace(0, np.nan)
            else:
                df['poc_dist_atr'] = 0.0
        else:
            df['poc_dist_atr'] = 0.0

        # Fill NaNs where appropriate (e.g. initial bars)
        # For rules, NaNs usually cause False, which is fine.

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
