from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Optional, Any
from utils.db import get_db_connection

class FlowFeatureEngine:
    def __init__(self, symbol: str, kind: str):
        self.symbol = symbol
        self.kind = kind

    def compute_all_features(self, df: pd.DataFrame, daily_df: Optional[pd.DataFrame] = None,
                             external_metrics_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Computes all features required for Flow Pillar scenarios in a vectorized manner.
        """
        if df.empty:
            return df

        # Ensure sorted
        df = df.sort_index()

        # 1. Basic Price/Vol features
        df = self._compute_price_vol_features(df)

        # 2. Indicators (MFI, RSI, OBV, VWAP)
        df = self._compute_indicators(df)

        # 3. Intraday VOI (Volume-OI Analysis)
        df = self._compute_voi(df)

        # 4. Merge Daily Futures Data (if provided)
        if daily_df is not None and not daily_df.empty:
            df = self._merge_daily_futures(df, daily_df)

        # 5. Merge External Metrics (if provided)
        if external_metrics_df is not None and not external_metrics_df.empty:
            df = self._merge_external_metrics(df, external_metrics_df)

        # 6. Session Logic
        df = self._compute_session_logic(df)

        return df

    def _compute_price_vol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Price changes
        df['price_change'] = df['close'].diff()
        df['price_hh'] = (df['high'] > df['high'].shift(1))
        df['price_ll'] = (df['low'] < df['low'].shift(1))
        df['price_higher_high'] = df['high'] > df['high'].rolling(5).max().shift(1) # Approximate for divergence
        df['price_lower_low'] = df['low'] < df['low'].rolling(5).min().shift(1)

        # Wick ratio (Upper wick vs Body)
        # body = abs(close - open)
        # upper_wick = high - max(open, close)
        # Avoid division by zero
        body = (df['close'] - df['open']).abs()
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        df['wick_ratio'] = upper_wick / body.replace(0, 0.01)

        # Reversal candle: strict definition or loose?
        # Simple Pinbar logic: long lower wick for bullish, long upper for bearish
        # Let's use a placeholder for "price_reversal_candle" based on INI context (usually reversal up)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        df['price_reversal_candle'] = (lower_wick > 2 * body) & (lower_wick > upper_wick)

        return df

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # RVOL (Relative Volume)
        # rvol = volume / average_volume_20
        avg_vol = df['volume'].rolling(20).mean()
        df['rvol_now'] = df['volume'] / avg_vol.replace(0, np.nan)
        df['rvol_now'] = df['rvol_now'].fillna(0)

        # Volatility CV (Coeff of Variation of Volume)
        vol_std = df['volume'].rolling(20).std(ddof=1)
        df['vol_cv20'] = (vol_std / avg_vol).fillna(0)

        # MFI (Money Flow Index) - 14 period
        # Typical Price
        tp = (df['high'] + df['low'] + df['close']) / 3
        rmf = tp * df['volume']

        # Positive/Negative Flow
        tp_diff = tp.diff()
        pos_flow = pd.Series(np.where(tp_diff > 0, rmf, 0), index=df.index)
        neg_flow = pd.Series(np.where(tp_diff < 0, rmf, 0), index=df.index)

        pos_mf = pos_flow.rolling(14).sum()
        neg_mf = neg_flow.rolling(14).sum()

        mfi_ratio = pos_mf / neg_mf.replace(0, np.nan)
        df['mfi_val'] = 100 - (100 / (1 + mfi_ratio))
        df['mfi_val'] = df['mfi_val'].fillna(50)

        # MFI Slope/Direction
        df['mfi_up'] = df['mfi_val'] > df['mfi_val'].shift(1)
        df['mfi_prev_high'] = df['mfi_val'].rolling(20).max().shift(1)

        # OBV (On Balance Volume)
        price_diff = df['close'].diff()
        obv_direction = np.sign(price_diff).fillna(0)
        df['obv'] = (obv_direction * df['volume']).cumsum()

        # OBV EMA
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        df['obv_above_ema'] = df['obv'] > df['obv_ema']

        # OBV HH/HL
        df['obv_hh'] = df['obv'] > df['obv'].rolling(5).max().shift(1)
        df['obv_higher_low'] = df['obv'] > df['obv'].rolling(20).min().shift(1) # simplistic

        return df

    def _compute_voi(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'oi' not in df.columns:
            # Create dummy columns if OI is missing (Spot data)
            for col in ['voi_long_build', 'voi_short_cover', 'voi_short_build',
                        'voi_long_unwind', 'roll_trap', 'voi_mag']:
                df[col] = False if col != 'voi_mag' else 0.0
            return df

        price_chg = df['close'].diff()
        oi_chg = df['oi'].diff().fillna(0)

        # Logic matches standard VOI analysis
        # Long Build: Price Up, OI Up
        df['voi_long_build'] = (price_chg > 0) & (oi_chg > 0)

        # Short Cover: Price Up, OI Down
        df['voi_short_cover'] = (price_chg > 0) & (oi_chg < 0)

        # Short Build: Price Down, OI Up
        df['voi_short_build'] = (price_chg < 0) & (oi_chg > 0)

        # Long Unwind: Price Down, OI Down
        df['voi_long_unwind'] = (price_chg < 0) & (oi_chg < 0)

        # Magnitude
        # Protect against div by zero
        prev_oi = df['oi'].shift(1).replace(0, np.nan)
        df['voi_mag'] = (oi_chg.abs() / prev_oi).fillna(0)

        # Roll Trap: OI drops significantly, Price moves?
        # INI: roll_drop_pct=0.35, rvol_strong=1.5
        # Implementation: Check single bar drop > 35%
        oi_drop_pct = (df['oi'].shift(1) - df['oi']) / df['oi'].shift(1).replace(0, 1)
        # Requires config usually, hardcoding reasonable default for vectorization
        # or assuming the caller might adjust later.
        # Using 0.35 as per INI default read in original file
        df['roll_trap'] = (oi_drop_pct > 0.35) & (df['rvol_now'] > 1.5)

        return df

    def _compute_session_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        # IST timezone conversion
        if df.index.tz is None:
            ts_local = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            ts_local = df.index.tz_convert('Asia/Kolkata')

        minutes = ts_local.hour * 60 + ts_local.minute

        # Session: 09:15 to 15:30
        open_min = 9 * 60 + 15
        close_min = 15 * 60 + 30

        df['in_session'] = (minutes >= open_min) & (minutes <= close_min)

        # Near Open (first 30 mins)
        df['near_open'] = (minutes >= open_min) & (minutes < open_min + 30)

        # Near Close (last 30 mins)
        df['near_close'] = (minutes > close_min - 30) & (minutes <= close_min)

        # Mid Lunch (12:30 - 13:30 approx)
        df['mid_lunch'] = (minutes >= 12 * 60 + 30) & (minutes <= 13 * 60 + 30)

        return df

    def _merge_daily_futures(self, df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges daily futures data (buildup, oi_change, etc.) onto 15m bars.
        Joins on Date.
        """
        # Prepare 15m date column for join
        if df.index.tz is None:
            ts_local = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            ts_local = df.index.tz_convert('Asia/Kolkata')

        df['join_date'] = ts_local.date

        # Prepare daily df
        # daily_df should have 'trade_date' as datetime.date or similar
        # Ensure daily_df columns are ready
        cols_to_use = ['trade_date', 'buildup', 'oi_change_pct', 'day_change_pct', 'volume_shares_chg_pct']
        # Map to what INI expects:
        # daily_fut_buildup -> buildup
        # daily_opt_call_oi_chg_pct -> (Wait, daily_futures table doesn't have options data, but INI asks for it)
        # Note: The User prompt only mentioned daily_futures for 'buildup'.
        # Options data (Call OI) might be in a different table, but for now we map what we have.

        daily_subset = daily_df[cols_to_use].copy()
        daily_subset['trade_date'] = pd.to_datetime(daily_subset['trade_date']).dt.date

        daily_subset = daily_subset.rename(columns={
            'buildup': 'daily_fut_buildup',
            'oi_change_pct': 'daily_fut_oi_chg_pct', # Disambiguate
            'day_change_pct': 'daily_fut_price_chg_pct'
        })

        # Merge
        # Ensure index name is preserved or handled
        idx_name = df.index.name or 'ts'
        df_reset = df.reset_index()
        if idx_name not in df_reset.columns:
             # Fallback if it was unnamed and became 'index'
             df_reset = df_reset.rename(columns={'index': idx_name})

        merged = df_reset.merge(daily_subset, left_on='join_date', right_on='trade_date', how='left')
        merged = merged.set_index(idx_name) # Restore index

        # Cleanup
        if 'join_date' in merged.columns:
            del merged['join_date']
        if 'trade_date' in merged.columns:
            del merged['trade_date']

        return merged

    def _merge_external_metrics(self, df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges external metrics (like FNO.pcr) pivoted onto the timeframe.
        """
        # Metric df is long: ts, metric, val
        # Pivot to wide: ts, [metric_names]

        # Ensure timestamps match (assuming metrics are on same interval, e.g. 15m)
        # If metrics are 15m, direct join.

        pivoted = metrics_df.pivot(index='ts', columns='metric', values='val')

        # Join
        # Use asof merge or left join?
        # Since metrics should be aligned (15m to 15m), left join is safer.
        merged = df.join(pivoted, how='left')

        return merged

def load_daily_futures(symbol: str) -> pd.DataFrame:
    sql = """
        SELECT trade_date, buildup, oi_change_pct, day_change_pct, volume_shares_chg_pct
        FROM raw_ingest.daily_futures
        WHERE symbol = %s
        ORDER BY trade_date ASC
    """
    with get_db_connection() as conn:
        return pd.read_sql(sql, conn, params=(symbol,))

def load_external_metrics(symbol: str, start_ts: Any, end_ts: Any) -> pd.DataFrame:
    # Fetch metrics relevant to Flow Pillar
    # FNO.pcr.vol.chg_pct, FNO.mwpl.pct, PIVOT.r1.dist_pct, FNO.pcr.oi.chg_pct
    metrics = [
        "FNO.pcr.vol.chg_pct",
        "FNO.mwpl.pct",
        "PIVOT.r1.dist_pct",
        "FNO.pcr.oi.chg_pct",
        "daily_opt_call_oi_chg_pct", # From INI
        "daily_opt_put_oi_chg_pct"
    ]

    placeholders = ','.join(['%s'] * len(metrics))
    sql = f"""
        SELECT ts, metric, val
        FROM indicators.values
        WHERE symbol = %s
          AND metric IN ({placeholders})
          AND ts >= %s AND ts <= %s
    """
    params = [symbol] + metrics + [start_ts, end_ts]

    with get_db_connection() as conn:
        return pd.read_sql(sql, conn, params=tuple(params))
