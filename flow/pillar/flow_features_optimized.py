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

    def _generate_synthetic_daily(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        """
        Generates synthetic daily data (buildup, OI/Price change) from 15m candles
        if the real daily_futures data is missing.
        """
        # If OI is missing (Spot), we cannot generate Buildup or OI change.
        if 'oi' not in df_15m.columns:
            return pd.DataFrame(columns=['trade_date', 'buildup', 'oi_change_pct', 'day_change_pct'])

        # Ensure UTC-localized index for resampling logic
        if df_15m.index.tz is None:
            df_work = df_15m.tz_localize('UTC')
        else:
            df_work = df_15m.copy()

        # Convert to IST for correct "Day" boundaries
        df_ist = df_work.tz_convert('Asia/Kolkata')

        # Resample to Daily (1D) based on IST days
        # Close = last close, OI = last OI, Volume = sum
        daily = df_ist.resample('1D').agg({
            'close': 'last',
            'oi': 'last',
            'volume': 'sum'
        }).dropna(subset=['close'])

        # Shift to get previous day values
        prev = daily.shift(1)

        # Calculate Percentage Changes
        # Avoid division by zero
        price_chg_pct = (daily['close'] - prev['close']) / prev['close'].replace(0, np.nan) * 100.0
        oi_chg_pct = (daily['oi'] - prev['oi']) / prev['oi'].replace(0, np.nan) * 100.0

        # Infer Buildup String
        # Price > Prev AND OI > Prev -> Long Build Up
        # Price > Prev AND OI < Prev -> Short Covering
        # Price < Prev AND OI > Prev -> Short Build Up
        # Price < Prev AND OI < Prev -> Long Unwind

        conditions = [
            (price_chg_pct > 0) & (oi_chg_pct > 0),
            (price_chg_pct > 0) & (oi_chg_pct < 0),
            (price_chg_pct < 0) & (oi_chg_pct > 0),
            (price_chg_pct < 0) & (oi_chg_pct < 0)
        ]
        choices = ["Long Build Up", "Short Covering", "Short Build Up", "Long Unwind"]

        daily['buildup'] = np.select(conditions, choices, default="Neutral")
        daily['oi_change_pct'] = oi_chg_pct.fillna(0.0)
        daily['day_change_pct'] = price_chg_pct.fillna(0.0)
        daily['trade_date'] = daily.index.date

        # Return dataframe matching daily_df schema subset
        return daily[['trade_date', 'buildup', 'oi_change_pct', 'day_change_pct']]

    def _merge_daily_futures(self, df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges daily futures data (buildup, oi_change, etc.) onto 15m bars.
        Joins on Date. Fills gaps with synthetic data if needed.
        """
        # Generate synthetic daily coverage from the 15m data itself
        synthetic_daily = self._generate_synthetic_daily(df)

        # Combine Real and Synthetic
        # We prioritize 'daily_df' (Real) where it exists.
        # merge: Real left join Synthetic? No, we want full coverage.
        # Concat and dedupe keeping real?

        # Prepare real DF
        real_subset = daily_df[['trade_date', 'buildup', 'oi_change_pct', 'day_change_pct']].copy()
        real_subset['trade_date'] = pd.to_datetime(real_subset['trade_date']).dt.date
        real_subset['source'] = 'real'

        synthetic_daily['source'] = 'synthetic'

        # Concatenate and keep 'real' if duplicates exist on trade_date
        combined = pd.concat([real_subset, synthetic_daily], ignore_index=True)
        # Drop duplicates on date, keeping first (which should be 'real' if we sort)
        # But concat order matters.
        # Actually, let's merge.

        # Re-index by date for easier combining
        real_indexed = real_subset.set_index('trade_date')
        syn_indexed = synthetic_daily.set_index('trade_date')

        # Combine: Real.combine_first(Synthetic) -> Fills Real NaNs with Synthetic values
        # This aligns indices (dates) and fills gaps.
        final_daily = real_indexed.combine_first(syn_indexed).reset_index()

        # Prepare 15m date column for join
        if df.index.tz is None:
            ts_local = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else:
            ts_local = df.index.tz_convert('Asia/Kolkata')

        df['join_date'] = ts_local.date

        # Use final_daily (Combined Real + Synthetic)
        daily_use = final_daily.rename(columns={
            'buildup': 'daily_fut_buildup',
            'oi_change_pct': 'daily_fut_oi_chg_pct',
            'day_change_pct': 'daily_fut_price_chg_pct'
        })

        # Merge
        # Ensure index name is preserved or handled
        idx_name = df.index.name or 'ts'
        df_reset = df.reset_index()
        if idx_name not in df_reset.columns:
             # Fallback if it was unnamed and became 'index'
             df_reset = df_reset.rename(columns={'index': idx_name})

        merged = df_reset.merge(daily_use, left_on='join_date', right_on='trade_date', how='left')
        merged = merged.set_index(idx_name) # Restore index

        # Cleanup
        if 'join_date' in merged.columns:
            del merged['join_date']
        if 'trade_date' in merged.columns:
            del merged['trade_date']
        if 'source' in merged.columns:
            del merged['source']

        return merged

    def _merge_external_metrics(self, df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges external metrics (like FNO.pcr) pivoted onto the timeframe.
        """
        # Metric df is long: ts, metric, val, interval
        # Pivot to wide: ts, [metric_names]

        # Ensure timestamps match (assuming metrics are on same interval, e.g. 15m)
        # If metrics are 15m, direct join.

        # Drop duplicates just in case
        metrics_dedup = metrics_df.drop_duplicates(subset=['ts', 'metric'])

        pivoted = metrics_dedup.pivot(index='ts', columns='metric', values='val')

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
    # Select interval as well to filter later
    sql = f"""
        SELECT ts, metric, val, interval
        FROM indicators.values
        WHERE symbol = %s
          AND metric IN ({placeholders})
          AND ts >= %s AND ts <= %s
    """
    params = [symbol] + metrics + [start_ts, end_ts]

    with get_db_connection() as conn:
        return pd.read_sql(sql, conn, params=tuple(params))
