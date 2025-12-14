import unittest
import pandas as pd
import numpy as np
from Trend.Pillar.trend_features import TrendFeatureEngine

class TestTrendFeatures(unittest.TestCase):
    def test_macd_features(self):
        """Verify correct MACD and hist_diff calculation"""
        # Create synthetic data: Price rising then falling
        close = [10, 11, 12, 13, 14, 13, 12, 11, 10, 9] * 5  # 50 bars
        df = pd.DataFrame({'close': close, 'high': close, 'low': close, 'open': close})

        # Calculate features
        engine = TrendFeatureEngine("TEST", "spot")
        res = engine.compute_all_features(df, None, None, cfg={})

        # MACD Line = EMA12 - EMA26
        # Signal = EMA9(MACD Line)
        # Hist = MACD Line - Signal
        # Hist Diff = Hist - Hist.shift(1)

        # Check columns exist
        self.assertIn('macd_line', res.columns)
        self.assertIn('macd_sig', res.columns)
        self.assertIn('hist', res.columns)
        self.assertIn('hist_diff', res.columns)

        # Verify calculation logic manually for last few rows
        last_hist = res['hist'].iloc[-1]
        prev_hist = res['hist'].iloc[-2]
        last_diff = res['hist_diff'].iloc[-1]

        # Tolerance for float
        self.assertAlmostEqual(last_diff, last_hist - prev_hist, places=5)

        print(f"MACD Hist: {last_hist}, Prev: {prev_hist}, Diff: {last_diff}")

if __name__ == '__main__':
    unittest.main()
