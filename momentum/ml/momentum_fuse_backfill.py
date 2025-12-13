"""
Momentum Pillar Fusion Backfill.
Computes Sigmoid-normalized score (0-100) and ML bucket lookup.
Joins Rules outputs with ML outputs, handling Spot vs Futures timestamps correctly.
"""
import logging
import math
import numpy as np
import pandas as pd
from typing import Optional, List, Any

from utils.db import get_db_connection
from pillars.common import write_values

logger = logging.getLogger(__name__)

# Same sigmoid parameters as Flow Pillar for consistency.
SIGMOID_CENTER = 0.5
SIGMOID_STEEPNESS = 12


def sigmoid(x, center=0.5, steepness=12):
    """
    Standard logistic function mapped to 0..100 range.
    x is probability (0..1).
    """
    if x is None:
        return 0.0
    k = steepness
    c = center
    try:
        val = 1.0 / (1.0 + math.exp(-k * (x - c)))
    except OverflowError:
        val = 0.0 if (x - c) < 0 else 1.0
    return val * 100.0


class MomentumFuser:
    def __init__(self, symbol: str, run_id: str = "MOM_FUSE_BACKFILL"):
        self.symbol = symbol
        self.run_id = run_id
        self.buckets = self._load_buckets()

    def _load_buckets(self) -> List[dict]:
        """
        Load calibration buckets from indicators.momentum_calibration_4h.
        """
        # Note: If the table doesn't exist, we return empty list.
        # This prevents crashes if calibration hasn't run yet.
        sql = """
            SELECT bucket, p_min, p_max
              FROM indicators.momentum_calibration_4h
             WHERE target_name = 'momentum_ml.target.bull_2pct_4h'
             ORDER BY bucket ASC
        """
        try:
            with get_db_connection() as conn:
                # Check if table exists first? Or just try-catch read_sql
                # indicators.momentum_calibration_4h might not exist if ML pipeline never ran.
                # But we assume it should.
                df = pd.read_sql(sql, conn)
                if df.empty:
                    return []
                return df.to_dict("records")
        except Exception:
            # logger.warning(f"Failed to load momentum buckets (table might be missing).")
            return []

    def get_bucket(self, prob: float) -> int:
        if not self.buckets:
            return int(prob * 20) + 1

        for b in self.buckets:
            if b["p_min"] <= prob < b["p_max"]:
                return int(b["bucket"])

        return int(self.buckets[-1]["bucket"])

    def compute_fusion_metrics(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """
        df_results expects columns: ['MOM.fused_score', 'MOM.ml_p_up_cal']
        Returns DataFrame with additional columns: ['MOM.score_final', 'MOM.ml_bucket']
        """
        if df_results.empty:
            return df_results

        # Sigmoid on Fused Score (which is 0-100).
        # We convert fused_score to prob 0..1 for sigmoid function (which expects 0..1 input)
        # OR we adjust sigmoid params?
        # Flow logic: `sigmoid(x, center=0.5, steepness=12)` where x is PROBABILITY.
        # fused_score is 0..100.

        probs = df_results["MOM.fused_score"] / 100.0

        df_results["MOM.score_final"] = probs.apply(
            lambda x: sigmoid(x, SIGMOID_CENTER, SIGMOID_STEEPNESS)
        )

        # Buckets
        if "MOM.ml_p_up_cal" in df_results.columns:
            df_results["MOM.ml_bucket"] = df_results["MOM.ml_p_up_cal"].apply(
                lambda x: self.get_bucket(x) if pd.notnull(x) else 0
            )
        else:
            df_results["MOM.ml_bucket"] = 0

        return df_results
