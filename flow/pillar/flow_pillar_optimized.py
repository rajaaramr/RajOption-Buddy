from __future__ import annotations

import configparser
import functools
import json
import ast
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path

from flow.pillar.flow_features_optimized import FlowFeatureEngine
from pillars.common import TZ, clamp, BaseCfg, write_values, maybe_trim_last_bar

# Paths
FLOW_DIR = Path(__file__).resolve().parent
DEFAULT_INI = FLOW_DIR / "flow_scenarios.ini"

@functools.lru_cache(maxsize=8)
def _cfg_cached(path_str: str, mtime_ns: int) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path_str)

    # Extract rules
    scenarios = []
    for section in cp.sections():
        if section.startswith("flow_scenario."):
            name = section.replace("flow_scenario.", "")
            when = cp.get(section, "when", fallback="").strip()
            score = cp.getfloat(section, "score", fallback=0.0)
            bonus_when = cp.get(section, "bonus_when", fallback="").strip()
            bonus = cp.getfloat(section, "bonus", fallback=0.0)
            set_veto = cp.getboolean(section, "set_veto", fallback=False)

            scenarios.append({
                "name": name,
                "when": when,
                "score": score,
                "bonus_when": bonus_when,
                "bonus": bonus,
                "set_veto": set_veto
            })

    return {
        "scenarios": scenarios,
        "clamp_low": cp.getfloat("flow", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("flow", "clamp_high", fallback=100.0),
    }

def _cfg(path: Path) -> dict:
    mtime = path.stat().st_mtime_ns if path.exists() else 0
    return _cfg_cached(str(path), mtime)

class VectorizedScorer:
    def __init__(self, ini_path: Path):
        self.config = _cfg(ini_path)

    def score(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies scoring rules to the dataframe.
        Returns:
            df_scores: DataFrame with 'score', 'veto' columns.
            df_debug: DataFrame with individual rule firings (optional).
        """
        # Initialize score to 0
        df['score'] = 0.0
        df['veto'] = False

        # We also track debug info (which rules fired)
        # For performance, maybe skip detail log unless requested
        # But 'indicators.values' usually expects a JSON of context.
        # We'll build a simplified context column.

        # Evaluate each scenario
        for scen in self.config["scenarios"]:
            name = scen["name"]

            # Main condition
            if scen["when"]:
                try:
                    mask = df.eval(scen["when"])
                    if isinstance(mask, pd.Series):
                        # Apply score
                        df.loc[mask, 'score'] += scen["score"]

                        # Apply veto
                        if scen["set_veto"]:
                            df.loc[mask, 'veto'] = True
                except Exception as e:
                    # Log error but continue?
                    # print(f"Error evaluating rule {name}: {e}")
                    pass

            # Bonus condition
            if scen["bonus_when"]:
                try:
                    mask_bonus = df.eval(scen["bonus_when"])
                    if isinstance(mask_bonus, pd.Series):
                        df.loc[mask_bonus, 'score'] += scen["bonus"]
                except Exception:
                    pass

        # Clamp scores
        df['score'] = df['score'].clip(self.config["clamp_low"], self.config["clamp_high"])

        return df[['score', 'veto']]

def process_symbol_vectorized(
    symbol: str,
    kind: str,
    tf: str,
    df15: pd.DataFrame,
    daily_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    base_cfg: BaseCfg,
    ini_path: Path = DEFAULT_INI
) -> List[tuple]:

    # 1. Resample if needed (Vectorized resample)
    if tf == "15m":
        dftf = df15.copy()
    else:
        # Import resample logic or implement simple one
        # For Flow, we need simple resample but logic like 'voi' depends on 15m granularity often?
        # No, standard is resample candles -> compute features -> score.
        from pillars.common import resample
        dftf = resample(df15, tf)

    if dftf.empty:
        return []

    # 2. Compute Features (Vectorized)
    engine = FlowFeatureEngine(symbol, kind)
    dftf = engine.compute_all_features(dftf, daily_df, metrics_df)

    # 3. Score (Vectorized)
    scorer = VectorizedScorer(ini_path)
    score_cols = scorer.score(dftf)
    dftf = dftf.join(score_cols)

    # 4. Convert to Rows for DB
    rows = []
    # Vectorized conversion to list of tuples is faster than iterrows
    # Reset index to access 'ts'
    dftf = dftf.reset_index()

    # We need to map boolean True/False to 1.0/0.0 for DB
    dftf['veto_val'] = dftf['veto'].astype(float)

    # Required columns
    # symbol, kind, tf, ts, metric, val, ctx, run_id, source

    # We can iterate now, it's cheaper than doing logic in the loop
    # Limit columns to iterate to save memory
    target_cols = ['ts', 'score', 'veto_val']

    # Prepare common fields
    run_id = base_cfg.run_id
    source = base_cfg.source

    # Bulk create rows
    # Metric: FLOW.score
    # Metric: FLOW.veto_flag
    # Metric: FLOW.score_final (same as score for now, ignoring ML fusion in this pass if purely rules)

    for row in dftf[target_cols].itertuples(index=False):
        ts, score, veto = row
        # Ensure ts is datetime
        ts_py = pd.Timestamp(ts).to_pydatetime()

        rows.append((symbol, kind, tf, ts_py, "FLOW.score", float(score), "{}", run_id, source))
        rows.append((symbol, kind, tf, ts_py, "FLOW.veto_flag", float(veto), "{}", run_id, source))
        rows.append((symbol, kind, tf, ts_py, "FLOW.score_final", float(score), "{}", run_id, source))

    return rows
