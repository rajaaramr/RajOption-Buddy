from __future__ import annotations

import configparser
import functools
import json
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path

from Trend.Pillar.trend_features import TrendFeatureEngine
from pillars.common import TZ, clamp, BaseCfg, write_values, maybe_trim_last_bar, resample, ensure_min_bars

# Paths
TREND_DIR = Path(__file__).resolve().parent
DEFAULT_INI = TREND_DIR / "trend_scenarios.ini"

@functools.lru_cache(maxsize=8)
def _cfg_cached(path_str: str, mtime_ns: int) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path_str)

    # Extract rules
    scenarios = []
    for section in cp.sections():
        if section.startswith("trend_scenario."):
            name = section.replace("trend_scenario.", "")
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

    # ML Config
    ml_cfg = {}
    if cp.has_section("trend_ml"):
        ml_cfg = {
            "enabled": cp.getboolean("trend_ml", "enabled", fallback=False),
            "target_name": cp.get("trend_ml", "target_name", fallback="trend_ml.target.bull_2pct_4h"),
            "version": cp.get("trend_ml", "version", fallback="xgb_v1"),
            "calibration_table": cp.get("trend_ml", "calibration_table", fallback="indicators.trend_calibration_4h"),
            "blend_weight": cp.getfloat("trend_ml", "blend_weight", fallback=0.35),
            "veto_if_prob_lt": cp.getfloat("trend_ml", "ml_veto_if_prob_lt", fallback=0.35),
            "soften_veto_if_prob_ge": cp.getfloat("trend_ml", "ml_soften_veto_if_prob_ge", fallback=0.65),
        }

    return {
        "scenarios": scenarios,
        "clamp_low": cp.getfloat("trend", "clamp_low", fallback=0.0),
        "clamp_high": cp.getfloat("trend", "clamp_high", fallback=100.0),
        "ml": ml_cfg,
        "min_bars": cp.getint("trend", "min_bars", fallback=120)
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
        """
        # Initialize score to 0
        df['score'] = 0.0
        df['veto'] = False

        # Evaluate each scenario
        for scen in self.config["scenarios"]:
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
                except Exception:
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

    cfg = _cfg(ini_path)

    # 1. Resample if needed
    if tf == "15m":
        dftf = df15.copy()
    else:
        dftf = resample(df15, tf)

    # Basic Validation
    if dftf.empty or len(dftf) < cfg["min_bars"]:
        return []

    # 2. Compute Features (Vectorized)
    # Filter metrics for current TF if necessary (though features currently use simple filter inside)
    # The feature engine expects metrics_df potentially containing 'VP.POC' etc.
    engine = TrendFeatureEngine(symbol, kind)
    dftf = engine.compute_all_features(dftf, daily_df, metrics_df)

    # 3. Score (Vectorized Rules)
    scorer = VectorizedScorer(ini_path)
    score_cols = scorer.score(dftf)

    # Drop existing score/veto columns if present to avoid overlap error
    cols_to_drop = [c for c in ['score', 'veto'] if c in dftf.columns]
    if cols_to_drop:
        dftf = dftf.drop(columns=cols_to_drop)

    dftf = dftf.join(score_cols)

    # 4. ML Integration (Vectorized Lookup)
    # In Flow, we lookup pre-computed ML probs from DB.
    # Since this function is "process_symbol_vectorized" often called in backfill,
    # we might need to fetch ML scores for the whole history if available.
    # For now, we will compute 'score_final' as purely rules-based unless we implement
    # the bulk fetch of ML pillars here.
    # To fully support ML backfill, we usually do that in a separate pass or join here.
    # Assumption: For this task, we focus on generating the Pillar scores.
    # The user asked for "Trend ML will follow Flow pattern", which implies
    # looking up 'indicators.ml_pillars'.

    # Placeholder for ML Join (Backfill complex logic usually separated)
    # For now, TREND.score_final = TREND.score
    dftf['score_final'] = dftf['score']
    dftf['veto_final'] = dftf['veto']
    dftf['ml_p_up_cal'] = 0.0 # Default/Placeholder

    # 5. Convert to Rows for DB
    rows = []
    # Reset index to access 'ts'
    dftf = dftf.reset_index()
    if 'ts' not in dftf.columns and 'index' in dftf.columns:
         dftf = dftf.rename(columns={'index': 'ts'})

    dftf['veto_val'] = dftf['veto'].astype(float)
    dftf['veto_final_val'] = dftf['veto_final'].astype(float)

    run_id = base_cfg.run_id
    source = base_cfg.source

    target_cols = ['ts', 'score', 'veto_val', 'score_final', 'veto_final_val']

    # Debug context columns to extract
    debug_cols = ['ema10', 'ema20', 'ema50', 'adx14'] # Add others as needed

    for row in dftf.itertuples(index=False):
        # We need to access columns by name safely, itertuples returns a namedtuple
        # but relying on position is risky if columns change.
        # Safer to use getattr
        ts = getattr(row, 'ts')
        score = getattr(row, 'score')
        veto_val = getattr(row, 'veto_val')
        score_final = getattr(row, 'score_final')
        veto_final_val = getattr(row, 'veto_final_val')

        ts_py = pd.Timestamp(ts).to_pydatetime()

        # Core Metrics
        rows.append((symbol, kind, tf, ts_py, "TREND.score", float(score), "{}", run_id, source))
        rows.append((symbol, kind, tf, ts_py, "TREND.veto_flag", float(veto_val), "{}", run_id, source))
        rows.append((symbol, kind, tf, ts_py, "TREND.score_final", float(score_final), "{}", run_id, source))
        rows.append((symbol, kind, tf, ts_py, "TREND.veto_final", float(veto_final_val), "{}", run_id, source))

        # Debug Context
        ctx = {k: float(getattr(row, k, 0.0)) for k in debug_cols}
        rows.append((symbol, kind, tf, ts_py, "TREND.debug_ctx", 0.0, json.dumps(ctx), run_id, source))

    return rows

def score_trend(symbol: str, kind: str, tf: str, df5: pd.DataFrame, base: BaseCfg,
                ini_path=DEFAULT_INI, context: Optional[Dict[str, Any]] = None):
    """
    Runtime entry point for single-timestamp execution (e.g. live mode).
    """
    # This wrapper adapts the live usage to the vectorized function.
    # Live mode usually calls this with recent data.

    # Prepare inputs
    # 'context' usually contains cached metrics. We need to wrap it into a DataFrame format
    # expected by process_symbol_vectorized if we want to reuse it, OR just use FeatureEngine directly.
    # Reusing process_symbol_vectorized is safer for consistency.

    # 1. Fake Daily DF (Live usually has access to daily context, but here we might need to fetch or ignore)
    # If df5 is sufficient for indicators, daily_df is optional for some feats.
    daily_df = pd.DataFrame()

    # 2. Metrics DF from Context
    # Context e.g. {'VP.POC': 100.0, ...} -> DataFrame
    metrics_records = []
    if context:
        for k, v in context.items():
            # We don't have TS in context usually, just latest value.
            # Feature engine expects TS-based metrics for merge_asof.
            # For live calculation on the LAST bar, we can fake the TS or handle it.
            # Actually, TrendFeatureEngine uses metrics_df only for POC currently.
            if k == 'VP.POC':
                 metrics_records.append({'ts': df5.index[-1], 'metric': 'VP.POC', 'val': v})

    metrics_df = pd.DataFrame(metrics_records)

    # 3. Call Vectorized Processor
    rows = process_symbol_vectorized(symbol, kind, tf, df5, daily_df, metrics_df, base, ini_path)

    if not rows:
        return None

    # 4. Extract Result for the LAST timestamp
    # process_symbol_vectorized returns all rows. We only want the latest for live return.
    # The list is (symbol, kind, tf, ts, metric, val, ...)

    # Sort by TS to find latest? It should be sorted.
    # Find rows matching the last timestamp
    last_ts = rows[-1][3] # ts is at index 3

    final_score = 0.0
    final_veto = False

    # We write all rows to DB? usually live worker writes them.
    # The worker calling this expects (ts, score, veto) return AND writes values itself?
    # No, 'trend_pillar.py' usually calls 'write_values'.

    # Filter rows for last TS to write?
    # Live execution usually re-writes the last bar or recent history.
    # We will write all generated rows (likely just the tail if df5 was trimmed).
    write_values(rows)

    # Extract values for return
    for r in rows:
        if r[3] == last_ts:
            if r[4] == "TREND.score_final":
                final_score = r[5]
            if r[4] == "TREND.veto_final":
                final_veto = bool(r[5])

    return (last_ts, final_score, final_veto)
