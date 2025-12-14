from __future__ import annotations

import configparser
import functools
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from Trend.Pillar.trend_features import TrendFeatureEngine
from pillars.common import TZ, clamp, BaseCfg, write_values, maybe_trim_last_bar, resample, ensure_min_bars
from utils.db import get_db_connection

# Paths
TREND_DIR = Path(__file__).resolve().parent
DEFAULT_INI = TREND_DIR / "trend_scenarios.ini"

@functools.lru_cache(maxsize=8)
def _cfg_cached(path_str: str, mtime_ns: int) -> dict:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path_str)

    # Parse list of scenarios
    scenario_list_raw = cp.get("trend_scenarios", "list", fallback="")
    allowed_scenarios = {s.strip() for s in scenario_list_raw.replace("\n", ",").split(",") if s.strip()}

    # Extract rules (only if in list)
    scenarios = []
    for section in cp.sections():
        if section.startswith("trend_scenario."):
            name = section.replace("trend_scenario.", "")

            # TRD-Fix-05: Enforce list control
            if name not in allowed_scenarios:
                continue

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

    # TRD-Fix-07: Parse write_scenarios_debug once
    write_debug = cp.getboolean("trend", "write_scenarios_debug", fallback=False)

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
        "min_bars": cp.getint("trend", "min_bars", fallback=120),
        "write_debug": write_debug
    }

def _cfg(path: Path) -> dict:
    mtime = path.stat().st_mtime_ns if path.exists() else 0
    return _cfg_cached(str(path), mtime)

class VectorizedScorer:
    def __init__(self, ini_path: Path):
        self.config = _cfg(ini_path)

    def score(self, df: pd.DataFrame, track_scenarios: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Applies scoring rules to the dataframe.
        Returns:
            df_scores: DataFrame with 'score', 'veto' columns.
            fired_scenarios: Dictionary of {scenario_name: Series(score)} if track_scenarios=True
        """
        # Initialize score to 0
        df['score'] = 0.0
        df['veto'] = False

        fired_scenarios = {}

        # Evaluate each scenario
        for scen in self.config["scenarios"]:
            name = scen["name"]
            # TRD-Fix-08: Use numexpr engine
            # Main condition
            if scen["when"]:
                try:
                    mask = df.eval(scen["when"], engine='numexpr')
                    if isinstance(mask, pd.Series):
                        # Apply score
                        score_val = scen["score"]
                        df.loc[mask, 'score'] += score_val

                        # Apply veto
                        if scen["set_veto"]:
                            df.loc[mask, 'veto'] = True

                        # Track firing
                        if track_scenarios:
                            # Initialize with 0
                            s_series = pd.Series(0.0, index=df.index)
                            s_series.loc[mask] = score_val
                            if name in fired_scenarios:
                                fired_scenarios[name] += s_series
                            else:
                                fired_scenarios[name] = s_series
                except Exception:
                    # Fallback to default engine if numexpr fails (e.g. missing op)
                    try:
                        mask = df.eval(scen["when"])
                        if isinstance(mask, pd.Series):
                            score_val = scen["score"]
                            df.loc[mask, 'score'] += score_val
                            if scen["set_veto"]:
                                df.loc[mask, 'veto'] = True
                            if track_scenarios:
                                s_series = pd.Series(0.0, index=df.index)
                                s_series.loc[mask] = score_val
                                if name in fired_scenarios:
                                    fired_scenarios[name] += s_series
                                else:
                                    fired_scenarios[name] = s_series
                    except Exception:
                        pass

            # Bonus condition
            if scen["bonus_when"]:
                try:
                    mask_bonus = df.eval(scen["bonus_when"], engine='numexpr')
                    if isinstance(mask_bonus, pd.Series):
                        bonus_val = scen["bonus"]
                        df.loc[mask_bonus, 'score'] += bonus_val

                        if track_scenarios:
                            # TRD-Fix-12: Use .bonus suffix
                            bonus_name = f"{name}.bonus"
                            b_series = pd.Series(0.0, index=df.index)
                            b_series.loc[mask_bonus] = bonus_val
                            fired_scenarios[bonus_name] = b_series
                except Exception:
                    try:
                        mask_bonus = df.eval(scen["bonus_when"])
                        if isinstance(mask_bonus, pd.Series):
                            bonus_val = scen["bonus"]
                            df.loc[mask_bonus, 'score'] += bonus_val

                            if track_scenarios:
                                bonus_name = f"{name}.bonus"
                                b_series = pd.Series(0.0, index=df.index)
                                b_series.loc[mask_bonus] = bonus_val
                                fired_scenarios[bonus_name] = b_series
                    except Exception:
                        pass

        # Clamp scores
        df['score'] = df['score'].clip(self.config["clamp_low"], self.config["clamp_high"])

        return df[['score', 'veto']], fired_scenarios

# -----------------------------
# ML Helpers
# -----------------------------
@functools.lru_cache(maxsize=1)
def _fetch_calibration_data(table_name: str) -> List[dict]:
    """
    Loads calibration buckets (e.g. indicators.trend_calibration_4h).
    Expected schema: min_prob, max_prob, realized_up_rate
    """
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            # We assume columns: min_prob, max_prob, realized_up_rate
            cur.execute(f"SELECT min_prob, max_prob, realized_up_rate FROM {table_name}")
            rows = cur.fetchall()
            # Convert to list of dicts
            return [
                {"min_p": float(r[0]), "max_p": float(r[1]), "cal_p": float(r[2])}
                for r in rows
            ]
    except Exception:
        # Fallback or log error
        return []

def _calibrate_prob(raw_p: float, buckets: List[dict]) -> float:
    """
    Maps raw probability to calibrated probability using buckets.
    """
    if not buckets:
        return raw_p

    # Simple linear search (buckets are few)
    for b in buckets:
        if b["min_p"] <= raw_p < b["max_p"]:
            return b["cal_p"]

    # Handle edges
    if raw_p >= buckets[-1]["max_p"]:
        return buckets[-1]["cal_p"]
    if raw_p < buckets[0]["min_p"]:
        return buckets[0]["cal_p"]

    return raw_p

def _fetch_ml_scores(symbol: str, kind: str, tf: str, target: str, version: str,
                     start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Fetches prob_up from indicators.ml_pillars.
    """
    try:
        query = """
            SELECT ts, prob_up
            FROM indicators.ml_pillars
            WHERE symbol = %s
              AND market_type = %s
              AND interval = %s
              AND pillar = 'trend'
              AND target_name = %s
              AND version = %s
              AND ts >= %s
              AND ts <= %s
            ORDER BY ts ASC
        """
        with get_db_connection() as conn:
            df = pd.read_sql(query, conn, params=(symbol, kind, tf, target, version, start_ts, end_ts), parse_dates=['ts'])
            if not df.empty:
                df = df.set_index('ts').sort_index()
            return df
    except Exception:
        return pd.DataFrame()

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
    # TRD-Fix-03: Pass cfg to feature engine
    # Let's re-read config efficiently.
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    cp.read(ini_path)
    trend_params = {k: v for k, v in cp.items("trend")} if cp.has_section("trend") else {}
    # Convert numeric
    for k in trend_params:
        try: trend_params[k] = float(trend_params[k])
        except: pass
        try: trend_params[k] = int(trend_params[k])
        except: pass

    dftf = engine.compute_all_features(dftf, daily_df, metrics_df, cfg=trend_params)

    # 3. Score (Vectorized Rules)
    # TRD-Fix-07: Use cached write_debug
    write_debug = cfg.get("write_debug", False)

    scorer = VectorizedScorer(ini_path)
    score_cols, fired_scenarios = scorer.score(dftf, track_scenarios=write_debug)

    # Drop existing score/veto columns if present to avoid overlap error
    cols_to_drop = [c for c in ['score', 'veto'] if c in dftf.columns]
    if cols_to_drop:
        dftf = dftf.drop(columns=cols_to_drop)

    dftf = dftf.join(score_cols)

    # If tracking scenarios, add them to dftf for row iteration or join?
    # dftf usually wide.
    if write_debug and fired_scenarios:
        for name, ser in fired_scenarios.items():
            col_name = f"__scen_{name}"
            dftf[col_name] = ser

    # 4. ML Integration (Vectorized Lookup)
    ml_cfg = cfg["ml"]

    # Default values
    dftf['ml_prob_raw'] = 0.0
    dftf['ml_p_up_cal'] = 0.0
    dftf['ml_p_down_cal'] = 0.0
    dftf['score_final'] = dftf['score']
    dftf['veto_final'] = dftf['veto']
    dftf['ml_ctx'] = "{}"

    if ml_cfg.get("enabled", False):
        start_ts = dftf.index[0]
        end_ts = dftf.index[-1]

        target = ml_cfg["target_name"]
        version = ml_cfg["version"]

        # Fetch ML scores
        ml_df = _fetch_ml_scores(symbol, kind, tf, target, version, start_ts, end_ts)

        if not ml_df.empty:
             # Merge (TRD-Fix-10: safer join)
             # Use merge_asof if indices are close but not exact, or simple join if exact.
             # ML data usually 15m aligned.
             # Ensure indices are timezone aware/naive consistent.
             # We assume both are datetime indexes.

             # Check index overlap
             # dftf = dftf.join(ml_df, how='left')

             # Better: merge_asof with small tolerance to handle slight clock skews
             # Reset index for merge
             dftf_reset = dftf.reset_index()
             if 'ts' not in dftf_reset.columns and 'index' in dftf_reset.columns:
                 dftf_reset = dftf_reset.rename(columns={'index': 'ts'})

             ml_df_reset = ml_df.reset_index().rename(columns={'index': 'ts'})

             # Sort keys
             dftf_reset = dftf_reset.sort_values('ts')
             ml_df_reset = ml_df_reset.sort_values('ts')

             # Merge asof
             merged = pd.merge_asof(
                 dftf_reset,
                 ml_df_reset[['ts', 'prob_up']],
                 on='ts',
                 tolerance=pd.Timedelta("5m"), # 5m tolerance
                 direction='nearest'
             )

             # Restore index
             dftf = merged.set_index('ts')

             # Fill missing ML with default (e.g. 0.5 or 0) - or handle logic.
             # If missing, we revert to rule score?
             # Let's fill with NaN and handle below

             # Load Calibration
             cal_table = ml_cfg.get("calibration_table", "")
             buckets = _fetch_calibration_data(cal_table) if cal_table else []

             # Vectorized Calibration (apply per row is slow, but buckets are small)
             # We can use pd.cut or apply. Given small buckets, apply is fine or a vectorized lookup.
             # Let's use apply for simplicity now, or optimize if bucket counts grow.
             if 'prob_up' in dftf.columns:
                  # Calculate Calibrated Probs
                  # Handle NaNs (missing ML) -> Default to neutral 0.5 or skip blend?
                  # If NaN, we likely shouldn't veto or boost.

                  def get_cal_prob(p):
                      if pd.isna(p): return 0.5 # Neutral
                      return _calibrate_prob(p, buckets)

                  dftf['ml_prob_raw'] = dftf['prob_up'].fillna(0.5)
                  dftf['ml_p_up_cal'] = dftf['prob_up'].apply(get_cal_prob)
                  dftf['ml_p_down_cal'] = 1.0 - dftf['ml_p_up_cal']

                  # Blending
                  # base_prob = TREND.score / 100
                  # final_prob = (1 - w) * base_prob + w * ml_p_up_cal
                  w = ml_cfg.get("blend_weight", 0.0)
                  w = max(0.0, min(1.0, w))

                  base_prob = dftf['score'] / 100.0
                  final_prob = (1.0 - w) * base_prob + w * dftf['ml_p_up_cal']
                  dftf['score_final'] = final_prob * 100.0
                  dftf['score_final'] = dftf['score_final'].clip(cfg["clamp_low"], cfg["clamp_high"])

                  # Veto Logic
                  # TRD-Fix-06: ML veto softening
                  # If final_prob < veto_if_prob_lt -> Force Veto
                  # If final_prob >= soften_veto_if_prob_ge -> Soften Veto (Clear Rules Veto)
                  # Else -> Keep Rules Veto

                  veto_hard_thresh = ml_cfg.get("veto_if_prob_lt", 0.35)
                  veto_soft_thresh = ml_cfg.get("soften_veto_if_prob_ge", 0.65)

                  has_ml = dftf['prob_up'].notna()

                  # 1. Hard Veto (Adds new veto)
                  ml_hard_veto = (final_prob < veto_hard_thresh) & has_ml

                  # 2. Soften Veto (Removes existing veto)
                  ml_soft_veto = (final_prob >= veto_soft_thresh) & has_ml

                  # Combine:
                  # Start with Rules Veto
                  current_veto = dftf['veto'].copy()

                  # Apply Softening (If ML says strong buy, ignore weak rule vetoes?)
                  # "ml_soften_veto_if_prob_ge" implies un-vetoing.
                  current_veto = current_veto & (~ml_soft_veto)

                  # Apply Hardening (If ML says strong sell, force veto)
                  dftf['veto_final'] = current_veto | ml_hard_veto

                  # Construct ML Context JSON
                  # We need a vectorized way to create JSON string?
                  # Or just do it in the loop below.
                  # Doing it in loop is safer.

        # Drop temp columns if needed
        cols_to_drop = ['prob_up']
        dftf = dftf.drop(columns=[c for c in cols_to_drop if c in dftf.columns])

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

    # Debug context columns to extract (TRD-07 requirements)
    debug_cols = [
        'ema10', 'ema20', 'ema50',
        'adx14',
        'slope_short', 'slope_mid', 'slope_long',
        'squeeze_flag',
        'rsi_now',
        'dip_gt_dim'
    ]

    # Identify scenario columns
    scen_cols = [c for c in dftf.columns if c.startswith("__scen_")] if write_debug else []

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

        # TRD-15 Metrics
        # TREND.ml_p_up_cal, TREND.ml_p_down_cal
        # TREND.ml_ctx

        # Extract ML values
        ml_p_up = getattr(row, 'ml_p_up_cal', 0.0)
        ml_p_down = getattr(row, 'ml_p_down_cal', 0.0)
        ml_prob_raw = getattr(row, 'ml_prob_raw', 0.0)

        rows.append((symbol, kind, tf, ts_py, "TREND.ml_p_up_cal", float(ml_p_up), "{}", run_id, source))
        rows.append((symbol, kind, tf, ts_py, "TREND.ml_p_down_cal", float(ml_p_down), "{}", run_id, source))
        rows.append((symbol, kind, tf, ts_py, "TREND.score_final", float(score_final), "{}", run_id, source))
        rows.append((symbol, kind, tf, ts_py, "TREND.veto_final", float(veto_final_val), "{}", run_id, source))

        # Build ML Context
        ml_ctx = {
            "prob_raw": float(ml_prob_raw),
            "cal_prob": float(ml_p_up),
            "target_name": ml_cfg.get("target_name", ""),
            "version": ml_cfg.get("version", ""),
            "blend_weight": ml_cfg.get("blend_weight", 0.0)
        }
        rows.append((symbol, kind, tf, ts_py, "TREND.ml_ctx", 0.0, json.dumps(ml_ctx), run_id, source))

        # Debug Context
        ctx = {}
        for k in debug_cols:
            val = getattr(row, k, 0.0)
            # handle boolean
            if isinstance(val, (bool, np.bool_)):
                val = 1.0 if val else 0.0
            ctx[k] = float(val)

        rows.append((symbol, kind, tf, ts_py, "TREND.debug_ctx", 0.0, json.dumps(ctx), run_id, source))

        # Scenario Debugging
        if write_debug:
            for sc in scen_cols:
                val = getattr(row, sc, 0.0)
                if val != 0:
                    # Strip prefix __scen_
                    real_name = sc.replace("__scen_", "")
                    rows.append((symbol, kind, tf, ts_py, f"TREND.scenario.{real_name}", float(val), "{}", run_id, source))

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
