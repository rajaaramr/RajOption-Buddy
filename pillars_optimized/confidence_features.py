# pillars_optimized/confidence_features.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import configparser
import numpy as np
import pandas as pd
from datetime import timedelta

@dataclass
class ConfidenceConfig:
    tfs: List[str]
    tf_weights: Dict[str, float]

    lookback_days: int

    # HMM / regime
    states: int
    barrier_atr_k: float
    horizon_bars: Dict[str, int]
    laplace_alpha: float

    # thresholds
    oi_z_hi: float
    vp_poc_atr_thr: float
    squeeze_bw_pct: float
    bb_score_hi: float

    # penalties / bonuses (logit lambda)
    lambda_poc: float
    lambda_crowd: float
    lambda_squeeze: float

    # OI integration
    oi_integration: str
    bayes_k: float
    bayes_m: float
    mtf_oi_weights: Dict[str, float]
    mtf_align_gate: float
    mtf_conflict_cap: float

    # identity
    run_id: str
    source: str


@dataclass
class OIConfig:
    window: int
    persistence_min: float
    voi_eff_min: float
    oi_turnover_low: float
    oi_turnover_high: float
    roll_dampen: float
    basis_confirm: int
    squeeze_bw_pct: float
    roll_pdrop: float
    roll_z: float

def load_confidence_config(path: str) -> Tuple[ConfidenceConfig, OIConfig]:
    cfg = configparser.ConfigParser()
    cfg.read(path)

    c = cfg["confidence"]
    o = cfg["oi"]

    def parse_weights(s: str) -> Dict[str, float]:
        out = {}
        for part in s.split(","):
            tf, w = part.split(":")
            out[tf.strip()] = float(w)
        return out

    conf = ConfidenceConfig(
        tfs=[tf.strip() for tf in c.get("tfs").split(",")],
        tf_weights=parse_weights(c.get("tf_weights", "")),
        lookback_days=c.getint("lookback_days", 180),

        states=c.getint("states", 3),
        barrier_atr_k=c.getfloat("barrier_atr_k", 1.0),
        horizon_bars=parse_weights(c.get("horizon_bars")),
        laplace_alpha=c.getfloat("laplace_alpha", 2.0),

        oi_z_hi=c.getfloat("oi_z_hi", 1.5),
        vp_poc_atr_thr=c.getfloat("vp_poc_atr_thr", 0.20),
        squeeze_bw_pct=c.getfloat("squeeze_bw_pct", 20.0),
        bb_score_hi=c.getfloat("bb_score_hi", 6.5),

        lambda_poc=c.getfloat("lambda_poc", 0.25),
        lambda_crowd=c.getfloat("lambda_crowd", 0.35),
        lambda_squeeze=c.getfloat("lambda_squeeze", 0.20),

        oi_integration=c.get("oi_integration", "bayes"),
        bayes_k=c.getfloat("bayes_k", 0.50),
        bayes_m=c.getfloat("bayes_m", 0.75),
        mtf_oi_weights=parse_weights(c.get("mtf_oi_weights")),
        mtf_align_gate=c.getfloat("mtf_align_gate", 0.60),
        mtf_conflict_cap=c.getfloat("mtf_conflict_cap", 0.60),

        run_id=c.get("run_id", "conf_run"),
        source=c.get("source", "conf_hmm_bayes"),
    )

    oi = OIConfig(
        window=o.getint("window", 20),
        persistence_min=o.getfloat("persistence_min", 0.60),
        voi_eff_min=o.getfloat("voi_eff_min", 0.05),
        oi_turnover_low=o.getfloat("oi_turnover_low", 0.10),
        oi_turnover_high=o.getfloat("oi_turnover_high", 1.20),
        roll_dampen=o.getfloat("roll_dampen", 0.50),
        basis_confirm=o.getint("basis_confirm", 1),
        squeeze_bw_pct=o.getfloat("squeeze_bw_pct", 20.0),
        roll_pdrop=o.getfloat("roll_pdrop", 0.30),
        roll_z=o.getfloat("roll_z", 2.0),
    )

    return conf, oi

def compute_price_vol_conf_features(
    df: pd.DataFrame,
    conf_cfg: ConfidenceConfig
) -> pd.DataFrame:
    out = df.copy()
    out["hmm_state"] = 1
    out["hmm_prob_trend_up"] = 0.33
    out["hmm_prob_trend_down"] = 0.33
    out["hmm_regime_label"] = "range"
    out["squeeze_flag"] = out["squeeze_bw_pct"] <= conf_cfg.squeeze_bw_pct
    out["squeeze_score"] = np.clip(
        (conf_cfg.squeeze_bw_pct - out["squeeze_bw_pct"]) / conf_cfg.squeeze_bw_pct,
        0.0, 1.0
    )
    out["vp_poc_trap_flag"] = out["vp_poc_dist_atr"] <= conf_cfg.vp_poc_atr_thr
    out["crowd_flag"] = False
    out["trap_flag"] = (
        out["vp_poc_trap_flag"] &
        out["squeeze_flag"] &
        out["crowd_flag"]
    )
    base_logit = (out["hmm_prob_trend_up"] - out["hmm_prob_trend_down"])
    base_logit -= conf_cfg.lambda_poc * out["vp_poc_trap_flag"].astype(float)
    base_logit -= conf_cfg.lambda_crowd * out["crowd_flag"].astype(float)
    base_logit -= conf_cfg.lambda_squeeze * out["squeeze_flag"].astype(float)
    out["conf_price_vol_raw"] = base_logit
    out["conf_price_vol"] = 1.0 / (1.0 + np.exp(-base_logit))
    return out

def compute_oi_conf_features(
    df: pd.DataFrame,
    conf_cfg: ConfidenceConfig,
    oi_cfg: OIConfig
) -> pd.DataFrame:
    out = df.copy()
    out["oi_turnover"] = out["oi_turnover"]
    out["voi_eff"] = out["voi_eff"]
    out["low_liq_flag"] = (
        (out["oi_turnover"] < oi_cfg.oi_turnover_low) |
        (out["oi_turnover"] > oi_cfg.oi_turnover_high)
    )
    out["roll_flag"] = (
        (out["oi_change_pct"].abs() > oi_cfg.roll_pdrop) &
        (out["oi_zscore"].abs() > oi_cfg.roll_z)
    ) if "oi_zscore" in out.columns else False
    base_prob = 0.5 + 0.5 * np.tanh(out["oi_price_corr"].fillna(0.0) * conf_cfg.bayes_k)
    base_prob = np.where(out["roll_flag"],
                         0.5 + (base_prob - 0.5) * oi_cfg.roll_dampen,
                         base_prob)
    base_prob = np.where(out["voi_eff"] < oi_cfg.voi_eff_min, 0.5, base_prob)
    out["oi_conf_score"] = np.clip(base_prob, 0.0, 1.0)
    out["mtf_oi_align_score"] = 2.0 * (out["oi_conf_score"] - 0.5)
    return out

def aggregate_mtf_oi_conf(
    per_tf_conf: Dict[str, float],
    conf_cfg: ConfidenceConfig
) -> Tuple[float, float]:
    scores = []
    weights = []
    for tf, c in per_tf_conf.items():
        if tf in conf_cfg.mtf_oi_weights:
            scores.append(c)
            weights.append(conf_cfg.mtf_oi_weights[tf])
    if not scores:
        return 0.5, 0.0
    scores = np.array(scores)
    weights = np.array(weights)
    weights = weights / weights.sum()
    oi_conf = float((scores * weights).sum())
    var = float(((scores - oi_conf) ** 2 * weights).sum())
    align_raw = 1.0 - var * 4.0
    mtf_align = float(np.clip(align_raw, -1.0, 1.0))
    if mtf_align < conf_cfg.mtf_align_gate:
        oi_conf = 0.5 + (oi_conf - 0.5) * conf_cfg.mtf_conflict_cap
    return oi_conf, mtf_align

def build_confidence_row(
    symbol: str,
    interval: str,
    ts,
    price_vol_row: pd.Series,
    oi_row: pd.Series,
    mtf_oi_conf: float,
    mtf_align_score: float,
    conf_cfg: ConfidenceConfig,
) -> Dict[str, Any]:
    conf_price_vol = float(price_vol_row["conf_price_vol"])
    conf_oi        = float(oi_row["oi_conf_score"])
    if conf_cfg.oi_integration == "bayes":
        logit_pv = np.log(conf_price_vol / max(1e-6, 1 - conf_price_vol))
        logit_oi = np.log(conf_oi / max(1e-6, 1 - conf_oi))
        combined_logit = (1 - conf_cfg.bayes_m) * logit_pv + conf_cfg.bayes_m * logit_oi
        conf_total = float(1.0 / (1.0 + np.exp(-combined_logit)))
    else:
        conf_total = float(0.5 * (conf_price_vol + conf_oi))
    row = {
        "symbol":            symbol,
        "interval":          interval,
        "ts":                ts,
        "conf_total":        conf_total,
        "conf_price_vol":    conf_price_vol,
        "conf_oi":           conf_oi,
        "hmm_state":         int(price_vol_row.get("hmm_state", 0)),
        "hmm_prob_trend_up": float(price_vol_row.get("hmm_prob_trend_up", 0.33)),
        "hmm_prob_trend_down": float(price_vol_row.get("hmm_prob_trend_down", 0.33)),
        "hmm_regime_label":  price_vol_row.get("hmm_regime_label", "unknown"),
        "mtf_oi_align_score": float(mtf_align_score),
        "oi_conf_score":      float(oi_row.get("oi_conf_score", 0.5)),
        "oi_turnover":        float(oi_row.get("oi_turnover", 0.0)),
        "voi_eff":            float(oi_row.get("voi_eff", 0.0)),
        "roll_flag":          bool(oi_row.get("roll_flag", False)),
        "squeeze_flag":       bool(price_vol_row.get("squeeze_flag", False)),
        "squeeze_score":      float(price_vol_row.get("squeeze_score", 0.0)),
        "crowd_flag":         bool(price_vol_row.get("crowd_flag", False)),
        "trap_flag":          bool(price_vol_row.get("trap_flag", False)),
        "vp_poc_trap_flag":   bool(price_vol_row.get("vp_poc_trap_flag", False)),
        "vp_poc_dist_atr":    float(price_vol_row.get("vp_poc_dist_atr", 0.0)),
        "low_liq_flag":       bool(oi_row.get("low_liq_flag", False)),
        "event_flag":         False,
        "squeeze_bw_pct":     float(price_vol_row.get("squeeze_bw_pct", 0.0)),
        "bb_score":           float(price_vol_row.get("bb_score", 0.0)),
        "run_id":             conf_cfg.run_id,
        "source":             conf_cfg.source,
        "debug_json":         None,
    }
    return row
