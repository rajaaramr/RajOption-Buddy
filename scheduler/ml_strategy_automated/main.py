# scheduler/ml_strategy_automated/main.py

"""
Main entry point for the automated ML strategy pipeline.
"""
import pandas as pd
import numpy as np
from . import data_loader, automated_feature_engineering
# Re-use the model trainer from the manual strategy
from ..ml_strategy import model_trainer, config

def run_pipeline(tune=False):
    """
    Executes the full automated ML pipeline.
    """
    print("======================================================")
    print("=== Starting Automated ML Strategy Pipeline Run ===")
    print("======================================================")

    # 1. Load Data
    engine = data_loader.get_engine()
    data_dict = data_loader.load_all_tables_for_featuretools(engine)

    # 2. Create EntitySet
    es = automated_feature_engineering.create_entityset(data_dict)

    # 3. Generate Automated Daily Features
    daily_feature_matrix, feature_defs = automated_feature_engineering.generate_automated_features(es)

    # 4. Merge automated daily features onto intraday 15m data
    spot_candles_df = data_dict["spot_candles"].copy()
    spot_candles_df['trade_date'] = spot_candles_df['ts'].dt.normalize()

    # Merge the rich daily features onto every 15-minute candle for that day
    hybrid_df = pd.merge(spot_candles_df, daily_feature_matrix, on=['symbol', 'trade_date'], how='left')
    hybrid_df = hybrid_df.sort_values('ts')

    # 5. Define Target Variable
    hybrid_df["future_close_SPOT"] = hybrid_df.groupby("symbol")["close"].shift(-config.TARGET_PERIODS_AHEAD)
    hybrid_df["future_ret"] = (hybrid_df["future_close_SPOT"] / hybrid_df["close"]) - 1.0

    hybrid_df["target"] = 0 # Sideways
    hybrid_df.loc[hybrid_df["future_ret"] > config.TARGET_THRESHOLD, "target"] = 1 # Up
    hybrid_df.loc[hybrid_df["future_ret"] < -config.TARGET_THRESHOLD, "target"] = -1 # Down

    # 6. Prepare Data for Training
    # Drop non-feature columns
    features_to_drop = ['future_close_SPOT', 'future_ret', 'symbol', 'ts', 'trade_date', 'target']
    X = hybrid_df.drop(columns=features_to_drop, errors='ignore')
    X = X.select_dtypes(include=np.number).fillna(0) # Keep only numeric types and fill NaNs
    y = hybrid_df['target']

    # Time-based split
    split_idx = int(len(X) * (1 - config.TEST_SET_FRAC))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"[INFO] Data prepared for training. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 7. Train and Evaluate Models
    models, predictions = model_trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

    # 8. Save Signals to a New Database Table
    if predictions:
        model_trainer.save_signals_to_db_automated(predictions, engine)

    print("\n======================================================")
    print("=== Automated ML Strategy Pipeline Run Finished ===")
    print("======================================================")

if __name__ == "__main__":
    # Example of how to run, though it's called from the entry script
    run_pipeline()
