# scheduler/ml_strategy/main.py

"""
Main entry point for the ML strategy pipeline.
Orchestrates data loading, feature engineering, model training, and signal generation.
"""

from . import data_loader, feature_engineering
from scheduler.ml_utils import model_trainer

import argparse

def run_pipeline(tune=False):
    """
    Executes the full ML pipeline.
    """
    print("=============================================")
    print("=== Starting ML Strategy Pipeline Run ===")
    print("=============================================")

    # 1. Load Data
    engine = data_loader.get_engine()
    data_dict = data_loader.load_all_tables(engine)

    if not data_dict:
        print("[FATAL] Data loading returned empty dict. Aborting.")
        return

    # 2. Feature Engineering
    df_features = feature_engineering.build_features(data_dict)

    if df_features.empty:
        print("[FATAL] Feature engineering resulted in an empty DataFrame. Aborting.")
        return

    # 3. Prepare Data for Training
    X_train, X_test, y_train, y_test = model_trainer.prepare_data_for_training(df_features)

    best_xgb_params, best_lgb_params = None, None
    if tune:
        best_xgb_params, best_lgb_params = model_trainer.hyperparameter_tuning(X_train, y_train)

    # 4. Train and Evaluate Models
    models, predictions = model_trainer.train_and_evaluate(X_train, X_test, y_train, y_test, best_xgb_params, best_lgb_params)

    # 5. Save Signals to Database
    if predictions:
        model_trainer.save_signals_to_db(predictions, engine, table_name="ml_signals_multiclass")
    else:
        print("[WARN] No predictions were generated. Skipping database write.")

    print("\n==========================================")
    print("=== ML Strategy Pipeline Run Finished ===")
    print("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning.")
    args = parser.parse_args()
    run_pipeline(tune=args.tune)
