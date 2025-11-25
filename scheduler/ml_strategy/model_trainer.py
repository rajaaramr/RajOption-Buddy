# scheduler/ml_strategy/model_trainer.py

"""
Model training, evaluation, and signal generation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from . import config

def prepare_data_for_training(df_features):
    """
    Prepares the feature dataframe for model training.
    - Defines features (X) and target (y)
    - Splits into training and testing sets
    - Handles NaNs and infinite values
    """
    print("[INFO] Preparing data for training...")

    # Define target and features
    y = df_features['target']

    # Drop columns that are not features
    cols_to_drop = [
        'target', 'future_close_SPOT', 'future_ret', 'high', 'low', 'volume_SPOT',
        'oi_FUT', 'interval', 'h-l', 'h-pc', 'l-pc'
    ]
    # Drop original indicator columns that now have lagged versions
    lag_cols_originals = [c for c in df_features.columns if "_SPOT_" in c or "_FUT_" in c and "_LAG_" not in c]
    cols_to_drop.extend(lag_cols_originals)

    X = df_features.drop(columns=cols_to_drop, errors='ignore')
    X = X.select_dtypes(include=np.number) # Keep only numeric types

    # Align X and y
    y = y.loc[X.index]

    # Time-based split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SET_FRAC, shuffle=False
    )

    # Clean NaNs and infs
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"[INFO] Data split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

def hyperparameter_tuning(X_train, y_train):
    """
    Performs hyperparameter tuning for XGBoost and LightGBM.
    """
    print("[INFO] Starting hyperparameter tuning...")

    # --- XGBoost ---
    xgb_param_grid = {
        'max_depth': [5, 7],
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1]
    }
    xgb_grid = GridSearchCV(xgb.XGBClassifier(**config.XGB_PARAMS), xgb_param_grid, cv=3, n_jobs=-1, verbose=1)
    xgb_grid.fit(X_train, y_train)
    print(f"[INFO] Best XGBoost params: {xgb_grid.best_params_}")

    # --- LightGBM ---
    lgb_param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50]
    }
    lgb_grid = GridSearchCV(lgb.LGBMClassifier(), lgb_param_grid, cv=3, n_jobs=-1, verbose=1)
    lgb_grid.fit(X_train, y_train)
    print(f"[INFO] Best LightGBM params: {lgb_grid.best_params_}")

    return xgb_grid.best_params_, lgb_grid.best_params_

def train_and_evaluate(X_train, X_test, y_train, y_test, best_xgb_params=None, best_lgb_params=None):
    """
    Trains XGBoost and LightGBM models, evaluates them, and returns predictions.
    """
    models = {}
    predictions = {}

    # --- Class Imbalance Handling ---
    y_mapped = y_train.map({-1: 2, 0: 0, 1: 1})
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_mapped)

    # --- XGBoost ---
    print("\n[INFO] Training XGBoost model...")
    y_train_xgb = y_train.map({-1: 2, 0: 0, 1: 1})
    y_test_xgb = y_test.map({-1: 2, 0: 0, 1: 1})

    xgb_params = {**config.XGB_PARAMS, **(best_xgb_params or {})}
    xgb_params['objective'] = 'multi:softprob'
    xgb_params['num_class'] = 3

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train_xgb, sample_weight=sample_weights)

    xgb_proba = xgb_model.predict_proba(X_test)
    predictions['xgb'] = pd.DataFrame(xgb_proba, index=X_test.index, columns=['prob_sideways_xgb', 'prob_up_xgb', 'prob_down_xgb'])
    models['xgb'] = xgb_model

    print("[INFO] XGBoost training complete.")
    print("XGBoost Classification Report:")
    print(classification_report(y_test_xgb, np.argmax(xgb_proba, axis=1), target_names=['sideways', 'up', 'down']))

    # --- LightGBM ---
    print("\n[INFO] Training LightGBM model...")
    y_train_lgb = y_train.map({-1: 2, 0: 0, 1: 1})
    y_test_lgb = y_test.map({-1: 2, 0: 0, 1: 1})

    lgb_params = {
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
        'verbose': -1, 'n_jobs': -1, 'seed': 42,
        **(best_lgb_params or {})
    }

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train_lgb, sample_weight=sample_weights)

    lgb_proba = lgb_model.predict_proba(X_test)
    predictions['lgb'] = pd.DataFrame(lgb_proba, index=X_test.index, columns=['prob_sideways_lgb', 'prob_up_lgb', 'prob_down_lgb'])
    models['lgb'] = lgb_model

    print("[INFO] LightGBM training complete.")
    print("LightGBM Classification Report:")
    print(classification_report(y_test_lgb, np.argmax(lgb_proba, axis=1), target_names=['sideways', 'up', 'down']))

    return models, predictions

def save_signals_to_db(predictions_dict, engine):
    """
    Combines model predictions and saves them to the database.
    """
    df_signals = pd.concat(predictions_dict.values(), axis=1)

    # Add a simple combined signal
    df_signals['prob_up_combined'] = df_signals[['prob_up_xgb', 'prob_up_lgb']].mean(axis=1)
    df_signals['prob_down_combined'] = df_signals[['prob_down_xgb', 'prob_down_lgb']].mean(axis=1)

    df_signals['ml_direction'] = 0
    up_mask = (df_signals['prob_up_combined'] > 0.6) & (df_signals['prob_up_combined'] > df_signals['prob_down_combined'])
    down_mask = (df_signals['prob_down_combined'] > 0.6) & (df_signals['prob_down_combined'] > df_signals['prob_up_combined'])
    df_signals.loc[up_mask, 'ml_direction'] = 1
    df_signals.loc[down_mask, 'ml_direction'] = -1

    df_to_write = df_signals.reset_index()

    print(f"\n[INFO] Writing {len(df_to_write)} signals to the database...")

    try:
        df_to_write.to_sql(
            "ml_signals_multiclass", # New table name
            engine,
            schema="indicators",
            if_exists="replace", # Use replace for backtesting, append/upsert for production
            index=False
        )
        print("[INFO] Successfully wrote signals to indicators.ml_signals_multiclass")
    except Exception as e:
        print(f"[ERROR] Failed to write signals to database: {e}")
