# scheduler/indicators_ml_signals_automated.py

"""
Main entry point for the automated ML strategy pipeline.
"""

import argparse
from ml_strategy_automated import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning.")
    args = parser.parse_args()

    # Call the main pipeline function from the new modular structure
    main.run_pipeline(tune=args.tune)
