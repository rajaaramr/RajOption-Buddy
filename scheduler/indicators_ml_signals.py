# scheduler/indicators_ml_signals.py

"""
This script now serves as an entry point to the refactored ML strategy pipeline.
It delegates the execution to the main function within the new modular structure.
"""

from ml_strategy import main

if __name__ == "__main__":
    # Call the main pipeline function from the new modular structure
    main.run_pipeline()
