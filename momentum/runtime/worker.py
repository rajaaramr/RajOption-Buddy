# momentum/runtime/worker.py

from __future__ import annotations
import sys
from datetime import datetime

from .backfill import main as backfill_main


def main(argv=None):
    """
    Daily rules worker for Momentum.
    Intended invocation (futures, all symbols):

        python -m momentum.runtime.worker --kind futures
    """
    # We just forward to backfill with sensible defaults if nothing else is passed.
    default_args = ["--all-symbols", "--kind", "futures", "--mode", "rules", "--lookback-days", "3"]
    args = default_args if argv is None else argv
    backfill_main(args)


if __name__ == "__main__":
    main(sys.argv[1:])
