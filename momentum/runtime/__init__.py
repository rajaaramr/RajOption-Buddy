# momentum/runtime/__init__.py

"""
Runtime entrypoints for Momentum pillar:

- backfill.py : batch rules / ML / both
- worker.py   : daily rules worker (used by dashboard)
- run_cli.py  : single-symbol runner for debugging
- utils.py    : DB status helpers
"""

from .backfill import main as backfill_main  # convenience

__all__ = ["backfill_main"]
