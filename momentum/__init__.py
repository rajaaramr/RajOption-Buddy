# momentum/__init__.py

"""
Momentum pillar package.

Folders:
- pillar/  : rules + scenarios + ML blend hook (score_momentum)
- runtime/ : backfill, worker, CLI runner, status helpers
- ml/      : training + calibration (to be wired per MOM-20+)

Usage:
    from momentum.pillar import score_momentum
    from momentum.runtime.backfill import main as momentum_backfill_main
"""

__all__ = ["pillar", "runtime", "ml"]
