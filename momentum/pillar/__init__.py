# momentum/pillar/__init__.py

from pathlib import Path

from .momentum_pillar import score_momentum, _cfg   # and any other public APIs

# Default INI path (mirror flow)
DEFAULT_MOMENTUM_INI = str(
    Path(__file__).with_suffix("")  # .../pillar/__init__.py → .../pillar
    .parent
    .joinpath("momentum_scenarios.ini")
)

__all__ = ["score_momentum", "DEFAULT_MOMENTUM_INI", "_cfg"]
