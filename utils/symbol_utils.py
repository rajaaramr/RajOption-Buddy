# File: utils/symbol_utils.py
# Purpose: Minimal, DB-free symbol helpers driven by zerodha.ini

from __future__ import annotations
import os
import configparser
from functools import lru_cache
from typing import Optional

# Path to your ini (adjust if you keep it elsewhere)
INI_PATH = os.environ.get("ZERODHA_INI_PATH", "zerodha.ini")

# ---------- INI helpers ----------

@lru_cache(maxsize=1)
def _cfg() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(INI_PATH)
    return cfg

def get_futures_expiry_suffix() -> str:
    """
    Read FUT expiry suffix from zerodha.ini, e.g. '25AUGFUT'.
    """
    return _cfg().get("settings", "fut_expiry_suffix", fallback="").upper().strip()

# ---------- Public API (DB-free) ----------

def get_spot_tradingsymbol(symbol: str) -> str:
    """
    Canonical NSE spot symbol. DB not required.
    If you later add an exchange map, update here.
    """
    return (symbol or "").upper().strip()

def get_futures_tradingsymbol(symbol: str) -> Optional[str]:
    """
    Build futures tradingsymbol using suffix from zerodha.ini.
    Returns None if suffix is not configured.
    Example: 'INFY' + '25AUGFUT' -> 'INFY25AUGFUT'
    """
    base = get_spot_tradingsymbol(symbol)
    suf = get_futures_expiry_suffix()
    if not base or not suf:
        return None
    return f"{base}{suf}"

def get_option_symbol_base(symbol: str) -> str:
    """
    For callers that previously reused the FUT base for options:
    returns '<SYMBOL><FUT_SUFFIX>' (e.g., 'INFY25AUGFUT' -> you can
    replace 'FUT' with strike/CE/PE in your option builder).
    If you move to instruments cache for options, you can retire this.
    """
    fut = get_futures_tradingsymbol(symbol)
    return fut or ""

def get_lot_size(symbol: str) -> int:
    """
    Placeholder (DB-free). Return 1 by default.
    If you create a metadata table later, wire it here.
    """
    return 1

# ---------- Convenience ----------

def is_index(symbol: str) -> bool:
    return get_spot_tradingsymbol(symbol) in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

def is_valid_tradingsymbol(symbol: str) -> bool:
    s = (symbol or "").upper().strip()
    return s.endswith("FUT") and len(s) > 10
