# File: utils/option_strikes.py
# Purpose: Resolve nearest expiry & strikes for NFO options, choose ATM ± n,
#          and pick the single nearest CE/PE by FUT (fallback SPOT) reference.

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple, Optional, Iterable
from collections import defaultdict

# Simple in-memory cache for instruments
_INSTR_CACHE: Dict[str, Optional[List[dict]]] = {"NFO": None, "NSE": None}


# ---------------------------
# Public data structures
# ---------------------------

@dataclass
class OptionSelection:
    symbol: str                 # underlying, e.g., "RELIANCE"
    expiry: date                # nearest expiry date
    strike: float               # chosen strike
    option_type: str            # "CE" or "PE"
    tradingsymbol: str          # NFO tradingsymbol chosen
    token_key: str              # e.g., "NFO:RELIANCE25AUG4200CE"
    ref_price: float            # price used to choose ATM (FUT or SPOT)
    ref_basis: str              # "FUT" or "SPOT"


# ---------------------------
# Instrument helpers
# ---------------------------

def load_nfo_instruments(kite, force: bool = False) -> List[dict]:
    """Cache NFO instruments to avoid repeated calls."""
    global _INSTR_CACHE
    if force or _INSTR_CACHE["NFO"] is None:
        _INSTR_CACHE["NFO"] = kite.instruments("NFO")
    return _INSTR_CACHE["NFO"] or []


def load_nse_instruments(kite, force: bool = False) -> List[dict]:
    global _INSTR_CACHE
    if force or _INSTR_CACHE["NSE"] is None:
        _INSTR_CACHE["NSE"] = kite.instruments("NSE")
    return _INSTR_CACHE["NSE"] or []


def nearest_expiry_for_symbol(symbol: str, instruments: Iterable[dict]) -> Optional[date]:
    """Find the nearest (>= today) option expiry for the given underlying."""
    today = date.today()
    expiries = sorted({
        ins.get("expiry").date()
        for ins in instruments
        if ins.get("name", "").upper() == symbol.upper()
        and ins.get("segment") == "NFO-OPT"
        and ins.get("instrument_type") in ("CE", "PE")
        and ins.get("expiry")
        and ins["expiry"].date() >= today
    })
    return expiries[0] if expiries else None


def strikes_for_expiry(symbol: str, instruments: Iterable[dict], expiry: date) -> Dict[float, Dict[str, str]]:
    """
    For a given underlying+expiry, return {strike: {'CE': tsym, 'PE': tsym}}.
    """
    strikes_map: Dict[float, Dict[str, str]] = defaultdict(dict)
    for ins in instruments:
        if (
            ins.get("name", "").upper() == symbol.upper()
            and ins.get("segment") == "NFO-OPT"
            and ins.get("instrument_type") in ("CE", "PE")
            and ins.get("expiry") and ins["expiry"].date() == expiry
        ):
            strikes_map[float(ins["strike"])][ins["instrument_type"]] = ins["tradingsymbol"]
    return dict(strikes_map)


# ---------------------------
# Price reference (FUT→SPOT fallback)
# ---------------------------

def _resolve_ref_price(symbol: str, kite, use_fut_for_atm: bool = True) -> Tuple[float, str]:
    """
    Return (price, basis) where basis is "FUT" or "SPOT".
    Tries nearest-expiry FUT first; if anything fails, uses NSE spot LTP.
    """
    spot_key = f"NSE:{symbol.upper()}"
    ref_basis = "SPOT"
    ref_price = 0.0

    # Spot first (always available for fallback)
    try:
        spot_ltp = kite.ltp([spot_key])[spot_key]["last_price"]
        ref_price = float(spot_ltp or 0)
    except Exception:
        ref_price = 0.0

    if not use_fut_for_atm:
        return ref_price, ref_basis

    # Try FUT (nearest expiry)
    try:
        instruments = load_nfo_instruments(kite)
        expiry = nearest_expiry_for_symbol(symbol, instruments)
        if expiry:
            fut_rows = [
                ins for ins in instruments
                if ins.get("name", "").upper() == symbol.upper()
                and ins.get("segment") == "NFO-FUT"
                and ins.get("instrument_type") == "FUT"
                and ins.get("expiry") and ins["expiry"].date() == expiry
            ]
            if fut_rows:
                fut_tsym = fut_rows[0]["tradingsymbol"]
                fut_key = f"NFO:{fut_tsym}"
                fut_ltp = kite.ltp([fut_key])[fut_key]["last_price"]
                if fut_ltp:
                    return float(fut_ltp), "FUT"
    except Exception:
        pass

    return ref_price, ref_basis


# ---------------------------
# Public API (backward-compat)
# ---------------------------

def atm_and_neighbors(symbol, kite, n: int = 3, use_fut_for_atm: bool = True):
    """
    Returns: (spot_price, expiry_date, tokens:list[str], meta:{token_key -> (strike, 'CE'/'PE')})
    - ATM chosen by closest available strike to FUT LTP (default) or SPOT.
    - tokens are 'NFO:{tradingsymbol}' strings, ready for kite.ltp()
    """
    # Reference price
    ref_price, basis = _resolve_ref_price(symbol, kite, use_fut_for_atm=use_fut_for_atm)

    # Instruments & nearest expiry
    instruments = load_nfo_instruments(kite)
    expiry = nearest_expiry_for_symbol(symbol, instruments)
    if not expiry:
        return ref_price, None, [], {}

    strikes_map = strikes_for_expiry(symbol, instruments, expiry)
    available_strikes = sorted(strikes_map.keys())
    if not available_strikes:
        return ref_price, expiry, [], {}

    # Choose ATM index by closeness to ref_price
    atm_idx = min(range(len(available_strikes)), key=lambda i: abs(available_strikes[i] - ref_price))
    idxs = [i for i in range(atm_idx - n, atm_idx + n + 1) if 0 <= i < len(available_strikes)]
    chosen = [available_strikes[i] for i in idxs]

    # Build tokens + meta
    tokens: List[str] = []
    meta: Dict[str, Tuple[float, str]] = {}
    for k in chosen:
        row = strikes_map.get(k, {})
        ce = row.get("CE")
        pe = row.get("PE")
        if ce:
            key = f"NFO:{ce}"
            tokens.append(key)
            meta[key] = (k, "CE")
        if pe:
            key = f"NFO:{pe}"
            tokens.append(key)
            meta[key] = (k, "PE")

    return ref_price, expiry, tokens, meta


# ---------------------------
# New API for ENTRY flow
# ---------------------------

def pick_nearest_option(symbol: str, side: str, kite, use_fut_for_atm: bool = True) -> Optional[OptionSelection]:
    """
    Choose a single nearest strike for the given side:
      - side in {"LONG","BUY"}  -> CE
      - side in {"SHORT","SELL"}-> PE
    Uses FUT LTP as reference (fallback to SPOT). Returns OptionSelection or None.
    """
    side_u = (side or "LONG").upper()
    want_ce = side_u in ("LONG", "BUY")
    want_pe = side_u in ("SHORT", "SELL")

    ref_price, basis = _resolve_ref_price(symbol, kite, use_fut_for_atm=use_fut_for_atm)

    instruments = load_nfo_instruments(kite)
    expiry = nearest_expiry_for_symbol(symbol, instruments)
    if not expiry:
        return None

    strikes_map = strikes_for_expiry(symbol, instruments, expiry)
    if not strikes_map:
        return None

    available_strikes = sorted(strikes_map.keys())
    # Pick nearest strike index to reference price
    atm_idx = min(range(len(available_strikes)), key=lambda i: abs(available_strikes[i] - ref_price))
    # The exact ATM strike
    strike = available_strikes[atm_idx]

    row = strikes_map.get(strike, {})
    tsym = row.get("CE") if want_ce else row.get("PE") if want_pe else None
    opt_type = "CE" if want_ce else "PE" if want_pe else None
    if not tsym or not opt_type:
        # Try neighbor if exact not present for that side
        for i in range(1, max(3, len(available_strikes))):
            for idx in (atm_idx - i, atm_idx + i):
                if 0 <= idx < len(available_strikes):
                    s = available_strikes[idx]
                    r = strikes_map.get(s, {})
                    tsym_try = r.get("CE") if want_ce else r.get("PE")
                    if tsym_try:
                        return OptionSelection(
                            symbol=symbol.upper(),
                            expiry=expiry,
                            strike=float(s),
                            option_type="CE" if want_ce else "PE",
                            tradingsymbol=tsym_try,
                            token_key=f"NFO:{tsym_try}",
                            ref_price=float(ref_price),
                            ref_basis=basis,
                        )
        return None

    return OptionSelection(
        symbol=symbol.upper(),
        expiry=expiry,
        strike=float(strike),
        option_type=opt_type,
        tradingsymbol=tsym,
        token_key=f"NFO:{tsym}",
        ref_price=float(ref_price),
        ref_basis=basis,
    )
