# File: utils/kite_utils.py
# Purpose: Kite auth helpers + resilient futures instrument resolver + safe LTP fetches

from __future__ import annotations

import configparser
from datetime import date
from typing import Optional, Dict, List, Tuple

from kiteconnect import KiteConnect

CONFIG_PATH = "zerodha.ini"

# ------------------- Config Loader -------------------

def load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config


# ------------------- Public Exports -------------------

__all__ = [
    "generate_login_url",
    "generate_access_token",
    "get_kite_session",
    "get_futures_instrument",
    "fetch_futures_data",
    "exchange_and_store_token",
    "load_kite",
]

# ------------------- Local cache -------------------

_INSTR_CACHE: dict[str, Optional[List[dict]]] = {"NFO": None, "NSE": None}


def _load_instruments(kite: KiteConnect, exchange: str, force: bool = False) -> List[dict]:
    """Cache instruments to avoid repeated full downloads during a run."""
    exchange = exchange.upper()
    global _INSTR_CACHE
    if force or _INSTR_CACHE.get(exchange) is None:
        _INSTR_CACHE[exchange] = kite.instruments(exchange)
    return _INSTR_CACHE[exchange] or []


# ------------------- Kite Auth & Token Flow -------------------

def generate_login_url(api_key: str) -> str:
    kite = KiteConnect(api_key=api_key)
    url = kite.login_url()
    print(f"🔗 Zerodha Login URL: {url}")
    return url


def generate_access_token(api_key: str, api_secret: str, request_token: str) -> str:
    kite = KiteConnect(api_key=api_key)
    session_data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session_data["access_token"]
    print(f"✅ Access token: {access_token[:8]}...")
    return access_token


def exchange_and_store_token(request_token: str) -> str:
    config = load_config()
    api_key = config["kite"]["api_key"]
    api_secret = config["kite"]["api_secret"]

    kite = KiteConnect(api_key=api_key)
    session = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session["access_token"]

    if "kite" not in config:
        config["kite"] = {}
    config["kite"]["access_token"] = access_token

    with open(CONFIG_PATH, "w") as f:
        config.write(f)

    print(f"✅ Stored access token in zerodha.ini: {access_token[:8]}...")
    return access_token


def get_kite_session(api_key: str, access_token: str) -> KiteConnect:
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def load_kite() -> KiteConnect:
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    api_key = config["kite"]["api_key"]
    access_token = config["kite"]["access_token"]

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


# ------------------- Futures Expiry Suffix -------------------

def get_futures_expiry_suffix() -> str:
    """
    Read configured futures suffix (e.g., 25AUGFUT). If missing, returns a sane default.
    """
    config = load_config()
    return config.get("settings", "fut_expiry_suffix", fallback="25AUGFUT").upper()


# ------------------- Instrument Resolver -------------------

def _nearest_expiry_for_symbol(symbol: str, nfo_instruments: List[dict]) -> Optional[date]:
    today = date.today()
    expiries = sorted({
        ins.get("expiry").date()
        for ins in nfo_instruments
        if ins.get("name", "").upper() == symbol.upper()
        and ins.get("segment") == "NFO-FUT"
        and ins.get("instrument_type") == "FUT"
        and ins.get("expiry")
        and ins["expiry"].date() >= today
    })
    return expiries[0] if expiries else None


def _find_fut_by_suffix(symbol: str, suffix: str, nfo_instruments: List[dict]) -> Optional[Dict]:
    tsym = f"{symbol.upper()}{suffix}"
    for ins in nfo_instruments:
        if (
            str(ins.get("tradingsymbol", "")).upper() == tsym
            and ins.get("segment") == "NFO-FUT"
            and ins.get("instrument_type") == "FUT"
        ):
            return {"tradingsymbol": ins["tradingsymbol"], "instrument_token": ins["instrument_token"]}
    return None


def _find_fut_by_nearest_expiry(symbol: str, nfo_instruments: List[dict]) -> Optional[Dict]:
    exp = _nearest_expiry_for_symbol(symbol, nfo_instruments)
    if not exp:
        return None
    for ins in nfo_instruments:
        if (
            ins.get("name", "").upper() == symbol.upper()
            and ins.get("segment") == "NFO-FUT"
            and ins.get("instrument_type") == "FUT"
            and ins.get("expiry")
            and ins["expiry"].date() == exp
        ):
            return {"tradingsymbol": ins["tradingsymbol"], "instrument_token": ins["instrument_token"]}
    return None


def get_futures_instrument(symbol: str, kite: KiteConnect) -> Optional[Dict]:
    """
    Return {'tradingsymbol','instrument_token'} for FUT of `symbol`.
    Tries config suffix first (fast path), then falls back to nearest-expiry FUT if not found.
    """
    try:
        nfo = _load_instruments(kite, "NFO")
    except Exception as e:
        print(f"❌ Error loading NFO instruments: {e}")
        return None

    # 1) Config-specified suffix
    try:
        suffix = get_futures_expiry_suffix()
        hit = _find_fut_by_suffix(symbol, suffix, nfo)
        if hit:
            return hit
    except Exception:
        pass

    # 2) Fallback to nearest-expiry FUT
    hit = _find_fut_by_nearest_expiry(symbol, nfo)
    if hit:
        return hit

    print(f"⚠️ FUT instrument not found for {symbol}")
    return None


# ------------------- Fetch Futures Data -------------------

def fetch_futures_data(symbol: str, kite: KiteConnect) -> Optional[Dict]:
    """
    Return a dict: {tradingsymbol, last_price, volume, instrument_token} for FUT.
    Falls back gracefully if fields are missing. Uses quote() (bulk-capable) path.
    """
    instrument = get_futures_instrument(symbol, kite)
    if not instrument:
        return None

    tradingsymbol = instrument["tradingsymbol"]
    instrument_token = instrument["instrument_token"]

    try:
        key = f"NFO:{tradingsymbol}"
        quote = (kite.quote([key]) or {}).get(key, {})

        # price fallback: last_price -> ohlc.close
        last_price = quote.get("last_price")
        if last_price is None:
            last_price = (quote.get("ohlc") or {}).get("close")

        # volume fallback: volume -> volume_traded -> 0
        volume = quote.get("volume")
        if volume is None:
            volume = quote.get("volume_traded")
        try:
            volume = int(volume) if volume is not None else 0
        except Exception:
            volume = 0

        if last_price is None:
            print(f"⚠️ Incomplete FUT quote for {symbol}: {quote}")
            return None

        return {
            "tradingsymbol": tradingsymbol,
            "last_price": float(last_price),
            "volume": volume,
            "instrument_token": instrument_token,
        }

    except Exception as e:
        print(f"❌ Error fetching quote for {symbol}: {e}")
        return None
