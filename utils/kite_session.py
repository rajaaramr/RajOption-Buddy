# File: utils/kite_session.py

import configparser
from kiteconnect import KiteConnect

def load_kite() -> KiteConnect:
    """
    Loads KiteConnect session from saved config file.
    """# File: utils/kite_session.py
# Purpose: Load an authenticated KiteConnect session from zerodha.ini

import os
import configparser
from kiteconnect import KiteConnect

CONFIG_PATH = os.getenv("ZCONFIG", "zerodha.ini")

def load_kite() -> KiteConnect:
    """
    Load an authenticated KiteConnect session from the INI file.
    Raises a clear error if keys are missing.
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"❌ Config file not found: {CONFIG_PATH}")

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    try:
        api_key = config["kite"]["api_key"]
        access_token = config["kite"]["access_token"]
    except KeyError as e:
        raise KeyError(f"❌ Missing required key in [kite] section: {e}")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite

    config = configparser.ConfigParser()
    config.read("zerodha.ini")
    
    api_key = config["kite"]["api_key"]
    access_token = config["kite"]["access_token"]

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite
