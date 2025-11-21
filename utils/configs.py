# utils/configs.py
"""
Centralized configuration management for the AlphaPivot trading system.

This module provides a set of functions for loading and accessing settings
from the unified `config.ini` file. It ensures that all parts of the
application can access configuration from a single, consistent source.
"""
import configparser
import os
from typing import List, Dict

DEFAULT_CONFIG = "config.ini"

def _load_config(path: str = DEFAULT_CONFIG) -> configparser.ConfigParser:
    """
    Loads the central INI file into a ConfigParser object.

    Args:
        path: The path to the configuration file. Defaults to "config.ini".

    Returns:
        A ConfigParser object with the loaded configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Main config file not found at: {path}")
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), interpolation=None, strict=False)
    cp.read(path)
    return cp

def get_config_parser(path: str = DEFAULT_CONFIG) -> configparser.ConfigParser:
    """
    Public accessor for the raw ConfigParser object.

    This function provides direct access to the ConfigParser object for advanced
    use cases where the specific helper functions are not sufficient.

    Args:
        path: The path to the configuration file.

    Returns:
        A ConfigParser object.
    """
    return _load_config(path)

def get_db_config(path: str = DEFAULT_CONFIG) -> Dict[str, str]:
    """
    Returns the [postgres] section from the config file as a dictionary.

    This is a convenience function for accessing the database connection
    parameters.

    Args:
        path: The path to the configuration file.

    Returns:
        A dictionary containing the database configuration.

    Raises:
        ValueError: If the [postgres] section is not found in the config file.
    """
    cp = _load_config(path)
    if "postgres" not in cp:
        raise ValueError("[postgres] section not found in config")
    return dict(cp["postgres"])

def get_symbols(path: str = DEFAULT_CONFIG) -> List[str]:
    """
    Returns the list of symbols from the [symbols] section of the config file.

    Args:
        path: The path to the configuration file.

    Returns:
        A list of trading symbols.
    """
    cp = _load_config(path)
    if "symbols" not in cp or "list" not in cp["symbols"]:
        return []
    return [s.strip() for s in cp.get("symbols", "list").split(",") if s.strip()]

def get_timeframes(path: str = DEFAULT_CONFIG) -> List[str]:
    """
    Returns the list of timeframes from the [timeframes] section of the config file.

    Args:
        path: The path to the configuration file.

    Returns:
        A list of timeframes (e.g., "15m", "60m").
    """
    cp = _load_config(path)
    if "timeframes" not in cp or "list" not in cp["timeframes"]:
        return []
    return [s.strip() for s in cp.get("timeframes", "list").split(",") if s.strip()]