"""
Utility Module
================

Provides logging setup, reproducibility controls, and common helper functions
used across both ML pipelines.
"""

import logging
import random
import sys
from typing import Optional

import numpy as np

from src.config import LOG_FORMAT, LOG_LEVEL, RANDOM_STATE


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    Create and configure a module-level logger.

    Args:
        name: Logger name (typically __name__ from the calling module).
        level: Logging level string (e.g., 'INFO', 'DEBUG'). Defaults to config.
        log_format: Custom log format string. Defaults to config.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(log_format or LOG_FORMAT)
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level or LOG_LEVEL))
    return logger


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """
    Set random seeds for full reproducibility across all libraries.

    Args:
        seed: Integer seed value. Defaults to config.RANDOM_STATE.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not required


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a float as a percentage string.

    Args:
        value: Float value (e.g., 0.85).
        decimals: Number of decimal places.

    Returns:
        Formatted string (e.g., '85.00%').
    """
    return f"{value * 100:.{decimals}f}%"


def print_section_header(title: str, char: str = "=", width: int = 70) -> None:
    """
    Print a formatted section header for notebook readability.

    Args:
        title: Section title text.
        char: Character to use for the separator line.
        width: Total width of the separator line.
    """
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform division with zero-division protection.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if denominator is zero.

    Returns:
        Result of division or default value.
    """
    if denominator == 0:
        return default
    return numerator / denominator
