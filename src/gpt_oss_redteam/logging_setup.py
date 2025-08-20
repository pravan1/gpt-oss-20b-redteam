"""Logging setup for the red-teaming framework."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_str: Optional[str] = None,
    use_rich: bool = True,
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        format_str: Custom format string for logging
        use_rich: Whether to use rich formatting for console output
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = []
    
    # Console handler with rich formatting
    if use_rich:
        console = Console(stderr=True)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
        )
        rich_handler.setLevel(log_level)
        handlers.append(rich_handler)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(format_str))
        handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(format_str))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )
    
    # Set third-party loggers to WARNING
    for logger_name in ["transformers", "torch", "urllib3", "httpx"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)