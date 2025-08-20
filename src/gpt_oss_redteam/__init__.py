"""GPT-OSS-20B Red Team Framework for Kaggle hackathon."""

__version__ = "0.1.0"
__author__ = "Red Team"

from .config import Config, load_config
from .logging_setup import setup_logging

__all__ = ["Config", "load_config", "setup_logging", "__version__"]