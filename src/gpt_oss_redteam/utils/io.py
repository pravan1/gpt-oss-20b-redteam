"""I/O utility functions."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], path: str) -> None:
    """Save data to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)