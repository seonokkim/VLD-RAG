"""Shared helpers for loading portable repository configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def get_project_root() -> Path:
    """Return the repository root directory."""
    return PROJECT_ROOT


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return a dictionary payload."""
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.warning(f"Failed to load config from {path}: {exc}")
        return {}


def load_named_config(filename: str) -> Dict[str, Any]:
    """Load a YAML config from the tracked `configs/` directory."""
    return load_yaml_file(CONFIGS_DIR / filename)


def get_data_config() -> Dict[str, Any]:
    """Load the repository data/artifact config."""
    return load_named_config("data.yml")


def get_model_config() -> Dict[str, Any]:
    """Load the repository model config."""
    return load_named_config("model.yml")


def get_database_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load database config from an explicit path or the tracked default."""
    if config_path is not None:
        payload = load_yaml_file(Path(config_path))
        return payload.get("database", payload.get("postgres", {}))

    payload = load_named_config("database.yml")
    if payload:
        return payload.get("database", payload.get("postgres", {}))

    legacy_payload = load_named_config("config.yaml")
    return legacy_payload.get("database", legacy_payload.get("postgres", {}))


def deep_get(payload: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely read a nested config value from a dictionary."""
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_repo_path(path_value: Optional[str]) -> Optional[Path]:
    """Resolve a relative repository path to an absolute filesystem path."""
    if not path_value:
        return None

    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()
