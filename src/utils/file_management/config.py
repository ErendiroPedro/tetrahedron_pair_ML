import os
from typing import Any, Dict

import yaml

PATH_EXTENSIONS = (".csv", ".pt", ".pth", ".ipynb")


def load_config(config_file: str) -> Dict[str, Any]:
    """Read a YAML config and resolve its paths against the `home` entry."""
    with open(config_file, "r") as f:
        return resolve_paths(yaml.safe_load(f))


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Join path-like string values onto `config['home']` unless they are already absolute."""
    home = config.get("home")
    if not home:
        raise ValueError("The 'home' path is missing from the configuration.")

    def _walk(node):
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, str) and _looks_like_path(node):
            return node if os.path.isabs(node) else os.path.join(home, node.lstrip("/"))
        return node

    return _walk(config)


def resolve_path(path: str, home: str) -> str:
    """Join one path onto `home` unless it is already absolute."""
    return path if os.path.isabs(path) else os.path.join(home, path)


def _looks_like_path(value: str) -> bool:
    return value.startswith("/") or "./" in value or any(ext in value for ext in PATH_EXTENSIONS)
