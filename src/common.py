from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    config_file = resolve_path(config_path)
    with config_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(path_like: str | Path) -> None:
    resolve_path(path_like).parent.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    ensure_parent_dir(output_path)
    with resolve_path(output_path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(input_path: str | Path) -> dict[str, Any]:
    with resolve_path(input_path).open("r", encoding="utf-8") as f:
        return json.load(f)

