from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "project.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = PROJECT_ROOT
    cfg["_config_path"] = cfg_path
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    return Path(cfg["_root"]) / cfg["paths"][key]


def gee_project(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_config()
    return str(cfg["project"]["gee_project"])
