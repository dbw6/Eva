from pathlib import Path
from typing import Any

import yaml


SIMULATOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SIMULATOR_ROOT.parent
CONFIG_ROOT = SIMULATOR_ROOT / "configs"
NOTEBOOK_ROOT = REPO_ROOT / "notebooks"
REFERENCE_ROOT = SIMULATOR_ROOT / "references"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_yaml_directory(path: Path, top_level_key: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for yaml_path in sorted(path.glob("*.yaml")):
        data = load_yaml(yaml_path)
        merged.update(data.get(top_level_key, {}))
    return merged


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_hardware_config(report_name: str = "table_vi_fig10", config_path: str | Path | None = None) -> dict:
    path = resolve_repo_path(config_path or (CONFIG_ROOT / "hardware.yaml"))
    payload = load_yaml(path)
    return payload["hardware_reports"][report_name]
