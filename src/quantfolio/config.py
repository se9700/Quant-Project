from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data_cache", "data/cache")
    cfg.setdefault("report_dir", "reports")
    Path(cfg["data_cache"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["report_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg
