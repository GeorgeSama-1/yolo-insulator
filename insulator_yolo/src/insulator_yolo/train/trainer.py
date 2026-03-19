from __future__ import annotations

from pathlib import Path
from typing import Any


_TRAIN_CONFIG_RESERVED_KEYS = {"model", "dataset_yaml", "data"}


def build_train_kwargs(train_config: dict[str, Any], dataset_yaml_path: str | Path) -> dict[str, Any]:
    kwargs = {
        key: value
        for key, value in train_config.items()
        if key not in _TRAIN_CONFIG_RESERVED_KEYS
    }
    kwargs["data"] = str(dataset_yaml_path)
    return kwargs


def train_model(train_config: dict[str, Any], dataset_yaml_path: str | Path) -> Any:
    from ultralytics import YOLO

    model = YOLO(train_config["model"])
    return model.train(**build_train_kwargs(train_config, dataset_yaml_path))
