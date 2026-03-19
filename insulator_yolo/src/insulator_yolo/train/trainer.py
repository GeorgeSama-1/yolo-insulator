from __future__ import annotations

from pathlib import Path
from typing import Any


def build_train_kwargs(train_config: dict[str, Any], dataset_yaml_path: str | Path) -> dict[str, Any]:
    return {
        "data": str(dataset_yaml_path),
        "imgsz": train_config["imgsz"],
        "epochs": train_config["epochs"],
        "batch": train_config["batch"],
        "workers": train_config["workers"],
        "device": train_config["device"],
        "project": train_config["project"],
        "name": train_config["name"],
    }


def train_model(train_config: dict[str, Any], dataset_yaml_path: str | Path) -> Any:
    from ultralytics import YOLO

    model = YOLO(train_config["model"])
    return model.train(**build_train_kwargs(train_config, dataset_yaml_path))
