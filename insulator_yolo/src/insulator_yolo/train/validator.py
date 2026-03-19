from __future__ import annotations

from pathlib import Path
from typing import Any


def build_val_kwargs(config: dict[str, Any], dataset_yaml_path: str | Path) -> dict[str, Any]:
    return {
        "data": str(dataset_yaml_path),
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "device": config["device"],
    }


def validate_model(
    weights_path: str | Path, config: dict[str, Any], dataset_yaml_path: str | Path
) -> Any:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    return model.val(**build_val_kwargs(config, dataset_yaml_path))
