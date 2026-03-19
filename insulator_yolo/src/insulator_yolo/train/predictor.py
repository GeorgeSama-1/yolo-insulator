from __future__ import annotations

from typing import Any


def build_predict_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "source": config["source"],
        "conf": config["conf"],
        "save": config.get("save", True),
    }
    if "project" in config:
        kwargs["project"] = config["project"]
    if "name" in config:
        kwargs["name"] = config["name"]
    return kwargs


def predict_with_model(weights_path: str, config: dict[str, Any]) -> Any:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    return model.predict(**build_predict_kwargs(config))
