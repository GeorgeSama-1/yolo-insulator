from __future__ import annotations

from typing import Any


_PREDICT_CONFIG_RESERVED_KEYS = {"weights"}


def build_predict_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        key: value
        for key, value in config.items()
        if key not in _PREDICT_CONFIG_RESERVED_KEYS
    }
    kwargs.setdefault("save", True)
    return kwargs


def predict_with_model(weights_path: str, config: dict[str, Any]) -> Any:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    return model.predict(**build_predict_kwargs(config))
