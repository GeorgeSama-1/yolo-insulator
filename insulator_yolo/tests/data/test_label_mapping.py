from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def test_load_yaml_config_reads_mapping(tmp_path: Path) -> None:
    from insulator_yolo.config import load_yaml_config

    config_path = tmp_path / "sample.yaml"
    config_path.write_text("value: 3\nname: demo\n", encoding="utf-8")

    assert load_yaml_config(config_path)["value"] == 3


@dataclass
class DummyObject:
    string: int | None
    conditions: dict[str, str]


def test_map_conditions_returns_broken_shell_for_shell_broken() -> None:
    from insulator_yolo.data.label_mapping import map_conditions

    assert map_conditions({"shell": "Broken"})[0] == "broken_shell"


def test_map_conditions_treats_notbroken_notflashed_as_normal() -> None:
    from insulator_yolo.data.label_mapping import map_conditions

    assert (
        map_conditions({"notbroken-notflashed": "notbroken-notflashed"})[0]
        == "normal_insulator"
    )


def test_is_trainable_object_filters_helper_boxes() -> None:
    from insulator_yolo.data.label_mapping import is_trainable_object

    assert not is_trainable_object(DummyObject(string=1, conditions={}))
    assert is_trainable_object(DummyObject(string=0, conditions={"shell": "Broken"}))


def test_build_train_kwargs_reads_expected_fields(tmp_path: Path) -> None:
    from insulator_yolo.train.trainer import build_train_kwargs

    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("path: demo\n", encoding="utf-8")
    train_config = {
        "model": "yolo11n.pt",
        "imgsz": 640,
        "epochs": 10,
        "batch": 8,
        "workers": 2,
        "device": "cpu",
        "project": "artifacts/runs",
        "name": "demo",
        "amp": False,
        "patience": 20,
        "cache": True,
        "cos_lr": True,
    }

    kwargs = build_train_kwargs(train_config, dataset_yaml)

    assert kwargs["data"] == str(dataset_yaml)
    assert kwargs["epochs"] == 10
    assert kwargs["amp"] is False
    assert kwargs["patience"] == 20
    assert kwargs["cache"] is True
    assert kwargs["cos_lr"] is True
    assert "model" not in kwargs
    assert "dataset_yaml" not in kwargs


def test_build_val_kwargs_reads_weights_and_dataset(tmp_path: Path) -> None:
    from insulator_yolo.train.validator import build_val_kwargs

    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("path: demo\n", encoding="utf-8")
    config = {"imgsz": 640, "batch": 4, "device": "cpu"}

    kwargs = build_val_kwargs(config, dataset_yaml)

    assert kwargs["data"] == str(dataset_yaml)
    assert kwargs["imgsz"] == 640


def test_build_predict_kwargs_reads_confidence_and_source(tmp_path: Path) -> None:
    from insulator_yolo.train.predictor import build_predict_kwargs

    source_dir = tmp_path / "images"
    source_dir.mkdir()
    config = {"source": str(source_dir), "conf": 0.4, "save": True}

    kwargs = build_predict_kwargs(config)

    assert kwargs["source"] == str(source_dir)
    assert kwargs["conf"] == 0.4
