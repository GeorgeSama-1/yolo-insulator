from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest


PNG_10X10 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAAEklEQVR4nGP8z4APMOGVHbHSAEEsAROxCnMTAAAAAElFTkSuQmCC"
)


@pytest.fixture
def fixture_labels_path() -> Path:
    return Path(__file__).parent / "data" / "fixtures" / "mini_labels.json"


@pytest.fixture
def sample_source_dataset(tmp_path: Path, fixture_labels_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "source"
    images_dir = dataset_root / "Train" / "Images"
    images_dir.mkdir(parents=True)

    with fixture_labels_path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)

    for item in labels:
        (images_dir / item["filename"]).write_bytes(PNG_10X10)

    labels_path = dataset_root / "Train" / "labels_v1.2.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    return dataset_root, labels_path
