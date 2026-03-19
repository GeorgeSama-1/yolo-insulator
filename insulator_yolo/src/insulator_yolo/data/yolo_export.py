from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from insulator_yolo.data.label_mapping import CLASS_NAMES, CLASS_TO_ID, is_trainable_object, map_conditions
from insulator_yolo.data.source_dataset import SourceRecord


def bbox_to_yolo_line(class_id: int, bbox: list[int], image_size: tuple[int, int]) -> str:
    x, y, width, height = bbox
    image_width, image_height = image_size
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    return (
        f"{class_id} "
        f"{x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
    )


def _read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def export_dataset(
    *,
    records: list[SourceRecord],
    split_assignment: dict[str, str],
    source_images_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_images_path = Path(source_images_dir)
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    manifests_dir = output_path / "manifests"
    for split in ("train", "val"):
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    exported_counts: Counter[str] = Counter()
    dropped_helper_boxes = 0
    anomalies: list[str] = []
    missing_files: list[str] = []
    multi_condition_conflicts = 0

    base_to_split: dict[str, str] = {}
    split_image_counts: Counter[str] = Counter()

    for record in records:
        split = split_assignment[record.stem]
        base_to_split[record.base_sample_id] = split
        split_image_counts[split] += 1

        source_image = source_images_path / record.filename
        if not source_image.exists():
            missing_files.append(record.filename)
            continue

        shutil.copy2(source_image, images_dir / split / record.filename)
        image_size = _read_image_size(source_image)

        label_lines: list[str] = []
        for index, source_object in enumerate(record.objects):
            if not is_trainable_object(source_object):
                dropped_helper_boxes += 1
                continue

            label_name, had_conflict = map_conditions(source_object.conditions)
            if had_conflict:
                multi_condition_conflicts += 1
                anomalies.append(f"{record.filename}#{index}: multi-condition object")
            exported_counts[label_name] += 1
            label_lines.append(
                bbox_to_yolo_line(
                    class_id=CLASS_TO_ID[label_name],
                    bbox=source_object.bbox,
                    image_size=image_size,
                )
            )

        label_path = labels_dir / split / f"{record.stem}.txt"
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    dataset_yaml = {
        "path": str(output_path),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (output_path / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False),
        encoding="utf-8",
    )
    (manifests_dir / "grouped_split.json").write_text(
        json.dumps(split_assignment, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    split_group_counts = Counter(base_to_split.values())
    return {
        "exported_counts": dict(exported_counts),
        "dropped_helper_boxes": dropped_helper_boxes,
        "anomalies": anomalies,
        "missing_files": missing_files,
        "multi_condition_conflicts": multi_condition_conflicts,
        "split_group_counts": dict(split_group_counts),
        "split_image_counts": dict(split_image_counts),
    }
