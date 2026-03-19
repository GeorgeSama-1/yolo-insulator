from __future__ import annotations

import json
from pathlib import Path


def test_bbox_to_yolo_line_normalizes_coordinates() -> None:
    from insulator_yolo.data.yolo_export import bbox_to_yolo_line

    line = bbox_to_yolo_line(class_id=1, bbox=[100, 50, 40, 20], image_size=(200, 100))

    assert line == "1 0.600000 0.600000 0.200000 0.200000"


def test_build_summary_reports_dropped_helper_boxes() -> None:
    from insulator_yolo.data.summary import build_summary

    summary = build_summary(
        source_image_count=3,
        source_object_count=5,
        exported_counts={"normal_insulator": 2},
        dropped_helper_boxes=3,
        anomalies=[],
        split_group_counts={"train": 1, "val": 1},
        split_image_counts={"train": 2, "val": 1},
        multi_condition_conflicts=0,
        missing_files=[],
    )

    assert summary["dropped_helper_boxes"] == 3


def test_export_dataset_writes_train_and_val_labels(
    tmp_path: Path, sample_source_dataset: tuple[Path, Path]
) -> None:
    from insulator_yolo.data.source_dataset import load_source_annotations
    from insulator_yolo.data.split import assign_grouped_splits
    from insulator_yolo.data.yolo_export import export_dataset

    dataset_root, labels_path = sample_source_dataset
    records = load_source_annotations(labels_path)
    assignment = assign_grouped_splits(records, val_fraction=0.5, seed=7)

    export_dataset(
        records=records,
        split_assignment=assignment,
        source_images_dir=dataset_root / "Train" / "Images",
        output_dir=tmp_path / "processed",
    )

    assert (tmp_path / "processed" / "images" / "train").exists()
    assert (tmp_path / "processed" / "labels" / "val").exists()
    assert (tmp_path / "processed" / "dataset.yaml").exists()
    assert (tmp_path / "processed" / "manifests" / "grouped_split.json").exists()

    manifest = json.loads(
        (tmp_path / "processed" / "manifests" / "grouped_split.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["100228"] in {"train", "val"}
