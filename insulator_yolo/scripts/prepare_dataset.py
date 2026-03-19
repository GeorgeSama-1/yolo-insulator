from __future__ import annotations

import argparse
from pathlib import Path

from insulator_yolo.cli.common import ensure_path_exists
from insulator_yolo.config import load_yaml_config
from insulator_yolo.data.source_dataset import load_source_annotations
from insulator_yolo.data.split import assign_grouped_splits
from insulator_yolo.data.summary import build_summary, write_summary_files
from insulator_yolo.data.yolo_export import export_dataset
from insulator_yolo.logging_utils import get_logger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logger = get_logger(__name__)
    config = load_yaml_config(args.config)

    dataset_root = ensure_path_exists(config["source_dataset_root"], "source_dataset_root")
    source_images_dir = ensure_path_exists(
        dataset_root / config["source_images_dir"], "source_images_dir"
    )
    labels_path = ensure_path_exists(
        dataset_root / config["source_labels_path"], "source_labels_path"
    )
    output_dir = Path(config["output_dir"])

    records = load_source_annotations(labels_path)
    split_assignment = assign_grouped_splits(
        records,
        val_fraction=float(config.get("val_fraction", 0.2)),
        seed=int(config.get("seed", 7)),
    )
    export_stats = export_dataset(
        records=records,
        split_assignment=split_assignment,
        source_images_dir=source_images_dir,
        output_dir=output_dir,
    )

    summary = build_summary(
        source_image_count=len(records),
        source_object_count=sum(len(record.objects) for record in records),
        exported_counts=export_stats["exported_counts"],
        dropped_helper_boxes=export_stats["dropped_helper_boxes"],
        anomalies=export_stats["anomalies"],
        split_group_counts=export_stats["split_group_counts"],
        split_image_counts=export_stats["split_image_counts"],
        multi_condition_conflicts=export_stats["multi_condition_conflicts"],
        missing_files=export_stats["missing_files"],
    )
    write_summary_files(summary, output_dir)
    logger.info("Prepared dataset at %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
