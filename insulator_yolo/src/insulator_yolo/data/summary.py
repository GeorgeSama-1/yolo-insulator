from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_summary(
    *,
    source_image_count: int,
    source_object_count: int,
    exported_counts: dict[str, int],
    dropped_helper_boxes: int,
    anomalies: list[str],
    split_group_counts: dict[str, int],
    split_image_counts: dict[str, int],
    multi_condition_conflicts: int,
    missing_files: list[str],
) -> dict[str, Any]:
    return {
        "source_image_count": source_image_count,
        "source_object_count": source_object_count,
        "exported_counts": exported_counts,
        "dropped_helper_boxes": dropped_helper_boxes,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "split_group_counts": split_group_counts,
        "split_image_counts": split_image_counts,
        "multi_condition_conflicts": multi_condition_conflicts,
        "missing_file_count": len(missing_files),
        "missing_files": missing_files,
    }


def write_summary_files(summary: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        f"source_image_count: {summary['source_image_count']}",
        f"source_object_count: {summary['source_object_count']}",
        f"dropped_helper_boxes: {summary['dropped_helper_boxes']}",
        f"anomaly_count: {summary['anomaly_count']}",
        f"multi_condition_conflicts: {summary['multi_condition_conflicts']}",
        f"missing_file_count: {summary['missing_file_count']}",
    ]
    (output_path / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
