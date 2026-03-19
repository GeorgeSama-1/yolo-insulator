from __future__ import annotations

from pathlib import Path


def test_ensure_path_exists_rejects_missing_path(tmp_path: Path) -> None:
    from insulator_yolo.cli.common import ensure_path_exists

    missing = tmp_path / "missing"

    try:
        ensure_path_exists(missing, "dataset")
    except FileNotFoundError as exc:
        assert "dataset" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_load_source_annotations_reads_filenames_and_objects(fixture_labels_path: Path) -> None:
    from insulator_yolo.data.source_dataset import load_source_annotations

    records = load_source_annotations(fixture_labels_path)

    assert records[0].filename == "100228.JPG"
    assert len(records[0].objects) == 3


def test_extract_base_sample_id_groups_variants() -> None:
    from insulator_yolo.data.source_dataset import extract_base_sample_id

    assert extract_base_sample_id("100228.JPG") == "100228"
    assert extract_base_sample_id("100228h.JPG") == "100228"
    assert extract_base_sample_id("100228v.JPG") == "100228"
    assert extract_base_sample_id("100228d.JPG") == "100228"


def test_grouped_split_keeps_variant_family_in_same_partition(fixture_labels_path: Path) -> None:
    from insulator_yolo.data.source_dataset import load_source_annotations
    from insulator_yolo.data.split import assign_grouped_splits

    records = load_source_annotations(fixture_labels_path)
    assignment = assign_grouped_splits(records, val_fraction=0.5, seed=7)

    assert assignment["100228"] == assignment["100228h"]
    assert set(assignment.values()) == {"train", "val"}
