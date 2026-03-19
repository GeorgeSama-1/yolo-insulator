from __future__ import annotations

from pathlib import Path


def test_compare_config_exists() -> None:
    assert Path("configs/compare.yaml").exists()


def test_select_split_images_returns_deterministic_subset(tmp_path: Path) -> None:
    from insulator_yolo.visualization.comparison import select_split_images

    image_paths = []
    for name in ["b.jpg", "a.jpg", "c.jpg", "d.jpg"]:
        path = tmp_path / name
        path.write_bytes(b"data")
        image_paths.append(path)

    selected = select_split_images(image_paths, limit=2, seed=7)

    assert [path.name for path in selected] == ["c.jpg", "a.jpg"]


def test_parse_yolo_label_line_returns_pixel_box() -> None:
    from insulator_yolo.visualization.comparison import parse_yolo_label_line

    class_id, box = parse_yolo_label_line("1 0.5 0.5 0.2 0.4", image_size=(100, 200))

    assert class_id == 1
    assert box == [40, 60, 60, 140]
