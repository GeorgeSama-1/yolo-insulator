from __future__ import annotations

from pathlib import Path

from PIL import Image


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


def test_filter_boxes_by_classes_hides_normal_class() -> None:
    from insulator_yolo.visualization.comparison import filter_boxes_by_classes

    boxes = [
        (0, [1, 1, 10, 10]),
        (1, [2, 2, 11, 11]),
        (2, [3, 3, 12, 12]),
    ]

    filtered = filter_boxes_by_classes(boxes, allowed_classes=[1, 2])

    assert filtered == [
        (1, [2, 2, 11, 11]),
        (2, [3, 3, 12, 12]),
    ]


def test_filter_prediction_boxes_by_classes_hides_normal_class() -> None:
    from insulator_yolo.visualization.comparison import filter_boxes_by_classes

    boxes = [
        (0, [1, 1, 10, 10], 0.99),
        (1, [2, 2, 11, 11], 0.95),
        (2, [3, 3, 12, 12], 0.88),
    ]

    filtered = filter_boxes_by_classes(boxes, allowed_classes=[1, 2])

    assert filtered == [
        (1, [2, 2, 11, 11], 0.95),
        (2, [3, 3, 12, 12], 0.88),
    ]


def test_compute_render_style_scales_for_large_images() -> None:
    from insulator_yolo.visualization.comparison import compute_render_style

    style = compute_render_style((3680, 2456))

    assert style["line_width"] > 2
    assert style["title_height"] > 20
    assert style["font_size"] > 10


def test_prepared_dataset_fixture_has_images_and_labels(prepared_dataset_fixture: Path) -> None:
    assert (prepared_dataset_fixture / "images" / "val").exists()
    assert (prepared_dataset_fixture / "labels" / "val").exists()


def test_save_comparison_image_writes_output(tmp_path: Path) -> None:
    from insulator_yolo.visualization.comparison import save_comparison_image

    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 24), "white").save(image_path)
    output_path = tmp_path / "sample_compare.jpg"

    save_comparison_image(
        image_path=image_path,
        gt_boxes=[(1, [4, 4, 16, 20])],
        pred_boxes=[(2, [8, 2, 20, 18], 0.9)],
        class_names={0: "normal", 1: "broken", 2: "flash"},
        output_path=output_path,
        title_suffix="sample.jpg | val",
    )

    assert output_path.exists()


def test_visualize_comparison_cli_generates_image(
    prepared_dataset_fixture: Path, tmp_path: Path
) -> None:
    from scripts.visualize_comparison import main

    config_path = tmp_path / "compare.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"prepared_root: {prepared_dataset_fixture}",
                "split: val",
                "limit: 1",
                "seed: 7",
                f"output_dir: {tmp_path / 'out'}",
                f"weights: {tmp_path / 'fake.pt'}",
                "conf: 0.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_predictor(image_path: Path):
        return [(1, [1, 1, 8, 8], 0.95)]

    exit_code = main(["--config", str(config_path)], predictor=fake_predictor)

    assert exit_code == 0
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 1
