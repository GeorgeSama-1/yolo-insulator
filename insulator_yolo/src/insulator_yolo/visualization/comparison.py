from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import yaml
from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = {
    0: "green",
    1: "red",
    2: "blue",
}


def select_split_images(
    image_paths: list[Path], limit: int, seed: int
) -> list[Path]:
    ordered_paths = sorted(image_paths, key=lambda path: path.name)
    if limit <= 0 or limit >= len(ordered_paths):
        return ordered_paths
    return random.Random(seed).sample(ordered_paths, limit)


def parse_yolo_label_line(
    line: str, image_size: tuple[int, int]
) -> tuple[int, list[int]]:
    class_id_text, x_center_text, y_center_text, width_text, height_text = line.split()
    image_width, image_height = image_size
    x_center = float(x_center_text) * image_width
    y_center = float(y_center_text) * image_height
    width = float(width_text) * image_width
    height = float(height_text) * image_height
    x1 = int(round(x_center - width / 2))
    y1 = int(round(y_center - height / 2))
    x2 = int(round(x_center + width / 2))
    y2 = int(round(y_center + height / 2))
    return int(class_id_text), [x1, y1, x2, y2]


def load_class_names(prepared_root: str | Path) -> dict[int, str]:
    with (Path(prepared_root) / "dataset.yaml").open("r", encoding="utf-8") as handle:
        dataset = yaml.safe_load(handle)
    names = dataset.get("names", {})
    return {int(key): value for key, value in names.items()}


def load_gt_boxes(label_path: Path, image_size: tuple[int, int]) -> list[tuple[int, list[int]]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[int, list[int]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        boxes.append(parse_yolo_label_line(stripped, image_size))
    return boxes


def _draw_panel(
    image: Image.Image,
    boxes: list[tuple[int, list[int]]] | list[tuple[int, list[int], float]],
    class_names: dict[int, str],
    title: str,
) -> Image.Image:
    panel = image.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, panel.width - 1, 20], fill="black")
    draw.text((4, 4), title, fill="white", font=font)

    for item in boxes:
        if len(item) == 2:
            class_id, box = item
            score = None
        else:
            class_id, box, score = item
        color = CLASS_COLORS.get(class_id, "yellow")
        draw.rectangle(box, outline=color, width=2)
        label = class_names.get(class_id, str(class_id))
        if score is not None:
            label = f"{label} {score:.2f}"
        draw.text((box[0] + 2, max(22, box[1] + 2)), label, fill=color, font=font)
    return panel


def save_comparison_image(
    *,
    image_path: Path,
    gt_boxes: list[tuple[int, list[int]]],
    pred_boxes: list[tuple[int, list[int], float]],
    class_names: dict[int, str],
    output_path: Path,
    title_suffix: str,
) -> None:
    original = Image.open(image_path).convert("RGB")
    original_panel = _draw_panel(original, [], class_names, f"Original | {title_suffix}")
    gt_panel = _draw_panel(original, gt_boxes, class_names, f"Ground Truth | {title_suffix}")
    pred_panel = _draw_panel(original, pred_boxes, class_names, f"Prediction | {title_suffix}")

    canvas = Image.new(
        "RGB",
        (original.width * 3, original.height),
        "white",
    )
    canvas.paste(original_panel, (0, 0))
    canvas.paste(gt_panel, (original.width, 0))
    canvas.paste(pred_panel, (original.width * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def comparison_output_name(image_path: Path) -> str:
    return f"{image_path.stem}_comparison.jpg"


def _default_predictor_factory(weights_path: Path, predict_kwargs: dict):
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    def predict(image_path: Path) -> list[tuple[int, list[int], float]]:
        result = model.predict(source=str(image_path), save=False, verbose=False, **predict_kwargs)[0]
        output: list[tuple[int, list[int], float]] = []
        if result.boxes is None:
            return output
        xyxy = result.boxes.xyxy.cpu().tolist()
        cls = result.boxes.cls.cpu().tolist()
        conf = result.boxes.conf.cpu().tolist()
        for box, class_id, score in zip(xyxy, cls, conf):
            output.append((int(class_id), [int(round(v)) for v in box], float(score)))
        return output

    return predict


def generate_comparisons(
    *,
    prepared_root: Path,
    split: str,
    limit: int,
    seed: int,
    output_dir: Path,
    weights_path: Path,
    predict_kwargs: dict,
    predictor: Callable[[Path], list[tuple[int, list[int], float]]] | None = None,
) -> list[Path]:
    image_dir = prepared_root / "images" / split
    label_dir = prepared_root / "labels" / split
    class_names = load_class_names(prepared_root)
    image_paths = sorted(image_dir.glob("*"))
    selected = select_split_images(image_paths, limit=limit, seed=seed)
    predictor_fn = predictor or _default_predictor_factory(weights_path, predict_kwargs)

    generated: list[Path] = []
    for image_path in selected:
        image = Image.open(image_path)
        gt_boxes = load_gt_boxes(label_dir / f"{image_path.stem}.txt", image.size)
        pred_boxes = predictor_fn(image_path)
        output_path = output_dir / comparison_output_name(image_path)
        save_comparison_image(
            image_path=image_path,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            class_names=class_names,
            output_path=output_path,
            title_suffix=f"{image_path.name} | {split}",
        )
        generated.append(output_path)
    return generated
