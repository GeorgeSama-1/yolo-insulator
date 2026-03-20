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


def filter_boxes_by_classes(
    boxes: list[tuple[int, list[int]]] | list[tuple[int, list[int], float]],
    allowed_classes: list[int] | None,
) -> list[tuple[int, list[int]]] | list[tuple[int, list[int], float]]:
    if not allowed_classes:
        return boxes
    allowed = set(allowed_classes)
    return [box for box in boxes if box[0] in allowed]


def compute_render_style(image_size: tuple[int, int]) -> dict[str, int]:
    image_width, image_height = image_size
    short_edge = min(image_width, image_height)
    line_width = max(3, short_edge // 320)
    font_size = max(14, short_edge // 90)
    title_height = max(28, font_size + 12)
    label_padding = max(3, font_size // 6)
    return {
        "line_width": line_width,
        "font_size": font_size,
        "title_height": title_height,
        "label_padding": label_padding,
    }


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


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
    style = compute_render_style(panel.size)
    font = _load_font(style["font_size"])
    title_height = min(panel.height - 1, style["title_height"])
    line_width = style["line_width"]
    label_padding = style["label_padding"]

    draw.rectangle([0, 0, panel.width - 1, title_height], fill="black")
    draw.text((label_padding, label_padding), title, fill="white", font=font)

    for item in boxes:
        if len(item) == 2:
            class_id, box = item
            score = None
        else:
            class_id, box, score = item
        color = CLASS_COLORS.get(class_id, "yellow")
        draw.rectangle(box, outline=color, width=line_width)
        label = class_names.get(class_id, str(class_id))
        if score is not None:
            label = f"{label} {score:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = max(0, box[0] + label_padding)
        max_text_x = max(0, panel.width - text_width - label_padding - 1)
        max_text_y = max(0, panel.height - text_height - label_padding - 1)
        text_x = min(text_x, max_text_x)
        text_y = min(max(title_height + label_padding, box[1] + label_padding), max_text_y)
        text_box = [
            text_x - label_padding,
            text_y - label_padding,
            max(text_x - label_padding, min(panel.width - 1, text_x + text_width + label_padding)),
            max(text_y - label_padding, min(panel.height - 1, text_y + text_height + label_padding)),
        ]
        draw.rectangle(text_box, fill=color)
        draw.text((text_x, text_y), label, fill="white", font=font)
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
    allowed_classes = predict_kwargs.get("classes")

    generated: list[Path] = []
    for image_path in selected:
        image = Image.open(image_path)
        gt_boxes = filter_boxes_by_classes(
            load_gt_boxes(label_dir / f"{image_path.stem}.txt", image.size),
            allowed_classes=allowed_classes,
        )
        pred_boxes = filter_boxes_by_classes(
            predictor_fn(image_path),
            allowed_classes=allowed_classes,
        )
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
