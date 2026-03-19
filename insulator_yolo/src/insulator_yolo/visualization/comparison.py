from __future__ import annotations

import random
from pathlib import Path


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
