from __future__ import annotations

import math
import random
from collections import defaultdict

from insulator_yolo.data.source_dataset import SourceRecord


def assign_grouped_splits(
    records: list[SourceRecord], val_fraction: float, seed: int
) -> dict[str, str]:
    grouped: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.base_sample_id].append(record)

    base_ids = sorted(grouped)
    shuffled = base_ids[:]
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) <= 1:
        val_base_ids: set[str] = set()
    else:
        val_count = max(1, min(len(shuffled) - 1, math.ceil(len(shuffled) * val_fraction)))
        val_base_ids = set(shuffled[:val_count])

    assignment: dict[str, str] = {}
    for base_id, base_records in grouped.items():
        split = "val" if base_id in val_base_ids else "train"
        for record in base_records:
            assignment[record.stem] = split
    return assignment
