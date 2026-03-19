from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SourceObject:
    bbox: list[int]
    string: int | None
    conditions: dict[str, str] = field(default_factory=dict)
    name: str = "insulator"
    material: str | None = None
    type: str | None = None
    comments: str | None = None


@dataclass(slots=True)
class SourceRecord:
    filename: str
    objects: list[SourceObject]

    @property
    def stem(self) -> str:
        return Path(self.filename).stem

    @property
    def base_sample_id(self) -> str:
        return extract_base_sample_id(self.filename)


def extract_base_sample_id(filename: str) -> str:
    stem = Path(filename).stem
    return stem[:-1] if stem and stem[-1] in {"d", "h", "v"} else stem


def _parse_object(raw_object: dict[str, Any]) -> SourceObject:
    bbox = raw_object.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox!r}")
    return SourceObject(
        bbox=[int(value) for value in bbox],
        string=raw_object.get("string"),
        conditions=dict(raw_object.get("conditions") or {}),
        name=raw_object.get("name", "insulator"),
        material=raw_object.get("material"),
        type=raw_object.get("type"),
        comments=raw_object.get("comments"),
    )


def load_source_annotations(labels_path: str | Path) -> list[SourceRecord]:
    with Path(labels_path).open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    records: list[SourceRecord] = []
    for raw_item in raw_data:
        objects = [
            _parse_object(raw_object)
            for raw_object in raw_item.get("Labels", {}).get("objects", [])
        ]
        records.append(SourceRecord(filename=raw_item["filename"], objects=objects))
    return records
