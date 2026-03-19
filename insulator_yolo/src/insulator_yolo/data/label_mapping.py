from __future__ import annotations

from collections.abc import Mapping

from insulator_yolo.data.source_dataset import SourceObject

CLASS_NAMES = ["normal_insulator", "broken_shell", "flashover_damage"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

_LABEL_PRIORITY = {
    "normal_insulator": 0,
    "flashover_damage": 1,
    "broken_shell": 2,
}


def is_trainable_object(source_object: SourceObject) -> bool:
    return not (source_object.string == 1 and not source_object.conditions)


def map_conditions(conditions: Mapping[str, str]) -> tuple[str, bool]:
    if not conditions:
        return "normal_insulator", False

    labels: set[str] = set()
    for key, value in conditions.items():
        if key == "No issues" and value == "No issues":
            labels.add("normal_insulator")
        elif key == "shell" and value == "Broken":
            labels.add("broken_shell")
        elif key == "glaze" and value == "Flashover damage":
            labels.add("flashover_damage")
        elif key == "notbroken-notflashed" and value == "notbroken-notflashed":
            labels.add("normal_insulator")
        else:
            raise ValueError(f"Unsupported condition mapping: {key}={value}")

    chosen = max(labels, key=_LABEL_PRIORITY.__getitem__)
    return chosen, len(labels) > 1
