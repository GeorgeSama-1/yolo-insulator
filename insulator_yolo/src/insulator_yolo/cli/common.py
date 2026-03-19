from __future__ import annotations

from pathlib import Path


def ensure_path_exists(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} path does not exist: {resolved}")
    return resolved
