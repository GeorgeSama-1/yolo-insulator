from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_package_imports() -> None:
    import insulator_yolo  # noqa: F401


def test_prepare_dataset_command_generates_dataset(
    tmp_path: Path, sample_source_dataset: tuple[Path, Path]
) -> None:
    dataset_root, _ = sample_source_dataset
    output_dir = tmp_path / "prepared"
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"source_dataset_root: {dataset_root}",
                "source_images_dir: Train/Images",
                "source_labels_path: Train/labels_v1.2.json",
                f"output_dir: {output_dir}",
                "val_fraction: 0.5",
                "seed: 7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = dict(**{"PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")})
    result = subprocess.run(
        [sys.executable, "scripts/prepare_dataset.py", "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "dataset.yaml").exists()
    assert (output_dir / "summary.json").exists()
