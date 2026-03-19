from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from insulator_yolo.config import load_yaml_config
from insulator_yolo.visualization.comparison import generate_comparisons


def main(
    argv: list[str] | None = None,
    predictor: Callable[[Path], list[tuple[int, list[int], float]]] | None = None,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--weights")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-dir")
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    split = args.split or config.get("split", "val")
    limit = args.limit if args.limit is not None else int(config.get("limit", 20))
    seed = args.seed if args.seed is not None else int(config.get("seed", 7))
    output_dir = Path(args.save_dir or config["output_dir"])
    weights_path = Path(args.weights or config["weights"])
    prepared_root = Path(config["prepared_root"])
    predict_kwargs = {
        key: value
        for key, value in config.items()
        if key in {"conf", "iou", "max_det", "classes"}
    }

    generate_comparisons(
        prepared_root=prepared_root,
        split=split,
        limit=limit,
        seed=seed,
        output_dir=output_dir,
        weights_path=weights_path,
        predict_kwargs=predict_kwargs,
        predictor=predictor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
