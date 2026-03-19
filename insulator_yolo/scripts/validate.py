from __future__ import annotations

import argparse

from insulator_yolo.config import load_yaml_config
from insulator_yolo.train.validator import validate_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    validate_model(args.weights, config, config["dataset_yaml"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
