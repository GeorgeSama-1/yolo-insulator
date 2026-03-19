from __future__ import annotations

import argparse

from insulator_yolo.config import load_yaml_config
from insulator_yolo.train.predictor import predict_with_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    predict_with_model(config["weights"], config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
