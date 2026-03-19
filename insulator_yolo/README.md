# Insulator YOLO

Reusable Ultralytics YOLO workflow for the IDID insulator defect dataset.

## Workflow

1. Prepare dataset
   - `PYTHONPATH=src python3 scripts/prepare_dataset.py --config configs/dataset.yaml`
2. Train
   - `PYTHONPATH=src python3 scripts/train.py --config configs/train.yaml`
   - Extra Ultralytics training options can be set directly in `configs/train.yaml`, such as `amp`, `patience`, `cache`, and `cos_lr`.
3. Validate
   - `PYTHONPATH=src python3 scripts/validate.py --config configs/train.yaml --weights <weights>`
4. Predict
   - `PYTHONPATH=src python3 scripts/predict.py --config configs/predict.yaml`
   - `configs/predict.yaml` supports direct Ultralytics options such as `classes`, `iou`, and `max_det`.

## Artifacts

- Prepared dataset: `artifacts/processed/`
- Training runs: `artifacts/runs/`
