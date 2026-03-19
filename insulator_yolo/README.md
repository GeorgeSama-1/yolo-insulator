# Insulator YOLO

Reusable Ultralytics YOLO workflow for the IDID insulator defect dataset.

## Workflow

1. Prepare dataset
   - `PYTHONPATH=src python3 scripts/prepare_dataset.py --config configs/dataset.yaml`
2. Train
   - `PYTHONPATH=src python3 scripts/train.py --config configs/train.yaml`
3. Validate
   - `PYTHONPATH=src python3 scripts/validate.py --config configs/train.yaml --weights <weights>`
4. Predict
   - `PYTHONPATH=src python3 scripts/predict.py --config configs/predict.yaml`

## Artifacts

- Prepared dataset: `artifacts/processed/`
- Training runs: `artifacts/runs/`
