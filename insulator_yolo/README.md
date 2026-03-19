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

## Comparison Visualization

Use the comparison tool when you want to inspect labeled samples side by side as:

1. original image
2. ground-truth annotations
3. model prediction

Default behavior:

- reads the prepared YOLO dataset
- uses the `val` split
- samples `20` images deterministically
- writes outputs under `artifacts/runs/comparisons/`

Example:

```bash
PYTHONPATH=src python3 scripts/visualize_comparison.py --config configs/compare.yaml
```

Common overrides:

```bash
PYTHONPATH=src python3 scripts/visualize_comparison.py \
  --config configs/compare.yaml \
  --split train \
  --limit 10 \
  --weights /path/to/best.pt \
  --save-dir artifacts/runs/comparisons/manual_check
```

This workflow complements validation metrics, but it does not replace quantitative evaluation such as mAP.

## Artifacts

- Prepared dataset: `artifacts/processed/`
- Training runs: `artifacts/runs/`
