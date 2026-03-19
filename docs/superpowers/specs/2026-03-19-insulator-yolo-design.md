# Insulator Defect Detection Training Design

## Context

This design defines a reusable training project for insulator defect detection using the dataset at `/mnt/f/yolo-cls/insulator-defect-detection`.

The implementation will live in a new project directory:

- `insulator_yolo/`

The project will use Ultralytics YOLO and preserve the source dataset in place instead of modifying it.

## Dataset Audit Summary

The source dataset is not a ready-made YOLO dataset and is not a native image-classification dataset.

- Training images: `1600`
- Base samples: `400`
- Derived variants per base sample: `4`
  - `orig`
  - `d`
  - `h`
  - `v`
- Official test images: `88`
- Training annotations: `Train_IDID_V1.2/Train/labels_v1.2.json`
- Annotation style: object detection with per-instance bounding boxes and condition metadata

Observed annotation characteristics:

- Each training image contains multiple insulator instances.
- `d/h/v` are derived variants of the same base image rather than separate labels.
- Annotation objects with `string=0` represent individual insulators used for learning targets.
- Annotation objects with `string=1` and no `conditions` appear to be helper boxes covering an insulator string rather than trainable targets.
- The official test set appears unlabeled and should be treated as inference-only input.

Implication:

- The project should train an object detection model, not a classification model.
- Dataset splitting must group derived variants by base sample to prevent leakage.

## Product Goal

Build a small but reusable YOLO training project that:

- converts the source JSON annotations into YOLO detection format,
- trains a multi-class insulator defect detector,
- validates on a grouped holdout split,
- runs inference on the official unlabeled test set or any user-provided directory,
- keeps data preparation, training, and inference separated behind stable CLI entry points.

## Non-Goals

The first version will not include:

- instance segmentation,
- model serving,
- active learning,
- web UI,
- experiment tracking services,
- multi-dataset federation.

## User Decisions Captured

- Task type: object detection
- Framework: Ultralytics YOLO
- Label strategy: multi-class defect detection
- Delivery scope: engineered workflow with reusable configuration and CLI
- Evaluation strategy: split grouped train/val from `Train_IDID_V1.2`, use official `Test_IDID` for unlabeled inference
- Project location: `/mnt/f/yolo-cls/insulator_yolo/`

## Architecture

The project is organized into four layers.

### 1. Configuration Layer

Configuration files define:

- source dataset paths,
- output locations,
- label mapping rules,
- grouped split parameters,
- training hyperparameters,
- inference parameters.

This keeps machine-specific paths and experiment choices out of the code.

### 2. Data Preparation Layer

This layer reads `labels_v1.2.json`, validates annotation structure, normalizes labels, creates grouped train/val splits, converts boxes into YOLO format, and emits a standard YOLO dataset plus summary artifacts.

This layer is the only part of the system that understands the original annotation schema.

### 3. Training and Validation Layer

This layer consumes prepared YOLO data and configuration values, then delegates the training and validation loop to Ultralytics YOLO.

This layer does not know about raw JSON annotations.

### 4. Prediction Layer

This layer loads trained weights and performs inference on unlabeled images, storing rendered outputs and raw prediction artifacts under a predictable directory structure.

## Planned Directory Structure

```text
/mnt/f/yolo-cls/
├── docs/
│   └── superpowers/
│       ├── specs/
│       └── plans/
├── insulator-defect-detection/
└── insulator_yolo/
    ├── pyproject.toml
    ├── README.md
    ├── configs/
    │   ├── dataset.yaml
    │   ├── train.yaml
    │   └── predict.yaml
    ├── scripts/
    │   ├── prepare_dataset.py
    │   ├── train.py
    │   ├── validate.py
    │   └── predict.py
    ├── src/
    │   └── insulator_yolo/
    │       ├── __init__.py
    │       ├── config.py
    │       ├── logging_utils.py
    │       ├── data/
    │       │   ├── __init__.py
    │       │   ├── source_dataset.py
    │       │   ├── label_mapping.py
    │       │   ├── split.py
    │       │   ├── yolo_export.py
    │       │   └── summary.py
    │       ├── train/
    │       │   ├── __init__.py
    │       │   ├── trainer.py
    │       │   ├── validator.py
    │       │   └── predictor.py
    │       └── cli/
    │           ├── __init__.py
    │           └── common.py
    ├── tests/
    │   ├── data/
    │   │   ├── test_label_mapping.py
    │   │   ├── test_group_split.py
    │   │   ├── test_yolo_export.py
    │   │   └── fixtures/
    │   └── smoke/
    │       └── test_prepare_dataset_smoke.py
    └── artifacts/
        ├── processed/
        └── runs/
```

## Label Mapping Rules

The prepared YOLO dataset will contain three detection classes:

- `normal_insulator`
- `broken_shell`
- `flashover_damage`

Mapping rules:

- Keep only annotation objects with `string=0` as trainable object instances.
- Drop helper objects with `string=1` and missing `conditions`.
- Map `{"No issues": "No issues"}` to `normal_insulator`.
- Map `{"shell": "Broken"}` to `broken_shell`.
- Map `{"glaze": "Flashover damage"}` to `flashover_damage`.
- Map `{"notbroken-notflashed": "notbroken-notflashed"}` to `normal_insulator`.

If a single object contains multiple non-normal conditions, export exactly one class label using this priority:

1. `broken_shell`
2. `flashover_damage`
3. `normal_insulator`

Whenever this priority rule is applied, the exporter must record the affected image and object in the summary output.

## Split Strategy

The prepared dataset must be split by base sample rather than by image filename.

Examples:

- `100228.JPG`
- `100228d.JPG`
- `100228h.JPG`
- `100228v.JPG`

These four files belong to the same split group and must never be separated across training and validation.

Default split behavior:

- grouped `train/val`
- deterministic
- configurable random seed
- default validation fraction: `0.2`

The official `Test_IDID/Test` directory is not used for quantitative evaluation in v1 because no labels were found during the dataset audit.

## Data Preparation Outputs

The preparation command must create:

- YOLO `images/` and `labels/` directories for `train` and `val`
- `dataset.yaml`
- split manifests listing grouped sample assignments
- a machine-readable summary file
- a human-readable summary file

Summary content must include:

- source image count
- source object count
- dropped helper box count
- exported object count per class
- anomaly count
- multi-condition conflict count
- missing-file count
- split sizes by grouped base sample and image file

## CLI Contract

The project exposes four stable entry points.

### `prepare_dataset.py`

Responsibilities:

- validate source paths,
- parse source annotations,
- convert annotations,
- create grouped splits,
- export YOLO labels and dataset config,
- emit summary artifacts.

### `train.py`

Responsibilities:

- load training config,
- ensure prepared dataset exists,
- invoke Ultralytics training,
- store run outputs under `artifacts/runs/`.

### `validate.py`

Responsibilities:

- run validation on a prepared dataset and selected weights,
- persist summarized metrics for later review.

### `predict.py`

Responsibilities:

- run inference on the official test set or a user-supplied directory,
- save rendered images and raw prediction outputs.

## Error Handling

The system should prefer early, explicit failures.

Fatal errors:

- missing source files,
- unreadable annotation JSON,
- malformed bounding boxes that cannot be normalized,
- unsupported annotation objects that violate required schema assumptions.

Recoverable issues that should be logged and summarized:

- dropped helper objects,
- multi-condition objects resolved via priority rule,
- images with zero exported targets,
- unknown but ignorable metadata fields.

The user should always be able to identify the exact failing filename or object source when a fatal error occurs.

## Testing Strategy

The implementation must include both focused unit tests and a light smoke test.

Unit tests:

- base sample grouping from `orig/d/h/v`
- condition-to-class mapping
- helper box filtering
- pixel bbox to YOLO bbox conversion
- deterministic grouped split behavior

Smoke test:

- run dataset preparation on a tiny fixture dataset
- verify YOLO directory creation
- verify split manifests and summary outputs
- verify no grouped leakage across train and val

## Success Criteria

The implementation is successful when:

- one command prepares a valid YOLO dataset from the source JSON,
- one command starts training with configurable YOLO parameters,
- one command runs validation for chosen weights,
- one command performs prediction on the official test set,
- grouped leakage is prevented,
- all outputs land in stable, documented directories,
- the code cleanly separates raw-data adaptation from model operations.

## Risks and Mitigations

### Risk: Misinterpreting helper boxes as trainable targets

Mitigation:

- explicitly filter `string=1` objects with missing conditions,
- report dropped counts in the summary.

### Risk: Leakage between train and val through derived variants

Mitigation:

- split on grouped base sample IDs rather than filenames.

### Risk: Hidden schema surprises in JSON annotations

Mitigation:

- centralize schema parsing and validation in the data layer,
- fail early with sample-specific messages.

### Risk: Low signal from unlabeled official test set

Mitigation:

- treat it as inference-only input,
- keep grouped validation inside the labeled training set.

## Open Assumptions

These assumptions are intentionally fixed for v1 to keep scope controlled:

- a lightweight YOLO checkpoint will be configurable in `train.yaml`,
- validation uses grouped holdout rather than a labeled external test set,
- three detection classes are sufficient for the current dataset.

If any of these change later, the project structure remains reusable because label mapping, split logic, and model configuration are isolated.
