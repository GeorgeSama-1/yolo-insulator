# Comparison Visualization Design

## Context

This design adds a manual inspection tool to the existing project at `/mnt/f/yolo-cls/insulator_yolo/`.

The current project can:

- prepare the dataset,
- train a YOLO detector,
- validate metrics,
- run prediction on unlabeled images.

What it cannot yet do well is help a human compare predictions against ground-truth labels on a labeled split. The user specifically wants a direct way to visually compare:

- the original image,
- the ground-truth annotations,
- the model prediction.

## Goal

Add a dedicated comparison visualization workflow that generates three-panel comparison images for labeled split samples:

- `original`
- `ground_truth`
- `prediction`

The tool must work on the prepared YOLO dataset, default to the validation split, sample a small number of images by default, and load model weights directly to run fresh predictions.

## User Decisions Captured

- Output format: three-panel comparison
- Panels: `original | GT annotation | prediction`
- Supported splits: `train` and `val`
- Default split: `val`
- Prediction source: load weights and run fresh inference
- Default output count: sampled `20` images, configurable
- README update required

## Non-Goals

This feature does not need to:

- evaluate the unlabeled `Test_IDID` set,
- replace quantitative validation metrics,
- compute mAP or confusion matrices,
- add a web UI,
- support segmentation masks,
- perform visual regression testing.

## Why This Exists

Prediction outputs on unlabeled data only support qualitative inspection. They do not answer whether the detector is actually correct.

By comparing predictions directly with validation-set ground truth, the user can immediately see:

- false positives,
- false negatives,
- class mistakes,
- systematic localization issues.

This makes debugging model behavior much faster and more trustworthy than only looking at prediction overlays.

## Architecture

The feature should be implemented as a small, isolated workflow inside the existing project.

### CLI Layer

Add a new script:

- `scripts/visualize_comparison.py`

Responsibilities:

- parse CLI arguments,
- load comparison config,
- resolve split, sample count, weight path, and output directory,
- invoke the comparison generation module,
- print the final output location.

This script should not contain image drawing logic.

### Visualization Module

Add a focused module:

- `src/insulator_yolo/visualization/comparison.py`

Responsibilities:

- enumerate prepared dataset images for `train` or `val`,
- pair each image with its YOLO label file,
- sample a deterministic subset,
- load YOLO weights and run fresh inference on each selected image,
- draw three panels,
- save combined comparison images.

This module should be the only place that knows how the comparison rendering works.

### Config Layer

Add a dedicated config file:

- `configs/compare.yaml`

Responsibilities:

- default prepared dataset root,
- default weights path,
- default split,
- default sample count,
- default random seed,
- default output directory,
- default prediction thresholds for comparison rendering.

This keeps server-specific or experiment-specific comparison settings out of the code.

### README Layer

Update the main README to explain:

- what the comparison tool is for,
- how to run it,
- where outputs go,
- why it complements but does not replace validation metrics.

## Data Sources

The comparison tool should consume the prepared YOLO dataset rather than the original source JSON.

It should read from:

- `artifacts/processed/images/<split>/`
- `artifacts/processed/labels/<split>/`
- `artifacts/processed/dataset.yaml`

This ensures the comparison tool uses exactly the same label representation as training and validation.

Ground-truth panels should be built from YOLO text labels, using the class mapping already defined in the prepared dataset.

Prediction panels should be built from fresh model inference using the configured weights.

## Output Format

Each generated image should contain three equal-width panels in a single horizontal layout:

1. original image
2. ground-truth annotations
3. prediction annotations

Each panel should include a short title rendered into the image:

- `Original`
- `Ground Truth`
- `Prediction`

The composed image should also include a simple footer or header with:

- source filename,
- split name,
- weights filename,
- prediction thresholds used for that run.

Suggested output directory:

- `artifacts/runs/comparisons/<run_name>/`

Suggested default run name format:

- `<weights_stem>_<split>_<limit>`

## Sampling Behavior

Supported splits:

- `train`
- `val`

Default behavior:

- use `val`
- sample `20` images
- deterministic sampling via configurable random seed

If the requested limit exceeds the number of images available in the split, the tool should automatically cap the output to the available image count.

The tool should also support processing all images in a split via a dedicated CLI option or a convention such as `limit: 0`.

## Drawing Rules

Ground-truth rendering should:

- read YOLO labels,
- convert normalized coordinates back into pixel coordinates,
- render class-colored rectangles,
- render compact class text labels.

Prediction rendering should:

- run the selected YOLO weights directly,
- respect configurable inference arguments such as `conf`, `iou`, `classes`, and `max_det`,
- render boxes and labels using the same class color scheme as GT whenever possible.

If a prediction produces no boxes, the prediction panel should still be saved as an empty panel with only the title.

## Error Handling

Fatal errors:

- missing prepared dataset directory,
- missing split image directory,
- missing weights file,
- unreadable image file.

Recoverable issues:

- missing label file for one image,
- malformed or empty label file,
- no predictions for one image.

Recoverable issues should be logged and skipped without aborting the whole batch.

## Testing Strategy

Keep testing lightweight and focused.

### Unit Tests

Add tests for:

- deterministic sample selection,
- conversion from YOLO text labels to drawable pixel boxes,
- comparison output filename rules,
- predictor argument propagation from config.

### Smoke Test

Add one smoke test using fixture images and labels that:

- creates a tiny prepared dataset fixture,
- runs the comparison script on a tiny split,
- verifies that at least one combined comparison image is created.

The smoke test does not need to validate rendered pixels exactly.

## Success Criteria

This feature is successful when:

- one command produces `original | GT | prediction` comparison images,
- the tool works for `train` and `val`,
- the default workflow runs on `val` with a small deterministic sample,
- output images land in a stable directory under `artifacts/runs/comparisons/`,
- README instructions are sufficient for a user to run the tool on a server,
- the feature reuses the prepared YOLO dataset instead of inventing a second label pipeline.

## Risks and Mitigations

### Risk: Comparison tool silently drifts from training labels

Mitigation:

- consume the prepared YOLO dataset directly,
- do not re-parse the original annotation JSON in this feature.

### Risk: Old prediction outputs get mistaken for new results

Mitigation:

- use a separate comparison output directory and explicit run names.

### Risk: Visual clutter makes comparisons hard to read

Mitigation:

- sample a small number of images by default,
- reuse prediction filters like `conf`, `iou`, `classes`, and `max_det`.

### Risk: Overbuilding a simple human-inspection tool

Mitigation:

- keep scope limited to one script, one module, one config, and a README update.
