# Comparison Visualization Readability Design

## Context

The existing comparison visualization workflow can generate three-panel images for labeled samples, but the rendered annotations are hard to inspect on large images:

- bounding boxes are too thin,
- labels are too small,
- labels do not have a readable background,
- `normal_insulator` is still shown even when the user only wants to inspect defect classes.

## Goal

Improve the comparison visualization workflow so it is practical for manual inspection on large images.

The updated workflow should:

- hide `normal_insulator` in both the `Ground Truth` and `Prediction` panels,
- default to defect-only classes in comparison config,
- render thicker boxes and more readable labels,
- keep the original image panel unchanged.

## User Decisions Captured

- Only defect classes should be shown.
- `normal_insulator` should be hidden in both GT and prediction panels.
- The fix should stay inside the comparison visualization workflow rather than changing training or generic prediction.

## Design

### Filtering

Comparison rendering will support class filtering through the existing comparison config.

- `configs/compare.yaml` will default to `classes: [1, 2]`
- GT boxes will be filtered using the same class allowlist
- Prediction boxes will continue to pass `classes` into Ultralytics and will also be filtered again before drawing for safety

This keeps GT and prediction visually aligned and makes the workflow focus on defect inspection.

### Readability

Rendering should scale with image size instead of using a fixed tiny style.

The comparison renderer will:

- compute line width from image dimensions,
- compute a readable font size from image dimensions,
- draw a filled label background behind class text,
- keep class-colored boxes and labels,
- show class name for GT and class name plus confidence for prediction.

### Scope Boundaries

This change will not:

- modify training labels,
- change `predict.py`,
- alter validation metrics,
- add new output formats.

## Testing

Add test coverage for:

- filtering out class `0` boxes from GT and prediction when `classes=[1, 2]`,
- preserving defect classes,
- continuing to generate output images successfully.

## Success Criteria

This change is successful when:

- comparison images no longer show `normal_insulator`,
- defect boxes remain visible,
- labels are visibly easier to read on large images,
- the comparison workflow remains configurable and test-covered.
