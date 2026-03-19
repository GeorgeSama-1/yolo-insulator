# Insulator YOLO Detection Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable Ultralytics YOLO project under `insulator_yolo/` that converts the IDID dataset into YOLO format, trains a grouped-split defect detector, validates it, and runs prediction on unlabeled test images.

**Architecture:** The implementation is split into a raw-data adaptation layer, a model-operation layer, and thin CLI entry points. The raw annotation schema is isolated inside the data modules so training, validation, and prediction only consume prepared YOLO artifacts and typed configuration.

**Tech Stack:** Python 3, Ultralytics YOLO, PyYAML, pytest

---

## File Structure

Planned files and responsibilities:

- Create: `insulator_yolo/pyproject.toml`
  - Project metadata and dependencies.
- Create: `insulator_yolo/README.md`
  - Project usage documentation.
- Create: `insulator_yolo/configs/dataset.yaml`
  - Raw dataset paths, split settings, and label mapping options.
- Create: `insulator_yolo/configs/train.yaml`
  - Training defaults such as model, image size, batch size, epochs, and output paths.
- Create: `insulator_yolo/configs/predict.yaml`
  - Prediction defaults such as weight path, source path, and confidence threshold.
- Create: `insulator_yolo/src/insulator_yolo/config.py`
  - Typed config loading helpers.
- Create: `insulator_yolo/src/insulator_yolo/logging_utils.py`
  - Shared logging setup.
- Create: `insulator_yolo/src/insulator_yolo/data/source_dataset.py`
  - Source JSON parsing, source object validation, and base-sample grouping.
- Create: `insulator_yolo/src/insulator_yolo/data/label_mapping.py`
  - Source condition normalization and class ID assignment.
- Create: `insulator_yolo/src/insulator_yolo/data/split.py`
  - Deterministic grouped train/val split logic.
- Create: `insulator_yolo/src/insulator_yolo/data/yolo_export.py`
  - YOLO label normalization and artifact export.
- Create: `insulator_yolo/src/insulator_yolo/data/summary.py`
  - Summary and anomaly report generation.
- Create: `insulator_yolo/src/insulator_yolo/train/trainer.py`
  - Training orchestration using Ultralytics.
- Create: `insulator_yolo/src/insulator_yolo/train/validator.py`
  - Validation orchestration.
- Create: `insulator_yolo/src/insulator_yolo/train/predictor.py`
  - Inference orchestration.
- Create: `insulator_yolo/src/insulator_yolo/cli/common.py`
  - Shared CLI helpers such as path validation.
- Create: `insulator_yolo/scripts/prepare_dataset.py`
  - Dataset preparation entry point.
- Create: `insulator_yolo/scripts/train.py`
  - Training entry point.
- Create: `insulator_yolo/scripts/validate.py`
  - Validation entry point.
- Create: `insulator_yolo/scripts/predict.py`
  - Prediction entry point.
- Create: `insulator_yolo/tests/data/test_label_mapping.py`
  - Unit tests for mapping and filtering.
- Create: `insulator_yolo/tests/data/test_group_split.py`
  - Unit tests for grouped split logic.
- Create: `insulator_yolo/tests/data/test_yolo_export.py`
  - Unit tests for bbox conversion and export.
- Create: `insulator_yolo/tests/data/fixtures/mini_labels.json`
  - Tiny annotation fixture for deterministic tests.
- Create: `insulator_yolo/tests/smoke/test_prepare_dataset_smoke.py`
  - Smoke test for end-to-end dataset preparation.

## Chunk 1: Project Skeleton and Config Plumbing

### Task 1: Create the project skeleton

**Files:**
- Create: `insulator_yolo/pyproject.toml`
- Create: `insulator_yolo/README.md`
- Create: `insulator_yolo/src/insulator_yolo/__init__.py`
- Create: `insulator_yolo/src/insulator_yolo/data/__init__.py`
- Create: `insulator_yolo/src/insulator_yolo/train/__init__.py`
- Create: `insulator_yolo/src/insulator_yolo/cli/__init__.py`

- [ ] **Step 1: Write the failing smoke import test**

```python
def test_package_imports():
    import insulator_yolo
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/smoke/test_prepare_dataset_smoke.py -k package_imports -v`
Expected: FAIL with import or file-not-found errors because the package skeleton does not exist yet.

- [ ] **Step 3: Write the minimal project files**

```toml
[project]
name = "insulator-yolo"
version = "0.1.0"
dependencies = ["ultralytics", "PyYAML", "pytest"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/smoke/test_prepare_dataset_smoke.py -k package_imports -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md src/insulator_yolo/__init__.py src/insulator_yolo/data/__init__.py src/insulator_yolo/train/__init__.py src/insulator_yolo/cli/__init__.py tests/smoke/test_prepare_dataset_smoke.py
git commit -m "chore: initialize insulator yolo project skeleton"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record that the step is blocked by missing repository initialization.

### Task 2: Add configuration files and loaders

**Files:**
- Create: `insulator_yolo/configs/dataset.yaml`
- Create: `insulator_yolo/configs/train.yaml`
- Create: `insulator_yolo/configs/predict.yaml`
- Create: `insulator_yolo/src/insulator_yolo/config.py`
- Test: `insulator_yolo/tests/data/test_label_mapping.py`

- [ ] **Step 1: Write the failing config loading test**

```python
from pathlib import Path

from insulator_yolo.config import load_yaml_config


def test_load_yaml_config_reads_mapping(tmp_path: Path):
    config_path = tmp_path / "sample.yaml"
    config_path.write_text("value: 3\n", encoding="utf-8")
    assert load_yaml_config(config_path)["value"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_label_mapping.py -k load_yaml_config_reads_mapping -v`
Expected: FAIL because `load_yaml_config` is not implemented.

- [ ] **Step 3: Write minimal config loader and default config files**

```python
def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_label_mapping.py -k load_yaml_config_reads_mapping -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/dataset.yaml configs/train.yaml configs/predict.yaml src/insulator_yolo/config.py tests/data/test_label_mapping.py
git commit -m "feat: add config files and yaml loader"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

### Task 3: Add shared logging and CLI path helpers

**Files:**
- Create: `insulator_yolo/src/insulator_yolo/logging_utils.py`
- Create: `insulator_yolo/src/insulator_yolo/cli/common.py`
- Test: `insulator_yolo/tests/data/test_group_split.py`

- [ ] **Step 1: Write the failing helper test**

```python
from pathlib import Path

from insulator_yolo.cli.common import ensure_path_exists


def test_ensure_path_exists_rejects_missing_path(tmp_path: Path):
    missing = tmp_path / "missing"
    try:
        ensure_path_exists(missing, "dataset")
    except FileNotFoundError as exc:
        assert "dataset" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_group_split.py -k ensure_path_exists_rejects_missing_path -v`
Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement minimal helpers**

```python
def ensure_path_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_group_split.py -k ensure_path_exists_rejects_missing_path -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/logging_utils.py src/insulator_yolo/cli/common.py tests/data/test_group_split.py
git commit -m "feat: add shared cli validation helpers"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

## Chunk 2: Source Dataset Parsing and Label Mapping

### Task 4: Parse source annotations into typed records

**Files:**
- Create: `insulator_yolo/src/insulator_yolo/data/source_dataset.py`
- Create: `insulator_yolo/tests/data/fixtures/mini_labels.json`
- Modify: `insulator_yolo/tests/data/test_group_split.py`

- [ ] **Step 1: Write the failing parse test**

```python
def test_load_source_annotations_reads_filenames_and_objects(fixture_labels_path):
    records = load_source_annotations(fixture_labels_path)
    assert records[0].filename == "100228.JPG"
    assert len(records[0].objects) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_group_split.py -k load_source_annotations_reads_filenames_and_objects -v`
Expected: FAIL because the parser is missing.

- [ ] **Step 3: Implement the parser and lightweight record types**

```python
@dataclass
class SourceObject:
    bbox: list[int]
    string: int | None
    conditions: dict[str, str]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_group_split.py -k load_source_annotations_reads_filenames_and_objects -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/data/source_dataset.py tests/data/fixtures/mini_labels.json tests/data/test_group_split.py
git commit -m "feat: parse source dataset annotations"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

### Task 5: Implement base-sample grouping for `orig/d/h/v`

**Files:**
- Modify: `insulator_yolo/src/insulator_yolo/data/source_dataset.py`
- Modify: `insulator_yolo/tests/data/test_group_split.py`

- [ ] **Step 1: Write the failing grouping test**

```python
def test_extract_base_sample_id_groups_variants():
    assert extract_base_sample_id("100228.JPG") == "100228"
    assert extract_base_sample_id("100228h.JPG") == "100228"
    assert extract_base_sample_id("100228v.JPG") == "100228"
    assert extract_base_sample_id("100228d.JPG") == "100228"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_group_split.py -k extract_base_sample_id_groups_variants -v`
Expected: FAIL because grouping logic is not implemented.

- [ ] **Step 3: Implement the grouping helper**

```python
def extract_base_sample_id(filename: str) -> str:
    stem = Path(filename).stem
    return stem[:-1] if stem and stem[-1] in {"d", "h", "v"} else stem
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_group_split.py -k extract_base_sample_id_groups_variants -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/data/source_dataset.py tests/data/test_group_split.py
git commit -m "feat: group derived dataset variants by base sample"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

### Task 6: Implement condition filtering and class mapping

**Files:**
- Create: `insulator_yolo/src/insulator_yolo/data/label_mapping.py`
- Modify: `insulator_yolo/tests/data/test_label_mapping.py`

- [ ] **Step 1: Write the failing mapping tests**

```python
def test_map_conditions_returns_broken_shell_for_shell_broken():
    assert map_conditions({"shell": "Broken"}) == "broken_shell"


def test_map_conditions_treats_notbroken_notflashed_as_normal():
    assert map_conditions({"notbroken-notflashed": "notbroken-notflashed"}) == "normal_insulator"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_label_mapping.py -k map_conditions -v`
Expected: FAIL because the mapping helper does not exist.

- [ ] **Step 3: Implement mapping and helper-box filtering**

```python
def is_trainable_object(source_object) -> bool:
    return not (source_object.string == 1 and not source_object.conditions)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_label_mapping.py -k map_conditions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/data/label_mapping.py tests/data/test_label_mapping.py
git commit -m "feat: map source conditions to yolo classes"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

## Chunk 3: Split, Export, and Summary Artifacts

### Task 7: Implement deterministic grouped train/val split

**Files:**
- Create: `insulator_yolo/src/insulator_yolo/data/split.py`
- Modify: `insulator_yolo/tests/data/test_group_split.py`

- [ ] **Step 1: Write the failing grouped split test**

```python
def test_grouped_split_keeps_variant_family_in_same_partition(records):
    split = grouped_train_val_split(records, val_fraction=0.2, seed=7)
    assert split["100228"] in {"train", "val"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_group_split.py -k grouped_split_keeps_variant_family_in_same_partition -v`
Expected: FAIL because the split helper is not implemented.

- [ ] **Step 3: Implement minimal deterministic grouped split**

```python
def grouped_train_val_split(records, val_fraction, seed):
    grouped = group_records_by_base_id(records)
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_group_split.py -k grouped_split_keeps_variant_family_in_same_partition -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/data/split.py tests/data/test_group_split.py
git commit -m "feat: add deterministic grouped dataset split"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

### Task 8: Convert pixel boxes into YOLO label lines

**Files:**
- Create: `insulator_yolo/src/insulator_yolo/data/yolo_export.py`
- Modify: `insulator_yolo/tests/data/test_yolo_export.py`

- [ ] **Step 1: Write the failing bbox conversion test**

```python
def test_bbox_to_yolo_line_normalizes_coordinates():
    line = bbox_to_yolo_line(class_id=1, bbox=[100, 50, 40, 20], image_size=(200, 100))
    assert line == "1 0.600000 0.600000 0.200000 0.200000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_yolo_export.py -k bbox_to_yolo_line_normalizes_coordinates -v`
Expected: FAIL because the exporter is not implemented.

- [ ] **Step 3: Implement bbox normalization**

```python
def bbox_to_yolo_line(class_id, bbox, image_size):
    x, y, w, h = bbox
    image_w, image_h = image_size
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_yolo_export.py -k bbox_to_yolo_line_normalizes_coordinates -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/data/yolo_export.py tests/data/test_yolo_export.py
git commit -m "feat: normalize source boxes into yolo labels"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

### Task 9: Export prepared YOLO dataset and manifests

**Files:**
- Modify: `insulator_yolo/src/insulator_yolo/data/yolo_export.py`
- Modify: `insulator_yolo/tests/data/test_yolo_export.py`

- [ ] **Step 1: Write the failing export test**

```python
def test_export_dataset_writes_train_and_val_labels(tmp_path):
    export_dataset(...)
    assert (tmp_path / "images" / "train").exists()
    assert (tmp_path / "labels" / "val").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_yolo_export.py -k export_dataset_writes_train_and_val_labels -v`
Expected: FAIL because dataset export is incomplete.

- [ ] **Step 3: Implement file export and manifest writing**

```python
def export_dataset(records, split_assignment, output_dir):
    write_images(...)
    write_labels(...)
    write_dataset_yaml(...)
    write_split_manifests(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_yolo_export.py -k export_dataset_writes_train_and_val_labels -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/data/yolo_export.py tests/data/test_yolo_export.py
git commit -m "feat: export grouped yolo dataset artifacts"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

### Task 10: Generate machine-readable and human-readable summaries

**Files:**
- Create: `insulator_yolo/src/insulator_yolo/data/summary.py`
- Modify: `insulator_yolo/tests/data/test_yolo_export.py`

- [ ] **Step 1: Write the failing summary test**

```python
def test_build_summary_reports_dropped_helper_boxes():
    summary = build_summary(dropped_helper_boxes=3, exported_counts={"normal_insulator": 2})
    assert summary["dropped_helper_boxes"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_yolo_export.py -k build_summary_reports_dropped_helper_boxes -v`
Expected: FAIL because summary generation is missing.

- [ ] **Step 3: Implement summary builders and writers**

```python
def build_summary(...):
    return {
        "dropped_helper_boxes": dropped_helper_boxes,
        ...
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_yolo_export.py -k build_summary_reports_dropped_helper_boxes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/data/summary.py tests/data/test_yolo_export.py
git commit -m "feat: add dataset preparation summaries"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

## Chunk 4: CLI Commands and Model Operations

### Task 11: Build the dataset preparation command

**Files:**
- Create: `insulator_yolo/scripts/prepare_dataset.py`
- Modify: `insulator_yolo/tests/smoke/test_prepare_dataset_smoke.py`

- [ ] **Step 1: Write the failing smoke command test**

```python
def test_prepare_dataset_command_generates_dataset(tmp_path):
    result = subprocess.run([...], check=False)
    assert result.returncode == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/smoke/test_prepare_dataset_smoke.py -k prepare_dataset_command_generates_dataset -v`
Expected: FAIL because the command does not exist.

- [ ] **Step 3: Implement the command wiring**

```python
def main():
    config = load_yaml_config(...)
    records = load_source_annotations(...)
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/smoke/test_prepare_dataset_smoke.py -k prepare_dataset_command_generates_dataset -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_dataset.py tests/smoke/test_prepare_dataset_smoke.py
git commit -m "feat: add dataset preparation command"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

### Task 12: Add training orchestration

**Files:**
- Create: `insulator_yolo/src/insulator_yolo/train/trainer.py`
- Create: `insulator_yolo/scripts/train.py`
- Modify: `insulator_yolo/README.md`

- [ ] **Step 1: Write the failing trainer wiring test**

```python
def test_build_train_kwargs_reads_expected_fields():
    kwargs = build_train_kwargs(train_config, dataset_yaml_path)
    assert kwargs["data"] == str(dataset_yaml_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_label_mapping.py -k build_train_kwargs_reads_expected_fields -v`
Expected: FAIL because the trainer helpers do not exist.

- [ ] **Step 3: Implement minimal trainer wiring**

```python
def train_model(train_config, dataset_yaml_path):
    model = YOLO(train_config["model"])
    return model.train(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_label_mapping.py -k build_train_kwargs_reads_expected_fields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/train/trainer.py scripts/train.py README.md tests/data/test_label_mapping.py
git commit -m "feat: add yolo training command"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

### Task 13: Add validation orchestration

**Files:**
- Create: `insulator_yolo/src/insulator_yolo/train/validator.py`
- Create: `insulator_yolo/scripts/validate.py`
- Modify: `insulator_yolo/README.md`

- [ ] **Step 1: Write the failing validation helper test**

```python
def test_build_val_kwargs_reads_weights_and_dataset():
    kwargs = build_val_kwargs(...)
    assert "data" in kwargs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_label_mapping.py -k build_val_kwargs_reads_weights_and_dataset -v`
Expected: FAIL because the validation helpers do not exist.

- [ ] **Step 3: Implement minimal validation wiring**

```python
def validate_model(weights_path, train_config, dataset_yaml_path):
    model = YOLO(str(weights_path))
    return model.val(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_label_mapping.py -k build_val_kwargs_reads_weights_and_dataset -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/train/validator.py scripts/validate.py README.md tests/data/test_label_mapping.py
git commit -m "feat: add validation command"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

### Task 14: Add prediction orchestration for unlabeled test images

**Files:**
- Create: `insulator_yolo/src/insulator_yolo/train/predictor.py`
- Create: `insulator_yolo/scripts/predict.py`
- Modify: `insulator_yolo/README.md`

- [ ] **Step 1: Write the failing prediction helper test**

```python
def test_build_predict_kwargs_reads_confidence_and_source():
    kwargs = build_predict_kwargs(...)
    assert "source" in kwargs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_label_mapping.py -k build_predict_kwargs_reads_confidence_and_source -v`
Expected: FAIL because prediction helpers do not exist.

- [ ] **Step 3: Implement minimal prediction wiring**

```python
def predict_with_model(weights_path, predict_config):
    model = YOLO(str(weights_path))
    return model.predict(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data/test_label_mapping.py -k build_predict_kwargs_reads_confidence_and_source -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/train/predictor.py scripts/predict.py README.md tests/data/test_label_mapping.py
git commit -m "feat: add prediction command"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

## Chunk 5: End-to-End Verification and Documentation

### Task 15: Expand the README with setup and workflow commands

**Files:**
- Modify: `insulator_yolo/README.md`

- [ ] **Step 1: Write the failing doc checklist**

Add a checklist to the current task notes requiring the README to document:

- environment setup,
- dataset preparation,
- training,
- validation,
- prediction,
- artifact locations.

- [ ] **Step 2: Verify the README is incomplete**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && rg -n "prepare_dataset|train.py|validate.py|predict.py|artifacts" README.md`
Expected: missing entries before the README is completed.

- [ ] **Step 3: Write the minimal complete README**

```markdown
## Workflow
1. Prepare dataset
2. Train
3. Validate
4. Predict
```

- [ ] **Step 4: Verify the README now covers the workflow**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && rg -n "prepare_dataset|train.py|validate.py|predict.py|artifacts" README.md`
Expected: each workflow command and artifact directory is mentioned.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document insulator yolo workflow"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

### Task 16: Run the focused test suite

**Files:**
- Verify: `insulator_yolo/tests/data/test_label_mapping.py`
- Verify: `insulator_yolo/tests/data/test_group_split.py`
- Verify: `insulator_yolo/tests/data/test_yolo_export.py`
- Verify: `insulator_yolo/tests/smoke/test_prepare_dataset_smoke.py`

- [ ] **Step 1: Run the unit tests**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data -v`
Expected: PASS

- [ ] **Step 2: Run the smoke test**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/smoke/test_prepare_dataset_smoke.py -v`
Expected: PASS

- [ ] **Step 3: Run both suites together**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && pytest tests/data tests/smoke -v`
Expected: PASS

- [ ] **Step 4: Record any environment-specific blockers**

Document missing dependencies, absent git repository setup, or dataset-path assumptions directly in the work log or PR notes.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/data tests/smoke
git commit -m "test: verify insulator yolo workflow"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

### Task 17: Run the first real dataset preparation command

**Files:**
- Verify: `insulator_yolo/configs/dataset.yaml`
- Verify: `insulator_yolo/scripts/prepare_dataset.py`
- Verify: `insulator_yolo/artifacts/processed/`

- [ ] **Step 1: Execute preparation against the real dataset**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && python scripts/prepare_dataset.py --config configs/dataset.yaml`
Expected: the command exits successfully and creates processed YOLO artifacts.

- [ ] **Step 2: Verify dataset outputs exist**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && find artifacts/processed -maxdepth 3 | sort`
Expected: `images/train`, `images/val`, `labels/train`, `labels/val`, `dataset.yaml`, manifests, and summary files.

- [ ] **Step 3: Spot-check grouped split outputs**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && rg -n "100228" artifacts/processed`
Expected: all members of the same base sample family appear in only one split manifest.

- [ ] **Step 4: Record dataset summary numbers**

Capture the exported class counts, helper-box drops, and anomaly counts from the generated summary output.

- [ ] **Step 5: Commit**

```bash
git add configs/dataset.yaml scripts/prepare_dataset.py artifacts/processed
git commit -m "feat: generate first prepared yolo dataset"
```

If `/mnt/f/yolo-cls` is not a git repository, skip the commit and record the block.

Plan complete and saved to `docs/superpowers/plans/2026-03-19-insulator-yolo-detection-pipeline.md`. Ready to execute?
