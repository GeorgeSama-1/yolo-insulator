# Comparison Visualization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated comparison visualization tool that generates `original | ground truth | prediction` panels for sampled `train` or `val` images from the prepared YOLO dataset.

**Architecture:** The implementation adds one new visualization module, one CLI entry point, one comparison config, and a small README update. It reuses the prepared YOLO dataset and fresh weight-based inference so comparison outputs stay consistent with training artifacts.

**Tech Stack:** Python 3, Pillow, Ultralytics YOLO, PyYAML, pytest

---

## File Structure

Planned files and responsibilities:

- Create: `insulator_yolo/configs/compare.yaml`
  - Default comparison settings such as weights path, split, sample limit, and output directory.
- Create: `insulator_yolo/src/insulator_yolo/visualization/__init__.py`
  - Package marker for visualization helpers.
- Create: `insulator_yolo/src/insulator_yolo/visualization/comparison.py`
  - Image sampling, GT rendering, prediction rendering, and three-panel composition.
- Create: `insulator_yolo/scripts/visualize_comparison.py`
  - CLI wrapper for comparison generation.
- Modify: `insulator_yolo/README.md`
  - Usage guide for the new comparison workflow.
- Modify: `insulator_yolo/tests/conftest.py`
  - Shared prepared-dataset fixtures for visualization tests.
- Create: `insulator_yolo/tests/visualization/test_comparison.py`
  - Unit tests for sampling, label conversion, and output writing.

## Chunk 1: Comparison Config and Pure Helpers

### Task 1: Add comparison config and package skeleton

**Files:**
- Create: `insulator_yolo/configs/compare.yaml`
- Create: `insulator_yolo/src/insulator_yolo/visualization/__init__.py`
- Test: `insulator_yolo/tests/visualization/test_comparison.py`

- [ ] **Step 1: Write the failing config existence test**

```python
from pathlib import Path


def test_compare_config_exists():
    assert Path("configs/compare.yaml").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k compare_config_exists -v`
Expected: FAIL because the config and test file do not exist yet.

- [ ] **Step 3: Create the minimal config and package marker**

```yaml
split: val
limit: 20
seed: 7
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k compare_config_exists -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/compare.yaml src/insulator_yolo/visualization/__init__.py tests/visualization/test_comparison.py
git commit -m "feat: add comparison visualization config scaffold"
```

### Task 2: Add deterministic image sampling helpers

**Files:**
- Create: `insulator_yolo/src/insulator_yolo/visualization/comparison.py`
- Modify: `insulator_yolo/tests/visualization/test_comparison.py`

- [ ] **Step 1: Write the failing sampling test**

```python
def test_select_split_images_returns_deterministic_subset(tmp_path):
    selected = select_split_images(image_paths, limit=2, seed=7)
    assert [path.name for path in selected] == ["a.jpg", "c.jpg"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k deterministic_subset -v`
Expected: FAIL because the helper is not implemented.

- [ ] **Step 3: Implement the minimal deterministic sampler**

```python
def select_split_images(image_paths, limit, seed):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k deterministic_subset -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/visualization/comparison.py tests/visualization/test_comparison.py
git commit -m "feat: add deterministic comparison sampling"
```

## Chunk 2: Ground-Truth Rendering and Comparison Composition

### Task 3: Convert YOLO label lines back into drawable boxes

**Files:**
- Modify: `insulator_yolo/src/insulator_yolo/visualization/comparison.py`
- Modify: `insulator_yolo/tests/visualization/test_comparison.py`

- [ ] **Step 1: Write the failing GT label conversion test**

```python
def test_parse_yolo_label_line_returns_pixel_box():
    box = parse_yolo_label_line("1 0.5 0.5 0.2 0.4", image_size=(100, 200))
    assert box == (1, [40, 60, 60, 140])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k pixel_box -v`
Expected: FAIL because parsing is not implemented.

- [ ] **Step 3: Implement the parser and box conversion**

```python
def parse_yolo_label_line(line, image_size):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k pixel_box -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/visualization/comparison.py tests/visualization/test_comparison.py
git commit -m "feat: decode yolo labels for gt visualization"
```

### Task 4: Render the three-panel comparison image

**Files:**
- Modify: `insulator_yolo/src/insulator_yolo/visualization/comparison.py`
- Modify: `insulator_yolo/tests/visualization/test_comparison.py`

- [ ] **Step 1: Write the failing output image test**

```python
def test_save_comparison_image_writes_output(tmp_path):
    save_comparison_image(...)
    assert (tmp_path / "sample_compare.jpg").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k writes_output -v`
Expected: FAIL because comparison composition is missing.

- [ ] **Step 3: Implement minimal panel composition**

```python
def save_comparison_image(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k writes_output -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/insulator_yolo/visualization/comparison.py tests/visualization/test_comparison.py
git commit -m "feat: add three-panel comparison rendering"
```

## Chunk 3: CLI Integration and Smoke Test

### Task 5: Add a fixture prepared dataset for visualization smoke tests

**Files:**
- Modify: `insulator_yolo/tests/conftest.py`
- Modify: `insulator_yolo/tests/visualization/test_comparison.py`

- [ ] **Step 1: Write the failing prepared-dataset fixture smoke test**

```python
def test_prepared_dataset_fixture_has_images_and_labels(prepared_dataset_fixture):
    assert (prepared_dataset_fixture / "images" / "val").exists()
    assert (prepared_dataset_fixture / "labels" / "val").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k prepared_dataset_fixture -v`
Expected: FAIL because the fixture does not exist.

- [ ] **Step 3: Add the minimal prepared dataset fixture**

```python
@pytest.fixture
def prepared_dataset_fixture(...):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k prepared_dataset_fixture -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/visualization/test_comparison.py
git commit -m "test: add prepared dataset fixture for comparisons"
```

### Task 6: Add the CLI entry point and smoke test

**Files:**
- Create: `insulator_yolo/scripts/visualize_comparison.py`
- Modify: `insulator_yolo/tests/visualization/test_comparison.py`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
def test_visualize_comparison_cli_generates_image(prepared_dataset_fixture, tmp_path):
    result = subprocess.run([...], check=False)
    assert result.returncode == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k visualize_comparison_cli_generates_image -v`
Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement the minimal CLI wiring**

```python
def main():
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -k visualize_comparison_cli_generates_image -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/visualize_comparison.py tests/visualization/test_comparison.py
git commit -m "feat: add comparison visualization cli"
```

## Chunk 4: Documentation and Full Verification

### Task 7: Update the README with comparison workflow instructions

**Files:**
- Modify: `insulator_yolo/README.md`

- [ ] **Step 1: Write the failing doc checklist**

Add a checklist entry requiring the README to mention:

- comparison purpose,
- comparison command,
- output directory,
- default split and sample count,
- the fact that comparison complements but does not replace metrics.

- [ ] **Step 2: Verify the README is incomplete**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && rg -n "visualize_comparison|comparison|ground truth|original" README.md`
Expected: missing entries before the README is updated.

- [ ] **Step 3: Update the README**

```markdown
## Comparison Visualization
...
```

- [ ] **Step 4: Verify the README now covers the workflow**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && rg -n "visualize_comparison|comparison|ground truth|original" README.md`
Expected: the comparison workflow is documented.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add comparison visualization guide"
```

### Task 8: Run the visualization test suite

**Files:**
- Verify: `insulator_yolo/tests/visualization/test_comparison.py`
- Verify: `insulator_yolo/tests/conftest.py`
- Verify: `insulator_yolo/src/insulator_yolo/visualization/comparison.py`
- Verify: `insulator_yolo/scripts/visualize_comparison.py`

- [ ] **Step 1: Run the visualization-focused tests**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests/visualization/test_comparison.py -v`
Expected: PASS

- [ ] **Step 2: Run the full test suite**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 -m pytest tests -q`
Expected: PASS

- [ ] **Step 3: Run the CLI help command**

Run: `cd /mnt/f/yolo-cls/insulator_yolo && PYTHONPATH=src python3 scripts/visualize_comparison.py --help`
Expected: help text displays successfully.

- [ ] **Step 4: Record environment-specific blockers**

Document any server-specific requirements such as available fonts, Pillow image support, or large-model runtime assumptions.

- [ ] **Step 5: Commit**

```bash
git add tests scripts src README.md configs
git commit -m "test: verify comparison visualization workflow"
```

Plan complete and saved to `docs/superpowers/plans/2026-03-19-comparison-visualization.md`. Ready to execute?
