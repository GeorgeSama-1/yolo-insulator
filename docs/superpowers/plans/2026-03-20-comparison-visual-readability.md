# Comparison Visualization Readability Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make comparison images readable on large inputs and hide `normal_insulator` so the workflow focuses on defect classes only.

**Architecture:** Keep the change local to the comparison visualization pipeline. Add config-driven class filtering that applies to both GT and prediction rendering, then update the drawing code to use adaptive line widths, readable label backgrounds, and larger text.

**Tech Stack:** Python, Pillow, Pytest, Ultralytics configuration passthrough

---

### Task 1: Add failing tests for filtering and rendering behavior

**Files:**
- Modify: `insulator_yolo/tests/visualization/test_comparison.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run targeted tests to verify the new expectations fail**
- [ ] **Step 3: Keep the tests focused on defect-only filtering and output generation**

### Task 2: Implement defect-only filtering and readable rendering

**Files:**
- Modify: `insulator_yolo/src/insulator_yolo/visualization/comparison.py`
- Modify: `insulator_yolo/scripts/visualize_comparison.py`
- Modify: `insulator_yolo/configs/compare.yaml`

- [ ] **Step 1: Add config-driven class filtering for GT and prediction**
- [ ] **Step 2: Add adaptive line width, font size, and label background rendering**
- [ ] **Step 3: Keep original-panel behavior unchanged**

### Task 3: Verify and document

**Files:**
- Modify: `insulator_yolo/README.md`

- [ ] **Step 1: Update README comparison instructions to mention defect-only default behavior**
- [ ] **Step 2: Run targeted and full test suites**
- [ ] **Step 3: Commit the finished change set**
