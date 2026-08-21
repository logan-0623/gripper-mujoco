# ICRA Representation Study Phase 1 Implementation Plan

> **For Codex:** Execute this plan test-first. Do not modify legacy report files or
> start training as part of Phase 1.

**Goal:** Establish trustworthy sensitivity measurements and a backend-independent
checkpoint/stage contract while preserving the completed ACT Graph study.

**Architecture:** Existing `interaction_vla.graph_control` remains the legacy ACT
mechanism pipeline. A new `interaction_vla.representation_study` package owns shared
stage manifests and backend protocols. Sensitivity v2 is emitted beside, never over,
the v1 report.

**Tech stack:** Python 3.12, dataclasses/protocols, NumPy, PyTorch, pytest, YAML/JSON.

---

### Task 1: Freeze legacy sensitivity semantics

**Files:**

- Modify: `interaction_vla/graph_control/sensitivity.py`
- Modify: `interaction_vla/graph_control/pipeline.py`
- Modify: `tests/interaction_vla/graph_control/test_sensitivity.py`

- [x] Add tests defining v2 raw action RMS metrics and action-IQR normalization.
- [x] Add tests showing zero/tiny token perturbations cannot create explosive primary
  sensitivity values.
- [x] Preserve v1 metric constants for reading historical reports.
- [x] Publish regenerated analysis to `diagnostics/<partition>/sensitivity_v2/` with
  schema `graph_policy_sensitivity_v2`.
- [x] Run the focused sensitivity test module.

### Task 2: Add policy stage manifests

**Files:**

- Create: `interaction_vla/representation_study/__init__.py`
- Create: `interaction_vla/representation_study/schemas/__init__.py`
- Create: `interaction_vla/representation_study/schemas/stages.py`
- Create: `tests/interaction_vla/representation_study/test_stage_schema.py`

- [x] Test supported ACT, SmolVLA, and pi0 stages.
- [x] Test rejection of missing hashes, unknown stages, duplicate taps, and invalid
  trainable groups.
- [x] Implement immutable stage and artifact descriptors with canonical JSON output.
- [x] Bind each manifest to checkpoint/config/dataset/source hashes and fixed taps.

### Task 3: Add backend and tap contracts

**Files:**

- Create: `interaction_vla/representation_study/backends/__init__.py`
- Create: `interaction_vla/representation_study/backends/base.py`
- Create: `interaction_vla/representation_study/taps/__init__.py`
- Create: `interaction_vla/representation_study/taps/registry.py`
- Create: `tests/interaction_vla/representation_study/test_backend_contract.py`

- [x] Test that every registered backend exposes the four predeclared scientific tap
  roles.
- [x] Test backend/stage compatibility and fixed-tap validation.
- [x] Define the runtime-checkable `PolicyBackend` protocol.
- [x] Register ACT, SmolVLA, and pi0 tap-role schemas without loading model weights.

### Task 4: Wire configuration and documentation

**Files:**

- Create: `configs/representation_study/base.yaml`
- Modify: `README.md`
- Modify: `ccfa.yaml`

- [x] Document ACT as the controlled mechanism study and SmolVLA as the modern VLA
  validation; mark pi0 optional.
- [x] Document Phase 1 test commands and explicitly state that no new training is
  started yet.
- [x] Advance the CCFA measurement gate only after focused and regression tests pass.

### Task 5: Verification

- [x] Run `pytest -q tests/interaction_vla/graph_control/test_sensitivity.py`.
- [x] Run `pytest -q tests/interaction_vla/representation_study`.
- [x] Run the relevant graph-control regression subset.
- [x] Inspect `git diff --check` and confirm no generated outputs or private PDF are
  staged or modified.
