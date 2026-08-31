# LIBERO Protocol-v3 Cross-fit Probes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run leakage-safe, group-cross-fitted linear interaction probes over all eight same-runtime SmolVLA conditions.

**Architecture:** Reuse the frozen State Bank, factor targets, linear probe solver, metrics, and protocol-v3 latent bindings. Add one protocol-v3 runner that owns deterministic folds, out-of-fold predictions, grouped uncertainty, paired condition contrasts, resumable cells, and report inspection without changing protocol-v2 artifacts.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pytest, existing JSON/NPY artifact helpers.

---

### Task 1: Freeze the cross-fit contract

**Files:**
- Modify: `interaction_vla/representation_study/libero/config.py`
- Modify: `configs/representation_study/libero_smolvla_linux_cuda.yaml`
- Modify: `configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml`
- Test: `tests/interaction_vla/representation_study/libero/test_config.py`

- [x] Add failing tests requiring five formal folds and three smoke folds.
- [x] Run the focused tests and confirm the missing config field fails.
- [x] Add `probes.crossfit_folds`, validate it is at least three, and rerun the tests.

### Task 2: Build deterministic leakage-safe folds

**Files:**
- Create: `interaction_vla/representation_study/libero/crossfit_probes.py`
- Create: `tests/interaction_vla/representation_study/libero/test_crossfit_probes.py`

- [x] Add failing tests proving task groups never cross partitions, episode groups stay intact, every group is test exactly once, and episode folds remain task-blocked.
- [x] Run the focused tests and confirm the fold builder is absent.
- [x] Implement deterministic hash-ordered folds with the next fold used for validation.
- [x] Rerun the tests and confirm they pass.

### Task 3: Produce out-of-fold factor metrics

**Files:**
- Modify: `interaction_vla/representation_study/libero/crossfit_probes.py`
- Modify: `tests/interaction_vla/representation_study/libero/test_crossfit_probes.py`

- [x] Add failing tests for complete out-of-fold coverage, explicit unsupported-class cells, train-only shortcut predictions, grouped bootstrap utility, and paired lower-is-better geometry deltas.
- [x] Run the focused tests and confirm the new runner behavior is missing.
- [x] Reuse `run_linear_probe` for each fold, concatenate out-of-fold predictions, compute preregistered metrics and shortcut baselines, and retain no MLP capacity check in protocol v3.
- [x] Rerun the tests and confirm they pass.

### Task 4: Bind resumable artifacts and CLI

**Files:**
- Modify: `interaction_vla/representation_study/libero/crossfit_probes.py`
- Modify: `interaction_vla/representation_study/libero/cli.py`
- Modify: `tests/interaction_vla/representation_study/libero/test_crossfit_probes.py`
- Modify: `tests/interaction_vla/representation_study/libero/test_cli.py`

- [x] Add failing tests for protocol-v3 latent-gate binding, stale-cell rejection, exact 8 x 4 x 6 grids, predefined paired contrasts, report inspection, and CLI routing.
- [x] Run the focused tests and confirm the commands do not exist.
- [x] Add `longitudinal probes` and `longitudinal probe-report`, writing only below `protocol_v3/probes/`.
- [x] Rerun the tests and confirm they pass.

### Task 5: Document and verify

**Files:**
- Modify: `README.md`
- Modify: `SERVER_RUNBOOK.md`
- Modify: `ccfa.yaml`

- [x] Document the two server commands, expected gate fields, resumability, and the explicit boundary between accessibility and functional/closed-loop utility.
- [x] Run all LIBERO representation-study tests.
- [x] Run CLI help and a synthetic smoke report check.
- [x] Review `git diff` for unrelated changes and preserve all existing protocol-v2 artifacts.
