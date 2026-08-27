# LIBERO Probe Protocol v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make longitudinal SmolVLA probe comparisons stage-matched, bootstrap-safe, paired, and versioned without overwriting the completed v1 pilot artifacts.

**Architecture:** Keep the frozen State Bank, splits, checkpoints, latent caches, factor targets, and primary metrics unchanged. Add a versioned v2 probe artifact root, derive probe seeds only from tap/factor/split/replicate identity, aggregate matched replicates, and compute paired stage deltas from predictions on identical held-out states and bootstrap groups.

**Tech Stack:** Python 3.12, NumPy, PyTorch, pytest, YAML, existing atomic JSON/cache utilities.

---

### Task 1: Freeze the protocol-v2 configuration contract

**Files:**
- Modify: `interaction_vla/representation_study/libero/config.py`
- Modify: `configs/representation_study/libero_smolvla_linux_cuda.yaml`
- Modify: `configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml`
- Test: `tests/interaction_vla/representation_study/libero/test_config.py`

- [x] Add a failing config test requiring three formal matched seeds and one smoke seed.
- [x] Run the focused config test and confirm it fails because `ProbeConfig` has no replicate field.
- [x] Add `matched_seeds` with positive, unique integer validation.
- [x] Run the focused config tests and confirm they pass.

### Task 2: Stabilize classification bootstrap labels

**Files:**
- Modify: `interaction_vla/representation_study/libero/probe_runner.py`
- Test: `tests/interaction_vla/representation_study/libero/test_probes.py`

- [x] Add a failing test where predictions contain a valid model class absent from the complete test targets.
- [x] Run the test and confirm `_bootstrap_ci` raises `classification label universe omits observed labels`.
- [x] Freeze the metric label universe as the union of complete targets and predictions before resampling.
- [x] Run the focused test and confirm the bootstrap interval is finite.

### Task 3: Match probe optimization seeds across stages

**Files:**
- Modify: `interaction_vla/representation_study/libero/probe_runner.py`
- Test: `tests/interaction_vla/representation_study/libero/test_probes.py`

- [x] Add failing tests proving seed derivation is invariant to stage and distinct across tap/factor/split/replicate identities.
- [x] Run the tests and confirm the stage-invariant API is absent.
- [x] Implement stable hash-based seed derivation without `stage_index`.
- [x] Store every replicate seed in each probe row and aggregate replicate metrics without treating seeds as independent environment evidence.
- [x] Add the hard sanity gate that identical latent matrices across stages must yield identical same-seed probe results.

### Task 4: Add paired longitudinal stage deltas

**Files:**
- Modify: `interaction_vla/representation_study/libero/probe_runner.py`
- Test: `tests/interaction_vla/representation_study/libero/test_probes.py`

- [x] Add failing synthetic tests for paired classification and geometry deltas clustered by held-out groups.
- [x] Implement per-seed matched prediction pairing and paired cluster-bootstrap confidence intervals.
- [x] Report `destination - pretrained` with metric direction explicit; never infer improvement from two independent accessibility intervals.
- [x] Mark a delta unavailable when either cell failed its gate or state/group identities differ.

### Task 5: Preserve v1 artifacts and verify v2

**Files:**
- Modify: `interaction_vla/representation_study/libero/probe_runner.py`
- Modify: `SERVER_RUNBOOK.md`
- Test: `tests/interaction_vla/representation_study/libero/test_probes.py`

- [x] Add a failing path test asserting v2 writes below `probes/protocol_v2/` and leaves `probes/report.json` untouched.
- [x] Bump the schema and bind v2 cells/report to implementation, config, State Bank, split, and latent hashes.
- [x] Run focused probe/config tests.
- [x] Run the complete LIBERO representation-study test directory, `compileall`, and `git diff --check`.
- [x] Document that State Bank, SFT checkpoints, and latent caches are reused; only probes must be rerun.
