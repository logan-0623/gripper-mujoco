# SmolVLA Sparse Feature Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover label-blind sparse features in the official SmolVLA `action_expert_input` cache and test their action sensitivity against matched random directions.

**Architecture:** Reuse the frozen State Bank, episode split, Protocol-v4 checkpoint binding, latent cache, deterministic policy inference, and clustered statistics. Add one focused Top-K SAE module and one pipeline module; store large arrays under ignored `outputs/` and commit only compact reports.

**Tech Stack:** Python 3.12, NumPy, PyTorch, existing LeRobot/SmolVLA runtime, pytest.

---

### Task 1: Sparse model and feature matching

**Files:**
- Create: `interaction_vla/representation_study/libero/sparse_autoencoder.py`
- Create: `tests/interaction_vla/representation_study/libero/test_sparse_features.py`

- [ ] Write tests showing Top-K activations contain at most 32 nonzero entries, model artifacts round-trip without pickle, and permuted decoder features match one-to-one.
- [ ] Run `pytest tests/interaction_vla/representation_study/libero/test_sparse_features.py -q`; expect import failure.
- [ ] Implement a minimal untied Top-K SAE, deterministic trainer, safe NPZ save/load, and decoder-cosine feature matching.
- [ ] Re-run the focused test; expect pass.

### Task 2: Label-blind discovery report

**Files:**
- Create: `interaction_vla/representation_study/libero/feature_discovery.py`
- Modify: `tests/interaction_vla/representation_study/libero/test_sparse_features.py`

- [ ] Add failing tests for episode-split fitting, three-seed reproducibility, task/episode coverage, concentration, temporal shuffle diagnostics, and label-blind top-eight selection.
- [ ] Run the focused tests; expect missing discovery functions.
- [ ] Implement `discover_sparse_features(config)` using only Protocol-v4 latents and episode-group train/validation partitions; use test labels only after candidates are frozen.
- [ ] Write `models/seed_{0,1,2}.npz`, `discovery.json`, and `candidates.json` with upstream hashes and deterministic settings.
- [ ] Re-run the focused tests; expect pass.

### Task 3: Feature action intervention and final gate

**Files:**
- Modify: `interaction_vla/representation_study/libero/feature_discovery.py`
- Modify: `tests/interaction_vla/representation_study/libero/test_sparse_features.py`

- [ ] Add failing tests for per-state decoder-contribution removal, orthogonal same-norm random deltas, episode-cluster intervals, BH correction, and stop/continue decisions.
- [ ] Run the focused tests; expect failures for the absent intervention path.
- [ ] Reuse Protocol-v4 `_predict`, `_action_effect`, context batching, and deterministic noise; evaluate original once and target/random per frozen candidate.
- [ ] Store compact per-state effects and write `action_sensitivity.json`; do not retain full action chunks.
- [ ] Implement `report_sparse_features(config)` with integrity, discovery, and causal gates.
- [ ] Re-run the focused tests; expect pass.

### Task 4: CLI and active-path cleanup

**Files:**
- Modify: `interaction_vla/representation_study/libero/cli.py`
- Modify: `interaction_vla/representation_study/cli.py`
- Delete: `interaction_vla/representation_study/libero/interfaces.py`
- Modify: `tests/interaction_vla/representation_study/libero/test_cli.py`

- [ ] Add failing CLI tests for `features discover`, `features intervene`, and `features report`, plus a test that importing the CLI does not load legacy RL training.
- [ ] Run the focused CLI tests; expect failure.
- [ ] Add the three commands and move generic imports into their dispatch branches.
- [ ] Confirm `libero/interfaces.py` has no imports, then delete it.
- [ ] Re-run CLI and LIBERO tests; expect pass.

### Task 5: Freeze compact evidence and project state

**Files:**
- Create: `docs/results/libero_smolvla_positive_control/` compact JSON artifacts
- Modify: `ccfa.yaml`
- Modify: `README.md`
- Modify: `SERVER_RUNBOOK.md`

- [ ] Copy Protocol-v4 evaluation, plan, latent/probe manifests, StableGrasp reports, and Contact reports; exclude checkpoints, row latents, NPZ arrays, and videos.
- [ ] Add an artifact manifest containing hashes for omitted large files.
- [ ] Mark the official positive control `failed_gate` and sparse discovery `not_run`; block longitudinal/closed-loop/RL work.
- [ ] Add only the three current server commands and remove stale wording that authorizes the rejected longitudinal route.

### Task 6: Verification and integration

- [ ] Run the new focused tests.
- [ ] Run `pytest tests/interaction_vla/representation_study/libero -q`.
- [ ] Run `python -m compileall -q interaction_vla` and all new CLI help commands.
- [ ] Review `git diff --check`, confirm user-owned untracked files remain untouched, commit, merge to `main`, and push.
