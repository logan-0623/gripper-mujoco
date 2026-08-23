# LIBERO Interaction Representation Implementation Plan

> **For Codex:** execute tasks in order with tests first. Stop before online RL.

**Goal:** Build an auditable shared LIBERO State Bank, then support fixed SmolVLA stages/taps, latent caches, probes, and factor-aligned paired interventions without changing legacy ACT/Graph artifacts.

**Architecture:** Add an isolated `representation_study.libero` package around two bound sources: standard LeRobotDataset observations and original LIBERO HDF5 simulator trajectories. Reuse existing artifact hashing, policy backends, and probe primitives, but use a new domain schema, task/episode group splits, and SmolVLA semantic tap capture. Commands fail closed when privileged source alignment or task semantics are unavailable.

**Stack:** Python 3.12, NumPy, PyTorch, LeRobot 0.6.1, hf-libero, MuJoCo, h5py, Pillow, pytest, YAML.

---

## Task 1: Register the new scientific track without rewriting old evidence

**Files:**
- Modify: `ccfa.yaml`
- Modify: `README.md`
- Modify: `requirements-lerobot-linux-cuda.txt`
- Test: `tests/interaction_vla/representation_study/libero/test_project_contract.py`

1. Write a failing test that loads `ccfa.yaml` and asserts the legacy classifications, failed recovery gate, new LIBERO dependency nodes, and explicit RL stop.
2. Assert the CUDA requirements contain LeRobot's `smolvla` and `libero` extras.
3. Update `ccfa.yaml` with `formal_evidence`, `pilot_complete`, `failed_gate`, `implementation_only`, and `not_started` states.
4. Add concise README scope, Linux dependency, source prerequisites, artifact roots, and command order.
5. Run the focused contract test.

## Task 2: Add typed LIBERO configuration and immutable schemas

**Files:**
- Create: `interaction_vla/representation_study/libero/__init__.py`
- Create: `interaction_vla/representation_study/libero/config.py`
- Create: `interaction_vla/representation_study/libero/schema.py`
- Create: `interaction_vla/representation_study/libero/interfaces.py`
- Create: `configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml`
- Create: `configs/representation_study/libero_smolvla_linux_cuda.yaml`
- Test: `tests/interaction_vla/representation_study/libero/test_config.py`
- Test: `tests/interaction_vla/representation_study/libero/test_schema.py`

1. Test fail-closed config parsing, relative paths, deterministic seeds, thresholds, suite coverage, and isolated output roots.
2. Define frozen dataclasses for source bindings, thresholds, labels/applicability, records, manifests, semantic taps, and stage metadata.
3. Validate finite continuous labels, dimensions, enum vocabularies, non-empty provenance, and absence of recovery labels.
4. Add smoke/full profiles with identical ontology and different sampling budgets.
5. Run config/schema tests.

## Task 3: Implement reviewed task semantics

**Files:**
- Create: `interaction_vla/representation_study/libero/task_semantics.py`
- Create: `interaction_vla/representation_study/libero/task_registry_v1.yaml`
- Test: `tests/interaction_vla/representation_study/libero/test_task_semantics.py`

1. Test supported single-goal relocation entries, role selectors, goal predicates, factor applicability, unknown-task failure, and unordered multi-goal failure.
2. Implement BDDL goal normalization and a versioned reviewed registry.
3. Register Spatial/Object as formal v1 coverage; keep Goal/Long disabled until explicitly reviewed.
4. Emit coverage and unsupported-reason summaries.
5. Run task-semantics tests.

## Task 4: Bind raw LIBERO and LeRobot episodes

**Files:**
- Create: `interaction_vla/representation_study/libero/sources.py`
- Create: `interaction_vla/representation_study/libero/alignment.py`
- Test: `tests/interaction_vla/representation_study/libero/test_alignment.py`

1. Build small synthetic HDF5 and LeRobot metadata fixtures.
2. Test unique alignment by task, length, and action fingerprint; reject ambiguity, mismatch, duplicate mappings, and missing model XML/states/actions.
3. Write a content-bound alignment manifest atomically and support exact-match resume.
4. Ensure the real loader imports heavy optional dependencies lazily.
5. Run alignment tests.

## Task 5: Implement deterministic replay adapter

**Files:**
- Create: `interaction_vla/representation_study/libero/replay.py`
- Test: `tests/interaction_vla/representation_study/libero/test_replay.py`
- Integration test: `tests/interaction_vla/representation_study/libero/test_libero_integration.py`

1. Use a fake simulator to test reset-from-XML, flattened-state initialization, action stepping, raw observation capture, contact capture, and per-frame mismatch metrics.
2. Implement a lazy `LiberoReplayAdapter` that follows official LIBERO playback semantics.
3. Validate frame/action offset explicitly; never infer it silently.
4. Add a Linux/`hf-libero` integration test for one configured episode, skipped only when the optional source/package is absent.
5. Run replay tests.

## Task 6: Implement privileged annotations

**Files:**
- Create: `interaction_vla/representation_study/libero/contacts.py`
- Create: `interaction_vla/representation_study/libero/annotation.py`
- Test: `tests/interaction_vla/representation_study/libero/test_contacts.py`
- Test: `tests/interaction_vla/representation_study/libero/test_annotation.py`

1. Test semantic geom/body contact grouping, bilateral finger contact, relative transforms/rotation-6D, stable-grasp windowing, phase ordering/hysteresis, and structured next relation.
2. Test undefined factors are masked, not encoded as false/zero.
3. Implement finite-value validation and event timelines.
4. Bind all stable-grasp and phase thresholds to the config/ontology hash and report binary support plus temporal sanity statistics; defer threshold sweeps until the primary annotation gate is inspected.
5. Run annotation tests.

## Task 7: Build, validate, cache, and visualize the shared State Bank

**Files:**
- Create: `interaction_vla/representation_study/libero/splits.py`
- Create: `interaction_vla/representation_study/libero/state_bank.py`
- Create: `interaction_vla/representation_study/libero/audit.py`
- Create: `interaction_vla/representation_study/libero/visualize.py`
- Test: `tests/interaction_vla/representation_study/libero/test_splits.py`
- Test: `tests/interaction_vla/representation_study/libero/test_state_bank.py`
- Test: `tests/interaction_vla/representation_study/libero/test_audit.py`

1. Test task-group and episode-group disjointness, exact coverage, duplicate-state rejection, and train-only normalization.
2. Test resumable per-episode shards, immutable manifest finalization, stale-source rejection, and no overwrite of existing incompatible banks.
3. Implement report fields for task/episode/state counts, applicability, label distributions, geometry ranges, missing/invalid values, replay mismatch, and gate reasons.
4. Produce deterministic contact/phase timelines with global/wrist thumbnails and event strips.
5. Run State Bank tests and the optional one-episode integration test.

## Task 8: Expose a lazy CLI family

**Files:**
- Create: `interaction_vla/representation_study/libero/cli.py`
- Modify: `interaction_vla/representation_study/cli.py`
- Test: `tests/interaction_vla/representation_study/libero/test_cli.py`

1. Test parseability and dispatch for `audit`, `state-bank collect|inspect|visualize|approve-timelines`, `stages plan`, `latents extract|inspect`, `probes run|report`, `interventions run`, and `evaluate`.
2. Import LIBERO-specific code only after the `libero` family is selected.
3. Return machine-readable passing/failing JSON and non-zero exit status on gate failure.
4. Run CLI tests.

## Task 9: Plan nested SFT stages and checkpoint identities

**Files:**
- Create: `interaction_vla/representation_study/libero/stages.py`
- Test: `tests/interaction_vla/representation_study/libero/test_stages.py`

1. Test task-balanced deterministic nested episode subsets `D25 ⊂ D50 ⊂ D100`.
2. Test fixed-epoch step derivation and metadata for base model, revision, seed, fraction, subset hash, code/config hash, and checkpoint status.
3. Mark absent checkpoints `not_run`; never create fake checkpoint content.
4. Reuse existing SFT trainer through a dataset-view adapter only after its schema binding passes.
5. Run stage tests.

## Task 10: Implement semantic SmolVLA taps

**Files:**
- Create: `interaction_vla/representation_study/libero/taps.py`
- Test: `tests/interaction_vla/representation_study/libero/test_taps.py`

1. Create a minimal fake SmolVLA call graph and test exact prefix-vs-suffix call selection, masks, view ordering, final denoising selection, and shapes.
2. Capture `embed_image`, first prefix hidden state, `action_time_mlp_out`, and `action_out_proj` input.
3. Implement preregistered `valid_token_mean`; expose alternatives by explicit name without auto-selection.
4. Store module path, raw/pooled shape, mask, call selector, and pooling provenance.
5. Run tap tests and compatibility tests against installed LeRobot when available.

## Task 11: Generate checkpoint-independent latent caches

**Files:**
- Create: `interaction_vla/representation_study/libero/latents.py`
- Test: `tests/interaction_vla/representation_study/libero/test_latents.py`

1. Test cache keys `(state_id, checkpoint_id, tap, pooling, inference_seed)`, exact coverage, restart/resume, stale checkpoint rejection, and shared label identity.
2. Batch/materialize State Bank observations without recomputing annotations.
3. Seed model inference deterministically from state/checkpoint identity.
4. Finalize only after finite tensors and shape consistency pass.
5. Run latent-cache tests.

## Task 12: Run the registered probe protocol

**Files:**
- Create: `interaction_vla/representation_study/libero/probes.py`
- Create: `interaction_vla/representation_study/libero/probe_runner.py`
- Test: `tests/interaction_vla/representation_study/libero/test_probes.py`
- Test: `tests/interaction_vla/representation_study/libero/test_report.py`

1. Test applicability masking, train-only normalization, factor metrics, majority/mean/task-ID/exact-instruction/time baselines, confidence-interval accessibility decisions, and task/episode clustered bootstrap.
2. Make linear primary and shallow MLP capacity check explicit.
3. Emit exact Stage×Tap×Factor rows and heatmap-ready long-form data.
4. Suppress unsupported metrics with reasons; never convert missing labels to negatives.
5. Run probe/report tests.

## Task 13: Add factor-aligned intervention and paired-evaluation infrastructure

**Files:**
- Create: `interaction_vla/representation_study/libero/interventions.py`
- Create: `interaction_vla/representation_study/libero/evaluation.py`
- Test: `tests/interaction_vla/representation_study/libero/test_interventions.py`
- Test: `tests/interaction_vla/representation_study/libero/test_evaluation.py`

1. Test probe-row-space donor intervention, nuisance matching, norm/distribution checks, target disruption, and non-target specificity.
2. Include original, matched-random, matched-mean, and OOD-zero controls.
3. Test paired case identity, inference-noise identity, outcome/failure schema, and paired task/episode bootstrap.
4. Keep reports `implementation_only`/`not_run` until validated probes and real paired rollouts exist.
5. Run intervention/evaluation tests.

## Task 14: Full verification and handoff

**Files:**
- Modify: `ccfa.yaml`
- Modify: `README.md`

1. Run all new focused tests.
2. Run the full representation-study regression suite.
3. Run `python -m compileall` on the new package, `git diff --check`, config parsing, and smoke CLI help.
4. Confirm legacy artifact paths and user-owned PDF are untouched.
5. Update statuses based only on actually executed gates: code is `implementation_only`; experiments stay `not_started` or `failed_gate` until real artifacts exist.
6. Provide exact Linux commands for source preparation, smoke State Bank, full State Bank, stage planning, latent extraction, probes, and paired evaluation. Stop before RL.
