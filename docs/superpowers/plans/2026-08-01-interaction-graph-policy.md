# Interaction-Graph Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a Mac-friendly, reproducible Graph-versus-Flat behavior-cloning pilot for multi-object tabletop pick-and-place.

**Architecture:** A deterministic kinematic tabletop environment and scripted expert generate current-state scene tensors and action demonstrations. Flat and message-passing encoders consume identical padded node/edge tensors and feed one shared action policy; paired closed-loop evaluation measures object-aware and gripper-aware behavior across two to five objects.

**Tech Stack:** Python 3.12, NumPy, PyTorch with MPS/CPU selection, PyYAML, pytest; optional MuJoCo adapter isolated behind the environment snapshot protocol.

**Repository note:** The supplied directory has no `.git` metadata, so commit steps cannot be executed unless the user later initializes or restores the repository history.

---

## File Map

- Create `interaction_vla/__init__.py`: public package exports.
- Create `interaction_vla/config.py`: validated experiment configuration and YAML loading.
- Create `interaction_vla/device.py`: MPS/CPU device resolution.
- Create `interaction_vla/graph/schema.py`: padded graph tensors and validation.
- Create `interaction_vla/graph/builder.py`: typed snapshot to graph conversion.
- Create `interaction_vla/env.py`: deterministic tabletop environment and snapshot protocol.
- Create `interaction_vla/mujoco_env.py`: optional headless MuJoCo mirror using primitive bodies.
- Create `interaction_vla/expert.py`: phase-based pick-and-place expert.
- Create `interaction_vla/data.py`: episode collection and `.npz` storage.
- Create `interaction_vla/models/encoders.py`: Flat and Graph encoders.
- Create `interaction_vla/models/policy.py`: shared action policy.
- Create `interaction_vla/train.py`: training, normalization, and checkpoints.
- Create `interaction_vla/evaluate.py`: paired closed-loop evaluation and reports.
- Create `configs/pilot_macos.yaml`: fast local run.
- Create `configs/main_macos.yaml`: larger future run.
- Create `requirements-macos.txt`: platform-neutral local dependencies.
- Modify `README.md`: Mac pilot commands and interpretation limits.
- Create `tests/interaction_vla/`: unit and integration tests matching the modules above.

## Task 1: Configuration and Device Selection

**Files:**
- Create: `interaction_vla/config.py`
- Create: `interaction_vla/device.py`
- Create: `configs/pilot_macos.yaml`
- Create: `configs/main_macos.yaml`
- Test: `tests/interaction_vla/test_config.py`

- [x] **Step 1: Write failing configuration tests**

```python
def test_pilot_config_has_disjoint_train_and_ood_counts():
    cfg = load_config("configs/pilot_macos.yaml")
    assert cfg.train.object_counts == (2, 3)
    assert cfg.eval.object_counts == (2, 3, 4, 5)

def test_invalid_capacity_fails_early():
    with pytest.raises(ValueError, match="max_objects"):
        ExperimentConfig(max_objects=1)

def test_device_falls_back_to_cpu_when_mps_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
```

- [x] **Step 2: Run `pytest tests/interaction_vla/test_config.py -q` and confirm imports or APIs are missing**
- [x] **Step 3: Implement frozen dataclass configuration with explicit validation and `resolve_device` supporting `auto`, `mps`, and `cpu`**
- [x] **Step 4: Add pilot/main YAML with seeds, object counts, model widths, episode counts, and output paths**
- [x] **Step 5: Re-run the focused tests and the full suite**

## Task 2: Scene-Graph Schema and Builder

**Files:**
- Create: `interaction_vla/graph/__init__.py`
- Create: `interaction_vla/graph/schema.py`
- Create: `interaction_vla/graph/builder.py`
- Test: `tests/interaction_vla/test_graph.py`

- [x] **Step 1: Write failing tests for shapes, masks, finite-value validation, and target flags**

```python
def test_builder_creates_complete_directed_masked_graph():
    graph = SceneGraphBuilder(max_objects=5).build(example_snapshot(2))
    assert graph.node_features.shape == (8, NODE_FEATURE_DIM)
    assert graph.edge_features.shape == (56, EDGE_FEATURE_DIM)
    assert graph.node_mask.sum() == 5  # gripper, 2 objects, receptacle, table
    assert graph.edge_mask.sum() == 20

def test_flat_payload_contains_the_exact_graph_values():
    graph = SceneGraphBuilder(max_objects=5).build(example_snapshot(3))
    payload = graph.flat_payload()
    np.testing.assert_array_equal(payload[: graph.node_features.size], graph.node_features.ravel())

def test_validate_rejects_non_finite_features():
    graph = SceneGraphBuilder(max_objects=5).build(example_snapshot(2))
    graph.node_features[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        graph.validate()
```

- [x] **Step 2: Run the graph tests and observe the expected missing-module failure**
- [x] **Step 3: Implement `EntityState`, `SceneSnapshot`, and `SceneGraph` dataclasses with stable feature constants**
- [x] **Step 4: Implement canonical node placement, complete directed `edge_index`, current-state relational features, and padding masks**
- [x] **Step 5: Implement `SceneGraph.permute_valid_nodes()` for invariance testing while preserving entity semantics**
- [x] **Step 6: Run focused and full tests**

## Task 3: Deterministic Environment and Scripted Expert

**Files:**
- Create: `interaction_vla/env.py`
- Create: `interaction_vla/expert.py`
- Test: `tests/interaction_vla/test_env.py`
- Test: `tests/interaction_vla/test_expert.py`

- [x] **Step 1: Write a failing deterministic-reset test**

```python
def test_reset_is_reproducible_and_non_overlapping():
    a = KinematicTabletopEnv(max_objects=5).reset(seed=17, object_count=4)
    b = KinematicTabletopEnv(max_objects=5).reset(seed=17, object_count=4)
    np.testing.assert_allclose(a.object_positions, b.object_positions)
    assert minimum_pairwise_distance(a.object_positions) >= 0.12
```

- [x] **Step 2: Write failing grasp-transition tests for wrong-object, holding, drop, release, and success result codes**
- [x] **Step 3: Run tests and confirm the APIs are absent**
- [x] **Step 4: Implement a bounded 3D Cartesian environment with four-dimensional actions `(dx, dy, dz, gripper)` and kinematically attached held objects**
- [x] **Step 5: Ensure `snapshot()` reports contact/holding/support only from the current state**
- [x] **Step 6: Implement `ScriptedExpert.act(snapshot)` with explicit `ExpertPhase` transitions and a timeout**
- [x] **Step 7: Run focused and full tests**

## Task 4: Episode Collection and Storage

**Files:**
- Create: `interaction_vla/data.py`
- Test: `tests/interaction_vla/test_data.py`

- [x] **Step 1: Write a failing temporal-ordering test**

```python
def test_collector_stores_pre_action_graph():
    env = KinematicTabletopEnv(max_objects=5)
    episode = collect_episode(env, ScriptedExpert(), seed=3, object_count=2)
    first = episode.steps[0]
    assert first.snapshot.gripper_position[0] == pytest.approx(0.0)
    assert first.action[0] != 0.0
```

- [x] **Step 2: Write failing round-trip and episode-level split tests**
- [x] **Step 3: Run tests and verify expected failures**
- [x] **Step 4: Implement typed `EpisodeStep`, `Episode`, collection, explicit termination reasons, and compressed NumPy storage**
- [x] **Step 5: Implement reproducible episode-level train/validation/test manifests with no seed overlap**
- [x] **Step 6: Add `python -m interaction_vla.data collect --config ...`**
- [x] **Step 7: Run focused and full tests**

## Task 4A: Headless MuJoCo Environment Adapter

**Files:**
- Create: `interaction_vla/mujoco_env.py`
- Test: `tests/interaction_vla/test_mujoco_env.py`

- [x] **Step 1: Write a test guarded by `pytest.importorskip("mujoco")` that resets two identical seeds and compares snapshots**
- [x] **Step 2: Write a failing test that constructs the adapter without GLFW or a viewer and steps one four-dimensional action**
- [x] **Step 3: Run tests and verify the adapter is missing**
- [x] **Step 4: Implement a primitive MuJoCo XML model with a mocap gripper, table, receptacle, and five colored movable object bodies**
- [x] **Step 5: Mirror the deterministic task state into MuJoCo positions before `mj_forward`, expose the standard snapshot contract, and keep rendering optional**
- [x] **Step 6: Run focused and full tests with MuJoCo installed; confirm no GUI opens**

## Task 5: Fair Flat and Graph Encoders

**Files:**
- Create: `interaction_vla/models/__init__.py`
- Create: `interaction_vla/models/encoders.py`
- Test: `tests/interaction_vla/test_encoders.py`

- [x] **Step 1: Write failing output-shape, padding, and permutation tests**

```python
def test_graph_pooling_is_invariant_to_valid_node_permutation():
    torch.manual_seed(0)
    encoder = GraphEncoder(...).eval()
    graph = sample_graph_batch()
    permuted = permute_graph_batch(graph, torch.tensor([2, 0, 1, 3, 4, 5, 6, 7]))
    torch.testing.assert_close(encoder(graph), encoder(permuted), atol=1e-5, rtol=1e-5)

def test_flat_and_graph_parameter_counts_are_within_ten_percent():
    flat, graph = build_matched_encoders(...)
    difference = abs(count_params(flat) - count_params(graph))
    assert difference / max(count_params(flat), count_params(graph)) <= 0.10
```

- [x] **Step 2: Run tests and confirm missing implementations**
- [x] **Step 3: Implement a masked two-round message-passing encoder with edge-conditioned messages and masked mean pooling**
- [x] **Step 4: Implement a flat MLP encoder over the exact graph payload**
- [x] **Step 5: Implement automatic hidden-width matching for the Flat encoder and assert the final parameter budget**
- [x] **Step 6: Run focused and full tests on CPU; run MPS shape tests when available**

## Task 6: Shared Action Policy and Training

**Files:**
- Create: `interaction_vla/models/policy.py`
- Create: `interaction_vla/train.py`
- Test: `tests/interaction_vla/test_policy.py`
- Test: `tests/interaction_vla/test_train.py`

- [x] **Step 1: Write failing action-range and shared-head tests**
- [x] **Step 2: Write a failing two-episode overfit test requiring final normalized MSE below `1e-3`**
- [x] **Step 3: Write a failing checkpoint-resume test that compares continued step counts and model weights**
- [x] **Step 4: Run tests and verify failures are caused by missing behavior**
- [x] **Step 5: Implement `ActionPolicy` with proprioception projection, optional context projection, scene encoder, and bounded four-dimensional action output**
- [x] **Step 6: Implement dataset normalization using training data only, deterministic batching, finite-loss guard, checkpoint save/resume, and JSONL metrics**
- [x] **Step 7: Add `python -m interaction_vla.train --config ... --representation flat|graph|proprio`**
- [x] **Step 8: Run focused and full tests**

## Task 7: Paired Closed-Loop Evaluation

**Files:**
- Create: `interaction_vla/evaluate.py`
- Test: `tests/interaction_vla/test_evaluate.py`

- [x] **Step 1: Write failing metric-aggregation and paired-seed tests**

```python
def test_report_separates_id_and_ood_counts():
    report = aggregate_results(example_results())
    assert report["by_object_count"]["4"]["split"] == "ood"
    assert "wrong_object_rate" in report["overall"]
```

- [x] **Step 2: Write a failing edge-shuffle test proving only valid edge assignments change**
- [x] **Step 3: Run tests and verify expected failures**
- [x] **Step 4: Implement closed-loop rollout on paired environment seeds and save per-episode CSV/JSON plus aggregate JSON**
- [x] **Step 5: Implement success, wrong-object, grasp, drop, step-count, and offline-action metrics**
- [x] **Step 6: Implement Graph edge-shuffle evaluation with the same trained weights and rollout seeds**
- [x] **Step 7: Add `python -m interaction_vla.evaluate --config ... --checkpoints ...`**
- [x] **Step 8: Run focused and full tests**

## Task 8: Mac Setup, Documentation, and End-to-End Smoke Run

**Files:**
- Create: `requirements-macos.txt`
- Modify: `README.md`
- Test: all tests and command-line smoke runs

- [x] **Step 1: Add platform-neutral dependencies (`numpy`, `torch`, `pytest`, `pyyaml`, `mujoco`) without a CUDA index URL**
- [x] **Step 2: Document `uv venv`, dependency installation, data collection, three-policy training, and paired evaluation commands**
- [x] **Step 3: Document that the core pilot uses privileged state and does not yet validate RGB graph extraction or a world model**
- [x] **Step 4: Run `pytest -q` and record the exact pass/skip totals**
- [x] **Step 5: Run a tiny collection command and inspect its manifest**
- [x] **Step 6: Train Flat and Graph tiny models on CPU/MPS, then run paired two- and four-object rollouts**
- [x] **Step 7: Run static compilation with `python -m compileall -q interaction_vla tests`**
- [x] **Step 8: Compare the delivered files against every design requirement and report any unimplemented future extension explicitly**

## Plan Self-Review

- The representation contract, action prediction, Mac device handling, automatic expert, episode-level data split, fair encoders, paired closed-loop metrics, edge-shuffle check, failure reporting, and tests each map to an implementation task.
- Names and tensor capacities are consistent across tasks: eight nodes, 56 directed edges, two to five movable objects, and four-dimensional Cartesian/gripper actions.
- SmolVLA integration and RGB-derived graphs remain explicitly outside the first implementation, as approved.
- No task requires Git history that is absent from this workspace.
