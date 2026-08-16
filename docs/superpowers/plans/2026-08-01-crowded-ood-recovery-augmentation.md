# Crowded OOD and Deterministic Recovery Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic target-centered crowded OOD condition, measure the existing checkpoints on it, then train and evaluate Flat/Graph with training-split-only deterministic recovery demonstrations.

**Architecture:** The environment gains an explicit layout mode and a state-consistent gripper perturbation API. A focused recovery module constructs deterministic perturbation specifications, while data collection saves post-perturbation trajectories in a separate manifest. Evaluation adds named conditions and model-seed-paired comparisons; Stage A and Stage B outputs remain isolated.

**Tech Stack:** Python 3.12, NumPy, PyTorch CPU/MPS, PyYAML, MuJoCo, pytest.

**Repository note:** This directory has no `.git` metadata, so commit steps cannot be performed.

---

## File Map

- Modify `interaction_vla/config.py`: crowded and recovery settings.
- Modify `interaction_vla/env.py`: layout modes, crowded sampling, perturbation API.
- Modify `interaction_vla/mujoco_env.py`: proxy the new environment contract.
- Modify `interaction_vla/expert.py`: recovery-aware state guards.
- Create `interaction_vla/recovery.py`: deterministic recovery specifications.
- Modify `interaction_vla/data.py`: recovery collection, metadata, manifests, CLI.
- Modify `interaction_vla/train.py`: combine base and recovery training paths.
- Modify `interaction_vla/evaluate.py`: conditioned cases, paired reports, stage comparison.
- Create `configs/crowded_ood_macos.yaml` and `configs/recovery_macos.yaml`.
- Modify tests, README, and the pilot result report.

## Task 1: Configuration and Deterministic Crowded Layout

**Files:**
- Modify: `interaction_vla/config.py`
- Modify: `interaction_vla/env.py`
- Modify: `interaction_vla/mujoco_env.py`
- Test: `tests/interaction_vla/test_config.py`
- Test: `tests/interaction_vla/test_env.py`
- Test: `tests/interaction_vla/test_mujoco_env.py`

- [ ] **Step 1: Write a failing crowded-configuration test**

```python
def test_crowded_distances_are_validated() -> None:
    cfg = EnvironmentConfig(
        crowded_anchor_min_distance=0.085,
        crowded_anchor_max_distance=0.105,
    )
    assert cfg.crowded_anchor_min_distance < cfg.crowded_anchor_max_distance
    with pytest.raises(ValueError, match="crowded"):
        EnvironmentConfig(
            crowded_anchor_min_distance=0.11,
            crowded_anchor_max_distance=0.09,
        )
```

- [ ] **Step 2: Run RED**

Run `.venv/bin/python -m pytest tests/interaction_vla/test_config.py::test_crowded_distances_are_validated -q` and confirm the fields are missing.

- [ ] **Step 3: Implement validated crowded settings**

Add `crowded_anchor_min_distance=0.085` and `crowded_anchor_max_distance=0.105` to `EnvironmentConfig`. Require positive ordered values, minimum `0.08`, and maximum below `min_object_distance`.

- [ ] **Step 4: Write failing layout tests**

```python
def test_crowded_reset_is_deterministic_and_places_anchor_near_target() -> None:
    first = KinematicTabletopEnv(max_objects=5).reset(
        seed=71, object_count=5, layout_mode="crowded"
    )
    second = KinematicTabletopEnv(max_objects=5).reset(
        seed=71, object_count=5, layout_mode="crowded"
    )
    np.testing.assert_allclose(object_positions(first), object_positions(second))
    target = first.target_object
    nearest = min(
        np.linalg.norm(entity.position[:2] - target.position[:2])
        for entity in first.objects
        if entity.name != target.name
    )
    assert 0.085 <= nearest <= 0.105


def test_normal_layout_retains_original_spacing() -> None:
    snapshot = KinematicTabletopEnv(max_objects=5).reset(
        seed=71, object_count=5, layout_mode="normal"
    )
    assert minimum_pairwise_distance(object_positions(snapshot)) >= 0.12
```

- [ ] **Step 5: Run layout tests and verify RED**

Expected: `reset()` rejects `layout_mode`.

- [ ] **Step 6: Implement `LayoutMode` and crowded sampling**

Add `LayoutMode.NORMAL` and `LayoutMode.CROWDED`. Select target index first. Preserve the current normal sampler. In crowded mode place the target, one seeded anchor distractor in the configured radial interval, then the remaining objects under the approved distance constraints. Restore positions to `object_<index>` order.

- [ ] **Step 6A: Verify crowded scenes remain solvable by the expert**

Add a parameterized real-environment test for object counts 2, 3, 4, and 5 over fixed crowded seeds. Require every scripted-expert rollout to terminate with `TerminationReason.SUCCESS`; do not weaken grasp or success thresholds if it fails.

- [ ] **Step 7: Write a failing held-object perturbation test**

After grasping a target, call the wished-for API:

```python
perturbed = env.perturb_gripper_state(
    np.asarray((0.04, -0.02, -0.02), dtype=np.float32)
)
np.testing.assert_allclose(
    perturbed.target_object.position,
    perturbed.gripper.position + env.hold_offset,
)
assert env.step_count == step_count_before
```

- [ ] **Step 8: Implement state-consistent perturbation**

Add `perturb_gripper_state(delta, gripper_open=None)`. Validate a finite 3-vector, clip the new gripper position, update velocity/open state without incrementing time, and move a held object with `hold_offset`. Never create or release a grasp in this method.

- [ ] **Step 9: Forward the contract through MuJoCo**

Allow `MujocoTabletopEnv.reset(..., layout_mode=...)` and proxy `perturb_gripper_state`, synchronizing MuJoCo data after mutation. Test one headless crowded reset and one held perturbation.

- [ ] **Step 10: Verify Task 1**

Run the three focused test files and then `.venv/bin/python -m pytest tests/interaction_vla -q`.

## Task 2: Condition-Aware Paired Evaluation

**Files:**
- Modify: `interaction_vla/config.py`
- Modify: `interaction_vla/evaluate.py`
- Create: `configs/crowded_ood_macos.yaml`
- Test: `tests/interaction_vla/test_evaluate.py`

- [ ] **Step 1: Write failing condition tests**

```python
cases = make_conditioned_evaluation_cases(
    id_object_counts=(2, 3),
    count_ood_object_counts=(4, 5),
    crowded_object_counts=(4, 5),
    episodes_per_count=2,
    base_seed=100,
)
assert {case.condition for case in cases} == {
    "id_normal", "count_ood", "crowded_ood"
}
assert all(
    case.layout_mode == "crowded"
    for case in cases
    if case.condition == "crowded_ood"
)
assert len({(case.condition, case.seed) for case in cases}) == len(cases)
```

Also require `by_policy_and_condition` and per-seed `condition_success_delta` in the aggregate report.

- [ ] **Step 2: Run RED**

Run `tests/interaction_vla/test_evaluate.py`; expect missing condition fields and builder.

- [ ] **Step 3: Add evaluation config**

Add `crowded_object_counts: tuple[int, ...] = (4, 5)` to `EvalConfig`. Validate that they belong to `object_counts` and do not overlap training object counts.

- [ ] **Step 4: Implement conditioned cases**

Add backward-compatible `condition="id_normal"` and `layout_mode="normal"` fields to `EvaluationCase` and `EpisodeResult`. Use distinct seed offsets for all three conditions and pass layout mode into environment reset.

- [ ] **Step 5: Implement conditioned metrics**

Add `by_condition`, `by_policy_and_condition`, `by_policy_condition_and_object_count`, and per-model-seed condition success/wrong-object deltas. Use crowded OOD for the strict criterion when it is present; otherwise retain old count-OOD behavior.

- [ ] **Step 6: Create `configs/crowded_ood_macos.yaml`**

Use `data_dir: outputs/interaction_vla/pilot/data`, `output_dir: outputs/interaction_vla/crowded_baseline`, pilot model/environment values, normal counts 2–5, crowded counts 4–5, 20 cases per count, and 120 steps.

- [ ] **Step 7: Verify Task 2**

Run evaluation tests and the full suite.

## Task 3: Execute Frozen-Checkpoint Stage A

**Files:**
- Generate: `outputs/interaction_vla/crowded_baseline/evaluation/report.json`
- Generate: `outputs/interaction_vla/crowded_baseline/evaluation/episodes.csv`
- Modify: `docs/interaction_graph_pilot_results.md`

- [ ] **Step 1: Check all six pilot checkpoints**

Require Flat and Graph checkpoints for model seeds 0, 1, and 2 before evaluation.

- [ ] **Step 2: Run Stage A**

```bash
.venv/bin/python -m interaction_vla.evaluate \
  --config configs/crowded_ood_macos.yaml \
  --checkpoints \
  outputs/interaction_vla/pilot/flat/seed_0/checkpoint.pt \
  outputs/interaction_vla/pilot/graph/seed_0/checkpoint.pt \
  outputs/interaction_vla/pilot/flat/seed_1/checkpoint.pt \
  outputs/interaction_vla/pilot/graph/seed_1/checkpoint.pt \
  outputs/interaction_vla/pilot/flat/seed_2/checkpoint.pt \
  outputs/interaction_vla/pilot/graph/seed_2/checkpoint.pt
```

- [ ] **Step 3: Validate Stage A artifacts**

Assert 1,080 episode rows, finite metrics, three model seeds, and expert success on all accepted crowded smoke cases.

- [ ] **Step 4: Freeze the Stage A summary**

Record per-seed ID, count-OOD, crowded-OOD, wrong-object, and edge-shuffle metrics before recovery generation begins.

## Task 4: Recovery-Aware Expert and Perturbation Specs

**Files:**
- Modify: `interaction_vla/expert.py`
- Create: `interaction_vla/recovery.py`
- Test: `tests/interaction_vla/test_expert.py`
- Create: `tests/interaction_vla/test_recovery.py`

- [ ] **Step 1: Write failing expert recovery tests**

Create real environment states and require:

```python
def test_closed_empty_gripper_reopens_and_realigns() -> None:
    action = expert.act(closed_unheld_offset_snapshot)
    assert action[3] == pytest.approx(1.0)
    assert expert.phase is ExpertPhase.ALIGN


def test_held_target_below_lift_height_returns_to_lift() -> None:
    action = expert.act(perturbed_held_snapshot)
    assert action[2] > 0
    assert action[3] == pytest.approx(0.0)
    assert expert.phase is ExpertPhase.LIFT
```

- [ ] **Step 2: Run RED**

Confirm the current monotonic phase machine does not resynchronize.

- [ ] **Step 3: Implement state-consistency guards**

At the start of `act()`, infer safe phase regressions from current holding, gripper openness, target distance, lift height, and receptacle-relative position. Never read a perturbation label or future state. Re-run the original unperturbed expert tests.

- [ ] **Step 4: Write failing spec-determinism tests**

Define the API:

```python
first = make_recovery_spec(source_seed=42, variant_id=0)
second = make_recovery_spec(source_seed=42, variant_id=0)
assert first.kind == second.kind
np.testing.assert_array_equal(first.delta, second.delta)
assert first.gripper_open == second.gripper_open
```

Test four consecutive variants, stable family ordering, and every magnitude bound.

- [ ] **Step 5: Implement `interaction_vla/recovery.py`**

Create:

```python
class PerturbationKind(str, Enum):
    ALIGN_OFFSET = "align_offset"
    FAILED_CLOSE = "failed_close"
    LIFT_OFFSET = "lift_offset"
    TRANSPORT_OFFSET = "transport_offset"


@dataclass(frozen=True)
class RecoverySpec:
    source_seed: int
    variant_id: int
    kind: PerturbationKind
    injection_phase: ExpertPhase
    delta: np.ndarray
    gripper_open: float | None
```

Use `np.random.SeedSequence((source_seed, variant_id, 0x5245434F))` and a fixed kind tuple. Implement `apply_recovery_spec(env, spec)` exclusively through `perturb_gripper_state`.

- [ ] **Step 6: Verify perturbation semantics**

Failed close must leave `held_object is None`. Lift and transport offsets must preserve holding and the held-object transform. Run recovery, expert, environment, and full tests.

## Task 5: Recovery Episode Metadata and Generation

**Files:**
- Modify: `interaction_vla/config.py`
- Modify: `interaction_vla/data.py`
- Create: `configs/recovery_macos.yaml`
- Modify: `tests/interaction_vla/test_config.py`
- Modify: `tests/interaction_vla/test_data.py`
- Modify: `tests/interaction_vla/test_smoke_pipeline.py`

- [ ] **Step 1: Write failing recovery-config tests**

```python
cfg = load_config("configs/recovery_macos.yaml")
assert cfg.recovery.enabled
assert cfg.recovery.variants_per_episode == 1
assert cfg.data_dir == "outputs/interaction_vla/pilot/data"
```

Reject negative variants and enabled recovery with zero variants.

- [ ] **Step 2: Implement recovery configuration**

Add:

```python
@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = False
    variants_per_episode: int = 0
```

Add it to `ExperimentConfig` and YAML loading. Create `configs/recovery_macos.yaml` with pilot base data, `output_dir: outputs/interaction_vla/recovery`, one variant, pilot optimizer/model settings, and crowded evaluation.

- [ ] **Step 3: Write failing metadata round-trip tests**

Extend the desired `Episode` and `EpisodeArrays` API with:

```text
trajectory_kind
source_seed
variant_id
perturbation_kind
injection_phase
```

Assert a recovery episode round-trips every value and an old base `.npz` loads with `trajectory_kind == "base"`.

- [ ] **Step 4: Implement backward-compatible metadata**

Store the new fields only in the JSON metadata scalar. Use explicit defaults when keys are absent. Do not change scene/action tensor shapes.

- [ ] **Step 5: Write a failing recovery-collection test**

```python
episode = collect_recovery_episode(
    env,
    ScriptedExpert(),
    source_seed=42,
    object_count=2,
    spec=make_recovery_spec(42, 0),
)
assert episode.reason is TerminationReason.SUCCESS
assert episode.trajectory_kind == "recovery"
assert episode.steps[0].phase == episode.injection_phase
```

Also assert the first saved snapshot differs from the unperturbed phase snapshot and two runs produce identical saved tensors.

- [ ] **Step 6: Implement post-perturbation collection**

Replay the unsaved base prefix. When the expert enters the injection phase, discard the already-computed base action, apply the spec, resnapshot, and begin storing labelled recovery frames. Require exactly one injection and terminal success.

- [ ] **Step 7: Write failing split-isolation tests**

On a six-episode manifest, require recovery source seeds to equal the base train split and remain disjoint from validation/test seeds.

- [ ] **Step 8: Implement atomic recovery generation**

Add `augment_recovery_from_config(config_path)`. Resolve only base `manifest.json`, reproduce the `0.1/0.1` split, generate configured variants, and atomically write:

```text
recovery_manifest.json
recovery_rejections.json
recovery_source_split.json
```

Reject an all-failed generation run.

- [ ] **Step 9: Add the CLI**

Support `python -m interaction_vla.data augment-recovery --config configs/recovery_macos.yaml` alongside the existing `collect` command.

- [ ] **Step 10: Verify Task 5**

Run config, data, recovery, expert, and smoke tests, followed by the complete suite.

## Task 6: Combined Training Data and Provenance

**Files:**
- Modify: `interaction_vla/data.py`
- Modify: `interaction_vla/train.py`
- Modify: `tests/interaction_vla/test_train.py`

- [ ] **Step 1: Write failing recovery-manifest resolver tests**

Add `recovery_paths_from_manifest()` with the same traversal, duplicate, missing-file, empty-manifest, and ordering guarantees as the base resolver.

- [ ] **Step 2: Write a failing combined-data equality test**

Construct one base and one recovery file. Build the ordered combined path tuple used by Flat and Graph, then assert exact equality of node, edge, masks, proprioception, actions, phases, and sample weights before encoder execution.

- [ ] **Step 3: Implement training-only recovery inclusion**

In `train_from_config()`:

1. split only base-manifest paths;
2. resolve recovery paths only when recovery is enabled;
3. require each recovery `source_seed` in the base training seed set;
4. append recovery paths in manifest order;
5. fit statistics and phase weights on the combined paths;
6. keep validation/test paths base-only.

- [ ] **Step 4: Store and validate provenance**

Checkpoint base train seeds, ordered recovery filenames, and recovery count. On resume, reject any provenance difference before loading optimizer state.

- [ ] **Step 5: Verify exact resume behavior with recovery data**

Extend the current uninterrupted-versus-resumed test to a combined dataset and require bit-identical weights.

- [ ] **Step 6: Run focused and full tests**

Run `tests/interaction_vla/test_train.py`, `test_data.py`, then the full suite.

## Task 7: Baseline-versus-Recovery Reporting

**Files:**
- Modify: `interaction_vla/evaluate.py`
- Modify: `tests/interaction_vla/test_evaluate.py`

- [ ] **Step 1: Write a failing stage-comparison test**

```python
comparison = compare_training_stages(baseline_results, recovery_results)
assert comparison["graph"]["0"]["crowded_ood_success_delta"] == pytest.approx(0.25)
assert comparison["flat"]["0"]["crowded_ood_success_delta"] == pytest.approx(0.10)
```

Require the join key `(representation, model_seed, condition, seed, object_count, ablation)` and reject duplicate or mismatched paired cases.

- [ ] **Step 2: Run RED**

Run the focused test and confirm `compare_training_stages` is missing.

- [ ] **Step 3: Implement CSV loading and stage comparison**

Add `load_episode_results_csv()` with explicit bool, int, and float parsing. Compute paired success, grasp, wrong-object, and step deltas by representation, model seed, and condition.

- [ ] **Step 4: Extend the evaluation CLI**

Add:

```text
--baseline-episodes outputs/interaction_vla/crowded_baseline/evaluation/episodes.csv
```

When present, add `recovery_vs_baseline` to the new recovery report without modifying Stage A files. Keep existing evaluation commands valid when absent.

- [ ] **Step 5: Verify Task 7**

Run evaluation tests and the full regression suite.

## Task 8: Stage B Execution, Documentation, and Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/interaction_graph_pilot_results.md`
- Generate: `outputs/interaction_vla/recovery/**`

- [ ] **Step 1: Document exact commands and interpretation**

Document Stage A, recovery augmentation, three-seed Flat/Graph training, optional proprio training, Stage B evaluation, resume, and report paths. State that crowded scenes never enter training.

- [ ] **Step 2: Generate recovery data**

Run:

```bash
.venv/bin/python -m interaction_vla.data augment-recovery \
  --config configs/recovery_macos.yaml
```

Assert every source seed belongs to the base train split and every accepted trajectory terminates in success.

- [ ] **Step 3: Train Stage B policies**

Train Flat and Graph for model seeds 0, 1, and 2 with the recovery config. Train the proprio reference at seed 0. Preserve summaries, split/provenance files, metrics, and checkpoints.

- [ ] **Step 4: Run fixed Stage B evaluation**

Evaluate the six Flat/Graph recovery checkpoints on the same conditioned cases and pass Stage A `episodes.csv` through `--baseline-episodes`.

- [ ] **Step 5: Record results without selection**

Append exact per-seed recovery deltas, Graph-minus-Flat crowded deltas, wrong-object rates, held-out errors, edge-shuffle deltas, and strict criterion status to `docs/interaction_graph_pilot_results.md`. Do not retune seeds, grasp radius, layouts, or thresholds after observing results.

- [ ] **Step 6: Run fresh verification**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q interaction_vla tests/interaction_vla
.venv/bin/python -m interaction_vla.data --help
.venv/bin/python -m interaction_vla.train --help
.venv/bin/python -m interaction_vla.evaluate --help
```

Expected: zero failures; MPS-only tests may skip when unavailable.

- [ ] **Step 7: Validate artifacts programmatically**

Assert finite Stage A/Stage B JSON metrics, exact expected paired row counts, three model seeds, encoder parameter difference within 10 percent, no recovery source leakage, and explicit criterion booleans.

- [ ] **Step 8: Audit design coverage**

Compare every approved spec section against code, tests, configs, documentation, and generated artifacts. Record unavailable MPS execution and future DAgger/VLA work explicitly.

## Plan Self-Review

- Spec coverage: crowded sampling, Stage A ordering, split isolation, four recovery families, expert guards, combined-data provenance, conditioned reporting, stage comparison, and real execution each have an explicit task.
- Type consistency: `LayoutMode`, `RecoveryConfig`, `PerturbationKind`, `RecoverySpec`, metadata names, condition names, and CLI arguments are defined once and reused consistently.
- Leakage check: only the base manifest determines splits; recovery is admitted after splitting and validated against train seeds; crowded layouts remain evaluation-only.
- Placeholder scan: no `TBD`, `TODO`, deferred core behavior, or ambiguous “similar to” implementation step remains.
- Scope check: obstacles, DAgger, RGB, world models, and SmolVLA remain outside this plan.
