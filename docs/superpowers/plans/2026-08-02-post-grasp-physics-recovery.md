# Post-Grasp Physics Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace reset-only physical recovery samples with three balanced post-grasp interventions and add closed-loop transport/release diagnostics before the user retrains seed 0.

**Architecture:** `physics_recovery.py` owns deterministic specifications, trigger predicates, and intervention actions; `physics_env.py` exposes a bounded unlabelled physics-step API; `physics_data.py` runs an unsaved expert prefix, injects the perturbation, validates it, and records only corrective suffixes. Evaluation derives transport progress and premature opening from each rollout without changing policy inputs or task success.

**Tech Stack:** Python 3.12, NumPy, MuJoCo, PyTorch, pytest, YAML, tqdm.

---

### Task 1: Deterministic Post-Grasp Recovery Specifications

**Files:**
- Modify: `tests/interaction_vla/test_physics_recovery.py`
- Modify: `interaction_vla/physics_recovery.py`

- [x] **Step 1: Replace the old spec assertions with failing tests for the three balanced kinds**

```python
def test_post_grasp_specs_are_deterministic_and_balanced_per_source() -> None:
    first = tuple(make_physics_recovery_spec(42, index) for index in range(3))
    second = tuple(make_physics_recovery_spec(42, index) for index in range(3))
    assert [spec.kind for spec in first] == list(PhysicsRecoveryKind)
    for left, right in zip(first, second, strict=True):
        assert left.metadata() == right.metadata()


def test_recovery_trigger_requires_transport_and_stable_target() -> None:
    spec = make_physics_recovery_spec(42, 0)
    assert not recovery_trigger_ready(spec, phase="lift", stable_target=True, distance=0.3)
    assert not recovery_trigger_ready(spec, phase="transport", stable_target=False, distance=0.3)
    assert recovery_trigger_ready(spec, phase="transport", stable_target=True, distance=0.3)
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_recovery.py -q`

Expected: failure because the new enum members, fields, and trigger helper do not exist.

- [x] **Step 3: Implement the minimal deterministic spec and pure trigger/action helpers**

```python
class PhysicsRecoveryKind(str, Enum):
    WRONG_WAY_TRANSPORT = "wrong_way_transport"
    PREMATURE_OPEN = "premature_open"
    RECEPTACLE_MISALIGNMENT = "receptacle_misalignment"


@dataclass(frozen=True)
class PhysicsRecoverySpec:
    source_seed: int
    variant_id: int
    kind: PhysicsRecoveryKind
    trigger_phase: str
    direction_sign: float
    translation_steps: int
    open_substeps: int


def recovery_trigger_ready(
    spec: PhysicsRecoverySpec,
    *,
    phase: str,
    stable_target: bool,
    distance: float,
) -> bool:
    if phase != spec.trigger_phase or not stable_target:
        return False
    if spec.kind is PhysicsRecoveryKind.RECEPTACLE_MISALIGNMENT:
        return distance <= 0.10
    return distance > 0.15


def recovery_translation_direction(
    spec: PhysicsRecoverySpec,
    target_xy: np.ndarray,
    receptacle_xy: np.ndarray,
) -> np.ndarray:
    radial = np.asarray(target_xy, dtype=np.float32) - np.asarray(
        receptacle_xy, dtype=np.float32
    )
    radial /= max(float(np.linalg.norm(radial)), 1e-8)
    if spec.kind is PhysicsRecoveryKind.WRONG_WAY_TRANSPORT:
        return radial
    return spec.direction_sign * np.asarray((-radial[1], radial[0]), dtype=np.float32)
```

`make_physics_recovery_spec` assigns kind by `variant_id % 3`, uses three/zero/two translation steps for wrong-way/open/misalignment, uses one open substep only for `PREMATURE_OPEN`, and serializes every field.

- [x] **Step 4: Run the recovery tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_recovery.py -q`

Expected: all tests in the file pass.

- [x] **Step 5: Record the task checkpoint**

No Git commit is possible because `/Users/loganluo/lerobot-mujoco` is not a Git worktree. Mark Task 1 complete in this plan after the focused tests pass.

### Task 2: Bounded Unlabelled MuJoCo Intervention Steps

**Files:**
- Modify: `tests/interaction_vla/test_physics_env.py`
- Modify: `interaction_vla/physics_env.py`

- [x] **Step 1: Add failing tests for intervention timing and validation**

```python
def test_unlabelled_intervention_advances_physics_without_policy_step() -> None:
    env = make_env()
    env.reset(seed=11, object_count=2)
    start_time = float(env.data.time)
    snapshot = env.advance_intervention(np.asarray([0, 0, 0, 0, 0, 0, 1]), substeps=4)
    assert env.step_count == 0
    assert float(env.data.time) == pytest.approx(start_time + 4 * env.physics.timestep)
    assert snapshot.gripper.name == "gripper"


@pytest.mark.parametrize("substeps", [0, 26])
def test_unlabelled_intervention_rejects_invalid_substep_count(substeps: int) -> None:
    env = make_env()
    env.reset(seed=11, object_count=2)
    with pytest.raises(ValueError, match="substeps"):
        env.advance_intervention(np.zeros(7), substeps=substeps)
```

- [x] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_env.py -q`

Expected: failure because `advance_intervention` does not exist.

- [x] **Step 3: Add the minimal environment API and remove reset-time recovery actions**

```python
def advance_intervention(self, action: np.ndarray, *, substeps: int) -> SceneSnapshot:
    self._require_initialized()
    if not 1 <= int(substeps) <= self.physics.substeps:
        raise ValueError("intervention substeps must be between 1 and physics.substeps")
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Cartesian action must be a finite vector with shape (7,)")
    self.controller.apply_action(values)
    for _ in range(int(substeps)):
        mujoco.mj_step(self.model, self.data)
        self.contact_diagnostics = self.contact_parser.parse(self.data)
        self.grasp_state = self.grasp_tracker.update(
            self.contact_diagnostics,
            object_bottom_heights=self._object_bottom_heights(),
        )
        self._update_placement_frames()
    self._update_wrist_camera()
    self._last_snapshot = self._build_snapshot()
    return self._last_snapshot
```

Remove `recovery` from `FrankaContactEnv.reset`, remove its reset-time action, and remove recovery metadata from `physics_metadata`; recovery provenance belongs to the episode archive.

- [x] **Step 4: Run environment and no-attachment audits and verify GREEN**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_env.py tests/interaction_vla/test_no_attachment_audit.py -q`

Expected: all selected tests pass and no object-state attachment/edit path is introduced.

- [x] **Step 5: Record the task checkpoint**

No Git commit is possible. Mark Task 2 complete after both focused files pass.

### Task 3: Recovery-Aware Expert and Suffix-Only Physical Collection

**Files:**
- Modify: `tests/interaction_vla/test_physics_expert.py`
- Modify: `tests/interaction_vla/test_physics_data.py`
- Modify: `interaction_vla/physics_expert.py`
- Modify: `interaction_vla/physics_data.py`

- [x] **Step 1: Add a failing expert test for premature-open correction**

```python
def test_transport_closes_a_partially_open_gripper_away_from_goal() -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2)
    target = snapshot.target_object.name
    expert = PhysicsScriptedExpert(env.physics)
    expert.reset(seed=11)
    expert.phase = PhysicsExpertPhase.TRANSPORT
    action = expert.act(snapshot, env.contact_diagnostics, grasp_state(bilateral=target, stable=target))
    assert expert.phase is PhysicsExpertPhase.TRANSPORT
    assert action[6] == 0.0
```

Extend this test with a snapshot whose target is at the receptacle but whose gripper is not holding the target, and assert that RELEASE is not selected.

- [x] **Step 2: Run the expert tests and verify RED for the new release guard**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_expert.py -q`

Expected: the release-guard assertion fails under the old phase-only behavior.

- [x] **Step 3: Implement observable post-grasp phase resynchronization**

Add `_resynchronize_post_grasp(snapshot, grasp)` at the start of `act`. When the target is bilaterally held, select LIFT below the safe TCP height and TRANSPORT above it unless already correctly releasing over the receptacle. Require bilateral target contact and target-to-receptacle XY distance at most `0.065` before entering or remaining in RELEASE. Remove recovery offsets and the recovery parameter from `PhysicsScriptedExpert.reset`.

- [x] **Step 4: Run the expert tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_expert.py -q`

Expected: all expert tests pass, including the real contact pick-and-place test.

- [x] **Step 5: Add failing collection tests for suffix-only data and corrective first labels**

```python
@pytest.mark.parametrize("variant_id", [0, 1, 2])
def test_post_grasp_recovery_saves_only_corrective_suffix(variant_id: int) -> None:
    env = make_env()
    spec = make_physics_recovery_spec(11, variant_id)
    episode = collect_physics_episode(
        env,
        PhysicsScriptedExpert(env.physics),
        seed=11,
        object_count=2,
        recovery=spec,
    )
    assert episode.reason == "success"
    assert episode.metadata["injection_phase"] == "transport"
    assert episode.phases[0] in {"lift", "transport"}
    assert episode.actions[0, 6] == 0.0
    assert "approach" not in set(episode.phases)
```

For wrong-way and misalignment variants, also assert that the first action has positive dot product with the post-intervention target-to-receptacle vector stored in the first graph state. For premature-open, assert the metadata reports one open substep.

- [x] **Step 6: Run the collection tests and verify RED**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_data.py -q`

Expected: old reset-time recoveries include the full prefix and report `injection_phase="reset"`.

- [x] **Step 7: Implement trigger, intervention, validation, and suffix recording**

Add `PhysicsRecoveryRejected` carrying a stable reason string. In `collect_physics_episode`, reset env/expert normally; run the expert prefix without appending arrays; call `recovery_trigger_ready`; apply three full-step wrong-way actions, one-substep premature-open, or two full-step tangent actions through `advance_intervention`; validate bilateral target contact and the premature-open finger change; then record only corrective expert frames. Raise `PhysicsRecoveryRejected` if the trigger is never reached or intervention postconditions fail.

In `collect_from_config`, catch `PhysicsRecoveryRejected`, write source seed, variant ID, kind, object count, and reason to `recovery_rejections.json`, and continue. Allocate `variant_id = source_index * variants_per_episode + local_variant_id`; with three configured variants this gives all kinds per source. Store actual injection phase and serialized spec in accepted metadata.

- [x] **Step 8: Run recovery, collection, and provenance tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_recovery.py tests/interaction_vla/test_physics_expert.py tests/interaction_vla/test_physics_data.py tests/interaction_vla/test_no_attachment_audit.py -q`

Expected: all selected tests pass.

- [x] **Step 9: Record the task checkpoint**

No Git commit is possible. Mark Task 3 complete after the focused recovery stack passes.

### Task 4: Transport Progress and Premature-Open Evaluation Metrics

**Files:**
- Modify: `tests/interaction_vla/test_physics_evaluate.py`
- Modify: `interaction_vla/physics_evaluate.py`

- [x] **Step 1: Extend the synthetic aggregation test and verify RED**

Construct Flat with `transport_progress=0.05`, `transport_progress_rate=0.2`, and `premature_open=True`; construct Graph with `0.20`, `0.8`, and `False`; assert:

```python
assert report["by_policy"]["graph_seed_0"]["mean_transport_progress"] == 0.20
assert report["by_policy"]["graph_seed_0"]["premature_open_rate"] == 0.0
paired = report["graph_vs_flat"]["by_model_seed"]["0"]
assert paired["transport_progress_rate_delta"] == pytest.approx(0.6)
assert paired["premature_open_delta"] == -1.0
```

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_evaluate.py::test_physics_aggregation_reports_interaction_metrics_and_paired_delta -q`

Expected: failure because the result fields and aggregates do not exist.

- [x] **Step 2: Add result fields and closed-loop tracking**

Append defaulted fields to `PhysicsEpisodeResult`:

```python
transport_progress: float = 0.0
transport_progress_rate: float = 0.0
premature_open: bool = False
```

During rollout, capture target-to-receptacle XY distance immediately after the first stable target grasp, maintain the minimum later distance, and mark premature opening when the pre-action state is post-stable-grasp, farther than 0.065 m, and `action[6] >= 0.5`. At termination compute non-negative progress and its clipped normalized rate.

- [x] **Step 3: Add aggregate and paired metric keys**

`_metrics` adds `mean_transport_progress`, `mean_transport_progress_rate`, and `premature_open_rate`. `_paired_deltas` adds right-minus-left `transport_progress_rate_delta` and `premature_open_delta`, both overall and inside each condition.

- [x] **Step 4: Run evaluation tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_physics_evaluate.py -q`

Expected: all evaluation tests pass, including subset progress behavior.

- [x] **Step 5: Record the task checkpoint**

No Git commit is possible. Mark Task 4 complete after evaluation tests pass.

### Task 5: Configuration, Documentation, and End-to-End Verification

**Files:**
- Modify: `configs/physics_recovery_pilot_macos.yaml`
- Modify: `configs/physics_recovery_smoke_macos.yaml`
- Modify: `README.md`
- Modify: `tests/interaction_vla/test_config.py`
- Modify: `tests/interaction_vla/test_physics_data.py`
- Modify: `docs/superpowers/plans/2026-08-02-post-grasp-physics-recovery.md`

- [x] **Step 1: Add failing configuration and manifest-balance assertions**

Assert both physical recovery configs load with `variants_per_episode == 3`. In the small collection test, group accepted and rejected recovery records together by source seed and assert attempted kinds are exactly `wrong_way_transport`, `premature_open`, and `receptacle_misalignment`, with unique variant IDs.

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_config.py tests/interaction_vla/test_physics_data.py -q`

Expected: config assertion fails while both YAML files still specify one variant.

- [x] **Step 2: Update the two YAML files**

```yaml
recovery:
  enabled: true
  variants_per_episode: 3
```

- [x] **Step 3: Replace stale README recovery text and add the seed-0 workflow**

Document that the physical recovery dataset now contains suffix-only post-grasp interventions, that all prior gate/data/checkpoint artifacts are stale, and provide this exact sequence:

```bash
.venv/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_recovery_pilot_macos.yaml

.venv/bin/python -m interaction_vla.physics_data collect \
  --config configs/physics_recovery_pilot_macos.yaml

for representation in flat graph; do
  .venv/bin/python -m interaction_vla.train \
    --config configs/physics_recovery_pilot_macos.yaml \
    --representation "$representation" \
    --model-seed 0
done

.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_recovery_pilot_macos.yaml \
  --model-seeds 0
```

State explicitly that `collect` replaces manifests and same-named episode files but does not delete unrelated output artifacts.

- [x] **Step 4: Run config and focused physical recovery tests**

Run: `.venv/bin/python -m pytest tests/interaction_vla/test_config.py tests/interaction_vla/test_physics_recovery.py tests/interaction_vla/test_physics_env.py tests/interaction_vla/test_physics_expert.py tests/interaction_vla/test_physics_data.py tests/interaction_vla/test_physics_evaluate.py tests/interaction_vla/test_no_attachment_audit.py -q`

Expected: all selected tests pass.

- [x] **Step 5: Run a deterministic real-MuJoCo recovery probe**

Use a temporary output/config copy with one or two successful base episodes, three variants, and the existing expert gate contract regenerated for that temporary config. Run physical collection, then read `recovery_manifest.json` and `recovery_rejections.json` to report accepted/rejected counts by kind. Do not alter the user's pilot output directory during this probe.

- [x] **Step 6: Run the full regression suite and compilation**

Run: `.venv/bin/python -m pytest -q`

Expected: zero failures.

Run: `.venv/bin/python -m compileall -q interaction_vla tests/interaction_vla`

Expected: exit code 0 with no output.

- [x] **Step 7: Review the implementation against the written specification**

Confirm each of the following from code and fresh command output: three balanced kinds; physical-only interventions; suffix-only accepted data; split isolation; audit metadata/rejections; unchanged Graph/Flat inputs; new evaluation metrics and paired deltas; provenance invalidation; no attachment or object-qpos edits.

- [x] **Step 8: Mark every completed checkbox and hand off commands**

Update this plan's checkboxes only for steps backed by executed command output. Report any recovery kind that lacks an accepted smoke trajectory instead of claiming completion.
