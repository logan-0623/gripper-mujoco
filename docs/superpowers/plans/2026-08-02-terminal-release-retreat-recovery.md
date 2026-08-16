# Terminal Release-and-Retreat Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the physical imitation dataset with correctly aligned phase labels and a fourth deterministic post-placement reclose recovery that teaches open-gripper vertical retreat.

**Architecture:** Keep the existing Franka expert, environment success rule, Graph/Flat inputs, and policy architecture unchanged. Extend the recovery specification and collector so the new failure is generated through real Cartesian control and contact physics, add per-kind quality gates, write to a separate provenance chain, and expose post-placement reclosing in evaluation.

**Tech Stack:** Python 3.12, NumPy, PyTorch, MuJoCo 3.3.4, YAML, pytest, tqdm.

**Design reference:** `docs/superpowers/specs/2026-08-02-terminal-release-retreat-recovery-design.md`

**Repository note:** This workspace is not a Git repository. Commit steps are omitted; old experiment artifacts are protected by explicit path and hash checks.

---

### Task 1: Align recorded phases with the actions that produced them

**Files:**

- Modify: `interaction_vla/physics_data.py`
- Modify: `tests/interaction_vla/test_physics_data.py`

- [x] **Step 1: Write the failing pre-action phase test**

Import `_expert_action_with_phase` and add:

```python
class TransitioningExpert:
    def __init__(self) -> None:
        self.phase = PhysicsExpertPhase.TRANSPORT

    def act(self, snapshot, contacts, grasp) -> np.ndarray:
        self.phase = PhysicsExpertPhase.RELEASE
        return np.zeros(7, dtype=np.float32)


def test_recorded_phase_is_the_phase_that_generated_the_action() -> None:
    expert = TransitioningExpert()
    phase, action = _expert_action_with_phase(
        expert, object(), object(), object()
    )
    assert phase == "transport"
    assert expert.phase is PhysicsExpertPhase.RELEASE
    np.testing.assert_array_equal(action, np.zeros(7, dtype=np.float32))
```

- [x] **Step 2: Run the test and verify RED**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_data.py::test_recorded_phase_is_the_phase_that_generated_the_action -q
```

Expected: FAIL because `_expert_action_with_phase` does not exist.

- [x] **Step 3: Implement phase capture before `expert.act()`**

Add after `_target_goal_distance`:

```python
def _expert_action_with_phase(
    expert: PhysicsScriptedExpert,
    snapshot,
    contacts,
    grasp,
) -> tuple[str, np.ndarray]:
    action_phase = expert.phase.value
    action = np.asarray(
        expert.act(snapshot, contacts, grasp), dtype=np.float32
    )
    return action_phase, action
```

Use it only for retained frames:

```python
action_phase, action = _expert_action_with_phase(
    expert,
    snapshot,
    env.contact_diagnostics,
    env.grasp_state,
)
actions.append(action.copy())
phases.append(action_phase)
```

Prefix actions before recovery injection remain unrecorded.

- [x] **Step 4: Add real-episode phase assertions**

Extend the saved physical episode test:

```python
release_actions = episode.actions[episode.phases == "release"]
retreat_actions = episode.actions[episode.phases == "retreat"]
assert len(release_actions) > 0
assert len(retreat_actions) > 0
assert np.all(release_actions[:, 6] == 1.0)
assert np.all(retreat_actions[:, 6] == 1.0)
assert float(np.mean(retreat_actions[:, 2])) > 0.5
```

- [x] **Step 5: Run the data tests GREEN**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_data.py -q
```

Expected: all physical-data tests PASS.

---

### Task 2: Add deterministic post-placement reclose recovery

**Files:**

- Modify: `interaction_vla/physics_recovery.py`
- Modify: `interaction_vla/physics_data.py`
- Modify: `tests/interaction_vla/test_physics_recovery.py`
- Modify: `tests/interaction_vla/test_physics_data.py`

- [x] **Step 1: Write failing spec and trigger tests**

Build four local variants and require:

```python
specs = tuple(
    make_physics_recovery_spec(42, index, kind_index=index)
    for index in range(4)
)
assert [spec.kind for spec in specs] == list(PhysicsRecoveryKind)
terminal = specs[3]
assert terminal.kind is PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE
assert terminal.trigger_phase == "retreat"
assert terminal.close_descent_steps == 5
assert recovery_trigger_ready(
    terminal,
    phase="retreat",
    stable_target=False,
    distance=0.04,
    supported_target=True,
)
assert not recovery_trigger_ready(
    terminal,
    phase="retreat",
    stable_target=False,
    distance=0.04,
    supported_target=False,
)
```

- [x] **Step 2: Run recovery tests and verify RED**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_recovery.py -q
```

Expected: FAIL because the fourth kind and fields do not exist.

- [x] **Step 3: Extend the recovery enum and dataclass**

```python
class PhysicsRecoveryKind(str, Enum):
    WRONG_WAY_TRANSPORT = "wrong_way_transport"
    PREMATURE_OPEN = "premature_open"
    RECEPTACLE_MISALIGNMENT = "receptacle_misalignment"
    POST_PLACEMENT_RECLOSE = "post_placement_reclose"


@dataclass(frozen=True)
class PhysicsRecoverySpec:
    source_seed: int
    variant_id: int
    kind: PhysicsRecoveryKind
    trigger_phase: str
    direction_sign: float
    translation_steps: int
    translation_distance: float
    open_substeps: int
    close_descent_steps: int = 0
```

Use kind-specific expected values:

```python
if self.kind is PhysicsRecoveryKind.WRONG_WAY_TRANSPORT:
    expected_phase, expected = "transport", (3, 0.06, 0, 0)
elif self.kind is PhysicsRecoveryKind.PREMATURE_OPEN:
    expected_phase, expected = "transport", (0, 0.0, 1, 0)
elif self.kind is PhysicsRecoveryKind.RECEPTACLE_MISALIGNMENT:
    expected_phase, expected = "transport", (2, 0.04, 0, 0)
else:
    expected_phase, expected = "retreat", (0, 0.0, 0, 5)
actual = (
    self.translation_steps,
    self.translation_distance,
    self.open_substeps,
    self.close_descent_steps,
)
if self.trigger_phase != expected_phase or actual != expected:
    raise ValueError(
        f"{self.kind.value} requires phase={expected_phase} and "
        f"steps/distance/open_substeps/close_descent_steps={expected}"
    )
```

Include `close_descent_steps` in `metadata()`.

- [x] **Step 4: Select recovery kind per source-local variant**

Change the factory to:

```python
def make_physics_recovery_spec(
    source_seed: int,
    variant_id: int,
    *,
    kind_index: int | None = None,
) -> PhysicsRecoverySpec:
    if source_seed < 0 or variant_id < 0:
        raise ValueError("source_seed and variant_id must be non-negative")
    resolved_kind_index = variant_id if kind_index is None else int(kind_index)
    if resolved_kind_index < 0:
        raise ValueError("kind_index must be non-negative")
    kind = tuple(PhysicsRecoveryKind)[
        resolved_kind_index % len(PhysicsRecoveryKind)
    ]
```

For the terminal kind set `trigger_phase="retreat"`, zero existing intervention
fields, and `close_descent_steps=5`. In collection call:

```python
spec = make_physics_recovery_spec(
    source_seed,
    variant_id,
    kind_index=local_variant_id,
)
```

This keeps old three-variant configs on the original three kinds.

- [x] **Step 5: Extend trigger readiness**

```python
def recovery_trigger_ready(
    spec: PhysicsRecoverySpec,
    *,
    phase: str,
    stable_target: bool,
    distance: float,
    supported_target: bool = False,
) -> bool:
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("recovery trigger distance must be finite and non-negative")
    if phase != spec.trigger_phase:
        return False
    if spec.kind is PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE:
        return supported_target and distance <= 0.065
    if not stable_target:
        return False
    if spec.kind is PhysicsRecoveryKind.RECEPTACLE_MISALIGNMENT:
        return distance <= 0.10
    return distance > 0.15
```

Pass `supported_target=target_name in
env.contact_diagnostics.object_receptacle` from collection.

- [x] **Step 6: Write the failing real-contact terminal test**

```python
def test_terminal_reclose_recovery_teaches_open_retreat() -> None:
    env = make_env()
    spec = make_physics_recovery_spec(11, 3, kind_index=3)
    episode = collect_physics_episode(
        env,
        PhysicsScriptedExpert(env.physics),
        seed=11,
        object_count=2,
        recovery=spec,
    )
    assert episode.reason == "success"
    assert episode.metadata["injection_phase"] == "retreat"
    assert set(episode.phases) == {"retreat"}
    assert episode.actions[0, 6] == 1.0
    assert episode.actions[0, 2] > 0.5
    assert episode.metadata["recovery"]["close_descent_steps"] == 5
```

- [x] **Step 7: Run the terminal test and verify RED**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_data.py::test_terminal_reclose_recovery_teaches_open_retreat -q
```

Expected: FAIL because the intervention has no terminal branch.

- [x] **Step 8: Implement the terminal intervention**

At the start of `_apply_recovery_intervention` add:

```python
if spec.kind is PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE:
    finger_before = float(np.mean(env.proprioception()[13:15]))
    action = np.asarray((0, 0, -1, 0, 0, 0, 0), dtype=np.float32)
    for _ in range(spec.close_descent_steps):
        snapshot = env.advance_intervention(
            action, substeps=env.physics.substeps
        )
        if target_name not in env.contact_diagnostics.object_receptacle:
            raise PhysicsRecoveryRejected(
                "target_support_lost_during_terminal_intervention"
            )
        if env.grasp_state.stable_object is not None:
            raise PhysicsRecoveryRejected(
                "stable_grasp_during_terminal_intervention"
            )
    finger_after = float(np.mean(env.proprioception()[13:15]))
    if finger_after >= finger_before - 1e-6:
        raise PhysicsRecoveryRejected("fingers_did_not_close")
    return snapshot
```

Keep the existing bilateral and unsupported postconditions only for transport
recovery kinds.

- [x] **Step 9: Run recovery/data tests GREEN**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_recovery.py tests/interaction_vla/test_physics_data.py -q
```

Expected: all tests PASS.

---

### Task 3: Enforce per-kind recovery data quality

**Files:**

- Modify: `interaction_vla/config.py`
- Modify: `interaction_vla/physics_data.py`
- Modify: `tests/interaction_vla/test_config.py`
- Modify: `tests/interaction_vla/test_physics_data.py`

- [x] **Step 1: Write failing config validation tests**

```python
def test_recovery_acceptance_rate_is_validated() -> None:
    assert RecoveryConfig(min_acceptance_rate=0.8).min_acceptance_rate == 0.8
    with pytest.raises(ValueError, match="acceptance"):
        RecoveryConfig(min_acceptance_rate=-0.1)
    with pytest.raises(ValueError, match="acceptance"):
        RecoveryConfig(min_acceptance_rate=1.1)
```

- [x] **Step 2: Run the config test and verify RED**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_config.py::test_recovery_acceptance_rate_is_validated -q
```

Expected: FAIL because `min_acceptance_rate` is not defined.

- [x] **Step 3: Extend `RecoveryConfig`**

```python
@dataclass(frozen=True)
class RecoveryConfig:
    enabled: bool = False
    variants_per_episode: int = 0
    min_acceptance_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.variants_per_episode < 0:
            raise ValueError("recovery.variants_per_episode must not be negative")
        if self.enabled and self.variants_per_episode < 1:
            raise ValueError(
                "recovery.variants_per_episode must be positive when recovery is enabled"
            )
        if (
            not math.isfinite(self.min_acceptance_rate)
            or not 0.0 <= self.min_acceptance_rate <= 1.0
        ):
            raise ValueError(
                "recovery.min_acceptance_rate must be finite and within [0, 1]"
            )
```

- [x] **Step 4: Write failing per-kind quality tests**

```python
def test_recovery_quality_requires_every_kind_and_minimum_rate() -> None:
    attempted = {
        PhysicsRecoveryKind.WRONG_WAY_TRANSPORT: 10,
        PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE: 10,
    }
    accepted = {
        PhysicsRecoveryKind.WRONG_WAY_TRANSPORT: 8,
        PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE: 7,
    }
    failed_summary = recovery_quality_summary(
        attempted, accepted, minimum_rate=0.8
    )
    with pytest.raises(RuntimeError, match="post_placement_reclose"):
        require_recovery_quality(failed_summary)
    accepted[PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE] = 8
    summary = recovery_quality_summary(
        attempted, accepted, minimum_rate=0.8
    )
    assert summary["post_placement_reclose"] == {
        "attempted": 10,
        "accepted": 8,
        "acceptance_rate": 0.8,
        "passed": True,
    }
```

- [x] **Step 5: Implement the quality summary and collection counters**

Add `Counter` and `Mapping` imports, then implement:

```python
def recovery_quality_summary(
    attempted: Mapping[PhysicsRecoveryKind, int],
    accepted: Mapping[PhysicsRecoveryKind, int],
    *,
    minimum_rate: float,
) -> dict[str, dict[str, object]]:
    if not np.isfinite(minimum_rate) or not 0.0 <= minimum_rate <= 1.0:
        raise ValueError("minimum recovery acceptance rate must be within [0, 1]")
    summary: dict[str, dict[str, object]] = {}
    for kind in PhysicsRecoveryKind:
        attempts = int(attempted.get(kind, 0))
        if attempts == 0:
            continue
        successes = int(accepted.get(kind, 0))
        rate = successes / attempts
        passed = successes > 0 and rate >= minimum_rate
        summary[kind.value] = {
            "attempted": attempts,
            "accepted": successes,
            "acceptance_rate": float(rate),
            "passed": passed,
        }
    return summary


def require_recovery_quality(
    summary: Mapping[str, Mapping[str, object]],
) -> None:
    failed = [
        (
            f"{kind}={metrics['accepted']}/{metrics['attempted']} "
            f"({float(metrics['acceptance_rate']):.1%})"
        )
        for kind, metrics in summary.items()
        if not bool(metrics["passed"])
    ]
    if failed:
        raise RuntimeError(
            "recovery quality gate failed: " + ", ".join(failed)
        )
```

In recovery collection maintain:

```python
attempted_by_kind: Counter[PhysicsRecoveryKind] = Counter()
accepted_by_kind: Counter[PhysicsRecoveryKind] = Counter()
attempted_by_kind[spec.kind] += 1
```

Increment accepted only after a successful NPZ save. Finish with this exact order
so a failed gate still leaves diagnostics:

```python
quality = recovery_quality_summary(
    attempted_by_kind,
    accepted_by_kind,
    minimum_rate=config.recovery.min_acceptance_rate,
)
_write_json(data_dir / "recovery_quality.json", quality)
require_recovery_quality(quality)
```

- [x] **Step 6: Run config and data tests GREEN**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_config.py tests/interaction_vla/test_physics_data.py -q
```

Expected: all tests PASS.

---

### Task 4: Add isolated terminal-recovery configurations

**Files:**

- Create: `configs/physics_terminal_recovery_smoke_macos.yaml`
- Create: `configs/physics_terminal_recovery_pilot_macos.yaml`
- Modify: `tests/interaction_vla/test_config.py`

- [x] **Step 1: Write the failing config contract test**

```python
@pytest.mark.parametrize(
    ("path", "episodes", "epochs", "minimum_rate"),
    [
        ("configs/physics_terminal_recovery_smoke_macos.yaml", 4, 1, 0.5),
        ("configs/physics_terminal_recovery_pilot_macos.yaml", 50, 80, 0.8),
    ],
)
def test_terminal_recovery_configs_are_isolated_and_use_four_variants(
    path: str,
    episodes: int,
    epochs: int,
    minimum_rate: float,
) -> None:
    config = load_config(path)
    assert config.backend == "franka_contact"
    assert config.train.episodes == episodes
    assert config.train.epochs == epochs
    assert config.recovery.variants_per_episode == 4
    assert config.recovery.min_acceptance_rate == minimum_rate
    assert "terminal_recovery" in config.data_dir
    assert "terminal_recovery" in config.output_dir
```

- [x] **Step 2: Run the test and verify RED**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_config.py::test_terminal_recovery_configs_are_isolated_and_use_four_variants -q
```

Expected: FAIL because the YAML files do not exist.

- [x] **Step 3: Create the smoke YAML**

```yaml
name: interaction_graph_physics_terminal_recovery_smoke
seed: 42
device: auto
backend: franka_contact
max_objects: 5
data_dir: outputs/interaction_graph_physics/terminal_recovery_smoke/data
output_dir: outputs/interaction_graph_physics/terminal_recovery_smoke
train:
  object_counts: [2, 3]
  episodes: 4
  batch_size: 32
  epochs: 1
  learning_rate: 0.003
  model_seeds: [0]
eval:
  object_counts: [2, 3, 4, 5]
  ood_object_counts: [4, 5]
  crowded_object_counts: [4, 5]
  episodes_per_count: 1
  max_steps: 180
model:
  embedding_dim: 16
  hidden_dim: 16
  message_rounds: 1
  action_dim: 7
environment:
  max_steps: 180
  workspace_low: [0.25, -0.35, 0.23]
  workspace_high: [0.78, 0.35, 0.75]
  min_object_distance: 0.12
  crowded_anchor_min_distance: 0.055
  crowded_anchor_max_distance: 0.075
recovery:
  enabled: true
  variants_per_episode: 4
  min_acceptance_rate: 0.5
physics:
  timestep: 0.002
  policy_hz: 20
  substeps: 25
  translation_delta: 0.02
  rotation_delta: 0.05235987755982988
  settle_steps: 250
  stable_grasp_frames: 10
  stable_lift_height: 0.01
  ik_damping: 0.05
  ik_iterations: 20
  ik_position_tolerance: 0.002
  ik_orientation_tolerance: 0.03490658503988659
  expert_gate_cases_per_condition: 20
  randomization:
    enabled: false
    object_mass_scale: [0.8, 1.2]
    friction_scale: [0.8, 1.2]
    joint_damping_scale: [0.9, 1.1]
recording:
  enabled: false
  width: 256
  height: 256
  cameras: [agentview, wristview, sideview, topview]
```

- [x] **Step 4: Create the pilot YAML**

```yaml
name: interaction_graph_physics_terminal_recovery_pilot
seed: 42
device: auto
backend: franka_contact
max_objects: 5
data_dir: outputs/interaction_graph_physics/terminal_recovery_pilot/data
output_dir: outputs/interaction_graph_physics/terminal_recovery_pilot
train:
  object_counts: [2, 3]
  episodes: 50
  batch_size: 64
  epochs: 80
  learning_rate: 0.001
  model_seeds: [0, 1, 2]
eval:
  object_counts: [2, 3, 4, 5]
  ood_object_counts: [4, 5]
  crowded_object_counts: [4, 5]
  episodes_per_count: 20
  max_steps: 180
model:
  embedding_dim: 64
  hidden_dim: 64
  message_rounds: 2
  action_dim: 7
environment:
  max_steps: 180
  workspace_low: [0.25, -0.35, 0.23]
  workspace_high: [0.78, 0.35, 0.75]
  min_object_distance: 0.12
  crowded_anchor_min_distance: 0.055
  crowded_anchor_max_distance: 0.075
recovery:
  enabled: true
  variants_per_episode: 4
  min_acceptance_rate: 0.8
physics:
  timestep: 0.002
  policy_hz: 20
  substeps: 25
  translation_delta: 0.02
  rotation_delta: 0.05235987755982988
  settle_steps: 250
  stable_grasp_frames: 10
  stable_lift_height: 0.01
  ik_damping: 0.05
  ik_iterations: 20
  ik_position_tolerance: 0.002
  ik_orientation_tolerance: 0.03490658503988659
  expert_gate_cases_per_condition: 20
  randomization:
    enabled: false
    object_mass_scale: [0.8, 1.2]
    friction_scale: [0.8, 1.2]
    joint_damping_scale: [0.9, 1.1]
recording:
  enabled: false
  width: 256
  height: 256
  cameras: [agentview, wristview, sideview, topview]
```

- [x] **Step 5: Run all config tests GREEN**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_config.py -q
```

Expected: new configs pass and existing three-variant configs remain unchanged.

---

### Task 5: Diagnose post-placement reclosing in evaluation

**Files:**

- Modify: `interaction_vla/physics_evaluate.py`
- Modify: `tests/interaction_vla/test_physics_evaluate.py`

- [x] **Step 1: Write failing episode and aggregate tests**

Construct a Flat result with `post_placement_reclose=True` and a Graph result with
it false, then require:

```python
assert report["by_policy"]["flat_seed_0"][
    "post_placement_reclose_rate"
] == 1.0
assert report["by_policy"]["graph_seed_0"][
    "post_placement_reclose_rate"
] == 0.0
```

Extend the rollout fixture so step one reports stable placement and step two
executes `g < 0.5`; require `result.post_placement_reclose is True`.

- [x] **Step 2: Run evaluation tests and verify RED**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_evaluate.py -q
```

Expected: FAIL because the result and metric do not exist.

- [x] **Step 3: Track reclosing after placement**

Add to `PhysicsEpisodeResult`:

```python
post_placement_reclose: bool = False
```

Initialize `post_placement_reclose = False`. Immediately before
`env.step(action)` add:

```python
if placement and float(action[6]) < 0.5:
    post_placement_reclose = True
```

Return the flag and aggregate it in `_metrics`:

```python
"post_placement_reclose_rate": float(
    np.mean([value.post_placement_reclose for value in values])
),
```

- [x] **Step 4: Run evaluation tests GREEN**

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_evaluate.py -q
```

Expected: all evaluation tests PASS.

---

### Task 6: Document, verify, and run the isolated smoke pipeline

**Files/artifacts:**

- Modify: `README.md`
- Verify: `interaction_vla/physics_provenance.py`
- Create at runtime: `outputs/interaction_graph_physics/terminal_recovery_smoke/`
- Preserve: `outputs/interaction_graph_physics/recovery_pilot/`

- [x] **Step 1: Document the v2 pilot commands**

Add a README section describing corrected phase alignment, the fourth recovery,
and the isolated output. Include:

```bash
.venv/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_terminal_recovery_pilot_macos.yaml

.venv/bin/python -m interaction_vla.physics_data collect \
  --config configs/physics_terminal_recovery_pilot_macos.yaml

for representation in flat graph; do
  .venv/bin/python -m interaction_vla.train \
    --config configs/physics_terminal_recovery_pilot_macos.yaml \
    --representation "$representation" \
    --model-seed 0
done

.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_terminal_recovery_pilot_macos.yaml \
  --model-seeds 0 \
  --episodes-per-count 5 \
  --conditions id_normal \
  --output outputs/interaction_graph_physics/terminal_recovery_pilot/evaluation/id_sanity_report.json
```

- [x] **Step 2: Run focused and full verification**

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_config.py \
  tests/interaction_vla/test_physics_recovery.py \
  tests/interaction_vla/test_physics_data.py \
  tests/interaction_vla/test_physics_evaluate.py -q
.venv/bin/python -m compileall -q interaction_vla tests/interaction_vla
.venv/bin/python -m pytest -q
```

Expected: compilation exits zero and all tests PASS.

- [x] **Step 3: Hash old artifacts before smoke**

Record SHA-256 for these exact files:

```text
outputs/interaction_graph_physics/recovery_pilot/data/manifest.json
outputs/interaction_graph_physics/recovery_pilot/flat/seed_0/checkpoint.pt
outputs/interaction_graph_physics/recovery_pilot/graph/seed_0/checkpoint.pt
outputs/interaction_graph_physics/recovery_pilot/evaluation/report.json
```

- [x] **Step 4: Run the complete new smoke chain**

```bash
.venv/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_terminal_recovery_smoke_macos.yaml
.venv/bin/python -m interaction_vla.physics_data collect \
  --config configs/physics_terminal_recovery_smoke_macos.yaml
.venv/bin/python -m interaction_vla.train \
  --config configs/physics_terminal_recovery_smoke_macos.yaml \
  --representation flat --model-seed 0
.venv/bin/python -m interaction_vla.train \
  --config configs/physics_terminal_recovery_smoke_macos.yaml \
  --representation graph --model-seed 0
.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_terminal_recovery_smoke_macos.yaml \
  --model-seeds 0 --episodes-per-count 1 --conditions id_normal \
  --output outputs/interaction_graph_physics/terminal_recovery_smoke/evaluation/id_sanity_report.json
```

Expected: tqdm completes each stage; four base episodes are accepted; all four
recovery kinds appear and pass the smoke quality gate; both checkpoints and the
four-rollout ID report are written.

- [x] **Step 5: Verify isolation and smoke data semantics**

Recompute the Step 3 hashes and require exact equality. Inspect
`recovery_quality.json` and require `passed=true` for every kind. Load each
terminal recovery NPZ and require only `retreat` phases plus an open/up first
action.

- [x] **Step 6: Request review and run final verification**

Review hardening completed before final verification:

- explicitly include configured zero-attempt kinds in `recovery_quality.json`
  and fail their gate;
- return controller/physics diagnostics from every unlabelled intervention step
  and reject failures with their subtype;
- assert the five terminal close/down actions, physical descent, finger closure,
  continuous receptacle support, and first-placement reclose metric boundary.

Use `requesting-code-review`, fix every Critical and Important issue, then run:

```bash
.venv/bin/python -m compileall -q interaction_vla tests/interaction_vla
.venv/bin/python -m pytest -q
```

Expected: zero failures. Do not automatically run the 50-episode pilot; hand the
tqdm commands to the user after the complete smoke chain is verified.
