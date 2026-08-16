# IK-Safe Learned-Policy Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply one deterministic IK-feasibility backtracking layer to every learned representation, expose its intervention burden in reports and visualizations, and provide a non-destructive ID sanity workflow.

**Architecture:** Put the safety algorithm in a new evaluation-only module that is not part of the physical provenance hash. Integrate it identically into Flat, Graph, edge-shuffle, and learned visualization paths; preserve raw-policy reproduction through a global disable flag. Extend evaluation scope, aggregate metrics, output routing, and ID sanity without changing controller, config, data, gate, or checkpoint files.

**Tech Stack:** Python 3.12, NumPy, PyTorch, MuJoCo controller diagnostics, argparse, pytest, tqdm.

**Design reference:** `docs/superpowers/specs/2026-08-02-ik-safe-policy-evaluation-design.md`

**Repository note:** This workspace is not a Git repository, so commit steps are omitted. Verification includes confirming the controller provenance hash remains unchanged.

---

### Task 1: Build deterministic IK action projection

**Files:**

- Create: `interaction_vla/physics_action_safety.py`
- Create: `tests/interaction_vla/test_physics_action_safety.py`

- [x] **Step 1: Write failing tests for full-scale acceptance and deterministic backtracking**

Use a fake controller whose `apply_action` returns `ControllerDiagnostics` based on pose norm. Assert that a feasible raw action returns scale `1.0`, while an initially infeasible action selects the first feasible scale `0.25` and preserves the gripper coordinate:

```python
result = project_cartesian_action(
    controller,
    np.asarray((0.8, 0, 0, 0, 0, 0, 1), dtype=np.float32),
    scales=(1.0, 0.5, 0.25, 0.0),
)
assert result.scale == 0.25
assert result.action[6] == 1.0
assert np.array_equal(result.action[:6], result.raw_action[:6] * 0.25)
assert result.raw_diagnostics.ik_limited is True
assert result.projected_diagnostics.ik_limited is False
```

- [x] **Step 2: Run the safety tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_action_safety.py -q
```

Expected: FAIL because `physics_action_safety` does not exist.

- [x] **Step 3: Implement the projection result and minimal backtracking loop**

Create:

```python
DEFAULT_IK_PROJECTION_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0)


@dataclass(frozen=True)
class IKProjectionResult:
    raw_action: np.ndarray
    action: np.ndarray
    scale: float
    raw_diagnostics: ControllerDiagnostics
    projected_diagnostics: ControllerDiagnostics


def project_cartesian_action(
    controller: FrankaCartesianController,
    action: np.ndarray,
    *,
    scales: tuple[float, ...] = DEFAULT_IK_PROJECTION_SCALES,
) -> IKProjectionResult:
```

Validate a finite `(7,)` action. Require at least two finite scales, first `1.0`, last `0.0`, values in `[0, 1]`, and strict decrease. Copy the raw action. For each scale, multiply only `candidate[:6]`, preserve `candidate[6]`, call `controller.apply_action`, retain the first diagnostics as `raw_diagnostics`, and return the first non-limited candidate. Raise `RuntimeError("zero-pose Cartesian action is unexpectedly IK-limited")` if every scale is limited.

- [x] **Step 4: Add RED tests for invalid schedules and zero-pose failure**

Parameterize schedules that omit `1.0`, omit `0.0`, contain duplicates, increase, contain non-finite values, or leave `[0, 1]`. Add a fake controller that always reports `ik_limited=True` and require the exact zero-pose error.

- [x] **Step 5: Implement validation and run GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_action_safety.py -q
```

Expected: all safety tests PASS.

---

### Task 2: Integrate projection and diagnostics into physics evaluation

**Files:**

- Modify: `interaction_vla/physics_evaluate.py`
- Modify: `tests/interaction_vla/test_physics_evaluate.py`

- [x] **Step 1: Write failing rollout tests for shared projection behavior**

Patch `project_cartesian_action` with a recorder, use the existing lightweight rollout fixtures, and assert:

```python
assert projection_calls == flat_steps + graph_steps
assert result.ik_projection_rate == projected_steps / result.steps
assert result.zero_pose_projection_rate == zero_steps / result.steps
assert result.mean_ik_projection_scale == pytest.approx(sum(scales) / result.steps)
```

Add a second test with `ik_projection=False` and assert the projector is never called and scale diagnostics remain `1.0`, `0.0`, `0.0`.

- [x] **Step 2: Run focused evaluation tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_evaluate.py -q
```

Expected: FAIL because rollout results do not expose projection diagnostics.

- [x] **Step 3: Extend `PhysicsEpisodeResult` and rollout accounting**

Add defaulted fields after existing defaulted metrics:

```python
action_saturation_rate: float = 0.0
ik_projection_rate: float = 0.0
zero_pose_projection_rate: float = 0.0
mean_ik_projection_scale: float = 1.0
```

Add `ik_projection: bool = True` to `rollout_physics_policy`. After clipping the raw learned action, count saturation when any absolute pose component is at least `0.95`. If enabled, call `project_cartesian_action(env.controller, raw_action)` and execute `projection.action`; otherwise execute the raw action with scale `1.0`. Accumulate selected scales and populate all four rates using the number of executed steps.

- [x] **Step 4: Write failing aggregate-metric tests**

Construct two `PhysicsEpisodeResult` instances with distinct projection metrics and termination reasons. Require `_metrics` to report their means plus:

```python
"termination_reason_counts": {"physics_failure": 1, "timeout": 1}
```

- [x] **Step 5: Extend `_metrics` and run GREEN**

Add mean action-saturation, projection, zero-projection, and scale metrics. Use `collections.Counter` over `termination_reason`. Broaden the return annotation to `dict[str, object]`.

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_evaluate.py -q
```

Expected: all evaluation tests PASS.

---

### Task 3: Add condition filtering, safe output routing, and ID sanity gates

**Files:**

- Modify: `interaction_vla/physics_evaluate.py`
- Modify: `tests/interaction_vla/test_physics_evaluate.py`

- [x] **Step 1: Write failing resolver and parser tests**

Add `resolve_evaluation_conditions(cases, requested)` tests that preserve generated order, reject empty, duplicate, and unknown selections, and retain paired cases. Extend parser expectations for:

```bash
--conditions id_normal --output id_sanity_report.json --disable-ik-projection
```

Require `main()` to forward `conditions`, `output`, and `ik_projection=False`.

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_evaluate.py -q
```

Expected: FAIL because resolver and CLI options are missing.

- [x] **Step 3: Implement filtering and global projection selection**

Define the allowed condition tuple from the four existing condition names. Add parameters to `evaluate_from_config`:

```python
conditions: Iterable[str] | None = None,
output: str | Path | None = None,
ik_projection: bool = True,
```

Generate the complete deterministic cases first, then filter by the resolved selection. Pass the same `ik_projection` boolean into every Flat, Graph, and edge-shuffle rollout. Record `conditions`, `ik_projection_enabled`, and `ik_projection_scales` in `evaluation_scope`.

- [x] **Step 4: Write failing output-routing tests**

Patch rollout/checkpoint loading as existing tests do. For default output require `evaluation/report.json` and `evaluation/episodes.csv`. For a custom report `evaluation/id_sanity_report.json`, require the episode table at `evaluation/id_sanity_report_episodes.csv`; ensure the existing full report bytes remain unchanged.

- [x] **Step 5: Implement output routing**

When `output is None`, preserve current destinations. Otherwise require a `.json` suffix, create its parent, and derive the CSV sibling with `report_path.with_name(f"{report_path.stem}_episodes.csv")`. Write both only after all rollouts complete.

- [x] **Step 6: Write failing ID sanity tests**

Feed aggregates with an `id_normal` condition and require per-policy results:

```python
{
    "control_passed": physics_failure_rate <= 0.10,
    "manipulation_passed": stable_lift_rate >= 0.10,
    "passed": control_passed and manipulation_passed,
}
```

Require an empty sanity mapping when no ID cases were selected.

- [x] **Step 7: Implement sanity aggregation and run GREEN**

Build sanity entries from each policy's `id_normal` metrics and add them as `learned_policy_sanity` in the report. Do not abort or suppress other conditions.

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_evaluate.py -q
```

Expected: all evaluation tests PASS.

---

### Task 4: Apply the same safety layer to learned visualization

**Files:**

- Modify: `interaction_vla/physics_visualize.py`
- Modify: `tests/interaction_vla/test_physics_visualize.py`

- [x] **Step 1: Write failing learned-only projection tests**

Construct a lightweight `PhysicsVisualizationSession`. Patch `project_cartesian_action` to return scale `0.25`. Assert a Flat/Graph `advance()` executes the projected action, while expert and teleop sessions never call the projector.

- [x] **Step 2: Write a failing overlay test**

After a learned action is projected, require the overlay to contain `IK scale 0.25`. Expert/teleop overlays retain the existing `IK ok/limited` wording without a projection scale.

- [x] **Step 3: Run focused visualization tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_visualize.py -q
```

Expected: FAIL because learned visualization executes raw policy actions.

- [x] **Step 4: Implement learned-only projection and overlay state**

Add `last_ik_projection_scale: float = 1.0` to the session. In the learned branch of `advance`, project `_policy_action()` through `self.env.controller`, store the scale, and execute the selected action. Reset the scale on session reset. Append the scale only when `controller_name in {"flat", "graph"}`.

- [x] **Step 5: Run visualization tests GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_visualize.py -q
```

Expected: all visualization tests PASS.

---

### Task 5: Verify provenance boundary and run the ID sanity pilot

**Files/artifacts:**

- Verify: `interaction_vla/physics_provenance.py`
- Create at runtime: `outputs/interaction_graph_physics/recovery_pilot/evaluation/id_sanity_report.json`
- Create at runtime: `outputs/interaction_graph_physics/recovery_pilot/evaluation/id_sanity_report_episodes.csv`
- Preserve: `outputs/interaction_graph_physics/recovery_pilot/evaluation/report.json`

- [x] **Step 1: Run focused and full automated verification**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_physics_action_safety.py tests/interaction_vla/test_physics_evaluate.py tests/interaction_vla/test_physics_visualize.py -q
.venv/bin/python -m compileall -q interaction_vla tests/interaction_vla
.venv/bin/python -m pytest -q
```

Expected: all tests PASS and compilation exits zero.

- [x] **Step 2: Confirm controller provenance is unchanged**

Run:

```bash
.venv/bin/python - <<'PY'
from interaction_vla.physics_data import expected_gate_hashes
print(expected_gate_hashes("configs/physics_recovery_pilot_macos.yaml"))
PY
```

Expected controller hash:

```text
420431f92ca1269b7a4f2bbd318b1453002cc436f81771f17727a48d26fc5e52
```

- [x] **Step 3: Preserve and hash the completed full report**

Print the SHA-256 of `evaluation/report.json` before the sanity run and compare it afterward. The values must match.

- [x] **Step 4: Run the ID sanity evaluation**

Run:

```bash
.venv/bin/python -m interaction_vla.physics_evaluate \
  --config configs/physics_recovery_pilot_macos.yaml \
  --model-seeds 0 \
  --episodes-per-count 5 \
  --conditions id_normal \
  --output outputs/interaction_graph_physics/recovery_pilot/evaluation/id_sanity_report.json
```

Expected: tqdm completes 20 rollouts, no stale-provenance error occurs, and both custom report files are written without changing `evaluation/report.json`.

- [x] **Step 5: Interpret the sanity result conservatively**

Report Flat/Graph physics-failure, projection, stable-lift, placement, and success rates. If `control_passed=true` but `manipulation_passed=false`, state that IK reachability is isolated and recovery-data/closed-loop imitation is the next bottleneck; do not claim Graph superiority.
