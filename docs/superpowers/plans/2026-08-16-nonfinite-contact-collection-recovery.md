# Non-finite Contact Collection Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert rare non-finite MuJoCo contact forces into rejected physics-failure rollouts so LeRobotDataset collection continues without accepting corrupted Graph labels.

**Architecture:** `ContactParser` will raise a narrow typed exception when raw or derived force components are non-finite. `FrankaContactEnv` will catch only that exception in normal step and intervention paths and return the existing physics-failure result contract. The collector's existing non-success path will clear the current episode and continue its deterministic attempt schedule.

**Tech Stack:** Python 3.12, MuJoCo 3.3.4, NumPy, pytest, LeRobotDataset.

---

### Task 1: Detect non-finite MuJoCo contact forces explicitly

**Files:**
- Modify: `interaction_vla/contact_physics.py:230-310`
- Test: `tests/interaction_vla/test_contact_physics.py`

- [ ] **Step 1: Write the failing force-validation test**

Add a focused test for the desired typed API:

```python
from interaction_vla.contact_physics import (
    NonFiniteContactForceError,
    contact_force_components,
)


@pytest.mark.parametrize(
    "force",
    (
        np.asarray((np.nan, 0.0, 0.0, 0.0, 0.0, 0.0)),
        np.asarray((1.0, np.inf, 0.0, 0.0, 0.0, 0.0)),
    ),
)
def test_contact_force_components_reject_non_finite_mujoco_output(force) -> None:
    with pytest.raises(NonFiniteContactForceError, match="non-finite"):
        contact_force_components(force)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_contact_physics.py::test_contact_force_components_reject_non_finite_mujoco_output \
  -q
```

Expected: collection fails because `NonFiniteContactForceError` and
`contact_force_components` do not exist.

- [ ] **Step 3: Add the minimal typed validation**

In `interaction_vla/contact_physics.py`, add:

```python
class NonFiniteContactForceError(FloatingPointError):
    """MuJoCo returned a contact force that cannot label a valid graph."""


def contact_force_components(force: np.ndarray) -> tuple[float, float]:
    values = np.asarray(force, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise NonFiniteContactForceError("MuJoCo contact force is non-finite")
    normal = abs(float(values[0]))
    tangential = float(np.linalg.norm(values[1:3]))
    if not np.isfinite(normal) or not np.isfinite(tangential):
        raise NonFiniteContactForceError("MuJoCo contact force is non-finite")
    return normal, tangential
```

Replace the inline conversion after `mj_contactForce` with:

```python
normal, tangential = contact_force_components(force)
```

Before constructing each `InteractionSignal`, reject a non-finite aggregate:

```python
if not np.isfinite(values).all():
    raise NonFiniteContactForceError("aggregated contact force is non-finite")
```

- [ ] **Step 4: Run the contact-physics tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/interaction_vla/test_contact_physics.py -q
```

Expected: all contact-physics tests pass.

- [ ] **Step 5: Commit the typed parser change**

```bash
git add interaction_vla/contact_physics.py \
  tests/interaction_vla/test_contact_physics.py
git commit -m "fix: classify non-finite MuJoCo contact forces"
```

### Task 2: Translate the typed error into physics-failure transitions

**Files:**
- Modify: `interaction_vla/physics_env.py:240-340`
- Test: `tests/interaction_vla/test_physics_env.py`
- Test: `tests/interaction_vla/lerobot_bridge/test_collector.py`

- [ ] **Step 1: Write failing environment and collector tests**

Add to `tests/interaction_vla/test_physics_env.py`:

```python
from interaction_vla.contact_physics import NonFiniteContactForceError


def test_non_finite_contact_force_terminates_as_physics_failure(monkeypatch) -> None:
    env = make_env()
    env.reset(seed=11, object_count=2)

    def invalid_force(*args, **kwargs):
        raise NonFiniteContactForceError("MuJoCo contact force is non-finite")

    monkeypatch.setattr(env.contact_parser, "parse", invalid_force)
    result = env.step(
        np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    )

    assert result.done
    assert result.reason is TerminationReason.PHYSICS_FAILURE
    assert result.info["physics_failure"] == "non_finite_contact_force"


def test_non_finite_contact_force_stops_intervention(monkeypatch) -> None:
    env = make_env()
    before = env.reset(seed=11, object_count=2)

    def invalid_force(*args, **kwargs):
        raise NonFiniteContactForceError("MuJoCo contact force is non-finite")

    monkeypatch.setattr(env.contact_parser, "parse", invalid_force)
    result = env.advance_intervention(
        np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        substeps=1,
    )

    assert result.snapshot is before
    assert result.physics_failure == "non_finite_contact_force"
```

Add to `tests/interaction_vla/lerobot_bridge/test_collector.py`:

```python
def test_physics_failure_attempt_is_rejected_and_cleared() -> None:
    events: list[str] = []
    writer = FakePolicyWriter(events)
    result = collect_attempt(
        env=FakeEnv(events, terminal_reason="physics_failure"),
        expert=FakeExpert(events),
        capture=FakeCapture(events),
        policy_writer=writer,
        teacher=FakeTeacher(events),
        seed=11,
        object_count=2,
        task="Pick up the green target object and place it inside the receptacle.",
    )

    assert result.accepted is False
    assert result.reason == "physics_failure"
    assert writer.clear_count == 1
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_physics_env.py::test_non_finite_contact_force_terminates_as_physics_failure \
  tests/interaction_vla/lerobot_bridge/test_collector.py::test_physics_failure_attempt_is_rejected_and_cleared \
  -q
```

Expected: the environment test raises `NonFiniteContactForceError` instead of
returning a transition. The collector test documents existing rejection
behavior and may already pass.

- [ ] **Step 3: Catch only the typed contact-force error**

Import `NonFiniteContactForceError` in `interaction_vla/physics_env.py`.
In `step`, wrap only `self.contact_parser.parse(self.data)`:

```python
try:
    self.contact_diagnostics = self.contact_parser.parse(self.data)
except NonFiniteContactForceError:
    self.step_count += 1
    return self._failure_transition(
        "non_finite_contact_force",
        diagnostics=diagnostics,
        refresh_snapshot=False,
    )
```

Add a `refresh_snapshot: bool = True` keyword to `_failure_transition` and
guard its `_build_snapshot` call with that flag. The non-finite-contact branch
must use the last valid snapshot because rebuilding would parse the same
invalid contact force again. Other failure transitions keep the default.

In `advance_intervention`, return the intervention failure contract:

```python
try:
    self.contact_diagnostics = self.contact_parser.parse(self.data)
except NonFiniteContactForceError:
    return PhysicsInterventionResult(
        snapshot=self._last_snapshot,
        controller_diagnostics=diagnostics,
        physics_failure="non_finite_contact_force",
    )
```

Do not catch `ValueError`, `FloatingPointError`, or `Exception` generally.

- [ ] **Step 4: Run the focused environment and collector tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/interaction_vla/test_physics_env.py \
  tests/interaction_vla/lerobot_bridge/test_collector.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the environment behavior**

```bash
git add interaction_vla/physics_env.py \
  tests/interaction_vla/test_physics_env.py \
  tests/interaction_vla/lerobot_bridge/test_collector.py
git commit -m "fix: reject rollouts with invalid contact forces"
```

### Task 3: Document recovery and verify the project

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the server recovery boundary**

Add a short Linux troubleshooting note stating that FFmpeg moov-atom output is
normal, `non_finite_contact_force` attempts are rejected automatically after
this fix, and pre-fix `INCOMPLETE` datasets must be moved aside before a fresh
collection. Use this exact text:

```markdown
During LeRobotDataset collection, FFmpeg's `moving the moov atom` message is
normal MP4 finalization. A rollout with a non-finite MuJoCo contact force is
recorded as `physics_failure=non_finite_contact_force`; its uncommitted frames
are cleared and collection continues with the next deterministic seed. A
dataset left `INCOMPLETE` by an older revision cannot be resumed safely: move
that directory aside and collect a new dataset root.
```

- [ ] **Step 2: Run targeted LeRobot tests**

```bash
HF_HOME=/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-lerobot-pycache \
.venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/test_contact_physics.py \
  tests/interaction_vla/test_physics_env.py \
  tests/interaction_vla/lerobot_bridge/test_collector.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run the complete regression suite**

```bash
HF_HOME=/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/tmp/gripper-mujoco-lerobot-pycache \
.venv-lerobot/bin/python -m pytest tests/interaction_vla -q
```

Expected: zero failures; platform-dependent tests may remain skipped.

- [ ] **Step 4: Check the final diff**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and no generated outputs staged.

- [ ] **Step 5: Commit the recovery documentation**

```bash
git add README.md
git commit -m "docs: explain contact-force collection recovery"
```

### Task 4: Recover the server run after the fix is published

**Files:**
- No repository files change in this task.

- [ ] **Step 1: Update the server checkout**

After the local commits are pushed, run on the server:

```bash
git pull --ff-only origin main
```

- [ ] **Step 2: Preserve the pre-fix incomplete dataset**

Run only after confirming no collector process is active:

```bash
pgrep -af 'interaction_vla.lerobot_bridge collect' \
  || echo NO_COLLECT_PROCESS

backup_dir="outputs/lerobot/franka_lerobot_act_pilot_incomplete_$(date +%Y%m%d_%H%M%S)"
mv -- outputs/lerobot/franka_lerobot_act_pilot "$backup_dir"
echo "Backup: $backup_dir"
```

- [ ] **Step 3: Collect and validate a fresh dataset**

```bash
export MUJOCO_GL=egl

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge collect \
  --config configs/lerobot_act_recovery_linux_cuda.yaml

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge validate \
  --config configs/lerobot_act_recovery_linux_cuda.yaml
```

Expected: collection reaches 50 accepted episodes, removes `INCOMPLETE`, and
validation prints `"passed": true`.
