# LeRobot v3 Dual-View Dataset and ACT Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, transactional LeRobotDataset v3 pipeline from the Franka MuJoCo scripted expert, then prove one-batch ACT training, Hugging Face-compatible checkpoint reload, and finite closed-loop rollout on the current Mac.

**Architecture:** Add an isolated `interaction_vla.lerobot_bridge` package around the existing `FrankaContactEnv`, `PhysicsScriptedExpert`, rendering, expert gate, and IK safety code. Standard LeRobot rows contain only two RGB views, a 10D end-effector state, a 7D gripper-local action, and task metadata; privileged geometry, depth, segmentation, and TC-TIG labels live only in hashed episode sidecars. ACT uses the official LeRobot 0.6.1 dataset, policy, processor, and pretrained APIs from a separate `.venv-lerobot` environment.

**Tech Stack:** Python 3.12, NumPy, MuJoCo 3.x, PyTorch 2.10, torchvision 0.25, LeRobot 0.6.1, Hugging Face Datasets/Hub, FFmpeg, PyYAML, pytest.

---

## Scope and execution order

This is one sequential subproject: codecs and synchronized capture are required by
the collector; the collector is required by the validator and ACT checks; the ACT
checkpoint is required by rollout. Do not begin the 50-episode pilot until the
five-episode smoke report is complete.

Use these versioned upstream interfaces while implementing:

- `LeRobotDataset.create/add_frame/save_episode/finalize` from LeRobot `v0.6.1`;
- `ACTConfig`, `make_policy`, and `make_pre_post_processors` from LeRobot `v0.6.1`;
- `PreTrainedPolicy.save_pretrained/from_pretrained` from LeRobot `v0.6.1`.

Do not copy imports or dependency pins from the older community tutorial.

## File map

Create these focused modules:

- `interaction_vla/lerobot_bridge/config.py`: bridge-only YAML schema and validation.
- `interaction_vla/lerobot_bridge/codecs.py`: 10D state and local/world action conversion.
- `interaction_vla/lerobot_bridge/capture.py`: synchronized agent/wrist RGB-D and raw segmentation.
- `interaction_vla/lerobot_bridge/teacher_schema.py`: fixed TC-TIG slot and feature registry.
- `interaction_vla/lerobot_bridge/teacher.py`: distractor tracking, relation extraction, and future-derived goals.
- `interaction_vla/lerobot_bridge/sidecar.py`: atomic teacher NPZ and manifest records.
- `interaction_vla/lerobot_bridge/provenance.py`: runtime, source, file, and dataset fingerprints.
- `interaction_vla/lerobot_bridge/dataset_writer.py`: the only LeRobotDataset write adapter.
- `interaction_vla/lerobot_bridge/collector.py`: successful scripted-expert episode orchestration.
- `interaction_vla/lerobot_bridge/validator.py`: schema, alignment, hash, and deterministic replay checks.
- `interaction_vla/lerobot_bridge/act_smoke.py`: one-batch and bounded ACT training/checkpoint logic.
- `interaction_vla/lerobot_bridge/rollout.py`: ACT observation processing and safe MuJoCo execution.
- `interaction_vla/lerobot_bridge/cli.py` and `__main__.py`: public commands only.

Modify only these existing files:

- `.gitignore`: ignore `.venv-lerobot` and `outputs/lerobot/`.
- `README.md`: document setup and the smoke workflow.

Do not edit a module listed by `physics_control_module_names()`: the already
validated `physics_pilot_macos.yaml` expert gate must remain current.

Tests mirror module boundaries under `tests/interaction_vla/lerobot_bridge/`. Tests
that import LeRobot use `pytest.importorskip("lerobot")`, so the original `.venv`
still runs every non-LeRobot test.

### Task 1: Dependency isolation and bridge configuration

**Files:**

- Create: `requirements-lerobot-macos.txt`
- Create: `configs/lerobot_act_smoke_macos.yaml`
- Create: `configs/lerobot_act_pilot_macos.yaml`
- Create: `interaction_vla/lerobot_bridge/__init__.py`
- Create: `interaction_vla/lerobot_bridge/config.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_config.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing configuration tests**

```python
from dataclasses import replace

import pytest

from interaction_vla.lerobot_bridge.config import load_bridge_config


def test_smoke_config_locks_the_model_visible_contract() -> None:
    cfg = load_bridge_config("configs/lerobot_act_smoke_macos.yaml")

    assert cfg.dataset.episodes == 5
    assert cfg.dataset.object_counts == (2, 3)
    assert cfg.dataset.fps == 20
    assert cfg.dataset.image_size == (256, 256)
    assert cfg.dataset.state_dim == 10
    assert cfg.dataset.action_dim == 7
    assert cfg.act.chunk_size == cfg.act.n_action_steps == 8
    assert cfg.act.steps == 500
    assert cfg.act.epochs is None
    assert cfg.source.backend == "franka_contact"


def test_pilot_config_uses_epochs_and_requires_the_smoke_report() -> None:
    cfg = load_bridge_config("configs/lerobot_act_pilot_macos.yaml")

    assert cfg.dataset.episodes == 50
    assert cfg.act.steps is None
    assert cfg.act.epochs == 5
    assert cfg.act.maximum_epochs == 10
    assert cfg.required_smoke_report is not None


def test_act_schedule_requires_exactly_one_stop_condition() -> None:
    cfg = load_bridge_config("configs/lerobot_act_smoke_macos.yaml")

    with pytest.raises(ValueError, match="exactly one"):
        replace(cfg.act, steps=500, epochs=5)
    with pytest.raises(ValueError, match="exactly one"):
        replace(cfg.act, steps=None, epochs=None)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_config.py -q
```

Expected: collection fails with `ModuleNotFoundError: interaction_vla.lerobot_bridge`.

- [ ] **Step 3: Add locked requirements and ignore rules**

`requirements-lerobot-macos.txt` must contain exactly:

```text
-r requirements-macos.txt
torch>=2.10,<2.11
torchvision>=0.25,<0.26
lerobot[dataset,training]==0.6.1
```

Append these two entries to `.gitignore`:

```text
.venv-lerobot/
outputs/lerobot/
```

- [ ] **Step 4: Add the exact bridge configuration types**

Implement frozen dataclasses with these public fields and validations:

```python
@dataclass(frozen=True)
class DatasetBridgeConfig:
    repo_id: str
    root: Path
    episodes: int
    object_counts: tuple[int, ...]
    task: str
    fps: int = 20
    image_size: tuple[int, int] = (256, 256)
    state_dim: int = 10
    action_dim: int = 7
    max_attempt_multiplier: int = 10

    def __post_init__(self) -> None:
        if not self.repo_id or not self.task.strip():
            raise ValueError("dataset repo_id and task must be non-empty")
        if self.episodes < 1 or self.max_attempt_multiplier < 1:
            raise ValueError("dataset episode and attempt counts must be positive")
        if not self.object_counts or min(self.object_counts) < 2:
            raise ValueError("dataset object_counts must contain values >= 2")
        if self.fps != 20 or self.image_size != (256, 256):
            raise ValueError("the first bridge requires 20 Hz and 256x256 images")
        if self.state_dim != 10 or self.action_dim != 7:
            raise ValueError("the first bridge requires state_dim=10 and action_dim=7")


@dataclass(frozen=True)
class TeacherConfig:
    distractor_count: int = 2
    replacement_margin: float = 0.10
    replacement_frames: int = 3
    dropout_frames: int = 3
    safety_margin_m: float = 0.01
    goal_horizon: int = 8
    goal_improvement_margin: float = 0.05


@dataclass(frozen=True)
class ACTBridgeConfig:
    output_dir: Path
    device: str = "auto"
    chunk_size: int = 8
    n_action_steps: int = 8
    batch_size: int = 2
    num_workers: int = 0
    steps: int | None = 500
    epochs: int | None = None
    maximum_epochs: int = 10
    learning_rate: float = 1e-5
    seed: int = 0
    dim_model: int = 256
    dim_feedforward: int = 1024
    encoder_layers: int = 2
    vae_encoder_layers: int = 2

    def __post_init__(self) -> None:
        if (self.steps is None) == (self.epochs is None):
            raise ValueError("ACT requires exactly one of steps or epochs")
        if self.chunk_size != 8 or self.n_action_steps != 8:
            raise ValueError("the first ACT bridge requires an 8-step chunk")
        if self.batch_size not in {1, 2} or self.num_workers != 0:
            raise ValueError("macOS ACT requires batch size 1/2 and num_workers=0")
        if self.device not in {"auto", "cpu", "mps"}:
            raise ValueError("ACT device must be auto, cpu, or mps")


@dataclass(frozen=True)
class BridgeConfig:
    config_path: Path
    source_config_path: Path
    expert_gate: Path
    dataset: DatasetBridgeConfig
    teacher: TeacherConfig
    act: ACTBridgeConfig
    seed: int
    required_smoke_report: Path | None
    source: ExperimentConfig
```

`load_bridge_config(path)` must parse YAML without mutating the loaded dictionary,
load `source_config_path` through the existing `load_config`, and reject a source
whose backend is not `franka_contact`, whose `policy_hz` differs from the dataset
FPS, or whose maximum object count cannot cover `dataset.object_counts`.

- [ ] **Step 5: Add the two concrete YAML files**

The smoke YAML uses:

```yaml
seed: 42
source_config: configs/physics_pilot_macos.yaml
expert_gate: outputs/interaction_graph_physics/pilot/expert_gate.json
required_smoke_report: null
dataset:
  repo_id: local/franka_lerobot_act_smoke
  root: outputs/lerobot/franka_lerobot_act_smoke
  episodes: 5
  object_counts: [2, 3]
  task: Pick up the green target object and place it inside the receptacle.
  fps: 20
  image_size: [256, 256]
  state_dim: 10
  action_dim: 7
  max_attempt_multiplier: 10
teacher:
  distractor_count: 2
  replacement_margin: 0.10
  replacement_frames: 3
  dropout_frames: 3
  safety_margin_m: 0.01
  goal_horizon: 8
  goal_improvement_margin: 0.05
act:
  output_dir: outputs/lerobot/act_smoke
  device: auto
  chunk_size: 8
  n_action_steps: 8
  batch_size: 2
  num_workers: 0
  steps: 500
  epochs: null
  maximum_epochs: 10
  learning_rate: 0.00001
  seed: 0
  dim_model: 256
  dim_feedforward: 1024
  encoder_layers: 2
  vae_encoder_layers: 2
```

The pilot YAML contains the complete independent configuration:

```yaml
seed: 42
source_config: configs/physics_pilot_macos.yaml
expert_gate: outputs/interaction_graph_physics/pilot/expert_gate.json
required_smoke_report: outputs/lerobot/act_smoke/smoke_report.json
dataset:
  repo_id: local/franka_lerobot_act_pilot
  root: outputs/lerobot/franka_lerobot_act_pilot
  episodes: 50
  object_counts: [2, 3]
  task: Pick up the green target object and place it inside the receptacle.
  fps: 20
  image_size: [256, 256]
  state_dim: 10
  action_dim: 7
  max_attempt_multiplier: 10
teacher:
  distractor_count: 2
  replacement_margin: 0.10
  replacement_frames: 3
  dropout_frames: 3
  safety_margin_m: 0.01
  goal_horizon: 8
  goal_improvement_margin: 0.05
act:
  output_dir: outputs/lerobot/act_pilot
  device: auto
  chunk_size: 8
  n_action_steps: 8
  batch_size: 2
  num_workers: 0
  steps: null
  epochs: 5
  maximum_epochs: 10
  learning_rate: 0.00001
  seed: 0
  dim_model: 256
  dim_feedforward: 1024
  encoder_layers: 2
  vae_encoder_layers: 2
```

- [ ] **Step 6: Run tests and create the isolated environment**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_config.py -q
python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install --upgrade pip
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-macos.txt
.venv-lerobot/bin/python -c "import importlib.metadata, torch; assert importlib.metadata.version('lerobot') == '0.6.1'; print(torch.__version__)"
.venv-lerobot/bin/python -m pip freeze --all > requirements-lerobot-macos.lock.txt
```

Expected: configuration tests pass; LeRobot prints `0.6.1`; PyTorch prints a
`2.10.x` version; the lock file contains the fully resolved environment.

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements-lerobot-macos.txt requirements-lerobot-macos.lock.txt configs/lerobot_act_smoke_macos.yaml configs/lerobot_act_pilot_macos.yaml interaction_vla/lerobot_bridge/__init__.py interaction_vla/lerobot_bridge/config.py tests/interaction_vla/lerobot_bridge/test_config.py
git commit -m "build: isolate LeRobot bridge dependencies"
```

### Task 2: Snapshot/proprioception coordinate codecs

**Files:**

- Create: `interaction_vla/lerobot_bridge/codecs.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_codecs.py`

- [ ] **Step 1: Write codec and environment contract tests**

```python
import numpy as np
import pytest

from interaction_vla.config import PhysicsConfig
from interaction_vla.lerobot_bridge.codecs import (
    EndEffectorStateCodec,
    LocalCartesianActionCodec,
)
from interaction_vla.physics_env import FrankaContactEnv


def test_rotation_6d_round_trip_is_right_handed() -> None:
    rotation = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    encoded = EndEffectorStateCodec.encode_rotation(rotation)
    decoded = EndEffectorStateCodec.decode_rotation(encoded)

    np.testing.assert_array_equal(encoded, (0.0, 1.0, 0.0, -1.0, 0.0, 0.0))
    np.testing.assert_allclose(decoded, rotation, atol=1e-7)
    np.testing.assert_allclose(decoded.T @ decoded, np.eye(3), atol=1e-7)
    assert np.linalg.det(decoded) == pytest.approx(1.0)


def test_local_action_round_trip_preserves_controller_command() -> None:
    rotation = EndEffectorStateCodec.decode_rotation(
        np.asarray((0.0, 1.0, 0.0, -1.0, 0.0, 0.0), dtype=np.float32)
    )
    controller_action = np.asarray((0.2, -0.4, 0.1, 0.3, -0.2, 0.5, 0.0), dtype=np.float32)

    stored = LocalCartesianActionCodec.encode(controller_action, rotation)
    restored = LocalCartesianActionCodec.decode(stored, rotation)

    np.testing.assert_allclose(restored, controller_action, atol=1e-6)
    assert stored[6] == 0.0


def test_public_snapshot_and_proprioception_encode_finite_10d_state() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    snapshot = env.reset(seed=11, object_count=2)

    state = EndEffectorStateCodec.encode_snapshot(snapshot, env.proprioception())

    assert state.shape == (10,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()
    assert 0.0 <= state[-1] <= 1.0
```

- [ ] **Step 2: Run the tests and verify missing methods fail**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_codecs.py -q
```

Expected: failure mentions missing `lerobot_bridge.codecs`.

- [ ] **Step 3: Implement the public observation adapter**

Use only `SceneSnapshot.gripper` and the public 23D `env.proprioception()` output.
Document the existing proprioception indices once and validate them:

```python
FINGER_POSITION_SLICE = slice(13, 15)
FINGER_POSITION_LOW = 0.0
FINGER_POSITION_HIGH = 0.04


@classmethod
def encode_snapshot(cls, snapshot: SceneSnapshot, proprioception: np.ndarray) -> np.ndarray:
    proprio = _finite(proprioception, (23,), "proprioception")
    fingers = proprio[FINGER_POSITION_SLICE]
    aperture = float(
        np.clip(
            np.mean((fingers - FINGER_POSITION_LOW) / (FINGER_POSITION_HIGH - FINGER_POSITION_LOW)),
            0.0,
            1.0,
        )
    )
    rotation = cls.quaternion_to_matrix(snapshot.gripper.orientation)
    return cls.encode(snapshot.gripper.position, rotation, aperture)
```

At collector startup, assert both named finger joint ranges in `env.model` equal
`[0.0,0.04]` within `1e-9`; fail before recording if the scene contract changes.

- [ ] **Step 4: Implement the exact codecs**

`EndEffectorStateCodec.encode_rotation` uses
`rotation[:, :2].T.reshape(6)`. `decode_rotation` normalizes the first column,
removes its projection from the second, normalizes the result, and obtains the
third column with `np.cross`. Reject non-finite, wrong-shaped, or near-collinear
inputs. `encode` concatenates position, rotation-6D, and aperture into
`float32[10]` and requires aperture in `[0,1]`.

`LocalCartesianActionCodec` must implement exactly:

```python
@staticmethod
def encode(action_world: np.ndarray, gripper_rotation: np.ndarray) -> np.ndarray:
    action = _finite(action_world, (7,), "action_world").copy()
    rotation = _rotation(gripper_rotation)
    action[:3] = rotation.T @ action[:3]
    return action.astype(np.float32)

@staticmethod
def decode(action_local: np.ndarray, gripper_rotation: np.ndarray) -> np.ndarray:
    action = _finite(action_local, (7,), "action_local").copy()
    rotation = _rotation(gripper_rotation)
    action[:3] = rotation @ action[:3]
    action[:6] = np.clip(action[:6], -1.0, 1.0)
    action[6] = float(action[6] >= 0.5)
    return action.astype(np.float32)
```

- [ ] **Step 5: Run focused and existing environment tests**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_codecs.py tests/interaction_vla/test_physics_env.py tests/interaction_vla/test_physics_expert.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add interaction_vla/lerobot_bridge/codecs.py tests/interaction_vla/lerobot_bridge/test_codecs.py
git commit -m "feat: add LeRobot Cartesian codecs"
```

### Task 3: Synchronized dual-view RGB-D and segmentation capture

**Files:**

- Create: `interaction_vla/lerobot_bridge/capture.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_capture.py`

- [ ] **Step 1: Write validation and non-advancement tests**

```python
import os

import numpy as np
import pytest

from interaction_vla.config import PhysicsConfig
from interaction_vla.lerobot_bridge.capture import CameraFrame, DualViewCapture
from interaction_vla.physics_env import FrankaContactEnv


def test_camera_frame_requires_rgb_metric_depth_and_raw_segmentation() -> None:
    frame = CameraFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        segmentation=np.full((8, 8, 2), -1, dtype=np.int32),
    )

    assert frame.segmentation.shape == (8, 8, 2)
    with pytest.raises(ValueError, match="segmentation"):
        CameraFrame(frame.rgb, frame.depth, np.zeros((8, 8), dtype=np.int32))


@pytest.mark.skipif(
    bool(os.environ.get("CODEX_SANDBOX")),
    reason="macOS CoreGraphics is unavailable inside the Codex sandbox",
)
def test_capture_has_two_views_and_does_not_advance_mujoco() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    env.reset(seed=11, object_count=2)
    qpos_before = env.data.qpos.copy()
    qvel_before = env.data.qvel.copy()
    ctrl_before = env.data.ctrl.copy()
    time_before = float(env.data.time)
    capture = DualViewCapture(env.model, width=64, height=64)
    try:
        frame = capture.capture(env, include_teacher=True)
    finally:
        capture.close()

    assert tuple(frame.views) == ("agent", "wrist")
    assert frame.policy_step == 0
    assert frame.timestamp == pytest.approx(0.0)
    for view in frame.views.values():
        assert view.rgb.shape == (64, 64, 3)
        assert view.depth.shape == (64, 64)
        assert view.segmentation.shape == (64, 64, 2)
        assert np.all(view.depth > 0.0)
    np.testing.assert_array_equal(env.data.qpos, qpos_before)
    np.testing.assert_array_equal(env.data.qvel, qvel_before)
    np.testing.assert_array_equal(env.data.ctrl, ctrl_before)
    assert env.data.time == time_before
```

- [ ] **Step 2: Run the test and verify the capture module is missing**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_capture.py -q
```

Expected: import failure for `lerobot_bridge.capture`.

- [ ] **Step 3: Implement immutable capture records**

`CameraFrame` validates `uint8[H,W,3]`, positive finite `float32[H,W]`, and
`int32[H,W,2]`. `DualViewFrame` validates ordered `agent,wrist` views, nonnegative
step/timestamp, and nonempty state hash. When `include_teacher=False`, use a
separate `PolicyCameraFrame` containing RGB only so rollout cannot accidentally
retain privileged buffers.

- [ ] **Step 4: Implement renderer mode switching**

For each of `agentview` and `wristview`, run this exact order on one renderer:

```python
self._renderer.disable_depth_rendering()
self._renderer.disable_segmentation_rendering()
self._renderer.update_scene(env.data, camera=camera_name)
rgb = np.asarray(self._renderer.render(), dtype=np.uint8).copy()

self._renderer.enable_depth_rendering()
self._renderer.update_scene(env.data, camera=camera_name)
depth = np.asarray(self._renderer.render(), dtype=np.float32).copy()
self._renderer.disable_depth_rendering()

self._renderer.enable_segmentation_rendering()
self._renderer.update_scene(env.data, camera=camera_name)
segmentation = np.asarray(self._renderer.render(), dtype=np.int32).copy()
self._renderer.disable_segmentation_rendering()
```

MuJoCo 3.3.4 returns segmentation channel 0 as object ID and channel 1 as
`mjtObj` object type. Preserve this raw pair until the teacher extractor maps it to
canonical entity IDs. Build the timestamp as `env.step_count / env.policy_hz` and
the hash with the existing `MultiViewRecorder.state_hash(env)`.

- [ ] **Step 5: Add calibrated camera metadata**

Implement `camera_calibration(env) -> dict[str, object]`. For each camera, compute
`fy = 0.5 * height / tan(0.5 * radians(fovy))`, `fx = fy`,
`cx = (width - 1) / 2`, and `cy = (height - 1) / 2`. Store the MuJoCo
camera-to-base position and rotation from `data.cam_xpos` and `data.cam_xmat`.
Store metric clip distances as `znear * model.stat.extent` and
`zfar * model.stat.extent`.
For the wrist view also store the constant XML camera transform relative to
`wrist_camera_rig`; verify at two EE poses that recomposition matches the runtime
camera pose within `1e-6`.

- [ ] **Step 6: Run capture and legacy recording tests**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_capture.py tests/interaction_vla/test_physics_recording.py -q
```

Expected: synthetic tests pass; graphical tests either pass in the user session or
skip only when `CODEX_SANDBOX` is set.

- [ ] **Step 7: Commit**

```bash
git add interaction_vla/lerobot_bridge/capture.py tests/interaction_vla/lerobot_bridge/test_capture.py
git commit -m "feat: capture synchronized LeRobot views"
```

### Task 4: Fixed TC-TIG teacher schema and distractor tracking

**Files:**

- Create: `interaction_vla/lerobot_bridge/teacher_schema.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_teacher_schema.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_distractor_tracker.py`

- [ ] **Step 1: Write schema and forbidden-key tests**

```python
import numpy as np

from interaction_vla.lerobot_bridge.teacher_schema import (
    ENTITY_SLOTS,
    FORBIDDEN_FIELD_FRAGMENTS,
    RELATION_FEATURE_DIM,
    RELATION_SLOTS,
    TeacherFrame,
    teacher_schema_payload,
)


def test_teacher_schema_has_six_entities_and_eight_sparse_relations() -> None:
    assert ENTITY_SLOTS == (
        "gripper",
        "target",
        "receptacle",
        "support",
        "distractor_0",
        "distractor_1",
    )
    assert RELATION_SLOTS == (
        "gripper_to_target",
        "target_to_receptacle",
        "target_to_support",
        "distractor_0_to_gripper",
        "distractor_0_to_target",
        "distractor_1_to_gripper",
        "distractor_1_to_target",
        "gripper_to_receptacle",
    )
    assert RELATION_FEATURE_DIM == 24


def test_teacher_payload_contains_no_privileged_forbidden_name() -> None:
    flattened = str(teacher_schema_payload()).lower()
    assert all(fragment not in flattened for fragment in FORBIDDEN_FIELD_FRAGMENTS)


def test_teacher_frame_validates_fixed_shapes() -> None:
    frame = TeacherFrame.zeros(frame_index=0, timestamp=0.0, state_hash="abc")

    assert frame.entity_pose.shape == (6, 9)
    assert frame.entity_visibility.shape == (6, 2)
    assert frame.relation_values.shape == (8, 24)
    assert frame.entity_mask.dtype == np.bool_
    assert frame.relation_mask.dtype == np.bool_
```

- [ ] **Step 2: Write deterministic hysteresis and dropout tests**

```python
from interaction_vla.lerobot_bridge.teacher_schema import DistractorTracker


def test_challenger_needs_point_one_margin_for_three_frames() -> None:
    tracker = DistractorTracker(count=2, replacement_margin=0.10, replacement_frames=3, dropout_frames=3)
    assert tracker.update({"a": 0.8, "b": 0.6, "c": 0.0}) == ("a", "b")
    assert tracker.update({"a": 0.8, "b": 0.6, "c": 0.71}) == ("a", "b")
    assert tracker.update({"a": 0.8, "b": 0.6, "c": 0.71}) == ("a", "b")
    assert tracker.update({"a": 0.8, "b": 0.6, "c": 0.71}) == ("a", "c")


def test_retained_track_survives_three_missing_frames_then_expires() -> None:
    tracker = DistractorTracker(count=2, replacement_margin=0.10, replacement_frames=3, dropout_frames=3)
    tracker.update({"a": 0.8, "b": 0.6})

    assert tracker.update({"a": 0.8}) == ("a", "b")
    assert tracker.update({"a": 0.8}) == ("a", "b")
    assert tracker.update({"a": 0.8}) == ("a", "b")
    assert tracker.update({"a": 0.8}) == ("a", None)


def test_ties_are_resolved_by_stable_track_name() -> None:
    tracker = DistractorTracker(count=2, replacement_margin=0.10, replacement_frames=3, dropout_frames=3)
    assert tracker.update({"c": 0.5, "a": 0.5, "b": 0.5}) == ("a", "b")
```

- [ ] **Step 3: Run tests and verify missing schema failures**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_teacher_schema.py tests/interaction_vla/lerobot_bridge/test_distractor_tracker.py -q
```

Expected: import failures for `teacher_schema`.

- [ ] **Step 4: Implement the fixed schema registry**

Use schema version `tc_tig_teacher_v1`, `ENTITY_POSE_DIM=9`,
`ENTITY_SIZE_DIM=3`, and `RELATION_FEATURE_DIM=24`. Define entity role IDs and
relation type IDs as zero-based integer vocabularies. Define the relation feature
layout once:

```python
RELATIVE_POSITION = slice(0, 3)
RELATIVE_ROTATION = slice(3, 6)
RELATIVE_LINEAR_VELOCITY = slice(6, 9)
RELATIVE_ANGULAR_VELOCITY = slice(9, 12)
SIGNED_MARGIN_0 = 12
SIGNED_MARGIN_1 = 13
SIGNED_MARGIN_2 = 14
SIGNED_MARGIN_3 = 15
PROBABILITY_0 = 16
PROBABILITY_1 = 17
RISK_0 = 18
RISK_1 = 19
ERROR_0 = 20
ERROR_1 = 21
VISIBILITY = 22
CONFIDENCE = 23
```

`TeacherFrame` contains only these fields: `frame_index`, `timestamp`,
`state_hash`, `entity_pose`, `entity_size`, `entity_role`, `entity_visibility`,
`entity_mask`, `relation_values`, `relation_type`, `relation_mask`,
`instance_agent`, `instance_wrist`, `depth_agent`, `depth_wrist`, and
`camera_intrinsics[2,3,3]` and `camera_extrinsics_base[2,4,4]`. Relation goals
are appended only after a successful episode. Do not store Python dictionaries or
object-dtype arrays in NPZ; every sidecar must load with `allow_pickle=False`.

Set:

```python
FORBIDDEN_FIELD_FRAGMENTS = (
    "contact_force",
    "normal_force",
    "tangential_force",
    "stable_grasp",
    "held_object",
    "expert_phase",
    "success",
    "termination_reason",
)
```

- [ ] **Step 5: Implement stateful distractor selection**

`DistractorTracker.update(scores)` sorts initial candidates by `(-score, name)`.
A challenger increments a per-pair counter only while it exceeds the lowest
retained score by at least `replacement_margin`; any violation resets the counter.
Replace after exactly `replacement_frames` consecutive wins. Missing tracks retain
their slot for `dropout_frames`, with confidence multipliers `2/3`, `1/3`, and
`0`; remove on the following update. Expose slot name, age, missing count, and
confidence without simulator IDs.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_teacher_schema.py tests/interaction_vla/lerobot_bridge/test_distractor_tracker.py -q
```

Expected: all pass.

```bash
git add interaction_vla/lerobot_bridge/teacher_schema.py tests/interaction_vla/lerobot_bridge/test_teacher_schema.py tests/interaction_vla/lerobot_bridge/test_distractor_tracker.py
git commit -m "feat: define TC-TIG teacher schema"
```

### Task 5: Visual-observable relation extraction and relation-goal labels

**Files:**

- Create: `interaction_vla/lerobot_bridge/teacher.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_teacher.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_relation_goals.py`

- [ ] **Step 1: Write invariance, slot, and privileged-input tests**

```python
import inspect

import numpy as np

from interaction_vla.config import PhysicsConfig
from dataclasses import replace

from interaction_vla.lerobot_bridge.teacher import (
    TCTIGTeacherExtractor,
    transform_snapshot_passive,
)
from interaction_vla.physics_env import FrankaContactEnv


def test_extractor_uses_task_roles_and_sparse_relation_slots() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    snapshot = env.reset(seed=11, object_count=3)
    extractor = TCTIGTeacherExtractor.from_defaults()

    frame = extractor.extract_geometry(snapshot, frame_index=0, timestamp=0.0, state_hash="state")

    assert frame.entity_mask.tolist() == [True, True, True, True, True, True]
    assert frame.relation_mask.tolist() == [True] * 8
    assert np.isfinite(frame.relation_values).all()


def test_extractor_source_never_reads_privileged_interaction_state() -> None:
    source = inspect.getsource(TCTIGTeacherExtractor).lower()
    forbidden = (
        "contact_diagnostics",
        "grasp_state",
        "held_object",
        "stable_grasp",
        "normal_force",
        "expert_phase",
        "termination_reason",
    )
    assert all(fragment not in source for fragment in forbidden)


def test_local_relations_are_invariant_to_passive_translation_and_yaw() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    snapshot = env.reset(seed=11, object_count=3)
    transformed = transform_snapshot_passive(
        snapshot,
        translation=np.asarray((1.2, -0.7, 0.3)),
        yaw_radians=0.8,
    )

    first = TCTIGTeacherExtractor.from_defaults().extract_geometry(
        snapshot, frame_index=0, timestamp=0.0, state_hash="first"
    )
    second = TCTIGTeacherExtractor.from_defaults().extract_geometry(
        transformed, frame_index=0, timestamp=0.0, state_hash="second"
    )

    np.testing.assert_allclose(
        first.relation_values[first.relation_mask],
        second.relation_values[second.relation_mask],
        atol=1e-5,
    )


def test_distractor_slots_do_not_depend_on_input_object_order() -> None:
    env = FrankaContactEnv(physics=PhysicsConfig(settle_steps=5))
    snapshot = env.reset(seed=11, object_count=3)
    reversed_snapshot = replace(snapshot, objects=tuple(reversed(snapshot.objects)))

    first = TCTIGTeacherExtractor.from_defaults().extract_geometry(
        snapshot, frame_index=0, timestamp=0.0, state_hash="first"
    )
    second = TCTIGTeacherExtractor.from_defaults().extract_geometry(
        reversed_snapshot, frame_index=0, timestamp=0.0, state_hash="second"
    )

    np.testing.assert_allclose(first.entity_pose, second.entity_pose, atol=1e-6)
    np.testing.assert_allclose(first.relation_values, second.relation_values, atol=1e-6)
```

- [ ] **Step 2: Write future-label tests**

```python
import numpy as np

from interaction_vla.lerobot_bridge.teacher import label_relation_goals


def test_goal_label_selects_largest_confident_future_improvement() -> None:
    errors = np.full((9, 8, 2), 0.8, dtype=np.float32)
    confidence = np.ones((9, 8), dtype=np.float32)
    errors[:, 0, 0] = np.linspace(0.8, 0.1, 9)
    errors[:, 1, 0] = np.linspace(0.8, 0.6, 9)

    labels = label_relation_goals(errors, confidence, horizon=8, minimum_improvement=0.05)

    assert labels[0, 0] == 0
    assert labels[0, 1] == 4
    assert labels[0, 2] == 0
    assert labels[0, 3] < 0.0
    assert 0.0 <= labels[0, 4] <= 1.0


def test_no_improvement_preserves_previous_relation() -> None:
    errors = np.ones((3, 8, 2), dtype=np.float32)
    confidence = np.ones((3, 8), dtype=np.float32)

    labels = label_relation_goals(errors, confidence, horizon=2, minimum_improvement=0.05)

    assert labels[:, 1].tolist() == [3.0, 3.0, 3.0]
```

Operator IDs are fixed as `establish=0, break=1, increase=2, preserve=3,
decrease=4`; predicate IDs are `proximity=0, alignment=1, enclosure=2,
co_motion=3, containment=4, support=5, clearance=6`.

- [ ] **Step 3: Run tests and verify the extractor is missing**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_teacher.py tests/interaction_vla/lerobot_bridge/test_relation_goals.py -q
```

Expected: import failures for `teacher`.

- [ ] **Step 4: Implement entity slots and instance masks**

The extractor receives only `SceneSnapshot`, raw segmentation pairs, camera
calibration, and the teacher configuration. It selects role-bound entities and
maps visible geoms to canonical instance values `0=background`, `1=gripper`,
`2=target`, `3=receptacle`, `4=support`, `5=distractor_0`, and
`6=distractor_1`. A gripper instance includes the `hand`, `left_finger`, and
`right_finger` body subtrees; receptacle includes all `receptacle_*` geoms.
Visibility is the fraction of image pixels assigned to an active entity.

Entity pose is base-frame position plus the column-major rotation-6D code and is
strictly teacher-only. The policy graph later consumes only local relations.

- [ ] **Step 5: Implement the four risk terms and Top-K score**

For every non-target object compute values clipped to `[0,1]`:

```python
wrong_grasp = np.exp(-distance(object_xy, target_xy) / 0.08)
closing_region = np.exp(-distance_in_gripper_frame / 0.08)
approach_risk = np.exp(-point_to_segment_distance(object_xyz, gripper_xyz, target_xyz) / 0.06)
transport_risk = np.exp(-point_to_segment_distance(object_xyz, target_xyz, receptacle_xyz) / 0.06)
risk = 0.25 * (wrong_grasp + closing_region + approach_risk + transport_risk)
```

Distances subtract object/gripper bounding radii and the configured `0.01 m`
safety margin before the exponential. Use track name only for deterministic tie
breaking, never as a feature.

- [ ] **Step 6: Implement typed local relation values**

Fill indices `0:12` with local relative pose and motion. Use the source frame
specified by the design: gripper for manipulation, receptacle for placement and
clearance, gravity-aligned support frame for support, and the relevant approach or
transport segment frame for risk. Fill indices `12:24` as follows and leave
unused typed positions at zero:

- manipulation: signed surface gap, two finger clearances, grasp-axis alignment,
  closing-region probability, co-motion probability, proximity/alignment errors,
  visibility, confidence;
- placement: x/y containment margins, bottom gap, entrance clearance, soft
  containment/support, containment/orientation errors, visibility, confidence;
- support: bottom gap, x/y projected overlap, vertical relative velocity, soft
  support, support error, visibility, confidence;
- risk: closing-region distance, swept clearance, clipped TTC, target distance,
  wrong-grasp/collision risks, the same two normalized errors, visibility,
  confidence;
- clearance: entrance and wall clearances, retreat-height residual, swept
  collision margin, clearance/collision probabilities, clearance error,
  visibility, confidence.

Use smooth probabilities `sigmoid(margin / 0.01)` and normalized errors clipped to
`[0,1]`. Never call `snapshot.interactions`, inspect contacts, or read expert state.

- [ ] **Step 7: Implement future-derived relation goals**

Construct a `[T,8,2]` error tensor containing each relation's two registered error
channels. At frame `t`, compute
`error[t] - min(error[t+1:min(T,t+H+1)])`, multiply by confidence, and choose the
largest valid improvement with deterministic relation/predicate tie breaking.
Require improvement at least `0.05`; otherwise use the preceding active relation
and operator `preserve`. Store `float32[T,5]` as active relation ID, operator ID,
predicate ID, signed best-step residual, and confidence. Labels may use future
relation arrays only.

The task template supplies this relation-derived candidate DAG, not a phase label:

```python
candidate_valid = {
    "establish_proximity": True,
    "establish_alignment": True,
    "establish_enclosure": proximity_error < 0.5,
    "establish_co_motion": enclosure_error < 0.5,
    "establish_containment": co_motion_error < 0.5,
    "establish_support": containment_error < 0.5,
    "break_co_motion": containment_error < 0.25 and support_error < 0.25,
    "increase_clearance": containment_error < 0.25 and co_motion_probability < 0.25,
}
```

For `break_co_motion`, the minimized error is co-motion probability; for
`increase_clearance`, it is the positive retreat-height residual. Thus the same
future-minimum rule covers relation creation and removal without reading expert
phase, contact, success, or termination.

- [ ] **Step 8: Run tests and commit**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_teacher.py tests/interaction_vla/lerobot_bridge/test_relation_goals.py -q
```

Expected: all pass, including invariance and source audits.

```bash
git add interaction_vla/lerobot_bridge/teacher.py tests/interaction_vla/lerobot_bridge/test_teacher.py tests/interaction_vla/lerobot_bridge/test_relation_goals.py
git commit -m "feat: extract visual-observable TC-TIG teachers"
```

### Task 6: Atomic teacher sidecars and reproducible provenance

**Files:**

- Create: `interaction_vla/lerobot_bridge/sidecar.py`
- Create: `interaction_vla/lerobot_bridge/provenance.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_sidecar.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_provenance.py`

- [ ] **Step 1: Write atomic round-trip and corruption tests**

```python
import numpy as np
import pytest

from interaction_vla.lerobot_bridge.sidecar import TeacherSidecarWriter, load_teacher_sidecar
from interaction_vla.lerobot_bridge.teacher_schema import TeacherFrame


def test_sidecar_round_trip_has_one_row_per_frame_and_a_sha256(tmp_path) -> None:
    writer = TeacherSidecarWriter(tmp_path)
    frames = [
        TeacherFrame.zeros(frame_index=index, timestamp=index / 20.0, state_hash=f"state-{index}")
        for index in range(2)
    ]

    record = writer.write_episode(0, frames, np.zeros((2, 5), dtype=np.float32))
    loaded = load_teacher_sidecar(tmp_path / record.path, expected_sha256=record.sha256)

    assert record.frames == 2
    assert loaded["annotation.tc_tig.entity_pose"].shape == (2, 6, 9)
    np.testing.assert_array_equal(loaded["frame_index"], (0, 1))


def test_hash_mismatch_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, frame_index=np.asarray((0,), dtype=np.int32))

    with pytest.raises(ValueError, match="SHA-256"):
        load_teacher_sidecar(path, expected_sha256="0" * 64)
```

- [ ] **Step 2: Write deterministic fingerprint tests**

```python
from interaction_vla.lerobot_bridge.provenance import fingerprint_tree, sha256_file


def test_tree_fingerprint_changes_with_content_not_mtime(tmp_path) -> None:
    path = tmp_path / "meta.json"
    path.write_text('{"value": 1}', encoding="utf-8")
    first = fingerprint_tree(tmp_path)
    path.touch()
    assert fingerprint_tree(tmp_path) == first
    path.write_text('{"value": 2}', encoding="utf-8")
    assert fingerprint_tree(tmp_path) != first
    assert len(sha256_file(path)) == 64
```

- [ ] **Step 3: Run tests and verify both modules are missing**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_sidecar.py tests/interaction_vla/lerobot_bridge/test_provenance.py -q
```

Expected: import failures for `sidecar` and `provenance`.

- [ ] **Step 4: Implement atomic sidecars**

Stack every `TeacherFrame` field under its schema key, append
`annotation.tc_tig.relation_goal`, and write to
`teacher/episode_000000.npz.tmp` using an open binary file handle so NumPy does
not append another suffix. Flush and `os.fsync`, then rename to `.npz`. Return a
frozen manifest record containing relative path, episode index, frame count,
SHA-256, first/last state hashes, schema version, seed, object count, and task ID.
Reject empty frames, noncontiguous indices, timestamps differing from `index/20`
by more than `1e-6`, or goal arrays not shaped `[T,5]`.

- [ ] **Step 5: Implement provenance functions**

Provide `sha256_file`, `fingerprint_tree`, `runtime_versions`, `git_commit`, and
`source_fingerprint`. `fingerprint_tree` hashes each sorted relative path, size,
and content hash while excluding only `INCOMPLETE`, macOS `.DS_Store`, and ACT
output directories. `runtime_versions` records Python, platform, architecture,
LeRobot, torch, torchvision, MuJoCo, NumPy, FFmpeg, and the resolved device probe.
Invoke FFmpeg as `ffmpeg -version` without a shell.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_sidecar.py tests/interaction_vla/lerobot_bridge/test_provenance.py -q
```

Expected: all pass.

```bash
git add interaction_vla/lerobot_bridge/sidecar.py interaction_vla/lerobot_bridge/provenance.py tests/interaction_vla/lerobot_bridge/test_sidecar.py tests/interaction_vla/lerobot_bridge/test_provenance.py
git commit -m "feat: add hashed TC-TIG sidecars"
```

### Task 7: Standard LeRobotDataset v3 writer

**Files:**

- Create: `interaction_vla/lerobot_bridge/dataset_writer.py`
- Create: `tests/interaction_vla/lerobot_bridge/conftest.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_dataset_writer.py`

- [ ] **Step 1: Write a two-frame standard-loader integration test**

```python
import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")
from lerobot.datasets import LeRobotDataset

from interaction_vla.lerobot_bridge.dataset_writer import LeRobotEpisodeWriter


def test_two_frame_dataset_loads_through_standard_lerobot_api(tmp_path) -> None:
    root = tmp_path / "dataset"
    writer = LeRobotEpisodeWriter.create(
        repo_id="local/test_dual_view",
        root=root,
        fps=20,
        width=256,
        height=256,
    )
    for value in (0, 1):
        writer.add_frame(
            agent_rgb=np.full((256, 256, 3), value, dtype=np.uint8),
            wrist_rgb=np.full((256, 256, 3), value + 1, dtype=np.uint8),
            state=np.zeros(10, dtype=np.float32),
            action=np.zeros(7, dtype=np.float32),
            task="Pick up the green target object and place it inside the receptacle.",
        )
    writer.save_episode()
    writer.finalize()

    dataset = LeRobotDataset("local/test_dual_view", root=root)
    sample = dataset[0]
    assert len(dataset) == 2
    assert tuple(sample["observation.images.agent"].shape) == (3, 256, 256)
    assert tuple(sample["observation.images.wrist"].shape) == (3, 256, 256)
    assert tuple(sample["observation.state"].shape) == (10,)
    assert tuple(sample["action"].shape) == (7,)
    assert dataset.meta.tasks.index.tolist() == [
        "Pick up the green target object and place it inside the receptacle."
    ]
```

- [ ] **Step 2: Run with the LeRobot environment and verify the writer is missing**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge/test_dataset_writer.py -q
```

Expected: import failure for `dataset_writer`.

- [ ] **Step 3: Implement the exact standard feature schema**

Use `LeRobotDataset.create` with `use_videos=True`, `robot_type="franka_mujoco"`,
`image_writer_processes=0`, and `image_writer_threads=2`. Features are exactly:

```python
{
    "observation.images.agent": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (10,),
        "names": [
            "ee_x", "ee_y", "ee_z",
            "rot_c0_x", "rot_c0_y", "rot_c0_z",
            "rot_c1_x", "rot_c1_y", "rot_c1_z",
            "gripper_aperture",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx_local", "dy_local", "dz_local", "droll", "dpitch", "dyaw", "gripper"],
    },
}
```

Assert `importlib.metadata.version("lerobot") == "0.6.1"` before creation.
`add_frame` passes only these four arrays plus `task`; it never supplies timestamp
or frame index because LeRobot 0.6.1 creates both automatically.

- [ ] **Step 4: Wrap buffer and finalization behavior**

Expose `add_frame`, `clear_episode`, `save_episode`, and idempotent `finalize`.
Reject an existing root before calling LeRobot. Validate all dtypes/shapes before
delegation. After `finalize`, no add/save operation is permitted. Do not expose
`push_to_hub` from this local writer.

- [ ] **Step 5: Run integration test and inspect layout**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge/test_dataset_writer.py -q
```

Expected: pass; the temporary root contains `data/`, `videos/`, and `meta/` and is
loadable without network access.

In `conftest.py`, add the shared nine-frame fixture used by later ACT tests:

```python
from pathlib import Path

import numpy as np
import pytest

from interaction_vla.lerobot_bridge.dataset_writer import LeRobotEpisodeWriter


@pytest.fixture
def tiny_lerobot_dataset(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "tiny_lerobot"
    repo_id = "local/tiny_lerobot"
    writer = LeRobotEpisodeWriter.create(
        repo_id=repo_id,
        root=root,
        fps=20,
        width=256,
        height=256,
    )
    for index in range(9):
        writer.add_frame(
            agent_rgb=np.full((256, 256, 3), index, dtype=np.uint8),
            wrist_rgb=np.full((256, 256, 3), 2 * index, dtype=np.uint8),
            state=np.full(10, 0.01 * index, dtype=np.float32),
            action=np.zeros(7, dtype=np.float32),
            task="Pick up the green target object and place it inside the receptacle.",
        )
    writer.save_episode()
    writer.finalize()
    return root, repo_id
```

- [ ] **Step 6: Commit**

```bash
git add interaction_vla/lerobot_bridge/dataset_writer.py tests/interaction_vla/lerobot_bridge/conftest.py tests/interaction_vla/lerobot_bridge/test_dataset_writer.py
git commit -m "feat: write dual-view LeRobot datasets"
```

### Task 8: Transactional scripted-expert collector

**Files:**

- Create: `interaction_vla/lerobot_bridge/collector.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_collector.py`

- [ ] **Step 1: Write an event-order unit test with fakes**

```python
from types import SimpleNamespace

import numpy as np

from interaction_vla.lerobot_bridge.collector import collect_attempt


class FakeEnv:
    policy_hz = 20

    def __init__(self, events: list[str], terminal_reason: str) -> None:
        self.events = events
        self.terminal_reason = terminal_reason
        self.contact_diagnostics = object()
        self.grasp_state = object()

    def reset(self, **kwargs):
        self.events.append("reset")
        return SimpleNamespace(
            gripper=SimpleNamespace(
                position=np.zeros(3),
                orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
            )
        )

    def proprioception(self):
        self.events.append("encode_state")
        value = np.zeros(23, dtype=np.float32)
        value[13:15] = 0.04
        return value

    def step(self, action):
        self.events.append("step")
        return SimpleNamespace(
            snapshot=SimpleNamespace(
                gripper=SimpleNamespace(
                    position=np.zeros(3),
                    orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
                )
            ),
            done=True,
            reason=SimpleNamespace(value=self.terminal_reason),
        )


class FakeExpert:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reset(self, *, seed: int) -> None:
        return None

    def act(self, snapshot, contacts, grasp):
        self.events.append("expert_action")
        return np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)


class FakeCapture:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def capture(self, env, *, include_teacher: bool):
        self.events.append("capture")
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        view = SimpleNamespace(rgb=rgb)
        return SimpleNamespace(views={"agent": view, "wrist": view})


class FakePolicyWriter:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.clear_count = 0

    def add_frame(self, **kwargs) -> None:
        self.events.append("add_frame")

    def clear_episode(self) -> None:
        self.clear_count += 1


class FakeTeacher:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reset(self) -> None:
        return None

    def extract(self, snapshot, camera_frame, *, state):
        self.events.append("teacher")
        return object()


def test_frame_is_recorded_before_the_matching_action_is_executed() -> None:
    events: list[str] = []
    env = FakeEnv(events, terminal_reason="success")
    expert = FakeExpert(events)
    capture = FakeCapture(events)
    policy_writer = FakePolicyWriter(events)
    teacher = FakeTeacher(events)

    result = collect_attempt(
        env=env,
        expert=expert,
        capture=capture,
        policy_writer=policy_writer,
        teacher=teacher,
        seed=11,
        object_count=2,
        task="Pick up the green target object and place it inside the receptacle.",
    )

    assert result.reason == "success"
    assert events[:7] == [
        "reset",
        "capture",
        "encode_state",
        "teacher",
        "expert_action",
        "add_frame",
        "step",
    ]


def test_rejected_attempt_clears_buffer_without_episode_commit() -> None:
    events: list[str] = []
    writer = FakePolicyWriter(events)
    result = collect_attempt(
        env=FakeEnv(events, terminal_reason="timeout"),
        expert=FakeExpert(events),
        capture=FakeCapture(events),
        policy_writer=writer,
        teacher=FakeTeacher(events),
        seed=11,
        object_count=2,
        task="Pick up the green target object and place it inside the receptacle.",
    )

    assert result.reason == "timeout"
    assert writer.clear_count == 1
    assert result.accepted is False
```

- [ ] **Step 2: Write output-root and seed-schedule tests**

```python
from pathlib import Path

import pytest

from interaction_vla.lerobot_bridge.collector import collection_seed, require_new_root


def test_seed_schedule_is_deterministic_and_collision_free() -> None:
    values = [collection_seed(42, attempt) for attempt in range(50)]
    assert values == [collection_seed(42, attempt) for attempt in range(50)]
    assert len(set(values)) == 50


def test_existing_root_is_never_reused(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new dataset root"):
        require_new_root(root)
    assert (root / "keep.txt").read_text(encoding="utf-8") == "user data"
```

- [ ] **Step 3: Run tests and verify the collector is missing**

Run:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_collector.py -q
```

Expected: import failure for `collector`.

- [ ] **Step 4: Implement one-attempt collection**

`collect_attempt` owns no files. It resets with `LayoutMode.NORMAL`, resets the
expert and teacher tracker, then executes exactly:

```python
while True:
    camera_frame = capture.capture(env, include_teacher=True)
    state = EndEffectorStateCodec.encode_snapshot(snapshot, env.proprioception())
    rotation = EndEffectorStateCodec.quaternion_to_matrix(snapshot.gripper.orientation)
    teacher_frame = teacher.extract(
        snapshot,
        camera_frame,
        state=state,
    )
    action_world = expert.act(snapshot, env.contact_diagnostics, env.grasp_state)
    action_local = LocalCartesianActionCodec.encode(action_world, rotation)
    policy_writer.add_frame(
        agent_rgb=camera_frame.views["agent"].rgb,
        wrist_rgb=camera_frame.views["wrist"].rgb,
        state=state,
        action=action_local,
        task=task,
    )
    teacher_frames.append(teacher_frame)
    world_actions.append(action_world.copy())
    transition = env.step(action_world)
    snapshot = transition.snapshot
    if transition.done:
        break
```

The only privileged use of `contact_diagnostics` and `grasp_state` is inside the
existing scripted expert and success gate, never inside the teacher extractor or
policy frame. Return frames/actions/seed/object count/final reason to the caller.

- [ ] **Step 5: Add staged sidecar commit semantics**

Extend `TeacherSidecarWriter` from Task 6 with `stage_episode` and
`commit_staged`. `stage_episode` writes `teacher/.episode_000000.pending.npz` and
returns its hash record; `commit_staged` atomically renames it to
`teacher/episode_000000.npz`. Keep `write_episode` as a tested convenience that
calls both. In the collector, commit in this order:

1. derive relation goals and stage the sidecar;
2. call `LeRobotEpisodeWriter.save_episode()`;
3. rename the pending sidecar;
4. append and atomically rewrite `meta/teacher_manifest.json`.

If any step fails, retain `INCOMPLETE` and the pending/final artifacts for audit;
never attempt rollback of Parquet or video files.

- [ ] **Step 6: Implement the full collection transaction**

`collect_from_config(config_path)` must:

1. load the bridge and source configurations;
2. call existing `require_expert_gate(source_config_path, expert_gate)`;
3. reject an existing dataset root;
4. create `LeRobotEpisodeWriter`, then immediately create `INCOMPLETE`;
5. write `meta/tc_tig_teacher_schema.json`, camera calibration, empty manifest,
   empty rejection log, and provenance with `complete=false` atomically;
6. try at most `episodes * max_attempt_multiplier` deterministic attempts, cycling
   object counts by attempt index;
7. accept only `TerminationReason.SUCCESS`, clear the LeRobot episode buffer on
   every rejection, and append seed/object count/frame count/reason to
   `meta/rejections.json`;
8. finalize LeRobot only after the requested accepted count is reached;
9. call the internal validator with `allow_incomplete=True`;
10. write final hashes and `complete=true`, fsync metadata, then remove
    `INCOMPLETE`;
11. run the normal validator once more.

Use `np.random.SeedSequence((config_seed, 0x4C45524F, attempt))` for attempt
seeds. Instantiate a fresh environment, expert, renderer, and teacher tracker for
each attempt and always close the renderer in `finally`.

- [ ] **Step 7: Run collector unit tests**

Run unit tests:

```bash
.venv/bin/pytest tests/interaction_vla/lerobot_bridge/test_collector.py -q
```

Expected: all fake-based tests pass.

- [ ] **Step 8: Commit**

```bash
git add interaction_vla/lerobot_bridge/collector.py interaction_vla/lerobot_bridge/sidecar.py tests/interaction_vla/lerobot_bridge/test_collector.py tests/interaction_vla/lerobot_bridge/test_sidecar.py
git commit -m "feat: collect transactional LeRobot episodes"
```

### Task 9: Dataset validation and deterministic action replay

**Files:**

- Create: `interaction_vla/lerobot_bridge/validator.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_validator.py`

- [ ] **Step 1: Write incomplete and sidecar mismatch tests**

```python
import numpy as np
import pytest

from interaction_vla.lerobot_bridge.validator import (
    validate_dataset_root,
    validate_teacher_manifest,
)


def test_validator_rejects_incomplete_dataset(tmp_path) -> None:
    (tmp_path / "INCOMPLETE").write_text("collection in progress\n", encoding="utf-8")

    with pytest.raises(ValueError, match="INCOMPLETE"):
        validate_dataset_root(tmp_path, allow_incomplete=False)


def test_validator_rejects_manifest_frame_mismatch() -> None:
    records = [{"episode_index": 0, "frames": 3, "path": "teacher/episode_000000.npz"}]
    with pytest.raises(ValueError, match="frame count"):
        validate_teacher_manifest(records, dataset_episode_lengths={0: 2})
```

- [ ] **Step 2: Write standard sample and replay tests**

```python
def test_standard_samples_have_only_the_policy_contract(tiny_lerobot_dataset) -> None:
    root, repo_id = tiny_lerobot_dataset
    report = validate_dataset_root(
        root,
        repo_id=repo_id,
        allow_incomplete=True,
        require_bridge_metadata=False,
        replay=False,
    )

    assert report["frames"] == 9
    assert report["image_shape"] == [3, 256, 256]
    assert report["state_shape"] == [10]
    assert report["action_shape"] == [7]
    assert report["forbidden_policy_keys"] == []


def test_replay_error_threshold_is_enforced() -> None:
    recorded = np.zeros((2, 10), dtype=np.float32)
    replayed = recorded.copy()
    replayed[1, 3] = 2e-5

    with pytest.raises(ValueError, match="replay"):
        validate_replay_states(recorded, replayed, tolerance=1e-5)
```

Import `validate_replay_states` from the validator in this test file. Mark the
module with `pytest.importorskip("lerobot")` before importing LeRobot-dependent
helpers.

- [ ] **Step 3: Run with LeRobot and verify the validator is missing**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge/test_validator.py -q
```

Expected: import failure for `validator`.

- [ ] **Step 4: Implement structural and privileged-field validation**

`validate_dataset_root(root, allow_incomplete=False, replay=True)` checks:

- marker and `provenance.complete` state;
- official `LeRobotDataset(repo_id, root=root, local_files_only behavior through a
  local root)` load;
- feature keys exactly equal the four model features plus LeRobot-generated
  index/timestamp/task indices;
- no key contains `annotation`, `depth`, `segmentation`, or any forbidden fragment;
- exactly two image keys with decoded `[3,256,256]` tensors;
- state/action shape, dtype, finite values, and action/gripper bounds;
- manifest episode/frame totals, contiguous indices, timestamps, state hashes,
  sidecar SHA-256, schema version, and calibration hash;
- all provenance source/config/gate hashes.

Return a JSON-serializable report with `passed`, episode/frame totals, checked
hashes, replay maximum error, image schema, and task list.

Expose `validate_from_config(config_path, no_replay=False)`: it loads the bridge
configuration and calls `validate_dataset_root` with the configured root/repo ID,
full bridge metadata required, and replay enabled unless explicitly disabled.

- [ ] **Step 5: Implement deterministic replay**

For each manifest episode, reset a fresh `FrankaContactEnv` using its seed and
object count. Before each action, encode the current `SceneSnapshot` plus public
proprioception and compare it with the raw standard row. Decode the stored local
action using the snapshot gripper rotation and
call `env.step`. Require the final reason recorded in the manifest and maximum
absolute state error at most `1e-5`. Refuse replay when the runtime/source/scene
hashes differ rather than reporting a misleading numerical failure.

- [ ] **Step 6: Run validation tests and commit**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge/test_validator.py -q
```

Expected: all pass.

```bash
git add interaction_vla/lerobot_bridge/validator.py tests/interaction_vla/lerobot_bridge/test_validator.py
git commit -m "feat: validate and replay LeRobot datasets"
```

### Task 10: ACT one-batch update and Hugging Face checkpoint round-trip

**Files:**

- Create: `interaction_vla/lerobot_bridge/act_smoke.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_act_smoke.py`

- [ ] **Step 1: Write a tiny ACT integration test**

```python
import pytest
import torch

pytest.importorskip("lerobot")

from interaction_vla.lerobot_bridge.act_smoke import run_one_batch_check


def test_one_batch_act_update_and_reload_are_finite(tiny_lerobot_dataset, tmp_path) -> None:
    dataset_root, repo_id = tiny_lerobot_dataset
    result = run_one_batch_check(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_dir=tmp_path / "checkpoint",
        device=torch.device("cpu"),
        batch_size=1,
        seed=0,
        architecture="test",
    )

    assert result.loss >= 0.0
    assert result.gradient_norm > 0.0
    assert result.reload_max_abs_error <= 1e-5
    assert (tmp_path / "checkpoint" / "model.safetensors").is_file()
    assert (tmp_path / "checkpoint" / "policy_preprocessor.json").is_file()
    assert (tmp_path / "checkpoint" / "policy_postprocessor.json").is_file()
```

The `test` architecture uses `dim_model=64`, `dim_feedforward=128`, one encoder
layer, one VAE encoder layer, and no pretrained backbone weights. It preserves
both image inputs, the state input, VAE loss, and 8-step action output.

- [ ] **Step 2: Run the integration test and verify the ACT adapter is missing**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge/test_act_smoke.py -q
```

Expected: import failure for `act_smoke`.

- [ ] **Step 3: Build the version-locked ACT bundle**

Implement `load_act_dataset` with delta timestamps:

```python
delta_timestamps = {
    "action": [index / metadata.fps for index in range(8)],
}
dataset = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=delta_timestamps)
```

Create `ACTConfig` with `device`, `chunk_size=8`, `n_action_steps=8`,
`vision_backbone="resnet18"`, `pretrained_backbone_weights=None`, the configured
model/feedforward/layer widths, and default `use_vae=True`. Call
`make_policy(config, ds_meta=dataset.meta)` and
`make_pre_post_processors(config, dataset_stats=dataset.meta.stats)`. Do not add
the task string as an ACT input; retain it in the dataset for future VLAs.

- [ ] **Step 4: Implement one finite optimizer update**

Use a deterministic `DataLoader` with `shuffle=False`, `drop_last=True`, and
`num_workers=0`. Audit the raw and processed batch for teacher/forbidden keys,
then execute:

```python
processed = preprocessor(raw_batch)
policy.train()
optimizer.zero_grad(set_to_none=True)
loss, loss_dict = policy.forward(processed)
if not torch.isfinite(loss):
    raise FloatingPointError("ACT loss is not finite")
loss.backward()
gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
optimizer.step()
```

Use `torch.optim.AdamW(policy.get_optim_params(), lr=config.optimizer_lr,
weight_decay=config.optimizer_weight_decay)`.

- [ ] **Step 5: Save and reload through official pretrained APIs**

Save the policy, preprocessor, and postprocessor into one new checkpoint directory.
On a fixed raw batch, compute `policy.predict_action_chunk(preprocessor(batch))`,
then load `ACTPolicy.from_pretrained(path, local_files_only=True)` and load
processors with `make_pre_post_processors(reloaded.config,
pretrained_path=str(path))`. Reset seeds, set both policies to eval, and require
maximum absolute chunk difference at most `1e-5` on the same device.

Write `bridge_checkpoint.json` containing dataset fingerprint, standard feature
contract, state/action codec versions, LeRobot version, ACT configuration, device,
and all source/config/gate hashes.

Expose `check_from_config(config_path, output=None)`: validate the configured
dataset first, choose `<act.output_dir>/integration_check` when output is absent,
reject an existing nonempty destination, and call `run_one_batch_check`.

- [ ] **Step 6: Run the ACT integration test and commit**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge/test_act_smoke.py -q
```

Expected: pass on CPU with a finite loss and reload error at most `1e-5`.

```bash
git add interaction_vla/lerobot_bridge/act_smoke.py tests/interaction_vla/lerobot_bridge/test_act_smoke.py
git commit -m "feat: add ACT checkpoint smoke check"
```

### Task 11: Bounded ACT training with Mac OOM fallback

**Files:**

- Modify: `interaction_vla/lerobot_bridge/act_smoke.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_act_training.py`

- [ ] **Step 1: Write fallback and restart tests**

```python
import pytest

from interaction_vla.lerobot_bridge.act_smoke import run_training_with_fallback


def test_oom_restarts_once_at_batch_one(monkeypatch) -> None:
    attempts: list[int] = []

    def fake_train(*, batch_size: int, **kwargs):
        attempts.append(batch_size)
        if batch_size == 2:
            raise RuntimeError("MPS backend out of memory")
        return {"steps": 3, "losses": [1.0, 0.9, 0.8], "initial_state_hash": "fresh"}

    monkeypatch.setattr("interaction_vla.lerobot_bridge.act_smoke.train_once", fake_train)
    result = run_training_with_fallback(object(), batch_size=2)

    assert attempts == [2, 1]
    assert result["fallback_from_batch_size"] == 2
    assert result["batch_size"] == 1


def test_non_oom_error_is_not_retried(monkeypatch) -> None:
    attempts = 0

    def fake_train(**kwargs):
        nonlocal attempts
        attempts += 1
        raise ValueError("bad schema")

    monkeypatch.setattr("interaction_vla.lerobot_bridge.act_smoke.train_once", fake_train)
    with pytest.raises(ValueError, match="bad schema"):
        run_training_with_fallback(object(), batch_size=2)
    assert attempts == 1
```

- [ ] **Step 2: Write deterministic step-count and checkpoint tests**

```python
from interaction_vla.lerobot_bridge.act_smoke import bounded_batches


def test_bounded_batches_restart_loader_and_stop_exactly() -> None:
    values = list(bounded_batches(lambda: iter(("a", "b", "c")), steps=5))

    assert values == [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "b")]


def test_bounded_batches_reject_an_empty_loader() -> None:
    with pytest.raises(ValueError, match="empty"):
        list(bounded_batches(lambda: iter(()), steps=1))
```

- [ ] **Step 3: Run tests and verify new functions fail**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge/test_act_training.py -q
```

Expected: import errors for the new training functions.

- [ ] **Step 4: Implement deterministic smoke training**

Resolve `auto` through the existing `resolve_device`. Seed Python `random`, NumPy,
and torch before constructing the dataset, policy, optimizer, and loader. For the
smoke configuration, cycle the loader until exactly 500 successful optimizer
steps complete. Record per-step loss, L1 loss, KL loss, gradient norm, wall time,
device, batch size, and source episode indices. Save only final pretrained files
plus `training_summary.json`; W&B is never imported and `push_to_hub` is never
called.

Define an OOM as `RuntimeError` whose lowercased message contains either
`"out of memory"` or `"mps backend out of memory"`. On the first OOM at batch 2:

1. delete Python references to policy, optimizer, processors, loader, and batch;
2. call `gc.collect()` and `torch.mps.empty_cache()` when available;
3. reset all seeds;
4. reconstruct dataset, policy, optimizer, and processors from scratch;
5. retry exactly once with batch 1 and zero completed steps.

Any other error, or an OOM at batch 1, propagates.

Expose `train_from_config(config_path, output=None)`: validate the configured
dataset and required smoke gate, choose `<act.output_dir>/checkpoint` when output
is absent, reject an existing nonempty checkpoint directory, and call the bounded
trainer with the configured stop condition.

- [ ] **Step 5: Implement the pilot episode split and epoch gate**

For 50 episodes, deterministically permute episode indices with
`np.random.default_rng(config.seed)` and assign 40 train, 5 validation, and 5
held-out test episodes. Train five epochs and compute validation loss in eval/no-grad
mode after every epoch. Permit one epoch at a time up to ten only while each of
the last two validation losses improved by at least `1e-4`; record every extension
decision. Never inspect held-out test episodes while selecting the epoch count.
Before pilot training, load `required_smoke_report` and require `passed=true` plus
matching dataset/schema/codec/source versions.

- [ ] **Step 6: Run training unit tests**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge/test_act_training.py tests/interaction_vla/lerobot_bridge/test_act_smoke.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add interaction_vla/lerobot_bridge/act_smoke.py tests/interaction_vla/lerobot_bridge/test_act_training.py
git commit -m "feat: train bounded ACT smoke policies"
```

### Task 12: Closed-loop ACT rollout adapter

**Files:**

- Create: `interaction_vla/lerobot_bridge/rollout.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_rollout.py`

- [ ] **Step 1: Write observation and gripper-hysteresis tests**

```python
import numpy as np
import torch

from interaction_vla.lerobot_bridge.rollout import BinaryGripperHysteresis, policy_observation


def test_policy_observation_is_rgb_only_chw_float_plus_state() -> None:
    observation = policy_observation(
        agent_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist_rgb=np.full((256, 256, 3), 255, dtype=np.uint8),
        state=np.zeros(10, dtype=np.float32),
    )

    assert set(observation) == {
        "observation.images.agent",
        "observation.images.wrist",
        "observation.state",
    }
    assert observation["observation.images.agent"].shape == (3, 256, 256)
    assert observation["observation.images.agent"].dtype == torch.float32
    assert observation["observation.images.wrist"].max().item() == 1.0


def test_gripper_hysteresis_suppresses_midrange_chatter() -> None:
    gate = BinaryGripperHysteresis(close_threshold=0.4, open_threshold=0.6, initially_open=True)
    assert [gate.resolve(value) for value in (0.55, 0.45, 0.39, 0.50, 0.61)] == [1.0, 1.0, 0.0, 0.0, 1.0]
```

- [ ] **Step 2: Write a finite fake-policy rollout test**

```python
from interaction_vla.lerobot_bridge.rollout import ActionChunkQueue


def test_chunk_queue_queries_policy_only_after_eight_actions() -> None:
    calls = 0

    def predict() -> np.ndarray:
        nonlocal calls
        calls += 1
        chunk = np.zeros((8, 7), dtype=np.float32)
        chunk[:, 0] = calls
        return chunk

    queue = ActionChunkQueue(chunk_size=8)
    selected = [queue.next(predict) for _ in range(9)]

    assert calls == 2
    assert [item.queue_index for item in selected] == list(range(8)) + [0]
    assert selected[0].action[0] == 1.0
    assert selected[8].action[0] == 2.0
    assert selected[0].raw_chunk.shape == (8, 7)


def test_chunk_queue_rejects_nonfinite_or_wrong_shape() -> None:
    queue = ActionChunkQueue(chunk_size=8)
    with pytest.raises(ValueError, match="shape"):
        queue.next(lambda: np.zeros((7, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        queue.next(lambda: np.full((8, 7), np.nan, dtype=np.float32))
```

Import `pytest` in this test file.

- [ ] **Step 3: Run tests and verify rollout module is missing**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge/test_rollout.py -q
```

Expected: import failure for `rollout`.

- [ ] **Step 4: Implement checkpoint-bound adapter construction**

Before loading weights, compare `bridge_checkpoint.json` against the selected
dataset fingerprint, feature keys/shapes, codec versions, source hash, and LeRobot
version. Load `ACTPolicy.from_pretrained(checkpoint,
local_files_only=True)` and processors through
`make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))`.
Reject a device unavailable in the current process. Keep policy inputs limited to
the three keys in the observation test; ACT does not receive task text.

- [ ] **Step 5: Implement explicit chunk execution and diagnostics**

When the queue is empty, preprocess the current observation, call
`predict_action_chunk`, postprocess the complete chunk, require shape `[1,8,7]`,
and enqueue its eight actions. For each selected local action:

1. require finite values and clip pose components to `[-1,1]`;
2. resolve gripper score with thresholds `0.4/0.6`;
3. decode translation with the current gripper rotation;
4. call existing `project_cartesian_action(env.controller, action_world)`;
5. execute the projected action through `env.step`;
6. record raw chunk, queue index, raw local action, decoded world action, clipped
   dimensions, gripper state/switch count, IK scale/errors, state hash, and terminal
   reason.

Render only RGB through `capture.capture(env, include_teacher=False)` so rollout
cannot access depth or segmentation. Reset policy queue and hysteresis on every
environment reset.

- [ ] **Step 6: Implement the real bounded rollout entry point**

`rollout_checkpoint(config_path, checkpoint, seed, object_count=2)` creates a
normal-layout environment from the source configuration, runs at most 180 policy
steps, and atomically writes `rollout.json`. Acceptance requires correct action
shape, finite values, no schema/device/image/checkpoint/controller exception, and
a terminal reason or the 180-step cap. Record task success but do not require it.

Expose `rollout_from_config(config_path, checkpoint, seed=None,
object_count=2)`. When seed is absent, derive it from
`SeedSequence((bridge.seed, 0x524F4C4C))`; never reuse a training episode seed.

- [ ] **Step 7: Run rollout tests and commit**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge/test_rollout.py -q
```

Expected: all fake-policy tests pass.

```bash
git add interaction_vla/lerobot_bridge/rollout.py tests/interaction_vla/lerobot_bridge/test_rollout.py
git commit -m "feat: run ACT in the Franka environment"
```

### Task 13: Public CLI, smoke report, documentation, and full verification

**Files:**

- Create: `interaction_vla/lerobot_bridge/cli.py`
- Create: `interaction_vla/lerobot_bridge/__main__.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Write parser and no-upload tests**

```python
from interaction_vla.lerobot_bridge.cli import build_parser


def test_cli_exposes_only_local_explicit_commands() -> None:
    parser = build_parser()
    for command in ("collect", "validate", "act-check", "act-train", "smoke"):
        args = parser.parse_args([command, "--config", "configs/lerobot_act_smoke_macos.yaml"])
        assert args.command == command
    rollout = parser.parse_args(
        [
            "rollout",
            "--config",
            "configs/lerobot_act_smoke_macos.yaml",
            "--checkpoint",
            "outputs/lerobot/act_smoke/checkpoint",
        ]
    )
    assert rollout.command == "rollout"
    help_text = parser.format_help().lower()
    assert "push" not in help_text
    assert "upload" not in help_text


def test_smoke_dispatch_order(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("interaction_vla.lerobot_bridge.cli.validate_from_config", lambda *a, **k: calls.append("validate") or {})
    monkeypatch.setattr("interaction_vla.lerobot_bridge.cli.check_from_config", lambda *a, **k: calls.append("act-check") or {})
    monkeypatch.setattr("interaction_vla.lerobot_bridge.cli.train_from_config", lambda *a, **k: calls.append("act-train") or {"checkpoint": "checkpoint"})
    monkeypatch.setattr("interaction_vla.lerobot_bridge.cli.rollout_from_config", lambda *a, **k: calls.append("rollout") or {})
    monkeypatch.setattr("interaction_vla.lerobot_bridge.cli.write_smoke_report", lambda *a, **k: calls.append("report") or {})

    main(["smoke", "--config", "configs/lerobot_act_smoke_macos.yaml"])

    assert calls == ["validate", "act-check", "act-train", "rollout", "report"]
```

Import `main` beside `build_parser`.

- [ ] **Step 2: Run tests and verify CLI module is missing**

Run:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge/test_cli.py -q
```

Expected: import failure for `cli`.

- [ ] **Step 3: Implement CLI commands**

Every command requires `--config`. `validate` accepts `--no-replay`; `rollout`
accepts required `--checkpoint`, optional `--seed`, and `--object-count` defaulting
to 2. `act-check` and `act-train` accept optional `--output`. `smoke` uses the
configured dataset and output roots, refuses an existing final checkpoint output,
and does not collect data.

Implement `main(argv: Sequence[str] | None = None) -> None` so tests can pass an
explicit argument list while normal execution uses `sys.argv`.

Print one final JSON object to stdout and return nonzero on failure. Add
`__main__.py` containing:

```python
from .cli import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Build the final smoke report**

After the pipeline succeeds, atomically write
`outputs/lerobot/act_smoke/smoke_report.json` with:

```text
outputs/lerobot/act_smoke/
  integration_check/
  checkpoint/
  training_summary.json
  rollout.json
  smoke_report.json
```

The integration checkpoint and final training checkpoint are distinct; the
one-batch check must never overwrite final weights.

```json
{
  "passed": true,
  "dataset_validation": true,
  "episodes": 5,
  "one_batch_update": true,
  "training_steps": 500,
  "checkpoint_reload_max_abs_error": 0.0,
  "finite_rollout": true,
  "task_success": false,
  "dataset_fingerprint": "64 lowercase hexadecimal characters",
  "checkpoint_fingerprint": "64 lowercase hexadecimal characters"
}
```

The two boolean values `passed` and `finite_rollout` do not depend on
`task_success`. Derive numeric values from component reports rather than copying
the illustrative zero above.

- [ ] **Step 5: Document exact setup and execution commands**

Append this README section:

````markdown
### LeRobot 双视角数据与 ACT smoke

这个桥接使用独立的 Python 环境，不升级项目原有 `.venv`：

```bash
python3.12 -m venv .venv-lerobot
.venv-lerobot/bin/python -m pip install -r requirements-lerobot-macos.txt

.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge collect \
  --config configs/lerobot_act_smoke_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge validate \
  --config configs/lerobot_act_smoke_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge smoke \
  --config configs/lerobot_act_smoke_macos.yaml
```

标准 LeRobotDataset 样本只有 agent RGB、wrist RGB、10D 末端状态、7D
局部动作和 task metadata。深度、分割、相机矩阵与 TC-TIG 标签只存在于
`teacher/` sidecar，ACT 和 VLA 的标准 batch 不会读取它们。

首个数据集使用一条固定语言指令，用来验证 language-conditioned dataset
兼容性；ACT 本身不使用语言。设备在进程启动时探测：MPS 可用则使用 MPS，
否则回退 CPU。5 个 episode 与 500-step 训练只验证工程闭环，不构成任务性能
或语言泛化证据。所有命令仅写本地 Hugging Face 兼容数据和 checkpoint，当前
CLI 不提供 Hub upload。
````

- [ ] **Step 6: Run all automated tests in both environments**

Run the original suite:

```bash
.venv/bin/pytest -q
```

Expected: all existing and all bridge tests that do not require LeRobot pass; only
explicit graphical/optional-dependency skips remain.

Run the LeRobot bridge suite:

```bash
.venv-lerobot/bin/python -m pytest tests/interaction_vla/lerobot_bridge -q
```

Expected: every bridge test passes; graphical tests skip only in the Codex sandbox.

- [ ] **Step 7: Collect and validate the five-episode smoke dataset**

Run in the user's graphical macOS session:

```bash
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge collect \
  --config configs/lerobot_act_smoke_macos.yaml
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge validate \
  --config configs/lerobot_act_smoke_macos.yaml
```

Expected: five successful episodes load through standard LeRobotDataset; each
frame has two images, `[10]` state, `[7]` action, task metadata, and one matching
teacher row; replay maximum error is at most `1e-5`; no `INCOMPLETE` marker remains.

- [ ] **Step 8: Run ACT smoke and inspect its report**

Run:

```bash
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge smoke \
  --config configs/lerobot_act_smoke_macos.yaml
```

Expected: one-batch update, 500 optimizer steps with at most one documented batch
fallback, pretrained checkpoint reload error at most `1e-5`, one finite rollout,
and `smoke_report.json` with `passed=true`.

- [ ] **Step 9: Audit isolation and commit**

Run:

```bash
git status --short
git diff --check
```

Expected: no file below existing physics dataset/checkpoint roots is modified and
no generated `.venv-lerobot` or `outputs/lerobot` artifact is staged.

```bash
git add README.md interaction_vla/lerobot_bridge/cli.py interaction_vla/lerobot_bridge/__main__.py tests/interaction_vla/lerobot_bridge/test_cli.py
git commit -m "docs: add LeRobot ACT smoke workflow"
```

## Completion boundary

The implementation is complete only after both automated suites pass and the real
five-episode smoke workflow produces a standard-loadable dataset, matching hashed
teacher sidecars, a finite ACT update, a reload-equivalent local checkpoint, and a
finite MuJoCo rollout. Do not collect the 50-episode pilot, train SmolVLA, run pi0,
or implement the RGB-token-to-TC-TIG model in this plan.
