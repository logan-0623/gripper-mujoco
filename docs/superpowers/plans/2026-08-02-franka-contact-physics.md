# Franka Panda Contact-Physics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 500 Hz MuJoCo Franka Panda manipulation backend with a shared 7D Cartesian delta-pose interface, genuine bilateral friction grasping, synchronized four-view RGB-D, an independently validated scripted expert, and a fair 18D-edge Graph/Flat behavior-cloning comparison.

**Architecture:** Preserve the existing 4D kinematic experiment and introduce a separate `FrankaContactEnv` path selected by configuration. The physics environment is the sole source of poses, velocities, contacts, forces, grasp state, and termination; a shared SE(3) controller turns every 20 Hz action into 25 MuJoCo steps. Physics episodes use a versioned 18D interaction schema and the existing model/training machinery only after an expert validation artifact proves at least 90% success on held-out seeds.

**Tech Stack:** Python 3.12, NumPy, MuJoCo 3.3.4, PyTorch, PyYAML, Pillow, pytest, macOS `mjpython`, GLFW supplied with MuJoCo.

---

## Scope and file boundaries

The workspace is not a Git repository, so the implementation uses passing-test checkpoints instead of commit checkpoints. Existing files under `outputs/interaction_vla/`, legacy configs, and legacy visualization remain untouched.

New focused modules:

- `interaction_vla/franka_controller.py`: 7D action validation/scaling, SE(3) target integration, DLS IK, and arm/finger actuator commands.
- `interaction_vla/contact_physics.py`: contact-force aggregation, bilateral-contact tracking, stable-grasp state, and physics termination diagnostics.
- `interaction_vla/physics_env.py`: reset/layout/randomization, 25-substep rollout, snapshot/proprioception, and environment metadata.
- `interaction_vla/physics_expert.py`: contact-aware 7D scripted expert and retry state machine.
- `interaction_vla/physics_recording.py`: four-view synchronized RGB-D, metric depth, episode image archive, dashboard panels, and GIF frames.
- `interaction_vla/physics_data.py`: successful expert/teleop trajectory collection, manifests, metadata hashes, and expert-gate enforcement.
- `interaction_vla/validate_physics_expert.py`: independent validation cases and the 90% gate artifact.
- `interaction_vla/physics_visualize.py`: native viewer, four-panel dashboard, teleoperation, and GIF CLI.
- `interaction_vla/physics_evaluate.py`: paired physics rollouts, contact/grasp/place metrics, and edge-shuffle ablation.

Existing modules modified only at explicit compatibility seams:

- `interaction_vla/config.py`: backend-aware physics/controller/randomization/recording configuration while retaining legacy defaults.
- `interaction_vla/graph/schema.py` and `interaction_vla/graph/builder.py`: versioned legacy-10D and physics-18D edge schemas.
- `interaction_vla/data.py`, `interaction_vla/train.py`, and `interaction_vla/models/policy.py`: infer action/proprioception/edge sizes from a physics dataset/checkpoint instead of hard-coding 4/7/10, without changing legacy behavior.
- `interaction_vla/assets/franka_emika_panda/franka_tabletop.xml`: physical table, receptacle, free objects, gravity, 0.002 s timestep, and cameras.
- `interaction_vla/franka.py`: pinned asset paths, names, home pose, and provenance.
- `README.md`: separate legacy and physical commands.

Stage 2 household meshes remain outside this implementation. Stage 1 exposes a geometry registry boundary so Stage 2 can add mesh objects without changing controller, snapshot, Graph/Flat, or evaluation interfaces.

### Task 1: Establish a clean physical Franka scene and asset contract

**Files:**
- Modify: `interaction_vla/franka.py`
- Modify: `interaction_vla/assets/franka_emika_panda/franka_tabletop.xml`
- Modify: `tests/interaction_vla/test_franka_assets.py`

- [x] **Step 1: Extend the failing asset tests**

Replace the current scene assertions with explicit physical invariants:

```python
from pathlib import Path

import mujoco
import numpy as np


def test_franka_scene_is_a_500hz_gravity_contact_model() -> None:
    from interaction_vla.franka import FRANKA_SCENE_PATH

    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
    assert model.opt.timestep == 0.002
    np.testing.assert_allclose(model.opt.gravity, (0.0, 0.0, -9.81))
    assert model.nu == 8
    assert [model.joint(f"joint{index}").id for index in range(1, 8)]
    assert model.joint("finger_joint1").id >= 0
    assert model.joint("finger_joint2").id >= 0
    for index in range(5):
        joint = model.joint(f"object_{index}_joint")
        assert model.jnt_type[joint.id] == mujoco.mjtJoint.mjJNT_FREE
        assert model.geom(f"object_{index}_geom").contype != 0
    assert model.geom("table").contype != 0
    assert model.body("receptacle").id >= 0
    assert {model.camera(i).name for i in range(model.ncam)} == {
        "agentview", "wristview", "sideview", "topview"
    }


def test_scene_has_no_object_attachment_or_object_mocap() -> None:
    from interaction_vla.franka import FRANKA_SCENE_PATH

    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
    assert model.neq == 1  # upstream finger-joint coupling only
    for index in range(5):
        body = model.body(f"object_{index}")
        assert model.body_mocapid[body.id] == -1


def test_franka_asset_provenance_and_license_are_preserved() -> None:
    from interaction_vla.franka import FRANKA_ASSET_ROOT, FRANKA_COMMIT

    assert FRANKA_COMMIT == "71f066ad0be9cd271f7ed58c030243ef157af9f4"
    assert (FRANKA_ASSET_ROOT / "LICENSE").is_file()
    assert "Apache License" in (FRANKA_ASSET_ROOT / "LICENSE").read_text()
```

- [x] **Step 2: Run the focused test and preserve the known red state**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -p no:cacheprovider tests/interaction_vla/test_franka_assets.py -q
```

Expected: failures for missing `FRANKA_ASSET_ROOT`, zero gravity, `0.02` timestep, and non-colliding table/receptacle.

- [x] **Step 3: Define pinned asset and model names**

Implement `interaction_vla/franka.py` with stable constants used by every later task:

```python
from pathlib import Path

import numpy as np


FRANKA_ASSET_ROOT = Path(__file__).parent / "assets" / "franka_emika_panda"
FRANKA_SCENE_PATH = FRANKA_ASSET_ROOT / "franka_tabletop.xml"
FRANKA_COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{index}" for index in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
FINGER_ACTUATOR_NAME = "actuator8"
OBJECT_NAMES = tuple(f"object_{index}" for index in range(5))
CAMERA_NAMES = ("agentview", "wristview", "sideview", "topview")
HOME_QPOS = np.asarray((0.0, -0.45, 0.0, -2.25, 0.0, 1.85, 0.785), dtype=np.float64)
TCP_OFFSET_IN_HAND = np.asarray((0.0, 0.0, 0.1034), dtype=np.float64)
```

- [x] **Step 4: Replace the tabletop layer with a real contact scene**

Keep the official `panda_integration.xml` include and use these exact scene properties in `franka_tabletop.xml`: `<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>`; a colliding table geom with `friction="1.0 0.02 0.001"`; a tray-shaped `receptacle` body with a base and four walls; five freejoint `0.022 m` half-size cubes with `mass="0.10"`, `friction="1.0 0.02 0.001"`, `condim="6"`, and `solref="0.01 1"`; and the four camera names from the test. Keep the wrist camera on the existing mocap camera rig because it has no dynamics and is updated from the hand pose only during snapshot/render operations. Do not add an equality, weld, object mocap body, suction site, or object actuator.

The receptacle body must contain these named collision geoms so contact aggregation can classify it without geometric guesses:

```xml
<body name="receptacle" pos="0.67 -0.12 0.245">
  <geom name="receptacle_base" type="box" size="0.065 0.065 0.006"
        rgba="0.10 0.70 0.75 0.65" friction="1.0 0.02 0.001"/>
  <geom name="receptacle_wall_pos_x" type="box" pos="0.060 0 0.025"
        size="0.005 0.065 0.025" rgba="0.10 0.70 0.75 0.65"/>
  <geom name="receptacle_wall_neg_x" type="box" pos="-0.060 0 0.025"
        size="0.005 0.065 0.025" rgba="0.10 0.70 0.75 0.65"/>
  <geom name="receptacle_wall_pos_y" type="box" pos="0 0.060 0.025"
        size="0.065 0.005 0.025" rgba="0.10 0.70 0.75 0.65"/>
  <geom name="receptacle_wall_neg_y" type="box" pos="0 -0.060 0.025"
        size="0.065 0.005 0.025" rgba="0.10 0.70 0.75 0.65"/>
</body>
```

- [x] **Step 5: Run asset tests and the legacy suite**

Run the focused test, then the full suite. Expected: asset tests pass and the suite returns to zero failures; CoreGraphics-dependent tests may remain skipped.

### Task 2: Add backend-aware physics configuration without breaking 4D configs

**Files:**
- Modify: `interaction_vla/config.py`
- Create: `configs/physics_smoke_macos.yaml`
- Create: `configs/physics_pilot_macos.yaml`
- Modify: `tests/interaction_vla/test_config.py`

- [x] **Step 1: Write backend/action/frequency/randomization tests**

Add tests that load all existing configs as `backend == "kinematic"` with `action_dim == 4`, load `physics_smoke_macos.yaml` with `backend == "franka_contact"`, `action_dim == 7`, `timestep == 0.002`, `policy_hz == 20`, and `substeps == 25`, and reject both mismatched action dimensions and frequency products where `timestep * policy_hz * substeps != 1`.

Use this contract for the new dataclasses:

```python
@dataclass(frozen=True)
class RandomizationConfig:
    enabled: bool = False
    object_mass_scale: tuple[float, float] = (1.0, 1.0)
    friction_scale: tuple[float, float] = (1.0, 1.0)
    joint_damping_scale: tuple[float, float] = (1.0, 1.0)


@dataclass(frozen=True)
class PhysicsConfig:
    timestep: float = 0.002
    policy_hz: int = 20
    substeps: int = 25
    translation_delta: float = 0.02
    rotation_delta: float = math.radians(3.0)
    settle_steps: int = 250
    stable_grasp_frames: int = 10
    stable_lift_height: float = 0.01
    ik_damping: float = 0.05
    ik_iterations: int = 20
    ik_position_tolerance: float = 0.002
    ik_orientation_tolerance: float = math.radians(2.0)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool = False
    width: int = 256
    height: int = 256
    cameras: tuple[str, ...] = ("agentview", "wristview", "sideview", "topview")
```

- [x] **Step 2: Run configuration tests and verify the physics cases fail**

Run `tests/interaction_vla/test_config.py`; expected failures are missing dataclasses/backend fields and missing physics config files.

- [x] **Step 3: Implement config loading and cross-field validation**

Add `backend: str = "kinematic"`, `physics: PhysicsConfig`, and `recording: RecordingConfig` to `ExperimentConfig`. Change `ModelConfig` to accept only action dimensions `{4, 7}` and enforce the actual pairing in `ExperimentConfig.__post_init__`:

```python
expected_action_dim = 4 if self.backend == "kinematic" else 7
if self.backend not in {"kinematic", "franka_contact"}:
    raise ValueError("backend must be one of: kinematic, franka_contact")
if self.model.action_dim != expected_action_dim:
    raise ValueError(
        f"backend {self.backend} requires model.action_dim={expected_action_dim}"
    )
```

Validate finite positive scales, ordered randomization ranges, exactly four declared camera names, and the exact 500/20/25 relationship. Parse tuple-valued YAML fields without mutating the raw dictionary after loading.

- [x] **Step 4: Add smoke and pilot physics configs**

The smoke config uses `data_dir: outputs/interaction_graph_physics/smoke/data` and
`output_dir: outputs/interaction_graph_physics/smoke`; the pilot config uses
`data_dir: outputs/interaction_graph_physics/pilot/data` and
`output_dir: outputs/interaction_graph_physics/pilot`. Both use `backend: franka_contact`
and `action_dim: 7`. Smoke uses 2 expert gate cases per condition, 4 demonstrations,
1 model seed, and 1 training epoch. Pilot uses 20 normal plus 20 crowded gate cases,
50 demonstrations, object counts 2/3 for training, 4/5 for count/crowded OOD, and
three model seeds. Canonical randomization is disabled; the declared ranges remain
`[0.8,1.2]`, `[0.8,1.2]`, and `[0.9,1.1]` for an explicitly enabled robustness slice.

- [x] **Step 5: Run every configuration test**

Expected: old and new config tests pass, proving the 4D legacy and 7D physics contracts coexist.

### Task 3: Implement the shared 7D SE(3) controller

**Files:**
- Create: `interaction_vla/franka_controller.py`
- Create: `tests/interaction_vla/test_franka_controller.py`

- [x] **Step 1: Write tests for action semantics and target composition**

Cover shape/non-finite rejection, normalized translation clipping, rotation-vector norm clipping, `g >= 0.5` open semantics, body-frame rotation composition, joint-range clipping, and finite IK diagnostics. The key assertions are:

```python
command = CartesianCommand.from_action(
    np.asarray((1.0, -1.0, 0.5, 1.0, 1.0, 0.0, 0.49)),
    translation_delta=0.02,
    rotation_delta=np.deg2rad(3.0),
)
np.testing.assert_allclose(command.translation, (0.02, -0.02, 0.01))
assert np.linalg.norm(command.rotation_vector) == pytest.approx(np.deg2rad(3.0))
assert command.gripper_open is False
```

Create a real model/data test that calls `controller.apply_action(action)` and asserts it only changes `data.ctrl`, controller target state, and private `ik_data`; the production `data.qpos` is unchanged until `mujoco.mj_step` runs.

- [x] **Step 2: Run the controller tests and verify import failure**

Expected: failure because `franka_controller.py` does not exist.

- [x] **Step 3: Implement action and diagnostic dataclasses**

Use these public interfaces:

```python
@dataclass(frozen=True)
class CartesianCommand:
    translation: np.ndarray
    rotation_vector: np.ndarray
    gripper_open: bool

    @classmethod
    def from_action(
        cls, action: np.ndarray, *, translation_delta: float, rotation_delta: float
    ) -> "CartesianCommand":
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("Cartesian action must be a finite vector with shape (7,)")
        translation = np.clip(values[:3], -1.0, 1.0) * translation_delta
        raw_rotation = np.clip(values[3:6], -1.0, 1.0)
        norm = float(np.linalg.norm(raw_rotation))
        if norm > 1.0:
            raw_rotation = raw_rotation / norm
        return cls(
            translation=translation,
            rotation_vector=raw_rotation * rotation_delta,
            gripper_open=bool(values[6] >= 0.5),
        )


@dataclass(frozen=True)
class ControllerDiagnostics:
    ik_limited: bool
    position_error: float
    orientation_error: float
    iterations: int
    joint_target: np.ndarray
```

`from_action` converts to `float64`, requires `(7,)`, requires finite values, clips each translation command to `[-1,1]`, clips the three-dimensional rotation command by vector norm, scales it, and uses the threshold declared in the design.

- [x] **Step 4: Implement warm-started DLS IK and actuator output**

`FrankaCartesianController` stores one target position/rotation matrix, seven arm qpos addresses, seven arm dof addresses, joint ranges, hand body id, finger actuator id, and a separate `MjData` named `ik_data`. Compute TCP position as `hand_xpos + hand_xmat @ TCP_OFFSET_IN_HAND`; compute a 6×7 Jacobian with `mujoco.mj_jac` at that point; solve

```python
dq = jacobian.T @ np.linalg.solve(
    jacobian @ jacobian.T + damping**2 * np.eye(6),
    np.concatenate((position_error, orientation_error)),
)
```

at most `ik_iterations` times. Compose orientation targets as `target_rotation @ exp(rotation_vector)` so rotations are body-frame deltas. Clip every trial qpos to the official joint ranges and clip target change to `0.12 rad` per policy step. Write arm joint targets to actuator controls 1–7 and write `255.0` for open or `0.0` for closed to actuator 8. Never assign production `data.qpos` in `apply_action`.

- [x] **Step 5: Pass focused tests and run the legacy suite**

Expected: controller tests pass and all legacy tests remain unchanged.

### Task 4: Parse contact forces and define physical grasp state

**Files:**
- Create: `interaction_vla/contact_physics.py`
- Modify: `interaction_vla/graph/schema.py`
- Create: `tests/interaction_vla/test_contact_physics.py`

- [x] **Step 1: Write deterministic state-machine and force tests**

Test an independent `StableGraspTracker` with this exact sequence: bilateral contact on the table does not grasp; bilateral contact 11 mm above the table for 9 frames does not grasp; the tenth frame does; one missing finger clears current contact; a previously stable object then contacting the table outside the tray emits a drop diagnostic. Also construct a model contact, call `mujoco.mj_contactForce`, and assert normal/tangential magnitudes are finite and non-negative.

- [x] **Step 2: Run the new tests and verify import failure**

Expected: failure because the contact module and extended schema do not exist.

- [x] **Step 3: Add version-neutral interaction records to snapshots**

Extend `graph/schema.py` without changing legacy defaults:

```python
@dataclass(frozen=True)
class InteractionSignal:
    first: str
    second: str
    contact: bool = False
    normal_force: float = 0.0
    tangential_force: float = 0.0
    stable_grasp: bool = False
    support: bool = False

    @property
    def key(self) -> frozenset[str]:
        return frozenset((self.first, self.second))


@dataclass(frozen=True)
class SceneSnapshot:
    # existing fields remain in their existing order
    interactions: tuple[InteractionSignal, ...] = ()
```

Validate distinct names and finite non-negative forces in `InteractionSignal.__post_init__`.

- [x] **Step 4: Implement contact aggregation and grasp tracking**

`ContactParser` resolves body ancestry for every contact geom, classifies left finger, right finger, object, table, and receptacle bodies, and aggregates force over all geom contacts for the same semantic pair. It calls `mujoco.mj_contactForce(model, data, contact_id, six_force)`, sums `abs(six_force[0])` as normal force and `norm(six_force[1:3])` as tangential force, records individual left/right sets for diagnostics, and emits aggregate `gripper-object` interactions for representations.

Use these immutable diagnostics:

```python
@dataclass(frozen=True)
class ContactDiagnostics:
    left_objects: frozenset[str]
    right_objects: frozenset[str]
    object_table: frozenset[str]
    object_receptacle: frozenset[str]
    interactions: tuple[InteractionSignal, ...]


@dataclass(frozen=True)
class GraspState:
    bilateral_object: str | None
    stable_object: str | None
    stable_frames: int
    ever_stable_target: bool
```

The tracker consumes semantic contacts plus object-bottom/table-top heights and never writes MuJoCo model/data.

- [x] **Step 5: Run contact and schema tests**

Expected: all new tests pass; legacy snapshots still construct without supplying interactions.

### Task 5: Build `FrankaContactEnv` as the authoritative physics backend

**Files:**
- Create: `interaction_vla/physics_env.py`
- Create: `tests/interaction_vla/test_physics_env.py`
- Modify: `interaction_vla/env.py`

- [x] **Step 1: Add physics-only termination enum value**

Add `PHYSICS_FAILURE = "physics_failure"` to `TerminationReason`. Do not alter the values of existing members.

- [x] **Step 2: Write reset, timing, determinism, no-rewrite, and failure tests**

Tests instantiate two environments, reset them with the same seed/layout/object count, and compare active object qpos, target index, sampled physics metadata, and initial snapshot exactly. Instrument `mujoco.mj_step` to count exactly 25 calls for one action. Copy every active object freejoint qpos before the controller call and assert the environment does not assign those slices; verify movement occurs after stepping. Assert two distinct seeds produce distinct layouts. Assert a NaN state ends with `PHYSICS_FAILURE` rather than `TIMEOUT`.

- [x] **Step 3: Implement deterministic layout and physics sampling**

Use independent streams:

```python
layout_rng = np.random.default_rng(np.random.SeedSequence((seed, 0x4C41594F)))
physics_rng = np.random.default_rng(np.random.SeedSequence((seed, 0x50485953)))
target_rng = np.random.default_rng(np.random.SeedSequence((seed, 0x54415247)))
```

Normal layouts use at least 0.12 m center spacing. Crowded layouts place exactly one
distractor 0.055–0.075 m from the target while preventing initial overlap. Reset may assign
robot/object qpos and zero qvel, set inactive bodies to a collision-disabled parked state,
apply the seeded mass/inertia, friction, and damping sample, call `mj_forward`, and settle
for the configured number of 500 Hz steps. Object body mass and all three inertia components
must use the same sampled mass scale. No other method may assign active object qpos/qvel.

- [x] **Step 4: Implement environment stepping and snapshot state**

Expose `reset(seed, object_count, target_index=None, layout_mode="normal", recovery=None)`,
`step(action)`, `snapshot()`, `proprioception()`, and `physics_metadata()` on a class with
`action_dim = 7` and `policy_hz = 20`. `step` validates `(7,)`, updates the controller target
once, executes exactly 25 `mj_step` calls, updates the contact/grasp tracker on every
substep, increments policy step once, updates the wrist mocap camera from `hand` only
after stepping, and returns the current snapshot.

Proprioception is exactly 23D in this order: TCP position 3, quaternion 4, linear velocity
3, angular velocity 3, two finger qpos, seven arm qpos minus `HOME_QPOS`, and
gripper-open scalar. Lock this with `assert result.shape == (23,)` and store the dimension
in dataset/checkpoint metadata.

- [x] **Step 5: Implement physical termination and diagnostics**

Success requires target-receptacle contact stable for 10 physics frames, fingers open, and TCP at least 0.08 m above target. Wrong object requires `stable_object` to be non-target. Dropped requires a previously stable target to lose bilateral contact and contact the table outside the receptacle. Reject non-finite state, object center outside declared bounds, speed above the declared safety threshold, and repeated IK limitation with `PHYSICS_FAILURE`. Include step count, IK errors, left/right contacts, normal/tangential forces, stable object, drop count, and physics sample hash in `EnvStep.info`.

- [x] **Step 6: Run focused tests and a 100-step zero-action stability rollout**

Expected: tests pass, all values remain finite, objects remain supported by the table, and no active object moves without contact beyond settling tolerance.

### Task 6: Add the versioned 18D physical interaction graph

**Files:**
- Modify: `interaction_vla/graph/schema.py`
- Modify: `interaction_vla/graph/builder.py`
- Modify: `interaction_vla/models/policy.py`
- Modify: `tests/interaction_vla/test_graph.py`
- Create: `tests/interaction_vla/test_physics_graph.py`

- [x] **Step 1: Write exact 18D layout and parity tests**

Define constants:

```python
LEGACY_EDGE_FEATURE_DIM = 10
PHYSICS_EDGE_FEATURE_DIM = 18
EDGE_SCHEMAS = {"kinematic_v1": 10, "physics_v2": 18}
```

For a physical snapshot with a known relative quaternion and `InteractionSignal`, assert this order exactly: relative position `[0:3]`, relative rotation vector `[3:6]`, relative linear velocity `[6:9]`, relative angular velocity `[9:12]`, distance `[12]`, contact `[13]`, normal force `[14]`, tangential force `[15]`, stable grasp `[16]`, support `[17]`. Assert `flat_payload()` is exactly the same node/edge/mask arrays flattened, and legacy graphs remain 10D.

- [x] **Step 2: Run graph tests and verify the physics schema fails**

Expected: physics graph construction/validation fails while legacy graph tests remain green.

- [x] **Step 3: Version `SceneGraph` validation**

Add `feature_schema: str = "kinematic_v1"` to `SceneGraph`, validate the field against `EDGE_SCHEMAS`, and derive the required last dimension from it. Keep `EDGE_FEATURE_DIM = LEGACY_EDGE_FEATURE_DIM` as a compatibility alias so legacy imports/checkpoints continue to work.

- [x] **Step 4: Extend `SceneGraphBuilder` with an explicit schema**

`SceneGraphBuilder(max_objects=5, feature_schema="kinematic_v1")` retains the old calculation exactly. `feature_schema="physics_v2"` computes relative orientation as the shortest signed quaternion difference converted to an axis-angle vector, looks up unordered interaction forces, and populates all 18 fields. It must not infer contact from distance. Both representations receive the same `SceneGraph` object; only the encoder differs.

- [x] **Step 5: Make action policies dimension-explicit at checkpoint boundaries**

Keep the existing default 4D action mode for old direct unit tests, but require
`train_from_config` and physics checkpoint loading to pass `action_mode="cartesian_7d"`,
the dataset action dimension, and the proprioception dimension. The legacy head remains
`tanh(xyz) * 0.04 + sigmoid(gripper)`; the physics head is
`tanh(dx,dy,dz,drx,dry,drz) + sigmoid(gripper)` and therefore directly predicts the
environment's normalized 7D command. Add checkpoint metadata `backend`, `action_mode`,
`action_dim`, `proprioception_dim`, and `feature_schema`; reject a physics checkpoint
missing those fields or a 4D checkpoint requested by `FrankaContactEnv`.

- [x] **Step 6: Run graph, encoder, policy, and legacy checkpoint tests**

Expected: 10D and 18D schemas both pass, and loading the wrong backend gives a precise compatibility error.

### Task 7: Implement and tune the real-contact scripted expert

**Files:**
- Create: `interaction_vla/physics_expert.py`
- Create: `tests/interaction_vla/test_physics_expert.py`

- [x] **Step 1: Write action/transition/retry tests before tuning success rate**

Test every action has shape `(7,)`, finite values, translation/rotation commands in `[-1,1]`, and binary gripper labels. Use constructed snapshots/diagnostics to assert: APPROACH→ALIGN by pose tolerance; ALIGN→CLOSE; CLOSE→LIFT on bilateral target contact without waiting for stable grasp; LIFT→TRANSPORT only after stable grasp; lost contact causes OPEN_RECOVER→APPROACH; RELEASE waits for receptacle contact; RETREAT ends only after the environment reports success.

- [x] **Step 2: Run the tests and verify import failure**

Expected: failure because the physical expert does not exist.

- [x] **Step 3: Implement the contact-aware phase machine**

Use these phases: `APPROACH`, `ALIGN`, `DESCEND`, `CLOSE`, `LIFT`, `TRANSPORT`, `RELEASE`, `RETREAT`, `OPEN_RECOVER`. The expert reads privileged physics snapshot/diagnostics, maintains retry count, and outputs normalized 7D actions through one helper:

```python
def delta_pose_action(
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    *,
    gripper_open: bool,
    translation_delta: float,
    rotation_delta: float,
) -> np.ndarray:
    translation = np.clip(
        (target_position - current_position) / translation_delta, -1.0, 1.0
    )
    rotation = relative_body_rotvec(current_rotation, target_rotation)
    rotation = clip_by_norm(rotation / rotation_delta, 1.0)
    return np.concatenate((translation, rotation, [float(gripper_open)])).astype(np.float32)
```

Approach 0.12 m above the target, align above its center, descend to the declared pre-grasp
height with open fingers, close until bilateral contact, lift to 0.12 m above the table,
transport 0.10 m above tray center, open, wait for placement contact/stability, and retreat
vertically. Retry at most three times with deterministic ±5 mm lateral correction derived
from the episode seed and retry number.

- [x] **Step 4: Tune only shared controller/expert parameters on development seeds**

Run 10 normal and 10 crowded development cases. Change only scene friction/solver, controller limits/gains, waypoint heights/tolerances, and retry timing. Record every change in `outputs/interaction_graph_physics/development/expert_tuning.json`; do not inspect or tune against validation seeds used by Task 8.

- [x] **Step 5: Run expert transition tests and deterministic repeated rollouts**

Expected: unit tests pass and repeated identical seeded rollouts have identical reason, length, and state arrays within `1e-8` absolute tolerance.

### Task 8: Create the independent ≥90% expert gate

**Files:**
- Create: `interaction_vla/validate_physics_expert.py`
- Create: `tests/interaction_vla/test_validate_physics_expert.py`

- [x] **Step 1: Write held-out namespace and gate tests**

Assert validation seeds are generated from a namespace distinct from development/data/model seeds, cases contain both normal and crowded layouts, success rate below 0.90 exits non-zero, and a passing report contains controller/scene hashes plus every episode result.

- [x] **Step 2: Implement deterministic validation cases and report schema**

Generate case seeds with `SeedSequence((config.seed, 0x56414C47, condition_id, index))`. Run the expert from fresh environment instances. Save:

```json
{
  "passed": true,
  "threshold": 0.9,
  "success_rate": 0.9,
  "controller_hash": "a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7",
  "scene_hash": "b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8",
  "config_hash": "c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9c9",
  "conditions": {"normal": {"success": 18, "total": 20}, "crowded": {"success": 18, "total": 20}},
  "episodes": [{"condition": "normal", "seed": 710001, "reason": "success", "steps": 84}]
}
```

The values above illustrate only the report shape; the command computes every value. Exit
status is 0 only when overall success and each declared condition are at least 0.90, all
successful lifts meet bilateral/stable criteria, and no physics/no-attachment audit failed.

- [x] **Step 3: Run smoke gate, then the full 40-case gate**

Run:

```bash
.venv/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_smoke_macos.yaml
.venv/bin/python -m interaction_vla.validate_physics_expert \
  --config configs/physics_pilot_macos.yaml
```

Expected: both commands exit 0 before proceeding. If the full gate fails, stop here, preserve its episode diagnostics, and return to Task 7/controller tuning; do not collect training data.

### Task 9: Record synchronized four-view RGB-D and render a dashboard

**Files:**
- Create: `interaction_vla/physics_recording.py`
- Create: `tests/interaction_vla/test_physics_recording.py`

- [x] **Step 1: Write RGB-D shape, metric-depth, synchronization, and isolation tests**

Assert each of `agent`, `wrist`, `side`, `top` returns RGB `(256,256,3)` `uint8` and depth `(256,256)` `float32`; depth is finite positive metric distance where geometry is visible; all frames share exactly one policy timestamp/state hash; and enabling recording does not change qpos/qvel/action/reason for the same seeded rollout. Skip only actual CoreGraphics rendering when no macOS graphical session exists.

- [x] **Step 2: Implement one-state four-camera capture**

Use these exact record types:

```python
@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray


@dataclass(frozen=True)
class MultiViewFrame:
    policy_step: int
    simulation_time: float
    state_hash: str
    views: dict[str, RGBDFrame]
```

`MultiViewRecorder.capture(env)` returns `MultiViewFrame`, and
`MultiViewRecorder.save_episode(frames, path)` returns the written `Path`. Capture all views
without calling `mj_step`. Use MuJoCo's depth-rendering mode and verify its returned
distances against near/far clipping; if raw depth is read through `mjr_readPixels`, convert
it with `near / (1 - z * (1 - near / far))`. Store four RGB and four float32 depth arrays
in one compressed NPZ with policy step/time/state hashes.

- [x] **Step 3: Implement 2×2 frame composition and GIF output**

Compose Agent/Wrist on the top row and Side/Top on the bottom row, with a 24-pixel label/diagnostic bar for controller, step, IK state, left/right contact, stable object, target, and reason. Reuse `save_animated_gif` but keep physical GIF output under `docs/media/franka_contact_expert.gif`.

- [x] **Step 4: Run recording tests in headless and Mac graphical modes**

Expected: logical/storage/isolation tests always pass; real RGB-D tests pass from the user's normal Terminal through `mjpython`.

### Task 10: Add physical recovery initialization and optional teleoperation

**Files:**
- Create: `interaction_vla/physics_recovery.py`
- Create: `interaction_vla/teleop.py`
- Create: `tests/interaction_vla/test_physics_recovery.py`
- Create: `tests/interaction_vla/test_teleop.py`

- [x] **Step 1: Write reset-only recovery and keyboard mapping tests**

Assert recovery specs are deterministic; affect only reset/controller targets before rollout; cannot be applied after `step_count > 0`; and produce no object qpos assignment after reset. Assert W/A/S/D, R/F, Q/E, arrow keys, Space, Z, and Escape map to the exact 7D semantics in the design.

- [x] **Step 2: Implement physical recovery specs**

Use reset variants `alignment_offset`, `premature_close`, `off_center_approach`, and `transport_target_offset`. Store seed, variant, source episode, initial controller target delta, and commanded gripper state. Apply these only inside `FrankaContactEnv.reset` before settling/first observation; rollout recovery is then driven solely by actions/contact/gravity.

- [x] **Step 3: Implement stateful teleop input**

`TeleopController` maintains gripper state, consumes GLFW key press/release events, and returns one normalized 7D action per 20 Hz tick. `Space` toggles only on a press edge, `Z` sets `discard_and_reset`, and `Escape` sets `quit`. Translation and rotation keys may be held and combine, with each vector clipped by norm to one.

- [x] **Step 4: Run recovery and teleop tests**

Expected: deterministic reset metadata and all keyboard mappings pass; the static audit finds no recovery-time object writes after reset.

### Task 11: Collect auditable 7D physics episodes and enforce the gate

**Files:**
- Create: `interaction_vla/physics_data.py`
- Modify: `interaction_vla/data.py`
- Modify: `interaction_vla/train.py`
- Create: `tests/interaction_vla/test_physics_data.py`
- Modify: `tests/interaction_vla/test_train.py`

- [x] **Step 1: Write dataset schema, manifest parity, and gate refusal tests**

Assert actions have final dimension 7, edge features 18, state/action frames are pre-action aligned, source is `scripted` or `teleop`, and metadata includes contact/stable-grasp diagnostics, scene/controller versions, physics sample/hash, recovery source, and optional RGB-D sidecar filename. Assert collection/training refuses a missing, failed, or hash-stale expert gate. Assert Flat and Graph resolve the exact same ordered episode filenames.

- [x] **Step 2: Implement physics episode collection**

`collect_physics_episode` uses `FrankaContactEnv`, `PhysicsScriptedExpert` or `TeleopController`, and `SceneGraphBuilder(feature_schema="physics_v2")`. Save only successful scripted demonstrations in the formal manifest; write failures to `rejections.json` with full termination diagnostics. Teleop is stored separately and enters the formal manifest only through an explicit shared manifest merge command.

The NPZ contract is:

```text
metadata: scalar JSON string
node_features: [T,N,23] float32
edge_index: [2,E] int64
edge_features: [T,E,18] float32
node_mask: [T,N] bool
edge_mask: [T,E] bool
proprioception: [T,P] float32
actions: [T,7] float32
phases: [T] unicode
contact_state: [T,O,2] bool
contact_force: [T,O,2] float32
relative_pose: [T,O,6] float32
stable_grasp: [T,O] bool
```

- [x] **Step 3: Make training dimensions dataset-driven**

In `train_from_config`, inspect the first manifest episode and assert all episodes agree on
node, edge, proprioception, and action dimensions. Replace hard-coded
`edge_feature_dim: 10` with the loaded final dimension; pass
`action_mode="cartesian_7d"`, `action_dim=7`, and the loaded proprioception dimension into
the policy. Train directly against the stored normalized 7D environment commands; action
statistics are retained for normalized-error reporting but are not used to transform
policy outputs. Store these values, backend, schema, manifest hash, expert-gate hash, and
controller/scene hash in checkpoints. Keep old 4D code paths and checkpoint fields valid
when `backend == "kinematic"`.

- [x] **Step 4: Add the collection command and run a smoke collection**

Run:

```bash
.venv/bin/python -m interaction_vla.physics_data collect \
  --config configs/physics_smoke_macos.yaml \
  --expert-gate outputs/interaction_graph_physics/smoke/expert_gate.json
```

Expected: the declared number of successful demonstrations, a manifest with unique filenames, auditable rejections, and optional RGB-D files only when recording is enabled.

- [x] **Step 5: Run physics data/train tests and all legacy data/train tests**

Expected: both schemas train without hard-coded dimension errors and legacy checkpoint resume tests remain green.

### Task 12: Run paired Graph/Flat physics evaluation with interaction metrics

**Files:**
- Create: `interaction_vla/physics_evaluate.py`
- Modify: `interaction_vla/evaluate.py`
- Create: `tests/interaction_vla/test_physics_evaluate.py`

- [x] **Step 1: Write paired-case, edge-shuffle, and metric tests**

Assert Flat, Graph, and Graph-edge-shuffled policies receive equal initial qpos/qvel, target, layout, physics sample/hash, controller settings, and max steps for each case. Assert shuffle permutes only valid edge rows and leaves every feature value/mask/node/action scale unchanged. Verify aggregation for success, bilateral contact, stable lift, wrong-object stable grasp, contact loss/drop, placement, episode length, and IK/physics failure.

- [x] **Step 2: Implement backend-checked policy rollout**

Load only `franka_contact`, 7D, `physics_v2` checkpoints. Create a fresh
`FrankaContactEnv` for every paired case and build the same 18D observation. The policy
already predicts the normalized seven-dimensional environment command, so evaluation
clips it to the declared command contract without statistical de-normalization and steps
the shared controller. A policy ending early does not alter the case presented to another
policy.

- [x] **Step 3: Implement reports and representation comparison**

Write pilot per-episode CSV and JSON under
`outputs/interaction_graph_physics/pilot/evaluation/`; use the config's explicit
`output_dir/evaluation` for other named runs. Group ID, count-OOD, crowded-OOD, and
controlled-randomization separately. Report every model seed and paired Graph−Flat deltas.
Treat the old +10 pp criterion as directional, not as a forced pass. Include edge-shuffle
deltas and wrong-object/stable-lift metrics so object/gripper awareness is independently
visible.

- [x] **Step 4: Train and evaluate the physics smoke path**

Run Flat and Graph for seed 0, then paired evaluation including shuffled Graph. Expected: commands complete, dimensional contracts match, and all metric fields are present. Smoke numbers are pipeline checks, not claims.

### Task 13: Provide four-panel viewer, native viewer, teleop, and GIF commands

**Files:**
- Create: `interaction_vla/physics_visualize.py`
- Create: `tests/interaction_vla/test_physics_visualize.py`
- Modify: `README.md`

- [x] **Step 1: Write CLI/parser/session tests**

Require subcommands `dashboard`, `native`, `teleop`, and `export-gif`; controllers `expert`, `flat`, and `graph`; normal/crowded layouts; seed/object count; recording path; and checkpoint compatibility. Assert the four-panel order and labels. Assert non-GIF output fails before renderer creation.

- [x] **Step 2: Implement the physical visualization session**

`PhysicsVisualizationSession` owns exactly one environment, one controller/expert/teleop source, one 18D builder, and one optional recorder. `advance` generates one 7D command and advances one 20 Hz policy step. It never synchronizes a fake backend and never writes object qpos.

- [x] **Step 3: Implement GLFW dashboard and native viewer**

Dashboard creates one GLFW window and one MuJoCo render context, draws the four cameras into four `MjrRect` viewports from the same data state, overlays diagnostics, and advances at 20 Hz. Register callbacks through `TeleopController` only in teleop mode. Native mode uses `mujoco.viewer.launch_passive` with the same environment and control loop. On macOS, catch launch errors and print the exact `mjpython` command.

- [x] **Step 4: Implement expert/policy GIF export**

Capture a four-panel frame at step zero and after each policy action until termination, then use Pillow at 20 FPS. For Flat/Graph comparison, compose two four-panel dashboards side by side from independently initialized but paired environments. Never reuse the legacy abstract-gripper GIF path.

- [x] **Step 5: Add runnable README commands**

Document these commands exactly:

```bash
.venv/bin/python -m interaction_vla.physics_visualize dashboard \
  --controller expert --layout crowded --object-count 4 --seed 2140049

.venv/bin/python -m interaction_vla.physics_visualize teleop \
  --layout normal --object-count 3 --seed 2140049 --record outputs/teleop_demo.npz

.venv/bin/mjpython -m interaction_vla.physics_visualize native \
  --controller expert --layout crowded --object-count 4 --seed 2140049

.venv/bin/mjpython -m interaction_vla.physics_visualize export-gif \
  --controller expert --layout crowded --object-count 4 --seed 2140049 \
  --output docs/media/franka_contact_expert.gif
```

Explain that the old `interaction_vla.visualize` command is the preserved kinematic baseline and the new `physics_visualize` command is the real Panda/contact system.

- [x] **Step 6: Run logical tests and Mac visual acceptance**

Expected: parser/layout/GIF storage tests pass headlessly. In a normal macOS graphical Terminal, the dashboard shows the complete Panda, both moving fingers, physical object lift/drop, and synchronized Agent/Wrist/Side/Top images.

### Task 14: Perform no-attachment audit and end-to-end verification

**Files:**
- Create: `tests/interaction_vla/test_no_attachment_audit.py`
- Modify: `README.md`

- [x] **Step 1: Add a source/model/runtime audit test**

Parse the compiled model and assert no equality references an object, no object is mocap, no object has an actuator, and only the upstream finger coupling equality exists. Inspect `FrankaContactEnv.step`, `FrankaCartesianController.apply_action`, contact tracking, expert, teleop, and visualization source with AST and reject assignments/subscript writes targeting production active-object qpos/qvel. At runtime, compare object acceleration against MuJoCo contact/gravity stepping and assert removing finger friction or either fingertip collision prevents the stable-grasp gate.

- [x] **Step 2: Run the complete deterministic test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -p no:cacheprovider tests/interaction_vla -q
```

Expected: zero failures; only explicitly marked CoreGraphics/MPS skips are allowed.

- [x] **Step 3: Run syntax compilation without repository caches**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/interaction_graph_physics_pycache \
  .venv/bin/python -m compileall -q interaction_vla tests/interaction_vla
```

Expected: exit 0 with no output.

- [x] **Step 4: Re-run the full expert gate and smoke experiment**

The expert gate must still pass from a clean process. Then collect smoke data, train Flat/Graph, run paired evaluation, and export the expert GIF. Record exact commands, versions, gate rates, artifact paths, and test counts in `outputs/interaction_graph_physics/verification.json`.

- [x] **Step 5: Verify legacy artifacts and commands remain available**

Assert `outputs/interaction_vla/` and `docs/media/flat_vs_graph_crowded.gif` are unchanged, legacy configs still load as 4D, and one legacy smoke test completes. The physical output namespace must contain no symlink or overwrite pointing into the legacy namespace.

- [x] **Step 6: Complete Mac acceptance**

Run the documented dashboard and native commands through `.venv/bin/mjpython`. Confirm visible Franka geometry, IK-linked 6D end-effector motion, actual bilateral friction grasp, gravity release, and four synchronized views. Export `docs/media/franka_contact_expert.gif` and add it to the physical section of the README only after this acceptance succeeds.
