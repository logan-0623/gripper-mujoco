# Franka Panda Four-View Visualization Implementation Plan

> **SUPERSEDED — DO NOT EXECUTE:** The user replaced deterministic attachment
> with full friction/contact grasping. Use the contact-physics design and its
> replacement implementation plan instead.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the abstract gripper mirror with a complete IK-driven Franka Panda and provide a live Agent/Wrist/Side/Top dashboard plus four-view GIF export without changing Graph/Flat policy behavior.

**Architecture:** `KinematicTabletopEnv` remains the authoritative task simulator. A new `FrankaMirror` maps every snapshot through one affine transform, solves warm-started damped-least-squares IK, writes Panda/finger/object state into one MuJoCo model, and exposes four fixed cameras. Visualization consumes that mirror for either a raw-GLFW 2×2 dashboard, the native free camera, or deterministic GIF frames.

**Tech Stack:** Python 3.12, MuJoCo 3.3.4 (`MjModel`, Jacobians, `MjvScene`, `MjrContext`), NumPy, Pillow, GLFW, pytest, official MuJoCo Menagerie Franka assets pinned to commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`.

**Repository note:** This workspace has no `.git` directory. Every task ends with a test checkpoint rather than a commit; do not initialize Git as part of this work.

---

## File map

- Create `interaction_vla/assets/franka_emika_panda/`: pinned official Panda MJCF, meshes, license and upstream README.
- Create `interaction_vla/assets/franka_tabletop.xml`: project table, semantic objects and four cameras around the included Panda.
- Create `interaction_vla/assets/README.md`: upstream URL, exact commit, license boundary and local integration notes.
- Create `interaction_vla/franka.py`: scene constants, affine coordinate mapping, IK solver and `FrankaMirror` synchronization.
- Modify `interaction_vla/mujoco_env.py`: preserve the existing task API while delegating visual state to `FrankaMirror`.
- Modify `interaction_vla/visualize.py`: semantic coloring, four-camera rendering, dashboard, single-controller GIF and CLI.
- Modify `README.md`: Mac setup, four-view viewer/native viewer/GIF commands and experiment-isolation explanation.
- Create `tests/interaction_vla/test_franka_assets.py`: model and license contract.
- Create `tests/interaction_vla/test_franka.py`: transform, IK, finger/object synchronization and isolation tests.
- Modify `tests/interaction_vla/test_mujoco_env.py`: update the mirror-level assertions for Panda.
- Modify `tests/interaction_vla/test_visualize.py`: four-panel composition, viewport layout, CLI and export tests.

### Task 1: Vendor the pinned Panda and compile the tabletop scene

**Files:**
- Create: `interaction_vla/assets/franka_emika_panda/**`
- Create: `interaction_vla/assets/franka_tabletop.xml`
- Create: `interaction_vla/assets/README.md`
- Create: `tests/interaction_vla/test_franka_assets.py`

- [ ] **Step 1: Write the failing asset contract test**

```python
from pathlib import Path

import mujoco

from interaction_vla.franka import FRANKA_SCENE_PATH


def test_franka_scene_contains_robot_fingers_objects_and_cameras() -> None:
    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
    assert [model.joint(f"joint{i}").id for i in range(1, 8)]
    assert model.joint("finger_joint1").id >= 0
    assert model.joint("finger_joint2").id >= 0
    assert model.body("hand").id >= 0
    for index in range(5):
        assert model.joint(f"object_{index}_joint").id >= 0
        assert model.geom(f"object_{index}_geom").id >= 0
    for camera in ("agentview", "wristview", "sideview", "topview"):
        assert model.camera(camera).id >= 0
    assert model.vis.global_.offwidth >= 640
    assert model.vis.global_.offheight >= 480


def test_franka_asset_provenance_and_license_are_preserved() -> None:
    root = FRANKA_SCENE_PATH.parent
    text = (root / "README.md").read_text()
    assert "71f066ad0be9cd271f7ed58c030243ef157af9f4" in text
    assert "google-deepmind/mujoco_menagerie" in text
    assert (root / "franka_emika_panda" / "LICENSE").is_file()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -p no:cacheprovider tests/interaction_vla/test_franka_assets.py -q
```

Expected: collection fails because `interaction_vla.franka` and the scene assets do not exist.

- [ ] **Step 3: Import only the pinned official model directory**

Use a disposable sparse checkout outside the project, then copy only the required model directory:

```bash
franka_tmp=$(mktemp -d /private/tmp/franka-menagerie.XXXXXX)
git clone --filter=blob:none --no-checkout \
  https://github.com/google-deepmind/mujoco_menagerie.git \
  "$franka_tmp/menagerie"
git -C "$franka_tmp/menagerie" sparse-checkout init --cone
git -C "$franka_tmp/menagerie" sparse-checkout set franka_emika_panda
git -C "$franka_tmp/menagerie" checkout \
  71f066ad0be9cd271f7ed58c030243ef157af9f4
mkdir -p interaction_vla/assets
cp -R "$franka_tmp/menagerie/franka_emika_panda" interaction_vla/assets/
```

Do not copy Menagerie `.git`, gallery assets, or any other robot. Preserve the model `LICENSE` and `README.md` unchanged.

- [ ] **Step 4: Add the project scene and provenance file**

Create `interaction_vla/assets/franka_tabletop.xml` with the official model include plus the task mirror:

```xml
<mujoco model="interaction graph franka tabletop">
  <include file="franka_emika_panda/panda.xml"/>
  <option timestep="0.02" gravity="0 0 0"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="640" offheight="480"/>
  </visual>
  <worldbody>
    <light pos="0.5 0 1.5" dir="0 0 -1" directional="true"/>
    <body name="table_focus" pos="0.5 0 0.28"/>
    <geom name="table" type="box" pos="0.5 0 0.205" size="0.30 0.24 0.02"
          rgba="0.65 0.55 0.45 1" contype="0" conaffinity="0"/>
    <geom name="receptacle" type="cylinder" pos="0.665 -0.099 0.239"
          size="0.055 0.006" rgba="0.10 0.70 0.75 0.65" contype="0" conaffinity="0"/>
    <body name="wrist_camera_rig" mocap="true" pos="0.3 0 0.45">
      <camera name="wristview" pos="0 0 0" quat="1 0 0 0" fovy="75"/>
    </body>
    <camera name="agentview" mode="targetbody" target="table_focus"
            pos="1.15 -1.05 0.95" fovy="45"/>
    <camera name="sideview" mode="targetbody" target="table_focus"
            pos="0.5 -1.25 0.58" fovy="50"/>
    <camera name="topview" mode="targetbody" target="table_focus"
            pos="0.5 0 1.45" fovy="42"/>
    <body name="object_0" pos="0 0 -2"><freejoint name="object_0_joint"/><geom name="object_0_geom" type="box" size="0.022 0.022 0.022" mass="0.1" rgba="0.2 0.4 0.9 1"/></body>
    <body name="object_1" pos="0 0 -2"><freejoint name="object_1_joint"/><geom name="object_1_geom" type="box" size="0.022 0.022 0.022" mass="0.1" rgba="0.2 0.4 0.9 1"/></body>
    <body name="object_2" pos="0 0 -2"><freejoint name="object_2_joint"/><geom name="object_2_geom" type="box" size="0.022 0.022 0.022" mass="0.1" rgba="0.2 0.4 0.9 1"/></body>
    <body name="object_3" pos="0 0 -2"><freejoint name="object_3_joint"/><geom name="object_3_geom" type="box" size="0.022 0.022 0.022" mass="0.1" rgba="0.2 0.4 0.9 1"/></body>
    <body name="object_4" pos="0 0 -2"><freejoint name="object_4_joint"/><geom name="object_4_geom" type="box" size="0.022 0.022 0.022" mass="0.1" rgba="0.2 0.4 0.9 1"/></body>
  </worldbody>
</mujoco>
```

If the upstream `home` keyframe is rejected after the scene adds free joints, remove only the upstream `<keyframe>` block in the vendored integration copy and document that exact local change in `interaction_vla/assets/README.md`. Do not change mesh geometry, inertials, joint ranges or visual materials.

Create `interaction_vla/assets/README.md` with the upstream URL, pinned commit, the fact that only `franka_emika_panda` was imported, its Apache-2.0 model license, and a statement that `franka_tabletop.xml` is project-owned integration code.

- [ ] **Step 5: Add the minimal scene-path constant**

Create the initial `interaction_vla/franka.py`:

```python
from pathlib import Path


FRANKA_SCENE_PATH = Path(__file__).parent / "assets" / "franka_tabletop.xml"
```

- [ ] **Step 6: Run the asset contract and verify GREEN**

Run the focused test command from Step 2. Expected: `2 passed` and no MuJoCo XML warnings or missing mesh errors.

### Task 2: Implement the coordinate transform and damped-least-squares IK

**Files:**
- Modify: `interaction_vla/franka.py`
- Create: `tests/interaction_vla/test_franka.py`

- [ ] **Step 1: Write failing transform and IK tests**

```python
import numpy as np
import mujoco

from interaction_vla.franka import (
    FRANKA_SCENE_PATH,
    POLICY_TO_SCENE_OFFSET,
    POLICY_TO_SCENE_SCALE,
    policy_to_scene,
    solve_franka_ik,
)


def test_policy_to_scene_is_uniform_and_preserves_nearest_neighbor() -> None:
    points = np.asarray(((-0.2, -0.1, 0.04), (0.1, 0.2, 0.04), (0.0, 0.0, 0.28)))
    mapped = policy_to_scene(points)
    np.testing.assert_allclose(mapped, points * POLICY_TO_SCENE_SCALE + POLICY_TO_SCENE_OFFSET)
    np.testing.assert_allclose(
        np.linalg.norm(mapped[0] - mapped[1]),
        POLICY_TO_SCENE_SCALE * np.linalg.norm(points[0] - points[1]),
    )


def test_franka_ik_reaches_readme_initial_target_with_legal_joints() -> None:
    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
    data = mujoco.MjData(model)
    target = policy_to_scene(np.asarray((-0.35, 0.0, 0.28)))
    result = solve_franka_ik(model, data, target)
    assert result.position_error < 0.01
    assert result.converged
    assert np.isfinite(result.joint_positions).all()
    for name, value in zip(result.joint_names, result.joint_positions, strict=True):
        joint = model.joint(name)
        assert model.jnt_range[joint.id, 0] <= value <= model.jnt_range[joint.id, 1]


def test_franka_ik_reaches_task_extremes_and_receptacle() -> None:
    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
    targets = ((-0.2, -0.25, 0.04), (0.12, 0.25, 0.04), (0.30, -0.18, 0.20))
    for policy_target in targets:
        data = mujoco.MjData(model)
        result = solve_franka_ik(model, data, policy_to_scene(np.asarray(policy_target)))
        assert result.position_error < 0.015
        assert np.isfinite(result.joint_positions).all()
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -p no:cacheprovider tests/interaction_vla/test_franka.py -q
```

Expected: import failures for the transform and IK symbols.

- [ ] **Step 3: Implement the transform, result type and IK solver**

Add these public contracts to `interaction_vla/franka.py`:

```python
from dataclasses import dataclass

import mujoco
import numpy as np


POLICY_TO_SCENE_SCALE = 0.55
POLICY_TO_SCENE_OFFSET = np.asarray((0.50, 0.0, 0.228), dtype=np.float64)
ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
TCP_LOCAL_OFFSET = np.asarray((0.0, 0.0, 0.1034), dtype=np.float64)
DOWNWARD_ROTATION = np.diag((1.0, -1.0, -1.0))


@dataclass(frozen=True)
class IKResult:
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    position_error: float
    rotation_error: float
    converged: bool
    iterations: int


def policy_to_scene(value: np.ndarray) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("policy positions must be finite (..., 3) arrays")
    return points * POLICY_TO_SCENE_SCALE + POLICY_TO_SCENE_OFFSET
```

Implement `solve_franka_ik(model, data, target, initial=None, max_iterations=80, position_tolerance=0.008)` using this loop:

1. Seed arm qpos with upstream home `(0, 0, 0, -1.57079, 0, 1.57079, -0.7853)` unless `initial` is supplied.
2. Call `mj_forward`.
3. Compute the TCP world point as `hand.xpos + hand.xmat @ TCP_LOCAL_OFFSET`.
4. Compute position error and the skew-vector rotation error from `DOWNWARD_ROTATION @ current_rotation.T`.
5. Fill `(3, nv)` translational and rotational Jacobians with `mujoco.mj_jac` at the TCP point and `hand` body id.
6. Select only the seven Panda DOF columns; solve `dq = J.T @ solve(J @ J.T + 1e-4 I, weighted_error)` with rotation weight `0.25`.
7. Clip each update to `[-0.2, 0.2]`, apply `0.35` of the update, and clip every joint to `model.jnt_range`.
8. Return an `IKResult` with the final errors even if the iteration budget is exhausted.

- [ ] **Step 4: Run the focused IK tests and tune only the declared transform/solver constants**

Expected: all tests pass. If reachability fails, adjust `POLICY_TO_SCENE_SCALE`, `POLICY_TO_SCENE_OFFSET`, the neutral pose or damping while preserving a uniform scale; do not change policy-space states or task thresholds.

- [ ] **Step 5: Add validation tests and keep them GREEN**

Add tests asserting non-finite targets raise `ValueError`, an invalid initial shape raises `ValueError`, and exhausted IK returns finite legal joints with `converged=False`. Run the whole `test_franka.py` after each validation.

### Task 3: Synchronize snapshots into the complete Panda mirror

**Files:**
- Modify: `interaction_vla/franka.py`
- Modify: `interaction_vla/mujoco_env.py`
- Modify: `tests/interaction_vla/test_franka.py`
- Modify: `tests/interaction_vla/test_mujoco_env.py`

- [ ] **Step 1: Write failing mirror synchronization tests**

```python
from interaction_vla.env import KinematicTabletopEnv
from interaction_vla.franka import FrankaMirror, policy_to_scene


def test_franka_mirror_syncs_objects_fingers_and_tcp() -> None:
    backend = KinematicTabletopEnv(max_objects=5)
    snapshot = backend.reset(seed=2_140_049, object_count=4, layout_mode="crowded")
    mirror = FrankaMirror(max_objects=5)
    status = mirror.sync(snapshot)
    assert status.position_error < 0.015
    for index, entity in enumerate(snapshot.objects):
        body = mirror.data.body(f"object_{index}")
        np.testing.assert_allclose(body.xpos, policy_to_scene(entity.position), atol=1e-6)
    assert mirror.data.qpos[mirror.finger_qpos_addresses].tolist() == [0.04, 0.04]
    hidden = mirror.data.body("object_4").xpos
    assert hidden[2] < -1.0


def test_franka_mirror_does_not_mutate_authoritative_task_state() -> None:
    backend = KinematicTabletopEnv(max_objects=5)
    before = backend.reset(seed=42, object_count=4, layout_mode="crowded")
    mirror = FrankaMirror(max_objects=5)
    mirror.sync(before)
    after = backend.snapshot()
    assert after.held_object == before.held_object
    assert after.contacts == before.contacts
    assert after.support_relations == before.support_relations
    for actual, expected in zip(
        (after.gripper, *after.objects, after.receptacle, after.support),
        (before.gripper, *before.objects, before.receptacle, before.support),
        strict=True,
    ):
        assert actual.name == expected.name
        assert actual.target == expected.target
        np.testing.assert_array_equal(actual.position, expected.position)
        np.testing.assert_array_equal(actual.orientation, expected.orientation)
```

Update existing MuJoCo tests to assert `link0`, `hand`, both finger joints and four cameras exist instead of asserting the old `gripper_geom` sphere.

- [ ] **Step 2: Run the focused tests and verify RED**

Expected: `FrankaMirror` is missing and the old `MujocoTabletopEnv` still loads the primitive XML.

- [ ] **Step 3: Implement `FrankaMirror`**

Add:

```python
@dataclass(frozen=True)
class MirrorStatus:
    converged: bool
    position_error: float
    rotation_error: float
    iterations: int


class FrankaMirror:
    def __init__(self, max_objects: int = 5) -> None:
        if max_objects != 5:
            raise ValueError("the Franka scene contains exactly five object slots")
        self.model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.arm_qpos_addresses = np.asarray([
            self.model.jnt_qposadr[self.model.joint(name).id] for name in ARM_JOINT_NAMES
        ])
        self.finger_qpos_addresses = np.asarray([
            self.model.jnt_qposadr[self.model.joint(name).id] for name in FINGER_JOINT_NAMES
        ])
        self.last_legal_arm_qpos: np.ndarray | None = None

    def sync(self, snapshot: SceneSnapshot) -> MirrorStatus:
        result = solve_franka_ik(
            self.model,
            self.data,
            policy_to_scene(snapshot.gripper.position),
            initial=self.last_legal_arm_qpos,
        )
        if np.isfinite(result.joint_positions).all():
            self.last_legal_arm_qpos = result.joint_positions.copy()
        self.data.qpos[self.arm_qpos_addresses] = self.last_legal_arm_qpos
        finger = float(np.clip(snapshot.gripper.gripper_open, 0.0, 1.0)) * 0.04
        self.data.qpos[self.finger_qpos_addresses] = finger
        for index in range(5):
            joint = self.model.joint(f"object_{index}_joint")
            address = int(self.model.jnt_qposadr[joint.id])
            if index < len(snapshot.objects):
                entity = snapshot.objects[index]
                quaternion = np.asarray(entity.orientation, dtype=np.float64)
                quaternion /= np.linalg.norm(quaternion)
                self.data.qpos[address : address + 3] = policy_to_scene(entity.position)
                self.data.qpos[address + 3 : address + 7] = quaternion
            else:
                self.data.qpos[address : address + 3] = (0.0, 0.0, -2.0 - index)
                self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)
        hand = self.data.body("hand")
        rotation = hand.xmat.reshape(3, 3).copy()
        tcp_position = hand.xpos + rotation @ TCP_LOCAL_OFFSET
        mocap_id = int(self.model.body("wrist_camera_rig").mocapid[0])
        self.data.mocap_pos[mocap_id] = tcp_position
        wrist_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(wrist_quaternion, rotation.ravel())
        self.data.mocap_quat[mocap_id] = wrist_quaternion
        mujoco.mj_forward(self.model, self.data)
        return MirrorStatus(result.converged, result.position_error, result.rotation_error, result.iterations)
```

- [ ] **Step 4: Delegate `MujocoTabletopEnv` visual state to the mirror**

Keep its constructor/reset/step/render API, but replace `_xml` and primitive `_sync` with:

```python
self.backend = KinematicTabletopEnv(...)
self.mirror = FrankaMirror(max_objects=max_objects)
self.model = self.mirror.model
self.data = self.mirror.data
self.mirror_status = None

def _sync(self, snapshot: SceneSnapshot) -> None:
    self.mirror_status = self.mirror.sync(snapshot)
```

Do not change calls into `self.backend`. This is the main experiment-isolation boundary.

- [ ] **Step 5: Run mirror, environment and policy regression tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider \
  tests/interaction_vla/test_franka.py \
  tests/interaction_vla/test_mujoco_env.py \
  tests/interaction_vla/test_env.py \
  tests/interaction_vla/test_visualize.py -q
```

Expected: all non-renderer tests pass; CoreGraphics-dependent tests may skip only under the Codex sandbox.

### Task 4: Render and compose the four camera grid

**Files:**
- Modify: `interaction_vla/visualize.py`
- Modify: `tests/interaction_vla/test_visualize.py`

- [ ] **Step 1: Write failing pure frame-composition tests**

```python
from interaction_vla.visualize import (
    CAMERA_GRID,
    compose_camera_grid,
    dashboard_viewports,
)


def test_camera_grid_order_is_agent_wrist_side_top() -> None:
    assert CAMERA_GRID == (
        ("agentview", "Agent View"),
        ("wristview", "Wrist / Egocentric"),
        ("sideview", "Side View"),
        ("topview", "Top View"),
    )


def test_compose_camera_grid_makes_labeled_two_by_two_image() -> None:
    frames = {name: np.full((60, 80, 3), index * 50, dtype=np.uint8)
              for index, (name, _label) in enumerate(CAMERA_GRID)}
    image = compose_camera_grid(frames, status_label="Expert · step 4 · running · IK ok")
    assert image.size == (160, 144)  # 24px global header + two 60px rows


def test_dashboard_viewports_cover_window_without_overlap() -> None:
    values = dashboard_viewports(1280, 960)
    assert values == {
        "agentview": (0, 480, 640, 480),
        "wristview": (640, 480, 640, 480),
        "sideview": (0, 0, 640, 480),
        "topview": (640, 0, 640, 480),
    }
```

- [ ] **Step 2: Run the tests and verify RED**

Expected: the new constants/functions do not exist.

- [ ] **Step 3: Implement the pure grid contracts**

Add `CAMERA_GRID`, `dashboard_viewports(width, height)` with positive/even dimension validation, and `compose_camera_grid(frames, status_label)` using Pillow. Each panel must have identical RGB shape; draw a 20px panel label inside its upper-left corner and a 24px global status header.

- [ ] **Step 4: Add a reusable four-camera renderer**

```python
class FourCameraRenderer:
    def __init__(self, env: MujocoTabletopEnv, *, width: int, height: int) -> None:
        self.env = env
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)

    def render(self) -> dict[str, np.ndarray]:
        return {
            name: render_rgb(self.env, self.renderer, camera=name)
            for name, _label in CAMERA_GRID
        }

    def close(self) -> None:
        self.renderer.close()
```

Add a real-render test guarded by the existing `CODEX_SANDBOX` skip. Assert all four frames have the configured shape, nonzero variance, and pairwise unequal pixel arrays.

- [ ] **Step 5: Update semantic coloring for the Panda scene**

Keep target/anchor/other/inactive object colors and receptacle color. Remove the lookup of deleted `gripper_geom`; do not recolor official Panda mesh materials. Update the existing color test accordingly.

- [ ] **Step 6: Run the focused visualization tests**

Expected: pure composition and semantic tests pass; only actual renderer tests skip inside the sandbox.

### Task 5: Add the live 2×2 GLFW dashboard and preserve native mode

**Files:**
- Modify: `interaction_vla/visualize.py`
- Modify: `tests/interaction_vla/test_visualize.py`

- [ ] **Step 1: Write failing CLI and viewport-status tests**

Extend parser tests to require:

```python
args = build_parser().parse_args([
    "viewer", "--controller", "expert", "--view", "dashboard"
])
assert args.view == "dashboard"

native = build_parser().parse_args([
    "viewer", "--controller", "expert", "--view", "native"
])
assert native.view == "native"
```

Add a status-label test asserting that a converged mirror prints `IK ok` and a non-converged one prints `IK limited`.

- [ ] **Step 2: Run and verify RED**

Expected: `--view` is rejected and dashboard dispatch is absent.

- [ ] **Step 3: Implement `run_camera_dashboard` with one MuJoCo state**

Lazy-import `glfw` inside the function. Initialize one 1280×960 window and one each of `MjvScene`, `MjvOption`, `MjrContext`. For every frame:

```python
for camera_name, label in CAMERA_GRID:
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = session.env.model.camera(camera_name).id
    mujoco.mjv_updateScene(
        session.env.model,
        session.env.data,
        option,
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    x, y, width, height = dashboard_viewports(*glfw.get_framebuffer_size(window))[camera_name]
    viewport = mujoco.MjrRect(x, y, width, height)
    mujoco.mjr_render(viewport, scene, context)
    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        viewport,
        label,
        status_label(session),
        context,
    )
```

Poll events, advance the one session once per loop, swap buffers and pace to `fps`. On exit, free the render context and terminate GLFW in `finally`. Pressing Escape or closing the window ends cleanly. Do not create four environments or four data objects.

- [ ] **Step 4: Make dashboard the default and retain native viewer**

Add `--view {dashboard,native}` with default `dashboard`. Dispatch `dashboard` to `run_camera_dashboard` and `native` to the existing `run_native_viewer`. Keep the existing macOS error message instructing use of `.venv/bin/mjpython`.

- [ ] **Step 5: Verify help and non-GUI tests**

Run parser tests and both help commands with `PYTHONDONTWRITEBYTECODE=1`. Expected: help shows `--view {dashboard,native}` and all focused tests pass.

### Task 6: Export a single-controller four-view GIF and update documentation

**Files:**
- Modify: `interaction_vla/visualize.py`
- Modify: `tests/interaction_vla/test_visualize.py`
- Modify: `README.md`
- Create: `docs/media/franka_four_view_crowded.gif`

- [ ] **Step 1: Write failing export tests**

Add parser coverage for:

```python
args = build_parser().parse_args([
    "export-rollout-gif",
    "--controller", "expert",
    "--output", "rollout.gif",
])
assert args.command == "export-rollout-gif"
assert args.controller == "expert"
```

Add a two-frame real export test guarded by the CoreGraphics skip. It must open the result with Pillow and assert `is_animated`, `n_frames >= 2`, and output size `(2 * panel_width, 24 + 2 * panel_height)`.

- [ ] **Step 2: Run and verify RED**

Expected: `export-rollout-gif` is not a recognized command.

- [ ] **Step 3: Implement `export_rollout_gif`**

Create one `VisualizationSession`, one `FourCameraRenderer`, render the initial grid, then advance and render until termination. Use existing `save_animated_gif`; close the renderer in `finally`. Support `expert`, `flat` and `graph`; learned controllers require their matching checkpoint. Reuse seed/layout/object-count/max-steps/fps/width/height/output arguments.

Extend the existing comparison exporter with `--camera {agentview,wristview}` and pass it to `render_rgb`; do not change default `agentview` behavior or overwrite the existing example unexpectedly.

- [ ] **Step 4: Update README commands and explanations**

Document these exact flows:

```bash
.venv/bin/python -m interaction_vla.macos_mjpython

.venv/bin/mjpython -m interaction_vla.visualize viewer \
  --controller expert --view dashboard \
  --layout crowded --object-count 4 --seed 2140049

.venv/bin/mjpython -m interaction_vla.visualize viewer \
  --controller graph --view native \
  --checkpoint outputs/interaction_vla/recovery/graph/seed_0/checkpoint.pt \
  --layout crowded --object-count 4 --seed 2140049

.venv/bin/python -m interaction_vla.visualize export-rollout-gif \
  --controller expert --layout crowded --object-count 4 --seed 2140049 \
  --width 320 --height 240 --fps 12 \
  --output docs/media/franka_four_view_crowded.gif
```

State explicitly that agent/wrist are camera outputs but not yet policy inputs, and that Franka mirror changes neither Graph/Flat state nor experiment metrics.

- [ ] **Step 5: Generate and validate the real example GIF outside the sandbox**

Run the documented export in a normal macOS graphical session. Validate with Pillow: animated, at least two frames, exact expected dimensions, nonzero file size. Embed it near the viewer section while retaining `flat_vs_graph_crowded.gif`.

### Task 7: Full regression and real Mac acceptance

**Files:**
- Verify all changed files and preserved artifacts

- [ ] **Step 1: Run the complete no-cache test suite**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -p no:cacheprovider tests/interaction_vla -q
```

Expected: every non-environmental test passes. The only allowed skips are the existing MPS check and CoreGraphics renderer tests under the sandbox.

- [ ] **Step 2: Compile into a temporary cache location**

```bash
PYTHONPYCACHEPREFIX=/private/tmp/interaction_vla_pycache \
  .venv/bin/python -m compileall -q interaction_vla tests/interaction_vla
```

Expected: exit code 0 and no project-local compile cache required.

- [ ] **Step 3: Verify experiment isolation on the fixed rollout**

Run the existing evaluation/session test that records expert and learned termination reasons for seed `2140049`. Assert the authoritative backend outcomes remain Flat timeout at 120 steps and Graph success at 29 steps for recovery seed 0. If numerical inference produces a platform-tolerance difference, compare the full snapshot/action sequence before changing any expected value.

- [ ] **Step 4: Run real dashboard acceptance through `mjpython`**

```bash
.venv/bin/mjpython -m interaction_vla.visualize viewer \
  --controller expert --view dashboard \
  --layout crowded --object-count 4 --seed 2140049 --fps 60
```

Acceptance: one window shows four labeled panels; Agent/Side/Top show the full Panda; Wrist follows the hand; fingers open/close; the selected object follows the gripper; the rollout terminates without a crash.

- [ ] **Step 5: Run native-view acceptance**

```bash
.venv/bin/mjpython -m interaction_vla.visualize viewer \
  --controller expert --view native \
  --layout crowded --object-count 4 --seed 2140049 --fps 60
```

Acceptance: the free camera shows official Panda meshes and articulated joints rather than the old sphere.

- [ ] **Step 6: Audit preserved and new artifacts**

Check that all prior checkpoints/reports and `docs/media/flat_vs_graph_crowded.gif` still exist, the new four-view GIF is readable, the pinned model license is present, and no deleted tutorial notebooks or old root `asset/` directory were restored.

- [ ] **Step 7: Record final evidence**

Report exact test counts, allowed skips, compile exit code, GIF dimensions/frame count/bytes, real dashboard/native exit codes, asset source SHA, and any IK-limited frames observed. Do not claim the Graph hypothesis was strengthened by the visual layer; report the prior numeric evidence unchanged.
