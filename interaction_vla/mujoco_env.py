from __future__ import annotations

import numpy as np

from .env import EnvStep, KinematicTabletopEnv, LayoutMode
from .graph.schema import SceneSnapshot

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised only without the optional package
    raise ImportError("MujocoTabletopEnv requires the 'mujoco' package") from exc


class MujocoTabletopEnv:
    """Headless MuJoCo mirror of :class:`KinematicTabletopEnv`.

    Task transitions remain deterministic and kinematic. MuJoCo owns a primitive
    scene representation so graph experiments can inspect or render simulator
    bodies without requiring GLFW or the tutorial viewer.
    """

    def __init__(
        self,
        max_objects: int = 5,
        max_steps: int = 120,
        min_object_distance: float = 0.12,
        workspace_low: tuple[float, float, float] = (-0.45, -0.35, 0.04),
        workspace_high: tuple[float, float, float] = (0.45, 0.35, 0.55),
        crowded_anchor_min_distance: float = 0.085,
        crowded_anchor_max_distance: float = 0.105,
    ) -> None:
        self.backend = KinematicTabletopEnv(
            max_objects=max_objects,
            max_steps=max_steps,
            min_object_distance=min_object_distance,
            workspace_low=workspace_low,
            workspace_high=workspace_high,
            crowded_anchor_min_distance=crowded_anchor_min_distance,
            crowded_anchor_max_distance=crowded_anchor_max_distance,
        )
        self.max_objects = max_objects
        self.model = mujoco.MjModel.from_xml_string(self._xml(max_objects))
        self.data = mujoco.MjData(self.model)

    def reset(
        self,
        seed: int,
        object_count: int,
        target_index: int | None = None,
        layout_mode: LayoutMode | str = LayoutMode.NORMAL,
    ) -> SceneSnapshot:
        snapshot = self.backend.reset(
            seed=seed,
            object_count=object_count,
            target_index=target_index,
            layout_mode=layout_mode,
        )
        self._sync(snapshot)
        return snapshot

    def step(self, action: np.ndarray) -> EnvStep:
        result = self.backend.step(action)
        self._sync(result.snapshot)
        return result

    def snapshot(self) -> SceneSnapshot:
        return self.backend.snapshot()

    def proprioception(self) -> np.ndarray:
        return self.backend.proprioception()

    def set_gripper_position(self, position: np.ndarray) -> None:
        self.backend.set_gripper_position(position)
        self._sync(self.backend.snapshot())

    def perturb_gripper_state(
        self,
        delta: np.ndarray,
        gripper_open: float | None = None,
    ) -> SceneSnapshot:
        snapshot = self.backend.perturb_gripper_state(delta, gripper_open=gripper_open)
        self._sync(snapshot)
        return snapshot

    def render_rgb(self, width: int = 256, height: int = 256) -> np.ndarray:
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(self.data)
            return renderer.render().copy()
        finally:
            renderer.close()

    def _sync(self, snapshot: SceneSnapshot) -> None:
        self.data.mocap_pos[0] = snapshot.gripper.position
        self.data.mocap_quat[0] = snapshot.gripper.orientation
        for index in range(self.max_objects):
            joint_id = self.model.joint(f"object_{index}_joint").id
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            if index < len(snapshot.objects):
                entity = snapshot.objects[index]
                self.data.qpos[qpos_address : qpos_address + 3] = entity.position
                self.data.qpos[qpos_address + 3 : qpos_address + 7] = entity.orientation
            else:
                self.data.qpos[qpos_address : qpos_address + 3] = (0.0, 0.0, -2.0 - index)
                self.data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _xml(max_objects: int) -> str:
        colors = (
            "0.85 0.15 0.15 1",
            "0.15 0.35 0.85 1",
            "0.15 0.75 0.30 1",
            "0.90 0.65 0.10 1",
            "0.65 0.20 0.75 1",
        )
        object_bodies = []
        for index in range(max_objects):
            object_bodies.append(
                f"""
                <body name="object_{index}" pos="0 0 -2">
                  <freejoint name="object_{index}_joint"/>
                  <geom name="object_{index}_geom" type="box" size="0.04 0.04 0.04"
                        mass="0.1" rgba="{colors[index % len(colors)]}"/>
                </body>
                """
            )
        return f"""
        <mujoco model="interaction_graph_pilot">
          <option timestep="0.02" gravity="0 0 0"/>
  <visual><global offwidth="640" offheight="480"/></visual>
          <worldbody>
            <light pos="0 0 2" dir="0 0 -1"/>
            <geom name="table" type="box" pos="0 0 -0.02" size="0.5 0.4 0.02"
                  rgba="0.65 0.55 0.45 1"/>
            <geom name="receptacle" type="cylinder" pos="0.30 -0.18 0.01"
                  size="0.10 0.01" rgba="0.15 0.65 0.75 0.55"/>
            <body name="gripper" mocap="true" pos="-0.35 0 0.28">
              <geom name="gripper_geom" type="sphere" size="0.035" rgba="0.15 0.15 0.15 1"/>
            </body>
            {''.join(object_bodies)}
            <camera name="agentview" pos="0 -1.1 0.85" xyaxes="1 0 0 0 0.65 0.76"/>
          </worldbody>
        </mujoco>
        """
