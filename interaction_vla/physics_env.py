from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import mujoco
import numpy as np

from .config import PhysicsConfig
from .contact_physics import (
    ContactDiagnostics,
    ContactParser,
    GraspState,
    NonFiniteContactForceError,
    StableGraspTracker,
)
from .env import EnvStep, LayoutMode, TerminationReason
from .franka import (
    ARM_JOINT_NAMES,
    FINGER_JOINT_NAMES,
    FRANKA_SCENE_PATH,
    HOME_QPOS,
    OBJECT_NAMES,
)
from .franka_controller import ControllerDiagnostics, FrankaCartesianController
from .graph.schema import EntityState, SceneSnapshot
from .placement import (
    PlacementDiagnostics,
    receptacle_inner_half_extents,
    strict_containment,
)


TARGET_RGBA = np.asarray((0.10, 0.80, 0.25, 1.0), dtype=np.float32)
ANCHOR_RGBA = np.asarray((1.00, 0.50, 0.05, 1.0), dtype=np.float32)
OTHER_RGBA = np.asarray((0.20, 0.40, 0.90, 1.0), dtype=np.float32)
INACTIVE_RGBA = np.asarray((0.35, 0.35, 0.35, 0.25), dtype=np.float32)


@dataclass(frozen=True)
class PhysicsInterventionResult:
    snapshot: SceneSnapshot
    controller_diagnostics: ControllerDiagnostics | None
    physics_failure: str | None


class FrankaContactEnv:
    action_dim = 7
    policy_hz = 20
    object_half_size = 0.022
    table_top = 0.225

    def __init__(
        self,
        *,
        max_objects: int = 5,
        max_steps: int = 180,
        min_object_distance: float = 0.12,
        workspace_low: tuple[float, float, float] = (0.25, -0.35, 0.23),
        workspace_high: tuple[float, float, float] = (0.78, 0.35, 0.75),
        crowded_anchor_min_distance: float = 0.055,
        crowded_anchor_max_distance: float = 0.075,
        physics: PhysicsConfig | None = None,
    ) -> None:
        if not 2 <= max_objects <= len(OBJECT_NAMES):
            raise ValueError(f"max_objects must be between 2 and {len(OBJECT_NAMES)}")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not (
            2.0 * self.object_half_size
            < crowded_anchor_min_distance
            < crowded_anchor_max_distance
            < min_object_distance
        ):
            raise ValueError(
                "crowded distances must prevent overlap and remain below normal spacing"
            )
        self.max_objects = int(max_objects)
        self.max_steps = int(max_steps)
        self.min_object_distance = float(min_object_distance)
        self.crowded_anchor_min_distance = float(crowded_anchor_min_distance)
        self.crowded_anchor_max_distance = float(crowded_anchor_max_distance)
        self.workspace_low = self._vector(workspace_low, "workspace_low")
        self.workspace_high = self._vector(workspace_high, "workspace_high")
        if np.any(self.workspace_low >= self.workspace_high):
            raise ValueError("workspace_low must be below workspace_high")
        self.physics = physics or PhysicsConfig()

        self.model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
        if not np.isclose(self.model.opt.timestep, self.physics.timestep):
            raise ValueError("physics timestep does not match the compiled Franka scene")
        self.data = mujoco.MjData(self.model)
        self._object_body_ids = np.asarray(
            [self.model.body(name).id for name in OBJECT_NAMES], dtype=np.int32
        )
        self._object_geom_ids = np.asarray(
            [self.model.geom(f"{name}_geom").id for name in OBJECT_NAMES], dtype=np.int32
        )
        self._object_joint_ids = np.asarray(
            [self.model.joint(f"{name}_joint").id for name in OBJECT_NAMES], dtype=np.int32
        )
        self._object_qpos_addresses = self.model.jnt_qposadr[
            self._object_joint_ids
        ].astype(np.int32)
        self._object_dof_addresses = self.model.jnt_dofadr[
            self._object_joint_ids
        ].astype(np.int32)
        self._arm_joint_ids = np.asarray(
            [self.model.joint(name).id for name in ARM_JOINT_NAMES], dtype=np.int32
        )
        self._arm_qpos_addresses = self.model.jnt_qposadr[
            self._arm_joint_ids
        ].astype(np.int32)
        self._arm_dof_addresses = self.model.jnt_dofadr[
            self._arm_joint_ids
        ].astype(np.int32)
        self._finger_qpos_addresses = np.asarray(
            [
                self.model.jnt_qposadr[self.model.joint(name).id]
                for name in FINGER_JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self._base_body_mass = self.model.body_mass.copy()
        self._base_body_inertia = self.model.body_inertia.copy()
        self._base_geom_friction = self.model.geom_friction.copy()
        self._base_geom_contype = self.model.geom_contype.copy()
        self._base_geom_conaffinity = self.model.geom_conaffinity.copy()
        self._base_body_gravcomp = self.model.body_gravcomp.copy()
        self._base_dof_damping = self.model.dof_damping.copy()
        self._randomizable_friction_geom_ids = self._find_randomizable_friction_geoms()
        self._receptacle_inner_half_extents = receptacle_inner_half_extents(self.model)
        self.contact_parser = ContactParser(self.model)
        self.grasp_tracker = StableGraspTracker(
            required_frames=self.physics.stable_grasp_frames,
            lift_height=self.physics.stable_lift_height,
        )
        self._initialized = False

    def reset(
        self,
        seed: int,
        object_count: int,
        target_index: int | None = None,
        layout_mode: LayoutMode | str = LayoutMode.NORMAL,
    ) -> SceneSnapshot:
        if object_count < 2 or object_count > self.max_objects:
            raise ValueError(f"object_count must be between 2 and {self.max_objects}")
        layout_mode = LayoutMode(layout_mode)
        target_rng = np.random.default_rng(
            np.random.SeedSequence((int(seed), 0x54415247))
        )
        if target_index is None:
            target_index = int(target_rng.integers(0, object_count))
        if not 0 <= target_index < object_count:
            raise ValueError("target_index is outside the active object range")
        layout_rng = np.random.default_rng(
            np.random.SeedSequence((int(seed), 0x4C41594F))
        )
        physics_rng = np.random.default_rng(
            np.random.SeedSequence((int(seed), 0x50485953))
        )
        positions = self._sample_positions(
            layout_rng,
            object_count=object_count,
            target_index=target_index,
            layout_mode=layout_mode,
        )

        self._restore_model_parameters()
        self._physics_metadata = self._apply_physics_sample(physics_rng, object_count)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._arm_qpos_addresses] = HOME_QPOS
        self.data.qpos[self._finger_qpos_addresses] = 0.04
        for index, address in enumerate(self._object_qpos_addresses):
            active = index < object_count
            if active:
                self.data.qpos[address : address + 3] = positions[index]
                self.model.geom_contype[self._object_geom_ids[index]] = self._base_geom_contype[
                    self._object_geom_ids[index]
                ]
                self.model.geom_conaffinity[
                    self._object_geom_ids[index]
                ] = self._base_geom_conaffinity[self._object_geom_ids[index]]
                self.model.body_gravcomp[self._object_body_ids[index]] = 0.0
            else:
                self.data.qpos[address : address + 3] = (0.0, 0.0, -2.0 - index)
                self.model.geom_contype[self._object_geom_ids[index]] = 0
                self.model.geom_conaffinity[self._object_geom_ids[index]] = 0
                self.model.body_gravcomp[self._object_body_ids[index]] = 1.0
            self.data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.controller = FrankaCartesianController(
            self.model,
            self.data,
            self.physics,
            workspace_low=self.workspace_low,
            workspace_high=self.workspace_high,
        )
        for _ in range(self.physics.settle_steps):
            mujoco.mj_step(self.model, self.data)
        self.seed = int(seed)
        self.object_count = int(object_count)
        self.target_index = int(target_index)
        self.target_name = f"object_{self.target_index}"
        self.layout_mode = layout_mode
        self.step_count = 0
        self._placement_frames = 0
        self._consecutive_ik_limited = 0
        self.grasp_tracker.reset(target_name=self.target_name, table_top=self.table_top)
        self.contact_diagnostics = self.contact_parser.parse(self.data)
        self.grasp_state = self.grasp_tracker.update(
            self.contact_diagnostics,
            object_bottom_heights=self._object_bottom_heights(),
        )
        self.last_placement = self._placement_diagnostics(stable_frames=0)
        self.controller.reset()
        self._update_wrist_camera()
        self._apply_semantic_colors()
        self._initialized = True
        self._last_snapshot = self._build_snapshot()
        return self._last_snapshot

    def step(self, action: np.ndarray) -> EnvStep:
        self._require_initialized()
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("Cartesian action must be a finite vector with shape (7,)")
        failure = self._physics_failure()
        if failure is not None:
            return self._failure_transition(failure)

        diagnostics = self.controller.apply_action(values)
        self._consecutive_ik_limited = (
            self._consecutive_ik_limited + 1 if diagnostics.ik_limited else 0
        )
        for _ in range(self.physics.substeps):
            mujoco.mj_step(self.model, self.data)
            try:
                self.contact_diagnostics = self.contact_parser.parse(self.data)
            except NonFiniteContactForceError:
                self.step_count += 1
                return self._failure_transition(
                    "non_finite_contact_force",
                    diagnostics=diagnostics,
                    refresh_snapshot=False,
                )
            self.grasp_state = self.grasp_tracker.update(
                self.contact_diagnostics,
                object_bottom_heights=self._object_bottom_heights(),
            )
            self._update_placement_frames()
        self.step_count += 1
        self._update_wrist_camera()

        failure = self._physics_failure()
        if failure is None and self._consecutive_ik_limited >= 20:
            failure = "ik_limited"
        if failure is not None:
            return self._failure_transition(failure, diagnostics=diagnostics)

        reason = TerminationReason.RUNNING
        done = False
        if self.grasp_state.ever_stable_wrong_object:
            reason, done = TerminationReason.WRONG_OBJECT, True
        elif self.grasp_state.dropped_target:
            reason, done = TerminationReason.DROPPED, True
        elif self._is_success():
            reason, done = TerminationReason.SUCCESS, True
        elif self.step_count >= self.max_steps:
            reason, done = TerminationReason.TIMEOUT, True

        self._last_snapshot = self._build_snapshot()
        return EnvStep(
            snapshot=self._last_snapshot,
            done=done,
            reason=reason,
            info=self._info(diagnostics),
        )

    def advance_intervention(
        self, action: np.ndarray, *, substeps: int
    ) -> PhysicsInterventionResult:
        self._require_initialized()
        if not 1 <= int(substeps) <= self.physics.substeps:
            raise ValueError(
                "intervention substeps must be between 1 and physics.substeps"
            )
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("Cartesian action must be a finite vector with shape (7,)")
        failure = self._physics_failure()
        if failure is not None:
            return PhysicsInterventionResult(
                snapshot=self._last_snapshot,
                controller_diagnostics=None,
                physics_failure=failure,
            )

        diagnostics = self.controller.apply_action(values)
        for _ in range(int(substeps)):
            mujoco.mj_step(self.model, self.data)
            failure = self._physics_failure()
            if failure is not None:
                return PhysicsInterventionResult(
                    snapshot=self._last_snapshot,
                    controller_diagnostics=diagnostics,
                    physics_failure=failure,
                )
            try:
                self.contact_diagnostics = self.contact_parser.parse(self.data)
            except NonFiniteContactForceError:
                return PhysicsInterventionResult(
                    snapshot=self._last_snapshot,
                    controller_diagnostics=diagnostics,
                    physics_failure="non_finite_contact_force",
                )
            self.grasp_state = self.grasp_tracker.update(
                self.contact_diagnostics,
                object_bottom_heights=self._object_bottom_heights(),
            )
            self._update_placement_frames()
        self._update_wrist_camera()
        self._last_snapshot = self._build_snapshot()
        return PhysicsInterventionResult(
            snapshot=self._last_snapshot,
            controller_diagnostics=diagnostics,
            physics_failure=None,
        )

    def snapshot(self) -> SceneSnapshot:
        self._require_initialized()
        return self._build_snapshot()

    def proprioception(self) -> np.ndarray:
        self._require_initialized()
        tcp_position, tcp_rotation = self.controller.tcp_pose()
        tcp_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(tcp_quaternion, tcp_rotation.reshape(-1))
        linear_velocity, angular_velocity = self._tcp_velocity()
        finger_qpos = self.data.qpos[self._finger_qpos_addresses]
        arm_delta = self.data.qpos[self._arm_qpos_addresses] - HOME_QPOS
        gripper_open = np.asarray((float(np.mean(finger_qpos) >= 0.02),))
        result = np.concatenate(
            (
                tcp_position,
                tcp_quaternion,
                linear_velocity,
                angular_velocity,
                finger_qpos,
                arm_delta,
                gripper_open,
            )
        ).astype(np.float32)
        assert result.shape == (23,)
        return result

    def physics_metadata(self) -> dict[str, Any]:
        self._require_initialized()
        return json.loads(json.dumps(self._physics_metadata, sort_keys=True))

    def _restore_model_parameters(self) -> None:
        self.model.body_mass[:] = self._base_body_mass
        self.model.body_inertia[:] = self._base_body_inertia
        self.model.geom_friction[:] = self._base_geom_friction
        self.model.geom_contype[:] = self._base_geom_contype
        self.model.geom_conaffinity[:] = self._base_geom_conaffinity
        self.model.body_gravcomp[:] = self._base_body_gravcomp
        self.model.dof_damping[:] = self._base_dof_damping

    def _apply_physics_sample(
        self, rng: np.random.Generator, object_count: int
    ) -> dict[str, Any]:
        randomization = self.physics.randomization
        if randomization.enabled:
            mass_scales = rng.uniform(
                *randomization.object_mass_scale, size=object_count
            )
            friction_scale = float(rng.uniform(*randomization.friction_scale))
            damping_scale = float(rng.uniform(*randomization.joint_damping_scale))
        else:
            mass_scales = np.ones(object_count, dtype=np.float64)
            friction_scale = 1.0
            damping_scale = 1.0
        for index, scale in enumerate(mass_scales):
            body_id = self._object_body_ids[index]
            self.model.body_mass[body_id] = self._base_body_mass[body_id] * scale
            self.model.body_inertia[body_id] = self._base_body_inertia[body_id] * scale
        self.model.geom_friction[self._randomizable_friction_geom_ids] = (
            self._base_geom_friction[self._randomizable_friction_geom_ids]
            * friction_scale
        )
        self.model.dof_damping[self._arm_dof_addresses] = (
            self._base_dof_damping[self._arm_dof_addresses] * damping_scale
        )
        metadata: dict[str, Any] = {
            "enabled": randomization.enabled,
            "object_mass_scales": [float(value) for value in mass_scales],
            "friction_scale": friction_scale,
            "joint_damping_scale": damping_scale,
            "timestep": self.physics.timestep,
            "policy_hz": self.physics.policy_hz,
            "substeps": self.physics.substeps,
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        metadata["physics_hash"] = hashlib.sha256(encoded).hexdigest()
        return metadata

    def _sample_positions(
        self,
        rng: np.random.Generator,
        *,
        object_count: int,
        target_index: int,
        layout_mode: LayoutMode,
    ) -> np.ndarray:
        low = np.asarray((0.32, -0.20), dtype=np.float64)
        high = np.asarray((0.60, 0.20), dtype=np.float64)
        receptacle_xy = np.asarray((0.67, -0.12), dtype=np.float64)

        def valid(candidate: np.ndarray, existing: list[np.ndarray], distance: float) -> bool:
            return bool(
                np.linalg.norm(candidate - receptacle_xy) >= 0.12
                and all(np.linalg.norm(candidate - value) >= distance for value in existing)
            )

        positions: dict[int, np.ndarray] = {}
        if layout_mode is LayoutMode.CROWDED:
            distractors = [index for index in range(object_count) if index != target_index]
            anchor_index = int(rng.choice(distractors))
            for _ in range(5000):
                target = rng.uniform(low, high)
                radius = float(
                    rng.uniform(
                        self.crowded_anchor_min_distance,
                        self.crowded_anchor_max_distance,
                    )
                )
                angle = float(rng.uniform(-np.pi, np.pi))
                anchor = target + radius * np.asarray((np.cos(angle), np.sin(angle)))
                if (
                    np.all(anchor >= low)
                    and np.all(anchor <= high)
                    and valid(target, [], 0.0)
                    and valid(anchor, [], 0.0)
                ):
                    positions[target_index] = target
                    positions[anchor_index] = anchor
                    break
            else:
                raise RuntimeError("could not sample a crowded target pair")

        for index in range(object_count):
            if index in positions:
                continue
            for _ in range(5000):
                candidate = rng.uniform(low, high)
                if valid(candidate, list(positions.values()), self.min_object_distance):
                    positions[index] = candidate
                    break
            else:
                raise RuntimeError(f"could not sample {object_count} physical objects")
        height = self.table_top + self.object_half_size + 0.003
        return np.asarray(
            [(*positions[index], height) for index in range(object_count)],
            dtype=np.float64,
        )

    def _build_snapshot(self) -> SceneSnapshot:
        contacts = self.contact_parser.parse(
            self.data, stable_object=self.grasp_state.stable_object
        )
        self.contact_diagnostics = contacts
        objects = tuple(self._object_entity(index) for index in range(self.object_count))
        interaction_contacts = frozenset(
            signal.key for signal in contacts.interactions if signal.contact
        )
        support_relations = frozenset(
            (signal.first, signal.second)
            if signal.first.startswith("object_")
            else (signal.second, signal.first)
            for signal in contacts.interactions
            if signal.support
        )
        tcp_position, tcp_rotation = self.controller.tcp_pose()
        tcp_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(tcp_quaternion, tcp_rotation.reshape(-1))
        linear_velocity, angular_velocity = self._tcp_velocity()
        finger_open = float(np.mean(self.data.qpos[self._finger_qpos_addresses]) >= 0.02)
        return SceneSnapshot(
            gripper=EntityState(
                name="gripper",
                entity_type="gripper",
                position=tcp_position,
                orientation=tcp_quaternion,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                size=np.asarray((0.08, 0.08, 0.10)),
                gripper_open=finger_open,
            ),
            objects=objects,
            receptacle=EntityState(
                name="receptacle",
                entity_type="receptacle",
                position=np.asarray(self.data.body("receptacle").xpos),
                orientation=np.asarray(self.data.body("receptacle").xquat),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                size=np.asarray((0.13, 0.13, 0.056)),
            ),
            support=EntityState(
                name="table",
                entity_type="support",
                position=np.asarray((0.5, 0.0, 0.205)),
                orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                size=np.asarray((0.60, 0.48, 0.04)),
            ),
            contacts=interaction_contacts,
            held_object=self.grasp_state.stable_object,
            support_relations=support_relations,
            interactions=contacts.interactions,
        )

    def _object_entity(self, index: int) -> EntityState:
        name = f"object_{index}"
        body = self.data.body(name)
        dof_address = self._object_dof_addresses[index]
        return EntityState(
            name=name,
            entity_type="object",
            position=np.asarray(body.xpos),
            orientation=np.asarray(body.xquat),
            linear_velocity=self.data.qvel[dof_address : dof_address + 3],
            angular_velocity=self.data.qvel[dof_address + 3 : dof_address + 6],
            size=np.asarray((0.044, 0.044, 0.044)),
            movable=True,
            target=index == self.target_index,
        )

    def _tcp_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.model.body("hand").id,
            velocity,
            0,
        )
        angular = velocity[:3]
        tcp_position, _ = self.controller.tcp_pose()
        offset = tcp_position - np.asarray(self.data.body("hand").xpos)
        linear = velocity[3:] + np.cross(angular, offset)
        return linear.copy(), angular.copy()

    def _object_bottom_heights(self) -> dict[str, float]:
        return {
            f"object_{index}": float(self.data.body(f"object_{index}").xpos[2])
            - self.object_half_size
            for index in range(self.object_count)
        }

    def _update_placement_frames(self) -> None:
        target_velocity = self.data.qvel[
            self._object_dof_addresses[self.target_index] :
            self._object_dof_addresses[self.target_index] + 6
        ]
        candidate = self._placement_diagnostics(stable_frames=self._placement_frames)
        if (
            candidate.fully_contained
            and candidate.base_contact
            and np.linalg.norm(target_velocity[:3]) < 0.05
            and np.linalg.norm(target_velocity[3:]) < 0.05
        ):
            self._placement_frames += 1
        else:
            self._placement_frames = 0
        self.last_placement = self._placement_diagnostics(
            stable_frames=self._placement_frames
        )

    def _placement_diagnostics(self, *, stable_frames: int) -> PlacementDiagnostics:
        receptacle_body = self.data.body("receptacle")
        target_body = self.data.body(self.target_name)
        receptacle_rotation = np.asarray(receptacle_body.xmat).reshape(3, 3)
        target_rotation = np.asarray(target_body.xmat).reshape(3, 3)
        local_center = receptacle_rotation.T @ (
            np.asarray(target_body.xpos) - np.asarray(receptacle_body.xpos)
        )
        relative_rotation = receptacle_rotation.T @ target_rotation
        object_half_extents = np.asarray(
            self.model.geom(f"{self.target_name}_geom").size,
            dtype=np.float64,
        )
        containment = strict_containment(
            local_center=local_center,
            relative_rotation=relative_rotation,
            object_half_extents=object_half_extents,
            inner_half_extents=self._receptacle_inner_half_extents,
        )
        base_contact = (
            self.target_name in self.contact_diagnostics.object_receptacle_base
        )
        wall_contact = (
            self.target_name in self.contact_diagnostics.object_receptacle_wall
        )
        return PlacementDiagnostics(
            local_center=containment.local_center,
            projected_half_extents=containment.projected_half_extents,
            containment_margin=containment.containment_margin,
            fully_contained=containment.fully_contained,
            base_contact=base_contact,
            wall_contact=wall_contact,
            wall_only_contact=wall_contact and not base_contact,
            stable_frames=int(stable_frames),
            strict_stable=bool(
                stable_frames >= 10
                and containment.fully_contained
                and base_contact
            ),
        )

    def _is_success(self) -> bool:
        finger_open = np.mean(self.data.qpos[self._finger_qpos_addresses]) >= 0.03
        tcp_position, _ = self.controller.tcp_pose()
        target_z = float(self.data.body(self.target_name).xpos[2])
        return bool(
            self.last_placement.strict_stable
            and finger_open
            and tcp_position[2] >= target_z + 0.08
        )

    def _physics_failure(self) -> str | None:
        arrays = (self.data.qpos, self.data.qvel, self.data.ctrl)
        if not all(np.isfinite(array).all() for array in arrays):
            return "non_finite_state"
        if np.max(np.abs(self.data.qvel)) > 100.0:
            return "excessive_velocity"
        if any(
            float(self.data.contact[index].dist) < -0.015
            for index in range(self.data.ncon)
        ):
            return "severe_penetration"
        for index in range(self.object_count):
            position = np.asarray(self.data.body(f"object_{index}").xpos)
            if (
                position[0] < 0.0
                or position[0] > 1.0
                or abs(position[1]) > 0.6
                or position[2] < -0.5
                or position[2] > 1.5
            ):
                return "object_out_of_bounds"
        return None

    def _failure_transition(
        self,
        failure: str,
        *,
        diagnostics: ControllerDiagnostics | None = None,
        refresh_snapshot: bool = True,
    ) -> EnvStep:
        finite_state = bool(
            np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        )
        if finite_state and refresh_snapshot:
            self._last_snapshot = self._build_snapshot()
        info = self._info(diagnostics)
        info["physics_failure"] = failure
        info["failure_qpos"] = self.data.qpos.copy()
        info["failure_qvel"] = self.data.qvel.copy()
        return EnvStep(
            snapshot=self._last_snapshot,
            done=True,
            reason=TerminationReason.PHYSICS_FAILURE,
            info=info,
        )

    def _info(self, diagnostics: ControllerDiagnostics | None) -> dict[str, object]:
        interactions = self.contact_diagnostics.interactions
        return {
            "step_count": self.step_count,
            "physics_substeps": self.physics.substeps,
            "target_index": self.target_index,
            "left_contact": ",".join(sorted(self.contact_diagnostics.left_objects)),
            "right_contact": ",".join(sorted(self.contact_diagnostics.right_objects)),
            "bilateral_object": self.grasp_state.bilateral_object or "",
            "stable_object": self.grasp_state.stable_object or "",
            "stable_frames": self.grasp_state.stable_frames,
            "ever_stable_target": self.grasp_state.ever_stable_target,
            "ever_stable_wrong_object": self.grasp_state.ever_stable_wrong_object,
            "ever_bilateral_contact": self.grasp_state.ever_bilateral_contact,
            "wrong_stable_object": self.grasp_state.wrong_stable_object or "",
            "drop_count": int(self.grasp_state.dropped_target),
            "strict_placement": self.last_placement.strict_stable,
            "stable_placement": self.last_placement.strict_stable,
            "wall_only_receptacle_contact": self.last_placement.wall_only_contact,
            "containment_margin_x": float(
                self.last_placement.containment_margin[0]
            ),
            "containment_margin_y": float(
                self.last_placement.containment_margin[1]
            ),
            "target_receptacle_local_position": self.last_placement.local_center.copy(),
            "normal_force": float(sum(signal.normal_force for signal in interactions)),
            "tangential_force": float(
                sum(signal.tangential_force for signal in interactions)
            ),
            "ik_limited": False if diagnostics is None else diagnostics.ik_limited,
            "ik_position_error": 0.0 if diagnostics is None else diagnostics.position_error,
            "ik_orientation_error": (
                0.0 if diagnostics is None else diagnostics.orientation_error
            ),
            "physics_sample_hash": self._physics_metadata["physics_hash"],
        }

    def _update_wrist_camera(self) -> None:
        mocap_id = int(self.model.body_mocapid[self.model.body("wrist_camera_rig").id])
        if mocap_id < 0:
            raise RuntimeError("wrist_camera_rig must be a mocap body")
        mujoco.mj_fwdPosition(self.model, self.data)
        position, rotation = self.controller.tcp_pose()
        quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
        self.data.mocap_pos[mocap_id] = position
        self.data.mocap_quat[mocap_id] = quaternion
        mujoco.mj_fwdPosition(self.model, self.data)

    def _apply_semantic_colors(self) -> None:
        active_names = {f"object_{index}" for index in range(self.object_count)}
        target_position = np.asarray(self.data.body(self.target_name).xpos)
        distractors = [
            name for name in active_names if name != self.target_name
        ]
        anchor = min(
            distractors,
            key=lambda name: np.linalg.norm(
                np.asarray(self.data.body(name).xpos[:2]) - target_position[:2]
            ),
        )
        for index, name in enumerate(OBJECT_NAMES):
            color = INACTIVE_RGBA
            if name in active_names:
                color = TARGET_RGBA if name == self.target_name else OTHER_RGBA
            if name == anchor:
                color = ANCHOR_RGBA
            self.model.geom_rgba[self._object_geom_ids[index]] = color

    def _find_randomizable_friction_geoms(self) -> np.ndarray:
        selected: list[int] = [self.model.geom("table").id]
        selected.extend(int(value) for value in self._object_geom_ids)
        finger_bodies = {
            self.model.body("left_finger").id,
            self.model.body("right_finger").id,
        }
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id in finger_bodies:
                selected.append(geom_id)
        return np.asarray(sorted(set(selected)), dtype=np.int32)

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("call reset before using the environment")

    @staticmethod
    def _vector(values: tuple[float, float, float], name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite 3D vector")
        return result
