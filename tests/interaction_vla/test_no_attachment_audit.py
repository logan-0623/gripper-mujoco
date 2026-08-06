from __future__ import annotations

import inspect

import mujoco

from interaction_vla.config import PhysicsConfig
from interaction_vla.contact_physics import StableGraspTracker
from interaction_vla.franka import FRANKA_SCENE_PATH, OBJECT_NAMES
from interaction_vla.franka_controller import FrankaCartesianController
from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.physics_data import _apply_recovery_intervention
from interaction_vla.physics_data import prepare_physics_recovery_start
from interaction_vla.chunked_controller import (
    ChunkedPolicyController,
    TemporalActionEnsembler,
)
from interaction_vla.physics_expert import PhysicsScriptedExpert
from interaction_vla.physics_visualize import PhysicsVisualizationSession
from interaction_vla.teleop import TeleopController
from interaction_vla.validate_physics_expert import no_attachment_audit


def test_compiled_scene_has_no_object_equality_mocap_or_actuator() -> None:
    model = mujoco.MjModel.from_xml_path(str(FRANKA_SCENE_PATH))
    object_body_ids = {model.body(name).id for name in OBJECT_NAMES}
    object_joint_ids = {model.joint(f"{name}_joint").id for name in OBJECT_NAMES}

    assert model.neq == 1  # upstream Panda finger coupling only
    for equality_id in range(model.neq):
        assert int(model.eq_obj1id[equality_id]) not in object_body_ids | object_joint_ids
        assert int(model.eq_obj2id[equality_id]) not in object_body_ids | object_joint_ids
    assert all(model.body_mocapid[body_id] == -1 for body_id in object_body_ids)
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT:
            assert int(model.actuator_trnid[actuator_id, 0]) not in object_joint_ids


def test_rollout_control_sources_do_not_write_object_qpos_or_qvel() -> None:
    methods = (
        FrankaContactEnv.step,
        FrankaContactEnv.advance_intervention,
        _apply_recovery_intervention,
        FrankaCartesianController.apply_action,
        StableGraspTracker.update,
        PhysicsScriptedExpert.act,
        TeleopController.action,
        PhysicsVisualizationSession.advance,
        ChunkedPolicyController.act,
        TemporalActionEnsembler.action_for_step,
        prepare_physics_recovery_start,
    )
    for method in methods:
        source = inspect.getsource(method)
        assert ".qpos[" not in source, method.__qualname__
        assert ".qvel[" not in source, method.__qualname__


def test_gate_attachment_audit_includes_unlabelled_interventions(
    monkeypatch,
) -> None:
    def unsafe_intervention(self, action, *, substeps):
        self.data.qpos[0] = 0.0

    monkeypatch.setattr(
        FrankaContactEnv,
        "advance_intervention",
        unsafe_intervention,
    )

    assert not no_attachment_audit()


def test_gate_attachment_audit_includes_chunked_controller(monkeypatch) -> None:
    def unsafe_act(self, env):
        env.data.qpos[0] = 0.0

    monkeypatch.setattr(ChunkedPolicyController, "act", unsafe_act)

    assert not no_attachment_audit()


def _run_expert_with_contact_change(change) -> tuple[bool, bool]:
    physics = PhysicsConfig(settle_steps=100)
    env = FrankaContactEnv(
        max_steps=180,
        physics=physics,
        workspace_low=(0.25, -0.35, 0.23),
        workspace_high=(0.78, 0.35, 0.75),
        crowded_anchor_min_distance=0.055,
        crowded_anchor_max_distance=0.075,
    )
    snapshot = env.reset(seed=11, object_count=2)
    change(env)
    expert = PhysicsScriptedExpert(physics)
    expert.reset(seed=11)
    for _ in range(180):
        action = expert.act(snapshot, env.contact_diagnostics, env.grasp_state)
        transition = env.step(action)
        snapshot = transition.snapshot
        if transition.done:
            break
    return env.grasp_state.ever_stable_target, transition.reason.value == "success"


def _finger_geom_ids(env: FrankaContactEnv) -> list[int]:
    finger_body_ids = {
        env.model.body("left_finger").id,
        env.model.body("right_finger").id,
    }
    return [
        geom_id
        for geom_id in range(env.model.ngeom)
        if int(env.model.geom_bodyid[geom_id]) in finger_body_ids
    ]


def test_stable_grasp_requires_fingertip_friction_and_bilateral_collision() -> None:
    baseline_lift, baseline_success = _run_expert_with_contact_change(lambda _env: None)

    def remove_friction(env: FrankaContactEnv) -> None:
        finger_geoms = _finger_geom_ids(env)
        # Higher priority makes the zeroed fingertip parameters authoritative for
        # the contact pair instead of being mixed with the object's friction.
        env.model.geom_priority[finger_geoms] = 1
        env.model.geom_friction[finger_geoms, :] = 0.0

    def remove_left_collision(env: FrankaContactEnv) -> None:
        left_body = env.model.body("left_finger").id
        for geom_id in range(env.model.ngeom):
            if int(env.model.geom_bodyid[geom_id]) == left_body:
                env.model.geom_contype[geom_id] = 0
                env.model.geom_conaffinity[geom_id] = 0

    friction_lift, friction_success = _run_expert_with_contact_change(remove_friction)
    bilateral_lift, bilateral_success = _run_expert_with_contact_change(remove_left_collision)

    assert baseline_lift and baseline_success
    assert not friction_lift and not friction_success
    assert not bilateral_lift and not bilateral_success
