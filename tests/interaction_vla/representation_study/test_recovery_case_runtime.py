from __future__ import annotations

import numpy as np

from interaction_vla.config import PhysicsConfig
from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.physics_expert import PhysicsScriptedExpert
from interaction_vla.representation_study.rl.distributions import RecoveryCase
from interaction_vla.representation_study.rl.environment import (
    ResidualMujocoRuntime,
    prepare_interaction_start,
)


def _env() -> FrankaContactEnv:
    return FrankaContactEnv(
        max_steps=180,
        physics=PhysicsConfig(settle_steps=100),
        workspace_low=(0.25, -0.35, 0.23),
        workspace_high=(0.78, 0.35, 0.75),
        crowded_anchor_min_distance=0.055,
        crowded_anchor_max_distance=0.075,
    )


def _case(family: str) -> RecoveryCase:
    if family == "nominal":
        phase, kind, severity = "approach", "nominal", 0.0
    elif family == "perturbation":
        phase, kind, severity = "approach", "approach_offset", 0.5
    else:
        phase, kind, severity = "transport", "wrong_way_transport", 1.0
    return RecoveryCase(
        case_id=f"training:11:{family}:0:{kind}",
        partition="training",
        family=family,
        source_seed=11,
        variant_id=0,
        object_count=2,
        layout="normal",
        phase=phase,
        intervention_kind=kind,
        severity=severity,
    )


class _OracleStub:
    def encode_runtime(self, env, prepared, case) -> np.ndarray:
        del env, prepared
        return np.full(36, case.severity, dtype=np.float32)


def _runtime() -> ResidualMujocoRuntime:
    runtime = ResidualMujocoRuntime.__new__(ResidualMujocoRuntime)
    runtime.env = _env()
    runtime.expert = PhysicsScriptedExpert(runtime.env.physics)
    runtime.oracle_codec = _OracleStub()

    def finish(snapshot) -> None:
        runtime.snapshot = snapshot
        runtime.current_observation = {"task": ["test"]}
        runtime.episode_return = 0.0
        runtime.episode_length = 0
        runtime.clipped_steps = 0
        runtime.projection_scales = []

    runtime._finish_reset = finish
    return runtime


def test_reset_case_reconstructs_nominal_state() -> None:
    runtime = _runtime()
    case = _case("nominal")
    first = runtime.reset_case(case)
    first_qpos = runtime.env.data.qpos.copy()
    second = runtime.reset_case(case)
    np.testing.assert_array_equal(runtime.env.data.qpos, first_qpos)
    assert np.isfinite(first.oracle_state).all()
    assert first.oracle_state.shape == (36,)
    assert second.case_id == first.case_id


def test_recovery_start_is_reconstructible_and_counts_prefix_steps() -> None:
    case = _case("recovery")

    def prepare_once():
        env = _env()
        prepared = prepare_interaction_start(
            env, PhysicsScriptedExpert(env.physics), case=case
        )
        return env.data.qpos.copy(), env.data.qvel.copy(), prepared

    first_qpos, first_qvel, first = prepare_once()
    second_qpos, second_qvel, second = prepare_once()
    np.testing.assert_array_equal(first_qpos, second_qpos)
    np.testing.assert_array_equal(first_qvel, second_qvel)
    assert first.prefix_steps == second.prefix_steps
    assert first.prefix_steps > 0


def test_perturbation_uses_intervention_action_not_set_state() -> None:
    env = _env()
    calls = 0
    advance = env.advance_intervention

    def track(action, *, substeps):
        nonlocal calls
        calls += 1
        return advance(action, substeps=substeps)

    env.advance_intervention = track
    env.set_state = lambda *_: (_ for _ in ()).throw(
        AssertionError("direct state write")
    )
    prepared = prepare_interaction_start(
        env, PhysicsScriptedExpert(env.physics), case=_case("perturbation")
    )
    assert calls == 1
    assert prepared.prefix_steps == 0
    assert np.isfinite(env.data.qpos).all()
