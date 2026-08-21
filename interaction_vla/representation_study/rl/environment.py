from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from interaction_vla.env import LayoutMode, TerminationReason
from interaction_vla.lerobot_bridge.capture import DualViewCapture
from interaction_vla.lerobot_bridge.codecs import EndEffectorStateCodec, LocalCartesianActionCodec
from interaction_vla.lerobot_bridge.config import BridgeConfig
from interaction_vla.lerobot_bridge.rollout import BinaryGripperHysteresis, policy_observation
from interaction_vla.physics_action_safety import project_cartesian_action
from interaction_vla.physics_data import (
    PhysicsRecoveryRejected,
    prepare_physics_recovery_start,
)
from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.physics_expert import PhysicsExpertPhase, PhysicsScriptedExpert
from interaction_vla.physics_recovery import (
    PhysicsRecoveryKind,
    PhysicsRecoverySpec,
    make_physics_recovery_spec,
)
from interaction_vla.graph.schema import SceneSnapshot

from ..backends.lerobot import LeRobotPolicyBackend
from .core import combine_residual_action
from .distributions import RecoveryCase


RECOVERY_KIND_INDEX = {
    kind.value: index
    for index, kind in enumerate(PhysicsRecoveryKind)
    if kind is not PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE
}
PERTURBATION_TRIGGER_PHASE = {
    "approach_offset": PhysicsExpertPhase.APPROACH,
    "grasp_offset": PhysicsExpertPhase.CLOSE,
    "lift_offset": PhysicsExpertPhase.LIFT,
}


class OracleStateEncoder(Protocol):
    def encode_runtime(
        self,
        env: FrankaContactEnv,
        prepared: "PreparedInteractionStart",
        case: RecoveryCase,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class PreparedInteractionStart:
    snapshot: SceneSnapshot
    case_id: str
    family: str
    intervention_kind: str
    severity: float
    prefix_steps: int


@dataclass(frozen=True)
class InteractionReset:
    case_id: str
    observation: dict[str, object]
    oracle_state: np.ndarray


@dataclass(frozen=True)
class ResidualTransition:
    observation: dict[str, object]
    latent: torch.Tensor
    residual: np.ndarray
    base_action: np.ndarray
    executed_local_action: np.ndarray
    action_was_clipped: bool
    ik_projection_scale: float
    episode_action_clipping_rate: float
    episode_mean_ik_projection_scale: float
    reward: float
    done: bool
    reason: str
    episode_return: float
    episode_length: int


def sparse_task_reward(reason: str) -> float:
    return float(reason == TerminationReason.SUCCESS.value)


def residual_clipping(
    base_action: object,
    residual: object,
    scale: object,
    *,
    base_was_clipped: bool = False,
) -> tuple[np.ndarray, bool]:
    base = np.asarray(base_action, dtype=np.float64)
    delta = np.asarray(residual, dtype=np.float64)
    alpha = np.asarray(scale, dtype=np.float64)
    unclipped = base + alpha * delta
    local = combine_residual_action(base, delta, alpha)
    residual_was_clipped = bool(
        np.any(np.abs(local.astype(np.float64) - unclipped) > 1.0e-7)
    )
    return local, bool(base_was_clipped or residual_was_clipped)


def interaction_potential(snapshot: Any, *, in_hand: bool) -> float:
    source = snapshot.target_object.position
    destination = snapshot.receptacle.position if in_hand else snapshot.gripper.position
    return -float(np.linalg.norm(np.asarray(source) - np.asarray(destination)))


def recovery_spec_from_case(case: RecoveryCase) -> PhysicsRecoverySpec:
    try:
        kind_index = RECOVERY_KIND_INDEX[case.intervention_kind]
    except KeyError as error:
        raise ValueError(
            f"case has an unsupported recovery kind: {case.intervention_kind}"
        ) from error
    return make_physics_recovery_spec(
        case.source_seed,
        case.variant_id,
        kind_index=kind_index,
    )


def _advance_expert_to_phase(
    env: FrankaContactEnv,
    expert: PhysicsScriptedExpert,
    *,
    case: RecoveryCase,
) -> tuple[SceneSnapshot, int]:
    try:
        trigger = PERTURBATION_TRIGGER_PHASE[case.intervention_kind]
    except KeyError as error:
        raise ValueError(
            f"case has an unsupported perturbation kind: {case.intervention_kind}"
        ) from error
    snapshot = env.reset(
        seed=case.source_seed,
        object_count=case.object_count,
        layout_mode=case.layout,
    )
    expert.reset(seed=case.source_seed)
    prefix_steps = 0
    while expert.phase is not trigger:
        transition = env.step(
            expert.act(snapshot, env.contact_diagnostics, env.grasp_state)
        )
        prefix_steps += 1
        snapshot = transition.snapshot
        if transition.done:
            reason = str(getattr(transition.reason, "value", transition.reason))
            raise PhysicsRecoveryRejected(
                f"perturbation_trigger_not_reached:{reason}"
            )
    return snapshot, prefix_steps


def _apply_phase_perturbation(
    env: FrankaContactEnv,
    *,
    case: RecoveryCase,
) -> SceneSnapshot:
    rng = np.random.default_rng(
        np.random.SeedSequence((case.source_seed, case.variant_id, 0x50545632))
    )
    direction = rng.normal(size=2)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    action = np.zeros(7, dtype=np.float32)
    action[:2] = direction * min(float(case.severity), 1.0)
    result = env.advance_intervention(action, substeps=env.physics.substeps)
    if result.physics_failure is not None:
        raise PhysicsRecoveryRejected(
            f"physics_failure_during_{case.intervention_kind}:"
            f"{result.physics_failure}"
        )
    if (
        result.controller_diagnostics is not None
        and result.controller_diagnostics.ik_limited
    ):
        raise PhysicsRecoveryRejected(
            f"ik_limited_during_{case.intervention_kind}"
        )
    return result.snapshot


def prepare_interaction_start(
    env: FrankaContactEnv,
    expert: PhysicsScriptedExpert,
    *,
    case: RecoveryCase,
) -> PreparedInteractionStart:
    if case.family == "nominal":
        snapshot = env.reset(
            seed=case.source_seed,
            object_count=case.object_count,
            layout_mode=case.layout,
        )
        expert.reset(seed=case.source_seed)
        return PreparedInteractionStart(
            snapshot=snapshot,
            case_id=case.case_id,
            family=case.family,
            intervention_kind=case.intervention_kind,
            severity=case.severity,
            prefix_steps=0,
        )
    if case.family == "recovery":
        prepared = prepare_physics_recovery_start(
            env,
            expert,
            spec=recovery_spec_from_case(case),
            object_count=case.object_count,
            source_split=case.partition,
            layout_mode=case.layout,
        )
        return PreparedInteractionStart(
            snapshot=prepared.snapshot,
            case_id=case.case_id,
            family=case.family,
            intervention_kind=case.intervention_kind,
            severity=case.severity,
            prefix_steps=prepared.prefix_steps,
        )
    snapshot, prefix_steps = _advance_expert_to_phase(
        env,
        expert,
        case=case,
    )
    snapshot = _apply_phase_perturbation(env, case=case)
    return PreparedInteractionStart(
        snapshot=snapshot,
        case_id=case.case_id,
        family=case.family,
        intervention_kind=case.intervention_kind,
        severity=case.severity,
        prefix_steps=prefix_steps,
    )


def make_physics_env(config: BridgeConfig, *, max_steps: int) -> FrankaContactEnv:
    source = config.source
    return FrankaContactEnv(
        max_objects=source.max_objects,
        max_steps=max_steps,
        min_object_distance=source.environment.min_object_distance,
        workspace_low=source.environment.workspace_low,
        workspace_high=source.environment.workspace_high,
        crowded_anchor_min_distance=source.environment.crowded_anchor_min_distance,
        crowded_anchor_max_distance=source.environment.crowded_anchor_max_distance,
        physics=source.physics,
    )


class ResidualMujocoRuntime:
    """Single-environment online sampler with the exact bridge observation/action codecs."""

    def __init__(
        self,
        *,
        bridge: BridgeConfig,
        backend: LeRobotPolicyBackend,
        tap_id: str,
        residual_scale: tuple[float, ...],
        max_steps: int,
        object_counts: tuple[int, ...],
        layouts: tuple[str, ...],
        seed: int,
        reward_mode: str,
        progress_reward_scale: float,
        oracle_codec: OracleStateEncoder | None = None,
    ) -> None:
        self.bridge = bridge
        self.backend = backend
        self.tap_id = tap_id
        self.residual_scale = np.asarray(residual_scale, dtype=np.float32)
        self.base_action_was_clipped = False
        self.max_steps = int(max_steps)
        self.object_counts = tuple(object_counts)
        self.layouts = tuple(layouts)
        self.rng = np.random.default_rng(seed)
        self.reward_mode = reward_mode
        self.progress_reward_scale = float(progress_reward_scale)
        self.oracle_codec = oracle_codec
        self.env = make_physics_env(bridge, max_steps=max_steps)
        self.expert = PhysicsScriptedExpert(self.env.physics)
        self.capture = DualViewCapture(
            self.env.model,
            width=bridge.dataset.image_size[1],
            height=bridge.dataset.image_size[0],
        )
        self.gripper = BinaryGripperHysteresis(
            close_threshold=0.4, open_threshold=0.6, initially_open=True
        )
        self.snapshot: Any | None = None
        self.current_observation: dict[str, object] | None = None
        self.episode_return = 0.0
        self.episode_length = 0
        self.previous_potential = 0.0
        self.clipped_steps = 0
        self.projection_scales: list[float] = []
        self.reset()

    def _observation(self) -> dict[str, object]:
        if self.snapshot is None:
            raise RuntimeError("residual environment is not reset")
        frame = self.capture.capture(self.env, include_teacher=False)
        state = EndEffectorStateCodec.encode_snapshot(
            self.snapshot, self.env.proprioception()
        )
        values = policy_observation(
            agent_rgb=frame.views["agent"].rgb,
            wrist_rgb=frame.views["wrist"].rgb,
            state=state,
        )
        return {
            **{key: value.unsqueeze(0) for key, value in values.items()},
            "task": [self.bridge.dataset.task],
        }

    def _finish_reset(self, snapshot: SceneSnapshot) -> None:
        self.snapshot = snapshot
        self.gripper.reset()
        if self.backend.policy is not None and hasattr(self.backend.policy, "reset"):
            self.backend.policy.reset()
        self.episode_return = 0.0
        self.episode_length = 0
        self.previous_potential = interaction_potential(self.snapshot, in_hand=False)
        self.clipped_steps = 0
        self.projection_scales = []
        self.current_observation = self._observation()

    def reset(
        self,
        *,
        case_seed: int | None = None,
        object_count: int | None = None,
        layout: str | None = None,
    ) -> dict[str, object]:
        seed = int(self.rng.integers(0, 2**32 - 1)) if case_seed is None else int(case_seed)
        count = (
            int(self.object_counts[int(self.rng.integers(0, len(self.object_counts)))])
            if object_count is None
            else int(object_count)
        )
        layout_mode = LayoutMode(
            self.layouts[int(self.rng.integers(0, len(self.layouts)))]
            if layout is None
            else layout
        )
        snapshot = self.env.reset(
            seed=seed,
            object_count=count,
            layout_mode=layout_mode,
        )
        self.expert.reset(seed=seed)
        self._finish_reset(snapshot)
        assert self.current_observation is not None
        return self.current_observation

    def reset_case(self, case: RecoveryCase) -> InteractionReset:
        if self.oracle_codec is None:
            raise RuntimeError("reset_case requires a Compact Oracle-State codec")
        prepared = prepare_interaction_start(self.env, self.expert, case=case)
        self._finish_reset(prepared.snapshot)
        oracle = np.asarray(
            self.oracle_codec.encode_runtime(self.env, prepared, case),
            dtype=np.float32,
        )
        if oracle.shape != (36,) or not np.isfinite(oracle).all():
            raise ValueError("reset_case Oracle-State must be finite with shape (36,)")
        assert self.current_observation is not None
        return InteractionReset(
            case_id=case.case_id,
            observation=self.current_observation,
            oracle_state=oracle,
        )

    def policy_features(self) -> tuple[np.ndarray, torch.Tensor]:
        if self.current_observation is None:
            raise RuntimeError("residual environment has no current observation")
        result = self.backend.get_latents(self.current_observation, (self.tap_id,))
        self.base_action_was_clipped = bool(
            self.backend.last_residual_action_was_clipped
        )
        actions = result["__action__"]
        latent = result[self.tap_id]
        if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
            raise ValueError("base policy must emit a batched action chunk")
        if not isinstance(latent, torch.Tensor) or latent.ndim != 2 or latent.shape[0] != 1:
            raise ValueError("residual tap must emit one pooled latent row")
        return actions[0, 0].numpy().astype(np.float32), latent.detach()

    def step(
        self,
        *,
        base_action: np.ndarray,
        latent: torch.Tensor,
        residual: np.ndarray,
    ) -> ResidualTransition:
        if self.snapshot is None or self.current_observation is None:
            raise RuntimeError("residual environment is not reset")
        local, action_was_clipped = residual_clipping(
            base_action,
            residual,
            self.residual_scale,
            base_was_clipped=self.base_action_was_clipped,
        )
        self.clipped_steps += int(action_was_clipped)
        local[6] = self.gripper.resolve(float(local[6]))
        rotation = EndEffectorStateCodec.quaternion_to_matrix(
            self.snapshot.gripper.orientation
        )
        world = LocalCartesianActionCodec.decode(local, rotation)
        projection = project_cartesian_action(self.env.controller, world)
        self.projection_scales.append(float(projection.scale))
        transition = self.env.step(projection.action)
        reason = str(getattr(transition.reason, "value", transition.reason))
        in_hand = self.env.grasp_state.stable_object == self.env.target_name
        potential = interaction_potential(transition.snapshot, in_hand=in_hand)
        reward = sparse_task_reward(reason)
        if self.reward_mode == "progress":
            reward += self.progress_reward_scale * (potential - self.previous_potential)
        self.previous_potential = potential
        self.snapshot = transition.snapshot
        self.episode_return += reward
        self.episode_length += 1
        result = ResidualTransition(
            observation=self.current_observation,
            latent=latent.detach().cpu(),
            residual=np.asarray(residual, dtype=np.float32).copy(),
            base_action=np.asarray(base_action, dtype=np.float32).copy(),
            executed_local_action=local.copy(),
            action_was_clipped=action_was_clipped,
            ik_projection_scale=float(projection.scale),
            episode_action_clipping_rate=float(
                self.clipped_steps / self.episode_length
            ),
            episode_mean_ik_projection_scale=float(
                np.mean(self.projection_scales)
            ),
            reward=float(reward),
            done=bool(transition.done),
            reason=reason,
            episode_return=float(self.episode_return),
            episode_length=int(self.episode_length),
        )
        self.current_observation = None if transition.done else self._observation()
        return result

    def close(self) -> None:
        self.capture.close()

    def rng_state(self) -> dict[str, object]:
        return dict(self.rng.bit_generator.state)

    def restore_rng_state(self, state: dict[str, object]) -> None:
        self.rng.bit_generator.state = dict(state)

    def __enter__(self) -> "ResidualMujocoRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
