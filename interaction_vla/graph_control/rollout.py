from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from interaction_vla.env import LayoutMode, TerminationReason
from interaction_vla.graph_finetune.data import CausalGraphV2Tracker
from interaction_vla.graph_finetune.schema import GraphV2Normalization
from interaction_vla.lerobot_bridge.capture import DualViewCapture
from interaction_vla.lerobot_bridge.codecs import (
    EndEffectorStateCodec,
    LocalCartesianActionCodec,
    validate_finger_joint_ranges,
)
from interaction_vla.lerobot_bridge.config import BridgeConfig
from interaction_vla.lerobot_bridge.rollout import (
    ActionChunkQueue,
    BinaryGripperHysteresis,
    policy_observation,
)
from interaction_vla.physics_action_safety import project_cartesian_action
from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.physics_evaluate import InteractionRolloutTracker

from .features import FrozenGraphRuntime, pack_oracle_current
from .schema import (
    ALL_CONDITIONS,
    ORACLE_CONDITIONS,
    TOKEN_DIM,
    empty_token,
    validate_token,
)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    seed: int
    layout: str
    object_count: int


class FlatTokenProvider:
    def reset(self) -> None:
        return None

    def token(
        self, *, snapshot: Any, camera_frame: Any, state: np.ndarray, task: str
    ) -> np.ndarray:
        del snapshot, camera_frame, state, task
        return empty_token()


def _chw_rgb(value: object, name: str) -> torch.Tensor:
    array = np.asarray(value)
    if array.shape != (256, 256, 3) or array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 with shape (256, 256, 3)")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)


class PredictedTokenProvider:
    def __init__(self, runtime: FrozenGraphRuntime | Any) -> None:
        self.runtime = runtime

    def reset(self) -> None:
        self.runtime.reset()

    def token(
        self, *, snapshot: Any, camera_frame: Any, state: np.ndarray, task: str
    ) -> np.ndarray:
        del snapshot
        values = self.runtime.predict_token(
            agent_rgb=_chw_rgb(camera_frame.views["agent"].rgb, "agent RGB"),
            wrist_rgb=_chw_rgb(camera_frame.views["wrist"].rgb, "wrist RGB"),
            state=state,
            task=task,
        )
        return validate_token(values)


class OracleGraphV2TokenProvider:
    def __init__(
        self, *, teacher: Any, normalization: GraphV2Normalization
    ) -> None:
        self.teacher = teacher
        self.normalization = normalization
        self.tracker = CausalGraphV2Tracker()

    def reset(self) -> None:
        self.teacher.reset()
        self.tracker.reset()

    def bind_model(self, model: Any) -> None:
        if hasattr(self.teacher, "model"):
            self.teacher.model = model

    def token(
        self, *, snapshot: Any, camera_frame: Any, state: np.ndarray, task: str
    ) -> np.ndarray:
        del task
        frame = self.teacher.extract(snapshot, camera_frame, state=state)
        targets = self.tracker.update(frame)
        return pack_oracle_current(
            targets,
            frame_index=0,
            normalization=self.normalization,
        )


def augment_policy_observation(
    *,
    agent_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: np.ndarray,
    token: object,
) -> dict[str, torch.Tensor]:
    observation = policy_observation(
        agent_rgb=agent_rgb,
        wrist_rgb=wrist_rgb,
        state=state,
    )
    observation["observation.environment_state"] = torch.from_numpy(
        validate_token(token).copy()
    )
    return observation


def paired_evaluation_cases(
    *,
    layouts: Sequence[str],
    object_counts: Sequence[int],
    cases_per_cell: int,
    master_seed: int,
) -> tuple[EvaluationCase, ...]:
    matrix = (tuple(layouts), tuple(int(value) for value in object_counts))
    if matrix not in {
        (("normal",), (2,)),
        (("normal", "crowded"), (2, 3)),
    }:
        raise ValueError("paired evaluation matrix is incompatible")
    if cases_per_cell < 1 or master_seed < 0:
        raise ValueError("paired evaluation counts and seed are invalid")
    cases: list[EvaluationCase] = []
    for layout_index, layout in enumerate(layouts):
        for object_count in object_counts:
            for replicate in range(cases_per_cell):
                seed = int(
                    np.random.SeedSequence(
                        (master_seed, layout_index, int(object_count), replicate)
                    ).generate_state(1, dtype=np.uint32)[0]
                )
                cases.append(
                    EvaluationCase(
                        case_id=f"{layout}_n{int(object_count)}_{replicate:03d}",
                        seed=seed,
                        layout=str(layout),
                        object_count=int(object_count),
                    )
                )
    return tuple(cases)


_AGGREGATE_FIELDS = {
    "success_rate": "success",
    "wrong_object_interaction_rate": "wrong_object_interaction",
    "wrong_object_stable_grasp_rate": "wrong_object_stable_grasp",
    "target_drop_rate": "target_drop",
    "timeout_rate": "timeout",
    "mean_steps": "steps",
    "mean_ik_projection_scale": "mean_ik_projection_scale",
    "action_clipping_rate": "action_clipping_rate",
    "mean_gripper_switch_count": "gripper_switch_count",
}
_FULL_CONTRASTS = (
    ("oracle_graph_v2", "flat"),
    ("predicted_random_v2", "flat"),
    ("predicted_reflect_v2", "flat"),
    ("predicted_reflect_v2", "predicted_random_v2"),
    ("oracle_graph_v2", "predicted_reflect_v2"),
)


def _active_conditions(conditions: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in conditions)
    if values not in {ORACLE_CONDITIONS, ALL_CONDITIONS}:
        raise ValueError("conditions must be the oracle pair or full Graph v2 matrix")
    return values


def _means(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for output_name, source_name in _AGGREGATE_FIELDS.items():
        values = [float(record[source_name]) for record in records]
        if not values or not np.isfinite(values).all():
            raise ValueError(f"rollout metric {source_name} is empty or non-finite")
        result[output_name] = float(np.mean(values))
    return result


def aggregate_rollouts(
    records: Sequence[Mapping[str, object]], *, conditions: Sequence[str]
) -> dict[str, object]:
    active = _active_conditions(conditions)
    contrasts_to_compute = (
        (("oracle_graph_v2", "flat"),)
        if active == ORACLE_CONDITIONS
        else _FULL_CONTRASTS
    )
    values = [dict(record) for record in records]
    if not values:
        raise ValueError("rollout aggregation requires records")
    required = {
        "condition",
        "policy_seed",
        "case_id",
        "environment_seed",
        "layout",
        "object_count",
        *_AGGREGATE_FIELDS.values(),
    }
    for record in values:
        missing = required - set(record)
        if missing:
            raise ValueError("rollout record is missing: " + ", ".join(sorted(missing)))
        if record["condition"] not in active:
            raise ValueError("rollout record condition is invalid")
    groups: dict[tuple[int, str], set[str]] = {}
    identities: dict[tuple[int, str], tuple[int, str, int]] = {}
    for record in values:
        key = (int(record["policy_seed"]), str(record["case_id"]))
        groups.setdefault(key, set()).add(str(record["condition"]))
        identity = (
            int(record["environment_seed"]),
            str(record["layout"]),
            int(record["object_count"]),
        )
        if key in identities and identities[key] != identity:
            fields = ("environment_seed", "layout", "object_count")
            differing = [
                field
                for field, expected, actual in zip(fields, identities[key], identity)
                if expected != actual
            ]
            raise ValueError(
                "rollout paired case identity mismatch: " + ", ".join(differing)
            )
        identities[key] = identity
    if any(group != set(active) for group in groups.values()):
        raise ValueError("rollout records are not paired across all conditions")
    if len(values) != len(groups) * len(active):
        raise ValueError("rollout records contain duplicate paired cases")

    records_by_case = {
        key: {
            str(record["condition"]): record
            for record in values
            if (int(record["policy_seed"]), str(record["case_id"])) == key
        }
        for key in groups
    }
    paired_case_deltas: list[dict[str, object]] = []
    for key in sorted(records_by_case):
        policy_seed, case_id = key
        paired = records_by_case[key]
        environment_seed, layout, object_count = identities[key]
        for first, second in contrasts_to_compute:
            for metric, source_name in _AGGREGATE_FIELDS.items():
                paired_case_deltas.append(
                    {
                        "policy_seed": policy_seed,
                        "case_id": case_id,
                        "environment_seed": environment_seed,
                        "layout": layout,
                        "object_count": object_count,
                        "training_distribution": (
                            "id" if layout == "normal" else "ood"
                        ),
                        "contrast": f"{first}-{second}",
                        "metric": metric,
                        "delta": float(paired[first][source_name])
                        - float(paired[second][source_name]),
                    }
                )

    by_condition = {
        condition: _means(
            [record for record in values if record["condition"] == condition]
        )
        for condition in active
    }
    policy_seeds = sorted({int(record["policy_seed"]) for record in values})
    by_seed_condition: dict[tuple[int, str], dict[str, float]] = {}
    for seed in policy_seeds:
        for condition in active:
            by_seed_condition[(seed, condition)] = _means(
                [
                    record
                    for record in values
                    if int(record["policy_seed"]) == seed
                    and record["condition"] == condition
                ]
            )
    contrasts: dict[str, object] = {}
    for first, second in contrasts_to_compute:
        metrics: dict[str, object] = {}
        for metric in _AGGREGATE_FIELDS:
            deltas = {
                str(seed): by_seed_condition[(seed, first)][metric]
                - by_seed_condition[(seed, second)][metric]
                for seed in policy_seeds
            }
            array = np.asarray(tuple(deltas.values()), dtype=np.float64)
            metrics[metric] = {
                "per_seed": deltas,
                "mean": float(array.mean()),
                "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "policy_seeds": len(array),
            }
        contrasts[f"{first}-{second}"] = metrics
    oracle_gate: dict[str, object] | None = None
    passed = True
    if active == ORACLE_CONDITIONS:
        success_delta = round(
            by_condition["oracle_graph_v2"]["success_rate"]
            - by_condition["flat"]["success_rate"],
            12,
        )
        wrong_grasp_delta = round(
            by_condition["oracle_graph_v2"]["wrong_object_stable_grasp_rate"]
            - by_condition["flat"]["wrong_object_stable_grasp_rate"],
            12,
        )
        passed = success_delta >= 0.10 and wrong_grasp_delta <= 0.0
        oracle_gate = {
            "passed": passed,
            "success_delta": success_delta,
            "wrong_object_stable_grasp_delta": wrong_grasp_delta,
            "required_success_delta": 0.10,
        }
    result: dict[str, object] = {
        "passed": passed,
        "records": len(values),
        "paired_cases": len(groups),
        "policy_seeds": policy_seeds,
        "replication_unit": "policy_seed",
        "paired_case_deltas": paired_case_deltas,
        "by_condition": by_condition,
        "contrasts": contrasts,
    }
    if active == ALL_CONDITIONS:
        oracle_gap = (
            by_condition["oracle_graph_v2"]["success_rate"]
            - by_condition["flat"]["success_rate"]
        )
        result["oracle_gap_recovered"] = (
            None
            if oracle_gap <= 0.0
            else round(
                (
                    by_condition["predicted_reflect_v2"]["success_rate"]
                    - by_condition["flat"]["success_rate"]
                )
                / oracle_gap,
                12,
            )
        )
    if oracle_gate is not None:
        result["oracle_gate"] = oracle_gate
    return result


@dataclass
class GraphPolicyRuntime:
    condition: str
    policy_seed: int
    checkpoint: Path
    policy: Any
    preprocessor: Any
    postprocessor: Any
    token_provider: Any

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.token_provider.reset()


def _make_env(config: BridgeConfig, *, max_steps: int) -> FrankaContactEnv:
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


def _predict_chunk(runtime: GraphPolicyRuntime, observation: dict[str, torch.Tensor]) -> np.ndarray:
    processed = runtime.preprocessor(observation)
    with torch.no_grad():
        normalized = runtime.policy.predict_action_chunk(processed)
        actions = runtime.postprocessor(normalized)
    if not isinstance(actions, torch.Tensor) or actions.shape != (1, 8, 7):
        raise ValueError("Graph-conditioned ACT action chunk must have shape (1, 8, 7)")
    result = actions[0].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Graph-conditioned ACT action chunk must be finite")
    return result


def _next_queued_action(
    queue: ActionChunkQueue,
    runtime: GraphPolicyRuntime | Any,
    observation_factory: Callable[[], dict[str, torch.Tensor]],
):
    return queue.next(lambda: _predict_chunk(runtime, observation_factory()))


def rollout_case(
    config: BridgeConfig,
    runtime: GraphPolicyRuntime,
    case: EvaluationCase,
    *,
    max_steps: int,
) -> dict[str, object]:
    if runtime.condition not in ALL_CONDITIONS:
        raise ValueError("Graph policy runtime condition is invalid")
    if max_steps < 1:
        raise ValueError("rollout max_steps must be positive")
    env = _make_env(config, max_steps=max_steps)
    validate_finger_joint_ranges(env.model)
    snapshot = env.reset(
        seed=case.seed,
        object_count=case.object_count,
        layout_mode=LayoutMode(case.layout),
    )
    capture = DualViewCapture(
        env.model,
        width=config.dataset.image_size[1],
        height=config.dataset.image_size[0],
    )
    runtime.reset()
    if hasattr(runtime.token_provider, "bind_model"):
        runtime.token_provider.bind_model(env.model)
    queue = ActionChunkQueue(
        chunk_size=config.act.chunk_size,
        n_action_steps=config.act.n_action_steps,
    )
    gripper = BinaryGripperHysteresis(
        close_threshold=0.4,
        open_threshold=0.6,
        initially_open=True,
    )
    interactions = InteractionRolloutTracker(target_name=env.target_name)
    projection_scales: list[float] = []
    clipped_steps = 0
    reason = TerminationReason.RUNNING
    try:
        for _ in range(max_steps):
            camera_frame = capture.capture(env, include_teacher=True)
            state = EndEffectorStateCodec.encode_snapshot(
                snapshot, env.proprioception()
            )

            def observation_factory() -> dict[str, torch.Tensor]:
                token = runtime.token_provider.token(
                    snapshot=snapshot,
                    camera_frame=camera_frame,
                    state=state,
                    task=config.dataset.task,
                )
                return augment_policy_observation(
                    agent_rgb=camera_frame.views["agent"].rgb,
                    wrist_rgb=camera_frame.views["wrist"].rgb,
                    state=state,
                    token=token,
                )

            selected = _next_queued_action(queue, runtime, observation_factory)
            if selected.queue_index != 0:
                raise ValueError(
                    "Graph-conditioned ACT requires receding-horizon queue index 0"
                )
            raw = selected.action.copy()
            action = raw.copy()
            action[:6] = np.clip(action[:6], -1.0, 1.0)
            clipped_steps += int(np.any(action[:6] != raw[:6]))
            action[6] = gripper.resolve(float(raw[6]))
            rotation = EndEffectorStateCodec.quaternion_to_matrix(
                snapshot.gripper.orientation
            )
            world = LocalCartesianActionCodec.decode(action, rotation)
            projection = project_cartesian_action(env.controller, world)
            projection_scales.append(float(projection.scale))
            transition = env.step(projection.action)
            reason = transition.reason
            interactions.observe(
                env.grasp_state,
                events=env.grasp_tracker.interaction_events_since(
                    interactions.last_observed_substep
                ),
            )
            snapshot = transition.snapshot
            if transition.done:
                break
    finally:
        capture.close()
    interaction = interactions.metrics()
    steps = len(projection_scales)
    return {
        "condition": runtime.condition,
        "policy_seed": runtime.policy_seed,
        "case_id": case.case_id,
        "environment_seed": case.seed,
        "layout": case.layout,
        "training_distribution": "id" if case.layout == "normal" else "ood",
        "object_count": case.object_count,
        "success": reason is TerminationReason.SUCCESS,
        "wrong_object_interaction": bool(interaction["wrong_object_interaction"]),
        "wrong_object_stable_grasp": bool(
            interaction["wrong_object_stable_grasp"]
        ),
        "target_drop": bool(interaction["dropped_target"]),
        "timeout": reason is TerminationReason.TIMEOUT,
        "termination_reason": reason.value,
        "steps": steps,
        "mean_ik_projection_scale": float(np.mean(projection_scales)) if steps else 1.0,
        "action_clipping_rate": float(clipped_steps / steps) if steps else 0.0,
        "gripper_switch_count": gripper.switch_count,
        "checkpoint": runtime.checkpoint.as_posix(),
    }


def evaluate_runtimes(
    config: BridgeConfig,
    runtimes: Mapping[tuple[int, str], GraphPolicyRuntime],
    cases: Sequence[EvaluationCase],
    *,
    max_steps: int,
    conditions: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    active = _active_conditions(conditions)
    seeds = sorted({seed for seed, _ in runtimes})
    expected = {(seed, condition) for seed in seeds for condition in active}
    if set(runtimes) != expected:
        raise ValueError("evaluation runtimes do not form the paired condition matrix")
    records = [
        rollout_case(config, runtimes[(seed, condition)], case, max_steps=max_steps)
        for seed in seeds
        for case in cases
        for condition in active
    ]
    return records, aggregate_rollouts(records, conditions=active)
