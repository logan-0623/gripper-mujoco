from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from interaction_vla.graph_pretrain.reflectvlm import Vocabulary
from interaction_vla.lerobot_bridge.codecs import EndEffectorStateCodec
from interaction_vla.lerobot_bridge.interaction_phase import (
    PHASE_IDS,
    causal_phase_step,
)
from interaction_vla.lerobot_bridge.teacher_schema import (
    CONFIDENCE,
    FORBIDDEN_FIELD_FRAGMENTS,
    OPERATOR_IDS,
    PREDICATE_IDS,
    PROBABILITY_0,
    PROBABILITY_1,
    RELATIVE_LINEAR_VELOCITY,
    RELATIVE_POSITION,
    RISK_0,
    RISK_1,
    SCHEMA_VERSION as TEACHER_SCHEMA,
    SIGNED_MARGIN_0,
    SIGNED_MARGIN_1,
    SIGNED_MARGIN_2,
    TeacherFrame,
)

from .schema import (
    TOKEN_DIM,
    GraphV2Normalization,
    GraphV2Targets,
    normalized_graph_v2_frame,
    pack_oracle_target,
)


_V2_TARGET_KEYS = {
    "annotation.tc_tig.entity_pose",
    "annotation.tc_tig.entity_mask",
    "annotation.tc_tig.entity_visibility",
    "annotation.tc_tig.relation_mask",
    "annotation.tc_tig.relation_values",
}
_RAW_REQUIRED_KEYS = {
    "observation.images.agent",
    "observation.images.wrist",
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "task",
}
_RAW_OPTIONAL_KEYS = {"action_is_pad"}
_FORBIDDEN_INPUT_FRAGMENTS = (
    "annotation",
    "depth",
    "segmentation",
) + FORBIDDEN_FIELD_FRAGMENTS
MODEL_BATCH_KEYS = {
    "agent_rgb",
    "wrist_rgb",
    "state",
    "language_tokens",
    "language_mask",
    "entity_mask",
    "entity_visibility",
    "relation_mask",
    "gripper_target_geometry",
    "target_receptacle_geometry",
    "distractor_geometry",
    "phase",
    "relation_trends",
    "goal_relation",
    "goal_operator",
    "goal_predicate",
    "goal_residual",
    "previous_graph",
}


PHASE_GOALS = {
    PHASE_IDS["approach"]: (
        0,
        OPERATOR_IDS["establish"],
        PREDICATE_IDS["proximity"],
    ),
    PHASE_IDS["grasp"]: (
        0,
        OPERATOR_IDS["establish"],
        PREDICATE_IDS["enclosure"],
    ),
    PHASE_IDS["lift"]: (
        0,
        OPERATOR_IDS["establish"],
        PREDICATE_IDS["co_motion"],
    ),
    PHASE_IDS["transport"]: (
        1,
        OPERATOR_IDS["establish"],
        PREDICATE_IDS["proximity"],
    ),
    PHASE_IDS["place"]: (
        1,
        OPERATOR_IDS["establish"],
        PREDICATE_IDS["containment"],
    ),
    PHASE_IDS["release"]: (
        0,
        OPERATOR_IDS["break"],
        PREDICATE_IDS["co_motion"],
    ),
}


def _closing_speed(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    displacement = np.asarray(position, dtype=np.float32)
    relative_velocity = np.asarray(velocity, dtype=np.float32)
    distance = np.linalg.norm(displacement, axis=-1)
    return np.divide(
        -np.sum(displacement * relative_velocity, axis=-1),
        distance,
        out=np.zeros_like(distance, dtype=np.float32),
        where=distance > 1e-6,
    ).astype(np.float32)


def teacher_frame_arrays(frame: TeacherFrame) -> dict[str, np.ndarray]:
    return {
        "frame_index": np.asarray([frame.frame_index], dtype=np.int64),
        "timestamp": np.asarray([frame.timestamp], dtype=np.float64),
        "state_hash": np.asarray([frame.state_hash]),
        "annotation.tc_tig.entity_pose": frame.entity_pose[None].copy(),
        "annotation.tc_tig.entity_size": frame.entity_size[None].copy(),
        "annotation.tc_tig.entity_mask": frame.entity_mask[None].copy(),
        "annotation.tc_tig.entity_visibility": (
            frame.entity_visibility[None].copy()
        ),
        "annotation.tc_tig.relation_mask": frame.relation_mask[None].copy(),
        "annotation.tc_tig.relation_values": frame.relation_values[None].copy(),
    }


def _geometry_teacher_frames(
    arrays: Mapping[str, np.ndarray],
) -> tuple[TeacherFrame, ...]:
    missing = _V2_TARGET_KEYS - set(arrays)
    if missing:
        raise ValueError("teacher arrays are missing: " + ", ".join(sorted(missing)))
    relation = np.asarray(
        arrays["annotation.tc_tig.relation_values"], dtype=np.float32
    )
    frames = len(relation)
    if (
        frames < 1
        or relation.shape != (frames, 8, 24)
        or not np.isfinite(relation).all()
    ):
        raise ValueError("teacher relation_values must be finite [frames, 8, 24]")

    def value(
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        default: np.ndarray,
    ) -> np.ndarray:
        result = np.asarray(arrays.get(name, default))
        if result.shape != shape or result.dtype != dtype:
            raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
        if np.issubdtype(dtype, np.floating) and not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite")
        return result

    entity_pose = value(
        "annotation.tc_tig.entity_pose",
        (frames, 6, 9),
        np.dtype(np.float32),
        np.zeros((frames, 6, 9), dtype=np.float32),
    )
    entity_size = value(
        "annotation.tc_tig.entity_size",
        (frames, 6, 3),
        np.dtype(np.float32),
        np.zeros((frames, 6, 3), dtype=np.float32),
    )
    entity_mask = value(
        "annotation.tc_tig.entity_mask",
        (frames, 6),
        np.dtype(np.bool_),
        np.zeros((frames, 6), dtype=np.bool_),
    )
    entity_visibility = value(
        "annotation.tc_tig.entity_visibility",
        (frames, 6, 2),
        np.dtype(np.float32),
        np.zeros((frames, 6, 2), dtype=np.float32),
    )
    relation_mask = value(
        "annotation.tc_tig.relation_mask",
        (frames, 8),
        np.dtype(np.bool_),
        np.zeros((frames, 8), dtype=np.bool_),
    )
    frame_indices = np.asarray(
        arrays.get("frame_index", np.arange(frames)), dtype=np.int64
    )
    timestamps = np.asarray(
        arrays.get("timestamp", np.arange(frames) / 20.0), dtype=np.float64
    )
    state_hashes = np.asarray(
        arrays.get("state_hash", [f"graph-v2-{index}" for index in range(frames)])
    ).astype(str)
    if (
        frame_indices.shape != (frames,)
        or timestamps.shape != (frames,)
        or state_hashes.shape != (frames,)
    ):
        raise ValueError("teacher frame metadata must align with geometry rows")
    result = []
    for index in range(frames):
        result.append(
            TeacherFrame(
                frame_index=int(frame_indices[index]),
                timestamp=float(timestamps[index]),
                state_hash=str(state_hashes[index]),
                entity_pose=entity_pose[index],
                entity_size=entity_size[index],
                entity_role=np.arange(6, dtype=np.int32),
                entity_visibility=entity_visibility[index],
                entity_mask=entity_mask[index],
                relation_values=relation[index],
                relation_type=np.arange(8, dtype=np.int32),
                relation_mask=relation_mask[index],
                instance_agent=np.zeros((1, 1), dtype=np.int32),
                instance_wrist=np.zeros((1, 1), dtype=np.int32),
                depth_agent=np.zeros((1, 1), dtype=np.float32),
                depth_wrist=np.zeros((1, 1), dtype=np.float32),
                camera_intrinsics=np.zeros((2, 3, 3), dtype=np.float32),
                camera_extrinsics_base=np.zeros((2, 4, 4), dtype=np.float32),
            )
        )
    return tuple(result)


def graph_v2_trend_scalars(targets: GraphV2Targets) -> np.ndarray:
    gripper_distance = targets.gripper_target_geometry[:, 3]
    grasp_confidence = (
        targets.gripper_target_geometry[:, 5]
        * targets.gripper_target_geometry[:, 6]
    )
    goal_distance = np.linalg.norm(
        targets.target_receptacle_geometry[:, :3], axis=1
    )
    active = targets.relation_mask[:, (3, 5)]
    clearances = targets.distractor_geometry[:, :, 3]
    minimum_clearance = np.min(
        np.where(active, clearances, np.inf), axis=1
    )
    minimum_clearance[~np.any(active, axis=1)] = 0.0
    return np.stack(
        (
            gripper_distance,
            grasp_confidence,
            goal_distance,
            minimum_clearance,
        ),
        axis=1,
    ).astype(np.float32)


def current_graph_v2_target(
    arrays: Mapping[str, np.ndarray],
    *,
    previous_scalars: np.ndarray | None,
    previous_phase: int,
) -> GraphV2Targets:
    missing = _V2_TARGET_KEYS - set(arrays)
    if missing:
        raise ValueError("teacher arrays are missing: " + ", ".join(sorted(missing)))
    relation = np.asarray(
        arrays["annotation.tc_tig.relation_values"], dtype=np.float32
    )
    pose = np.asarray(arrays["annotation.tc_tig.entity_pose"], dtype=np.float32)
    if relation.shape != (1, 8, 24) or not np.isfinite(relation).all():
        raise ValueError("current relation must be finite with shape [1, 8, 24]")
    if pose.shape != (1, 6, 9) or not np.isfinite(pose).all():
        raise ValueError("current entity_pose must be finite with shape [1, 6, 9]")
    entity_mask = np.asarray(
        arrays["annotation.tc_tig.entity_mask"], dtype=np.bool_
    )
    visibility = np.asarray(
        arrays["annotation.tc_tig.entity_visibility"], dtype=np.float32
    )
    relation_mask = np.asarray(
        arrays["annotation.tc_tig.relation_mask"], dtype=np.bool_
    )
    if entity_mask.shape != (1, 6) or relation_mask.shape != (1, 8):
        raise ValueError("current entity/relation masks have invalid shapes")
    if visibility.shape != (1, 6, 2) or not np.isfinite(visibility).all():
        raise ValueError("current entity_visibility must be finite [1, 6, 2]")

    gripper_delta = relation[:, 0, RELATIVE_POSITION]
    gripper_target = np.concatenate(
        (
            gripper_delta,
            np.linalg.norm(gripper_delta, axis=1, keepdims=True),
            _closing_speed(
                gripper_delta,
                relation[:, 0, RELATIVE_LINEAR_VELOCITY],
            )[:, None],
            relation[:, 0, [PROBABILITY_0, PROBABILITY_1, CONFIDENCE]],
        ),
        axis=1,
    ).astype(np.float32)

    gripper_rotation = EndEffectorStateCodec.decode_rotation(pose[0, 0, 3:])
    target_to_goal_action = gripper_rotation.T @ (pose[0, 2, :3] - pose[0, 1, :3])
    placement = relation[0, 1]
    target_receptacle = np.asarray(
        [
            *target_to_goal_action,
            *placement[RELATIVE_POSITION],
            min(placement[SIGNED_MARGIN_0], placement[SIGNED_MARGIN_1]),
            -abs(float(placement[SIGNED_MARGIN_2])),
            placement[PROBABILITY_0],
            placement[CONFIDENCE],
        ],
        dtype=np.float32,
    )[None]

    distractors = np.zeros((1, 2, 7), dtype=np.float32)
    for slot, relation_index in enumerate((3, 5)):
        if relation_mask[0, relation_index]:
            values = relation[0, relation_index]
            distractors[0, slot] = np.asarray(
                [
                    *values[RELATIVE_POSITION],
                    values[SIGNED_MARGIN_0],
                    values[RISK_1],
                    values[RISK_0],
                    values[CONFIDENCE],
                ],
                dtype=np.float32,
            )

    phase = causal_phase_step(relation[0], int(previous_phase))
    goal_relation, goal_operator, goal_predicate = PHASE_GOALS[phase]
    contact = float(gripper_target[0, 5])
    co_motion = float(gripper_target[0, 6])
    goal_distance = float(np.linalg.norm(target_to_goal_action))
    containment = float(target_receptacle[0, 8])
    residuals = {
        PHASE_IDS["approach"]: -float(gripper_target[0, 3]),
        PHASE_IDS["grasp"]: -(1.0 - contact),
        PHASE_IDS["lift"]: -(1.0 - co_motion),
        PHASE_IDS["transport"]: -goal_distance,
        PHASE_IDS["place"]: -(1.0 - containment),
        PHASE_IDS["release"]: -co_motion,
    }
    provisional = GraphV2Targets(
        entity_mask=entity_mask,
        entity_visibility=visibility,
        relation_mask=relation_mask,
        gripper_target_geometry=gripper_target,
        target_receptacle_geometry=target_receptacle,
        distractor_geometry=distractors,
        phase=np.asarray([phase], dtype=np.int64),
        relation_trends=np.zeros((1, 4), dtype=np.float32),
        goal_relation=np.asarray([goal_relation], dtype=np.int64),
        goal_operator=np.asarray([goal_operator], dtype=np.int64),
        goal_predicate=np.asarray([goal_predicate], dtype=np.int64),
        goal_residual=np.asarray([residuals[phase]], dtype=np.float32),
    )
    scalars = graph_v2_trend_scalars(provisional)[0]
    if previous_scalars is None:
        trends = np.zeros((1, 4), dtype=np.float32)
    else:
        previous = np.asarray(previous_scalars, dtype=np.float32)
        if previous.shape != (4,) or not np.isfinite(previous).all():
            raise ValueError("previous Graph v2 scalars must be finite with shape [4]")
        trends = (scalars - previous)[None].astype(np.float32)
    return GraphV2Targets(
        **{
            field.name: (
                trends
                if field.name == "relation_trends"
                else getattr(provisional, field.name)
            )
            for field in fields(GraphV2Targets)
        }
    )


class CausalGraphV2Tracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._previous_scalars: np.ndarray | None = None
        self._previous_phase = PHASE_IDS["approach"]

    def update(self, frame: TeacherFrame) -> GraphV2Targets:
        current = current_graph_v2_target(
            teacher_frame_arrays(frame),
            previous_scalars=self._previous_scalars,
            previous_phase=self._previous_phase,
        )
        self._previous_scalars = graph_v2_trend_scalars(current)[0].copy()
        self._previous_phase = int(current.phase[0])
        return current


def graph_v2_targets(arrays: Mapping[str, np.ndarray]) -> GraphV2Targets:
    tracker = CausalGraphV2Tracker()
    targets = [tracker.update(frame) for frame in _geometry_teacher_frames(arrays)]
    return GraphV2Targets(
        **{
            field.name: np.concatenate(
                [getattr(target, field.name) for target in targets], axis=0
            )
            for field in fields(GraphV2Targets)
        }
    )


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    values = np.asarray(ratios, dtype=np.float64)
    if (
        values.shape != (3,)
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or not np.isclose(values.sum(), 1.0)
    ):
        raise ValueError("split ratios must be three positive values summing to one")


def split_episode_indices(
    records: Sequence[Mapping[str, object]],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, list[int]]:
    _validate_ratios(ratios)
    if len(records) < 3:
        raise ValueError("episode split requires at least three episodes")
    episode_indices = [int(record.get("episode_index", -1)) for record in records]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("teacher episode indices must be unique")
    if any(record.get("schema_version") != TEACHER_SCHEMA for record in records):
        raise ValueError("teacher manifest schema version is incompatible")

    def digest(record: Mapping[str, object]) -> bytes:
        episode_index = int(record["episode_index"])
        teacher_seed = int(record["seed"])
        return hashlib.sha256(
            f"{int(seed)}:{episode_index}:{teacher_seed}".encode()
        ).digest()

    ordered = sorted(records, key=lambda record: (digest(record), int(record["episode_index"])))
    validation_count = max(1, int(math.floor(len(ordered) * ratios[1])))
    test_count = max(1, int(math.floor(len(ordered) * ratios[2])))
    train_count = len(ordered) - validation_count - test_count
    if train_count < 1:
        raise ValueError("split ratios leave no training episode")
    values = [int(record["episode_index"]) for record in ordered]
    return {
        "train": sorted(values[:train_count]),
        "validation": sorted(values[train_count : train_count + validation_count]),
        "test": sorted(values[train_count + validation_count :]),
    }


def select_training_fraction(
    episodes: Sequence[int], *, fraction: float, seed: int
) -> list[int]:
    values = [int(value) for value in episodes]
    if not values or len(set(values)) != len(values):
        raise ValueError("training episodes must be non-empty and unique")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("training fraction must lie within (0, 1]")
    ordered = sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{int(seed)}:{value}".encode()).digest(),
            value,
        ),
    )
    count = max(1, math.ceil(len(ordered) * float(fraction)))
    return sorted(ordered[:count])


@dataclass(frozen=True)
class PreparedMuJoCoCorpus:
    source: Any
    records: tuple[dict[str, object], ...]
    targets: dict[int, GraphV2Targets]
    tasks: dict[int, str]
    states: np.ndarray
    row_indices: dict[int, tuple[int, ...]]
    splits: dict[str, list[int]]
    pretrained_vocabulary: Vocabulary

    def for_training_fraction(self, fraction: float, seed: int) -> "TrainingCorpus":
        selected = select_training_fraction(
            self.splits["train"], fraction=fraction, seed=seed
        )
        vocabulary = extend_vocabulary(
            self.pretrained_vocabulary,
            [self.tasks[episode] for episode in selected],
        )
        normalization = fit_normalization(self, selected)
        return TrainingCorpus(
            corpus=self,
            selected_train_episodes=tuple(selected),
            vocabulary=vocabulary,
            normalization=normalization,
        )


@dataclass(frozen=True)
class TrainingCorpus:
    corpus: PreparedMuJoCoCorpus
    selected_train_episodes: tuple[int, ...]
    vocabulary: Vocabulary
    normalization: GraphV2Normalization


def _source_columns(source: Any) -> Mapping[str, Sequence[Any]]:
    raw = getattr(source, "hf_dataset", None)
    if raw is None:
        raise ValueError("source must expose metadata through hf_dataset")
    return raw


def _source_tasks(source: Any) -> tuple[str, ...]:
    direct = getattr(source, "tasks", None)
    if direct is not None:
        values = tuple(str(value) for value in direct)
    else:
        meta = getattr(source, "meta", None)
        task_index = getattr(meta, "tasks", None)
        if task_index is None or not hasattr(task_index, "index"):
            raise ValueError("source must expose language tasks")
        values = tuple(str(value) for value in task_index.index.tolist())
    if not values or any(not value.strip() for value in values):
        raise ValueError("source language tasks must be non-empty")
    return values


def extend_vocabulary(base: Vocabulary, tasks: Sequence[str]) -> Vocabulary:
    additions = Vocabulary.build((str(task),) for task in tasks).tokens[2:]
    tokens = (*base.tokens, *(token for token in additions if token not in base.tokens))
    return Vocabulary(tuple(tokens))


def prepare_corpus(
    source: Any,
    records: Sequence[Mapping[str, object]],
    sidecars: Mapping[int, Mapping[str, np.ndarray]],
    *,
    split_seed: int,
    split_ratios: tuple[float, float, float],
    pretrained_vocabulary: Vocabulary | None = None,
) -> PreparedMuJoCoCorpus:
    normalized_records = tuple(dict(record) for record in records)
    splits = split_episode_indices(
        normalized_records, seed=split_seed, ratios=split_ratios
    )
    record_by_episode = {
        int(record["episode_index"]): record for record in normalized_records
    }
    if set(sidecars) != set(record_by_episode):
        raise ValueError("teacher sidecars must match manifest episodes")
    targets = {
        int(episode): graph_v2_targets(arrays)
        for episode, arrays in sidecars.items()
    }
    columns = _source_columns(source)
    required_columns = {
        "observation.state",
        "episode_index",
        "frame_index",
        "task_index",
    }
    column_names = getattr(columns, "column_names", None)
    if column_names is None:
        column_names = columns.keys()
    missing = required_columns - set(column_names)
    if missing:
        raise ValueError("source metadata is missing: " + ", ".join(sorted(missing)))
    lengths = {len(columns[name]) for name in required_columns}
    if lengths != {len(source)}:
        raise ValueError("source metadata columns have inconsistent lengths")
    states = np.asarray(columns["observation.state"], dtype=np.float32)
    if states.shape != (len(source), 10) or not np.isfinite(states).all():
        raise ValueError("source end-effector state must be finite with shape [rows, 10]")
    episode_values = np.asarray(columns["episode_index"], dtype=np.int64)
    frame_values = np.asarray(columns["frame_index"], dtype=np.int64)
    task_values = np.asarray(columns["task_index"], dtype=np.int64)
    task_names = _source_tasks(source)
    row_indices: dict[int, tuple[int, ...]] = {}
    tasks: dict[int, str] = {}
    for episode, record in record_by_episode.items():
        rows = np.flatnonzero(episode_values == episode)
        order = np.argsort(frame_values[rows])
        rows = rows[order]
        frames = int(record.get("frames", -1))
        if len(rows) != frames or not np.array_equal(
            frame_values[rows], np.arange(frames)
        ):
            raise ValueError(f"source frame alignment mismatch for episode {episode}")
        if len(targets[episode].phase) != frames:
            raise ValueError(f"teacher frame alignment mismatch for episode {episode}")
        episode_tasks = np.unique(task_values[rows])
        if len(episode_tasks) != 1 or not 0 <= int(episode_tasks[0]) < len(task_names):
            raise ValueError(f"task metadata mismatch for episode {episode}")
        row_indices[episode] = tuple(int(value) for value in rows)
        tasks[episode] = task_names[int(episode_tasks[0])]
    if set(np.unique(episode_values).tolist()) != set(record_by_episode):
        raise ValueError("source episodes do not match teacher manifest")
    return PreparedMuJoCoCorpus(
        source=source,
        records=normalized_records,
        targets=targets,
        tasks=tasks,
        states=states,
        row_indices=row_indices,
        splits=splits,
        pretrained_vocabulary=(
            pretrained_vocabulary
            if pretrained_vocabulary is not None
            else Vocabulary(("<pad>", "<unk>"))
        ),
    )


def _safe_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, np.float32(1e-4))


def fit_normalization(
    corpus: PreparedMuJoCoCorpus, episodes: Sequence[int]
) -> GraphV2Normalization:
    rows = [row for episode in episodes for row in corpus.row_indices[int(episode)]]
    if not rows:
        raise ValueError("normalization requires training rows")
    state_mean, state_std = _safe_mean_std(corpus.states[rows])
    workspace_values: list[np.ndarray] = []
    velocity_values: list[np.ndarray] = []
    for episode in episodes:
        target = corpus.targets[int(episode)]
        gripper_active = target.relation_mask[:, 0]
        gripper = target.gripper_target_geometry[gripper_active]
        workspace_values.extend(np.abs(gripper[:, index]) for index in range(4))
        velocity_values.append(np.abs(gripper[:, 4]))

        goal_active = target.relation_mask[:, 1]
        goal = target.target_receptacle_geometry[goal_active]
        workspace_values.extend(np.abs(goal[:, index]) for index in range(8))

        for distractor, relation_index in enumerate((3, 5)):
            active = target.relation_mask[:, relation_index]
            geometry = target.distractor_geometry[active, distractor]
            workspace_values.extend(
                np.abs(geometry[:, index]) for index in range(4)
            )
    workspace_percentiles = [
        float(np.percentile(values, 95.0))
        for values in workspace_values
        if len(values)
    ]
    active_speeds = [values for values in velocity_values if len(values)]
    closing_speeds = (
        np.concatenate(active_speeds, axis=0)
        if active_speeds
        else np.zeros(0, dtype=np.float32)
    )
    workspace_scale = max([0.10, *workspace_percentiles])
    velocity_scale = max(
        0.05,
        float(np.percentile(closing_speeds, 95.0)) if len(closing_speeds) else 0.0,
    )
    return GraphV2Normalization(
        state_mean=state_mean,
        state_std=state_std,
        workspace_scale=workspace_scale,
        velocity_scale=velocity_scale,
    )


def resize_rgb(value: Any, image_size: int, name: str) -> torch.Tensor:
    image = torch.as_tensor(value, dtype=torch.float32)
    if image.ndim != 3 or image.shape[0] != 3 or not torch.isfinite(image).all():
        raise ValueError(f"{name} must be a finite CHW RGB tensor")
    if torch.any((image < 0.0) | (image > 1.0)):
        raise ValueError(f"{name} values must lie within [0, 1]")
    return F.interpolate(
        image.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


class MuJoCoGraphDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        training: TrainingCorpus,
        *,
        partition: str,
        image_size: int,
        max_language_tokens: int,
    ) -> None:
        if partition not in {"train", "validation", "test"}:
            raise ValueError("partition must be train, validation, or test")
        if image_size < 8 or max_language_tokens < 1:
            raise ValueError("image size and language token count must be positive")
        self.training = training
        self.partition = partition
        episodes = (
            training.selected_train_episodes
            if partition == "train"
            else tuple(training.corpus.splits[partition])
        )
        self.row_indices = tuple(
            row
            for episode in sorted(episodes)
            for row in training.corpus.row_indices[episode]
        )
        self.image_size = int(image_size)
        self.max_language_tokens = int(max_language_tokens)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        source_index = self.row_indices[item]
        sample = self.training.corpus.source[source_index]
        keys = set(sample)
        forbidden = sorted(
            key
            for key in keys
            if any(fragment in key.lower() for fragment in _FORBIDDEN_INPUT_FRAGMENTS)
        )
        if forbidden:
            raise ValueError(f"model sample contains forbidden input keys: {forbidden}")
        missing = _RAW_REQUIRED_KEYS - keys
        extra = keys - _RAW_REQUIRED_KEYS - _RAW_OPTIONAL_KEYS
        if missing or extra:
            raise ValueError(
                f"standard sample keys mismatch; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        episode = int(torch.as_tensor(sample["episode_index"]).item())
        frame = int(torch.as_tensor(sample["frame_index"]).item())
        rows = self.training.corpus.row_indices[episode]
        if frame < 0 or frame >= len(rows) or rows[frame] != source_index:
            raise ValueError("source episode/frame alignment changed after preparation")
        target = self.training.corpus.targets[episode]
        normalization = self.training.normalization
        state = np.asarray(sample["observation.state"], dtype=np.float32)
        if state.shape != (10,) or not np.isfinite(state).all():
            raise ValueError("model state must be finite with shape [10]")
        task = str(sample["task"])
        if task != self.training.corpus.tasks[episode]:
            raise ValueError("sample task differs from prepared episode language")
        tokens, token_mask = self.training.vocabulary.encode(
            (task,), self.max_language_tokens
        )
        normalized = normalized_graph_v2_frame(target, frame, normalization)
        previous_graph = (
            np.zeros(TOKEN_DIM, dtype=np.float32)
            if frame == 0
            else pack_oracle_target(target, frame - 1, normalization)
        )
        return {
            "agent_rgb": resize_rgb(
                sample["observation.images.agent"], self.image_size, "agent RGB"
            ),
            "wrist_rgb": resize_rgb(
                sample["observation.images.wrist"], self.image_size, "wrist RGB"
            ),
            "state": torch.from_numpy(
                ((state - normalization.state_mean) / normalization.state_std).copy()
            ),
            "language_tokens": torch.from_numpy(tokens),
            "language_mask": torch.from_numpy(token_mask),
            "entity_mask": torch.from_numpy(target.entity_mask[frame].copy()),
            "entity_visibility": torch.from_numpy(
                target.entity_visibility[frame].copy()
            ),
            "relation_mask": torch.from_numpy(target.relation_mask[frame].copy()),
            "gripper_target_geometry": torch.from_numpy(
                normalized["gripper_target_geometry"].copy()
            ),
            "target_receptacle_geometry": torch.from_numpy(
                normalized["target_receptacle_geometry"].copy()
            ),
            "distractor_geometry": torch.from_numpy(
                normalized["distractor_geometry"].copy()
            ),
            "phase": torch.tensor(target.phase[frame], dtype=torch.long),
            "relation_trends": torch.from_numpy(
                normalized["relation_trends"].copy()
            ),
            "goal_relation": torch.tensor(target.goal_relation[frame], dtype=torch.long),
            "goal_operator": torch.tensor(target.goal_operator[frame], dtype=torch.long),
            "goal_predicate": torch.tensor(
                target.goal_predicate[frame], dtype=torch.long
            ),
            "goal_residual": torch.tensor(
                float(normalized["goal_residual"]), dtype=torch.float32
            ),
            "previous_graph": torch.from_numpy(previous_graph),
        }
