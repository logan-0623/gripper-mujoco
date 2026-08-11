from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from interaction_vla.graph_pretrain.reflectvlm import Vocabulary
from interaction_vla.lerobot_bridge.teacher_schema import (
    FORBIDDEN_FIELD_FRAGMENTS,
    SCHEMA_VERSION as TEACHER_SCHEMA,
)

from .schema import MuJoCoGraphTargets, SEMANTIC_CHANNELS


_TARGET_KEYS = {
    "annotation.tc_tig.entity_mask",
    "annotation.tc_tig.entity_visibility",
    "annotation.tc_tig.relation_mask",
    "annotation.tc_tig.relation_values",
    "annotation.tc_tig.relation_goal",
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
    "relation_semantics",
    "goal_relation",
    "goal_operator",
    "goal_predicate",
    "goal_residual",
}


def semantic_targets(arrays: Mapping[str, np.ndarray]) -> MuJoCoGraphTargets:
    missing = _TARGET_KEYS - set(arrays)
    if missing:
        raise ValueError("teacher arrays are missing: " + ", ".join(sorted(missing)))
    relation_values = np.asarray(
        arrays["annotation.tc_tig.relation_values"], dtype=np.float32
    )
    if relation_values.ndim != 3 or relation_values.shape[1:] != (8, 24):
        raise ValueError("teacher relation_values must have shape [frames, 8, 24]")
    goals = np.asarray(arrays["annotation.tc_tig.relation_goal"], dtype=np.float32)
    if goals.shape != (len(relation_values), 5) or not np.isfinite(goals).all():
        raise ValueError("teacher relation_goal must be finite with shape [frames, 5]")
    if not np.equal(goals[:, :3], np.floor(goals[:, :3])).all():
        raise ValueError("teacher relation_goal categorical IDs must be integers")
    return MuJoCoGraphTargets(
        entity_mask=np.asarray(
            arrays["annotation.tc_tig.entity_mask"], dtype=np.bool_
        ),
        entity_visibility=np.asarray(
            arrays["annotation.tc_tig.entity_visibility"], dtype=np.float32
        ),
        relation_mask=np.asarray(
            arrays["annotation.tc_tig.relation_mask"], dtype=np.bool_
        ),
        relation_semantics=relation_values[:, :, SEMANTIC_CHANNELS],
        goal_relation=goals[:, 0].astype(np.int64),
        goal_operator=goals[:, 1].astype(np.int64),
        goal_predicate=goals[:, 2].astype(np.int64),
        goal_residual=goals[:, 3].astype(np.float32),
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


def _float_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


@dataclass(frozen=True)
class GraphNormalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    relation_mean: np.ndarray
    relation_std: np.ndarray
    residual_mean: float
    residual_std: float

    def __post_init__(self) -> None:
        values = {
            "state_mean": _float_array(self.state_mean, (10,), "state_mean"),
            "state_std": _float_array(self.state_std, (10,), "state_std"),
            "relation_mean": _float_array(
                self.relation_mean, (8, 10), "relation_mean"
            ),
            "relation_std": _float_array(
                self.relation_std, (8, 10), "relation_std"
            ),
        }
        if np.any(values["state_std"] <= 0.0) or np.any(
            values["relation_std"] <= 0.0
        ):
            raise ValueError("normalization standard deviations must be positive")
        residual_mean = float(self.residual_mean)
        residual_std = float(self.residual_std)
        if not np.isfinite(residual_mean) or not np.isfinite(residual_std):
            raise ValueError("residual normalization must be finite")
        if residual_std <= 0.0:
            raise ValueError("residual_std must be positive")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "residual_mean", residual_mean)
        object.__setattr__(self, "residual_std", residual_std)


@dataclass(frozen=True)
class PreparedMuJoCoCorpus:
    source: Any
    records: tuple[dict[str, object], ...]
    targets: dict[int, MuJoCoGraphTargets]
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
    normalization: GraphNormalization


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
        int(episode): semantic_targets(arrays)
        for episode, arrays in sidecars.items()
    }
    columns = _source_columns(source)
    required_columns = {
        "observation.state",
        "episode_index",
        "frame_index",
        "task_index",
    }
    missing = required_columns - set(columns)
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
        if len(targets[episode].goal_relation) != frames:
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
) -> GraphNormalization:
    rows = [row for episode in episodes for row in corpus.row_indices[int(episode)]]
    if not rows:
        raise ValueError("normalization requires training rows")
    state_mean, state_std = _safe_mean_std(corpus.states[rows])
    relation_mean = np.zeros((8, 10), dtype=np.float32)
    relation_std = np.ones((8, 10), dtype=np.float32)
    residuals = [corpus.targets[int(episode)].goal_residual for episode in episodes]
    for relation in range(8):
        active_values = []
        for episode in episodes:
            target = corpus.targets[int(episode)]
            active_values.append(
                target.relation_semantics[target.relation_mask[:, relation], relation]
            )
        combined = np.concatenate(active_values, axis=0)
        if len(combined):
            relation_mean[relation], relation_std[relation] = _safe_mean_std(combined)
    residual_values = np.concatenate(residuals)
    residual_mean_values, residual_std_values = _safe_mean_std(
        residual_values[:, None]
    )
    return GraphNormalization(
        state_mean=state_mean,
        state_std=state_std,
        relation_mean=relation_mean,
        relation_std=relation_std,
        residual_mean=float(residual_mean_values[0]),
        residual_std=float(residual_std_values[0]),
    )


def _resize_rgb(value: Any, image_size: int, name: str) -> torch.Tensor:
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
        relation_semantics = (
            target.relation_semantics[frame] - normalization.relation_mean
        ) / normalization.relation_std
        goal_residual = (
            float(target.goal_residual[frame]) - normalization.residual_mean
        ) / normalization.residual_std
        return {
            "agent_rgb": _resize_rgb(
                sample["observation.images.agent"], self.image_size, "agent RGB"
            ),
            "wrist_rgb": _resize_rgb(
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
            "relation_semantics": torch.from_numpy(relation_semantics.copy()),
            "goal_relation": torch.tensor(target.goal_relation[frame], dtype=torch.long),
            "goal_operator": torch.tensor(target.goal_operator[frame], dtype=torch.long),
            "goal_predicate": torch.tensor(
                target.goal_predicate[frame], dtype=torch.long
            ),
            "goal_residual": torch.tensor(goal_residual, dtype=torch.float32),
        }
