from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


STAGE_SCHEMA = "libero_smolvla_stage_v1"


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    suite: str
    task_id: int
    frames: int

    def __post_init__(self) -> None:
        if min(self.episode_index, self.task_id) < 0 or self.frames <= 0:
            raise ValueError("episode identity and frame count are invalid")


@dataclass(frozen=True)
class StageManifest:
    stage: str
    status: str
    base_model: str
    base_revision: str
    dataset_repo_id: str
    dataset_revision: str
    data_fraction: float
    episode_indices: tuple[int, ...]
    subset_sha256: str
    seed: int
    epochs: int
    training_steps: int
    checkpoint: str
    checkpoint_sha256: str | None
    code_hash: str
    config_hash: str
    schema_version: str = STAGE_SCHEMA

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["episode_indices"] = list(self.episode_indices)
        return value


def _subset_hash(episodes: Sequence[int]) -> str:
    payload = json.dumps(list(episodes), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def plan_nested_subsets(
    episodes: Sequence[EpisodeInfo],
    *,
    fractions: tuple[float, ...],
    seed: int,
) -> dict[float, tuple[int, ...]]:
    if (
        not fractions
        or fractions[-1] != 1.0
        or any(left >= right for left, right in zip(fractions, fractions[1:], strict=False))
    ):
        raise ValueError("fractions must be increasing and end at 1.0")
    if len({episode.episode_index for episode in episodes}) != len(episodes):
        raise ValueError("episode indices must be unique")
    by_task: dict[tuple[str, int], list[int]] = {}
    for episode in episodes:
        by_task.setdefault((episode.suite, episode.task_id), []).append(
            episode.episode_index
        )
    ordered_by_task: dict[tuple[str, int], tuple[int, ...]] = {}
    for task_position, (task, indices) in enumerate(sorted(by_task.items())):
        task_seed = np.random.SeedSequence([seed, task_position]).generate_state(1)[0]
        permutation = np.random.default_rng(task_seed).permutation(sorted(indices))
        ordered_by_task[task] = tuple(int(item) for item in permutation)
    result: dict[float, tuple[int, ...]] = {}
    for fraction in fractions:
        selected: list[int] = []
        for indices in ordered_by_task.values():
            count = len(indices) if fraction == 1.0 else max(1, int(math.floor(len(indices) * fraction)))
            selected.extend(indices[:count])
        result[float(fraction)] = tuple(sorted(selected))
    for left, right in zip(fractions, fractions[1:], strict=False):
        if not set(result[left]).issubset(result[right]):
            raise RuntimeError("nested subset construction failed")
    return result


def build_stage_manifests(
    *,
    episodes: Sequence[EpisodeInfo],
    subsets: Mapping[float, tuple[int, ...]],
    output_dir: str | Path,
    base_model: str,
    base_revision: str,
    dataset_repo_id: str,
    dataset_revision: str,
    seed: int,
    epochs: int,
    batch_size: int,
    code_hash: str,
    config_hash: str,
) -> dict[str, StageManifest]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    frame_counts = {episode.episode_index: episode.frames for episode in episodes}
    root = Path(output_dir) / "stages"
    result: dict[str, StageManifest] = {}
    pretrained_path = root / "pretrained" / "checkpoint"
    pretrained_hash = None
    result["pretrained"] = StageManifest(
        stage="pretrained",
        status="complete" if pretrained_hash is not None else "not_run",
        base_model=base_model,
        base_revision=base_revision,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
        data_fraction=0.0,
        episode_indices=(),
        subset_sha256=_subset_hash(()),
        seed=seed,
        epochs=0,
        training_steps=0,
        checkpoint=str(pretrained_path),
        checkpoint_sha256=pretrained_hash,
        code_hash=code_hash,
        config_hash=config_hash,
    )
    for fraction in sorted(subsets):
        episode_indices = tuple(subsets[fraction])
        unknown = sorted(set(episode_indices).difference(frame_counts))
        if unknown:
            raise ValueError(f"subset contains unknown episodes: {unknown}")
        label = f"sft_{int(round(fraction * 100))}"
        checkpoint = root / label / "run" / "checkpoints" / "last" / "pretrained_model"
        checkpoint_hash = None
        total_frames = sum(frame_counts[index] for index in episode_indices)
        result[label] = StageManifest(
            stage=label,
            status="complete" if checkpoint_hash is not None else "not_run",
            base_model=base_model,
            base_revision=base_revision,
            dataset_repo_id=dataset_repo_id,
            dataset_revision=dataset_revision,
            data_fraction=fraction,
            episode_indices=episode_indices,
            subset_sha256=_subset_hash(episode_indices),
            seed=seed,
            epochs=epochs,
            training_steps=math.ceil(total_frames / batch_size) * epochs,
            checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_hash,
            code_hash=code_hash,
            config_hash=config_hash,
        )
    return result
