from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .data import EpisodeArrays, load_episode_arrays
from .models.encoders import SceneBatch

if TYPE_CHECKING:
    from .train import TrainingStatistics


@dataclass(frozen=True)
class SequenceWindow:
    episode_index: int
    start: int
    sample_group: int
    phase: str


@dataclass(frozen=True)
class _PreparedEpisode:
    path: Path
    source_seed: int
    sample_group: int
    node_features: Tensor
    edge_features: Tensor
    node_mask: Tensor
    edge_mask: Tensor
    proprioception: Tensor
    actions: Tensor
    phases: np.ndarray


class EpisodeSequenceDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        *,
        base_paths: tuple[Path, ...],
        recovery_paths: tuple[Path, ...],
        statistics: TrainingStatistics,
        horizon: int,
    ) -> None:
        if horizon < 1:
            raise ValueError("sequence horizon must be positive")
        if not base_paths:
            raise ValueError("at least one base episode is required")
        self.horizon = int(horizon)
        entries = tuple((Path(path), 0) for path in base_paths) + tuple(
            (Path(path), 1) for path in recovery_paths
        )
        loaded = [(path, group, load_episode_arrays(path)) for path, group in entries]
        edge_index = loaded[0][2].edge_index
        if any(
            not np.array_equal(episode.edge_index, edge_index)
            for _, _, episode in loaded[1:]
        ):
            raise ValueError("all sequence episodes must use the same edge ordering")
        self.edge_index = torch.from_numpy(edge_index).long()
        self._episodes = tuple(
            self._prepare_episode(path, group, episode, statistics)
            for path, group, episode in loaded
        )
        self.windows = tuple(
            SequenceWindow(
                episode_index=episode_index,
                start=start,
                sample_group=episode.sample_group,
                phase=str(episode.phases[start]),
            )
            for episode_index, episode in enumerate(self._episodes)
            for start in range(len(episode.actions))
        )
        self.indices_by_group_phase = {
            group: {
                phase: tuple(
                    index
                    for index, window in enumerate(self.windows)
                    if window.sample_group == group and window.phase == phase
                )
                for phase in sorted(
                    {
                        window.phase
                        for window in self.windows
                        if window.sample_group == group
                    }
                )
            }
            for group in (0, 1)
        }

    @staticmethod
    def _prepare_episode(
        path: Path,
        sample_group: int,
        episode: EpisodeArrays,
        statistics: TrainingStatistics,
    ) -> _PreparedEpisode:
        raw_scene = SceneBatch(
            node_features=torch.from_numpy(episode.node_features).float(),
            edge_index=torch.from_numpy(episode.edge_index).long(),
            edge_features=torch.from_numpy(episode.edge_features).float(),
            node_mask=torch.from_numpy(episode.node_mask).bool(),
            edge_mask=torch.from_numpy(episode.edge_mask).bool(),
        )
        normalized_scene = statistics.normalize_scene(raw_scene)
        normalized_proprioception = statistics.normalize_proprioception(
            torch.from_numpy(episode.proprioception).float()
        )
        source_seed = (
            episode.seed
            if sample_group == 0
            else int(
                episode.source_seed
                if episode.source_seed is not None
                else episode.seed
            )
        )
        return _PreparedEpisode(
            path=path,
            source_seed=int(source_seed),
            sample_group=int(sample_group),
            node_features=normalized_scene.node_features,
            edge_features=normalized_scene.edge_features,
            node_mask=normalized_scene.node_mask,
            edge_mask=normalized_scene.edge_mask,
            proprioception=normalized_proprioception,
            actions=torch.from_numpy(episode.actions).float(),
            phases=episode.phases.copy(),
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        window = self.windows[index]
        episode = self._episodes[window.episode_index]
        available = min(self.horizon, len(episode.actions) - window.start)
        action_chunk = torch.zeros(
            (self.horizon, episode.actions.shape[-1]),
            dtype=episode.actions.dtype,
        )
        action_chunk[:available] = episode.actions[
            window.start : window.start + available
        ]
        horizon_mask = torch.zeros(self.horizon, dtype=torch.bool)
        horizon_mask[:available] = True
        return {
            "node_features": episode.node_features[window.start],
            "edge_features": episode.edge_features[window.start],
            "node_mask": episode.node_mask[window.start],
            "edge_mask": episode.edge_mask[window.start],
            "proprioception": episode.proprioception[window.start],
            "action_chunk": action_chunk,
            "horizon_mask": horizon_mask,
            "sample_group": torch.tensor(window.sample_group, dtype=torch.long),
            "phase": window.phase,
            "source_seed": torch.tensor(episode.source_seed, dtype=torch.long),
            "episode_path": str(episode.path),
            "frame_index": torch.tensor(window.start, dtype=torch.long),
        }


class _PhaseDrawer:
    def __init__(
        self,
        indices_by_phase: Mapping[str, tuple[int, ...]],
        *,
        seed: int,
    ) -> None:
        self.phases = tuple(sorted(indices_by_phase))
        if not self.phases or any(not indices_by_phase[phase] for phase in self.phases):
            raise ValueError("each sampled group requires non-empty phases")
        self._values = {
            phase: np.asarray(indices_by_phase[phase], dtype=np.int64)
            for phase in self.phases
        }
        self._rngs = {
            phase: np.random.default_rng(
                np.random.SeedSequence((int(seed), phase_index, 0x50484153))
            )
            for phase_index, phase in enumerate(self.phases)
        }
        self._queues: dict[str, list[int]] = {phase: [] for phase in self.phases}

    def draw(self, phase: str, count: int) -> list[int]:
        selected: list[int] = []
        while len(selected) < count:
            if not self._queues[phase]:
                permutation = self._rngs[phase].permutation(self._values[phase])
                self._queues[phase].extend(int(value) for value in permutation)
            take = min(count - len(selected), len(self._queues[phase]))
            selected.extend(self._queues[phase][:take])
            del self._queues[phase][:take]
        return selected


class StratifiedSequenceBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        base_indices_by_phase: Mapping[str, Iterable[int]],
        recovery_indices_by_phase: Mapping[str, Iterable[int]],
        batch_size: int,
        recovery_fraction: float,
        seed: int,
    ) -> None:
        if batch_size < 2:
            raise ValueError("stratified batch size must be at least two")
        if not 0.0 < recovery_fraction < 1.0:
            raise ValueError("recovery fraction must be within (0, 1)")
        recovery_count_float = batch_size * recovery_fraction
        recovery_count = int(round(recovery_count_float))
        if not math.isclose(recovery_count_float, recovery_count, abs_tol=1e-12):
            raise ValueError(
                "recovery fraction must produce an integer batch group count"
            )
        self.recovery_per_batch = recovery_count
        self.base_per_batch = batch_size - recovery_count
        if min(self.base_per_batch, self.recovery_per_batch) < 1:
            raise ValueError("each batch must contain base and recovery samples")
        self.base_indices_by_phase = self._canonical_groups(base_indices_by_phase)
        self.recovery_indices_by_phase = self._canonical_groups(
            recovery_indices_by_phase
        )
        base_indices = tuple(
            index
            for indices in self.base_indices_by_phase.values()
            for index in indices
        )
        recovery_indices = tuple(
            index
            for indices in self.recovery_indices_by_phase.values()
            for index in indices
        )
        if not base_indices or not recovery_indices:
            raise ValueError("both base and recovery sampler groups are required")
        if len(set(base_indices)) != len(base_indices):
            raise ValueError("base sampler indices must be unique")
        if len(set(recovery_indices)) != len(recovery_indices):
            raise ValueError("recovery sampler indices must be unique")
        if set(base_indices) & set(recovery_indices):
            raise ValueError("base and recovery sampler indices must be disjoint")
        self.batch_size = int(batch_size)
        self.recovery_fraction = float(recovery_fraction)
        self.seed = int(seed)
        self.epoch = 0
        self._length = math.ceil(len(base_indices) / self.base_per_batch)

    @staticmethod
    def _canonical_groups(
        values: Mapping[str, Iterable[int]],
    ) -> dict[str, tuple[int, ...]]:
        result: dict[str, tuple[int, ...]] = {}
        for phase, indices in sorted(values.items(), key=lambda item: str(item[0])):
            resolved = tuple(int(index) for index in indices)
            if resolved:
                result[str(phase)] = resolved
        return result

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def _phase_quotas(
        phases: tuple[str, ...],
        total: int,
        *,
        offset: int,
    ) -> dict[str, int]:
        quotient, remainder = divmod(total, len(phases))
        quotas = {phase: quotient for phase in phases}
        for index in range(remainder):
            quotas[phases[(offset + index) % len(phases)]] += 1
        return quotas

    def __iter__(self):
        epoch_seed = int(
            np.random.SeedSequence((self.seed, self.epoch, 0x53414D50))
            .generate_state(1, dtype=np.uint32)[0]
        )
        base_drawer = _PhaseDrawer(
            self.base_indices_by_phase,
            seed=epoch_seed ^ 0x42415345,
        )
        recovery_drawer = _PhaseDrawer(
            self.recovery_indices_by_phase,
            seed=epoch_seed ^ 0x5245434F,
        )
        batch_rng = np.random.default_rng(
            np.random.SeedSequence((epoch_seed, 0x42415443))
        )
        for batch_index in range(self._length):
            batch: list[int] = []
            base_quotas = self._phase_quotas(
                base_drawer.phases,
                self.base_per_batch,
                offset=batch_index + self.epoch,
            )
            recovery_quotas = self._phase_quotas(
                recovery_drawer.phases,
                self.recovery_per_batch,
                offset=batch_index + self.epoch,
            )
            for phase in base_drawer.phases:
                batch.extend(base_drawer.draw(phase, base_quotas[phase]))
            for phase in recovery_drawer.phases:
                batch.extend(recovery_drawer.draw(phase, recovery_quotas[phase]))
            yield [int(batch[index]) for index in batch_rng.permutation(len(batch))]


@dataclass(frozen=True)
class SequenceLoss:
    total: Tensor
    base: Tensor
    recovery: Tensor


def sequence_behavior_cloning_loss(
    prediction: Tensor,
    target: Tensor,
    horizon_mask: Tensor,
    sample_group: Tensor,
    *,
    future_loss_decay: float,
    recovery_loss_fraction: float,
) -> SequenceLoss:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("sequence prediction and target must share shape [B, H, A]")
    if horizon_mask.shape != prediction.shape[:2]:
        raise ValueError("horizon mask must have shape [B, H]")
    if sample_group.shape != prediction.shape[:1]:
        raise ValueError("sample group must have shape [B]")
    if not 0.0 < future_loss_decay <= 1.0:
        raise ValueError("future loss decay must be within (0, 1]")
    if not 0.0 < recovery_loss_fraction < 1.0:
        raise ValueError("recovery loss fraction must be within (0, 1)")
    base_mask = sample_group == 0
    recovery_mask = sample_group == 1
    if not bool(base_mask.any()) or not bool(recovery_mask.any()):
        raise ValueError("sequence loss requires both base and recovery samples")
    if bool(((sample_group != 0) & (sample_group != 1)).any()):
        raise ValueError("sample group values must be zero or one")
    horizon_weights = torch.pow(
        torch.as_tensor(
            future_loss_decay,
            dtype=prediction.dtype,
            device=prediction.device,
        ),
        torch.arange(
            prediction.shape[1],
            dtype=prediction.dtype,
            device=prediction.device,
        ),
    )
    valid_weights = horizon_mask.to(prediction.dtype) * horizon_weights.unsqueeze(0)
    denominators = valid_weights.sum(dim=1)
    if bool((denominators <= 0.0).any()):
        raise ValueError("every sequence sample requires at least one valid target")
    per_step = (prediction - target).square().mean(dim=-1)
    per_sample = (per_step * valid_weights).sum(dim=1) / denominators
    base_mean = per_sample[base_mask].mean()
    recovery_mean = per_sample[recovery_mask].mean()
    total = (
        (1.0 - recovery_loss_fraction) * base_mean
        + recovery_loss_fraction * recovery_mean
    )
    return SequenceLoss(total=total, base=base_mean, recovery=recovery_mean)
