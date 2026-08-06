from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import ExperimentConfig, load_config
from .data import (
    EpisodeSplits,
    episode_paths_from_manifest,
    load_episode_arrays,
    recovery_paths_from_manifest,
    split_episode_seeds,
)
from .device import resolve_device
from .models.encoders import SceneBatch
from .models.policy import ActionPolicy, build_action_policy
from .sequence_training import (
    EpisodeSequenceDataset,
    StratifiedSequenceBatchSampler,
    sequence_behavior_cloning_loss,
)
from .source_split import load_source_data_layout
from .physics_provenance import (
    learned_rollout_source_hash,
    training_pipeline_source_hash,
)


def _validate_resume_contract(
    payload: Mapping[str, object],
    checkpoint_metadata: Mapping[str, object] | None,
) -> None:
    expected = dict(checkpoint_metadata or {})
    for key in (
        "representation_contract",
        "temporal_contract",
        "training_pipeline_source_hash",
        "learned_rollout_source_hash",
    ):
        if key in expected and payload.get(key) != expected[key]:
            raise ValueError(f"resume checkpoint differs for {key}")


def _safe_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    standard_deviation = values.std(axis=axis).astype(np.float32)
    return np.where(standard_deviation < 1e-6, 1.0, standard_deviation).astype(np.float32)


@dataclass(frozen=True)
class TrainingStatistics:
    node_mean: np.ndarray
    node_std: np.ndarray
    edge_mean: np.ndarray
    edge_std: np.ndarray
    proprio_mean: np.ndarray
    proprio_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @classmethod
    def fit(cls, episode_paths: Iterable[str | Path]) -> "TrainingStatistics":
        episodes = [load_episode_arrays(path) for path in episode_paths]
        if not episodes:
            raise ValueError("at least one training episode is required")
        valid_nodes = np.concatenate(
            [episode.node_features[episode.node_mask] for episode in episodes], axis=0
        )
        valid_edges = np.concatenate(
            [episode.edge_features[episode.edge_mask] for episode in episodes], axis=0
        )
        proprioception = np.concatenate([episode.proprioception for episode in episodes], axis=0)
        actions = np.concatenate([episode.actions for episode in episodes], axis=0)
        return cls(
            node_mean=valid_nodes.mean(axis=0).astype(np.float32),
            node_std=_safe_std(valid_nodes),
            edge_mean=valid_edges.mean(axis=0).astype(np.float32),
            edge_std=_safe_std(valid_edges),
            proprio_mean=proprioception.mean(axis=0).astype(np.float32),
            proprio_std=_safe_std(proprioception),
            action_mean=actions.mean(axis=0).astype(np.float32),
            action_std=_safe_std(actions),
        )

    def normalize_scene(self, batch: SceneBatch) -> SceneBatch:
        device = batch.node_features.device
        node_mean = torch.as_tensor(self.node_mean, device=device)
        node_std = torch.as_tensor(self.node_std, device=device)
        edge_mean = torch.as_tensor(self.edge_mean, device=device)
        edge_std = torch.as_tensor(self.edge_std, device=device)
        node_mask = batch.node_mask.unsqueeze(-1)
        edge_mask = batch.edge_mask.unsqueeze(-1)
        return SceneBatch(
            node_features=((batch.node_features - node_mean) / node_std) * node_mask,
            edge_index=batch.edge_index,
            edge_features=((batch.edge_features - edge_mean) / edge_std) * edge_mask,
            node_mask=batch.node_mask,
            edge_mask=batch.edge_mask,
        )

    def normalize_proprioception(self, values: Tensor) -> Tensor:
        mean = torch.as_tensor(self.proprio_mean, device=values.device)
        std = torch.as_tensor(self.proprio_std, device=values.device)
        return (values - mean) / std

    def normalize_actions(self, values: Tensor) -> Tensor:
        mean = torch.as_tensor(self.action_mean, device=values.device)
        std = torch.as_tensor(self.action_std, device=values.device)
        return (values - mean) / std

    def checkpoint_state(self) -> dict[str, Tensor]:
        return {
            name: torch.from_numpy(getattr(self, name)).clone()
            for name in (
                "node_mean",
                "node_std",
                "edge_mean",
                "edge_std",
                "proprio_mean",
                "proprio_std",
                "action_mean",
                "action_std",
            )
        }

    @classmethod
    def from_checkpoint_state(cls, state: Mapping[str, Tensor]) -> "TrainingStatistics":
        return cls(**{name: value.detach().cpu().numpy() for name, value in state.items()})


class EpisodeFrameDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        episode_paths: Iterable[str | Path],
        statistics: TrainingStatistics,
    ) -> None:
        episodes = [load_episode_arrays(path) for path in episode_paths]
        if not episodes:
            raise ValueError("at least one episode is required")
        edge_index = episodes[0].edge_index
        if any(not np.array_equal(episode.edge_index, edge_index) for episode in episodes[1:]):
            raise ValueError("all episodes must use the same edge ordering")
        self.edge_index = torch.from_numpy(edge_index).long()
        raw_batch = SceneBatch(
            node_features=torch.from_numpy(
                np.concatenate([episode.node_features for episode in episodes], axis=0)
            ).float(),
            edge_index=self.edge_index,
            edge_features=torch.from_numpy(
                np.concatenate([episode.edge_features for episode in episodes], axis=0)
            ).float(),
            node_mask=torch.from_numpy(
                np.concatenate([episode.node_mask for episode in episodes], axis=0)
            ).bool(),
            edge_mask=torch.from_numpy(
                np.concatenate([episode.edge_mask for episode in episodes], axis=0)
            ).bool(),
        )
        normalized = statistics.normalize_scene(raw_batch)
        self.node_features = normalized.node_features
        self.edge_features = normalized.edge_features
        self.node_mask = normalized.node_mask
        self.edge_mask = normalized.edge_mask
        raw_proprioception = torch.from_numpy(
            np.concatenate([episode.proprioception for episode in episodes], axis=0)
        ).float()
        self.proprioception = statistics.normalize_proprioception(raw_proprioception)
        self.actions = torch.from_numpy(
            np.concatenate([episode.actions for episode in episodes], axis=0)
        ).float()
        self.phases = np.concatenate([episode.phases for episode in episodes], axis=0)
        phase_names, phase_counts = np.unique(self.phases, return_counts=True)
        count_by_phase = dict(zip(phase_names.tolist(), phase_counts.tolist(), strict=True))
        phase_count = len(phase_names)
        frame_count = len(self.phases)
        self.sample_weights = torch.as_tensor(
            [
                frame_count / (phase_count * count_by_phase[str(phase)])
                for phase in self.phases
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.actions.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "node_features": self.node_features[index],
            "edge_features": self.edge_features[index],
            "node_mask": self.node_mask[index],
            "edge_mask": self.edge_mask[index],
            "proprioception": self.proprioception[index],
            "actions": self.actions[index],
            "sample_weights": self.sample_weights[index],
        }

    def scene_batch(self, indices: Tensor | None = None) -> SceneBatch:
        selection = slice(None) if indices is None else indices
        return SceneBatch(
            node_features=self.node_features[selection],
            edge_index=self.edge_index,
            edge_features=self.edge_features[selection],
            node_mask=self.node_mask[selection],
            edge_mask=self.edge_mask[selection],
        )


@dataclass(frozen=True)
class TrainingResult:
    global_step: int
    completed_epochs: int
    final_normalized_mse: float
    final_base_mse: float | None = None
    final_recovery_mse: float | None = None


@dataclass(frozen=True)
class TrainingDataSelection:
    splits: EpisodeSplits
    base_train_paths: tuple[Path, ...]
    recovery_train_paths: tuple[Path, ...]
    validation_paths: tuple[Path, ...] = ()
    test_paths: tuple[Path, ...] = ()
    recovery_benchmark_paths: tuple[Path, ...] = ()
    source_split_hash: str = ""
    manifest_hash: str = ""
    recovery_manifest_hash: str = ""
    recovery_benchmark_manifest_hash: str = ""

    @property
    def base_training_paths(self) -> tuple[Path, ...]:
        return self.base_train_paths

    @property
    def recovery_paths(self) -> tuple[Path, ...]:
        return self.recovery_train_paths

    @property
    def combined_paths(self) -> tuple[Path, ...]:
        return self.base_train_paths + self.recovery_train_paths

    def provenance(self) -> dict[str, object]:
        return {
            "base_train_seeds": list(self.splits.train),
            "recovery_filenames": [path.name for path in self.recovery_train_paths],
            "recovery_count": len(self.recovery_train_paths),
            "source_split_hash": self.source_split_hash,
            "manifest_hash": self.manifest_hash,
            "recovery_manifest_hash": self.recovery_manifest_hash,
            "recovery_benchmark_filenames": [
                path.name for path in self.recovery_benchmark_paths
            ],
            "recovery_benchmark_manifest_hash": (
                self.recovery_benchmark_manifest_hash
            ),
        }


def inspect_episode_dimensions(
    episode_paths: Iterable[str | Path],
) -> dict[str, int]:
    episodes = [load_episode_arrays(path) for path in episode_paths]
    if not episodes:
        raise ValueError("at least one episode is required to inspect dimensions")
    first = episodes[0]
    dimensions = {
        "node_feature_dim": int(first.node_features.shape[-1]),
        "edge_feature_dim": int(first.edge_features.shape[-1]),
        "proprioception_dim": int(first.proprioception.shape[-1]),
        "action_dim": int(first.actions.shape[-1]),
    }
    for episode in episodes[1:]:
        observed = {
            "node_feature_dim": int(episode.node_features.shape[-1]),
            "edge_feature_dim": int(episode.edge_features.shape[-1]),
            "proprioception_dim": int(episode.proprioception.shape[-1]),
            "action_dim": int(episode.actions.shape[-1]),
        }
        if observed != dimensions:
            raise ValueError("all training episodes must use identical feature dimensions")
    return dimensions


def hash_episode_contents(episode_paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    paths = tuple(Path(path) for path in episode_paths)
    if not paths:
        raise ValueError("at least one episode is required to hash dataset contents")
    for path in paths:
        name = path.name.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def hash_training_statistics(statistics: TrainingStatistics) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(statistics.checkpoint_state().items()):
        encoded_name = name.encode("utf-8")
        array = value.detach().cpu().numpy()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(array.shape.__repr__().encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_training_provenance(
    config: ExperimentConfig,
    selection: TrainingDataSelection,
    *,
    expert_gate_hash: str,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    if selection.source_split_hash:
        all_episode_paths = (
            selection.base_train_paths
            + selection.validation_paths
            + selection.test_paths
            + selection.recovery_train_paths
            + selection.recovery_benchmark_paths
        )
        provenance = selection.provenance()
        provenance.update(
            {
                "backend": config.backend,
                "expert_gate_hash": expert_gate_hash,
                "dataset_content_hash": hash_episode_contents(all_episode_paths),
            }
        )
        return provenance, all_episode_paths
    all_episode_paths = episode_paths_from_manifest(config.data_dir)
    if config.recovery.enabled:
        all_episode_paths += recovery_paths_from_manifest(config.data_dir)
    provenance = selection.provenance()
    manifest_bytes = (Path(config.data_dir) / "manifest.json").read_bytes()
    provenance.update(
        {
            "backend": config.backend,
            "manifest_hash": hashlib.sha256(manifest_bytes).hexdigest(),
            "expert_gate_hash": expert_gate_hash,
            "dataset_content_hash": hash_episode_contents(all_episode_paths),
        }
    )
    if config.recovery.enabled:
        recovery_manifest_bytes = (
            Path(config.data_dir) / "recovery_manifest.json"
        ).read_bytes()
        provenance["recovery_manifest_hash"] = hashlib.sha256(
            recovery_manifest_bytes
        ).hexdigest()
    return provenance, all_episode_paths


def _sequence_provenance_fields(
    statistics: TrainingStatistics,
    dataset: EpisodeSequenceDataset,
    sampler: StratifiedSequenceBatchSampler,
) -> dict[str, object]:
    return {
        "statistics_hash": hash_training_statistics(statistics),
        "base_window_count": sum(
            len(indices)
            for indices in dataset.indices_by_group_phase[0].values()
        ),
        "recovery_window_count": sum(
            len(indices)
            for indices in dataset.indices_by_group_phase[1].values()
        ),
        "base_phase_counts": {
            phase: len(indices)
            for phase, indices in dataset.indices_by_group_phase[0].items()
        },
        "recovery_phase_counts": {
            phase: len(indices)
            for phase, indices in dataset.indices_by_group_phase[1].items()
        },
        "base_per_batch": sampler.base_per_batch,
        "recovery_per_batch": sampler.recovery_per_batch,
    }


def build_sequence_provenance_fields(
    config: ExperimentConfig,
    selection: TrainingDataSelection,
    *,
    model_seed: int,
) -> dict[str, object]:
    if not config.sequence.enabled:
        return {}
    statistics = TrainingStatistics.fit(selection.base_train_paths)
    dataset = EpisodeSequenceDataset(
        base_paths=selection.base_train_paths,
        recovery_paths=selection.recovery_train_paths,
        statistics=statistics,
        horizon=config.sequence.horizon,
    )
    sampler = StratifiedSequenceBatchSampler(
        base_indices_by_phase=dataset.indices_by_group_phase[0],
        recovery_indices_by_phase=dataset.indices_by_group_phase[1],
        batch_size=config.train.batch_size,
        recovery_fraction=config.sequence.recovery_loss_fraction,
        seed=int(model_seed),
    )
    return _sequence_provenance_fields(statistics, dataset, sampler)


def resolve_training_data(
    data_dir: str | Path,
    *,
    split_seed: int | None = None,
    include_recovery: bool,
) -> TrainingDataSelection:
    directory = Path(data_dir)
    if (directory / "source_split.json").is_file():
        layout = load_source_data_layout(directory)
        recovery_paths = (
            layout.training_recovery_paths if include_recovery else ()
        )
        return TrainingDataSelection(
            splits=EpisodeSplits(
                train=layout.split.train,
                validation=layout.split.validation,
                test=layout.split.test,
            ),
            base_train_paths=layout.base_by_split["train"],
            recovery_train_paths=recovery_paths,
            validation_paths=layout.base_by_split["validation"],
            test_paths=layout.base_by_split["test"],
            recovery_benchmark_paths=layout.benchmark_recovery_paths,
            source_split_hash=layout.source_split_hash,
            manifest_hash=layout.manifest_hash,
            recovery_manifest_hash=layout.training_recovery_manifest_hash,
            recovery_benchmark_manifest_hash=(
                layout.benchmark_recovery_manifest_hash
            ),
        )
    if split_seed is None:
        raise ValueError("legacy training data requires split_seed")
    base_paths = episode_paths_from_manifest(data_dir)
    base_by_path = {path: load_episode_arrays(path) for path in base_paths}
    splits = split_episode_seeds(
        (episode.seed for episode in base_by_path.values()),
        validation_fraction=0.1,
        test_fraction=0.1,
        seed=split_seed,
    )
    training_seeds = set(splits.train)
    base_training_paths = tuple(
        path for path, episode in base_by_path.items() if episode.seed in training_seeds
    )
    recovery_paths: tuple[Path, ...] = ()
    if include_recovery:
        recovery_paths = recovery_paths_from_manifest(data_dir)
        for path in recovery_paths:
            episode = load_episode_arrays(path)
            if episode.trajectory_kind != "recovery":
                raise ValueError(f"recovery manifest contains a non-recovery episode: {path}")
            if episode.source_seed not in training_seeds:
                raise ValueError(
                    f"recovery source seed {episode.source_seed} is outside the base training split"
                )
    return TrainingDataSelection(
        splits=splits,
        base_train_paths=base_training_paths,
        recovery_train_paths=recovery_paths,
        validation_paths=tuple(
            path
            for path, episode in base_by_path.items()
            if episode.seed in set(splits.validation)
        ),
        test_paths=tuple(
            path
            for path, episode in base_by_path.items()
            if episode.seed in set(splits.test)
        ),
    )


def _batch_to_inputs(batch: Mapping[str, Tensor], edge_index: Tensor, device: torch.device):
    scene = SceneBatch(
        node_features=batch["node_features"].to(device),
        edge_index=edge_index.to(device),
        edge_features=batch["edge_features"].to(device),
        node_mask=batch["node_mask"].to(device),
        edge_mask=batch["edge_mask"].to(device),
    )
    return (
        scene,
        batch["proprioception"].to(device),
        batch["actions"].to(device),
        batch["sample_weights"].to(device),
    )


@torch.no_grad()
def evaluate_normalized_mse(
    policy: ActionPolicy,
    dataset: EpisodeFrameDataset,
    statistics: TrainingStatistics,
    device: torch.device | str,
) -> float:
    resolved_device = torch.device(device)
    policy.to(resolved_device).eval()
    scene = dataset.scene_batch().to(resolved_device)
    proprioception = dataset.proprioception.to(resolved_device)
    actions = dataset.actions.to(resolved_device)
    predictions = policy(scene if policy.scene_encoder is not None else None, proprioception)
    error = statistics.normalize_actions(predictions) - statistics.normalize_actions(actions)
    return float(torch.mean(error.square()).item())


def train_policy(
    policy: ActionPolicy,
    dataset: EpisodeFrameDataset,
    statistics: TrainingStatistics,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device | str,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    representation: str = "flat",
    model_kwargs: Mapping[str, object] | None = None,
    metrics_path: str | Path | None = None,
    training_provenance: Mapping[str, object] | None = None,
    checkpoint_metadata: Mapping[str, object] | None = None,
    show_progress: bool = False,
) -> TrainingResult:
    if min(epochs, batch_size) < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    resolved_device = torch.device(device)
    torch.manual_seed(seed)
    policy.to(resolved_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    global_step = 0
    completed_epochs = 0
    current_provenance = dict(training_provenance or {})
    if resume_from is not None:
        payload = torch.load(Path(resume_from), map_location=resolved_device, weights_only=False)
        _validate_resume_contract(payload, checkpoint_metadata)
        if dict(payload.get("training_provenance", {})) != current_provenance:
            raise ValueError(
                "training provenance differs from the checkpoint; refusing to resume"
            )
        restored_statistics = TrainingStatistics.from_checkpoint_state(payload["statistics"])
        statistics_match = all(
            np.array_equal(getattr(statistics, name), getattr(restored_statistics, name))
            for name in statistics.checkpoint_state()
        )
        if not statistics_match:
            raise ValueError(
                "resume statistics differ from the checkpoint; rebuild the dataset "
                "with the checkpoint training split and normalization"
            )
        policy.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        global_step = int(payload["global_step"])
        completed_epochs = int(payload["completed_epochs"])

    metrics_destination = Path(metrics_path) if metrics_path is not None else None
    if metrics_destination is not None:
        metrics_destination.parent.mkdir(parents=True, exist_ok=True)
        if resume_from is None:
            metrics_destination.write_text("", encoding="utf-8")

    policy.train()
    progress = (
        tqdm(
            range(epochs),
            desc=f"{representation} seed={seed}",
            total=epochs,
            unit="epoch",
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )
    epoch_indices = progress if progress is not None else range(epochs)
    for _ in epoch_indices:
        generator = torch.Generator().manual_seed(seed + completed_epochs)
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for batch in loader:
            scene, proprioception, actions, sample_weights = _batch_to_inputs(
                batch, dataset.edge_index, resolved_device
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = policy(
                scene if policy.scene_encoder is not None else None, proprioception
            )
            normalized_error = (
                statistics.normalize_actions(predictions)
                - statistics.normalize_actions(actions)
            )
            per_sample_loss = normalized_error.square().mean(dim=-1)
            loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("training loss became non-finite")
            loss.backward()
            optimizer.step()
            global_step += 1
            weighted_loss_sum += float(
                (per_sample_loss.detach() * sample_weights).sum().item()
            )
            weight_sum += float(sample_weights.sum().item())
        completed_epochs += 1
        epoch_mse = weighted_loss_sum / weight_sum
        if progress is not None:
            progress.set_postfix(mse=f"{epoch_mse:.6f}")
        if metrics_destination is not None:
            with metrics_destination.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": completed_epochs,
                            "global_step": global_step,
                            "weighted_train_mse": epoch_mse,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    final_mse = evaluate_normalized_mse(policy, dataset, statistics, resolved_device)
    if checkpoint_path is not None:
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "global_step": global_step,
                "completed_epochs": completed_epochs,
                "representation": representation,
                "model_seed": seed,
                "model_kwargs": dict(model_kwargs or {}),
                "statistics": statistics.checkpoint_state(),
                "training_provenance": current_provenance,
            }
        checkpoint_payload.update(dict(checkpoint_metadata or {}))
        torch.save(checkpoint_payload, destination)
    return TrainingResult(
        global_step=global_step,
        completed_epochs=completed_epochs,
        final_normalized_mse=final_mse,
    )


def _sequence_batch_to_inputs(
    batch: Mapping[str, object],
    edge_index: Tensor,
    device: torch.device,
) -> tuple[SceneBatch, Tensor, Tensor, Tensor, Tensor]:
    scene = SceneBatch(
        node_features=batch["node_features"].to(device),
        edge_index=edge_index.to(device),
        edge_features=batch["edge_features"].to(device),
        node_mask=batch["node_mask"].to(device),
        edge_mask=batch["edge_mask"].to(device),
    )
    return (
        scene,
        batch["proprioception"].to(device),
        batch["action_chunk"].to(device),
        batch["horizon_mask"].to(device),
        batch["sample_group"].to(device),
    )


def train_sequence_policy(
    policy: ActionPolicy,
    dataset: EpisodeSequenceDataset,
    sampler: StratifiedSequenceBatchSampler,
    statistics: TrainingStatistics,
    *,
    epochs: int,
    learning_rate: float,
    future_loss_decay: float,
    recovery_loss_fraction: float,
    seed: int,
    device: torch.device | str,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    representation: str = "flat",
    model_kwargs: Mapping[str, object] | None = None,
    metrics_path: str | Path | None = None,
    training_provenance: Mapping[str, object] | None = None,
    checkpoint_metadata: Mapping[str, object] | None = None,
    show_progress: bool = False,
) -> TrainingResult:
    if epochs < 1 or learning_rate <= 0.0:
        raise ValueError("epochs and learning_rate must be positive")
    resolved_device = torch.device(device)
    torch.manual_seed(seed)
    policy.to(resolved_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    global_step = 0
    completed_epochs = 0
    current_provenance = dict(training_provenance or {})
    if resume_from is not None:
        payload = torch.load(
            Path(resume_from),
            map_location=resolved_device,
            weights_only=False,
        )
        _validate_resume_contract(payload, checkpoint_metadata)
        if dict(payload.get("training_provenance", {})) != current_provenance:
            raise ValueError(
                "training provenance differs from the checkpoint; refusing to resume"
            )
        restored_statistics = TrainingStatistics.from_checkpoint_state(
            payload["statistics"]
        )
        if hash_training_statistics(restored_statistics) != hash_training_statistics(
            statistics
        ):
            raise ValueError("resume statistics differ from the checkpoint")
        policy.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        global_step = int(payload["global_step"])
        completed_epochs = int(payload["completed_epochs"])

    metrics_destination = Path(metrics_path) if metrics_path is not None else None
    if metrics_destination is not None:
        metrics_destination.parent.mkdir(parents=True, exist_ok=True)
        if resume_from is None:
            metrics_destination.write_text("", encoding="utf-8")
    progress = (
        tqdm(
            range(epochs),
            desc=f"{representation} seed={seed}",
            total=epochs,
            unit="epoch",
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )
    epoch_indices = progress if progress is not None else range(epochs)
    final_total = float("nan")
    final_base = float("nan")
    final_recovery = float("nan")
    policy.train()
    for _ in epoch_indices:
        sampler.set_epoch(completed_epochs)
        loader = DataLoader(dataset, batch_sampler=sampler)
        total_sum = 0.0
        base_sum = 0.0
        recovery_sum = 0.0
        batch_count = 0
        for batch in loader:
            scene, proprioception, actions, horizon_mask, sample_group = (
                _sequence_batch_to_inputs(
                    batch,
                    dataset.edge_index,
                    resolved_device,
                )
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = policy.predict_action_chunk(
                scene if policy.scene_encoder is not None else None,
                proprioception,
            )
            loss = sequence_behavior_cloning_loss(
                statistics.normalize_actions(predictions),
                statistics.normalize_actions(actions),
                horizon_mask,
                sample_group,
                future_loss_decay=future_loss_decay,
                recovery_loss_fraction=recovery_loss_fraction,
            )
            if not torch.isfinite(loss.total):
                raise FloatingPointError("sequence training loss became non-finite")
            loss.total.backward()
            optimizer.step()
            global_step += 1
            batch_count += 1
            total_sum += float(loss.total.detach().item())
            base_sum += float(loss.base.detach().item())
            recovery_sum += float(loss.recovery.detach().item())
        if batch_count == 0:
            raise RuntimeError("sequence sampler produced no batches")
        completed_epochs += 1
        final_total = total_sum / batch_count
        final_base = base_sum / batch_count
        final_recovery = recovery_sum / batch_count
        if progress is not None:
            progress.set_postfix(
                mse=f"{final_total:.6f}",
                base=f"{final_base:.6f}",
                recovery=f"{final_recovery:.6f}",
                mix=f"{sampler.base_per_batch}+{sampler.recovery_per_batch}",
            )
        if metrics_destination is not None:
            with metrics_destination.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": completed_epochs,
                            "global_step": global_step,
                            "total_train_mse": final_total,
                            "base_train_mse": final_base,
                            "recovery_train_mse": final_recovery,
                            "base_per_batch": sampler.base_per_batch,
                            "recovery_per_batch": sampler.recovery_per_batch,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    if checkpoint_path is not None:
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "model_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
            "completed_epochs": completed_epochs,
            "representation": representation,
            "model_seed": seed,
            "model_kwargs": dict(model_kwargs or {}),
            "statistics": statistics.checkpoint_state(),
            "statistics_hash": hash_training_statistics(statistics),
            "training_provenance": current_provenance,
        }
        checkpoint_payload.update(dict(checkpoint_metadata or {}))
        torch.save(checkpoint_payload, destination)
    return TrainingResult(
        global_step=global_step,
        completed_epochs=completed_epochs,
        final_normalized_mse=final_total,
        final_base_mse=final_base,
        final_recovery_mse=final_recovery,
    )


def load_training_checkpoint(
    path: str | Path, device: torch.device | str
) -> tuple[ActionPolicy, TrainingStatistics, dict]:
    resolved_device = torch.device(device)
    payload = torch.load(Path(path), map_location=resolved_device, weights_only=False)
    policy = build_action_policy(
        representation=str(payload["representation"]), **dict(payload["model_kwargs"])
    )
    policy.load_state_dict(payload["model_state"])
    policy.to(resolved_device)
    statistics = TrainingStatistics.from_checkpoint_state(payload["statistics"])
    return policy, statistics, payload


def train_from_config(
    config_path: str | Path,
    representation: str,
    model_seed: int | None = None,
    resume: bool = False,
) -> Path:
    cfg = load_config(config_path)
    gate_hash = ""
    physical_hashes: dict[str, str] = {}
    if cfg.backend == "franka_contact":
        from .physics_data import expert_gate_provenance

        physical_hashes = expert_gate_provenance(
            config_path, Path(cfg.output_dir) / "expert_gate.json"
        )
        gate_hash = physical_hashes["expert_gate_hash"]
    device = resolve_device(cfg.device)
    selection = resolve_training_data(
        cfg.data_dir,
        split_seed=cfg.seed,
        include_recovery=cfg.recovery.enabled,
    )
    splits = selection.splits
    training_paths = selection.combined_paths
    provenance, all_episode_paths = build_training_provenance(
        cfg, selection, expert_gate_hash=gate_hash
    )
    if cfg.backend == "franka_contact":
        from .physics_data import require_episode_gate_provenance

        require_episode_gate_provenance(all_episode_paths, gate_hash)
    statistics = TrainingStatistics.fit(
        selection.base_train_paths if cfg.sequence.enabled else training_paths
    )
    dataset: EpisodeFrameDataset | EpisodeSequenceDataset
    sequence_sampler: StratifiedSequenceBatchSampler | None = None
    if cfg.sequence.enabled:
        dataset = EpisodeSequenceDataset(
            base_paths=selection.base_train_paths,
            recovery_paths=selection.recovery_train_paths,
            statistics=statistics,
            horizon=cfg.sequence.horizon,
        )
        sequence_sampler = StratifiedSequenceBatchSampler(
            base_indices_by_phase=dataset.indices_by_group_phase[0],
            recovery_indices_by_phase=dataset.indices_by_group_phase[1],
            batch_size=cfg.train.batch_size,
            recovery_fraction=cfg.sequence.recovery_loss_fraction,
            seed=(cfg.train.model_seeds[0] if model_seed is None else model_seed),
        )
        provenance.update(
            _sequence_provenance_fields(
                statistics,
                dataset,
                sequence_sampler,
            )
        )
    else:
        dataset = EpisodeFrameDataset(training_paths, statistics)
    selected_seed = cfg.train.model_seeds[0] if model_seed is None else model_seed
    dimensions = inspect_episode_dimensions(training_paths)
    expected_schema = "physics_v2" if cfg.backend == "franka_contact" else "kinematic_v1"
    expected_edge_dim = 18 if cfg.backend == "franka_contact" else 10
    expected_proprio_dim = 23 if cfg.backend == "franka_contact" else 7
    if (
        dimensions["action_dim"] != cfg.model.action_dim
        or dimensions["edge_feature_dim"] != expected_edge_dim
        or dimensions["proprioception_dim"] != expected_proprio_dim
    ):
        raise ValueError(
            f"dataset dimensions {dimensions} do not match backend {cfg.backend}"
        )
    action_mode = (
        "cartesian_7d" if cfg.backend == "franka_contact" else "legacy_cartesian_4d"
    )
    model_kwargs: dict[str, object] = {
        "max_nodes": cfg.max_objects + 3,
        "max_edges": (cfg.max_objects + 3) * (cfg.max_objects + 2),
        "node_feature_dim": dimensions["node_feature_dim"],
        "edge_feature_dim": dimensions["edge_feature_dim"],
        "graph_hidden_dim": cfg.model.hidden_dim,
        "embedding_dim": cfg.model.embedding_dim,
        "policy_hidden_dim": cfg.model.hidden_dim,
        "message_rounds": cfg.model.message_rounds,
        "proprio_dim": dimensions["proprioception_dim"],
        "action_dim": dimensions["action_dim"],
        "action_mode": action_mode,
        "action_horizon": cfg.sequence.horizon if cfg.sequence.enabled else 1,
    }
    torch.manual_seed(selected_seed)
    policy = build_action_policy(representation=representation, **model_kwargs)
    run_dir = Path(cfg.output_dir) / representation / f"seed_{selected_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    split_record = {
        "train": list(splits.train),
        "validation": list(splits.validation),
        "test": list(splits.test),
    }
    split_path = run_dir / "episode_splits.json"
    if resume:
        if not split_path.exists():
            raise FileNotFoundError(f"cannot resume because {split_path} does not exist")
        saved_split_record = json.loads(split_path.read_text(encoding="utf-8"))
        if saved_split_record != split_record:
            raise ValueError(
                "current episode manifest produces a different train/validation/test split; "
                "refusing to resume"
            )
    else:
        split_path.write_text(
            json.dumps(split_record, indent=2, sort_keys=True), encoding="utf-8"
        )
    checkpoint = run_dir / "checkpoint.pt"
    if resume and not checkpoint.exists():
        raise FileNotFoundError(f"cannot resume because {checkpoint} does not exist")
    representation_contract = {
        "experimental_variable": "encoder_only",
        "state_information": "identical",
        "temporal_head": "identical",
        "rollout_controller": "identical",
    }
    temporal_contract = {
        "contribution": "shared_infrastructure",
        "horizon": cfg.sequence.horizon if cfg.sequence.enabled else 1,
        "future_loss_decay": cfg.sequence.future_loss_decay,
        "temporal_decay": cfg.sequence.temporal_decay,
        "gripper_close_threshold": cfg.sequence.gripper_close_threshold,
        "gripper_open_threshold": cfg.sequence.gripper_open_threshold,
        "recovery_loss_fraction": cfg.sequence.recovery_loss_fraction,
        "rollout_device": cfg.sequence.rollout_device,
    }
    checkpoint_metadata = {
        "backend": cfg.backend,
        "feature_schema": expected_schema,
        "action_mode": action_mode,
        "action_dim": dimensions["action_dim"],
        "proprioception_dim": dimensions["proprioception_dim"],
        "expert_gate_hash": gate_hash,
        "representation_contract": representation_contract,
        "temporal_contract": temporal_contract,
        "data_provenance": dict(provenance),
        "training_pipeline_source_hash": training_pipeline_source_hash(),
        "learned_rollout_source_hash": learned_rollout_source_hash(),
        **physical_hashes,
    }
    if cfg.sequence.enabled:
        assert isinstance(dataset, EpisodeSequenceDataset)
        assert sequence_sampler is not None
        result = train_sequence_policy(
            policy,
            dataset,
            sequence_sampler,
            statistics,
            epochs=cfg.train.epochs,
            learning_rate=cfg.train.learning_rate,
            future_loss_decay=cfg.sequence.future_loss_decay,
            recovery_loss_fraction=cfg.sequence.recovery_loss_fraction,
            seed=selected_seed,
            device=device,
            checkpoint_path=checkpoint,
            resume_from=checkpoint if resume else None,
            representation=representation,
            model_kwargs=model_kwargs,
            metrics_path=run_dir / "metrics.jsonl",
            training_provenance=provenance,
            checkpoint_metadata=checkpoint_metadata,
            show_progress=True,
        )
    else:
        assert isinstance(dataset, EpisodeFrameDataset)
        result = train_policy(
            policy,
            dataset,
            statistics,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            learning_rate=cfg.train.learning_rate,
            seed=selected_seed,
            device=device,
            checkpoint_path=checkpoint,
            resume_from=checkpoint if resume else None,
            representation=representation,
            model_kwargs=model_kwargs,
            metrics_path=run_dir / "metrics.jsonl",
            training_provenance=provenance,
            checkpoint_metadata=checkpoint_metadata,
            show_progress=True,
        )
    summary = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "final_normalized_mse": result.final_normalized_mse,
        "global_step": result.global_step,
        "training_episodes": len(training_paths),
        "base_training_episodes": len(selection.base_train_paths),
        "recovery_training_episodes": len(selection.recovery_train_paths),
        "recovery_episodes": len(selection.recovery_train_paths),
        "final_base_mse": result.final_base_mse,
        "final_recovery_mse": result.final_recovery_mse,
        "effective_loss_mass": {
            "base": 1.0 - cfg.sequence.recovery_loss_fraction,
            "recovery": cfg.sequence.recovery_loss_fraction,
        },
        "representation_contract": representation_contract,
        "temporal_contract": temporal_contract,
        "training_provenance": provenance,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if cfg.sequence.enabled:
        (run_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an interaction-representation policy")
    parser.add_argument("--config", default="configs/pilot_macos.yaml")
    parser.add_argument(
        "--representation", required=True, choices=("flat", "graph", "proprio")
    )
    parser.add_argument("--model-seed", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the run checkpoint and train for the configured number of additional epochs",
    )
    args = parser.parse_args()
    print(train_from_config(args.config, args.representation, args.model_seed, args.resume))


if __name__ == "__main__":
    main()
