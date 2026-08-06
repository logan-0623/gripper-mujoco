from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from interaction_vla.sequence_training import (
    EpisodeSequenceDataset,
    StratifiedSequenceBatchSampler,
    sequence_behavior_cloning_loss,
)
from interaction_vla.train import TrainingStatistics


def identity_statistics() -> TrainingStatistics:
    return TrainingStatistics(
        node_mean=np.zeros(2, dtype=np.float32),
        node_std=np.ones(2, dtype=np.float32),
        edge_mean=np.zeros(2, dtype=np.float32),
        edge_std=np.ones(2, dtype=np.float32),
        proprio_mean=np.zeros(3, dtype=np.float32),
        proprio_std=np.ones(3, dtype=np.float32),
        action_mean=np.zeros(7, dtype=np.float32),
        action_std=np.ones(7, dtype=np.float32),
    )


def write_numbered_episode(
    path: Path,
    *,
    frames: int,
    source_seed: int = 11,
    recovery: bool = False,
) -> Path:
    actions = np.zeros((frames, 7), dtype=np.float32)
    actions[:, 0] = np.arange(frames, dtype=np.float32)
    node_features = np.zeros((frames, 2, 2), dtype=np.float32)
    node_features[:, :, 0] = np.arange(frames, dtype=np.float32)[:, None]
    phases = np.asarray(
        tuple(("approach", "transport", "retreat")[index % 3] for index in range(frames))
    )
    metadata = {
        "seed": source_seed,
        "object_count": 2,
        "target_name": "object_0",
        "reason": "success",
        "trajectory_kind": "recovery" if recovery else "base",
        "source_seed": source_seed if recovery else None,
        "variant_id": 0 if recovery else None,
        "perturbation_kind": "wrong_way_transport" if recovery else None,
        "injection_phase": "transport" if recovery else None,
    }
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        node_features=node_features,
        edge_index=np.asarray(((0, 1), (1, 0)), dtype=np.int64),
        edge_features=np.zeros((frames, 2, 2), dtype=np.float32),
        node_mask=np.ones((frames, 2), dtype=np.bool_),
        edge_mask=np.ones((frames, 2), dtype=np.bool_),
        proprioception=np.zeros((frames, 3), dtype=np.float32),
        actions=actions,
        phases=phases,
    )
    return path


def test_sequence_windows_never_cross_episode_boundaries(tmp_path: Path) -> None:
    episode = write_numbered_episode(tmp_path / "episode.npz", frames=5)
    dataset = EpisodeSequenceDataset(
        base_paths=(episode,),
        recovery_paths=(),
        statistics=identity_statistics(),
        horizon=3,
    )

    penultimate = dataset[3]
    torch.testing.assert_close(
        penultimate["horizon_mask"],
        torch.tensor((True, True, False)),
    )
    torch.testing.assert_close(
        penultimate["action_chunk"][:, 0],
        torch.tensor((3.0, 4.0, 0.0)),
    )
    final = dataset[4]
    torch.testing.assert_close(
        final["horizon_mask"],
        torch.tensor((True, False, False)),
    )
    assert final["node_features"][0, 0].item() == pytest.approx(4.0)
    assert final["sample_group"].item() == 0
    assert final["source_seed"].item() == 11


@pytest.mark.parametrize("batch_size,expected", [(64, (48, 16)), (8, (6, 2))])
def test_stratified_sampler_has_exact_recovery_mass(batch_size, expected) -> None:
    sampler = StratifiedSequenceBatchSampler(
        base_indices_by_phase={
            "approach": tuple(range(0, 40)),
            "grasp": tuple(range(40, 80)),
            "transport": tuple(range(80, 120)),
        },
        recovery_indices_by_phase={
            "lift": (120, 121),
            "transport": (122, 123),
            "retreat": (124, 125),
        },
        batch_size=batch_size,
        recovery_fraction=0.25,
        seed=7,
    )

    base_phase_by_index = {
        index: phase
        for phase, indices in sampler.base_indices_by_phase.items()
        for index in indices
    }
    recovery_phase_by_index = {
        index: phase
        for phase, indices in sampler.recovery_indices_by_phase.items()
        for index in indices
    }
    cumulative_base: Counter[str] = Counter()
    cumulative_recovery: Counter[str] = Counter()
    for batch in sampler:
        groups = [index >= 120 for index in batch]
        assert groups.count(False) == expected[0]
        assert groups.count(True) == expected[1]
        cumulative_base.update(base_phase_by_index[index] for index in batch if index < 120)
        cumulative_recovery.update(
            recovery_phase_by_index[index] for index in batch if index >= 120
        )
    assert max(cumulative_base.values()) - min(cumulative_base.values()) <= 1
    assert max(cumulative_recovery.values()) - min(cumulative_recovery.values()) <= 1


def test_stratified_sampler_is_deterministic_per_epoch() -> None:
    kwargs = {
        "base_indices_by_phase": {"a": tuple(range(12)), "b": tuple(range(12, 24))},
        "recovery_indices_by_phase": {"r": (24, 25)},
        "batch_size": 8,
        "recovery_fraction": 0.25,
        "seed": 17,
    }
    first = StratifiedSequenceBatchSampler(**kwargs)
    second = StratifiedSequenceBatchSampler(**kwargs)

    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)


def test_sequence_loss_has_exact_group_mass_and_respects_masks() -> None:
    prediction = torch.zeros((4, 3, 7), dtype=torch.float32)
    target = torch.ones_like(prediction)
    target[2:] = 2.0
    mask = torch.tensor(
        (
            (True, True, True),
            (True, False, False),
            (True, True, False),
            (True, False, False),
        )
    )
    sample_group = torch.tensor((0, 0, 1, 1))

    loss = sequence_behavior_cloning_loss(
        prediction,
        target,
        mask,
        sample_group,
        future_loss_decay=0.9,
        recovery_loss_fraction=0.25,
    )

    assert loss.base.item() == pytest.approx(1.0)
    assert loss.recovery.item() == pytest.approx(4.0)
    assert loss.total.item() == pytest.approx(0.75 * 1.0 + 0.25 * 4.0)


def test_sequence_loss_requires_both_training_groups() -> None:
    prediction = torch.zeros((2, 2, 7))
    with pytest.raises(ValueError, match="both base and recovery"):
        sequence_behavior_cloning_loss(
            prediction,
            prediction,
            torch.ones((2, 2), dtype=torch.bool),
            torch.zeros(2, dtype=torch.long),
            future_loss_decay=0.9,
            recovery_loss_fraction=0.25,
        )
