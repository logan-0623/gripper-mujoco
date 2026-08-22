from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from interaction_vla.representation_study.rl.replay import RecoveryReplay
from interaction_vla.representation_study.rl.snapshots import SnapshotStore


def example_transition(index: int) -> dict[str, object]:
    image = np.full((4, 5, 3), index % 255, dtype=np.uint8)
    return {
        "transition_id": f"transition-{index:04d}",
        "case_id": f"case-{index % 3}",
        "family": ("recovery", "perturbation", "nominal")[index % 3],
        "task": "Pick and place the target.",
        "agent_rgb": image,
        "wrist_rgb": image + np.uint8(1),
        "state": np.full(10, index, dtype=np.float32),
        "next_agent_rgb": image + np.uint8(2),
        "next_wrist_rgb": image + np.uint8(3),
        "next_state": np.full(10, index + 1, dtype=np.float32),
        "oracle_state": np.full(36, index, dtype=np.float32),
        "next_oracle_state": np.full(36, index + 1, dtype=np.float32),
        "actor_observation": np.full(16, index, dtype=np.float32),
        "next_actor_observation": np.full(16, index + 1, dtype=np.float32),
        "residual": np.full(7, index / 100.0, dtype=np.float32),
        "reward": float(index),
        "done": bool(index % 5 == 0),
    }


def test_replay_round_trip_preserves_sample_sequence(tmp_path: Path) -> None:
    replay = RecoveryReplay(
        root=tmp_path / "replay", capacity=32, seed=5, shard_size=8
    )
    for index in range(20):
        replay.add(example_transition(index))
    state = replay.state_dict()
    expected = replay.sample(8).transition_ids
    replay.load_state_dict(state)
    assert replay.sample(8).transition_ids == expected


def test_replay_uses_uint8_images_and_float32_numeric_arrays(tmp_path: Path) -> None:
    replay = RecoveryReplay(
        root=tmp_path / "replay", capacity=32, seed=5, shard_size=4
    )
    for index in range(4):
        replay.add(example_transition(index))
    batch = replay.sample(3)
    assert batch.agent_rgb.dtype == np.uint8
    assert batch.wrist_rgb.dtype == np.uint8
    assert batch.state.dtype == np.float32
    assert batch.oracle_state.dtype == np.float32
    assert batch.actor_observation.shape == (3, 16)
    assert batch.next_actor_observation.shape == (3, 16)


def test_replay_resume_rejects_modified_shard(tmp_path: Path) -> None:
    replay = RecoveryReplay(
        root=tmp_path / "replay", capacity=32, seed=5, shard_size=4
    )
    for index in range(4):
        replay.add(example_transition(index))
    state = replay.state_dict()
    shard = next((tmp_path / "replay" / "shards").glob("*.npz"))
    shard.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="shard hash"):
        replay.load_state_dict(state)


def test_replay_resume_skips_uncommitted_orphan_shards(tmp_path: Path) -> None:
    root = tmp_path / "replay"
    replay = RecoveryReplay(root=root, capacity=32, seed=5, shard_size=4)
    for index in range(4):
        replay.add(example_transition(index))
    committed = replay.state_dict()
    for index in range(4, 8):
        replay.add(example_transition(index))
    assert (root / "shards" / "shard_00000001.npz").is_file()

    resumed = RecoveryReplay(root=root, capacity=32, seed=5, shard_size=4)
    resumed.load_state_dict(committed)
    for index in range(4, 8):
        resumed.add(example_transition(index))
    resumed.state_dict()
    assert (root / "shards" / "shard_00000002.npz").is_file()


def example_payload() -> dict[str, object]:
    return {
        "actor": {"weight": torch.ones(2)},
        "environment_steps": 4096,
        "rng": {"seed": 7},
    }


def test_completed_snapshot_is_immutable(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.save(step=4096, payload=example_payload(), binding="abc")
    with pytest.raises(FileExistsError, match="immutable"):
        store.save(step=4096, payload=example_payload(), binding="abc")


def test_snapshot_load_validates_binding(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    store.save(step=4096, payload=example_payload(), binding="abc")
    with pytest.raises(ValueError, match="binding"):
        store.load(step=4096, expected_binding="different")
    loaded = store.load(step=4096, expected_binding="abc")
    assert loaded["environment_steps"] == 4096


def test_snapshot_inspection_rejects_corrupted_payload_without_loading(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    snapshot = store.save(step=4096, payload=example_payload(), binding="abc")
    (snapshot / "training_state.pt").write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="payload SHA-256"):
        store.inspect(step=4096, expected_binding="abc")
