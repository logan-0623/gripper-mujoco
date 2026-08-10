import numpy as np
import pytest

pytest.importorskip("lerobot")

from interaction_vla.lerobot_bridge.validator import (
    validate_dataset_contract,
    validate_dataset_root,
    validate_replay_states,
    validate_teacher_schema,
    validate_teacher_manifest,
)
from interaction_vla.lerobot_bridge.teacher_schema import teacher_schema_payload


def test_validator_rejects_incomplete_dataset(tmp_path) -> None:
    (tmp_path / "INCOMPLETE").write_text(
        "collection in progress\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="INCOMPLETE"):
        validate_dataset_root(tmp_path, allow_incomplete=False)


def test_validator_rejects_manifest_frame_mismatch() -> None:
    records = [
        {
            "episode_index": 0,
            "frames": 3,
            "path": "teacher/episode_000000.npz",
        }
    ]
    with pytest.raises(ValueError, match="frame count"):
        validate_teacher_manifest(records, dataset_episode_lengths={0: 2})


def test_validator_rejects_teacher_schema_semantic_drift() -> None:
    schema = teacher_schema_payload()
    schema["predicate_ids"]["clearance"] = 99

    with pytest.raises(ValueError, match="teacher schema"):
        validate_teacher_schema(schema)


def test_standard_samples_have_only_the_policy_contract(
    tiny_lerobot_dataset,
) -> None:
    root, repo_id = tiny_lerobot_dataset
    report = validate_dataset_root(
        root,
        repo_id=repo_id,
        allow_incomplete=True,
        require_bridge_metadata=False,
        replay=False,
    )

    assert report["frames"] == 9
    assert report["image_shape"] == [3, 256, 256]
    assert report["state_shape"] == [10]
    assert report["action_shape"] == [7]
    assert report["forbidden_policy_keys"] == []


def test_replay_error_threshold_is_enforced() -> None:
    recorded = np.zeros((2, 10), dtype=np.float32)
    replayed = recorded.copy()
    replayed[1, 3] = 2e-5

    with pytest.raises(ValueError, match="replay"):
        validate_replay_states(recorded, replayed, tolerance=1e-5)


def test_dataset_contract_rejects_task_metadata_drift() -> None:
    from interaction_vla.lerobot_bridge.config import load_bridge_config

    config = load_bridge_config("configs/lerobot_act_smoke_macos.yaml")
    records = [
        {
            "episode_index": index,
            "task": config.dataset.task,
            "task_id": 0,
            "object_count": config.dataset.object_counts[
                index % len(config.dataset.object_counts)
            ],
        }
        for index in range(config.dataset.episodes)
    ]
    provenance = {
        "repo_id": config.dataset.repo_id,
        "task": config.dataset.task,
        "requested_episodes": config.dataset.episodes,
        "accepted_episodes": config.dataset.episodes,
        "fps": config.dataset.fps,
        "image_size": list(config.dataset.image_size),
    }

    with pytest.raises(ValueError, match="task"):
        validate_dataset_contract(
            resolved_repo_id=config.dataset.repo_id,
            tasks=["a different task"],
            episode_lengths={index: 1 for index in range(config.dataset.episodes)},
            records=records,
            provenance=provenance,
            bridge_config=config,
        )
