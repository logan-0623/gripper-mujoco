import numpy as np
import pytest

pytest.importorskip("lerobot")

from interaction_vla.lerobot_bridge.validator import (
    validate_dataset_root,
    validate_replay_states,
    validate_teacher_manifest,
)


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
