import numpy as np
import pytest

from interaction_vla.lerobot_bridge.sidecar import (
    TeacherSidecarWriter,
    load_teacher_sidecar,
)
from interaction_vla.lerobot_bridge.teacher_schema import TeacherFrame


def test_sidecar_round_trip_has_one_row_per_frame_and_a_sha256(tmp_path) -> None:
    writer = TeacherSidecarWriter(tmp_path)
    frames = [
        TeacherFrame.zeros(
            frame_index=index,
            timestamp=index / 20.0,
            state_hash=f"state-{index}",
        )
        for index in range(2)
    ]

    record = writer.write_episode(0, frames, np.zeros((2, 5), dtype=np.float32))
    loaded = load_teacher_sidecar(
        tmp_path / record.path, expected_sha256=record.sha256
    )

    assert record.frames == 2
    assert loaded["annotation.tc_tig.entity_pose"].shape == (2, 6, 9)
    np.testing.assert_array_equal(loaded["frame_index"], (0, 1))


def test_hash_mismatch_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, frame_index=np.asarray((0,), dtype=np.int32))

    with pytest.raises(ValueError, match="SHA-256"):
        load_teacher_sidecar(path, expected_sha256="0" * 64)


def test_staged_sidecar_is_hidden_until_atomic_commit(tmp_path) -> None:
    writer = TeacherSidecarWriter(tmp_path)
    frames = [
        TeacherFrame.zeros(frame_index=0, timestamp=0.0, state_hash="state-0")
    ]

    record = writer.stage_episode(0, frames, np.zeros((1, 5), dtype=np.float32))

    assert (tmp_path / "teacher" / ".episode_000000.pending.npz").is_file()
    assert not (tmp_path / record.path).exists()
    writer.commit_staged(record)
    assert (tmp_path / record.path).is_file()
