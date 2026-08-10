from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from interaction_vla.lerobot_bridge.rollout_gif import (
    HEADER_HEIGHT,
    RolloutGIFRecorder,
    compose_rollout_frame,
)


def _views() -> tuple[np.ndarray, np.ndarray]:
    agent = np.zeros((256, 256, 3), dtype=np.uint8)
    agent[..., 0] = 255
    wrist = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist[..., 1] = 255
    return agent, wrist


def test_compose_rollout_frame_preserves_both_policy_views() -> None:
    agent, wrist = _views()

    frame = compose_rollout_frame(
        agent,
        wrist,
        step=7,
        gripper_open=False,
        terminal_reason="running",
    )
    pixels = np.asarray(frame.convert("RGB"))

    assert frame.size == (512, 256 + HEADER_HEIGHT)
    np.testing.assert_array_equal(pixels[HEADER_HEIGHT:, :256], agent)
    np.testing.assert_array_equal(pixels[HEADER_HEIGHT:, 256:], wrist)
    assert np.any(pixels[:HEADER_HEIGHT] != pixels[0, 0])


def test_recorder_samples_at_10_fps_and_always_keeps_final_frame(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "rollout.gif"
    recorder = RolloutGIFRecorder(destination, source_fps=20, playback_fps=10)
    agent, wrist = _views()
    for step in range(4):
        recorder.add(
            agent,
            wrist,
            step=step,
            gripper_open=step < 2,
            terminal_reason="timeout" if step == 3 else "running",
        )

    frame_count = recorder.write()

    assert frame_count == 3
    assert destination.is_file()
    with Image.open(destination) as image:
        assert image.format == "GIF"
        assert image.n_frames == 3
        assert image.info["duration"] == 100


def test_recorder_rejects_non_gif_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".gif"):
        RolloutGIFRecorder(tmp_path / "rollout.mp4")


def test_encoding_failure_keeps_previous_gif_and_removes_temporary_file(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "rollout.gif"
    destination.write_bytes(b"previous")
    recorder = RolloutGIFRecorder(destination)
    agent, wrist = _views()
    recorder.add(
        agent,
        wrist,
        step=0,
        gripper_open=True,
        terminal_reason="running",
    )

    def fail_after_partial_write(image, target, *args, **kwargs) -> None:
        Path(target).write_bytes(b"partial")
        raise RuntimeError("encode failed")

    monkeypatch.setattr(Image.Image, "save", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="encode failed"):
        recorder.write()

    assert destination.read_bytes() == b"previous"
    assert not (tmp_path / ".rollout.gif.tmp").exists()
