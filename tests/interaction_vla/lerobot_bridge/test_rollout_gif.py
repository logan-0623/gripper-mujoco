from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from interaction_vla.lerobot_bridge import rollout_gif
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
        step_agent = agent.copy()
        step_agent[..., 2] = step * 20
        recorder.add(
            step_agent,
            wrist,
            step=step,
            gripper_open=True,
            terminal_reason="timeout" if step == 3 else "running",
        )

    frame_count = recorder.write()

    assert frame_count == 3
    assert destination.is_file()
    with Image.open(destination) as image:
        assert image.format == "GIF"
        assert image.n_frames == 3
        assert image.info["duration"] == 100
        image.seek(1)
        preceding = np.asarray(image.convert("RGB"))
        image.seek(2)
        final = np.asarray(image.convert("RGB"))
        assert final[HEADER_HEIGHT, 0, 2] > preceding[HEADER_HEIGHT, 0, 2]
        assert np.any(final[:HEADER_HEIGHT] != preceding[:HEADER_HEIGHT])


def test_recorder_rejects_non_gif_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".gif"):
        RolloutGIFRecorder(tmp_path / "rollout.mp4")


@pytest.mark.parametrize(
    ("agent", "wrist", "terminal_reason", "message"),
    (
        (
            np.zeros((8, 8, 3), dtype=np.float32),
            np.zeros((8, 8, 3), dtype=np.uint8),
            "running",
            "uint8",
        ),
        (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((7, 8, 3), dtype=np.uint8),
            "running",
            "matching shapes",
        ),
        (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((8, 8, 3), dtype=np.uint8),
            "",
            "terminal_reason",
        ),
    ),
)
def test_compose_rollout_frame_rejects_invalid_inputs(
    agent: np.ndarray,
    wrist: np.ndarray,
    terminal_reason: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compose_rollout_frame(
            agent,
            wrist,
            step=0,
            gripper_open=True,
            terminal_reason=terminal_reason,
        )


def test_recorder_rejects_empty_output(tmp_path: Path) -> None:
    recorder = RolloutGIFRecorder(tmp_path / "empty.gif")

    with pytest.raises(ValueError, match="empty"):
        recorder.write()


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
    temporary_paths: list[Path] = []

    def fail_after_partial_write(image, target, *args, **kwargs) -> None:
        temporary = Path(target)
        temporary_paths.append(temporary)
        temporary.write_bytes(b"partial")
        raise RuntimeError("encode failed")

    monkeypatch.setattr(Image.Image, "save", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="encode failed"):
        recorder.write()

    assert destination.read_bytes() == b"previous"
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()


def test_recorder_uses_a_unique_sibling_temporary_file(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "rollout.gif"
    recorder = RolloutGIFRecorder(destination)
    agent, wrist = _views()
    recorder.add(
        agent,
        wrist,
        step=0,
        gripper_open=True,
        terminal_reason="running",
    )
    targets: list[Path] = []
    original_save = Image.Image.save

    def remember_target(image, target, *args, **kwargs) -> None:
        targets.append(Path(target))
        original_save(image, target, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", remember_target)

    recorder.write()

    assert len(targets) == 1
    assert targets[0].parent == tmp_path
    assert targets[0].name.startswith(".rollout.gif.")
    assert targets[0] != tmp_path / ".rollout.gif.tmp"


def test_unsupported_directory_fsync_does_not_fail_after_publish(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "rollout.gif"
    recorder = RolloutGIFRecorder(destination)
    agent, wrist = _views()
    recorder.add(
        agent,
        wrist,
        step=0,
        gripper_open=True,
        terminal_reason="running",
    )
    original_open = rollout_gif.os.open

    def reject_directory(path, flags, *args, **kwargs):
        if Path(path) == tmp_path:
            raise OSError("directory fsync unsupported")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(rollout_gif.os, "open", reject_directory)

    assert recorder.write() == 1
    assert destination.is_file()
