from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


HEADER_HEIGHT = 40


def _rgb(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"{name} must be an HxWx3 uint8 array")
    return array


def compose_rollout_frame(
    agent_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    *,
    step: int,
    gripper_open: bool,
    terminal_reason: str,
) -> Image.Image:
    agent = _rgb(agent_rgb, "agent_rgb")
    wrist = _rgb(wrist_rgb, "wrist_rgb")
    if agent.shape != wrist.shape:
        raise ValueError("agent and wrist RGB views must have matching shapes")
    if step < 0 or not terminal_reason:
        raise ValueError("step and terminal_reason must be valid")

    height, width = agent.shape[:2]
    canvas = Image.new("RGB", (width * 2, height + HEADER_HEIGHT), "#111827")
    canvas.paste(Image.fromarray(agent), (0, HEADER_HEIGHT))
    canvas.paste(Image.fromarray(wrist), (width, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), "ACT smoke rollout", fill="white")
    draw.text(
        (8, 21),
        (
            f"step={step:03d}  "
            f"gripper={'open' if gripper_open else 'closed'}  "
            f"status={terminal_reason}"
        ),
        fill="#d1d5db",
    )
    draw.text((width - 66, 5), "agent RGB", fill="#93c5fd")
    draw.text((width * 2 - 70, 5), "wrist RGB", fill="#86efac")
    return canvas


class RolloutGIFRecorder:
    def __init__(
        self,
        destination: str | Path,
        *,
        source_fps: int = 20,
        playback_fps: int = 10,
    ) -> None:
        self.destination = Path(destination)
        if self.destination.suffix.lower() != ".gif":
            raise ValueError("rollout GIF destination must end in .gif")
        if source_fps < 1 or playback_fps < 1 or source_fps % playback_fps:
            raise ValueError("source_fps must be divisible by playback_fps")
        self.playback_fps = int(playback_fps)
        self.sample_interval = source_fps // playback_fps
        self._frames: list[tuple[int, Image.Image]] = []
        self._latest: tuple[int, Image.Image] | None = None

    def add(
        self,
        agent_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        *,
        step: int,
        gripper_open: bool,
        terminal_reason: str,
    ) -> None:
        frame = compose_rollout_frame(
            agent_rgb,
            wrist_rgb,
            step=step,
            gripper_open=gripper_open,
            terminal_reason=terminal_reason,
        )
        self._latest = (step, frame)
        if step % self.sample_interval == 0:
            self._frames.append((step, frame))

    def write(self) -> int:
        if self._latest is None:
            raise ValueError("cannot write an empty rollout GIF")
        if not self._frames or self._frames[-1][0] != self._latest[0]:
            self._frames.append(self._latest)
        images = [
            frame.convert("P", palette=Image.Palette.ADAPTIVE)
            for _, frame in self._frames
        ]
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.destination.with_name(f".{self.destination.name}.tmp")
        try:
            images[0].save(
                temporary,
                format="GIF",
                save_all=True,
                append_images=images[1:],
                duration=round(1000 / self.playback_fps),
                loop=0,
                disposal=2,
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, self.destination)
            directory_fd = os.open(self.destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
        return len(images)
