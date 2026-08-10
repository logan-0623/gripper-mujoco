# ACT Rollout GIF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `--gif` output to the existing ACT checkpoint rollout and generate a truthful dual-view animation from a real Mac MPS run.

**Architecture:** Put GIF composition and atomic encoding in a focused `rollout_gif.py` module so visualization cannot affect policy inference or MuJoCo state. The existing rollout loop passes the exact agent/wrist observations and post-step diagnostics to an optional recorder; the CLI only constructs it when `--gif` is supplied.

**Tech Stack:** Python 3.12, NumPy, Pillow, MuJoCo, PyTorch/LeRobot, argparse, pytest.

---

## File map

- Create `interaction_vla/lerobot_bridge/rollout_gif.py`: validate, compose, sample, and atomically encode rollout frames.
- Create `tests/interaction_vla/lerobot_bridge/test_rollout_gif.py`: pure recorder and GIF integrity tests.
- Modify `interaction_vla/lerobot_bridge/rollout.py`: feed policy observations and truthful terminal diagnostics to the optional recorder.
- Modify `interaction_vla/lerobot_bridge/cli.py`: expose and forward `rollout --gif PATH`.
- Modify `tests/interaction_vla/lerobot_bridge/test_cli.py`: verify argument parsing and forwarding.
- Modify `README.md`: document the exact command and smoke-checkpoint limitation.

### Task 1: Focused atomic GIF recorder

**Files:**

- Create: `interaction_vla/lerobot_bridge/rollout_gif.py`
- Create: `tests/interaction_vla/lerobot_bridge/test_rollout_gif.py`

- [ ] **Step 1: Write failing composition and encoding tests**

Create `tests/interaction_vla/lerobot_bridge/test_rollout_gif.py` with:

```python
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
```

- [ ] **Step 2: Run the new test and verify it fails for the missing module**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gripper-mujoco-pycache-current \
  .venv/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_rollout_gif.py -q
```

Expected: collection fails with `ModuleNotFoundError: interaction_vla.lerobot_bridge.rollout_gif`.

- [ ] **Step 3: Implement the minimal recorder**

Create `interaction_vla/lerobot_bridge/rollout_gif.py` with this public contract:

```python
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
        f"step={step:03d}  gripper={'open' if gripper_open else 'closed'}  status={terminal_reason}",
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
        images = [frame.convert("P", palette=Image.Palette.ADAPTIVE) for _, frame in self._frames]
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
        finally:
            if temporary.exists():
                temporary.unlink()
        return len(images)
```

- [ ] **Step 4: Run recorder tests and verify they pass**

Run the Step 2 command again.

Expected: `4 passed`.

- [ ] **Step 5: Commit the recorder**

```bash
git add interaction_vla/lerobot_bridge/rollout_gif.py \
  tests/interaction_vla/lerobot_bridge/test_rollout_gif.py
git commit -m "feat: add atomic ACT rollout GIF recorder"
```

### Task 2: Integrate GIF recording into rollout and CLI

**Files:**

- Modify: `interaction_vla/lerobot_bridge/rollout.py`
- Modify: `interaction_vla/lerobot_bridge/cli.py`
- Modify: `tests/interaction_vla/lerobot_bridge/test_cli.py`

- [ ] **Step 1: Write failing CLI forwarding test**

Append to `tests/interaction_vla/lerobot_bridge/test_cli.py`:

```python
def test_rollout_cli_forwards_gif_destination(monkeypatch, capsys) -> None:
    received = {}

    def fake_rollout(*args, **kwargs):
        received.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(
        "interaction_vla.lerobot_bridge.cli.rollout_from_config", fake_rollout
    )

    main(
        [
            "rollout",
            "--config",
            "configs/lerobot_act_smoke_macos.yaml",
            "--checkpoint",
            "outputs/lerobot/act_smoke/checkpoint",
            "--gif",
            "outputs/lerobot/act_smoke/rollout.gif",
        ]
    )

    assert received["gif_path"] == Path("outputs/lerobot/act_smoke/rollout.gif")
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_rollout_cli_defaults_to_no_gif() -> None:
    args = build_parser().parse_args(
        [
            "rollout",
            "--config",
            "configs/lerobot_act_smoke_macos.yaml",
            "--checkpoint",
            "outputs/lerobot/act_smoke/checkpoint",
        ]
    )

    assert args.gif_path is None
```

- [ ] **Step 2: Run the CLI test and verify the unrecognized argument failure**

Run:

```bash
HF_HOME=/private/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/private/tmp/gripper-mujoco-pycache-lerobot \
  .venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_cli.py::test_rollout_cli_forwards_gif_destination -q
```

Expected: FAIL because `--gif` is unrecognized or `gif_path` is absent.

- [ ] **Step 3: Add the CLI argument and preserve the no-GIF default**

In `build_parser`, add:

```python
rollout.add_argument(
    "--gif",
    dest="gif_path",
    type=Path,
    help="write a 10 FPS side-by-side ACT rollout GIF",
)
```

Forward it in `_dispatch`:

```python
return rollout_from_config(
    args.config,
    args.checkpoint,
    seed=args.seed,
    object_count=args.object_count,
    gif_path=args.gif_path,
)
```

- [ ] **Step 4: Add rollout integration using the exact captured observations**

Import `RolloutGIFRecorder` in `rollout.py`. Add `gif_path: str | Path | None = None`
to both public rollout functions and forward it from `rollout_from_config` to
`rollout_checkpoint`.

After `DualViewCapture` is created, initialize:

```python
gif_recorder = (
    RolloutGIFRecorder(
        gif_path,
        source_fps=config.dataset.fps,
        playback_fps=10,
    )
    if gif_path is not None
    else None
)
```

Immediately after appending each post-step diagnostic and before replacing the
snapshot, record the exact pre-action policy images with the resulting status:

```python
if gif_recorder is not None:
    gif_recorder.add(
        camera_frame.views["agent"].rgb,
        camera_frame.views["wrist"].rgb,
        step=step,
        gripper_open=gripper.is_open,
        terminal_reason=final_reason,
    )
```

After the loop and capture cleanup, write only when requested:

```python
gif_frame_count = gif_recorder.write() if gif_recorder is not None else None
```

Add fields without changing the legacy result shape:

```python
if gif_recorder is not None:
    result["gif"] = gif_recorder.destination
    result["gif_frames"] = gif_frame_count
```

- [ ] **Step 5: Run CLI, rollout, and recorder tests**

Run:

```bash
HF_HOME=/private/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/private/tmp/gripper-mujoco-pycache-lerobot \
  .venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge/test_cli.py \
  tests/interaction_vla/lerobot_bridge/test_rollout_gif.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit rollout integration**

```bash
git add interaction_vla/lerobot_bridge/cli.py \
  interaction_vla/lerobot_bridge/rollout.py \
  tests/interaction_vla/lerobot_bridge/test_cli.py
git commit -m "feat: export ACT rollout GIFs"
```

### Task 3: Document, verify, and generate the real artifact

**Files:**

- Modify: `README.md`
- Generate locally: `outputs/lerobot/act_smoke/rollout.gif`
- Update locally: `outputs/lerobot/act_smoke/rollout.json`

- [ ] **Step 1: Document the exact rollout command and evidence boundary**

Add this example below the LeRobot smoke commands in `README.md`:

````markdown
生成 ACT 闭环双视角 GIF：

```bash
.venv-lerobot/bin/python -m interaction_vla.lerobot_bridge rollout \
  --config configs/lerobot_act_smoke_macos.yaml \
  --checkpoint outputs/lerobot/act_smoke/checkpoint \
  --object-count 2 \
  --gif outputs/lerobot/act_smoke/rollout.gif
```

GIF 展示实际送入 ACT 的 agent/wrist RGB 和执行状态。500-step checkpoint
只通过工程 smoke；GIF 中的 timeout 或失败状态不是任务性能成功证据。
````

- [ ] **Step 2: Run both regression suites**

Run the original environment:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gripper-mujoco-pycache-current \
  .venv/bin/python -m pytest -q
```

Expected: all runnable tests pass, with only declared skips.

Run the LeRobot bridge environment:

```bash
HF_HOME=/private/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/private/tmp/gripper-mujoco-pycache-lerobot \
  .venv-lerobot/bin/python -m pytest \
  tests/interaction_vla/lerobot_bridge -q
```

Expected: all bridge tests pass, with only the declared sandbox graphics skip.

- [ ] **Step 3: Commit the documentation**

```bash
git add README.md
git commit -m "docs: document ACT rollout GIF command"
```

- [ ] **Step 4: Move provenance-stale artifacts to recoverable backups**

Any bridge source change intentionally changes `source_fingerprint`. The current
dataset provenance and checkpoint binding will therefore reject the new runtime.
After all source commits are final, move—not delete—the old artifacts:

```bash
mv outputs/lerobot/franka_lerobot_act_smoke \
  /private/tmp/gripper-mujoco-dataset-pre-rollout-gif
mv outputs/lerobot/act_smoke \
  /private/tmp/gripper-mujoco-act-pre-rollout-gif
```

Expected: both output destinations are absent and both backup directories exist.

- [ ] **Step 5: Regenerate the source-bound dataset and ACT checkpoint**

Run collection outside the graphics sandbox:

```bash
HF_HOME=/private/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/private/tmp/gripper-mujoco-pycache-lerobot \
  .venv-lerobot/bin/python -m interaction_vla.lerobot_bridge collect \
  --config configs/lerobot_act_smoke_macos.yaml
```

Expected: `passed: true`, 5 episodes, 581 deterministic frames, and replay error
no greater than `1e-5`.

Run the bounded ACT gate outside the sandbox:

```bash
HF_HOME=/private/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/private/tmp/gripper-mujoco-pycache-lerobot \
  .venv-lerobot/bin/python -m interaction_vla.lerobot_bridge smoke \
  --config configs/lerobot_act_smoke_macos.yaml
```

Expected: `passed: true`, `device: mps`, `training_steps: 500`, finite rollout,
and zero or finite checkpoint reload error no greater than `1e-5`.

- [ ] **Step 6: Run the real Mac MPS rollout and write the GIF**

Run outside the graphics sandbox:

```bash
HF_HOME=/private/tmp/gripper-mujoco-hf-cache \
PYTHONPYCACHEPREFIX=/private/tmp/gripper-mujoco-pycache-lerobot \
  .venv-lerobot/bin/python -m interaction_vla.lerobot_bridge rollout \
  --config configs/lerobot_act_smoke_macos.yaml \
  --checkpoint outputs/lerobot/act_smoke/checkpoint \
  --object-count 2 \
  --gif outputs/lerobot/act_smoke/rollout.gif
```

Expected JSON fields:

```json
{
  "device": "mps",
  "finite_rollout": true,
  "gif": "outputs/lerobot/act_smoke/rollout.gif",
  "gif_frames": 91,
  "passed": true,
  "steps": 180,
  "task_success": false,
  "terminal_reason": "timeout"
}
```

The exact task outcome must be reported as observed; only the finite rollout and
artifact integrity are required by this feature.

- [ ] **Step 7: Verify the generated GIF and clean worktree**

Run:

```bash
.venv-lerobot/bin/python -c \
  'from PIL import Image; p="outputs/lerobot/act_smoke/rollout.gif"; im=Image.open(p); print({"format": im.format, "size": im.size, "frames": im.n_frames, "duration_ms": im.info.get("duration")})'
git status --short
```

Expected GIF audit:

```text
{'format': 'GIF', 'size': (512, 296), 'frames': 91, 'duration_ms': 100}
```

Expected Git status: clean. Generated `outputs/lerobot/` artifacts remain ignored.
