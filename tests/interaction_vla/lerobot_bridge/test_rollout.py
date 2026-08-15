import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from interaction_vla.lerobot_bridge import rollout as rollout_module
from interaction_vla.lerobot_bridge.rollout import (
    ActionChunkQueue,
    BinaryGripperHysteresis,
    LoadedACTRuntime,
    policy_observation,
)


def test_policy_observation_is_rgb_only_chw_float_plus_state() -> None:
    observation = policy_observation(
        agent_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        wrist_rgb=np.full((256, 256, 3), 255, dtype=np.uint8),
        state=np.zeros(10, dtype=np.float32),
    )

    assert set(observation) == {
        "observation.images.agent",
        "observation.images.wrist",
        "observation.state",
    }
    assert observation["observation.images.agent"].shape == (3, 256, 256)
    assert observation["observation.images.agent"].dtype == torch.float32
    assert observation["observation.images.wrist"].max().item() == 1.0


def test_gripper_hysteresis_suppresses_midrange_chatter() -> None:
    gate = BinaryGripperHysteresis(
        close_threshold=0.4,
        open_threshold=0.6,
        initially_open=True,
    )
    assert [
        gate.resolve(value) for value in (0.55, 0.45, 0.39, 0.50, 0.61)
    ] == [1.0, 1.0, 0.0, 0.0, 1.0]


def test_loaded_act_runtime_resets_policy_state() -> None:
    policy = SimpleNamespace(reset_calls=0)
    policy.reset = lambda: setattr(policy, "reset_calls", policy.reset_calls + 1)
    runtime = LoadedACTRuntime(Path("checkpoint"), policy, object(), object())

    runtime.reset()

    assert policy.reset_calls == 1


def test_chunk_queue_requeries_after_configured_action_horizon() -> None:
    calls = 0

    def predict() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.full((8, 7), calls, dtype=np.float32)

    queue = ActionChunkQueue(chunk_size=8, n_action_steps=1)
    selected = [queue.next(predict) for _ in range(3)]

    assert calls == 3
    assert [item.queue_index for item in selected] == [0, 0, 0]
    assert [float(item.action[0]) for item in selected] == [1.0, 2.0, 3.0]


def test_chunk_queue_can_retain_legacy_eight_action_execution() -> None:
    calls = 0

    def predict() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.zeros((8, 7), dtype=np.float32)

    queue = ActionChunkQueue(chunk_size=8, n_action_steps=8)
    for _ in range(9):
        queue.next(predict)

    assert calls == 2


def test_chunk_queue_rejects_nonfinite_or_wrong_shape() -> None:
    queue = ActionChunkQueue(chunk_size=8, n_action_steps=8)
    with pytest.raises(ValueError, match="shape"):
        queue.next(lambda: np.zeros((7, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        queue.next(lambda: np.full((8, 7), np.nan, dtype=np.float32))


def test_rollout_from_config_forwards_optional_gif_path(
    tmp_path: Path, monkeypatch
) -> None:
    gif_path = tmp_path / "rollout.gif"
    received = {}
    monkeypatch.setattr(
        rollout_module,
        "load_bridge_config",
        lambda path: SimpleNamespace(seed=42),
    )

    def fake_rollout_checkpoint(*args, **kwargs):
        received.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(
        rollout_module, "rollout_checkpoint", fake_rollout_checkpoint
    )

    result = rollout_module.rollout_from_config(
        "config.yaml",
        "checkpoint",
        seed=7,
        gif_path=gif_path,
    )

    assert result == {"passed": True}
    assert received["gif_path"] == gif_path


def test_record_rollout_gif_frame_uses_exact_views_and_status() -> None:
    agent = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist = np.ones((256, 256, 3), dtype=np.uint8)
    camera_frame = SimpleNamespace(
        views={
            "agent": SimpleNamespace(rgb=agent),
            "wrist": SimpleNamespace(rgb=wrist),
        }
    )

    class Recorder:
        def add(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    recorder = Recorder()

    rollout_module.record_rollout_gif_frame(
        recorder,
        camera_frame,
        step=17,
        gripper_open=False,
        terminal_reason="timeout",
    )

    assert recorder.args == (agent, wrist)
    assert recorder.kwargs == {
        "step": 17,
        "gripper_open": False,
        "terminal_reason": "timeout",
    }


def test_rollout_checkpoint_records_gif_lifecycle_and_json(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[object] = []
    agent = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist = np.full((256, 256, 3), 7, dtype=np.uint8)
    snapshot = SimpleNamespace(
        gripper=SimpleNamespace(orientation=np.asarray((1.0, 0.0, 0.0, 0.0)))
    )
    transition = SimpleNamespace(
        snapshot=snapshot,
        reason=rollout_module.TerminationReason.TIMEOUT,
        done=True,
    )

    class Environment:
        model = object()
        controller = object()

        def reset(self, **kwargs):
            return snapshot

        def proprioception(self):
            return object()

        def step(self, action):
            events.append("step")
            return transition

    class Capture:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def capture(self, env, *, include_teacher: bool):
            events.append("capture")
            assert include_teacher is False
            return SimpleNamespace(
                state_hash="a" * 64,
                views={
                    "agent": SimpleNamespace(rgb=agent),
                    "wrist": SimpleNamespace(rgb=wrist),
                },
            )

        def close(self) -> None:
            events.append("close")

    class Recorder:
        def __init__(self, destination, **kwargs) -> None:
            self.destination = Path(destination)
            events.append(("init", self.destination, kwargs))

        def add(self, *args, **kwargs) -> None:
            events.append(("add", args, kwargs))

        def write(self) -> int:
            events.append("write")
            return 1

    config = SimpleNamespace(
        dataset=SimpleNamespace(
            root=tmp_path / "dataset",
            repo_id="local/test",
            fps=20,
            image_size=(256, 256),
        ),
        act=SimpleNamespace(
            device="cpu",
            output_dir=tmp_path / "output",
            chunk_size=8,
            n_action_steps=1,
        ),
        source=SimpleNamespace(
            max_objects=4,
            environment=SimpleNamespace(max_steps=180),
        ),
    )
    projection = SimpleNamespace(
        action=np.zeros(7, dtype=np.float32),
        scale=1.0,
        projected_diagnostics=SimpleNamespace(
            position_error=0.0,
            orientation_error=0.0,
        ),
    )
    monkeypatch.setattr(rollout_module, "load_bridge_config", lambda path: config)
    monkeypatch.setattr(rollout_module, "validate_dataset_root", lambda *a, **k: None)
    monkeypatch.setattr(
        rollout_module, "resolve_device", lambda requested: torch.device("cpu")
    )
    monkeypatch.setattr(
        rollout_module,
        "_load_checkpoint_bundle",
        lambda **kwargs: (object(), object(), object(), {"dataset_fingerprint": "d" * 64}),
    )
    monkeypatch.setattr(
        rollout_module,
        "_make_env",
        lambda config, max_steps=None: Environment(),
    )
    monkeypatch.setattr(
        rollout_module, "validate_finger_joint_ranges", lambda model: None
    )
    monkeypatch.setattr(rollout_module, "DualViewCapture", Capture)
    monkeypatch.setattr(rollout_module, "RolloutGIFRecorder", Recorder)
    monkeypatch.setattr(
        rollout_module,
        "EndEffectorStateCodec",
        SimpleNamespace(
            encode_snapshot=lambda snapshot, proprioception: np.zeros(
                10, dtype=np.float32
            ),
            quaternion_to_matrix=lambda quaternion: np.eye(3, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        rollout_module,
        "policy_observation",
        lambda **kwargs: {"observation.state": torch.zeros(10)},
    )
    monkeypatch.setattr(
        rollout_module,
        "_predict_chunk",
        lambda **kwargs: np.ones((8, 7), dtype=np.float32),
    )
    monkeypatch.setattr(
        rollout_module,
        "LocalCartesianActionCodec",
        SimpleNamespace(
            decode=lambda local_action, rotation: np.zeros(7, dtype=np.float32)
        ),
    )
    monkeypatch.setattr(
        rollout_module, "project_cartesian_action", lambda *a, **k: projection
    )
    gif_path = tmp_path / "output" / "rollout.gif"

    result = rollout_module.rollout_checkpoint(
        "config.yaml",
        tmp_path / "checkpoint",
        seed=7,
        gif_path=gif_path,
    )

    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "init",
        "capture",
        "step",
        "add",
        "close",
        "write",
    ]
    add_event = events[3]
    assert add_event[1] == (agent, wrist)
    assert add_event[2]["terminal_reason"] == "timeout"
    assert result["gif"] == gif_path
    assert result["gif_frames"] == 1
    saved = json.loads((config.act.output_dir / "rollout.json").read_text())
    assert saved["gif"] == gif_path.as_posix()
    assert saved["gif_frames"] == 1

    events.clear()
    legacy = rollout_module.rollout_checkpoint(
        "config.yaml",
        tmp_path / "checkpoint",
        seed=7,
    )
    assert "gif" not in legacy
    assert "gif_frames" not in legacy
    assert "write" not in events
