from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from interaction_vla.lerobot_bridge.collector import (
    collect_attempt,
    collection_seed,
    require_new_root,
)


class FakeEnv:
    policy_hz = 20

    def __init__(self, events: list[str], terminal_reason: str) -> None:
        self.events = events
        self.terminal_reason = terminal_reason
        self.contact_diagnostics = object()
        self.grasp_state = object()

    def reset(self, **kwargs):
        self.events.append("reset")
        return SimpleNamespace(
            gripper=SimpleNamespace(
                position=np.zeros(3),
                orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
            )
        )

    def proprioception(self):
        self.events.append("encode_state")
        value = np.zeros(23, dtype=np.float32)
        value[13:15] = 0.04
        return value

    def step(self, action):
        self.events.append("step")
        return SimpleNamespace(
            snapshot=SimpleNamespace(
                gripper=SimpleNamespace(
                    position=np.zeros(3),
                    orientation=np.asarray((1.0, 0.0, 0.0, 0.0)),
                )
            ),
            done=True,
            reason=SimpleNamespace(value=self.terminal_reason),
        )


class FakeExpert:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reset(self, *, seed: int) -> None:
        return None

    def act(self, snapshot, contacts, grasp):
        self.events.append("expert_action")
        return np.asarray(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32
        )


class FakeCapture:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def capture(self, env, *, include_teacher: bool):
        self.events.append("capture")
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        view = SimpleNamespace(rgb=rgb)
        return SimpleNamespace(views={"agent": view, "wrist": view})


class FakePolicyWriter:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.clear_count = 0

    def add_frame(self, **kwargs) -> None:
        self.events.append("add_frame")

    def clear_episode(self) -> None:
        self.clear_count += 1


class FakeTeacher:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reset(self) -> None:
        return None

    def extract(self, snapshot, camera_frame, *, state):
        self.events.append("teacher")
        return object()


def test_frame_is_recorded_before_the_matching_action_is_executed() -> None:
    events: list[str] = []
    env = FakeEnv(events, terminal_reason="success")
    expert = FakeExpert(events)
    capture = FakeCapture(events)
    policy_writer = FakePolicyWriter(events)
    teacher = FakeTeacher(events)

    result = collect_attempt(
        env=env,
        expert=expert,
        capture=capture,
        policy_writer=policy_writer,
        teacher=teacher,
        seed=11,
        object_count=2,
        task="Pick up the green target object and place it inside the receptacle.",
    )

    assert result.reason == "success"
    assert events[:7] == [
        "reset",
        "capture",
        "encode_state",
        "teacher",
        "expert_action",
        "add_frame",
        "step",
    ]


def test_rejected_attempt_clears_buffer_without_episode_commit() -> None:
    events: list[str] = []
    writer = FakePolicyWriter(events)
    result = collect_attempt(
        env=FakeEnv(events, terminal_reason="timeout"),
        expert=FakeExpert(events),
        capture=FakeCapture(events),
        policy_writer=writer,
        teacher=FakeTeacher(events),
        seed=11,
        object_count=2,
        task="Pick up the green target object and place it inside the receptacle.",
    )

    assert result.reason == "timeout"
    assert writer.clear_count == 1
    assert result.accepted is False


def test_physics_failure_attempt_is_rejected_and_cleared() -> None:
    events: list[str] = []
    writer = FakePolicyWriter(events)
    result = collect_attempt(
        env=FakeEnv(events, terminal_reason="physics_failure"),
        expert=FakeExpert(events),
        capture=FakeCapture(events),
        policy_writer=writer,
        teacher=FakeTeacher(events),
        seed=11,
        object_count=2,
        task="Pick up the green target object and place it inside the receptacle.",
    )

    assert result.accepted is False
    assert result.reason == "physics_failure"
    assert writer.clear_count == 1


def test_seed_schedule_is_deterministic_and_collision_free() -> None:
    values = [collection_seed(42, attempt) for attempt in range(50)]
    assert values == [collection_seed(42, attempt) for attempt in range(50)]
    assert len(set(values)) == 50


def test_existing_root_is_never_reused(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="new dataset root"):
        require_new_root(root)
    assert (root / "keep.txt").read_text(encoding="utf-8") == "user data"
