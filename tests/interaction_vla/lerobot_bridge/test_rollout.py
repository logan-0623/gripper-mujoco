import numpy as np
import pytest
import torch

from interaction_vla.lerobot_bridge.rollout import (
    ActionChunkQueue,
    BinaryGripperHysteresis,
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


def test_chunk_queue_queries_policy_only_after_eight_actions() -> None:
    calls = 0

    def predict() -> np.ndarray:
        nonlocal calls
        calls += 1
        chunk = np.zeros((8, 7), dtype=np.float32)
        chunk[:, 0] = calls
        return chunk

    queue = ActionChunkQueue(chunk_size=8)
    selected = [queue.next(predict) for _ in range(9)]

    assert calls == 2
    assert [item.queue_index for item in selected] == list(range(8)) + [0]
    assert selected[0].action[0] == 1.0
    assert selected[8].action[0] == 2.0
    assert selected[0].raw_chunk.shape == (8, 7)


def test_chunk_queue_rejects_nonfinite_or_wrong_shape() -> None:
    queue = ActionChunkQueue(chunk_size=8)
    with pytest.raises(ValueError, match="shape"):
        queue.next(lambda: np.zeros((7, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        queue.next(lambda: np.full((8, 7), np.nan, dtype=np.float32))
