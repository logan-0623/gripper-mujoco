from dataclasses import dataclass

import numpy as np
import pytest

from interaction_vla.representation_study.libero.replay import (
    RawReplayEpisode,
    replay_episode,
)


class FakeSimulator:
    def __init__(self, *, drift: float = 0.0) -> None:
        self.state = np.zeros(2, dtype=np.float64)
        self.drift = drift
        self.xml = ""

    def reset_from_xml_string(self, xml: str) -> None:
        self.xml = xml

    def set_state_from_flattened(self, state: np.ndarray) -> None:
        self.state = np.asarray(state, dtype=np.float64).copy()

    def get_state_flattened(self) -> np.ndarray:
        return self.state.copy()

    def step(self, action: np.ndarray) -> None:
        self.state = self.state + np.asarray(action[:2]) + self.drift

    def observation(self) -> dict[str, object]:
        return {"robot_state": self.state.copy()}

    def contacts(self) -> tuple[tuple[str, str], ...]:
        return (("left_finger", "black_bowl"),) if self.state[0] > 0 else ()


def _episode() -> RawReplayEpisode:
    actions = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    states = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    return RawReplayEpisode(
        suite="libero_spatial",
        task_id=0,
        episode_id="demo_0",
        model_xml="<mujoco/>",
        states=states,
        actions=actions,
    )


def test_replay_uses_pre_action_frames_and_checks_post_action_state() -> None:
    result = replay_episode(_episode(), FakeSimulator(), action_atol=1e-8)
    assert result.passed
    assert len(result.frames) == 2
    assert np.array_equal(result.frames[0].simulator_state, [0.0, 0.0])
    assert np.array_equal(result.frames[1].simulator_state, [1.0, 0.0])
    assert result.frames[1].contacts == (("left_finger", "black_bowl"),)
    assert result.max_abs_error == 0.0


def test_replay_reports_drift_instead_of_silently_accepting() -> None:
    result = replay_episode(_episode(), FakeSimulator(drift=0.1), action_atol=1e-8)
    assert not result.passed
    assert result.max_abs_error > 0.09


def test_replay_rejects_unknown_frame_action_offset() -> None:
    bad = _episode()
    with pytest.raises(ValueError, match=r"states.*actions or actions \+ 1"):
        RawReplayEpisode(
            suite=bad.suite,
            task_id=bad.task_id,
            episode_id=bad.episode_id,
            model_xml=bad.model_xml,
            states=bad.states[:-2],
            actions=bad.actions,
        )


def test_replay_accepts_official_equal_state_action_lengths() -> None:
    episode = _episode()
    equal_length = RawReplayEpisode(
        suite=episode.suite,
        task_id=episode.task_id,
        episode_id=episode.episode_id,
        model_xml=episode.model_xml,
        states=episode.states[:-1],
        actions=episode.actions,
    )
    result = replay_episode(equal_length, FakeSimulator(), action_atol=1e-8)
    assert result.passed
    assert len(result.l2_errors) == 1
    assert np.isnan(result.frames[-1].post_step_l2_error)
