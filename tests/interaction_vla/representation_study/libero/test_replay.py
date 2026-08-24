from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from interaction_vla.representation_study.libero import replay as replay_module
from interaction_vla.representation_study.libero.replay import (
    RawReplayEpisode,
    replay_episode,
)


def test_recorded_asset_paths_are_relocated_to_installed_roots(tmp_path: Path) -> None:
    robosuite_root = tmp_path / "robosuite"
    libero_assets_root = tmp_path / "libero-assets"
    robot_mesh = robosuite_root / "models/assets/robots/panda/meshes/link0.stl"
    object_texture = libero_assets_root / "textures/object.png"
    robot_mesh.parent.mkdir(parents=True)
    object_texture.parent.mkdir(parents=True)
    robot_mesh.touch()
    object_texture.touch()
    xml = """
    <mujoco><asset>
      <mesh name="robot" file="/Users/author/work/robosuite/robosuite/models/assets/robots/panda/meshes/link0.stl"/>
      <texture name="object" file="/Users/author/work/libero/chiliocosm/assets/scenes/../textures/object.png"/>
    </asset></mujoco>
    """

    relocated = replay_module.relocate_model_asset_paths(
        xml,
        robosuite_root=robosuite_root,
        libero_assets_root=libero_assets_root,
    )

    files = [element.get("file") for element in ET.fromstring(relocated).iter() if element.get("file")]
    assert files == [str(robot_mesh.resolve()), str(object_texture.resolve())]


def test_recorded_asset_relocation_rejects_unknown_absolute_paths(
    tmp_path: Path,
) -> None:
    xml = '<mujoco><asset><mesh file="/unknown/assets/object.stl"/></asset></mujoco>'

    with pytest.raises(ValueError, match="unrecognized absolute paths"):
        replay_module.relocate_model_asset_paths(
            xml,
            robosuite_root=tmp_path / "robosuite",
            libero_assets_root=tmp_path / "libero-assets",
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
