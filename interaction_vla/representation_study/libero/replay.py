from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol
import xml.etree.ElementTree as ET

import numpy as np


def relocate_model_asset_paths(
    xml: str,
    *,
    robosuite_root: str | Path,
    libero_assets_root: str | Path,
) -> str:
    """Relocate author-machine asset paths embedded in official LIBERO demos."""
    robot_root = Path(robosuite_root).resolve()
    libero_root = Path(libero_assets_root).resolve()
    tree = ET.fromstring(xml)
    unresolved: list[str] = []
    for element in tree.iter():
        old_path = element.get("file")
        if not old_path or not PurePosixPath(old_path).is_absolute():
            continue
        parts = PurePosixPath(old_path).parts
        robot_indices = [index for index, value in enumerate(parts) if value == "robosuite"]
        chiliocosm_indices = [
            index
            for index in range(len(parts) - 1)
            if parts[index : index + 2] == ("chiliocosm", "assets")
        ]
        if robot_indices:
            target = robot_root.joinpath(*parts[robot_indices[-1] + 1 :]).resolve()
            root = robot_root
        elif chiliocosm_indices:
            target = libero_root.joinpath(*parts[chiliocosm_indices[-1] + 2 :]).resolve()
            root = libero_root
        else:
            unresolved.append(old_path)
            continue
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError(f"recorded asset path escapes its installed root: {old_path}") from error
        if not target.is_file():
            raise FileNotFoundError(f"relocated LIBERO asset does not exist: {target}")
        element.set("file", str(target))
    if unresolved:
        raise ValueError(f"unrecognized absolute paths in recorded model XML: {unresolved}")
    return ET.tostring(tree, encoding="unicode")


@dataclass(frozen=True)
class RawReplayEpisode:
    suite: str
    task_id: int
    episode_id: str
    model_xml: str
    states: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        states = np.asarray(self.states)
        actions = np.asarray(self.actions)
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError("replay states and actions must be matrices")
        if states.shape[0] not in {actions.shape[0], actions.shape[0] + 1}:
            raise ValueError("replay states must contain actions or actions + 1 rows")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("replay states and actions must be finite")
        if not self.model_xml.strip():
            raise ValueError("replay model XML must be non-empty")


class Simulator(Protocol):
    def reset_from_xml_string(self, xml: str) -> None: ...
    def set_state_from_flattened(self, state: np.ndarray) -> None: ...
    def get_state_flattened(self) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> None: ...
    def observation(self) -> Mapping[str, object]: ...
    def contacts(self) -> tuple[tuple[str, str], ...]: ...


@dataclass(frozen=True)
class ReplayedFrame:
    frame_index: int
    action: np.ndarray
    simulator_state: np.ndarray
    observation: Mapping[str, object]
    contacts: tuple[tuple[str, str], ...]
    post_step_l2_error: float
    post_step_max_abs_error: float


@dataclass(frozen=True)
class ReplayResult:
    suite: str
    task_id: int
    episode_id: str
    frames: tuple[ReplayedFrame, ...]
    l2_errors: tuple[float, ...]
    max_abs_errors: tuple[float, ...]
    passed: bool

    @property
    def max_abs_error(self) -> float:
        return max(self.max_abs_errors, default=0.0)

    @property
    def l2_p95_error(self) -> float:
        return float(np.quantile(self.l2_errors, 0.95)) if self.l2_errors else 0.0


def replay_episode(
    episode: RawReplayEpisode,
    simulator: Simulator,
    *,
    action_atol: float,
    state_l2_p95_tolerance: float | None = None,
    state_max_abs_tolerance: float | None = None,
) -> ReplayResult:
    if action_atol <= 0:
        raise ValueError("action_atol must be positive")
    l2_tolerance = action_atol if state_l2_p95_tolerance is None else state_l2_p95_tolerance
    max_tolerance = action_atol if state_max_abs_tolerance is None else state_max_abs_tolerance
    simulator.reset_from_xml_string(episode.model_xml)
    simulator.set_state_from_flattened(np.asarray(episode.states[0], dtype=np.float64))
    frames: list[ReplayedFrame] = []
    l2_errors: list[float] = []
    max_errors: list[float] = []
    for index, action in enumerate(np.asarray(episode.actions, dtype=np.float64)):
        current = np.asarray(simulator.get_state_flattened(), dtype=np.float64).copy()
        expected_current = np.asarray(episode.states[index], dtype=np.float64)
        if current.shape != expected_current.shape:
            raise ValueError("simulator state shape differs from raw state shape")
        pre_error = float(np.max(np.abs(current - expected_current), initial=0.0))
        if pre_error > max_tolerance and index == 0:
            raise ValueError("simulator failed to restore the raw initial state")
        observation = dict(simulator.observation())
        contacts = tuple(tuple(str(name) for name in pair) for pair in simulator.contacts())
        simulator.step(action.copy())
        if index + 1 < len(episode.states):
            observed_next = np.asarray(simulator.get_state_flattened(), dtype=np.float64)
            expected_next = np.asarray(episode.states[index + 1], dtype=np.float64)
            difference = observed_next - expected_next
            l2_error = float(np.linalg.norm(difference))
            max_error = float(np.max(np.abs(difference), initial=0.0))
            l2_errors.append(l2_error)
            max_errors.append(max_error)
        else:
            l2_error = float("nan")
            max_error = float("nan")
        frames.append(
            ReplayedFrame(
                frame_index=index,
                action=action.copy(),
                simulator_state=current,
                observation=observation,
                contacts=contacts,
                post_step_l2_error=l2_error,
                post_step_max_abs_error=max_error,
            )
        )
    l2_p95 = float(np.quantile(l2_errors, 0.95)) if l2_errors else 0.0
    maximum = max(max_errors, default=0.0)
    return ReplayResult(
        suite=episode.suite,
        task_id=episode.task_id,
        episode_id=episode.episode_id,
        frames=tuple(frames),
        l2_errors=tuple(l2_errors),
        max_abs_errors=tuple(max_errors),
        passed=l2_p95 <= l2_tolerance and maximum <= max_tolerance,
    )


def load_raw_replay_episode(
    path: str | Path,
    *,
    suite: str,
    task_id: int,
    demo_key: str,
) -> RawReplayEpisode:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - depends on optional Linux env
        raise RuntimeError(
            "h5py is required to read original LIBERO demonstration files"
        ) from error
    source = Path(path)
    with h5py.File(source, "r") as handle:
        group = handle["data"][demo_key]
        model_xml = group.attrs.get("model_file")
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")
        if model_xml is None:
            raise ValueError(f"raw LIBERO demo {demo_key} has no model_file attribute")
        states = np.asarray(group["states"], dtype=np.float64)
        actions = np.asarray(group["actions"], dtype=np.float64)
    return RawReplayEpisode(
        suite=suite,
        task_id=task_id,
        episode_id=demo_key,
        model_xml=str(model_xml),
        states=states,
        actions=actions,
    )
