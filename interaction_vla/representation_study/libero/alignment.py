from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np


ALIGNMENT_SCHEMA = "libero_source_alignment_v2"


@dataclass(frozen=True)
class EpisodeDescriptor:
    source_kind: Literal["raw", "lerobot"]
    suite: str
    task_id: int
    episode_id: str
    actions: np.ndarray
    robot_states: np.ndarray | None = None
    relative_path: str | None = None
    demo_key: str | None = None
    model_xml_sha256: str | None = None

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError("episode actions must have shape [frames, 7]")
        if not np.isfinite(actions).all():
            raise ValueError("episode actions must be finite")
        if self.source_kind == "raw":
            if not self.relative_path or not self.demo_key or not self.model_xml_sha256:
                raise ValueError("raw episode requires path, demo key, and model XML hash")
        elif self.robot_states is not None:
            states = np.asarray(self.robot_states)
            if states.shape != (actions.shape[0], 8) or not np.isfinite(states).all():
                raise ValueError("LeRobot robot_states must have shape [frames, 8]")


@dataclass(frozen=True)
class AlignmentRow:
    suite: str
    task_id: int
    raw_episode_id: str
    lerobot_episode_id: str
    frames: int
    raw_relative_path: str
    raw_demo_key: str
    model_xml_sha256: str
    action_sha256: str
    max_action_error: float


@dataclass(frozen=True)
class AlignmentManifest:
    rows: tuple[AlignmentRow, ...]
    action_atol: float
    semantic_sha256: str
    raw_episode_count: int
    lerobot_episode_count: int
    unmatched_raw_episode_ids: tuple[str, ...]
    unmatched_lerobot_episode_ids: tuple[str, ...]
    schema_version: str = ALIGNMENT_SCHEMA

    @property
    def matched_episode_count(self) -> int:
        return len(self.rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_atol": self.action_atol,
            "semantic_sha256": self.semantic_sha256,
            "rows": [asdict(row) for row in self.rows],
            "coverage": {
                "raw_episode_count": self.raw_episode_count,
                "lerobot_episode_count": self.lerobot_episode_count,
                "matched_episode_count": self.matched_episode_count,
                "raw_match_rate": self.matched_episode_count / self.raw_episode_count,
                "lerobot_match_rate": (
                    self.matched_episode_count / self.lerobot_episode_count
                ),
                "unmatched_raw_episode_ids": list(self.unmatched_raw_episode_ids),
                "unmatched_lerobot_episode_ids": list(
                    self.unmatched_lerobot_episode_ids
                ),
            },
        }


def _array_sha(array: np.ndarray) -> str:
    canonical = np.asarray(array, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _episode_key(row: EpisodeDescriptor) -> str:
    return f"{row.suite}:{row.task_id}:{row.episode_id}"


def _semantic_hash(
    rows: Sequence[AlignmentRow],
    action_atol: float,
    *,
    raw_episode_count: int,
    lerobot_episode_count: int,
    unmatched_raw_episode_ids: Sequence[str],
    unmatched_lerobot_episode_ids: Sequence[str],
) -> str:
    payload = {
        "schema_version": ALIGNMENT_SCHEMA,
        "action_atol": float(action_atol),
        "rows": [asdict(row) for row in rows],
        "raw_episode_count": raw_episode_count,
        "lerobot_episode_count": lerobot_episode_count,
        "unmatched_raw_episode_ids": list(unmatched_raw_episode_ids),
        "unmatched_lerobot_episode_ids": list(unmatched_lerobot_episode_ids),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def align_episode_sources(
    raw_episodes: Sequence[EpisodeDescriptor],
    lerobot_episodes: Sequence[EpisodeDescriptor],
    *,
    action_atol: float,
    require_all_raw: bool = True,
) -> AlignmentManifest:
    if action_atol <= 0:
        raise ValueError("action_atol must be positive")
    raw_ids = {(row.suite, row.task_id, row.episode_id) for row in raw_episodes}
    lerobot_ids = {(row.suite, row.task_id, row.episode_id) for row in lerobot_episodes}
    if len(raw_ids) != len(raw_episodes) or len(lerobot_ids) != len(lerobot_episodes):
        raise ValueError("source episode identities must be unique")
    if any(row.source_kind != "raw" for row in raw_episodes):
        raise ValueError("raw_episodes contains a non-raw descriptor")
    if any(row.source_kind != "lerobot" for row in lerobot_episodes):
        raise ValueError("lerobot_episodes contains a non-LeRobot descriptor")

    used_lerobot: set[str] = set()
    aligned: list[AlignmentRow] = []
    unmatched_raw: list[str] = []
    for raw in sorted(raw_episodes, key=lambda item: (item.suite, item.task_id, item.episode_id)):
        candidates: list[tuple[EpisodeDescriptor, float]] = []
        raw_actions = np.asarray(raw.actions, dtype=np.float64)
        for candidate in lerobot_episodes:
            candidate_actions = np.asarray(candidate.actions, dtype=np.float64)
            if (
                candidate.suite != raw.suite
                or candidate.task_id != raw.task_id
                or candidate_actions.shape != raw_actions.shape
            ):
                continue
            error = float(np.max(np.abs(candidate_actions - raw_actions), initial=0.0))
            if error <= action_atol:
                candidates.append((candidate, error))
        if not candidates:
            if require_all_raw:
                raise ValueError(
                    "no matching LeRobot episode for "
                    f"{raw.suite}/{raw.task_id}/{raw.episode_id}"
                )
            unmatched_raw.append(_episode_key(raw))
            continue
        if len(candidates) != 1:
            names = sorted(candidate.episode_id for candidate, _ in candidates)
            raise ValueError(
                f"ambiguous LeRobot alignment for {raw.episode_id}: {names}"
            )
        candidate, error = candidates[0]
        if candidate.episode_id in used_lerobot:
            raise ValueError(f"LeRobot episode matched more than once: {candidate.episode_id}")
        used_lerobot.add(candidate.episode_id)
        aligned.append(
            AlignmentRow(
                suite=raw.suite,
                task_id=raw.task_id,
                raw_episode_id=raw.episode_id,
                lerobot_episode_id=candidate.episode_id,
                frames=int(raw_actions.shape[0]),
                raw_relative_path=str(raw.relative_path),
                raw_demo_key=str(raw.demo_key),
                model_xml_sha256=str(raw.model_xml_sha256),
                action_sha256=_array_sha(raw_actions),
                max_action_error=error,
            )
        )
    rows = tuple(aligned)
    if not rows:
        raise ValueError("the raw and LeRobot sources have no exact shared episodes")
    unmatched_raw_ids = tuple(sorted(unmatched_raw))
    unmatched_lerobot_ids = tuple(
        sorted(
            _episode_key(row)
            for row in lerobot_episodes
            if row.episode_id not in used_lerobot
        )
    )
    return AlignmentManifest(
        rows=rows,
        action_atol=float(action_atol),
        semantic_sha256=_semantic_hash(
            rows,
            action_atol,
            raw_episode_count=len(raw_episodes),
            lerobot_episode_count=len(lerobot_episodes),
            unmatched_raw_episode_ids=unmatched_raw_ids,
            unmatched_lerobot_episode_ids=unmatched_lerobot_ids,
        ),
        raw_episode_count=len(raw_episodes),
        lerobot_episode_count=len(lerobot_episodes),
        unmatched_raw_episode_ids=unmatched_raw_ids,
        unmatched_lerobot_episode_ids=unmatched_lerobot_ids,
    )
