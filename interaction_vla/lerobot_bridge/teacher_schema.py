from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np


SCHEMA_VERSION = "tc_tig_teacher_v1"
ENTITY_SLOTS = (
    "gripper",
    "target",
    "receptacle",
    "support",
    "distractor_0",
    "distractor_1",
)
RELATION_SLOTS = (
    "gripper_to_target",
    "target_to_receptacle",
    "target_to_support",
    "distractor_0_to_gripper",
    "distractor_0_to_target",
    "distractor_1_to_gripper",
    "distractor_1_to_target",
    "gripper_to_receptacle",
)
ENTITY_POSE_DIM = 9
ENTITY_SIZE_DIM = 3
RELATION_FEATURE_DIM = 24

ENTITY_ROLE_IDS = {name: index for index, name in enumerate(ENTITY_SLOTS)}
RELATION_TYPE_IDS = {name: index for index, name in enumerate(RELATION_SLOTS)}

RELATIVE_POSITION = slice(0, 3)
RELATIVE_ROTATION = slice(3, 6)
RELATIVE_LINEAR_VELOCITY = slice(6, 9)
RELATIVE_ANGULAR_VELOCITY = slice(9, 12)
SIGNED_MARGIN_0 = 12
SIGNED_MARGIN_1 = 13
SIGNED_MARGIN_2 = 14
SIGNED_MARGIN_3 = 15
PROBABILITY_0 = 16
PROBABILITY_1 = 17
RISK_0 = 18
RISK_1 = 19
ERROR_0 = 20
ERROR_1 = 21
VISIBILITY = 22
CONFIDENCE = 23

FORBIDDEN_FIELD_FRAGMENTS = (
    "contact_force",
    "normal_force",
    "tangential_force",
    "stable_grasp",
    "held_object",
    "expert_phase",
    "success",
    "termination_reason",
)


def teacher_schema_payload() -> dict[str, object]:
    return {
        "version": SCHEMA_VERSION,
        "entity_slots": list(ENTITY_SLOTS),
        "relation_slots": list(RELATION_SLOTS),
        "entity_role_ids": dict(ENTITY_ROLE_IDS),
        "relation_type_ids": dict(RELATION_TYPE_IDS),
        "entity_pose_layout": ["position_x", "position_y", "position_z", "rotation_6d"],
        "entity_size_layout": ["extent_x", "extent_y", "extent_z"],
        "relation_feature_layout": [
            "relative_position",
            "relative_rotation",
            "relative_linear_velocity",
            "relative_angular_velocity",
            "signed_margin_0",
            "signed_margin_1",
            "signed_margin_2",
            "signed_margin_3",
            "probability_0",
            "probability_1",
            "risk_0",
            "risk_1",
            "error_0",
            "error_1",
            "visibility",
            "confidence",
        ],
    }


def _array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    name: str,
) -> np.ndarray:
    values = np.asarray(value)
    if values.shape != shape or values.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values.copy()


@dataclass(frozen=True)
class TeacherFrame:
    frame_index: int
    timestamp: float
    state_hash: str
    entity_pose: np.ndarray
    entity_size: np.ndarray
    entity_role: np.ndarray
    entity_visibility: np.ndarray
    entity_mask: np.ndarray
    relation_values: np.ndarray
    relation_type: np.ndarray
    relation_mask: np.ndarray
    instance_agent: np.ndarray
    instance_wrist: np.ndarray
    depth_agent: np.ndarray
    depth_wrist: np.ndarray
    camera_intrinsics: np.ndarray
    camera_extrinsics_base: np.ndarray

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not np.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("timestamp must be finite and non-negative")
        if not self.state_hash:
            raise ValueError("state_hash must not be empty")
        specs = {
            "entity_pose": ((6, 9), np.dtype(np.float32)),
            "entity_size": ((6, 3), np.dtype(np.float32)),
            "entity_role": ((6,), np.dtype(np.int32)),
            "entity_visibility": ((6, 2), np.dtype(np.float32)),
            "entity_mask": ((6,), np.dtype(np.bool_)),
            "relation_values": ((8, 24), np.dtype(np.float32)),
            "relation_type": ((8,), np.dtype(np.int32)),
            "relation_mask": ((8,), np.dtype(np.bool_)),
            "camera_intrinsics": ((2, 3, 3), np.dtype(np.float32)),
            "camera_extrinsics_base": ((2, 4, 4), np.dtype(np.float32)),
        }
        for name, (shape, dtype) in specs.items():
            object.__setattr__(
                self,
                name,
                _array(getattr(self, name), shape=shape, dtype=dtype, name=name),
            )
        images = {
            "instance_agent": np.dtype(np.int32),
            "instance_wrist": np.dtype(np.int32),
            "depth_agent": np.dtype(np.float32),
            "depth_wrist": np.dtype(np.float32),
        }
        image_shape = np.asarray(self.instance_agent).shape
        if len(image_shape) != 2:
            raise ValueError("teacher image arrays must be two-dimensional")
        for name, dtype in images.items():
            object.__setattr__(
                self,
                name,
                _array(getattr(self, name), shape=image_shape, dtype=dtype, name=name),
            )
        if np.any(self.depth_agent < 0.0) or np.any(self.depth_wrist < 0.0):
            raise ValueError("teacher depths must be non-negative")

    @classmethod
    def zeros(
        cls,
        *,
        frame_index: int,
        timestamp: float,
        state_hash: str,
        image_size: tuple[int, int] = (1, 1),
    ) -> TeacherFrame:
        height, width = image_size
        if height < 1 or width < 1:
            raise ValueError("image_size values must be positive")
        return cls(
            frame_index=frame_index,
            timestamp=timestamp,
            state_hash=state_hash,
            entity_pose=np.zeros((6, 9), dtype=np.float32),
            entity_size=np.zeros((6, 3), dtype=np.float32),
            entity_role=np.arange(6, dtype=np.int32),
            entity_visibility=np.zeros((6, 2), dtype=np.float32),
            entity_mask=np.zeros(6, dtype=np.bool_),
            relation_values=np.zeros((8, 24), dtype=np.float32),
            relation_type=np.arange(8, dtype=np.int32),
            relation_mask=np.zeros(8, dtype=np.bool_),
            instance_agent=np.zeros((height, width), dtype=np.int32),
            instance_wrist=np.zeros((height, width), dtype=np.int32),
            depth_agent=np.zeros((height, width), dtype=np.float32),
            depth_wrist=np.zeros((height, width), dtype=np.float32),
            camera_intrinsics=np.zeros((2, 3, 3), dtype=np.float32),
            camera_extrinsics_base=np.zeros((2, 4, 4), dtype=np.float32),
        )


@dataclass(frozen=True)
class DistractorTrack:
    name: str
    age: int
    missing_count: int
    confidence: float
    score: float


class DistractorTracker:
    def __init__(
        self,
        *,
        count: int,
        replacement_margin: float,
        replacement_frames: int,
        dropout_frames: int,
    ) -> None:
        if count < 1 or replacement_frames < 1 or dropout_frames < 1:
            raise ValueError("tracker counts must be positive")
        if not np.isfinite(replacement_margin) or replacement_margin < 0.0:
            raise ValueError("replacement_margin must be finite and non-negative")
        self.count = int(count)
        self.replacement_margin = float(replacement_margin)
        self.replacement_frames = int(replacement_frames)
        self.dropout_frames = int(dropout_frames)
        self._tracks: list[DistractorTrack | None] = [None] * self.count
        self._challenger_counts: dict[tuple[str, str], int] = {}

    @property
    def tracks(self) -> tuple[DistractorTrack | None, ...]:
        return tuple(self._tracks)

    def update(self, scores: Mapping[str, float]) -> tuple[str | None, ...]:
        normalized: dict[str, float] = {}
        for name, score in scores.items():
            value = float(score)
            if not name or not np.isfinite(value):
                raise ValueError("distractor scores require non-empty names and finite values")
            normalized[str(name)] = value

        for index, track in enumerate(self._tracks):
            if track is None:
                continue
            if track.name in normalized:
                self._tracks[index] = replace(
                    track,
                    age=track.age + 1,
                    missing_count=0,
                    confidence=1.0,
                    score=normalized[track.name],
                )
            else:
                missing_count = track.missing_count + 1
                if missing_count > self.dropout_frames:
                    self._tracks[index] = None
                else:
                    self._tracks[index] = replace(
                        track,
                        age=track.age + 1,
                        missing_count=missing_count,
                        confidence=max(
                            0.0,
                            (self.dropout_frames - missing_count) / self.dropout_frames,
                        ),
                    )

        retained = {track.name for track in self._tracks if track is not None}
        candidates = sorted(
            ((score, name) for name, score in normalized.items() if name not in retained),
            key=lambda item: (-item[0], item[1]),
        )
        for index, track in enumerate(self._tracks):
            if track is None and candidates:
                score, name = candidates.pop(0)
                self._tracks[index] = DistractorTrack(name, 1, 0, 1.0, score)

        retained_present = [
            (index, track)
            for index, track in enumerate(self._tracks)
            if track is not None and track.name in normalized
        ]
        retained = {track.name for track in self._tracks if track is not None}
        challengers = sorted(
            ((score, name) for name, score in normalized.items() if name not in retained),
            key=lambda item: (-item[0], item[1]),
        )
        if retained_present and challengers:
            lowest_index, lowest = min(
                retained_present,
                key=lambda item: (normalized[item[1].name], item[1].name),
            )
            challenger_score, challenger_name = challengers[0]
            pair = (lowest.name, challenger_name)
            if challenger_score >= normalized[lowest.name] + self.replacement_margin:
                wins = self._challenger_counts.get(pair, 0) + 1
                self._challenger_counts = {pair: wins}
                if wins >= self.replacement_frames:
                    self._tracks[lowest_index] = DistractorTrack(
                        challenger_name, 1, 0, 1.0, challenger_score
                    )
                    self._challenger_counts.clear()
            else:
                self._challenger_counts.clear()
        else:
            self._challenger_counts.clear()

        return tuple(track.name if track is not None else None for track in self._tracks)
