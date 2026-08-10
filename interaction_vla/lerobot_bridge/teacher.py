from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping

import mujoco
import numpy as np

from interaction_vla.graph.schema import EntityState, SceneSnapshot
from interaction_vla.lerobot_bridge.codecs import EndEffectorStateCodec
from interaction_vla.lerobot_bridge.config import TeacherConfig
from interaction_vla.lerobot_bridge.teacher_schema import (
    CONFIDENCE,
    ENTITY_ROLE_IDS,
    ERROR_0,
    ERROR_1,
    OPERATOR_IDS,
    PREDICATE_IDS,
    PROBABILITY_0,
    PROBABILITY_1,
    RELATION_TYPE_IDS,
    RELATIVE_ANGULAR_VELOCITY,
    RELATIVE_LINEAR_VELOCITY,
    RELATIVE_POSITION,
    RELATIVE_ROTATION,
    RISK_0,
    RISK_1,
    SIGNED_MARGIN_0,
    SIGNED_MARGIN_1,
    SIGNED_MARGIN_2,
    SIGNED_MARGIN_3,
    VISIBILITY,
    DistractorTracker,
    TeacherFrame,
)


ESTABLISH = OPERATOR_IDS["establish"]
BREAK = OPERATOR_IDS["break"]
INCREASE = OPERATOR_IDS["increase"]
PRESERVE = OPERATOR_IDS["preserve"]
DECREASE = OPERATOR_IDS["decrease"]

PROXIMITY = PREDICATE_IDS["proximity"]
ALIGNMENT = PREDICATE_IDS["alignment"]
ENCLOSURE = PREDICATE_IDS["enclosure"]
CO_MOTION = PREDICATE_IDS["co_motion"]
CONTAINMENT = PREDICATE_IDS["containment"]
SUPPORT = PREDICATE_IDS["support"]
CLEARANCE = PREDICATE_IDS["clearance"]

GOAL_CANDIDATES = (
    (0, ESTABLISH, PROXIMITY),
    (0, ESTABLISH, ALIGNMENT),
    (0, ESTABLISH, ENCLOSURE),
    (0, ESTABLISH, CO_MOTION),
    (1, ESTABLISH, CONTAINMENT),
    (2, ESTABLISH, SUPPORT),
    (0, BREAK, CO_MOTION),
    (7, INCREASE, CLEARANCE),
)


def _rotation(entity: EntityState) -> np.ndarray:
    return EndEffectorStateCodec.quaternion_to_matrix(entity.orientation)


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, matrix.reshape(-1))
    return quaternion.astype(np.float32)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    quaternion = _matrix_to_quaternion(rotation).astype(np.float64)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    scalar = float(np.clip(quaternion[0], -1.0, 1.0))
    angle = 2.0 * math.acos(scalar)
    sine_half = math.sqrt(max(0.0, 1.0 - scalar * scalar))
    if sine_half < 1e-8:
        return (2.0 * quaternion[1:]).astype(np.float32)
    return (angle * quaternion[1:] / sine_half).astype(np.float32)


def transform_snapshot_passive(
    snapshot: SceneSnapshot,
    *,
    translation: np.ndarray,
    yaw_radians: float,
) -> SceneSnapshot:
    offset = np.asarray(translation, dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("translation must be a finite 3D vector")
    if not np.isfinite(yaw_radians):
        raise ValueError("yaw_radians must be finite")
    cosine = math.cos(yaw_radians)
    sine = math.sin(yaw_radians)
    yaw = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )

    def transformed(entity: EntityState) -> EntityState:
        return replace(
            entity,
            position=yaw @ entity.position + offset,
            orientation=_matrix_to_quaternion(yaw @ _rotation(entity)),
            linear_velocity=yaw @ entity.linear_velocity,
            angular_velocity=yaw @ entity.angular_velocity,
        )

    return replace(
        snapshot,
        gripper=transformed(snapshot.gripper),
        objects=tuple(transformed(entity) for entity in snapshot.objects),
        receptacle=transformed(snapshot.receptacle),
        support=transformed(snapshot.support),
    )


def _radius(entity: EntityState, *, xy: bool = False) -> float:
    dimensions = entity.size[:2] if xy else entity.size
    return float(np.linalg.norm(dimensions) / 2.0)


def _surface_distance(
    first: EntityState,
    second: EntityState,
    *,
    safety_margin: float,
    xy: bool = False,
) -> float:
    dimensions = slice(0, 2) if xy else slice(0, 3)
    center_distance = float(np.linalg.norm(first.position[dimensions] - second.position[dimensions]))
    return max(0.0, center_distance - _radius(first, xy=xy) - _radius(second, xy=xy) - safety_margin)


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    denominator = float(np.dot(segment, segment))
    if denominator < 1e-12:
        return float(np.linalg.norm(np.asarray(point) - np.asarray(start)))
    fraction = float(np.clip(np.dot(np.asarray(point) - start, segment) / denominator, 0.0, 1.0))
    closest = np.asarray(start) + fraction * segment
    return float(np.linalg.norm(np.asarray(point) - closest))


def _risk_scores(
    snapshot: SceneSnapshot,
    *,
    safety_margin: float,
) -> dict[str, float]:
    target = snapshot.target_object
    gripper = snapshot.gripper
    receptacle = snapshot.receptacle
    values: dict[str, float] = {}
    for entity in snapshot.objects:
        if entity.target:
            continue
        wrong_gap = _surface_distance(
            entity, target, safety_margin=safety_margin, xy=True
        )
        closing_gap = _surface_distance(
            entity, gripper, safety_margin=safety_margin
        )
        approach_gap = max(
            0.0,
            _point_to_segment_distance(
                entity.position, gripper.position, target.position
            )
            - _radius(entity)
            - safety_margin,
        )
        transport_gap = max(
            0.0,
            _point_to_segment_distance(
                entity.position, target.position, receptacle.position
            )
            - _radius(entity)
            - safety_margin,
        )
        terms = (
            math.exp(-wrong_gap / 0.08),
            math.exp(-closing_gap / 0.08),
            math.exp(-approach_gap / 0.06),
            math.exp(-transport_gap / 0.06),
        )
        values[entity.name] = float(np.clip(0.25 * sum(terms), 0.0, 1.0))
    return values


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-clipped))


def _relative_values(source: EntityState, destination: EntityState) -> np.ndarray:
    source_rotation = _rotation(source)
    destination_rotation = _rotation(destination)
    values = np.zeros(24, dtype=np.float32)
    values[RELATIVE_POSITION] = source_rotation.T @ (
        destination.position - source.position
    )
    values[RELATIVE_ROTATION] = _rotation_vector(
        source_rotation.T @ destination_rotation
    )
    values[RELATIVE_LINEAR_VELOCITY] = source_rotation.T @ (
        destination.linear_velocity - source.linear_velocity
    )
    values[RELATIVE_ANGULAR_VELOCITY] = source_rotation.T @ (
        destination.angular_velocity - source.angular_velocity
    )
    return values


def _manipulation_values(gripper: EntityState, target: EntityState) -> np.ndarray:
    values = _relative_values(gripper, target)
    gap = _surface_distance(gripper, target, safety_margin=0.0)
    local = values[RELATIVE_POSITION]
    alignment = abs(float(np.dot(_rotation(gripper)[:, 2], _rotation(target)[:, 2])))
    co_motion_error = float(
        np.linalg.norm(target.linear_velocity - gripper.linear_velocity)
    )
    values[SIGNED_MARGIN_0] = gap
    values[SIGNED_MARGIN_1] = abs(float(local[0])) - target.size[0] / 2.0
    values[SIGNED_MARGIN_2] = abs(float(local[1])) - target.size[1] / 2.0
    values[SIGNED_MARGIN_3] = alignment
    values[PROBABILITY_0] = _sigmoid(-gap / 0.01)
    values[PROBABILITY_1] = math.exp(-co_motion_error / 0.10)
    values[ERROR_0] = np.clip(gap / 0.20, 0.0, 1.0)
    values[ERROR_1] = np.clip(1.0 - alignment, 0.0, 1.0)
    return values


def _placement_values(receptacle: EntityState, target: EntityState) -> np.ndarray:
    values = _relative_values(receptacle, target)
    local = values[RELATIVE_POSITION]
    x_margin = receptacle.size[0] / 2.0 - target.size[0] / 2.0 - abs(float(local[0]))
    y_margin = receptacle.size[1] / 2.0 - target.size[1] / 2.0 - abs(float(local[1]))
    bottom_gap = float(local[2] - (receptacle.size[2] + target.size[2]) / 2.0)
    entrance = min(x_margin, y_margin)
    containment_error = np.clip(max(-x_margin, -y_margin, abs(bottom_gap)) / 0.10, 0.0, 1.0)
    orientation_error = np.clip(
        1.0 - abs(float(np.dot(_rotation(receptacle)[:, 2], _rotation(target)[:, 2]))),
        0.0,
        1.0,
    )
    values[SIGNED_MARGIN_0] = x_margin
    values[SIGNED_MARGIN_1] = y_margin
    values[SIGNED_MARGIN_2] = bottom_gap
    values[SIGNED_MARGIN_3] = entrance
    values[PROBABILITY_0] = _sigmoid(min(x_margin, y_margin, -abs(bottom_gap)) / 0.01)
    values[PROBABILITY_1] = _sigmoid(-abs(bottom_gap) / 0.01)
    values[ERROR_0] = containment_error
    values[ERROR_1] = orientation_error
    return values


def _support_values(support: EntityState, target: EntityState) -> np.ndarray:
    values = _relative_values(support, target)
    local = values[RELATIVE_POSITION]
    bottom_gap = float(local[2] - (support.size[2] + target.size[2]) / 2.0)
    x_overlap = support.size[0] / 2.0 + target.size[0] / 2.0 - abs(float(local[0]))
    y_overlap = support.size[1] / 2.0 + target.size[1] / 2.0 - abs(float(local[1]))
    vertical_velocity = float(values[RELATIVE_LINEAR_VELOCITY][2])
    values[SIGNED_MARGIN_0] = bottom_gap
    values[SIGNED_MARGIN_1] = x_overlap
    values[SIGNED_MARGIN_2] = y_overlap
    values[SIGNED_MARGIN_3] = vertical_velocity
    values[PROBABILITY_0] = _sigmoid(min(x_overlap, y_overlap, -abs(bottom_gap)) / 0.01)
    values[ERROR_0] = np.clip(
        (abs(bottom_gap) + max(0.0, -x_overlap) + max(0.0, -y_overlap)) / 0.10,
        0.0,
        1.0,
    )
    values[ERROR_1] = np.clip(abs(vertical_velocity) / 0.20, 0.0, 1.0)
    return values


def _risk_values(
    source: EntityState,
    distractor: EntityState,
    target: EntityState,
    receptacle: EntityState,
    *,
    safety_margin: float,
) -> np.ndarray:
    values = _relative_values(source, distractor)
    closing = _surface_distance(
        source, distractor, safety_margin=safety_margin
    )
    swept = max(
        0.0,
        _point_to_segment_distance(
            distractor.position, source.position, target.position
        )
        - _radius(distractor)
        - safety_margin,
    )
    relative_position = distractor.position - source.position
    relative_velocity = distractor.linear_velocity - source.linear_velocity
    closing_speed = max(
        0.0,
        -float(np.dot(relative_position, relative_velocity))
        / max(float(np.linalg.norm(relative_position)), 1e-6),
    )
    ttc = min(1.0, closing / max(closing_speed, 1e-6))
    target_gap = _surface_distance(
        distractor, target, safety_margin=safety_margin
    )
    wrong_risk = math.exp(-target_gap / 0.08)
    transport_clearance = max(
        0.0,
        _point_to_segment_distance(
            distractor.position, target.position, receptacle.position
        )
        - _radius(distractor)
        - safety_margin,
    )
    collision_risk = math.exp(-min(swept, transport_clearance) / 0.06)
    values[SIGNED_MARGIN_0] = closing
    values[SIGNED_MARGIN_1] = swept
    values[SIGNED_MARGIN_2] = ttc
    values[SIGNED_MARGIN_3] = target_gap
    values[RISK_0] = wrong_risk
    values[RISK_1] = collision_risk
    values[ERROR_0] = wrong_risk
    values[ERROR_1] = collision_risk
    return values


def _clearance_values(receptacle: EntityState, gripper: EntityState) -> np.ndarray:
    values = _relative_values(receptacle, gripper)
    local = values[RELATIVE_POSITION]
    entrance_x = receptacle.size[0] / 2.0 - abs(float(local[0]))
    entrance_y = receptacle.size[1] / 2.0 - abs(float(local[1]))
    retreat_residual = max(0.0, receptacle.size[2] - float(local[2]))
    swept_margin = min(entrance_x, entrance_y)
    values[SIGNED_MARGIN_0] = entrance_x
    values[SIGNED_MARGIN_1] = entrance_y
    values[SIGNED_MARGIN_2] = retreat_residual
    values[SIGNED_MARGIN_3] = swept_margin
    values[PROBABILITY_0] = _sigmoid(swept_margin / 0.01)
    values[PROBABILITY_1] = _sigmoid(-swept_margin / 0.01)
    values[ERROR_0] = np.clip(retreat_residual / 0.10, 0.0, 1.0)
    values[ERROR_1] = np.clip(max(0.0, -swept_margin) / 0.10, 0.0, 1.0)
    return values


def _geom_instance_lookup(
    model: mujoco.MjModel,
    slot_entities: tuple[EntityState | None, ...],
) -> np.ndarray:
    lookup = np.zeros(model.ngeom, dtype=np.int32)
    object_instances = {
        entity.name: index + 1
        for index, entity in enumerate(slot_entities)
        if entity is not None and entity.entity_type == "object"
    }
    gripper_bodies = {"hand", "left_finger", "right_finger"}
    for geom_id in range(model.ngeom):
        geom_name = model.geom(geom_id).name or ""
        body_id = int(model.geom_bodyid[geom_id])
        ancestry: set[str] = set()
        cursor = body_id
        while cursor > 0:
            ancestry.add(model.body(cursor).name or "")
            cursor = int(model.body_parentid[cursor])
        if ancestry & gripper_bodies:
            lookup[geom_id] = 1
        elif any(name in ancestry for name in object_instances):
            lookup[geom_id] = next(
                instance for name, instance in object_instances.items() if name in ancestry
            )
        elif any(name.startswith("receptacle") for name in ancestry) or geom_name.startswith(
            "receptacle_"
        ):
            lookup[geom_id] = 3
        elif geom_name == "table" or "table" in ancestry:
            lookup[geom_id] = 4
    return lookup


def _canonical_instance(raw: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    segmentation = np.asarray(raw, dtype=np.int32)
    if segmentation.ndim != 3 or segmentation.shape[2] != 2:
        raise ValueError("raw segmentation must have shape HxWx2")
    result = np.zeros(segmentation.shape[:2], dtype=np.int32)
    geom_mask = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    geom_ids = segmentation[..., 0]
    valid = geom_mask & (geom_ids >= 0) & (geom_ids < len(lookup))
    result[valid] = lookup[geom_ids[valid]]
    return result


def _with_visual_observations(
    frame: TeacherFrame,
    *,
    camera_frame: Any,
    model: mujoco.MjModel | None,
    calibration: Mapping[str, object] | None,
    slot_entities: tuple[EntityState | None, ...],
) -> TeacherFrame:
    if model is None or calibration is None:
        raise ValueError("visual teacher extraction requires model and calibration")
    lookup = _geom_instance_lookup(model, slot_entities)
    instances: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    visibility = np.zeros((6, 2), dtype=np.float32)
    for view_index, view_name in enumerate(("agent", "wrist")):
        view = camera_frame.views[view_name]
        instance = _canonical_instance(view.segmentation, lookup)
        instances.append(instance)
        depths.append(np.asarray(view.depth, dtype=np.float32))
        for entity_index in range(6):
            visibility[entity_index, view_index] = float(
                np.count_nonzero(instance == entity_index + 1) / instance.size
            )
        view_calibration = calibration[view_name]
        intrinsics.append(np.asarray(view_calibration["intrinsics"], dtype=np.float32))
        extrinsics.append(
            np.asarray(view_calibration["camera_to_base"], dtype=np.float32)
        )
    relation_values = frame.relation_values.copy()
    pairs = ((0, 1), (2, 1), (3, 1), (0, 4), (1, 4), (0, 5), (1, 5), (2, 0))
    for relation_index, (first, second) in enumerate(pairs):
        relation_values[relation_index, VISIBILITY] = float(
            np.max(np.minimum(visibility[first], visibility[second]))
        )
    return replace(
        frame,
        entity_visibility=visibility,
        relation_values=relation_values,
        instance_agent=instances[0],
        instance_wrist=instances[1],
        depth_agent=depths[0],
        depth_wrist=depths[1],
        camera_intrinsics=np.stack(intrinsics),
        camera_extrinsics_base=np.stack(extrinsics),
    )


class TCTIGTeacherExtractor:
    def __init__(
        self,
        config: TeacherConfig,
        *,
        model: mujoco.MjModel | None = None,
        calibration: Mapping[str, object] | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.calibration = calibration
        self.reset()

    @classmethod
    def from_defaults(
        cls,
        *,
        model: mujoco.MjModel | None = None,
        calibration: Mapping[str, object] | None = None,
    ) -> TCTIGTeacherExtractor:
        return cls(TeacherConfig(), model=model, calibration=calibration)

    def reset(self) -> None:
        self.tracker = DistractorTracker(
            count=self.config.distractor_count,
            replacement_margin=self.config.replacement_margin,
            replacement_frames=self.config.replacement_frames,
            dropout_frames=self.config.dropout_frames,
        )
        self._last_slot_entities: tuple[EntityState | None, ...] | None = None

    def extract_geometry(
        self,
        snapshot: SceneSnapshot,
        *,
        frame_index: int,
        timestamp: float,
        state_hash: str,
    ) -> TeacherFrame:
        frame, slot_entities = _geometry_frame(
            snapshot,
            tracker=self.tracker,
            config=self.config,
            frame_index=frame_index,
            timestamp=timestamp,
            state_hash=state_hash,
        )
        self._last_slot_entities = slot_entities
        return frame

    def extract(
        self,
        snapshot: SceneSnapshot,
        camera_frame: Any,
        *,
        state: np.ndarray,
    ) -> TeacherFrame:
        state_values = np.asarray(state, dtype=np.float32)
        if state_values.shape != (10,) or not np.isfinite(state_values).all():
            raise ValueError("state must be a finite 10D vector")
        frame = self.extract_geometry(
            snapshot,
            frame_index=camera_frame.policy_step,
            timestamp=camera_frame.timestamp,
            state_hash=camera_frame.state_hash,
        )
        assert self._last_slot_entities is not None
        frame_calibration = getattr(camera_frame, "calibration", None)
        return _with_visual_observations(
            frame,
            camera_frame=camera_frame,
            model=self.model,
            calibration=frame_calibration or self.calibration,
            slot_entities=self._last_slot_entities,
        )


def _geometry_frame(
    snapshot: SceneSnapshot,
    *,
    tracker: DistractorTracker,
    config: TeacherConfig,
    frame_index: int,
    timestamp: float,
    state_hash: str,
) -> tuple[TeacherFrame, tuple[EntityState | None, ...]]:
    target = snapshot.target_object
    by_name = {entity.name: entity for entity in snapshot.objects if not entity.target}
    tracked_names = tracker.update(
        _risk_scores(snapshot, safety_margin=config.safety_margin_m)
    )
    distractors = tuple(by_name.get(name) if name is not None else None for name in tracked_names)
    while len(distractors) < 2:
        distractors += (None,)
    entities: tuple[EntityState | None, ...] = (
        snapshot.gripper,
        target,
        snapshot.receptacle,
        snapshot.support,
        distractors[0],
        distractors[1],
    )
    entity_pose = np.zeros((6, 9), dtype=np.float32)
    entity_size = np.zeros((6, 3), dtype=np.float32)
    entity_visibility = np.zeros((6, 2), dtype=np.float32)
    entity_mask = np.zeros(6, dtype=np.bool_)
    confidence = np.ones(6, dtype=np.float32)
    for index, entity in enumerate(entities):
        if entity is None:
            continue
        entity_pose[index, :3] = entity.position
        entity_pose[index, 3:] = EndEffectorStateCodec.encode_rotation(_rotation(entity))
        entity_size[index] = entity.size
        entity_visibility[index] = 1.0
        entity_mask[index] = True
    for offset, track in enumerate(tracker.tracks):
        if track is not None:
            confidence[4 + offset] = track.confidence

    relation_values = np.zeros((8, 24), dtype=np.float32)
    relation_values[0] = _manipulation_values(snapshot.gripper, target)
    relation_values[1] = _placement_values(snapshot.receptacle, target)
    relation_values[2] = _support_values(snapshot.support, target)
    for offset, distractor in enumerate(distractors[:2]):
        if distractor is None:
            continue
        relation_values[3 + 2 * offset] = _risk_values(
            snapshot.gripper,
            distractor,
            target,
            snapshot.receptacle,
            safety_margin=config.safety_margin_m,
        )
        relation_values[4 + 2 * offset] = _risk_values(
            target,
            distractor,
            target,
            snapshot.receptacle,
            safety_margin=config.safety_margin_m,
        )
    relation_values[7] = _clearance_values(snapshot.receptacle, snapshot.gripper)

    relation_entity_indices = (
        (0, 1),
        (2, 1),
        (3, 1),
        (0, 4),
        (1, 4),
        (0, 5),
        (1, 5),
        (2, 0),
    )
    relation_mask = np.asarray(
        [entity_mask[first] and entity_mask[second] for first, second in relation_entity_indices],
        dtype=np.bool_,
    )
    for relation_index, (first, second) in enumerate(relation_entity_indices):
        relation_values[relation_index, VISIBILITY] = float(
            min(entity_visibility[first].max(), entity_visibility[second].max())
        )
        relation_values[relation_index, CONFIDENCE] = float(
            min(confidence[first], confidence[second])
        )
    return (
        TeacherFrame(
            frame_index=frame_index,
            timestamp=timestamp,
            state_hash=state_hash,
            entity_pose=entity_pose,
            entity_size=entity_size,
            entity_role=np.asarray(list(ENTITY_ROLE_IDS.values()), dtype=np.int32),
            entity_visibility=entity_visibility,
            entity_mask=entity_mask,
            relation_values=relation_values,
            relation_type=np.asarray(list(RELATION_TYPE_IDS.values()), dtype=np.int32),
            relation_mask=relation_mask,
            instance_agent=np.zeros((1, 1), dtype=np.int32),
            instance_wrist=np.zeros((1, 1), dtype=np.int32),
            depth_agent=np.zeros((1, 1), dtype=np.float32),
            depth_wrist=np.zeros((1, 1), dtype=np.float32),
            camera_intrinsics=np.zeros((2, 3, 3), dtype=np.float32),
            camera_extrinsics_base=np.zeros((2, 4, 4), dtype=np.float32),
        ),
        entities,
    )


def _typed_candidate_errors(relation_values: np.ndarray) -> np.ndarray:
    values = np.asarray(relation_values, dtype=np.float32)
    return np.stack(
        (
            values[:, 0, ERROR_0],
            values[:, 0, ERROR_1],
            1.0 - values[:, 0, PROBABILITY_0],
            1.0 - values[:, 0, PROBABILITY_1],
            values[:, 1, ERROR_0],
            values[:, 2, ERROR_0],
            values[:, 0, PROBABILITY_1],
            values[:, 7, ERROR_0],
        ),
        axis=1,
    ).astype(np.float32)


def _typed_candidate_valid(relation_values: np.ndarray) -> np.ndarray:
    values = np.asarray(relation_values, dtype=np.float32)
    proximity_error = values[:, 0, ERROR_0]
    enclosure_error = 1.0 - values[:, 0, PROBABILITY_0]
    co_motion_error = 1.0 - values[:, 0, PROBABILITY_1]
    co_motion_probability = values[:, 0, PROBABILITY_1]
    containment_error = values[:, 1, ERROR_0]
    support_error = values[:, 2, ERROR_0]
    return np.stack(
        (
            np.ones(len(values), dtype=np.bool_),
            np.ones(len(values), dtype=np.bool_),
            proximity_error < 0.5,
            enclosure_error < 0.5,
            co_motion_error < 0.5,
            containment_error < 0.5,
            (containment_error < 0.25) & (support_error < 0.25),
            (containment_error < 0.25) & (co_motion_probability < 0.25),
        ),
        axis=1,
    )


def _label_typed_relation_goals(
    relation_values: np.ndarray,
    confidence: np.ndarray,
    *,
    horizon: int,
    minimum_improvement: float,
) -> np.ndarray:
    candidate_errors = _typed_candidate_errors(relation_values)
    candidate_valid = _typed_candidate_valid(relation_values)
    candidate_relations = np.asarray(
        [candidate[0] for candidate in GOAL_CANDIDATES], dtype=np.int64
    )
    labels = np.zeros((len(relation_values), 5), dtype=np.float32)
    previous_relation = 0
    previous_predicate = PROXIMITY
    for frame_index in range(len(relation_values)):
        stop = min(len(relation_values), frame_index + horizon + 1)
        improvements = np.full(len(GOAL_CANDIDATES), -np.inf, dtype=np.float32)
        residuals = np.zeros(len(GOAL_CANDIDATES), dtype=np.float32)
        selected_confidence = np.zeros(len(GOAL_CANDIDATES), dtype=np.float32)
        if frame_index + 1 < stop:
            future_errors = candidate_errors[frame_index + 1 : stop]
            future_confidence = confidence[frame_index + 1 : stop]
            for candidate_index, relation_index in enumerate(candidate_relations):
                if not candidate_valid[frame_index, candidate_index]:
                    continue
                best_offset = int(np.argmin(future_errors[:, candidate_index]))
                residual = (
                    future_errors[best_offset, candidate_index]
                    - candidate_errors[frame_index, candidate_index]
                )
                combined_confidence = min(
                    confidence[frame_index, relation_index],
                    future_confidence[best_offset, relation_index],
                )
                residuals[candidate_index] = residual
                selected_confidence[candidate_index] = combined_confidence
                improvements[candidate_index] = -residual * combined_confidence
        candidate_index = int(np.argmax(improvements))
        improvement = float(improvements[candidate_index])
        if improvement >= minimum_improvement:
            relation_index, operator_index, predicate_index = GOAL_CANDIDATES[
                candidate_index
            ]
            previous_relation = relation_index
            previous_predicate = predicate_index
            labels[frame_index] = (
                relation_index,
                operator_index,
                predicate_index,
                residuals[candidate_index],
                selected_confidence[candidate_index],
            )
        else:
            labels[frame_index] = (
                previous_relation,
                PRESERVE,
                previous_predicate,
                0.0,
                confidence[frame_index, previous_relation],
            )
    return labels


def label_relation_goals(
    errors: np.ndarray,
    confidence: np.ndarray,
    *,
    relation_values: np.ndarray | None = None,
    horizon: int,
    minimum_improvement: float,
) -> np.ndarray:
    error_values = np.asarray(errors, dtype=np.float32)
    confidence_values = np.asarray(confidence, dtype=np.float32)
    if error_values.ndim != 3 or error_values.shape[1:] != (8, 2):
        raise ValueError("errors must have shape [T, 8, 2]")
    if confidence_values.shape != error_values.shape[:2]:
        raise ValueError("confidence must have shape [T, 8]")
    if not np.isfinite(error_values).all() or not np.isfinite(confidence_values).all():
        raise ValueError("goal inputs must be finite")
    if np.any(confidence_values < 0.0) or np.any(confidence_values > 1.0):
        raise ValueError("confidence must lie within [0, 1]")
    if horizon < 1 or not np.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("goal horizon must be positive and margin non-negative")

    if relation_values is not None:
        typed_values = np.asarray(relation_values, dtype=np.float32)
        if typed_values.shape != (len(error_values), 8, 24):
            raise ValueError("relation_values must have shape [T, 8, 24]")
        if not np.isfinite(typed_values).all():
            raise ValueError("relation_values must be finite")
        if not np.allclose(
            error_values,
            typed_values[:, :, (ERROR_0, ERROR_1)],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError("errors must match relation_values error channels")
        if not np.allclose(
            confidence_values,
            typed_values[:, :, CONFIDENCE],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError("confidence must match relation_values")
        return _label_typed_relation_goals(
            typed_values,
            confidence_values,
            horizon=horizon,
            minimum_improvement=minimum_improvement,
        )

    labels = np.zeros((len(error_values), 5), dtype=np.float32)
    previous_relation = 0
    previous_predicate = 0
    for frame_index in range(len(error_values)):
        stop = min(len(error_values), frame_index + horizon + 1)
        if frame_index + 1 < stop:
            future = error_values[frame_index + 1 : stop]
            future_confidence = confidence_values[frame_index + 1 : stop]
            weighted_improvement = np.full((8, 2), -np.inf, dtype=np.float32)
            residuals = np.zeros((8, 2), dtype=np.float32)
            selected_confidence = np.zeros((8, 2), dtype=np.float32)
            for relation_index in range(8):
                for predicate_index in range(2):
                    best_offset = int(np.argmin(future[:, relation_index, predicate_index]))
                    residual = (
                        future[best_offset, relation_index, predicate_index]
                        - error_values[frame_index, relation_index, predicate_index]
                    )
                    combined_confidence = min(
                        confidence_values[frame_index, relation_index],
                        future_confidence[best_offset, relation_index],
                    )
                    residuals[relation_index, predicate_index] = residual
                    selected_confidence[relation_index, predicate_index] = combined_confidence
                    weighted_improvement[relation_index, predicate_index] = (
                        -residual * combined_confidence
                    )
            flat_index = int(np.argmax(weighted_improvement))
            relation_index, predicate_index = np.unravel_index(
                flat_index, weighted_improvement.shape
            )
            improvement = float(weighted_improvement[relation_index, predicate_index])
        else:
            relation_index = previous_relation
            predicate_index = previous_predicate
            improvement = -math.inf
            residuals = np.zeros((8, 2), dtype=np.float32)
            selected_confidence = np.zeros((8, 2), dtype=np.float32)

        if improvement >= minimum_improvement:
            previous_relation = int(relation_index)
            previous_predicate = int(predicate_index)
            labels[frame_index] = (
                previous_relation,
                DECREASE,
                previous_predicate,
                residuals[relation_index, predicate_index],
                selected_confidence[relation_index, predicate_index],
            )
        else:
            labels[frame_index] = (
                previous_relation,
                PRESERVE,
                previous_predicate,
                0.0,
                confidence_values[frame_index, previous_relation],
            )
    return labels


def validate_typed_relation_goals(
    relation_values: np.ndarray,
    labels: np.ndarray,
    *,
    horizon: int,
    minimum_improvement: float,
) -> None:
    values = np.asarray(relation_values)
    goals = np.asarray(labels)
    if values.ndim != 3 or values.shape[1:] != (8, 24):
        raise ValueError("typed relation values must have shape [T, 8, 24]")
    if goals.shape != (len(values), 5) or goals.dtype != np.float32:
        raise ValueError("typed relation goals must be float32 with shape [T, 5]")
    expected = label_relation_goals(
        values[:, :, (ERROR_0, ERROR_1)],
        values[:, :, CONFIDENCE],
        relation_values=values,
        horizon=horizon,
        minimum_improvement=minimum_improvement,
    )
    if not np.allclose(goals, expected, rtol=0.0, atol=1e-7):
        mismatch = int(np.flatnonzero(np.any(np.abs(goals - expected) > 1e-7, axis=1))[0])
        raise ValueError(f"relation goal semantic mismatch at frame {mismatch}")
