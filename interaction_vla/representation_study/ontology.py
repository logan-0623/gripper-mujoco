from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np

from interaction_vla.graph_finetune.data import graph_v2_targets
from interaction_vla.graph_finetune.schema import GraphV2Targets, TOKEN_SLICES
from interaction_vla.lerobot_bridge.interaction_phase import PHASE_IDS, PHASE_NAMES
from interaction_vla.lerobot_bridge.teacher_schema import (
    OPERATOR_IDS,
    PREDICATE_IDS,
    RELATION_SLOTS,
)


ONTOLOGY_SCHEMA_VERSION: Final[str] = "interaction_measurement_ontology_v1"
RECOVERY_STATES: Final[tuple[str, ...]] = (
    "nominal",
    "perturbed",
    "recovering",
    "terminal",
)
RECOVERY_STATE_IDS: Final[dict[str, int]] = {
    name: index for index, name in enumerate(RECOVERY_STATES)
}


def ontology_payload() -> dict[str, object]:
    return {
        "schema_version": ONTOLOGY_SCHEMA_VERSION,
        "role": "measurement_language",
        "policy_input": False,
        "groups": {
            "entity": {
                "kind": "multilabel",
                "target": "task_conditioned_entity_presence",
                "width": 6,
            },
            "geometry": {
                "kind": "continuous",
                "target": "gripper_target_and_target_receptacle_geometry",
                "width": 18,
            },
            "contact": {"kind": "binary", "width": 1},
            "stable_grasp": {"kind": "binary", "width": 1},
            "phase": {
                "kind": "categorical",
                "classes": list(PHASE_NAMES),
            },
            "next_relation": {
                "kind": "structured_categorical",
                "relation_classes": list(RELATION_SLOTS),
                "operator_classes": [
                    name for name, _ in sorted(OPERATOR_IDS.items(), key=lambda item: item[1])
                ],
                "predicate_classes": [
                    name for name, _ in sorted(PREDICATE_IDS.items(), key=lambda item: item[1])
                ],
            },
            "recovery_state": {
                "kind": "categorical",
                "classes": list(RECOVERY_STATES),
            },
        },
    }


def _recovery_target(name: str) -> dict[str, object]:
    value = str(name).strip()
    if value not in RECOVERY_STATE_IDS:
        raise ValueError(f"unknown recovery state: {value}")
    return {"id": RECOVERY_STATE_IDS[value], "name": value}


def labels_from_teacher_arrays(
    arrays: Mapping[str, np.ndarray], *, frame: int, recovery_state: str
) -> dict[str, object]:
    targets = graph_v2_targets(arrays)
    return labels_from_graph_targets(
        targets, frame=frame, recovery_state=recovery_state
    )


def labels_from_graph_targets(
    targets: GraphV2Targets, *, frame: int, recovery_state: str
) -> dict[str, object]:
    if not isinstance(targets, GraphV2Targets):
        raise ValueError("targets must be GraphV2Targets")
    index = int(frame)
    if index != frame or not 0 <= index < len(targets.phase):
        raise IndexError("ontology frame is outside the teacher episode")
    contact_probability = float(targets.gripper_target_geometry[index, 5])
    co_motion_probability = float(targets.gripper_target_geometry[index, 6])
    return {
        "schema_version": ONTOLOGY_SCHEMA_VERSION,
        "targets": {
            "entity": targets.entity_mask[index].astype(np.int64).tolist(),
            "geometry": {
                "gripper_target": targets.gripper_target_geometry[index].tolist(),
                "target_receptacle": targets.target_receptacle_geometry[index].tolist(),
            },
            "contact": int(contact_probability >= 0.45),
            "stable_grasp": int(
                contact_probability >= 0.45 and co_motion_probability >= 0.70
            ),
            "phase": int(targets.phase[index]),
            "next_relation": {
                "relation_id": int(targets.goal_relation[index]),
                "operator_id": int(targets.goal_operator[index]),
                "predicate_id": int(targets.goal_predicate[index]),
            },
            "recovery_state": _recovery_target(recovery_state),
        },
    }


def labels_from_trace_record(
    record: Mapping[str, object], *, recovery_state: str
) -> dict[str, object]:
    token = np.asarray(record.get("teacher_token"), dtype=np.float32)
    if token.ndim != 1 or token.shape[0] < TOKEN_SLICES["goal_residual"].stop:
        raise ValueError("trace teacher_token is incompatible with Graph v2")
    if not np.isfinite(token).all():
        raise ValueError("trace teacher_token must be finite")

    def category(name: str) -> int:
        values = token[TOKEN_SLICES[name]]
        if float(np.sum(np.clip(values, 0.0, None))) <= 1.0e-12:
            raise ValueError(f"trace teacher_token has no {name} category")
        return int(np.argmax(values))

    phase_name = str(record.get("phase", ""))
    if phase_name not in PHASE_IDS or category("phase") != PHASE_IDS[phase_name]:
        raise ValueError("trace phase and teacher token disagree")
    entity = token[TOKEN_SLICES["entity_presence"]]
    gripper_geometry = token[TOKEN_SLICES["gripper_target_geometry"]]
    target_geometry = token[TOKEN_SLICES["target_receptacle_geometry"]]
    return {
        "schema_version": ONTOLOGY_SCHEMA_VERSION,
        "targets": {
            "entity": (entity >= 0.5).astype(np.int64).tolist(),
            "geometry": {
                "gripper_target": gripper_geometry.tolist(),
                "target_receptacle": target_geometry.tolist(),
            },
            "contact": int(bool(record.get("target_contact"))),
            "stable_grasp": int(bool(record.get("stable_target_grasp"))),
            "phase": category("phase"),
            "next_relation": {
                "relation_id": category("next_relation"),
                "operator_id": category("relation_operator"),
                "predicate_id": category("predicate"),
            },
            "recovery_state": _recovery_target(recovery_state),
        },
    }
