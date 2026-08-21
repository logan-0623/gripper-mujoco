from __future__ import annotations

import numpy as np

from interaction_vla.graph_finetune.data import graph_v2_targets
from interaction_vla.lerobot_bridge.teacher_schema import (
    CONFIDENCE,
    PROBABILITY_0,
    PROBABILITY_1,
)
from interaction_vla.representation_study.ontology import (
    ONTOLOGY_SCHEMA_VERSION,
    labels_from_teacher_arrays,
)


def _arrays(frames: int = 3) -> dict[str, np.ndarray]:
    relation = np.zeros((frames, 8, 24), dtype=np.float32)
    relation[:, 0, :3] = (0.1, -0.2, 0.3)
    relation[:, 0, CONFIDENCE] = 1.0
    relation[:, 1, CONFIDENCE] = 1.0
    relation[:, 2, CONFIDENCE] = 1.0
    relation[1:, 0, PROBABILITY_0] = 0.8
    relation[1:, 0, PROBABILITY_1] = 0.9
    entity_pose = np.zeros((frames, 6, 9), dtype=np.float32)
    entity_pose[:, :, 3:9] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    entity_mask = np.ones((frames, 6), dtype=np.bool_)
    entity_visibility = np.ones((frames, 6, 2), dtype=np.float32)
    relation_mask = np.ones((frames, 8), dtype=np.bool_)
    goals = np.zeros((frames, 5), dtype=np.float32)
    goals[:, 0] = 1
    goals[:, 1] = 0
    goals[:, 2] = 4
    goals[:, 4] = 1
    return {
        "annotation.tc_tig.entity_pose": entity_pose,
        "annotation.tc_tig.entity_mask": entity_mask,
        "annotation.tc_tig.entity_visibility": entity_visibility,
        "annotation.tc_tig.relation_values": relation,
        "annotation.tc_tig.relation_mask": relation_mask,
        "annotation.tc_tig.relation_goal": goals,
    }


def test_teacher_ontology_exposes_all_measurement_groups() -> None:
    arrays = _arrays()
    labels = labels_from_teacher_arrays(arrays, frame=1, recovery_state="nominal")

    assert labels["schema_version"] == ONTOLOGY_SCHEMA_VERSION
    assert set(labels["targets"]) == {
        "entity",
        "geometry",
        "contact",
        "stable_grasp",
        "phase",
        "next_relation",
        "recovery_state",
    }
    assert labels["targets"]["contact"] == 1
    assert labels["targets"]["stable_grasp"] == 1
    assert labels["targets"]["next_relation"]["relation_id"] == 0


def test_teacher_ontology_matches_graph_v2_phase_and_geometry() -> None:
    arrays = _arrays()
    graph = graph_v2_targets(arrays)
    labels = labels_from_teacher_arrays(arrays, frame=0, recovery_state="nominal")

    assert labels["targets"]["phase"] == int(graph.phase[0])
    assert np.allclose(
        labels["targets"]["geometry"]["gripper_target"],
        graph.gripper_target_geometry[0],
    )
