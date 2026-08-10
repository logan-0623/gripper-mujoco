import numpy as np

from interaction_vla.lerobot_bridge.teacher_schema import (
    ENTITY_SLOTS,
    FORBIDDEN_FIELD_FRAGMENTS,
    RELATION_FEATURE_DIM,
    RELATION_SLOTS,
    TeacherFrame,
    teacher_schema_payload,
)


def test_teacher_schema_has_six_entities_and_eight_sparse_relations() -> None:
    assert ENTITY_SLOTS == (
        "gripper",
        "target",
        "receptacle",
        "support",
        "distractor_0",
        "distractor_1",
    )
    assert RELATION_SLOTS == (
        "gripper_to_target",
        "target_to_receptacle",
        "target_to_support",
        "distractor_0_to_gripper",
        "distractor_0_to_target",
        "distractor_1_to_gripper",
        "distractor_1_to_target",
        "gripper_to_receptacle",
    )
    assert RELATION_FEATURE_DIM == 24


def test_teacher_payload_contains_no_privileged_forbidden_name() -> None:
    flattened = str(teacher_schema_payload()).lower()
    assert all(fragment not in flattened for fragment in FORBIDDEN_FIELD_FRAGMENTS)


def test_teacher_frame_validates_fixed_shapes() -> None:
    frame = TeacherFrame.zeros(frame_index=0, timestamp=0.0, state_hash="abc")

    assert frame.entity_pose.shape == (6, 9)
    assert frame.entity_visibility.shape == (6, 2)
    assert frame.relation_values.shape == (8, 24)
    assert frame.entity_mask.dtype == np.bool_
    assert frame.relation_mask.dtype == np.bool_
