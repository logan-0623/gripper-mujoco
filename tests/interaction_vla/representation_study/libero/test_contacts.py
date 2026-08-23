from interaction_vla.representation_study.libero.contacts import (
    SemanticContactMap,
    contact_labels_from_pairs,
)


def test_semantic_contacts_require_actual_geom_pairs_and_preserve_finger_groups() -> None:
    mapping = SemanticContactMap(
        target_geoms=frozenset({"bowl_collision"}),
        goal_geoms=frozenset({"plate_collision"}),
        source_geoms=frozenset({"table_collision"}),
        finger_geoms={
            "left": frozenset({"left_pad"}),
            "right": frozenset({"right_pad"}),
        },
    )
    labels = contact_labels_from_pairs(
        (
            ("bowl_collision", "left_pad"),
            ("right_pad", "bowl_collision"),
            ("bowl_collision", "plate_collision"),
        ),
        mapping,
    )
    assert labels.gripper_target
    assert labels.finger_groups == ("left", "right")
    assert labels.target_goal
    assert not labels.target_source
