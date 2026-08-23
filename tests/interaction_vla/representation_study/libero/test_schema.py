import math

import pytest

from interaction_vla.representation_study.libero.schema import (
    FACTORS,
    PHASES,
    ContactLabels,
    EntityLabels,
    FactorApplicability,
    GeometryLabels,
    InteractionLabels,
    NextRelation,
    ObservationReference,
    ReplayReference,
    StateRecord,
)


def _record() -> StateRecord:
    return StateRecord(
        state_id="libero:libero_spatial:0:demo_0:3:abcdef123456",
        suite="libero_spatial",
        task_id=0,
        task_name="pick_up_the_black_bowl_and_place_it_on_the_plate",
        language="pick up the black bowl and place it on the plate",
        source_episode_id="demo_0",
        lerobot_episode_index=4,
        frame_index=3,
        simulator_seed=None,
        observation=ObservationReference(
            dataset_index=31,
            global_rgb_key="observation.images.image",
            wrist_rgb_key="observation.images.image2",
            robot_state=(0.0,) * 8,
            action=(0.0,) * 7,
            timestamp=0.3,
        ),
        replay=ReplayReference(
            hdf5_relative_path="libero_spatial/task_0_demo.hdf5",
            demo_key="demo_0",
            simulator_state_index=3,
            action_index=3,
            model_xml_sha256="a" * 64,
            initial_state_sha256="b" * 64,
        ),
        labels=InteractionLabels(
            applicability=FactorApplicability.all_applicable(),
            entity=EntityLabels(
                target="black_bowl",
                goal="plate",
                source="table",
                distractors=("white_bowl",),
            ),
            geometry=GeometryLabels(
                gripper_to_target=(0.0,) * 9,
                target_to_goal=(0.0,) * 9,
                gripper_target_distance=0.04,
                target_goal_distance=0.20,
            ),
            contact=ContactLabels(
                gripper_target=True,
                target_goal=False,
                target_source=True,
                finger_groups=("left", "right"),
            ),
            stable_grasp=False,
            phase="contact",
            next_relation=NextRelation(
                active_goal_index=0,
                subject_role="gripper",
                predicate="stable_grasp",
                object_role="target",
                operator="establish",
            ),
        ),
        source_revision="abcdef1234567890",
        annotator_sha256="c" * 64,
    )


def test_record_round_trip_has_no_recovery_factor() -> None:
    record = _record()
    decoded = StateRecord.from_dict(record.to_dict())
    assert decoded == record
    assert set(decoded.labels.applicability.to_dict()) == set(FACTORS)
    assert "recovery" not in decoded.to_dict()["labels"]
    assert decoded.labels.phase in PHASES


def test_record_rejects_nonfinite_geometry() -> None:
    record = _record().to_dict()
    record["labels"]["geometry"]["gripper_target_distance"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        StateRecord.from_dict(record)


def test_masked_factor_must_not_carry_a_convenient_label() -> None:
    record = _record().to_dict()
    record["labels"]["applicability"]["stable_grasp"] = False
    record["labels"]["stable_grasp"] = False
    with pytest.raises(ValueError, match="masked.*null"):
        StateRecord.from_dict(record)


def test_observation_dimensions_are_explicitly_validated() -> None:
    record = _record().to_dict()
    record["observation"]["robot_state"] = [0.0] * 7
    with pytest.raises(ValueError, match="robot_state.*8"):
        StateRecord.from_dict(record)
