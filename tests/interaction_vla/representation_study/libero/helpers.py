from dataclasses import replace

from interaction_vla.representation_study.libero.schema import (
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


def make_record(
    *,
    task_id: int,
    episode: int,
    frame: int,
    suite: str = "libero_spatial",
    phase: str = "approach",
    contact: bool = False,
    stable: bool = False,
) -> StateRecord:
    source_episode = f"demo_{episode}"
    return StateRecord(
        state_id=f"libero:{suite}:{task_id}:{source_episode}:{frame}:abcdef123456",
        suite=suite,
        task_id=task_id,
        task_name=f"task_{task_id}",
        language=f"perform task {task_id}",
        source_episode_id=source_episode,
        lerobot_episode_index=episode,
        frame_index=frame,
        simulator_seed=None,
        observation=ObservationReference(
            dataset_index=episode * 100 + frame,
            global_rgb_key="observation.images.image",
            wrist_rgb_key="observation.images.image2",
            robot_state=(0.0,) * 8,
            action=(0.0,) * 7,
            timestamp=frame / 10.0,
        ),
        replay=ReplayReference(
            hdf5_relative_path=f"task_{task_id}.hdf5",
            demo_key=source_episode,
            simulator_state_index=frame,
            action_index=frame,
            model_xml_sha256="a" * 64,
            initial_state_sha256="b" * 64,
        ),
        labels=InteractionLabels(
            applicability=FactorApplicability.all_applicable(),
            entity=EntityLabels(
                target=f"object_{task_id}",
                goal=f"goal_{task_id}",
                source="table",
                distractors=(),
            ),
            geometry=GeometryLabels(
                gripper_to_target=(float(frame), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                target_to_goal=(0.0, float(task_id), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                gripper_target_distance=float(frame + 1) / 100.0,
                target_goal_distance=float(task_id + 1) / 10.0,
            ),
            contact=ContactLabels(contact, False, not stable, ("left", "right") if contact else ()),
            stable_grasp=stable,
            phase=phase,
            next_relation=NextRelation(0, "gripper", "near", "target", "establish"),
        ),
        source_revision="abcdef1234567890",
        annotator_sha256="c" * 64,
    )
