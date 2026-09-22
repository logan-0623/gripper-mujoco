from types import SimpleNamespace

import numpy as np

from interaction_vla.representation_study.libero.source_contrast import (
    donor_diagnostics,
    matched_donors,
)


def test_matched_donors_are_same_task_and_cross_episode():
    records = []
    for task in range(2):
        for episode in range(3):
            records.append(SimpleNamespace(
                task_id=task, source_episode_id=f"e{episode}", frame_index=episode,
                observation=SimpleNamespace(robot_state=tuple(
                    float(task + episode + index) for index in range(8))),
                labels=SimpleNamespace(geometry=SimpleNamespace(
                    gripper_target_distance=float(episode),
                    target_goal_distance=float(3 - episode))),
            ))
    donors = matched_donors(records)
    assert set(donors) == {"visual_geometry", "robot_state"}
    for indices in donors.values():
        assert indices.shape == (len(records),)
        for receiver, donor in enumerate(indices):
            assert records[receiver].task_id == records[donor].task_id
            assert records[receiver].source_episode_id != records[donor].source_episode_id

    diagnostics = donor_diagnostics(records, donors)
    assert all(row["coverage"] == 1.0 for row in diagnostics.values())
    assert all(row["p95_distance"] >= row["mean_distance"] for row in diagnostics.values())
