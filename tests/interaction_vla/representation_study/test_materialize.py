from __future__ import annotations

import pytest
import torch

from interaction_vla.representation_study.state_bank.materialize import (
    MaterializedObservation,
    collate_observations,
    validate_replayed_position,
)
from interaction_vla.representation_study.state_bank.schema import (
    CanonicalJson,
    ObservationReference,
    StateBankRecord,
)


def _record(index: int) -> StateBankRecord:
    return StateBankRecord(
        state_id=f"state-{index}",
        source_episode_id=f"episode-{index}",
        split_group_id=f"episode-{index}",
        frame_index=0,
        task_id="pick-place",
        stratum="nominal",
        domain="expert_support",
        phase="approach",
        seed=index,
        instruction="pick up the target",
        observation=ObservationReference(
            source_uri="dataset",
            source_index=index,
            agent_rgb_key="observation.images.agent",
            wrist_rgb_key="observation.images.wrist",
        ),
        robot_state=tuple(0.0 for _ in range(10)),
        privileged_state=CanonicalJson.from_value({}),
        ontology_labels=CanonicalJson.from_value({}),
        provenance=CanonicalJson.from_value({}),
    )


def test_replay_position_validation_is_strict() -> None:
    assert validate_replayed_position((0.0, 0.1, 0.2), (0.0, 0.1, 0.2), tolerance=1e-5) == 0.0
    with pytest.raises(ValueError, match="replay drifted"):
        validate_replayed_position((0.0, 0.1, 0.2), (0.0, 0.1, 0.3), tolerance=1e-5)


def test_materialized_observations_collate_policy_inputs() -> None:
    items = [
        MaterializedObservation(
            record=record,
            agent_rgb=torch.zeros(3, 256, 256),
            wrist_rgb=torch.ones(3, 256, 256),
            robot_state=torch.zeros(10),
        )
        for record in (_record(0), _record(1))
    ]
    batch = collate_observations(items)
    assert tuple(batch["observation.images.agent"].shape) == (2, 3, 256, 256)
    assert tuple(batch["observation.state"].shape) == (2, 10)
    assert len(batch["task"]) == 2
