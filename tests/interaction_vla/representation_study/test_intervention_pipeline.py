from __future__ import annotations

import numpy as np
import json
from types import SimpleNamespace

from interaction_vla.representation_study.interventions import (
    action_change,
    intervention_probe_binding,
    paired_intervention_batches,
)
from interaction_vla.representation_study.state_bank.schema import (
    CanonicalJson,
    ObservationReference,
    StateBankRecord,
)


def _record(index: int, phase: str = "approach") -> StateBankRecord:
    return StateBankRecord(
        state_id=f"state-{index}",
        source_episode_id=f"episode-{index}",
        split_group_id=f"episode-{index}",
        frame_index=0,
        task_id="task",
        stratum="nominal",
        domain="expert_support",
        phase=phase,
        seed=index,
        instruction="pick target",
        observation=ObservationReference("dataset", index, "agent", "wrist"),
        robot_state=tuple(0.0 for _ in range(10)),
        privileged_state=CanonicalJson.from_value({}),
        ontology_labels=CanonicalJson.from_value({}),
        provenance=CanonicalJson.from_value({}),
    )


def test_intervention_batches_preserve_matching_axes_and_no_singletons() -> None:
    batches = paired_intervention_batches(
        [_record(index) for index in range(7)], batch_size=3, max_states=7
    )
    assert sum(len(batch) for batch in batches) == 7
    assert all(len(batch) >= 2 for batch in batches)


def test_action_change_separates_translation_rotation_and_gripper() -> None:
    baseline = np.zeros((8, 7), dtype=np.float32)
    changed = baseline.copy()
    changed[0, 0] = 1.0
    changed[0, 3] = 2.0
    changed[0, 6] = 0.5
    metrics = action_change(baseline, changed)
    assert metrics["first_translation_l2"] == 1.0
    assert metrics["first_rotation_l2"] == 2.0
    assert metrics["first_gripper_absolute_change"] == 0.5


def test_intervention_binding_requires_the_matching_linear_probe(tmp_path) -> None:
    path = tmp_path / "act" / "sft" / "linear" / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "schema_version": "interaction_frozen_probe_v2",
                "backend": "act",
                "stage": "sft",
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(probes=SimpleNamespace(output_dir=tmp_path))

    binding = intervention_probe_binding(config, backend="act", stage="sft")

    assert binding["uri"] == path.as_posix()
    assert len(binding["sha256"]) == 64
