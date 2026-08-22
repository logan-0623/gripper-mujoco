from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from interaction_vla.representation_study.probes.training import (
    FORMAL_PRIMARY_TARGETS,
    FORMAL_SECONDARY_TARGETS,
    v2_probe_targets,
)
from interaction_vla.representation_study.rl.timeline import (
    SnapshotMeasurementContext,
    measurement_timeline,
    measure_snapshot,
)


def test_timeline_registers_six_linear_and_two_mlp_measurements() -> None:
    timeline = measurement_timeline()
    assert timeline.linear_steps == (0, 4096, 8192, 12288, 16384, 20480)
    assert timeline.mlp_steps == (0, 20480)


def test_timeline_rejects_incomplete_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "step_004096"
    snapshot.mkdir()
    context = SnapshotMeasurementContext(
        condition="rl_head",
        seed_index=0,
        environment_steps=4096,
        expected_binding="b" * 64,
        output_dir=tmp_path / "measurement",
    )
    with pytest.raises(ValueError, match="COMPLETED"):
        measure_snapshot(snapshot, context)


def test_measure_snapshot_verifies_binding_and_payload_hash(tmp_path: Path) -> None:
    import torch

    from interaction_vla.lerobot_bridge.provenance import sha256_file

    snapshot = tmp_path / "step_004096"
    snapshot.mkdir()
    payload = snapshot / "training_state.pt"
    torch.save({"schema_version": "recovery_rl_formal_act_v2"}, payload)
    (snapshot / "COMPLETED").write_text("complete\n", encoding="utf-8")
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "recovery_rl_snapshot_v1",
                "environment_steps": 4096,
                "binding": "b" * 64,
                "payload_sha256": sha256_file(payload),
            }
        ),
        encoding="utf-8",
    )
    context = SnapshotMeasurementContext(
        condition="rl_head",
        seed_index=0,
        environment_steps=4096,
        expected_binding="b" * 64,
        output_dir=tmp_path / "measurement",
    )
    point = measure_snapshot(snapshot, context)
    assert point.environment_steps == 4096
    assert len(point.snapshot_hash) == 64


def test_v2_probe_targets_register_primary_and_secondary_roles() -> None:
    rows = [
        {
            "labels": {
                "geometry": [float(index)] * 16,
                "phase": index % 6,
                "recovery_state": index % 3,
                "recovery_type": index % 4,
                "next_relation": index % 6,
                "contact": index % 2,
                "stable_grasp": (index + 1) % 2,
            }
        }
        for index in range(12)
    ]
    targets = v2_probe_targets(rows)
    assert set(FORMAL_PRIMARY_TARGETS) <= set(targets)
    assert set(FORMAL_SECONDARY_TARGETS) <= set(targets)
    assert targets["geometry"].values.shape == (12, 16)
    assert targets["recovery_state"].output_dim == 3
