from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..state_bank.schema import StateBankRecord


@dataclass(frozen=True)
class ProbeTarget:
    name: str
    kind: str
    values: np.ndarray
    output_dim: int
    head_widths: tuple[int, ...] = ()


def probe_targets(records: Sequence[StateBankRecord]) -> dict[str, ProbeTarget]:
    if not records:
        raise ValueError("probe targets require State Bank records")
    payloads = [record.ontology_labels.to_value()["targets"] for record in records]
    entity = np.asarray([value["entity"] for value in payloads], dtype=np.float32)
    geometry = np.asarray(
        [
            [*value["geometry"]["gripper_target"], *value["geometry"]["target_receptacle"]]
            for value in payloads
        ],
        dtype=np.float32,
    )
    next_relation = np.asarray(
        [
            (
                value["next_relation"]["relation_id"],
                value["next_relation"]["operator_id"],
                value["next_relation"]["predicate_id"],
            )
            for value in payloads
        ],
        dtype=np.int64,
    )
    result = {
        "entity": ProbeTarget("entity", "multilabel", entity, entity.shape[1]),
        "geometry": ProbeTarget("geometry", "continuous", geometry, geometry.shape[1]),
        "contact": ProbeTarget(
            "contact",
            "binary",
            np.asarray([value["contact"] for value in payloads], dtype=np.int64),
            2,
        ),
        "stable_grasp": ProbeTarget(
            "stable_grasp",
            "binary",
            np.asarray([value["stable_grasp"] for value in payloads], dtype=np.int64),
            2,
        ),
        "phase": ProbeTarget(
            "phase",
            "categorical",
            np.asarray([value["phase"] for value in payloads], dtype=np.int64),
            6,
        ),
        "next_relation": ProbeTarget(
            "next_relation", "structured", next_relation, 8 + 5 + 7, (8, 5, 7)
        ),
        "recovery_state": ProbeTarget(
            "recovery_state",
            "categorical",
            np.asarray(
                [value["recovery_state"]["id"] for value in payloads], dtype=np.int64
            ),
            4,
        ),
    }
    for target in result.values():
        if len(target.values) != len(records) or not np.isfinite(target.values).all():
            raise ValueError(f"probe target {target.name} is not finite/aligned")
    return result

