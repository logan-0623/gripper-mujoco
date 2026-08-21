from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .schema import (
    STATE_BANK_DOMAINS,
    STATE_BANK_PARTITIONS,
    STATE_BANK_STRATA,
    StateBankRecord,
    StateBankSplit,
)


def validate_state_bank(
    records: Sequence[StateBankRecord], split: StateBankSplit
) -> dict[str, object]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("State Bank records must be a sequence")
    typed = tuple(records)
    if not typed or any(not isinstance(record, StateBankRecord) for record in typed):
        raise ValueError("State Bank records must be non-empty StateBankRecord values")
    if not isinstance(split, StateBankSplit):
        raise ValueError("State Bank split must be a StateBankSplit")

    state_ids = [record.state_id for record in typed]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("State Bank state IDs must be unique")
    references = [
        (record.observation.source_uri, record.observation.source_index)
        for record in typed
    ]
    if len(set(references)) != len(references):
        raise ValueError("State Bank observation references must be unique")
    state_dims = {len(record.robot_state) for record in typed}
    if len(state_dims) != 1:
        raise ValueError("State Bank robot_state dimensions must match")

    split_ids = {
        state_id
        for partition in STATE_BANK_PARTITIONS
        for state_id in getattr(split, partition)
    }
    if split_ids != set(state_ids):
        raise ValueError("State Bank split must cover every state exactly once")
    by_id = {record.state_id: record for record in typed}
    partition_episodes = {
        partition: {
            by_id[state_id].source_episode_id
            for state_id in getattr(split, partition)
        }
        for partition in STATE_BANK_PARTITIONS
    }
    partition_groups = {
        partition: {
            by_id[state_id].split_group_id
            for state_id in getattr(split, partition)
        }
        for partition in STATE_BANK_PARTITIONS
    }
    for index, left in enumerate(STATE_BANK_PARTITIONS):
        for right in STATE_BANK_PARTITIONS[index + 1 :]:
            if partition_episodes[left] & partition_episodes[right]:
                raise ValueError(
                    f"source episode leakage between {left} and {right} State Bank splits"
                )
            if partition_groups[left] & partition_groups[right]:
                raise ValueError(
                    f"split group leakage between {left} and {right} State Bank splits"
                )

    stratum_counts = Counter(record.stratum for record in typed)
    missing_primary = [
        name for name in ("nominal", "perturbation", "recovery") if not stratum_counts[name]
    ]
    if missing_primary:
        raise ValueError(
            "State Bank is missing primary strata: " + ", ".join(missing_primary)
        )
    domain_counts = Counter(record.domain for record in typed)
    return {
        "passed": True,
        "schema_version": "interaction_state_bank_validation_v1",
        "record_count": len(typed),
        "state_dim": next(iter(state_dims)),
        "episode_overlap": False,
        "stratum_counts": {name: stratum_counts[name] for name in STATE_BANK_STRATA},
        "domain_counts": {name: domain_counts[name] for name in STATE_BANK_DOMAINS},
        "partition_records": {
            name: len(getattr(split, name)) for name in STATE_BANK_PARTITIONS
        },
        "partition_episodes": {
            name: len(partition_episodes[name]) for name in STATE_BANK_PARTITIONS
        },
        "partition_groups": {
            name: len(partition_groups[name]) for name in STATE_BANK_PARTITIONS
        },
    }
