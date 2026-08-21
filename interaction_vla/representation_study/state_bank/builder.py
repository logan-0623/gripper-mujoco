from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from interaction_vla.graph_control.tracing import load_trace_episode
from interaction_vla.graph_finetune.data import graph_v2_targets
from interaction_vla.lerobot_bridge.codecs import EndEffectorStateCodec
from interaction_vla.lerobot_bridge.config import load_bridge_config
from interaction_vla.lerobot_bridge.provenance import (
    sha256_file,
    source_fingerprint,
    standard_dataset_fingerprint,
)
from interaction_vla.lerobot_bridge.sidecar import load_teacher_sidecar
from interaction_vla.lerobot_bridge.interaction_phase import PHASE_NAMES
from interaction_vla.representation_study.config import RepresentationStudyConfig
from interaction_vla.representation_study.ontology import (
    labels_from_graph_targets,
    labels_from_trace_record,
    ontology_payload,
)
from interaction_vla.representation_study.schemas.stages import ArtifactBinding

from .io import encode_records, write_bytes_atomic, write_json_atomic
from .schema import (
    CanonicalJson,
    ObservationReference,
    StateBankManifest,
    StateBankRecord,
    StateBankSplit,
)
from .selection import (
    assign_groups_to_partitions,
    classify_trace_strata,
    select_stratified_indices,
)
from .validation import validate_state_bank


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _hash_payload(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parquet_columns(root: Path) -> dict[str, np.ndarray]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"LeRobotDataset parquet data not found: {root}")
    table = pa.concat_tables([pq.read_table(path) for path in paths])
    required = {"observation.state", "episode_index", "frame_index", "index"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError("LeRobotDataset metadata is missing: " + ", ".join(sorted(missing)))
    result = {
        "observation.state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
        "episode_index": np.asarray(table["episode_index"].to_pylist(), dtype=np.int64),
        "frame_index": np.asarray(table["frame_index"].to_pylist(), dtype=np.int64),
        "index": np.asarray(table["index"].to_pylist(), dtype=np.int64),
    }
    rows = len(result["index"])
    if result["observation.state"].shape != (rows, 10):
        raise ValueError("LeRobotDataset robot state must have shape [rows, 10]")
    if not np.isfinite(result["observation.state"]).all():
        raise ValueError("LeRobotDataset robot state must be finite")
    if not np.array_equal(result["index"], np.arange(rows)):
        raise ValueError("LeRobotDataset global indices must be contiguous")
    return result


def _even_indices(values: np.ndarray, *, per_value: int) -> tuple[int, ...]:
    selected: list[int] = []
    for value in sorted(np.unique(values).tolist()):
        candidates = np.flatnonzero(values == value)
        if len(candidates) <= per_value:
            selected.extend(int(index) for index in candidates)
        else:
            offsets = np.rint(np.linspace(0, len(candidates) - 1, per_value)).astype(np.int64)
            selected.extend(int(candidates[index]) for index in offsets)
    return tuple(sorted(set(selected)))


def _expert_records(
    config: RepresentationStudyConfig,
) -> tuple[list[StateBankRecord], dict[str, str], list[dict[str, object]]]:
    root = config.dataset.root
    manifest_path = root / "meta" / "teacher_manifest.json"
    manifest = _json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("teacher manifest must be a non-empty list")
    split_raw = _json(config.dataset.split_manifest)
    if not isinstance(split_raw, Mapping) or not isinstance(split_raw.get("episode_indices"), Mapping):
        raise ValueError("split manifest is missing episode_indices")
    partition_by_episode = {
        int(episode): str(partition)
        for partition, episodes in split_raw["episode_indices"].items()
        for episode in episodes
    }
    columns = _parquet_columns(root)
    result: list[StateBankRecord] = []
    sources: list[dict[str, object]] = []
    for raw in manifest:
        if not isinstance(raw, Mapping):
            raise ValueError("teacher manifest records must be mappings")
        episode = int(raw["episode_index"])
        if episode not in partition_by_episode:
            raise ValueError(f"teacher episode {episode} is absent from split manifest")
        sidecar_path = root / str(raw["path"])
        arrays = load_teacher_sidecar(sidecar_path, expected_sha256=str(raw["sha256"]))
        targets = graph_v2_targets(arrays)
        row_positions = np.flatnonzero(columns["episode_index"] == episode)
        row_positions = row_positions[np.argsort(columns["frame_index"][row_positions])]
        if len(row_positions) != len(targets.phase):
            raise ValueError(f"expert episode {episode} frame alignment changed")
        selected = set(_even_indices(targets.phase, per_value=config.state_bank.expert_per_phase))
        selected.add(len(row_positions) - 1)
        for frame in sorted(selected):
            row = int(row_positions[frame])
            terminal = frame == len(row_positions) - 1
            state_hash = str(arrays["state_hash"][frame])
            state_id = f"expert-{episode:06d}-{frame:04d}-{state_hash[:12]}"
            relation = arrays["annotation.tc_tig.relation_values"][frame]
            result.append(
                StateBankRecord(
                    state_id=state_id,
                    source_episode_id=f"expert/{episode:06d}",
                    split_group_id=f"expert/{episode:06d}",
                    frame_index=frame,
                    task_id=str(raw.get("task_id", 0)),
                    stratum="terminal" if terminal else "nominal",
                    domain="expert_support",
                    phase=PHASE_NAMES[int(targets.phase[frame])],
                    seed=int(raw["seed"]),
                    instruction=str(raw["task"]),
                    observation=ObservationReference(
                        source_uri=root.as_posix(),
                        source_index=int(columns["index"][row]),
                        agent_rgb_key="observation.images.agent",
                        wrist_rgb_key="observation.images.wrist",
                    ),
                    robot_state=tuple(float(value) for value in columns["observation.state"][row]),
                    privileged_state=CanonicalJson.from_value(
                        {
                            "state_hash": state_hash,
                            "entity_pose": arrays["annotation.tc_tig.entity_pose"][frame].tolist(),
                            "entity_mask": arrays["annotation.tc_tig.entity_mask"][frame].tolist(),
                            "relation_values": relation.tolist(),
                            "relation_mask": arrays["annotation.tc_tig.relation_mask"][frame].tolist(),
                        }
                    ),
                    ontology_labels=CanonicalJson.from_value(
                        labels_from_graph_targets(
                            targets,
                            frame=frame,
                            recovery_state="terminal" if terminal else "nominal",
                        )
                    ),
                    provenance=CanonicalJson.from_value(
                        {
                            "collector": "lerobot_teacher_sidecar",
                            "episode_index": episode,
                            "global_row": row,
                            "sidecar": str(raw["path"]),
                            "sidecar_sha256": str(raw["sha256"]),
                        }
                    ),
                )
            )
        sources.append(
            {"uri": sidecar_path.as_posix(), "sha256": str(raw["sha256"])}
        )
    return result, {f"expert/{episode:06d}": partition for episode, partition in partition_by_episode.items()}, sources


def _trace_robot_state(record: Mapping[str, object], previous_gripper: float) -> tuple[float, ...]:
    rotation = EndEffectorStateCodec.quaternion_to_matrix(
        np.asarray(record["end_effector_orientation"], dtype=np.float64)
    )
    state = EndEffectorStateCodec.encode(
        np.asarray(record["end_effector_position"], dtype=np.float64),
        rotation,
        previous_gripper,
    )
    return tuple(float(value) for value in state)


def _trace_records(
    config: RepresentationStudyConfig,
) -> tuple[list[StateBankRecord], dict[str, str], list[dict[str, object]]]:
    task = load_bridge_config(config.dataset.bridge_config).dataset.task
    paths = sorted(config.trace.root.glob(f"traces/seed_*/{config.trace.condition}/*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"no trace episodes found for condition {config.trace.condition}: {config.trace.root}"
        )
    episodes: list[tuple[Path, list[dict[str, object]], tuple[str, ...]]] = []
    group_ids: list[str] = []
    for path in paths:
        rows = load_trace_episode(path)
        strata = classify_trace_strata(rows)
        episodes.append((path, rows, strata))
        group_ids.append(f"trace/{rows[0]['case_id']}")
    group_partition = assign_groups_to_partitions(
        group_ids,
        seed=config.state_bank.selection_seed,
        ratios=config.state_bank.split_ratios,
    )
    result: list[StateBankRecord] = []
    episode_partition: dict[str, str] = {}
    sources: list[dict[str, object]] = []
    for path, rows, strata in episodes:
        first = rows[0]
        case_id = str(first["case_id"])
        group_id = f"trace/{case_id}"
        source_episode = str(first["episode_id"])
        episode_partition[source_episode] = group_partition[group_id]
        selected = select_stratified_indices(
            strata, per_stratum=config.state_bank.policy_per_stratum
        )
        previous_gripper = 1.0
        state_by_step: dict[int, tuple[float, ...]] = {}
        for index, row in enumerate(rows):
            state_by_step[index] = _trace_robot_state(row, previous_gripper)
            previous_gripper = float(row["gripper_command"])
        for frame in selected:
            row = rows[frame]
            stratum = strata[frame]
            recovery_state = {
                "nominal": "nominal",
                "perturbation": "perturbed",
                "recovery": "recovering",
                "terminal": "terminal",
            }[stratum]
            identity = hashlib.sha256(
                f"{source_episode}:{frame}".encode("utf-8")
            ).hexdigest()[:12]
            result.append(
                StateBankRecord(
                    state_id=f"policy-{identity}-{frame:04d}",
                    source_episode_id=source_episode,
                    split_group_id=group_id,
                    frame_index=frame,
                    task_id="0",
                    stratum=stratum,
                    domain="policy_shift",
                    phase=str(row["phase"]),
                    seed=int(row["environment_seed"]),
                    instruction=task,
                    observation=ObservationReference(
                        source_uri=path.as_posix(),
                        source_index=frame,
                        agent_rgb_key="replay.observation.images.agent",
                        wrist_rgb_key="replay.observation.images.wrist",
                    ),
                    robot_state=state_by_step[frame],
                    privileged_state=CanonicalJson.from_value(
                        {
                            "case_id": case_id,
                            "layout": row["layout"],
                            "object_count": row["object_count"],
                            "end_effector_position": row["end_effector_position"],
                            "target_relative_position": row["target_relative_position"],
                            "receptacle_relative_position": row["receptacle_relative_position"],
                            "minimum_distractor_clearance": row["minimum_distractor_clearance"],
                            "target_contact": row["target_contact"],
                            "stable_target_grasp": row["stable_target_grasp"],
                            "termination_reason": row["termination_reason"],
                        }
                    ),
                    ontology_labels=CanonicalJson.from_value(
                        labels_from_trace_record(row, recovery_state=recovery_state)
                    ),
                    provenance=CanonicalJson.from_value(
                        {
                            "collector": "deterministic_trace_replay",
                            "trace": path.as_posix(),
                            "trace_sha256": sha256_file(path),
                            "checkpoint": row["checkpoint"],
                            "policy_seed": row["policy_seed"],
                            "executed_action_prefix_length": frame,
                        }
                    ),
                )
            )
        sources.append({"uri": path.as_posix(), "sha256": sha256_file(path)})
    return result, episode_partition, sources


def _split_records(
    records: Sequence[StateBankRecord], partitions: Mapping[str, str]
) -> StateBankSplit:
    grouped = {name: [] for name in ("train", "validation", "test")}
    for record in records:
        partition = partitions.get(record.source_episode_id)
        if partition not in grouped:
            raise ValueError(f"State Bank record has no valid partition: {record.state_id}")
        grouped[partition].append(record.state_id)
    return StateBankSplit(**{name: tuple(values) for name, values in grouped.items()})


def _publish(
    config: RepresentationStudyConfig,
    records: Sequence[StateBankRecord],
    split: StateBankSplit,
    sources: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    destination = config.state_bank.output_dir
    if destination.exists():
        raise FileExistsError(f"State Bank output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        ontology = ontology_payload()
        write_json_atomic(staging / "ontology.json", ontology)
        provenance = {
            "schema_version": "interaction_state_bank_provenance_v1",
            "study_id": config.study_id,
            "source_fingerprint": source_fingerprint(),
            "dataset_fingerprint": standard_dataset_fingerprint(config.dataset.root),
            "teacher_manifest": {
                "uri": (config.dataset.root / "meta/teacher_manifest.json").as_posix(),
                "sha256": sha256_file(config.dataset.root / "meta/teacher_manifest.json"),
            },
            "split_manifest": {
                "uri": config.dataset.split_manifest.as_posix(),
                "sha256": sha256_file(config.dataset.split_manifest),
            },
            "trace_manifest": {
                "uri": (config.trace.root / "manifest.json").as_posix(),
                "sha256": sha256_file(config.trace.root / "manifest.json"),
            },
            "sources": sorted(
                ({"uri": str(value["uri"]), "sha256": str(value["sha256"])} for value in sources),
                key=lambda value: value["uri"],
            ),
        }
        write_json_atomic(staging / "provenance.json", provenance)
        write_json_atomic(
            staging / "selection_contract.json",
            config.state_bank_selection_payload(),
        )
        records_bytes = encode_records(records)
        write_bytes_atomic(staging / "records.jsonl", records_bytes)
        write_json_atomic(staging / "split.json", split.to_dict())
        validation = validate_state_bank(records, split)
        write_json_atomic(staging / "validation_report.json", validation)
        manifest = StateBankManifest(
            bank_id=f"{config.study_id}_state_bank_v1",
            dataset=ArtifactBinding(
                uri=config.dataset.root.as_posix(),
                sha256=standard_dataset_fingerprint(config.dataset.root),
            ),
            ontology=ArtifactBinding(
                uri=(destination / "ontology.json").as_posix(),
                sha256=sha256_file(staging / "ontology.json"),
            ),
            source=ArtifactBinding(
                uri=(destination / "provenance.json").as_posix(),
                sha256=sha256_file(staging / "provenance.json"),
            ),
            selection_config=ArtifactBinding(
                uri=f"{config.config_path.as_posix()}#state_bank_selection",
                sha256=config.state_bank_selection_sha256(),
            ),
            records_sha256=hashlib.sha256(records_bytes).hexdigest(),
            record_count=len(records),
            state_dim=len(records[0].robot_state),
            split=split,
        )
        write_json_atomic(staging / "manifest.json", manifest.to_dict())
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {**validation, "output_dir": destination.as_posix(), "manifest_sha256": manifest.sha256()}


def collect_state_bank(config: RepresentationStudyConfig) -> dict[str, object]:
    expert, expert_partitions, expert_sources = _expert_records(config)
    policy, policy_partitions, policy_sources = _trace_records(config)
    records = tuple(sorted((*expert, *policy), key=lambda record: record.state_id))
    split = _split_records(records, {**expert_partitions, **policy_partitions})
    return _publish(config, records, split, (*expert_sources, *policy_sources))


def inspect_state_bank(config: RepresentationStudyConfig) -> dict[str, object]:
    from .io import load_manifest, load_records, load_split

    root = config.state_bank.output_dir
    records_path = root / "records.jsonl"
    split_path = root / "split.json"
    manifest_path = root / "manifest.json"
    records = load_records(records_path)
    split = load_split(split_path)
    manifest = load_manifest(manifest_path)
    report = validate_state_bank(records, split)
    differing: list[str] = []
    if manifest.records_sha256 != sha256_file(records_path):
        differing.append("records_sha256")
    if manifest.split != split or manifest.record_count != len(records):
        differing.append("manifest_counts_or_split")
    if manifest.selection_config.sha256 != config.state_bank_selection_sha256():
        differing.append("selection_config")
    if differing:
        raise ValueError("State Bank manifest mismatch: " + ", ".join(differing))
    counts = Counter((record.domain, record.stratum) for record in records)
    return {
        **report,
        "manifest_sha256": manifest.sha256(),
        "domain_strata": {
            f"{domain}/{stratum}": count
            for (domain, stratum), count in sorted(counts.items())
        },
    }
