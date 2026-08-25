from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from ..state_bank.io import write_json_atomic
from .alignment import AlignmentRow, align_episode_sources
from .annotation import AnnotationThresholds, annotate_relocation_episode
from .config import LiberoStudyConfig
from .replay import load_raw_replay_episode, replay_episode
from .runtime import LiberoOffscreenSimulator
from .schema import ObservationReference, ReplayReference, StateRecord
from .sources import (
    LeRobotEpisodeSource,
    load_source_catalog,
    sha256_bytes,
    source_catalog_binding,
)
from .state_bank import EpisodeShardWriter, finalize_state_bank, load_state_bank


_COMPATIBLE_REPLAY_SOURCE_BINDINGS = (
    # Exact binding of the interrupted formal run produced by 4a90cfe. Its
    # replay and annotation semantics match this migration; only candidate
    # backfill was missing. No other historical binding is accepted.
    "f8bb416f7a33b69b15eecf7908408340c2b42fef22da0f000cab64de276330ae",
)


@dataclass(frozen=True)
class _ReplaySelection:
    selected: tuple[AlignmentRow, ...]
    rejected: tuple[AlignmentRow, ...]
    attempted: tuple[AlignmentRow, ...]


def _select_passing_rows(
    candidates: Sequence[AlignmentRow],
    *,
    required: int,
    seed: int,
    is_acceptable: Callable[[AlignmentRow], bool],
) -> _ReplaySelection:
    ranked = sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row.suite}:{row.task_id}:{row.raw_episode_id}".encode()
        ).hexdigest(),
    )
    selected: list[AlignmentRow] = []
    rejected: list[AlignmentRow] = []
    attempted: list[AlignmentRow] = []
    for row in ranked:
        attempted.append(row)
        if is_acceptable(row):
            selected.append(row)
            if len(selected) == required:
                break
        else:
            rejected.append(row)
    if len(selected) != required:
        task = (
            f"{ranked[0].suite}/{ranked[0].task_id}" if ranked else "unknown task"
        )
        raise ValueError(
            f"{task} has only {len(selected)} replay-valid episodes; "
            f"requires {required}"
        )
    return _ReplaySelection(tuple(selected), tuple(rejected), tuple(attempted))


def _replay_row_key(row: Mapping[str, object]) -> tuple[str, int, str]:
    return str(row["suite"]), int(row["task_id"]), str(row["episode_id"])


def _build_replay_statistics(
    *,
    selected_rows: Sequence[Mapping[str, object]],
    attempted_rows: Sequence[Mapping[str, object]],
    expected_task_keys: set[tuple[str, int]],
    required_episodes_per_task: int,
) -> dict[str, object]:
    if not selected_rows:
        raise ValueError("no replay-valid LIBERO episodes were selected")
    selected_keys = {_replay_row_key(row) for row in selected_rows}
    if len(selected_keys) != len(selected_rows):
        raise ValueError("selected replay episode identities must be unique")
    if any(not bool(row["passed"]) for row in selected_rows):
        raise ValueError("selected State Bank episodes must pass deterministic replay")
    selected_counts = {
        task_key: sum(
            (str(row["suite"]), int(row["task_id"])) == task_key
            for row in selected_rows
        )
        for task_key in expected_task_keys
    }
    incomplete = {
        task_key: count
        for task_key, count in selected_counts.items()
        if count != required_episodes_per_task
    }
    if incomplete:
        raise ValueError(f"replay selection did not preserve per-task quota: {incomplete}")

    validation_vectors = sorted(
        {str(row.get("validation_vector", "unknown")) for row in attempted_rows}
    )
    replay_modes = sorted(
        {str(row.get("replay_mode", "unknown")) for row in attempted_rows}
    )
    replay_protocols = sorted(
        {str(row.get("replay_protocol", "unknown")) for row in attempted_rows}
    )
    if (
        len(validation_vectors) != 1
        or len(replay_modes) != 1
        or len(replay_protocols) != 1
    ):
        raise ValueError(
            "replay episodes used inconsistent protocols: "
            f"modes={replay_modes}, vectors={validation_vectors}, "
            f"protocols={replay_protocols}"
        )

    selected_l2 = [float(row["l2_p95"]) for row in selected_rows]
    selected_maximum = [float(row["max_abs"]) for row in selected_rows]
    candidate_accepted = sum(bool(row["passed"]) for row in attempted_rows)
    annotated_attempts = [
        {**dict(row), "selected": _replay_row_key(row) in selected_keys}
        for row in attempted_rows
    ]
    return {
        "episodes": len(selected_rows),
        "accepted": len(selected_rows),
        "acceptance_rate": 1.0,
        "expected_tasks": len(expected_task_keys),
        "accepted_tasks": len(expected_task_keys),
        "missing_tasks": [],
        "required_episodes_per_task": required_episodes_per_task,
        "selected_episodes_per_task": [
            {"suite": suite, "task_id": task_id, "episodes": selected_counts[(suite, task_id)]}
            for suite, task_id in sorted(expected_task_keys)
        ],
        "replay_mode": replay_modes[0],
        "validation_vector": validation_vectors[0],
        "replay_protocol": replay_protocols[0],
        "l2_p95": float(np.quantile(selected_l2, 0.95)),
        "max_abs": max(selected_maximum),
        "candidate_attempts": len(attempted_rows),
        "candidate_accepted": candidate_accepted,
        "candidate_rejected": len(attempted_rows) - candidate_accepted,
        "candidate_acceptance_rate": candidate_accepted / len(attempted_rows),
        "rows": annotated_attempts,
    }


def _validate_cached_candidate(
    records: Sequence[StateRecord],
    replay_row: Mapping[str, object],
    *,
    ontology_sha256: str,
    episode_key: str,
) -> None:
    passed = bool(replay_row.get("passed"))
    if passed and not records:
        raise ValueError(f"cached passing episode has no records: {episode_key}")
    if not passed and records:
        raise ValueError(f"cached rejected episode contains records: {episode_key}")
    if any(record.annotator_sha256 != ontology_sha256 for record in records):
        raise ValueError(
            f"cached episode uses a different annotation ontology: {episode_key}"
        )


def _config_hash(config: LiberoStudyConfig) -> str:
    return sha256_bytes(config.source_path.read_bytes())


def _ontology_hash(config: LiberoStudyConfig) -> str:
    package = Path(__file__).parent
    implementation = {
        name: sha256_bytes((package / name).read_bytes())
        for name in (
            "annotation.py",
            "contacts.py",
            "runtime.py",
            "task_semantics.py",
            "task_registry_v1.yaml",
        )
    }
    payload = {
        "schema": "libero_interaction_annotation_v1",
        "formal_suites": ["libero_spatial", "libero_object"],
        "annotations": config.annotations.__dict__,
        "implementation": implementation,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def _pipeline_hash() -> str:
    package = Path(__file__).parent
    payload = {
        name: sha256_bytes((package / name).read_bytes())
        for name in (
            "alignment.py",
            "collector.py",
            "replay.py",
            "schema.py",
            "sources.py",
            "splits.py",
            "state_bank.py",
        )
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def _source_binding_sha256(
    *,
    catalog_sha256: str,
    alignment_sha256: str,
    config_sha256: str,
    pipeline_sha256: str,
) -> str:
    return sha256_bytes(
        json.dumps(
            {
                "catalog_sha256": catalog_sha256,
                "alignment_sha256": alignment_sha256,
                "config_sha256": config_sha256,
                "pipeline_sha256": pipeline_sha256,
            },
            sort_keys=True,
        ).encode("utf-8")
    )


def _selected_frames(length: int, limit: int | None) -> tuple[int, ...]:
    if limit is None or limit >= length:
        return tuple(range(length))
    if limit <= 0:
        raise ValueError("states_per_episode must be positive when set")
    return tuple(sorted(set(int(item) for item in np.linspace(0, length - 1, limit))))


def collect_libero_state_bank(config: LiberoStudyConfig) -> dict[str, object]:
    catalog = load_source_catalog(config)
    catalog_binding = source_catalog_binding(catalog)
    alignment = align_episode_sources(
        catalog.raw_episodes,
        tuple(item.descriptor for item in catalog.lerobot_episodes),
        action_atol=config.replay.action_atol,
        require_all_raw=False,
    )
    config_hash = _config_hash(config)
    pipeline_hash = _pipeline_hash()
    source_binding = _source_binding_sha256(
        catalog_sha256=catalog_binding,
        alignment_sha256=alignment.semantic_sha256,
        config_sha256=config_hash,
        pipeline_sha256=pipeline_hash,
    )
    root = config.output_dir
    ontology_hash = _ontology_hash(config)
    bank_root = root / "state_bank"
    if (bank_root / "manifest.json").is_file():
        _, existing, _, _ = load_state_bank(bank_root)
        expected = {
            "source_binding_sha256": source_binding,
            "ontology_sha256": ontology_hash,
            "config_sha256": config_hash,
            "alignment_sha256": alignment.semantic_sha256,
        }
        differing = [
            key for key, value in expected.items() if existing.get(key) != value
        ]
        if differing:
            raise FileExistsError(
                "State Bank output has a different scientific binding "
                f"({', '.join(differing)}); use a new output root: {bank_root}"
            )
        return json.loads(
            (bank_root / "audit" / "report.json").read_text(encoding="utf-8")
        )
    alignment_path = root / "source_alignment" / "manifest.json"
    if alignment_path.is_file():
        existing_alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        if existing_alignment.get("semantic_sha256") != alignment.semantic_sha256:
            raise FileExistsError(
                "source alignment has a different scientific binding; "
                f"use a new output root: {alignment_path}"
            )
    else:
        write_json_atomic(alignment_path, alignment.to_dict())
    raw_lookup = {
        (row.suite, row.task_id, row.episode_id): row for row in catalog.raw_episodes
    }
    lerobot_lookup: dict[tuple[str, int, str], LeRobotEpisodeSource] = {
        (
            row.descriptor.suite,
            row.descriptor.task_id,
            row.descriptor.episode_id,
        ): row
        for row in catalog.lerobot_episodes
    }
    task_lookup = {(task.suite, task.task_id): task for task in catalog.tasks}
    shard_writer = EpisodeShardWriter(
        root / "state_bank",
        source_binding_sha256=source_binding,
        compatible_source_bindings=_COMPATIBLE_REPLAY_SOURCE_BINDINGS,
    )
    records: list[StateRecord] = []
    selected_replay_rows: list[dict[str, object]] = []
    attempted_replay_rows: list[dict[str, object]] = []
    cached_attempts = 0
    rows_by_task: dict[tuple[str, int], list[AlignmentRow]] = {}
    for row in alignment.rows:
        rows_by_task.setdefault((row.suite, row.task_id), []).append(row)
    thresholds = AnnotationThresholds(
        stable_window_frames=config.annotations.stable_window_frames,
        relative_translation_drift_m=config.annotations.relative_translation_drift_m,
        relative_rotation_drift_deg=config.annotations.relative_rotation_drift_deg,
        minimum_comotion_m=config.annotations.minimum_comotion_m,
        lift_clearance_m=config.annotations.lift_clearance_m,
        approach_surface_distance_m=config.annotations.approach_surface_distance_m,
        hysteresis_m=config.annotations.hysteresis_m,
        minimum_finger_groups=config.annotations.minimum_finger_groups,
    )
    candidate_artifacts: dict[
        tuple[str, int, str], tuple[tuple[StateRecord, ...], dict[str, object]]
    ] = {}
    required_per_task = config.state_bank.holdout_episodes_per_task
    progress = tqdm(
        total=len(task_lookup) * required_per_task,
        desc="LIBERO deterministic replay",
        unit="accepted episode",
    )
    try:
        for task_key in sorted(task_lookup):
            task_rows = rows_by_task.get(task_key, [])
            if len(task_rows) < required_per_task:
                raise ValueError(
                    f"task {task_key} has only {len(task_rows)} aligned episodes; "
                    f"requires {required_per_task}"
                )

            def candidate_passes(row: AlignmentRow) -> bool:
                nonlocal cached_attempts
                episode_identity = (row.suite, row.task_id, row.raw_episode_id)
                episode_key = f"{row.suite}:{row.task_id}:{row.raw_episode_id}"
                cached = shard_writer.load(episode_key)
                if cached is not None:
                    cached_records, cached_metadata = cached
                    cached_replay = cached_metadata.get("replay")
                    if not isinstance(cached_replay, dict):
                        raise ValueError(
                            f"cached episode has no replay audit: {episode_key}"
                        )
                    episode_records = tuple(cached_records)
                    replay_row = dict(cached_replay)
                    _validate_cached_candidate(
                        episode_records,
                        replay_row,
                        ontology_sha256=ontology_hash,
                        episode_key=episode_key,
                    )
                    cached_attempts += 1
                else:
                    raw_descriptor = raw_lookup[episode_identity]
                    lerobot = lerobot_lookup[
                        (row.suite, row.task_id, row.lerobot_episode_id)
                    ]
                    task = task_lookup[task_key]
                    raw_path = config.sources.raw_hdf5_root / str(
                        raw_descriptor.relative_path
                    )
                    episode = load_raw_replay_episode(
                        raw_path,
                        suite=row.suite,
                        task_id=row.task_id,
                        demo_key=row.raw_episode_id,
                    )
                    simulator = LiberoOffscreenSimulator(
                        suite=row.suite,
                        task_id=row.task_id,
                        task_name=task.task_name,
                        language=task.language,
                        bddl_path=task.bddl_path,
                        seed=config.seed,
                        control_freq=config.replay.control_freq,
                    )
                    try:
                        replay = replay_episode(
                            episode,
                            simulator,
                            action_atol=config.replay.action_atol,
                            state_l2_p95_tolerance=(
                                config.replay.state_l2_p95_tolerance
                            ),
                            state_max_abs_tolerance=(
                                config.replay.state_max_abs_tolerance
                            ),
                        )
                        privileged = tuple(
                            replace(
                                frame.observation["_privileged_frame"],
                                frame_index=frame.frame_index,
                            )
                            for frame in replay.frames
                        )
                        labels = annotate_relocation_episode(
                            privileged, simulator.semantics, thresholds
                        )
                    finally:
                        simulator.close()
                    replay_row = {
                        "suite": row.suite,
                        "task_id": row.task_id,
                        "episode_id": row.raw_episode_id,
                        "passed": replay.passed,
                        "replay_mode": replay.replay_mode,
                        "validation_vector": replay.validation_vector,
                        "replay_protocol": replay.replay_protocol,
                        "validated_transitions": len(replay.l2_errors),
                        "l2_p95": replay.l2_p95_error,
                        "max_abs": replay.max_abs_error,
                    }
                    if replay.passed:
                        selected_frames = _selected_frames(
                            len(replay.frames), config.state_bank.states_per_episode
                        )
                        built_records: list[StateRecord] = []
                        for frame_index in selected_frames:
                            state_id = (
                                f"libero:{row.suite}:{row.task_id}:"
                                f"{row.raw_episode_id}:{frame_index}:"
                                f"{catalog.dataset_revision[:12]}"
                            )
                            built_records.append(
                                StateRecord(
                                    state_id=state_id,
                                    suite=row.suite,
                                    task_id=row.task_id,
                                    task_name=task.task_name,
                                    language=lerobot.language,
                                    source_episode_id=row.raw_episode_id,
                                    lerobot_episode_index=lerobot.episode_index,
                                    frame_index=frame_index,
                                    simulator_seed=config.seed,
                                    observation=ObservationReference(
                                        dataset_index=(
                                            lerobot.dataset_from_index + frame_index
                                        ),
                                        global_rgb_key="observation.images.image",
                                        wrist_rgb_key="observation.images.image2",
                                        robot_state=tuple(
                                            float(value)
                                            for value in lerobot.descriptor.robot_states[
                                                frame_index
                                            ]
                                        ),
                                        action=tuple(
                                            float(value)
                                            for value in lerobot.descriptor.actions[
                                                frame_index
                                            ]
                                        ),
                                        timestamp=frame_index / catalog.dataset_fps,
                                    ),
                                    replay=ReplayReference(
                                        hdf5_relative_path=str(
                                            raw_descriptor.relative_path
                                        ),
                                        demo_key=row.raw_episode_id,
                                        simulator_state_index=frame_index,
                                        action_index=frame_index,
                                        model_xml_sha256=str(
                                            raw_descriptor.model_xml_sha256
                                        ),
                                        initial_state_sha256=sha256_bytes(
                                            np.asarray(
                                                episode.states[0], dtype="<f8"
                                            ).tobytes()
                                        ),
                                    ),
                                    labels=labels[frame_index],
                                    source_revision=catalog.dataset_revision,
                                    annotator_sha256=ontology_hash,
                                )
                            )
                        episode_records = tuple(built_records)
                    else:
                        episode_records = ()
                    shard_writer.write(
                        episode_key,
                        episode_records,
                        metadata={"replay": replay_row},
                    )
                attempted_replay_rows.append(replay_row)
                candidate_artifacts[episode_identity] = (
                    episode_records,
                    replay_row,
                )
                if bool(replay_row["passed"]):
                    progress.update()
                else:
                    progress.set_postfix(
                        rejected=sum(
                            not bool(item["passed"])
                            for item in attempted_replay_rows
                        )
                    )
                return bool(replay_row["passed"])

            selection = _select_passing_rows(
                task_rows,
                required=required_per_task,
                seed=config.seed,
                is_acceptable=candidate_passes,
            )
            for row in selection.selected:
                episode_identity = (row.suite, row.task_id, row.raw_episode_id)
                episode_records, replay_row = candidate_artifacts[episode_identity]
                records.extend(episode_records)
                selected_replay_rows.append(replay_row)
    finally:
        progress.close()

    replay_statistics = _build_replay_statistics(
        selected_rows=selected_replay_rows,
        attempted_rows=attempted_replay_rows,
        expected_task_keys=set(task_lookup),
        required_episodes_per_task=required_per_task,
    )
    write_json_atomic(root / "replay" / "report.json", replay_statistics)
    audit = finalize_state_bank(
        records,
        output_dir=root / "state_bank",
        source_binding_sha256=source_binding,
        ontology_sha256=ontology_hash,
        config_sha256=config_hash,
        alignment_sha256=alignment.semantic_sha256,
        split_seed=config.splits.seed,
        task_ratios=config.splits.task_ratios,
        episode_ratios=config.splits.episode_ratios,
        replay_statistics=replay_statistics,
        minimum_acceptance_rate=config.replay.minimum_acceptance_rate,
        l2_p95_tolerance=config.replay.state_l2_p95_tolerance,
        max_abs_tolerance=config.replay.state_max_abs_tolerance,
    )
    write_json_atomic(
        root / "state_bank" / "collection_report.json",
        {
            "schema_version": "libero_state_bank_collection_v1",
            "passed": bool(audit.get("passed")),
            "source_binding_sha256": source_binding,
            "source_catalog_sha256": catalog_binding,
            "alignment_sha256": alignment.semantic_sha256,
            "alignment_coverage": alignment.to_dict()["coverage"],
            "config_sha256": config_hash,
            "pipeline_sha256": pipeline_hash,
            "ontology_sha256": ontology_hash,
            "selected_episodes": len(selected_replay_rows),
            "accepted_episodes": len(selected_replay_rows),
            "candidate_attempts": len(attempted_replay_rows),
            "candidate_rejections": (
                len(attempted_replay_rows) - len(selected_replay_rows)
            ),
            "cached_attempts": cached_attempts,
            "cached_episode_shards": len(
                list((root / "state_bank" / ".episode_shards").glob("*.json"))
            ),
        },
    )
    return audit
