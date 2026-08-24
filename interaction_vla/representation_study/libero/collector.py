from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ..state_bank.io import write_json_atomic
from .alignment import AlignmentRow, align_episode_sources
from .annotation import AnnotationThresholds, PrivilegedFrame, annotate_relocation_episode
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
    source_binding = sha256_bytes(
        json.dumps(
            {
                "catalog_sha256": catalog_binding,
                "alignment_sha256": alignment.semantic_sha256,
                "config_sha256": config_hash,
                "pipeline_sha256": pipeline_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
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
        root / "state_bank", source_binding_sha256=source_binding
    )
    records: list[StateRecord] = []
    replay_rows: list[dict[str, object]] = []
    selected_rows: set[tuple[str, int, str]] = set()
    rows_by_task: dict[tuple[str, int], list[AlignmentRow]] = {}
    for row in alignment.rows:
        rows_by_task.setdefault((row.suite, row.task_id), []).append(row)
    for task_key in sorted(task_lookup):
        task_rows = rows_by_task.get(task_key, [])
        if len(task_rows) < config.state_bank.holdout_episodes_per_task:
            raise ValueError(
                f"task {task_key} has only {len(task_rows)} aligned episodes; "
                f"requires {config.state_bank.holdout_episodes_per_task}"
            )
        ranked = sorted(
            task_rows,
            key=lambda row: hashlib.sha256(
                f"{config.seed}:{row.suite}:{row.task_id}:{row.raw_episode_id}".encode()
            ).hexdigest(),
        )
        count = config.state_bank.holdout_episodes_per_task
        selected_rows.update(
            (row.suite, row.task_id, row.raw_episode_id) for row in ranked[:count]
        )
    thresholds = AnnotationThresholds(
        stable_window_frames=config.annotations.stable_window_frames,
        relative_translation_drift_m=config.annotations.relative_translation_drift_m,
        relative_rotation_drift_deg=config.annotations.relative_rotation_drift_deg,
        minimum_comotion_m=config.annotations.minimum_comotion_m,
        lift_clearance_m=config.annotations.lift_clearance_m,
        approach_surface_distance_m=config.annotations.approach_surface_distance_m,
        hysteresis_m=config.annotations.hysteresis_m,
        grasp_aperture_threshold=config.annotations.grasp_aperture_threshold,
        minimum_finger_groups=config.annotations.minimum_finger_groups,
    )
    selected_alignment = tuple(
        row
        for row in alignment.rows
        if (row.suite, row.task_id, row.raw_episode_id) in selected_rows
    )
    for row in tqdm(
        selected_alignment, desc="LIBERO deterministic replay", unit="episode"
    ):
        task_key = (row.suite, row.task_id)
        episode_key = f"{row.suite}:{row.task_id}:{row.raw_episode_id}"
        cached = shard_writer.load(episode_key)
        if cached is not None:
            cached_records, cached_metadata = cached
            cached_replay = cached_metadata.get("replay")
            if not isinstance(cached_replay, dict):
                raise ValueError(f"cached episode has no replay audit: {episode_key}")
            replay_rows.append(cached_replay)
            records.extend(cached_records)
            continue
        raw_descriptor = raw_lookup[(row.suite, row.task_id, row.raw_episode_id)]
        lerobot = lerobot_lookup[(row.suite, row.task_id, row.lerobot_episode_id)]
        task = task_lookup[task_key]
        raw_path = config.sources.raw_hdf5_root / str(raw_descriptor.relative_path)
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
                state_l2_p95_tolerance=config.replay.state_l2_p95_tolerance,
                state_max_abs_tolerance=config.replay.state_max_abs_tolerance,
            )
            privileged = tuple(
                replace(frame.observation["_privileged_frame"], frame_index=frame.frame_index)
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
                "validated_transitions": len(replay.l2_errors),
                "l2_p95": replay.l2_p95_error,
                "max_abs": replay.max_abs_error,
            }
        replay_rows.append(replay_row)
        if not replay.passed:
            shard_writer.write(episode_key, (), metadata={"replay": replay_row})
            continue
        selected = _selected_frames(len(replay.frames), config.state_bank.states_per_episode)
        episode_records: list[StateRecord] = []
        for frame_index in selected:
            state_id = (
                f"libero:{row.suite}:{row.task_id}:{row.raw_episode_id}:"
                f"{frame_index}:{catalog.dataset_revision[:12]}"
            )
            record = StateRecord(
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
                    dataset_index=lerobot.dataset_from_index + frame_index,
                    global_rgb_key="observation.images.image",
                    wrist_rgb_key="observation.images.image2",
                    robot_state=tuple(
                        float(value)
                        for value in lerobot.descriptor.robot_states[frame_index]
                    ),
                    action=tuple(
                        float(value) for value in lerobot.descriptor.actions[frame_index]
                    ),
                    timestamp=frame_index / catalog.dataset_fps,
                ),
                replay=ReplayReference(
                    hdf5_relative_path=str(raw_descriptor.relative_path),
                    demo_key=row.raw_episode_id,
                    simulator_state_index=frame_index,
                    action_index=frame_index,
                    model_xml_sha256=str(raw_descriptor.model_xml_sha256),
                    initial_state_sha256=sha256_bytes(
                        np.asarray(episode.states[0], dtype="<f8").tobytes()
                    ),
                ),
                labels=labels[frame_index],
                source_revision=catalog.dataset_revision,
                annotator_sha256=ontology_hash,
            )
            episode_records.append(record)
        shard_writer.write(
            episode_key, episode_records, metadata={"replay": replay_row}
        )
        records.extend(episode_records)
    if not replay_rows:
        raise ValueError("no aligned LIBERO episodes were selected for replay")
    accepted = sum(bool(row["passed"]) for row in replay_rows)
    accepted_task_keys = {
        (str(row["suite"]), int(row["task_id"]))
        for row in replay_rows
        if bool(row["passed"])
    }
    l2 = [float(row["l2_p95"]) for row in replay_rows]
    maximum = [float(row["max_abs"]) for row in replay_rows]
    replay_statistics = {
        "episodes": len(replay_rows),
        "accepted": accepted,
        "acceptance_rate": accepted / len(replay_rows),
        "expected_tasks": len(task_lookup),
        "accepted_tasks": len(accepted_task_keys),
        "missing_tasks": [
            {"suite": suite, "task_id": task_id}
            for suite, task_id in sorted(set(task_lookup).difference(accepted_task_keys))
        ],
        "l2_p95": float(np.quantile(l2, 0.95)),
        "max_abs": max(maximum, default=float("inf")),
        "rows": replay_rows,
    }
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
            "selected_episodes": len(replay_rows),
            "accepted_episodes": accepted,
            "cached_episode_shards": len(list((root / "state_bank" / ".episode_shards").glob("*.json"))),
        },
    )
    return audit
