from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from tqdm.auto import tqdm

from .config import load_config
from .data import split_episode_seeds
from .env import LayoutMode
from .graph.builder import SceneGraphBuilder
from .physics_env import FrankaContactEnv
from .physics_expert import PhysicsScriptedExpert
from .physics_recovery import (
    PhysicsRecoveryKind,
    PhysicsRecoverySpec,
    make_physics_recovery_spec,
    recovery_translation_direction,
    recovery_trigger_ready,
)
from .physics_recording import MultiViewFrame, MultiViewRecorder
from .physics_provenance import (
    config_file_hash,
    controller_source_hash,
    learned_rollout_source_hash,
    scene_asset_hash,
)
from .source_split import (
    deterministic_source_split,
    save_source_split,
    select_training_recovery_sources,
    validate_derived_sources,
)


@dataclass(frozen=True)
class PhysicsEpisode:
    seed: int
    object_count: int
    target_name: str
    reason: str
    trajectory_source: str
    metadata: dict[str, object]
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    node_mask: np.ndarray
    edge_mask: np.ndarray
    proprioception: np.ndarray
    actions: np.ndarray
    phases: np.ndarray
    contact_state: np.ndarray
    contact_force: np.ndarray
    relative_pose: np.ndarray
    stable_grasp: np.ndarray
    rgbd_frames: tuple[MultiViewFrame, ...] = ()


class PhysicsRecoveryRejected(RuntimeError):
    def __init__(self, reason: str) -> None:
        if not reason:
            raise ValueError("recovery rejection reason must not be empty")
        self.reason = str(reason)
        super().__init__(self.reason)


@dataclass(frozen=True)
class PreparedRecoveryStart:
    snapshot: object
    source_seed: int
    source_split: str
    variant_id: int
    kind: str
    interaction_baseline: Mapping[str, int | bool | str | None]


def recovery_quality_summary(
    attempted: Mapping[PhysicsRecoveryKind, int],
    accepted: Mapping[PhysicsRecoveryKind, int],
    *,
    minimum_rate: float,
    expected_kinds: Iterable[PhysicsRecoveryKind],
) -> dict[str, dict[str, object]]:
    if not np.isfinite(minimum_rate) or not 0.0 <= minimum_rate <= 1.0:
        raise ValueError("minimum recovery acceptance rate must be within [0, 1]")
    summary: dict[str, dict[str, object]] = {}
    kinds = tuple(dict.fromkeys(PhysicsRecoveryKind(kind) for kind in expected_kinds))
    if not kinds:
        raise ValueError("expected recovery kinds must not be empty")
    for kind in kinds:
        attempts = int(attempted.get(kind, 0))
        successes = int(accepted.get(kind, 0))
        rate = successes / attempts if attempts else 0.0
        summary[kind.value] = {
            "attempted": attempts,
            "accepted": successes,
            "acceptance_rate": float(rate),
            "passed": successes > 0 and rate >= minimum_rate,
        }
    return summary


def require_recovery_quality(
    summary: Mapping[str, Mapping[str, object]],
) -> None:
    failed = [
        (
            f"{kind}={metrics['accepted']}/{metrics['attempted']} "
            f"({float(metrics['acceptance_rate']):.1%})"
        )
        for kind, metrics in summary.items()
        if not bool(metrics["passed"])
    ]
    if failed:
        raise RuntimeError(
            "recovery quality gate failed: " + ", ".join(failed)
        )


def _target_goal_distance(snapshot) -> float:
    return float(
        np.linalg.norm(
            snapshot.target_object.position[:2] - snapshot.receptacle.position[:2]
        )
    )


def _expert_action_with_phase(
    expert: PhysicsScriptedExpert,
    snapshot,
    contacts,
    grasp,
) -> tuple[str, np.ndarray]:
    action_phase = expert.phase.value
    action = np.asarray(
        expert.act(snapshot, contacts, grasp),
        dtype=np.float32,
    )
    return action_phase, action


def _advance_recovery_intervention(
    env: FrankaContactEnv,
    action: np.ndarray,
    *,
    substeps: int,
    context: str,
):
    result = env.advance_intervention(action, substeps=substeps)
    if result.physics_failure is not None:
        raise PhysicsRecoveryRejected(
            f"physics_failure_during_{context}:{result.physics_failure}"
        )
    if (
        result.controller_diagnostics is not None
        and result.controller_diagnostics.ik_limited
    ):
        raise PhysicsRecoveryRejected(f"ik_limited_during_{context}")
    return result.snapshot


def _apply_recovery_intervention(
    env: FrankaContactEnv,
    snapshot,
    spec: PhysicsRecoverySpec,
):
    target_name = snapshot.target_object.name
    finger_before = float(np.mean(env.proprioception()[13:15]))
    if spec.kind is PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE:
        action = np.asarray((0, 0, -1, 0, 0, 0, 0), dtype=np.float32)
        for _ in range(spec.close_descent_steps):
            snapshot = _advance_recovery_intervention(
                env,
                action,
                substeps=env.physics.substeps,
                context="terminal_intervention",
            )
            if target_name not in env.contact_diagnostics.object_receptacle:
                raise PhysicsRecoveryRejected(
                    "target_support_lost_during_terminal_intervention"
                )
            if env.grasp_state.stable_object is not None:
                raise PhysicsRecoveryRejected(
                    "stable_grasp_during_terminal_intervention"
                )
        finger_after = float(np.mean(env.proprioception()[13:15]))
        if finger_after >= finger_before - 1e-6:
            raise PhysicsRecoveryRejected("fingers_did_not_close")
        return snapshot
    if spec.kind is PhysicsRecoveryKind.PREMATURE_OPEN:
        action = np.zeros(7, dtype=np.float32)
        action[6] = 1.0
        snapshot = _advance_recovery_intervention(
            env,
            action,
            substeps=spec.open_substeps,
            context="intervention",
        )
    else:
        direction = recovery_translation_direction(
            spec,
            snapshot.target_object.position[:2],
            snapshot.receptacle.position[:2],
        )
        action = np.zeros(7, dtype=np.float32)
        action[:2] = direction
        for _ in range(spec.translation_steps):
            snapshot = _advance_recovery_intervention(
                env,
                action,
                substeps=env.physics.substeps,
                context="intervention",
            )
            if env.grasp_state.bilateral_object != target_name:
                raise PhysicsRecoveryRejected("target_contact_lost_during_intervention")

    if env.grasp_state.bilateral_object != target_name:
        raise PhysicsRecoveryRejected("target_not_bilateral_after_intervention")
    if (
        target_name in env.contact_diagnostics.object_table
        or target_name in env.contact_diagnostics.object_receptacle
    ):
        raise PhysicsRecoveryRejected("target_supported_after_intervention")
    if spec.kind is PhysicsRecoveryKind.PREMATURE_OPEN:
        finger_after = float(np.mean(env.proprioception()[13:15]))
        if finger_after <= finger_before + 1e-6:
            raise PhysicsRecoveryRejected("fingers_did_not_open")
    return snapshot


def _interaction_baseline(env: FrankaContactEnv) -> dict[str, int | bool | str | None]:
    grasp = env.grasp_state
    return {
        "tracker_substep": int(grasp.tracker_substep),
        "ever_bilateral_contact": bool(grasp.ever_bilateral_contact),
        "first_bilateral_object": grasp.first_bilateral_object,
        "ever_bilateral_target_contact": bool(
            grasp.ever_bilateral_target_contact
        ),
        "ever_bilateral_wrong_object": bool(grasp.ever_bilateral_wrong_object),
        "total_bilateral_target_substeps": int(
            grasp.total_bilateral_target_substeps
        ),
        "total_bilateral_wrong_substeps": int(
            grasp.total_bilateral_wrong_substeps
        ),
        "first_bilateral_target_substep": grasp.first_bilateral_target_substep,
        "ever_stable_target": bool(grasp.ever_stable_target),
        "ever_stable_wrong_object": bool(grasp.ever_stable_wrong_object),
        "first_stable_target_substep": grasp.first_stable_target_substep,
        "total_stable_target_substeps": int(grasp.total_stable_target_substeps),
        "longest_stable_target_run": int(grasp.longest_stable_target_run),
        "dropped_target": bool(grasp.dropped_target),
    }


def prepare_physics_recovery_start(
    env: FrankaContactEnv,
    expert: PhysicsScriptedExpert,
    *,
    spec: PhysicsRecoverySpec,
    object_count: int,
    source_split: str,
    layout_mode: LayoutMode | str = LayoutMode.NORMAL,
) -> PreparedRecoveryStart:
    """Recreate one recovery handoff without modifying object state directly.

    The expert owns only the prefix needed to reach the deterministic trigger.
    The returned snapshot is immediately after the physical intervention; callers
    own every action after this function returns.
    """

    if not source_split:
        raise ValueError("source_split must not be empty")
    snapshot = env.reset(
        seed=spec.source_seed,
        object_count=object_count,
        layout_mode=layout_mode,
    )
    expert.reset(seed=spec.source_seed)
    target_name = snapshot.target_object.name
    while True:
        ready = recovery_trigger_ready(
            spec,
            phase=expert.phase.value,
            stable_target=env.grasp_state.stable_object == target_name,
            distance=_target_goal_distance(snapshot),
            supported_target=(
                target_name in env.contact_diagnostics.object_receptacle
            ),
        )
        if ready:
            snapshot = _apply_recovery_intervention(env, snapshot, spec)
            return PreparedRecoveryStart(
                snapshot=snapshot,
                source_seed=int(spec.source_seed),
                source_split=str(source_split),
                variant_id=int(spec.variant_id),
                kind=spec.kind.value,
                interaction_baseline=_interaction_baseline(env),
            )

        prefix_action = expert.act(
            snapshot,
            env.contact_diagnostics,
            env.grasp_state,
        )
        transition = env.step(prefix_action)
        snapshot = transition.snapshot
        if transition.done:
            raise PhysicsRecoveryRejected(
                f"trigger_not_reached:{transition.reason.value}"
            )


def expected_gate_hashes(config_path: str | Path) -> dict[str, str]:
    return {
        "controller_hash": controller_source_hash(),
        "rollout_integrity_hash": learned_rollout_source_hash(),
        "scene_hash": scene_asset_hash(),
        "config_hash": config_file_hash(config_path),
    }


def require_expert_gate(config_path: str | Path, gate_path: str | Path) -> str:
    path = Path(gate_path)
    if not path.exists():
        raise FileNotFoundError(f"expert gate not found: {path}")
    raw = path.read_bytes()
    report = json.loads(raw)
    if not report.get("passed", False):
        raise ValueError(f"expert gate did not pass: {path}")
    expected = expected_gate_hashes(config_path)
    differing = [
        name for name, value in expected.items() if report.get(name) != value
    ]
    if differing:
        raise ValueError(
            "expert gate is stale for the current config, scene, or controller; "
            f"differing keys: {', '.join(differing)}; rerun validation"
        )
    return hashlib.sha256(raw).hexdigest()


def expert_gate_provenance(
    config_path: str | Path, gate_path: str | Path
) -> dict[str, str]:
    return {
        "expert_gate_hash": require_expert_gate(config_path, gate_path),
        **expected_gate_hashes(config_path),
    }


def require_episode_gate_provenance(
    episode_paths: tuple[Path, ...] | list[Path] | tuple[str | Path, ...],
    expected_gate_hash: str,
) -> None:
    if not expected_gate_hash:
        raise ValueError("expected expert gate hash must not be empty")
    for episode_path in episode_paths:
        path = Path(episode_path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
        if (
            metadata.get("backend") != "franka_contact"
            or metadata.get("feature_schema") != "physics_v2"
            or metadata.get("expert_gate_hash") != expected_gate_hash
        ):
            raise ValueError(
                f"episode gate provenance does not match the current expert gate: {path}"
            )


def collect_physics_episode(
    env: FrankaContactEnv,
    expert: PhysicsScriptedExpert,
    *,
    seed: int,
    object_count: int,
    trajectory_source: str = "scripted",
    layout_mode: LayoutMode | str = LayoutMode.NORMAL,
    recovery: PhysicsRecoverySpec | None = None,
    source_split: str = "unspecified",
    expert_gate_hash: str = "",
    recorder: MultiViewRecorder | None = None,
) -> PhysicsEpisode:
    if trajectory_source not in {"scripted", "teleop"}:
        raise ValueError("trajectory_source must be scripted or teleop")
    if trajectory_source != "scripted":
        raise ValueError("teleop collection is driven by the visualization event loop")
    builder = SceneGraphBuilder(max_objects=env.max_objects, feature_schema="physics_v2")
    if recovery is not None and recovery.source_seed != int(seed):
        raise ValueError("recovery source_seed must match the episode seed")
    if recovery is None:
        snapshot = env.reset(
            seed=seed,
            object_count=object_count,
            layout_mode=layout_mode,
        )
        expert.reset(seed=seed)
    else:
        prepared = prepare_physics_recovery_start(
            env,
            expert,
            spec=recovery,
            object_count=object_count,
            source_split=source_split,
            layout_mode=layout_mode,
        )
        snapshot = prepared.snapshot
    target_name = snapshot.target_object.name
    node_features: list[np.ndarray] = []
    edge_features: list[np.ndarray] = []
    node_masks: list[np.ndarray] = []
    edge_masks: list[np.ndarray] = []
    proprioception: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phases: list[str] = []
    contact_states: list[np.ndarray] = []
    contact_forces: list[np.ndarray] = []
    relative_poses: list[np.ndarray] = []
    stable_grasps: list[np.ndarray] = []
    rgbd_frames: list[MultiViewFrame] = []
    edge_index: np.ndarray | None = None

    while True:
        graph = builder.build(snapshot)
        action_phase, action = _expert_action_with_phase(
            expert,
            snapshot,
            env.contact_diagnostics,
            env.grasp_state,
        )
        node_features.append(graph.node_features.copy())
        edge_features.append(graph.edge_features.copy())
        node_masks.append(graph.node_mask.copy())
        edge_masks.append(graph.edge_mask.copy())
        proprioception.append(env.proprioception().copy())
        actions.append(action.copy())
        phases.append(action_phase)
        edge_index = graph.edge_index.copy()

        contact_state = np.zeros((object_count, 2), dtype=np.bool_)
        contact_force = np.zeros((object_count, 2), dtype=np.float32)
        relative_pose = np.zeros((object_count, 6), dtype=np.float32)
        stable = np.zeros(object_count, dtype=np.bool_)
        interaction_by_key = {
            signal.key: signal for signal in snapshot.interactions
        }
        for index in range(object_count):
            name = f"object_{index}"
            contact_state[index] = (
                name in env.contact_diagnostics.left_objects,
                name in env.contact_diagnostics.right_objects,
            )
            signal = interaction_by_key.get(frozenset(("gripper", name)))
            if signal is not None:
                contact_force[index] = (
                    signal.normal_force,
                    signal.tangential_force,
                )
            edge_id = np.flatnonzero(
                (graph.edge_index[0] == 0) & (graph.edge_index[1] == index + 1)
            )
            if len(edge_id) != 1:
                raise RuntimeError("physical graph is missing a gripper-object edge")
            relative_pose[index] = graph.edge_features[int(edge_id[0]), :6]
            stable[index] = env.grasp_state.stable_object == name
        contact_states.append(contact_state)
        contact_forces.append(contact_force)
        relative_poses.append(relative_pose)
        stable_grasps.append(stable)
        if recorder is not None:
            rgbd_frames.append(recorder.capture(env))

        transition = env.step(action)
        snapshot = transition.snapshot
        if transition.done:
            break

    assert edge_index is not None
    metadata: dict[str, object] = {
        "seed": int(seed),
        "object_count": int(object_count),
        "target_name": target_name,
        "reason": transition.reason.value,
        "trajectory_kind": "recovery" if recovery is not None else "base",
        "trajectory_source": trajectory_source,
        "feature_schema": "physics_v2",
        "backend": "franka_contact",
        "action_dim": 7,
        "proprioception_dim": 23,
        "scene_version": "franka_contact_v1",
        "controller_version": "cartesian_7d_dls_v1",
        "physics": env.physics_metadata(),
        "expert_gate_hash": expert_gate_hash,
        "recovery": None if recovery is None else recovery.metadata(),
        "source_seed": None if recovery is None else int(recovery.source_seed),
        "source_split": None if recovery is None else source_split,
        "variant_id": None if recovery is None else int(recovery.variant_id),
        "perturbation_kind": None if recovery is None else recovery.kind.value,
        "injection_phase": None if recovery is None else recovery.trigger_phase,
        "rgbd_sidecar": None,
    }
    return PhysicsEpisode(
        seed=int(seed),
        object_count=int(object_count),
        target_name=target_name,
        reason=transition.reason.value,
        trajectory_source=trajectory_source,
        metadata=metadata,
        node_features=np.stack(node_features),
        edge_index=edge_index,
        edge_features=np.stack(edge_features),
        node_mask=np.stack(node_masks),
        edge_mask=np.stack(edge_masks),
        proprioception=np.stack(proprioception),
        actions=np.stack(actions),
        phases=np.asarray(phases),
        contact_state=np.stack(contact_states),
        contact_force=np.stack(contact_forces),
        relative_pose=np.stack(relative_poses),
        stable_grasp=np.stack(stable_grasps),
        rgbd_frames=tuple(rgbd_frames),
    )


def save_physics_episode(
    episode: PhysicsEpisode,
    path: str | Path,
    *,
    rgbd_path: str | Path | None = None,
) -> Path:
    if len(episode.actions) < 1:
        raise ValueError("cannot save an empty physics episode")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(episode.metadata)
    if episode.rgbd_frames:
        if rgbd_path is None:
            raise ValueError("rgbd_path is required when the episode contains RGB-D frames")
        sidecar = MultiViewRecorder.save_episode(episode.rgbd_frames, rgbd_path)
        metadata["rgbd_sidecar"] = sidecar.name
    elif rgbd_path is not None:
        raise ValueError("rgbd_path was provided but the episode has no RGB-D frames")
    np.savez_compressed(
        destination,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        node_features=episode.node_features,
        edge_index=episode.edge_index,
        edge_features=episode.edge_features,
        node_mask=episode.node_mask,
        edge_mask=episode.edge_mask,
        proprioception=episode.proprioception,
        actions=episode.actions,
        phases=episode.phases,
        contact_state=episode.contact_state,
        contact_force=episode.contact_force,
        relative_pose=episode.relative_pose,
        stable_grasp=episode.stable_grasp,
    )
    return destination


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _collect_recovery_group(
    *,
    config,
    records_by_seed: Mapping[int, Mapping[str, object]],
    source_seeds: tuple[int, ...],
    source_split_by_seed: Mapping[int, str],
    gate_hash: str,
    manifest_path: Path,
    rejection_path: Path,
    quality_path: Path,
    filename_prefix: str,
    progress_description: str,
    show_progress: bool,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    recovery_records: list[dict[str, object]] = []
    recovery_rejections: list[dict[str, object]] = []
    _write_json(manifest_path, recovery_records)
    _write_json(rejection_path, recovery_rejections)
    attempted_by_kind: Counter[PhysicsRecoveryKind] = Counter()
    accepted_by_kind: Counter[PhysicsRecoveryKind] = Counter()
    progress = (
        tqdm(
            total=len(source_seeds) * config.recovery.variants_per_episode,
            desc=progress_description,
            unit="attempt",
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )
    try:
        for source_index, source_seed in enumerate(source_seeds):
            record = records_by_seed[source_seed]
            source_split = source_split_by_seed[source_seed]
            for local_variant_id in range(config.recovery.variants_per_episode):
                variant_id = (
                    source_index * config.recovery.variants_per_episode
                    + local_variant_id
                )
                spec = make_physics_recovery_spec(
                    source_seed,
                    variant_id,
                    kind_index=local_variant_id,
                )
                attempted_by_kind[spec.kind] += 1
                env = FrankaContactEnv(
                    max_objects=config.max_objects,
                    max_steps=config.environment.max_steps,
                    min_object_distance=config.environment.min_object_distance,
                    workspace_low=config.environment.workspace_low,
                    workspace_high=config.environment.workspace_high,
                    crowded_anchor_min_distance=(
                        config.environment.crowded_anchor_min_distance
                    ),
                    crowded_anchor_max_distance=(
                        config.environment.crowded_anchor_max_distance
                    ),
                    physics=config.physics,
                )
                recorder = (
                    MultiViewRecorder(
                        env.model,
                        width=config.recording.width,
                        height=config.recording.height,
                    )
                    if config.recording.enabled
                    else None
                )
                recovery_reason = ""
                try:
                    try:
                        episode = collect_physics_episode(
                            env,
                            PhysicsScriptedExpert(config.physics),
                            seed=source_seed,
                            object_count=int(record["object_count"]),
                            recovery=spec,
                            source_split=source_split,
                            expert_gate_hash=gate_hash,
                            recorder=recorder,
                        )
                    except PhysicsRecoveryRejected as error:
                        recovery_reason = error.reason
                        recovery_rejections.append(
                            {
                                "source_seed": source_seed,
                                "source_split": source_split,
                                "variant_id": variant_id,
                                "kind": spec.kind.value,
                                "object_count": int(record["object_count"]),
                                "frames": 0,
                                "reason": error.reason,
                            }
                        )
                        _write_json(rejection_path, recovery_rejections)
                    except Exception as error:
                        recovery_reason = f"exception:{type(error).__name__}"
                        raise
                    else:
                        recovery_reason = episode.reason
                        recovery_record: dict[str, object] = {
                            "source_seed": source_seed,
                            "source_split": source_split,
                            "variant_id": variant_id,
                            "kind": spec.kind.value,
                            "object_count": int(record["object_count"]),
                            "frames": len(episode.actions),
                            "reason": episode.reason,
                        }
                        if episode.reason == "success":
                            episode.metadata["source_split"] = source_split
                            path = save_physics_episode(
                                episode,
                                manifest_path.parent
                                / (
                                    f"{filename_prefix}_{source_seed:010d}_v"
                                    f"{variant_id:03d}.npz"
                                ),
                                rgbd_path=(
                                    manifest_path.parent
                                    / (
                                        f"{filename_prefix}_{source_seed:010d}_v"
                                        f"{variant_id:03d}_rgbd.npz"
                                    )
                                    if config.recording.enabled
                                    else None
                                ),
                            )
                            recovery_record["path"] = path.name
                            recovery_records.append(recovery_record)
                            accepted_by_kind[spec.kind] += 1
                            _write_json(manifest_path, recovery_records)
                        else:
                            recovery_rejections.append(recovery_record)
                            _write_json(rejection_path, recovery_rejections)
                finally:
                    if recorder is not None:
                        recorder.close()
                    if progress is not None:
                        progress.set_postfix(
                            kind=spec.kind.value,
                            source_seed=source_seed,
                            accepted=len(recovery_records),
                            rejected=len(recovery_rejections),
                            reason=recovery_reason,
                        )
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    expected_kinds = tuple(
        dict.fromkeys(
            make_physics_recovery_spec(
                config.seed,
                local_variant_id,
                kind_index=local_variant_id,
            ).kind
            for local_variant_id in range(config.recovery.variants_per_episode)
        )
    )
    quality = recovery_quality_summary(
        attempted_by_kind,
        accepted_by_kind,
        minimum_rate=config.recovery.min_acceptance_rate,
        expected_kinds=expected_kinds,
    )
    _write_json(quality_path, quality)
    require_recovery_quality(quality)
    if not recovery_records:
        raise RuntimeError(
            "recovery augmentation produced no successful trajectories; "
            f"see {rejection_path}"
        )
    return recovery_records, quality


def collect_from_config(
    config_path: str | Path,
    *,
    expert_gate: str | Path | None = None,
    show_progress: bool = False,
) -> Path:
    config_file = Path(config_path)
    config = load_config(config_file)
    if config.backend != "franka_contact":
        raise ValueError("physics collection requires backend=franka_contact")
    gate_path = (
        Path(expert_gate)
        if expert_gate is not None
        else Path(config.output_dir) / "expert_gate.json"
    )
    gate_hash = require_expert_gate(config_file, gate_path)
    data_dir = Path(config.data_dir)
    manifest_path = data_dir / "manifest.json"
    rejection_path = data_dir / "rejections.json"
    records: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    _write_json(manifest_path, records)
    _write_json(rejection_path, rejections)
    attempt = 0
    max_attempts = config.train.episodes * 10
    base_progress = (
        tqdm(
            total=config.train.episodes,
            desc="base data",
            unit="episode",
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )
    try:
        while len(records) < config.train.episodes and attempt < max_attempts:
            object_count = config.train.object_counts[
                attempt % len(config.train.object_counts)
            ]
            sequence = np.random.SeedSequence((config.seed, 0x44415441, attempt))
            seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
            env = FrankaContactEnv(
                max_objects=config.max_objects,
                max_steps=config.environment.max_steps,
                min_object_distance=config.environment.min_object_distance,
                workspace_low=config.environment.workspace_low,
                workspace_high=config.environment.workspace_high,
                crowded_anchor_min_distance=(
                    config.environment.crowded_anchor_min_distance
                ),
                crowded_anchor_max_distance=(
                    config.environment.crowded_anchor_max_distance
                ),
                physics=config.physics,
            )
            recorder = (
                MultiViewRecorder(
                    env.model,
                    width=config.recording.width,
                    height=config.recording.height,
                )
                if config.recording.enabled
                else None
            )
            try:
                episode = collect_physics_episode(
                    env,
                    PhysicsScriptedExpert(config.physics),
                    seed=seed,
                    object_count=object_count,
                    expert_gate_hash=gate_hash,
                    recorder=recorder,
                )
            finally:
                if recorder is not None:
                    recorder.close()
            if episode.reason == "success":
                episode_index = len(records)
                path = save_physics_episode(
                    episode,
                    data_dir / f"episode_{episode_index:06d}.npz",
                    rgbd_path=(
                        data_dir / f"episode_{episode_index:06d}_rgbd.npz"
                        if config.recording.enabled
                        else None
                    ),
                )
                records.append(
                    {
                        "episode_index": episode_index,
                        "attempt": attempt,
                        "seed": seed,
                        "object_count": object_count,
                        "frames": len(episode.actions),
                        "reason": episode.reason,
                        "trajectory_source": episode.trajectory_source,
                        "path": path.name,
                    }
                )
                _write_json(manifest_path, records)
                if base_progress is not None:
                    base_progress.update(1)
            else:
                rejections.append(
                    {
                        "attempt": attempt,
                        "seed": seed,
                        "object_count": object_count,
                        "reason": episode.reason,
                        "frames": len(episode.actions),
                    }
                )
                _write_json(rejection_path, rejections)
            if base_progress is not None:
                base_progress.set_postfix(
                    attempts=attempt + 1,
                    accepted=len(records),
                    rejected=len(rejections),
                    objects=object_count,
                    reason=episode.reason,
                )
            attempt += 1
    finally:
        if base_progress is not None:
            base_progress.close()
    if len(records) != config.train.episodes:
        raise RuntimeError(
            f"collected {len(records)}/{config.train.episodes} successful episodes; "
            f"see {rejection_path}"
        )
    if config.recovery.enabled and config.sequence.enabled:
        split = deterministic_source_split(
            (int(record["seed"]) for record in records),
            seed=config.seed,
        )
        training_recovery_sources = select_training_recovery_sources(
            split.train,
            fraction=config.recovery.training_source_fraction,
            seed=config.seed,
        )
        benchmark_sources = tuple(sorted(split.validation + split.test))
        validate_derived_sources(
            split,
            training_recovery_sources=training_recovery_sources,
            benchmark_sources=benchmark_sources,
        )
        save_source_split(
            data_dir / "source_split.json",
            split,
            training_recovery_sources=training_recovery_sources,
            benchmark_sources=benchmark_sources,
        )
        source_split_by_seed = {
            source_seed: split_name
            for split_name, source_seeds in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
            for source_seed in source_seeds
        }
        for record in records:
            source_seed = int(record["seed"])
            record["source_seed"] = source_seed
            record["source_split"] = source_split_by_seed[source_seed]
        _write_json(manifest_path, records)
        records_by_seed = {int(record["seed"]): record for record in records}
        _collect_recovery_group(
            config=config,
            records_by_seed=records_by_seed,
            source_seeds=training_recovery_sources,
            source_split_by_seed=source_split_by_seed,
            gate_hash=gate_hash,
            manifest_path=data_dir / "recovery_manifest.json",
            rejection_path=data_dir / "recovery_rejections.json",
            quality_path=data_dir / "recovery_quality.json",
            filename_prefix="train_recovery",
            progress_description="training recovery",
            show_progress=show_progress,
        )
        _collect_recovery_group(
            config=config,
            records_by_seed=records_by_seed,
            source_seeds=benchmark_sources,
            source_split_by_seed=source_split_by_seed,
            gate_hash=gate_hash,
            manifest_path=data_dir / "recovery_benchmark_manifest.json",
            rejection_path=data_dir / "recovery_benchmark_rejections.json",
            quality_path=data_dir / "recovery_benchmark_quality.json",
            filename_prefix="benchmark_recovery",
            progress_description="held-out recovery",
            show_progress=show_progress,
        )
        return manifest_path
    if config.recovery.enabled:
        splits = split_episode_seeds(
            (int(record["seed"]) for record in records),
            validation_fraction=0.1,
            test_fraction=0.1,
            seed=config.seed,
        )
        training_seeds = set(splits.train)
        recovery_records: list[dict[str, object]] = []
        recovery_rejections: list[dict[str, object]] = []
        recovery_manifest = data_dir / "recovery_manifest.json"
        recovery_rejection_path = data_dir / "recovery_rejections.json"
        split_path = data_dir / "recovery_source_split.json"
        _write_json(recovery_manifest, recovery_records)
        _write_json(recovery_rejection_path, recovery_rejections)
        _write_json(
            split_path,
            {
                "train": list(splits.train),
                "validation": list(splits.validation),
                "test": list(splits.test),
            },
        )
        training_records = tuple(
            (source_index, record)
            for source_index, record in enumerate(records)
            if int(record["seed"]) in training_seeds
        )
        attempted_by_kind: Counter[PhysicsRecoveryKind] = Counter()
        accepted_by_kind: Counter[PhysicsRecoveryKind] = Counter()
        recovery_progress = (
            tqdm(
                total=(
                    len(training_records) * config.recovery.variants_per_episode
                ),
                desc="recovery",
                unit="attempt",
                dynamic_ncols=True,
            )
            if show_progress
            else None
        )
        try:
            for source_index, record in training_records:
                source_seed = int(record["seed"])
                for local_variant_id in range(
                    config.recovery.variants_per_episode
                ):
                    variant_id = (
                        source_index * config.recovery.variants_per_episode
                        + local_variant_id
                    )
                    spec = make_physics_recovery_spec(
                        source_seed,
                        variant_id,
                        kind_index=local_variant_id,
                    )
                    attempted_by_kind[spec.kind] += 1
                    env = FrankaContactEnv(
                        max_objects=config.max_objects,
                        max_steps=config.environment.max_steps,
                        min_object_distance=config.environment.min_object_distance,
                        workspace_low=config.environment.workspace_low,
                        workspace_high=config.environment.workspace_high,
                        crowded_anchor_min_distance=(
                            config.environment.crowded_anchor_min_distance
                        ),
                        crowded_anchor_max_distance=(
                            config.environment.crowded_anchor_max_distance
                        ),
                        physics=config.physics,
                    )
                    recorder = (
                        MultiViewRecorder(
                            env.model,
                            width=config.recording.width,
                            height=config.recording.height,
                        )
                        if config.recording.enabled
                        else None
                    )
                    recovery_reason = ""
                    try:
                        try:
                            episode = collect_physics_episode(
                                env,
                                PhysicsScriptedExpert(config.physics),
                                seed=source_seed,
                                object_count=int(record["object_count"]),
                                recovery=spec,
                                expert_gate_hash=gate_hash,
                                recorder=recorder,
                            )
                        except PhysicsRecoveryRejected as error:
                            recovery_reason = error.reason
                            recovery_rejections.append(
                                {
                                    "source_seed": source_seed,
                                    "variant_id": variant_id,
                                    "kind": spec.kind.value,
                                    "object_count": int(record["object_count"]),
                                    "frames": 0,
                                    "reason": error.reason,
                                }
                            )
                            _write_json(
                                recovery_rejection_path,
                                recovery_rejections,
                            )
                        except Exception as error:
                            recovery_reason = f"exception:{type(error).__name__}"
                            raise
                        else:
                            recovery_reason = episode.reason
                            recovery_record = {
                                "source_seed": source_seed,
                                "variant_id": variant_id,
                                "kind": spec.kind.value,
                                "object_count": int(record["object_count"]),
                                "frames": len(episode.actions),
                                "reason": episode.reason,
                            }
                            if episode.reason == "success":
                                path = save_physics_episode(
                                    episode,
                                    data_dir
                                    / (
                                        f"recovery_{source_seed:010d}_v"
                                        f"{variant_id:03d}.npz"
                                    ),
                                    rgbd_path=(
                                        data_dir
                                        / (
                                            f"recovery_{source_seed:010d}_v"
                                            f"{variant_id:03d}_rgbd.npz"
                                        )
                                        if config.recording.enabled
                                        else None
                                    ),
                                )
                                recovery_record["path"] = path.name
                                recovery_records.append(recovery_record)
                                accepted_by_kind[spec.kind] += 1
                                _write_json(recovery_manifest, recovery_records)
                            else:
                                recovery_rejections.append(recovery_record)
                                _write_json(
                                    recovery_rejection_path,
                                    recovery_rejections,
                                )
                    finally:
                        if recorder is not None:
                            recorder.close()
                        if recovery_progress is not None:
                            recovery_progress.set_postfix(
                                kind=spec.kind.value,
                                source_seed=source_seed,
                                accepted=len(recovery_records),
                                rejected=len(recovery_rejections),
                                reason=recovery_reason,
                            )
                            recovery_progress.update(1)
        finally:
            if recovery_progress is not None:
                recovery_progress.close()
        quality = recovery_quality_summary(
            attempted_by_kind,
            accepted_by_kind,
            minimum_rate=config.recovery.min_acceptance_rate,
            expected_kinds=tuple(
                dict.fromkeys(
                    make_physics_recovery_spec(
                        config.seed,
                        local_variant_id,
                        kind_index=local_variant_id,
                    ).kind
                    for local_variant_id in range(
                        config.recovery.variants_per_episode
                    )
                )
            ),
        )
        _write_json(data_dir / "recovery_quality.json", quality)
        require_recovery_quality(quality)
        if not recovery_records:
            raise RuntimeError(
                "recovery augmentation produced no successful trajectories; "
                f"see {recovery_rejection_path}"
            )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect physical Franka expert episodes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--config", default="configs/physics_smoke_macos.yaml")
    collect.add_argument("--expert-gate")
    args = parser.parse_args()
    print(
        collect_from_config(
            args.config,
            expert_gate=args.expert_gate,
            show_progress=True,
        )
    )


if __name__ == "__main__":
    main()
