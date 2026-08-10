from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from interaction_vla.env import LayoutMode, TerminationReason
from interaction_vla.lerobot_bridge.capture import DualViewCapture, camera_calibration
from interaction_vla.lerobot_bridge.codecs import (
    EndEffectorStateCodec,
    LocalCartesianActionCodec,
    validate_finger_joint_ranges,
)
from interaction_vla.lerobot_bridge.config import BridgeConfig, load_bridge_config
from interaction_vla.lerobot_bridge.dataset_writer import LeRobotEpisodeWriter
from interaction_vla.lerobot_bridge.provenance import (
    git_commit,
    runtime_versions,
    sha256_file,
    source_fingerprint,
)
from interaction_vla.lerobot_bridge.sidecar import (
    TeacherSidecarRecord,
    TeacherSidecarWriter,
)
from interaction_vla.lerobot_bridge.teacher import (
    TCTIGTeacherExtractor,
    label_relation_goals,
)
from interaction_vla.lerobot_bridge.teacher_schema import (
    CONFIDENCE,
    ERROR_0,
    ERROR_1,
    TeacherFrame,
    teacher_schema_payload,
)
from interaction_vla.physics_data import require_expert_gate
from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.physics_expert import PhysicsScriptedExpert


@dataclass(frozen=True)
class AttemptResult:
    accepted: bool
    reason: str
    teacher_frames: tuple[Any, ...]
    world_actions: tuple[np.ndarray, ...]
    seed: int
    object_count: int

    @property
    def frames(self) -> int:
        return len(self.teacher_frames)


def collection_seed(config_seed: int, attempt: int) -> int:
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    sequence = np.random.SeedSequence((int(config_seed), 0x4C45524F, int(attempt)))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def require_new_root(root: str | Path) -> Path:
    destination = Path(root)
    if destination.exists():
        raise FileExistsError(f"collection requires a new dataset root: {destination}")
    return destination


def collect_attempt(
    *,
    env: Any,
    expert: Any,
    capture: Any,
    policy_writer: Any,
    teacher: Any,
    seed: int,
    object_count: int,
    task: str,
) -> AttemptResult:
    snapshot = env.reset(
        seed=seed,
        object_count=object_count,
        layout_mode=LayoutMode.NORMAL,
    )
    expert.reset(seed=seed)
    teacher.reset()
    teacher_frames: list[Any] = []
    world_actions: list[np.ndarray] = []
    while True:
        camera_frame = capture.capture(env, include_teacher=True)
        state = EndEffectorStateCodec.encode_snapshot(
            snapshot, env.proprioception()
        )
        rotation = EndEffectorStateCodec.quaternion_to_matrix(
            snapshot.gripper.orientation
        )
        teacher_frame = teacher.extract(
            snapshot,
            camera_frame,
            state=state,
        )
        action_world = expert.act(
            snapshot, env.contact_diagnostics, env.grasp_state
        )
        action_local = LocalCartesianActionCodec.encode(action_world, rotation)
        policy_writer.add_frame(
            agent_rgb=camera_frame.views["agent"].rgb,
            wrist_rgb=camera_frame.views["wrist"].rgb,
            state=state,
            action=action_local,
            task=task,
        )
        teacher_frames.append(teacher_frame)
        world_actions.append(np.asarray(action_world, dtype=np.float32).copy())
        transition = env.step(action_world)
        snapshot = transition.snapshot
        if transition.done:
            break
    reason = str(getattr(transition.reason, "value", transition.reason))
    accepted = reason == TerminationReason.SUCCESS.value
    if not accepted:
        policy_writer.clear_episode()
    return AttemptResult(
        accepted=accepted,
        reason=reason,
        teacher_frames=tuple(teacher_frames),
        world_actions=tuple(world_actions),
        seed=int(seed),
        object_count=int(object_count),
    )


def _make_env(config: BridgeConfig) -> FrankaContactEnv:
    source = config.source
    return FrankaContactEnv(
        max_objects=source.max_objects,
        max_steps=source.environment.max_steps,
        min_object_distance=source.environment.min_object_distance,
        workspace_low=source.environment.workspace_low,
        workspace_high=source.environment.workspace_high,
        crowded_anchor_min_distance=source.environment.crowded_anchor_min_distance,
        crowded_anchor_max_distance=source.environment.crowded_anchor_max_distance,
        physics=source.physics,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _create_incomplete_marker(path: Path) -> None:
    with path.open("xb") as handle:
        handle.write(b"collection in progress\n")
        handle.flush()
        os.fsync(handle.fileno())


def _relation_goals(frames: tuple[TeacherFrame, ...], config: BridgeConfig) -> np.ndarray:
    relation_values = np.stack([frame.relation_values for frame in frames])
    errors = relation_values[:, :, (ERROR_0, ERROR_1)]
    confidence = relation_values[:, :, CONFIDENCE]
    return label_relation_goals(
        errors,
        confidence,
        horizon=config.teacher.goal_horizon,
        minimum_improvement=config.teacher.goal_improvement_margin,
    )


def _require_smoke_report(path: Path | None) -> None:
    if path is None:
        return
    if not path.is_file():
        raise FileNotFoundError(f"required smoke report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed", False):
        raise ValueError(f"required smoke report did not pass: {path}")


def _manifest_entry(
    record: TeacherSidecarRecord,
    *,
    reason: str,
    task: str,
) -> dict[str, object]:
    result = asdict(record)
    result["final_reason"] = reason
    result["task"] = task
    return result


def collect_from_config(config_path: str | Path) -> dict[str, object]:
    config = load_bridge_config(config_path)
    gate_hash = require_expert_gate(config.source_config_path, config.expert_gate)
    _require_smoke_report(config.required_smoke_report)
    root = require_new_root(config.dataset.root)
    writer = LeRobotEpisodeWriter.create(
        repo_id=config.dataset.repo_id,
        root=root,
        fps=config.dataset.fps,
        width=config.dataset.image_size[1],
        height=config.dataset.image_size[0],
    )
    incomplete = root / "INCOMPLETE"
    _create_incomplete_marker(incomplete)

    probe_env = _make_env(config)
    probe_env.reset(
        seed=collection_seed(config.seed, 0),
        object_count=config.dataset.object_counts[0],
        layout_mode=LayoutMode.NORMAL,
    )
    validate_finger_joint_ranges(probe_env.model)
    calibration = camera_calibration(
        probe_env,
        width=config.dataset.image_size[1],
        height=config.dataset.image_size[0],
    )

    metadata = root / "meta"
    schema_path = metadata / "tc_tig_teacher_schema.json"
    calibration_path = metadata / "camera_calibration.json"
    manifest_path = metadata / "teacher_manifest.json"
    rejections_path = metadata / "rejections.json"
    provenance_path = metadata / "bridge_provenance.json"
    _write_json_atomic(schema_path, teacher_schema_payload())
    _write_json_atomic(calibration_path, calibration)
    _write_json_atomic(manifest_path, [])
    _write_json_atomic(rejections_path, [])
    provenance: dict[str, object] = {
        "complete": False,
        "repo_id": config.dataset.repo_id,
        "bridge_config": config.config_path,
        "bridge_config_sha256": sha256_file(config.config_path),
        "source_config": config.source_config_path,
        "source_config_sha256": sha256_file(config.source_config_path),
        "expert_gate": config.expert_gate,
        "expert_gate_sha256": gate_hash,
        "teacher_schema_sha256": sha256_file(schema_path),
        "camera_calibration_sha256": sha256_file(calibration_path),
        "source_fingerprint": source_fingerprint(),
        "git_commit": git_commit(),
        "runtime": runtime_versions(requested_device=config.act.device),
        "requested_episodes": config.dataset.episodes,
        "accepted_episodes": 0,
        "rejected_attempts": 0,
    }
    _write_json_atomic(provenance_path, provenance)

    sidecars = TeacherSidecarWriter(root, fps=config.dataset.fps)
    manifest: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    maximum_attempts = config.dataset.episodes * config.dataset.max_attempt_multiplier
    for attempt in range(maximum_attempts):
        if len(manifest) >= config.dataset.episodes:
            break
        seed = collection_seed(config.seed, attempt)
        object_count = config.dataset.object_counts[
            attempt % len(config.dataset.object_counts)
        ]
        env = _make_env(config)
        capture = DualViewCapture(
            env.model,
            width=config.dataset.image_size[1],
            height=config.dataset.image_size[0],
        )
        expert = PhysicsScriptedExpert(config.source.physics)
        teacher = TCTIGTeacherExtractor(
            config.teacher,
            model=env.model,
        )
        try:
            result = collect_attempt(
                env=env,
                expert=expert,
                capture=capture,
                policy_writer=writer,
                teacher=teacher,
                seed=seed,
                object_count=object_count,
                task=config.dataset.task,
            )
            if not result.accepted:
                rejections.append(
                    {
                        "attempt": attempt,
                        "seed": seed,
                        "object_count": object_count,
                        "frames": result.frames,
                        "reason": result.reason,
                    }
                )
                _write_json_atomic(rejections_path, rejections)
                continue
            teacher_frames = tuple(result.teacher_frames)
            goals = _relation_goals(teacher_frames, config)
            episode_index = len(manifest)
            record = sidecars.stage_episode(
                episode_index,
                teacher_frames,
                goals,
                seed=seed,
                object_count=object_count,
                task_id=0,
            )
            writer.save_episode()
            sidecars.commit_staged(record)
            manifest.append(
                _manifest_entry(
                    record,
                    reason=result.reason,
                    task=config.dataset.task,
                )
            )
            _write_json_atomic(manifest_path, manifest)
        finally:
            capture.close()

    if len(manifest) != config.dataset.episodes:
        raise RuntimeError(
            f"accepted {len(manifest)} of {config.dataset.episodes} requested episodes "
            f"after {maximum_attempts} attempts; incomplete dataset retained at {root}"
        )

    writer.finalize()
    from interaction_vla.lerobot_bridge.validator import validate_dataset_root

    validate_dataset_root(
        root,
        repo_id=config.dataset.repo_id,
        allow_incomplete=True,
        require_bridge_metadata=True,
        replay=True,
        bridge_config=config,
    )
    provenance.update(
        {
            "complete": True,
            "accepted_episodes": len(manifest),
            "rejected_attempts": len(rejections),
            "teacher_manifest_sha256": sha256_file(manifest_path),
            "rejections_sha256": sha256_file(rejections_path),
        }
    )
    _write_json_atomic(provenance_path, provenance)
    incomplete.unlink()
    root_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    return validate_dataset_root(
        root,
        repo_id=config.dataset.repo_id,
        allow_incomplete=False,
        require_bridge_metadata=True,
        replay=True,
        bridge_config=config,
    )
