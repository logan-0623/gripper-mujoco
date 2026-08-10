from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from interaction_vla.env import LayoutMode
from interaction_vla.lerobot_bridge.codecs import (
    EndEffectorStateCodec,
    LocalCartesianActionCodec,
    validate_finger_joint_ranges,
)
from interaction_vla.lerobot_bridge.config import BridgeConfig, load_bridge_config
from interaction_vla.lerobot_bridge.dataset_writer import standard_features
from interaction_vla.lerobot_bridge.provenance import (
    sha256_file,
    source_fingerprint,
    standard_dataset_fingerprint,
)
from interaction_vla.lerobot_bridge.sidecar import load_teacher_sidecar
from interaction_vla.lerobot_bridge.teacher import validate_typed_relation_goals
from interaction_vla.lerobot_bridge.teacher_schema import (
    FORBIDDEN_FIELD_FRAGMENTS,
    SCHEMA_VERSION,
    teacher_schema_payload,
)
from interaction_vla.physics_data import require_expert_gate
from interaction_vla.physics_env import FrankaContactEnv


POLICY_FEATURE_KEYS = {
    "observation.images.agent",
    "observation.images.wrist",
    "observation.state",
    "action",
}
GENERATED_FEATURE_KEYS = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}
ALLOWED_SAMPLE_KEYS = POLICY_FEATURE_KEYS | GENERATED_FEATURE_KEYS | {"task"}
FORBIDDEN_POLICY_FRAGMENTS = (
    "annotation",
    "depth",
    "segmentation",
) + FORBIDDEN_FIELD_FRAGMENTS


def validate_replay_states(
    recorded: np.ndarray,
    replayed: np.ndarray,
    *,
    tolerance: float = 1e-5,
) -> float:
    first = np.asarray(recorded, dtype=np.float64)
    second = np.asarray(replayed, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 10:
        raise ValueError("replay state arrays must have matching shape [T, 10]")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("replay state arrays must be finite")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("replay tolerance must be finite and non-negative")
    maximum = float(np.max(np.abs(first - second), initial=0.0))
    if maximum > tolerance:
        raise ValueError(
            f"deterministic replay state error {maximum:.8g} exceeds {tolerance:.8g}"
        )
    return maximum


def validate_teacher_manifest(
    records: list[Mapping[str, Any]],
    *,
    dataset_episode_lengths: Mapping[int, int],
) -> int:
    expected_indices = list(range(len(dataset_episode_lengths)))
    actual_indices = [int(record.get("episode_index", -1)) for record in records]
    if actual_indices != expected_indices:
        raise ValueError("teacher manifest episode indices must be contiguous")
    if set(actual_indices) != set(dataset_episode_lengths):
        raise ValueError("teacher manifest episodes do not match the standard dataset")
    total = 0
    for record in records:
        episode_index = int(record["episode_index"])
        frames = int(record.get("frames", -1))
        if frames != int(dataset_episode_lengths[episode_index]):
            raise ValueError(
                f"teacher manifest frame count mismatch for episode {episode_index}"
            )
        expected_path = f"teacher/episode_{episode_index:06d}.npz"
        if record.get("path") != expected_path:
            raise ValueError(f"teacher manifest path mismatch for episode {episode_index}")
        total += frames
    return total


def validate_teacher_schema(schema: Mapping[str, Any]) -> None:
    if dict(schema) != teacher_schema_payload():
        raise ValueError("teacher schema differs from the TC-TIG contract")


def validate_dataset_contract(
    *,
    resolved_repo_id: str,
    tasks: list[str],
    episode_lengths: Mapping[int, int],
    records: list[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    bridge_config: BridgeConfig,
) -> None:
    dataset = bridge_config.dataset
    expected_provenance = {
        "repo_id": dataset.repo_id,
        "task": dataset.task,
        "requested_episodes": dataset.episodes,
        "accepted_episodes": dataset.episodes,
        "fps": dataset.fps,
        "image_size": list(dataset.image_size),
        "object_counts": list(dataset.object_counts),
    }
    if resolved_repo_id != dataset.repo_id:
        raise ValueError("standard dataset repo_id differs from the bridge config")
    if tasks != [dataset.task]:
        raise ValueError("standard dataset task metadata differs from the bridge config")
    if len(episode_lengths) != dataset.episodes:
        raise ValueError("standard dataset episode count differs from the bridge config")
    differing = [
        key for key, value in expected_provenance.items() if provenance.get(key) != value
    ]
    if differing:
        raise ValueError(
            "bridge provenance dataset contract mismatch: " + ", ".join(differing)
        )
    for record in records:
        if record.get("task") != dataset.task or int(record.get("task_id", -1)) != 0:
            raise ValueError("teacher manifest task metadata differs from the dataset")
        if int(record.get("object_count", -1)) not in dataset.object_counts:
            raise ValueError("teacher manifest object count is outside the bridge config")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid required JSON metadata: {path}") from error


def _repo_id(root: Path, requested: str | None) -> str:
    if requested:
        return requested
    info = _load_json(root / "meta" / "info.json")
    value = info.get("repo_id")
    if not isinstance(value, str) or not value:
        raise ValueError("repo_id is required and is missing from meta/info.json")
    return value


def _episode_lengths(episode_indices: np.ndarray) -> dict[int, int]:
    values = np.asarray(episode_indices, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("standard dataset must contain episode indices")
    unique, counts = np.unique(values, return_counts=True)
    expected = np.arange(len(unique), dtype=np.int64)
    if not np.array_equal(unique, expected):
        raise ValueError("standard dataset episode indices must be contiguous")
    return {int(index): int(count) for index, count in zip(unique, counts, strict=True)}


def _validate_standard_rows(dataset: Any) -> tuple[dict[int, int], dict[str, object]]:
    feature_keys = set(dataset.features)
    expected_features = POLICY_FEATURE_KEYS | GENERATED_FEATURE_KEYS
    if feature_keys != expected_features:
        missing = sorted(expected_features - feature_keys)
        extra = sorted(feature_keys - expected_features)
        raise ValueError(f"standard feature keys mismatch; missing={missing}, extra={extra}")
    forbidden = sorted(
        key
        for key in feature_keys
        if any(fragment in key.lower() for fragment in FORBIDDEN_POLICY_FRAGMENTS)
    )
    if forbidden:
        raise ValueError(f"forbidden policy feature keys: {forbidden}")
    expected_schema = standard_features(width=256, height=256)
    for key, expected in expected_schema.items():
        actual = dataset.features[key]
        if actual["dtype"] != expected["dtype"]:
            raise ValueError(f"standard feature dtype mismatch: {key}")
        if tuple(actual["shape"]) != tuple(expected["shape"]):
            raise ValueError(f"standard feature shape mismatch: {key}")
        if actual.get("names") != expected.get("names"):
            raise ValueError(f"standard feature names mismatch: {key}")

    raw = dataset.hf_dataset
    states = np.asarray(raw["observation.state"], dtype=np.float32)
    actions = np.asarray(raw["action"], dtype=np.float32)
    episode_indices = np.asarray(raw["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(raw["frame_index"], dtype=np.int64)
    timestamps = np.asarray(raw["timestamp"], dtype=np.float64)
    if states.shape != (len(dataset), 10) or actions.shape != (len(dataset), 7):
        raise ValueError("standard state/action arrays have invalid shapes")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("standard state/action arrays must be finite")
    if np.any(states[:, 9] < 0.0) or np.any(states[:, 9] > 1.0):
        raise ValueError("gripper aperture state must lie within [0, 1]")
    if np.any(actions[:, :6] < -1.0) or np.any(actions[:, :6] > 1.0):
        raise ValueError("Cartesian actions must lie within [-1, 1]")
    if not np.all(np.isin(actions[:, 6], (0.0, 1.0))):
        raise ValueError("gripper actions must be binary")

    lengths = _episode_lengths(episode_indices)
    for episode_index, length in lengths.items():
        mask = episode_indices == episode_index
        if not np.array_equal(frame_indices[mask], np.arange(length)):
            raise ValueError(f"frame indices are not contiguous in episode {episode_index}")
        if not np.allclose(
            timestamps[mask],
            np.arange(length) / 20.0,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"timestamps are not 20 Hz in episode {episode_index}")

    sample = dataset[0]
    sample_keys = set(sample)
    if sample_keys != ALLOWED_SAMPLE_KEYS:
        raise ValueError(
            f"standard sample keys mismatch; missing={sorted(ALLOWED_SAMPLE_KEYS - sample_keys)}, "
            f"extra={sorted(sample_keys - ALLOWED_SAMPLE_KEYS)}"
        )
    sample_forbidden = sorted(
        key
        for key in sample_keys
        if any(fragment in key.lower() for fragment in FORBIDDEN_POLICY_FRAGMENTS)
    )
    if sample_forbidden:
        raise ValueError(f"forbidden policy sample keys: {sample_forbidden}")
    for image_key in ("observation.images.agent", "observation.images.wrist"):
        if tuple(sample[image_key].shape) != (3, 256, 256):
            raise ValueError(f"decoded image shape mismatch: {image_key}")
    if tuple(sample["observation.state"].shape) != (10,):
        raise ValueError("decoded state shape mismatch")
    if tuple(sample["action"].shape) != (7,):
        raise ValueError("decoded action shape mismatch")
    return lengths, {
        "states": states,
        "actions": actions,
        "episode_indices": episode_indices,
        "image_shape": [3, 256, 256],
        "state_shape": [10],
        "action_shape": [7],
        "forbidden_policy_keys": [],
    }


def _validate_bridge_metadata(
    root: Path,
    *,
    episode_lengths: Mapping[int, int],
    allow_incomplete: bool,
    bridge_config: BridgeConfig | None,
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    meta = root / "meta"
    schema_path = meta / "tc_tig_teacher_schema.json"
    calibration_path = meta / "camera_calibration.json"
    manifest_path = meta / "teacher_manifest.json"
    rejections_path = meta / "rejections.json"
    provenance_path = meta / "bridge_provenance.json"
    for path in (
        schema_path,
        calibration_path,
        manifest_path,
        rejections_path,
        provenance_path,
    ):
        if not path.is_file():
            raise ValueError(f"missing bridge metadata: {path}")
    schema = _load_json(schema_path)
    if not isinstance(schema, dict):
        raise ValueError("teacher schema must be a JSON object")
    validate_teacher_schema(schema)
    _load_json(calibration_path)
    records = _load_json(manifest_path)
    if not isinstance(records, list):
        raise ValueError("teacher manifest must be a JSON list")
    _load_json(rejections_path)
    provenance = _load_json(provenance_path)
    if not isinstance(provenance, dict):
        raise ValueError("bridge provenance must be a JSON object")
    if not allow_incomplete and not provenance.get("complete", False):
        raise ValueError("bridge provenance is not complete")
    validate_teacher_manifest(records, dataset_episode_lengths=episode_lengths)
    checked_hashes = []
    expected_hashes = {
        "teacher_schema_sha256": sha256_file(schema_path),
        "camera_calibration_sha256": sha256_file(calibration_path),
    }
    if provenance.get("complete", False):
        expected_hashes.update(
            {
                "teacher_manifest_sha256": sha256_file(manifest_path),
                "rejections_sha256": sha256_file(rejections_path),
                "standard_dataset_fingerprint": standard_dataset_fingerprint(root),
            }
        )
    for key, expected in expected_hashes.items():
        if provenance.get(key) != expected:
            raise ValueError(f"bridge provenance hash mismatch: {key}")
        checked_hashes.append(key)

    for record in records:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("teacher manifest schema version mismatch")
        sidecar = root / str(record["path"])
        arrays = load_teacher_sidecar(
            sidecar, expected_sha256=str(record.get("sha256", ""))
        )
        episode_index = int(record["episode_index"])
        length = int(episode_lengths[episode_index])
        if not np.array_equal(arrays["frame_index"], np.arange(length)):
            raise ValueError("teacher sidecar frame indices are not contiguous")
        if not np.allclose(
            arrays["timestamp"],
            np.arange(length) / 20.0,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("teacher sidecar timestamps are not 20 Hz")
        state_hashes = arrays["state_hash"].astype(str)
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in state_hashes
        ):
            raise ValueError("teacher sidecar contains an invalid state hash")
        if record.get("first_state_hash") != state_hashes[0]:
            raise ValueError("teacher manifest first state hash mismatch")
        if record.get("last_state_hash") != state_hashes[-1]:
            raise ValueError("teacher manifest last state hash mismatch")
        if bridge_config is not None:
            validate_typed_relation_goals(
                arrays["annotation.tc_tig.relation_values"],
                arrays["annotation.tc_tig.relation_goal"],
                horizon=bridge_config.teacher.goal_horizon,
                minimum_improvement=bridge_config.teacher.goal_improvement_margin,
            )

    if bridge_config is not None:
        config_hashes = {
            "bridge_config_sha256": sha256_file(bridge_config.config_path),
            "source_config_sha256": sha256_file(bridge_config.source_config_path),
            "expert_gate_sha256": sha256_file(bridge_config.expert_gate),
            "source_fingerprint": source_fingerprint(),
        }
        for key, expected in config_hashes.items():
            if provenance.get(key) != expected:
                raise ValueError(f"bridge source/config/gate hash mismatch: {key}")
            checked_hashes.append(key)
        require_expert_gate(
            bridge_config.source_config_path, bridge_config.expert_gate
        )
        runtime = provenance.get("runtime", {})
        if runtime.get("lerobot") != importlib.metadata.version("lerobot"):
            raise ValueError("LeRobot runtime version differs from collection")
        if runtime.get("mujoco") != importlib.metadata.version("mujoco"):
            raise ValueError("MuJoCo runtime version differs from collection")
    return records, provenance, checked_hashes


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


def _replay(
    dataset: Any,
    *,
    records: list[dict[str, object]],
    bridge_config: BridgeConfig,
) -> float:
    raw = dataset.hf_dataset
    episode_indices = np.asarray(raw["episode_index"], dtype=np.int64)
    all_states = np.asarray(raw["observation.state"], dtype=np.float32)
    all_actions = np.asarray(raw["action"], dtype=np.float32)
    maximum_error = 0.0
    for record in records:
        episode_index = int(record["episode_index"])
        mask = episode_indices == episode_index
        states = all_states[mask]
        actions = all_actions[mask]
        env = _make_env(bridge_config)
        validate_finger_joint_ranges(env.model)
        snapshot = env.reset(
            seed=int(record["seed"]),
            object_count=int(record["object_count"]),
            layout_mode=LayoutMode.NORMAL,
        )
        replayed: list[np.ndarray] = []
        final_reason = "running"
        for action_local in actions:
            replayed.append(
                EndEffectorStateCodec.encode_snapshot(
                    snapshot, env.proprioception()
                )
            )
            rotation = EndEffectorStateCodec.quaternion_to_matrix(
                snapshot.gripper.orientation
            )
            action_world = LocalCartesianActionCodec.decode(action_local, rotation)
            transition = env.step(action_world)
            snapshot = transition.snapshot
            final_reason = str(getattr(transition.reason, "value", transition.reason))
        maximum_error = max(
            maximum_error,
            validate_replay_states(states, np.stack(replayed), tolerance=1e-5),
        )
        if final_reason != record.get("final_reason"):
            raise ValueError(
                f"replay final reason mismatch for episode {episode_index}: {final_reason}"
            )
    return maximum_error


def validate_dataset_root(
    root: str | Path,
    *,
    repo_id: str | None = None,
    allow_incomplete: bool = False,
    require_bridge_metadata: bool = True,
    replay: bool = True,
    bridge_config: BridgeConfig | None = None,
) -> dict[str, object]:
    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root does not exist: {dataset_root}")
    incomplete = dataset_root / "INCOMPLETE"
    if incomplete.exists() and not allow_incomplete:
        raise ValueError(f"dataset contains INCOMPLETE marker: {incomplete}")

    from lerobot.datasets import LeRobotDataset

    resolved_repo_id = _repo_id(dataset_root, repo_id)
    dataset = LeRobotDataset(resolved_repo_id, root=dataset_root)
    episode_lengths, standard = _validate_standard_rows(dataset)
    records: list[dict[str, object]] = []
    provenance: dict[str, object] = {}
    checked_hashes: list[str] = []
    if require_bridge_metadata:
        records, provenance, checked_hashes = _validate_bridge_metadata(
            dataset_root,
            episode_lengths=episode_lengths,
            allow_incomplete=allow_incomplete,
            bridge_config=bridge_config,
        )
        if bridge_config is not None and provenance.get("complete", False):
            validate_dataset_contract(
                resolved_repo_id=resolved_repo_id,
                tasks=dataset.meta.tasks.index.tolist(),
                episode_lengths=episode_lengths,
                records=records,
                provenance=provenance,
                bridge_config=bridge_config,
            )
    replay_error = 0.0
    if replay:
        if bridge_config is None or not records:
            raise ValueError("deterministic replay requires bridge metadata and config")
        replay_error = _replay(dataset, records=records, bridge_config=bridge_config)
    return {
        "passed": True,
        "episodes": len(episode_lengths),
        "frames": len(dataset),
        "image_shape": standard["image_shape"],
        "state_shape": standard["state_shape"],
        "action_shape": standard["action_shape"],
        "forbidden_policy_keys": standard["forbidden_policy_keys"],
        "tasks": dataset.meta.tasks.index.tolist(),
        "checked_hashes": sorted(set(checked_hashes)),
        "replay_max_abs_error": replay_error,
    }


def validate_from_config(
    config_path: str | Path,
    *,
    no_replay: bool = False,
) -> dict[str, object]:
    config = load_bridge_config(config_path)
    return validate_dataset_root(
        config.dataset.root,
        repo_id=config.dataset.repo_id,
        allow_incomplete=False,
        require_bridge_metadata=True,
        replay=not no_replay,
        bridge_config=config,
    )
