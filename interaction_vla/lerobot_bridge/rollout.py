from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from interaction_vla.device import resolve_device
from interaction_vla.env import LayoutMode, TerminationReason
from interaction_vla.lerobot_bridge.act_smoke import (
    ACTION_CODEC_VERSION,
    STATE_CODEC_VERSION,
)
from interaction_vla.lerobot_bridge.capture import DualViewCapture
from interaction_vla.lerobot_bridge.codecs import (
    EndEffectorStateCodec,
    LocalCartesianActionCodec,
    validate_finger_joint_ranges,
)
from interaction_vla.lerobot_bridge.config import BridgeConfig, load_bridge_config
from interaction_vla.lerobot_bridge.provenance import (
    fingerprint_tree,
    sha256_file,
    source_fingerprint,
)
from interaction_vla.lerobot_bridge.validator import validate_dataset_root
from interaction_vla.physics_action_safety import project_cartesian_action
from interaction_vla.physics_env import FrankaContactEnv


def policy_observation(
    *,
    agent_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: np.ndarray,
) -> dict[str, torch.Tensor]:
    def image(value: np.ndarray, name: str) -> torch.Tensor:
        array = np.asarray(value)
        if array.shape != (256, 256, 3) or array.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 with shape (256, 256, 3)")
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)

    state_array = np.asarray(state)
    if state_array.shape != (10,) or state_array.dtype != np.float32:
        raise ValueError("state must be float32 with shape (10,)")
    if not np.isfinite(state_array).all():
        raise ValueError("state must be finite")
    return {
        "observation.images.agent": image(agent_rgb, "agent_rgb"),
        "observation.images.wrist": image(wrist_rgb, "wrist_rgb"),
        "observation.state": torch.from_numpy(state_array.copy()),
    }


class BinaryGripperHysteresis:
    def __init__(
        self,
        *,
        close_threshold: float,
        open_threshold: float,
        initially_open: bool,
    ) -> None:
        if not (
            np.isfinite(close_threshold)
            and np.isfinite(open_threshold)
            and 0.0 <= close_threshold < open_threshold <= 1.0
        ):
            raise ValueError("gripper thresholds must satisfy 0 <= close < open <= 1")
        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)
        self.initially_open = bool(initially_open)
        self.reset()

    def reset(self) -> None:
        self.is_open = self.initially_open
        self.switch_count = 0

    def resolve(self, score: float) -> float:
        value = float(score)
        if not np.isfinite(value):
            raise ValueError("gripper score must be finite")
        previous = self.is_open
        if self.is_open and value <= self.close_threshold:
            self.is_open = False
        elif not self.is_open and value >= self.open_threshold:
            self.is_open = True
        if self.is_open != previous:
            self.switch_count += 1
        return float(self.is_open)


@dataclass(frozen=True)
class QueuedAction:
    action: np.ndarray
    raw_chunk: np.ndarray
    queue_index: int


class ActionChunkQueue:
    def __init__(self, *, chunk_size: int) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        self.reset()

    def reset(self) -> None:
        self._chunk: np.ndarray | None = None
        self._index = 0

    def next(self, predict: Callable[[], np.ndarray]) -> QueuedAction:
        if self._chunk is None or self._index >= self.chunk_size:
            chunk = np.asarray(predict(), dtype=np.float32)
            expected = (self.chunk_size, 7)
            if chunk.shape != expected:
                raise ValueError(f"predicted action chunk must have shape {expected}")
            if not np.isfinite(chunk).all():
                raise ValueError("predicted action chunk must be finite")
            self._chunk = chunk.copy()
            self._index = 0
        assert self._chunk is not None
        queue_index = self._index
        action = self._chunk[queue_index].copy()
        raw_chunk = self._chunk.copy()
        self._index += 1
        return QueuedAction(action, raw_chunk, queue_index)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _load_checkpoint_bundle(
    *,
    config: BridgeConfig,
    checkpoint: Path,
    device: torch.device,
):
    metadata_path = checkpoint / "bridge_checkpoint.json"
    if not metadata_path.is_file():
        raise ValueError(f"ACT bridge checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_fingerprint": fingerprint_tree(config.dataset.root),
        "state_codec_version": STATE_CODEC_VERSION,
        "action_codec_version": ACTION_CODEC_VERSION,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "source_fingerprint": source_fingerprint(),
        "bridge_config_sha256": sha256_file(config.config_path),
        "source_config_sha256": sha256_file(config.source_config_path),
        "expert_gate_sha256": sha256_file(config.expert_gate),
    }
    differing = [key for key, value in expected.items() if metadata.get(key) != value]
    if differing:
        raise ValueError(f"ACT checkpoint binding mismatch: {', '.join(differing)}")

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    dataset_metadata = LeRobotDatasetMetadata(
        config.dataset.repo_id,
        root=config.dataset.root,
    )
    if _jsonable(metadata.get("features")) != _jsonable(dataset_metadata.features):
        raise ValueError("ACT checkpoint feature contract differs from the dataset")
    policy = ACTPolicy.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
    )
    policy.eval()
    return policy, preprocessor, postprocessor, metadata


def _predict_chunk(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    observation: dict[str, torch.Tensor],
) -> np.ndarray:
    if set(observation) != {
        "observation.images.agent",
        "observation.images.wrist",
        "observation.state",
    }:
        raise ValueError("ACT rollout observation violates the RGB/state contract")
    processed = preprocessor(observation)
    with torch.no_grad():
        normalized = policy.predict_action_chunk(processed)
        actions = postprocessor(normalized)
    if not isinstance(actions, torch.Tensor) or actions.shape != (1, 8, 7):
        shape = getattr(actions, "shape", None)
        raise ValueError(f"ACT postprocessed chunk must have shape (1, 8, 7), got {shape}")
    values = actions[0].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("ACT postprocessed chunk must be finite")
    return values


def rollout_checkpoint(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    seed: int,
    object_count: int = 2,
) -> dict[str, object]:
    config = load_bridge_config(config_path)
    validate_dataset_root(
        config.dataset.root,
        repo_id=config.dataset.repo_id,
        allow_incomplete=False,
        require_bridge_metadata=True,
        replay=True,
        bridge_config=config,
    )
    if object_count < 2 or object_count > config.source.max_objects:
        raise ValueError("rollout object_count is outside the source environment range")
    device = resolve_device(config.act.device)
    checkpoint_path = Path(checkpoint)
    policy, preprocessor, postprocessor, metadata = _load_checkpoint_bundle(
        config=config,
        checkpoint=checkpoint_path,
        device=device,
    )
    env = _make_env(config)
    validate_finger_joint_ranges(env.model)
    snapshot = env.reset(
        seed=seed,
        object_count=object_count,
        layout_mode=LayoutMode.NORMAL,
    )
    capture = DualViewCapture(
        env.model,
        width=config.dataset.image_size[1],
        height=config.dataset.image_size[0],
    )
    queue = ActionChunkQueue(chunk_size=8)
    gripper = BinaryGripperHysteresis(
        close_threshold=0.4,
        open_threshold=0.6,
        initially_open=True,
    )
    diagnostics: list[dict[str, object]] = []
    final_reason = TerminationReason.RUNNING.value
    try:
        for step in range(180):
            camera_frame = capture.capture(env, include_teacher=False)
            state = EndEffectorStateCodec.encode_snapshot(
                snapshot, env.proprioception()
            )
            observation = policy_observation(
                agent_rgb=camera_frame.views["agent"].rgb,
                wrist_rgb=camera_frame.views["wrist"].rgb,
                state=state,
            )
            selected = queue.next(
                lambda: _predict_chunk(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    observation=observation,
                )
            )
            raw_local = selected.action.copy()
            local_action = raw_local.copy()
            local_action[:6] = np.clip(local_action[:6], -1.0, 1.0)
            clipped_dimensions = np.flatnonzero(
                local_action[:6] != raw_local[:6]
            ).astype(int)
            local_action[6] = gripper.resolve(float(raw_local[6]))
            rotation = EndEffectorStateCodec.quaternion_to_matrix(
                snapshot.gripper.orientation
            )
            action_world = LocalCartesianActionCodec.decode(local_action, rotation)
            projection = project_cartesian_action(env.controller, action_world)
            transition = env.step(projection.action)
            final_reason = str(
                getattr(transition.reason, "value", transition.reason)
            )
            diagnostics.append(
                {
                    "step": step,
                    "state_hash": camera_frame.state_hash,
                    "raw_chunk": selected.raw_chunk,
                    "queue_index": selected.queue_index,
                    "raw_local_action": raw_local,
                    "decoded_world_action": action_world,
                    "projected_world_action": projection.action,
                    "clipped_dimensions": clipped_dimensions,
                    "gripper_open": gripper.is_open,
                    "gripper_switch_count": gripper.switch_count,
                    "ik_projection_scale": projection.scale,
                    "ik_position_error": projection.projected_diagnostics.position_error,
                    "ik_orientation_error": projection.projected_diagnostics.orientation_error,
                    "terminal_reason": final_reason,
                }
            )
            snapshot = transition.snapshot
            if transition.done:
                break
    finally:
        capture.close()
    result: dict[str, object] = {
        "passed": True,
        "finite_rollout": True,
        "task_success": final_reason == TerminationReason.SUCCESS.value,
        "terminal_reason": final_reason,
        "steps": len(diagnostics),
        "seed": int(seed),
        "object_count": int(object_count),
        "device": str(device),
        "checkpoint": checkpoint_path,
        "checkpoint_dataset_fingerprint": metadata["dataset_fingerprint"],
        "diagnostics": diagnostics,
    }
    _write_json_atomic(config.act.output_dir / "rollout.json", result)
    return result


def rollout_from_config(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    seed: int | None = None,
    object_count: int = 2,
) -> dict[str, object]:
    config = load_bridge_config(config_path)
    if seed is None:
        seed = int(
            np.random.SeedSequence((config.seed, 0x524F4C4C)).generate_state(
                1, dtype=np.uint32
            )[0]
        )
    return rollout_checkpoint(
        config_path,
        checkpoint,
        seed=seed,
        object_count=object_count,
    )
