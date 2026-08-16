from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from interaction_vla.device import resolve_device
from interaction_vla.lerobot_bridge.act_smoke import (
    _audit_batch,
    load_act_dataset,
    pilot_episode_split,
)
from interaction_vla.lerobot_bridge.config import load_bridge_config
from interaction_vla.lerobot_bridge.interaction_phase import (
    PHASE_NAMES,
    causal_phase_ids,
)
from interaction_vla.lerobot_bridge.rollout import (
    _load_checkpoint_bundle,
    _write_json_atomic,
)
from interaction_vla.lerobot_bridge.sidecar import load_teacher_sidecar
from interaction_vla.lerobot_bridge.validator import validate_dataset_root


def action_chunk_metrics(
    predicted: object, target: object, action_is_pad: object
) -> dict[str, Any]:
    prediction = np.asarray(predicted, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    padding = np.asarray(action_is_pad, dtype=np.bool_)
    if (
        prediction.ndim != 3
        or prediction.shape[-1] != 7
        or prediction.shape != truth.shape
    ):
        raise ValueError(
            "predicted and target actions must share shape [batch, chunk, 7]"
        )
    if padding.shape != prediction.shape[:2]:
        raise ValueError("action_is_pad must have shape [batch, chunk]")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("action diagnostics require finite values")
    valid = ~padding
    if not np.any(valid):
        raise ValueError("action diagnostics require at least one valid action")
    errors = np.abs(prediction - truth)
    translation_prediction = prediction[..., :3][valid]
    translation_truth = truth[..., :3][valid]
    denominator = np.linalg.norm(translation_prediction, axis=1) * np.linalg.norm(
        translation_truth, axis=1
    )
    directional = denominator > 1e-8
    cosines = np.divide(
        np.sum(translation_prediction * translation_truth, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=directional,
    )
    first_valid = ~padding[:, 0]
    if not np.any(first_valid):
        raise ValueError("action diagnostics require a valid first action")
    first_errors = errors[first_valid, 0]
    first_prediction = prediction[first_valid, 0, :3]
    first_truth = truth[first_valid, 0, :3]
    first_denominator = np.linalg.norm(first_prediction, axis=1) * np.linalg.norm(
        first_truth, axis=1
    )
    first_directional = first_denominator > 1e-8
    first_cosines = np.divide(
        np.sum(first_prediction * first_truth, axis=1),
        first_denominator,
        out=np.zeros_like(first_denominator),
        where=first_directional,
    )
    first_sign = np.sign(first_prediction) == np.sign(first_truth)
    return {
        "valid_actions": int(valid.sum()),
        "translation_mae_x": float(errors[..., 0][valid].mean()),
        "translation_mae_y": float(errors[..., 1][valid].mean()),
        "translation_mae_z": float(errors[..., 2][valid].mean()),
        "rotation_mae": float(errors[..., 3:6][valid].mean()),
        "gripper_mae": float(errors[..., 6][valid].mean()),
        "first_translation_mae_x": float(first_errors[:, 0].mean()),
        "first_translation_mae_y": float(first_errors[:, 1].mean()),
        "first_translation_mae_z": float(first_errors[:, 2].mean()),
        "first_rotation_mae": float(first_errors[:, 3:6].mean()),
        "first_gripper_mae": float(first_errors[:, 6].mean()),
        "translation_direction_cosine": float(cosines[directional].mean())
        if np.any(directional)
        else 0.0,
        "first_translation_direction_cosine": float(
            first_cosines[first_directional].mean()
        )
        if np.any(first_directional)
        else 0.0,
        "first_translation_sign_accuracy": float(first_sign.mean()),
    }


def partition_action_metrics(
    predicted: object,
    target: object,
    action_is_pad: object,
    phase_ids: object,
) -> dict[str, Any]:
    prediction = np.asarray(predicted, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    padding = np.asarray(action_is_pad, dtype=np.bool_)
    phases = np.asarray(phase_ids, dtype=np.int64)
    if prediction.ndim != 3 or phases.shape != (prediction.shape[0],):
        raise ValueError("phase IDs must have shape [batch]")
    if np.any((phases < 0) | (phases >= len(PHASE_NAMES))):
        raise ValueError("phase IDs contain an unknown interaction phase")
    report = {
        "overall": action_chunk_metrics(prediction, truth, padding),
        "by_phase": {},
    }
    by_phase = report["by_phase"]
    assert isinstance(by_phase, dict)
    for phase_id, phase_name in enumerate(PHASE_NAMES):
        selected = phases == phase_id
        if np.any(selected):
            by_phase[phase_name] = action_chunk_metrics(
                prediction[selected],
                truth[selected],
                padding[selected],
            )
    return report


def phase_lookup_from_manifest(root: str | Path) -> dict[tuple[int, int], int]:
    dataset_root = Path(root)
    manifest_path = dataset_root / "meta" / "teacher_manifest.json"
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid teacher manifest: {manifest_path}") from error
    if not isinstance(records, list):
        raise ValueError("teacher manifest must be a JSON list")
    lookup: dict[tuple[int, int], int] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("teacher manifest records must be JSON objects")
        episode = int(record.get("episode_index", -1))
        frames = int(record.get("frames", -1))
        relative_path = record.get("path")
        digest = record.get("sha256")
        if episode < 0 or frames < 1 or not isinstance(relative_path, str):
            raise ValueError("teacher manifest phase metadata is invalid")
        if not isinstance(digest, str):
            raise ValueError("teacher manifest sidecar hash is invalid")
        arrays = load_teacher_sidecar(
            dataset_root / relative_path,
            expected_sha256=digest,
        )
        phases = causal_phase_ids(arrays["annotation.tc_tig.relation_values"])
        if len(phases) != frames:
            raise ValueError("teacher phase count differs from its manifest")
        for frame, phase in enumerate(phases):
            key = (episode, frame)
            if key in lookup:
                raise ValueError("teacher phase lookup contains duplicate frames")
            lookup[key] = int(phase)
    return lookup


def _evaluate_partition(
    dataset: Any,
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    batch_size: int,
    phase_lookup: dict[tuple[int, int], int],
    partition: str,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    if hasattr(policy, "reset"):
        policy.reset()
    policy.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    paddings: list[np.ndarray] = []
    phases: list[int] = []
    for raw_batch in loader:
        _audit_batch(raw_batch, stage=f"{partition} diagnostics raw")
        processed = preprocessor(raw_batch)
        _audit_batch(processed, stage=f"{partition} diagnostics processed")
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(processed)
            predicted = postprocessor(normalized)
        if not isinstance(predicted, torch.Tensor):
            raise ValueError("ACT action diagnostics require tensor predictions")
        predictions.append(predicted.detach().cpu().numpy())
        targets.append(raw_batch["action"].detach().cpu().numpy())
        paddings.append(raw_batch["action_is_pad"].detach().cpu().numpy())
        episode_ids = raw_batch["episode_index"].detach().cpu().reshape(-1)
        frame_ids = raw_batch["frame_index"].detach().cpu().reshape(-1)
        if len(episode_ids) != len(frame_ids):
            raise ValueError("diagnostic episode/frame IDs have different lengths")
        for episode, frame in zip(episode_ids.tolist(), frame_ids.tolist(), strict=True):
            key = (int(episode), int(frame))
            if key not in phase_lookup:
                raise ValueError(f"missing causal phase label for episode/frame {key}")
            phases.append(phase_lookup[key])
    if not predictions:
        raise ValueError(f"{partition} action diagnostics dataset is empty")
    return partition_action_metrics(
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(paddings, axis=0),
        np.asarray(phases, dtype=np.int64),
    )


def evaluate_checkpoint_actions(
    config_path: str | Path,
    checkpoint: str | Path,
) -> dict[str, Any]:
    config = load_bridge_config(config_path)
    if config.recovery is None:
        raise ValueError("bridge config does not define recovery diagnostics")
    validate_dataset_root(
        config.dataset.root,
        repo_id=config.dataset.repo_id,
        allow_incomplete=False,
        require_bridge_metadata=True,
        replay=True,
        bridge_config=config,
        require_collection_identity=False,
    )
    split = pilot_episode_split(
        total_episodes=config.dataset.episodes,
        seed=config.act.seed,
    )
    datasets = {
        partition: load_act_dataset(
            dataset_root=config.dataset.root,
            repo_id=config.dataset.repo_id,
            episodes=split[partition],
        )
        for partition in ("train", "validation")
    }
    device = resolve_device(config.act.device)
    checkpoint_path = Path(checkpoint)
    policy, preprocessor, postprocessor, _ = _load_checkpoint_bundle(
        config=config,
        checkpoint=checkpoint_path,
        device=device,
    )
    phase_lookup = phase_lookup_from_manifest(config.dataset.root)
    report: dict[str, Any] = {
        "checkpoint": checkpoint_path.as_posix(),
        "episode_split": split,
    }
    for partition, dataset in datasets.items():
        report[partition] = _evaluate_partition(
            dataset,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            batch_size=config.act.batch_size,
            phase_lookup=phase_lookup,
            partition=partition,
        )
    destination = config.recovery.output_dir / "action_diagnostics.json"
    _write_json_atomic(destination, report)
    report["path"] = destination.as_posix()
    return report
