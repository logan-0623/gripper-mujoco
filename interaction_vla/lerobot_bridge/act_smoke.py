from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import importlib.metadata
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from interaction_vla.device import resolve_device
from interaction_vla.lerobot_bridge.config import BridgeConfig, load_bridge_config
from interaction_vla.lerobot_bridge.provenance import (
    fingerprint_tree,
    sha256_file,
    source_fingerprint,
)
from interaction_vla.lerobot_bridge.teacher_schema import FORBIDDEN_FIELD_FRAGMENTS
from interaction_vla.lerobot_bridge.validator import validate_dataset_root


STATE_CODEC_VERSION = "ee_position_rotation6d_aperture_v1"
ACTION_CODEC_VERSION = "local_translation_world_rotvec_binary_gripper_v1"
FORBIDDEN_BATCH_FRAGMENTS = (
    "annotation",
    "depth",
    "segmentation",
) + FORBIDDEN_FIELD_FRAGMENTS


@dataclass(frozen=True)
class ACTCheckResult:
    loss: float
    gradient_norm: float
    reload_max_abs_error: float
    checkpoint: Path


@dataclass
class ACTBundle:
    dataset: Any
    config: Any
    policy: Any
    preprocessor: Any
    postprocessor: Any


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_act_dataset(*, dataset_root: str | Path, repo_id: str):
    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo_id, root=Path(dataset_root))
    delta_timestamps = {
        "action": [index / metadata.fps for index in range(8)],
    }
    return LeRobotDataset(
        repo_id,
        root=Path(dataset_root),
        delta_timestamps=delta_timestamps,
    )


def _act_config(
    *,
    device: torch.device,
    architecture: str,
    bridge_config: BridgeConfig | None = None,
):
    from lerobot.policies.act.configuration_act import ACTConfig

    if architecture == "test":
        dim_model = 64
        dim_feedforward = 128
        encoder_layers = 1
        vae_encoder_layers = 1
    elif architecture == "configured" and bridge_config is not None:
        dim_model = bridge_config.act.dim_model
        dim_feedforward = bridge_config.act.dim_feedforward
        encoder_layers = bridge_config.act.encoder_layers
        vae_encoder_layers = bridge_config.act.vae_encoder_layers
    else:
        raise ValueError("architecture must be test or configured with a bridge config")
    return ACTConfig(
        device=device.type,
        push_to_hub=False,
        chunk_size=8,
        n_action_steps=8,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        dim_model=dim_model,
        dim_feedforward=dim_feedforward,
        n_encoder_layers=encoder_layers,
        n_vae_encoder_layers=vae_encoder_layers,
        use_vae=True,
    )


def build_act_bundle(
    *,
    dataset_root: str | Path,
    repo_id: str,
    device: torch.device,
    architecture: str,
    bridge_config: BridgeConfig | None = None,
) -> ACTBundle:
    from lerobot.policies import make_policy, make_pre_post_processors

    dataset = load_act_dataset(dataset_root=dataset_root, repo_id=repo_id)
    config = _act_config(
        device=device,
        architecture=architecture,
        bridge_config=bridge_config,
    )
    policy = make_policy(config, ds_meta=dataset.meta).to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=dataset.meta.stats,
    )
    return ACTBundle(dataset, config, policy, preprocessor, postprocessor)


def _audit_batch(batch: dict[str, Any], *, stage: str) -> None:
    forbidden = sorted(
        key
        for key in batch
        if any(fragment in key.lower() for fragment in FORBIDDEN_BATCH_FRAGMENTS)
    )
    if forbidden:
        raise ValueError(f"{stage} batch contains forbidden teacher keys: {forbidden}")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_checkpoint_metadata(
    path: Path,
    *,
    bundle: ACTBundle,
    dataset_root: Path,
    device: torch.device,
    extra: dict[str, object] | None,
) -> None:
    payload: dict[str, object] = {
        "dataset_fingerprint": fingerprint_tree(dataset_root),
        "features": bundle.dataset.features,
        "state_codec_version": STATE_CODEC_VERSION,
        "action_codec_version": ACTION_CODEC_VERSION,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "act_config": bundle.config,
        "device": str(device),
        "source_fingerprint": source_fingerprint(),
    }
    if extra:
        payload.update(extra)
    destination = path / "bridge_checkpoint.json"
    destination.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_one_batch_check(
    *,
    dataset_root: str | Path,
    repo_id: str,
    output_dir: str | Path,
    device: torch.device,
    batch_size: int,
    seed: int,
    architecture: str,
    bridge_config: BridgeConfig | None = None,
    checkpoint_metadata: dict[str, object] | None = None,
) -> ACTCheckResult:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"ACT checkpoint destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    _seed_all(seed)
    bundle = build_act_bundle(
        dataset_root=dataset_root,
        repo_id=repo_id,
        device=device,
        architecture=architecture,
        bridge_config=bridge_config,
    )
    loader = DataLoader(
        bundle.dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    try:
        raw_batch = next(iter(loader))
    except StopIteration as error:
        raise ValueError("ACT dataset loader is empty") from error
    _audit_batch(raw_batch, stage="raw")
    processed = bundle.preprocessor(raw_batch)
    _audit_batch(processed, stage="processed")
    optimizer = torch.optim.AdamW(
        bundle.policy.get_optim_params(),
        lr=bundle.config.optimizer_lr,
        weight_decay=bundle.config.optimizer_weight_decay,
    )
    bundle.policy.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = bundle.policy.forward(processed)
    if not torch.isfinite(loss):
        raise FloatingPointError("ACT loss is not finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        bundle.policy.parameters(), max_norm=10.0
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("ACT gradient norm is not finite")
    optimizer.step()

    bundle.policy.eval()
    with torch.no_grad():
        expected_chunk = bundle.policy.predict_action_chunk(
            bundle.preprocessor(raw_batch)
        ).detach()
    bundle.policy.save_pretrained(destination, push_to_hub=False)
    bundle.preprocessor.save_pretrained(destination, push_to_hub=False)
    bundle.postprocessor.save_pretrained(destination, push_to_hub=False)
    _write_checkpoint_metadata(
        destination,
        bundle=bundle,
        dataset_root=Path(dataset_root),
        device=device,
        extra=checkpoint_metadata,
    )

    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    reloaded = ACTPolicy.from_pretrained(
        destination,
        local_files_only=True,
    ).to(device)
    reloaded_preprocessor, _ = make_pre_post_processors(
        reloaded.config,
        pretrained_path=str(destination),
    )
    reloaded.eval()
    with torch.no_grad():
        actual_chunk = reloaded.predict_action_chunk(
            reloaded_preprocessor(raw_batch)
        ).detach()
    reload_error = float(torch.max(torch.abs(expected_chunk - actual_chunk)).item())
    if not np.isfinite(reload_error) or reload_error > 1e-5:
        raise ValueError(f"ACT checkpoint reload error is too large: {reload_error}")
    return ACTCheckResult(
        loss=float(loss.detach().item()),
        gradient_norm=float(gradient_norm.detach().item()),
        reload_max_abs_error=reload_error,
        checkpoint=destination,
    )


def check_from_config(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
) -> ACTCheckResult:
    config = load_bridge_config(config_path)
    validate_dataset_root(
        config.dataset.root,
        repo_id=config.dataset.repo_id,
        allow_incomplete=False,
        require_bridge_metadata=True,
        replay=True,
        bridge_config=config,
    )
    device = resolve_device(config.act.device)
    destination = (
        Path(output) if output is not None else config.act.output_dir / "integration_check"
    )
    metadata = {
        "bridge_config_sha256": sha256_file(config.config_path),
        "source_config_sha256": sha256_file(config.source_config_path),
        "expert_gate_sha256": sha256_file(config.expert_gate),
    }
    return run_one_batch_check(
        dataset_root=config.dataset.root,
        repo_id=config.dataset.repo_id,
        output_dir=destination,
        device=device,
        batch_size=config.act.batch_size,
        seed=config.act.seed,
        architecture="configured",
        bridge_config=config,
        checkpoint_metadata=metadata,
    )
