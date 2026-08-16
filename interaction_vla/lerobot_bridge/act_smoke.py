from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
from typing import Any
from urllib.parse import urlparse

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
from interaction_vla.lerobot_bridge.teacher_schema import (
    FORBIDDEN_FIELD_FRAGMENTS,
    SCHEMA_VERSION,
)
from interaction_vla.lerobot_bridge.validator import validate_dataset_root


STATE_CODEC_VERSION = "ee_position_rotation6d_aperture_v1"
ACTION_CODEC_VERSION = "local_translation_world_rotvec_binary_gripper_v1"
POLICY_FEATURE_CONTRACT = {
    "observation.images.agent": {"shape": [3, 256, 256], "dtype": "float32"},
    "observation.images.wrist": {"shape": [3, 256, 256], "dtype": "float32"},
    "observation.state": {"shape": [10], "dtype": "float32"},
    "action": {"shape": [7], "dtype": "float32"},
    "task": {"kind": "language_metadata", "act_model_input": False},
}
FORBIDDEN_BATCH_FRAGMENTS = (
    "annotation",
    "depth",
    "segmentation",
) + FORBIDDEN_FIELD_FRAGMENTS


def expected_smoke_report_contract() -> dict[str, object]:
    try:
        lerobot_version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        lerobot_version = "unavailable"
    return {
        "state_codec_version": STATE_CODEC_VERSION,
        "action_codec_version": ACTION_CODEC_VERSION,
        "schema_version": SCHEMA_VERSION,
        "lerobot_version": lerobot_version,
        "policy_feature_contract": POLICY_FEATURE_CONTRACT,
        "source_fingerprint": source_fingerprint(),
    }


def validate_smoke_report_compatibility(path: str | Path | None) -> None:
    if path is None:
        return
    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"required smoke report not found: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid required smoke report: {report_path}") from error
    expected = {"passed": True, **expected_smoke_report_contract()}
    differing = [key for key, value in expected.items() if report.get(key) != value]
    if differing:
        raise ValueError(
            "required smoke report is stale or incompatible: "
            + ", ".join(differing)
        )


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
    backbone_weights_sha256: str | None = None


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_act_dataset(
    *,
    dataset_root: str | Path,
    repo_id: str,
    episodes: list[int] | None = None,
):
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
        episodes=episodes,
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
    n_action_steps = (
        8
        if architecture == "test"
        else bridge_config.act.n_action_steps
    )
    pretrained_weights = (
        None
        if architecture == "test"
        else bridge_config.act.pretrained_backbone_weights
    )
    return ACTConfig(
        device=device.type,
        push_to_hub=False,
        chunk_size=8,
        n_action_steps=n_action_steps,
        vision_backbone="resnet18",
        pretrained_backbone_weights=pretrained_weights,
        dim_model=dim_model,
        dim_feedforward=dim_feedforward,
        n_encoder_layers=encoder_layers,
        n_vae_encoder_layers=vae_encoder_layers,
        use_vae=True,
        optimizer_lr=(
            bridge_config.act.learning_rate
            if architecture == "configured" and bridge_config is not None
            else 1e-5
        ),
    )


def require_cached_backbone_weights(identifier: str | None) -> Path | None:
    if identifier is None:
        return None
    from torchvision.models import get_weight

    weight = get_weight(identifier)
    filename = Path(urlparse(weight.url).path).name
    cached = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cached.is_file():
        raise FileNotFoundError(
            "ACT pretrained backbone is not cached: "
            f"{identifier}; expected {cached}"
        )
    return cached


def build_act_bundle(
    *,
    dataset_root: str | Path,
    repo_id: str,
    device: torch.device,
    architecture: str,
    bridge_config: BridgeConfig | None = None,
    episodes: list[int] | None = None,
) -> ACTBundle:
    dataset = load_act_dataset(
        dataset_root=dataset_root,
        repo_id=repo_id,
        episodes=episodes,
    )
    return build_act_bundle_from_dataset(
        dataset,
        device=device,
        architecture=architecture,
        bridge_config=bridge_config,
    )


def build_act_bundle_from_dataset(
    dataset: Any,
    *,
    device: torch.device,
    architecture: str,
    bridge_config: BridgeConfig | None = None,
) -> ACTBundle:
    from lerobot.policies import make_policy, make_pre_post_processors

    if not hasattr(dataset, "meta"):
        raise ValueError("ACT dataset must expose metadata")
    config = _act_config(
        device=device,
        architecture=architecture,
        bridge_config=bridge_config,
    )
    cached_weights = require_cached_backbone_weights(
        config.pretrained_backbone_weights
    )
    policy = make_policy(config, ds_meta=dataset.meta).to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        dataset_stats=dataset.meta.stats,
    )
    return ACTBundle(
        dataset,
        config,
        policy,
        preprocessor,
        postprocessor,
        None if cached_weights is None else sha256_file(cached_weights),
    )


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
    pretrained_identifier = getattr(
        bundle.config, "pretrained_backbone_weights", None
    )
    if pretrained_identifier is not None:
        if bundle.backbone_weights_sha256 is None:
            raise ValueError("pretrained ACT bundle is missing its weight archive hash")
        payload.update(
            {
                "pretrained_backbone_weights": pretrained_identifier,
                "backbone_weights_sha256": bundle.backbone_weights_sha256,
            }
        )
    if extra:
        payload.update(extra)
    destination = path / "bridge_checkpoint.json"
    destination.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def checkpoint_reload_max_abs_error(
    *,
    bundle: ACTBundle,
    checkpoint: Path,
    raw_batch: dict[str, Any],
    device: torch.device,
) -> float:
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    bundle.policy.eval()
    with torch.no_grad():
        expected_chunk = bundle.policy.predict_action_chunk(
            bundle.preprocessor(raw_batch)
        ).detach()
    reloaded = ACTPolicy.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to(device)
    reloaded_preprocessor, _ = make_pre_post_processors(
        reloaded.config,
        pretrained_path=str(checkpoint),
    )
    reloaded.eval()
    with torch.no_grad():
        actual_chunk = reloaded.predict_action_chunk(
            reloaded_preprocessor(raw_batch)
        ).detach()
    reload_error = float(
        torch.max(torch.abs(expected_chunk - actual_chunk)).item()
    )
    if not np.isfinite(reload_error) or reload_error > 1e-5:
        raise ValueError(f"ACT checkpoint reload error is too large: {reload_error}")
    return reload_error


def seeded_train_loader(
    dataset: Any, *, batch_size: int, seed: int, shuffle: bool
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=True,
        num_workers=0,
    )


def iter_seeded_batches(
    dataset: Any, *, batch_size: int, seed: int
) -> list[object]:
    return list(
        seeded_train_loader(
            dataset,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
        )
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
    loader = seeded_train_loader(
        bundle.dataset,
        batch_size=batch_size,
        seed=seed,
        shuffle=bool(
            architecture == "configured"
            and bridge_config is not None
            and bridge_config.act.shuffle_train
        ),
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

    reload_error = checkpoint_reload_max_abs_error(
        bundle=bundle,
        checkpoint=destination,
        raw_batch=raw_batch,
        device=device,
    )
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
        require_collection_identity=False,
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


def bounded_batches(loader_factory, *, steps: int):
    if steps < 1:
        raise ValueError("steps must be positive")
    completed = 0
    while completed < steps:
        iterator = iter(loader_factory())
        produced = False
        for batch in iterator:
            produced = True
            yield completed, batch
            completed += 1
            if completed == steps:
                return
        if not produced:
            raise ValueError("ACT dataset loader is empty")


def _is_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def run_training_with_fallback(
    config: Any,
    *,
    batch_size: int,
    **kwargs,
) -> dict[str, object]:
    try:
        result = train_once(config=config, batch_size=batch_size, **kwargs)
        result["batch_size"] = batch_size
        return result
    except RuntimeError as error:
        if batch_size != 2 or not _is_oom(error):
            raise
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
    if (
        torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            pass
    result = train_once(config=config, batch_size=1, **kwargs)
    result["fallback_from_batch_size"] = 2
    result["batch_size"] = 1
    return result


def pilot_episode_split(*, total_episodes: int, seed: int) -> dict[str, list[int]]:
    if total_episodes != 50:
        raise ValueError("the first ACT pilot requires exactly 50 episodes")
    indices = np.random.default_rng(seed).permutation(total_episodes).tolist()
    return {
        "train": indices[:40],
        "validation": indices[40:45],
        "test": indices[45:50],
    }


def _initial_state_hash(policy: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _loss_value(loss_dict: dict[str, Any], key: str) -> float:
    value = loss_dict.get(key, 0.0)
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def optimizer_metric_from_loss_dict(
    *, total_loss: float, loss_dict: dict[str, Any]
) -> dict[str, float]:
    return {
        "loss": float(total_loss),
        "l1_loss": _loss_value(loss_dict, "l1_loss"),
        "kld_loss": _loss_value(loss_dict, "kld_loss"),
    }


def row_order_sha256(rows: list[int]) -> str:
    values = np.asarray(rows, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _optimizer_update(
    *,
    bundle: ACTBundle,
    optimizer: torch.optim.Optimizer,
    raw_batch: dict[str, Any],
) -> dict[str, object]:
    started = time.perf_counter()
    _audit_batch(raw_batch, stage="raw")
    processed = bundle.preprocessor(raw_batch)
    _audit_batch(processed, stage="processed")
    bundle.policy.train()
    optimizer.zero_grad(set_to_none=True)
    loss, loss_dict = bundle.policy.forward(processed)
    if not torch.isfinite(loss):
        raise FloatingPointError("ACT loss is not finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        bundle.policy.parameters(), max_norm=10.0
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("ACT gradient norm is not finite")
    optimizer.step()
    episode_indices = raw_batch.get("episode_index")
    if isinstance(episode_indices, torch.Tensor):
        sources = sorted(
            {int(value) for value in episode_indices.detach().cpu().reshape(-1)}
        )
    else:
        sources = []
    row_indices = raw_batch.get("index")
    if isinstance(row_indices, torch.Tensor):
        source_rows = [
            int(value) for value in row_indices.detach().cpu().reshape(-1)
        ]
    else:
        source_rows = []
    return {
        **optimizer_metric_from_loss_dict(
            total_loss=float(loss.detach().item()),
            loss_dict=loss_dict,
        ),
        "gradient_norm": float(gradient_norm.detach().item()),
        "wall_time_s": time.perf_counter() - started,
        "source_episode_indices": sources,
        "source_row_indices": source_rows,
    }


def _validation_loss(
    *,
    bundle: ACTBundle,
    loader: DataLoader,
) -> float:
    losses: list[float] = []
    bundle.policy.eval()
    with torch.no_grad():
        for raw_batch in loader:
            _audit_batch(raw_batch, stage="validation raw")
            processed = bundle.preprocessor(raw_batch)
            _audit_batch(processed, stage="validation processed")
            loss, _ = bundle.policy.forward(processed)
            if not torch.isfinite(loss):
                raise FloatingPointError("ACT validation loss is not finite")
            losses.append(float(loss.item()))
    if not losses:
        raise ValueError("ACT validation loader is empty")
    return float(np.mean(losses))


def _save_training_checkpoint(
    *,
    bundle: ACTBundle,
    output_dir: Path,
    dataset_root: Path,
    device: torch.device,
    summary: dict[str, object],
    config: BridgeConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.policy.save_pretrained(output_dir, push_to_hub=False)
    bundle.preprocessor.save_pretrained(output_dir, push_to_hub=False)
    bundle.postprocessor.save_pretrained(output_dir, push_to_hub=False)
    _write_checkpoint_metadata(
        output_dir,
        bundle=bundle,
        dataset_root=dataset_root,
        device=device,
        extra={
            "bridge_config_sha256": sha256_file(config.config_path),
            "source_config_sha256": sha256_file(config.source_config_path),
            "expert_gate_sha256": sha256_file(config.expert_gate),
        },
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_once(
    config: BridgeConfig,
    *,
    batch_size: int,
    output_dir: str | Path | None = None,
    architecture: str = "configured",
) -> dict[str, object]:
    _seed_all(config.act.seed)
    device = resolve_device(config.act.device)
    full_dataset = load_act_dataset(
        dataset_root=config.dataset.root,
        repo_id=config.dataset.repo_id,
    )
    splits: dict[str, list[int]] | None = None
    train_episodes: list[int] | None = None
    validation_episodes: list[int] | None = None
    if config.act.epochs is not None:
        splits = pilot_episode_split(
            total_episodes=full_dataset.meta.total_episodes,
            seed=config.act.seed,
        )
        train_episodes = splits["train"]
        validation_episodes = splits["validation"]
    del full_dataset
    bundle = build_act_bundle(
        dataset_root=config.dataset.root,
        repo_id=config.dataset.repo_id,
        device=device,
        architecture=architecture,
        bridge_config=config,
        episodes=train_episodes,
    )
    loader = seeded_train_loader(
        bundle.dataset,
        batch_size=batch_size,
        seed=config.act.seed,
        shuffle=config.act.shuffle_train if architecture == "configured" else False,
    )
    optimizer = torch.optim.AdamW(
        bundle.policy.get_optim_params(),
        lr=bundle.config.optimizer_lr,
        weight_decay=bundle.config.optimizer_weight_decay,
    )
    initial_hash = _initial_state_hash(bundle.policy)
    metrics: list[dict[str, object]] = []
    validation_losses: list[float] = []
    extension_decisions: list[dict[str, object]] = []
    epoch_order_hashes: list[str] = []
    epochs_completed = 0
    reload_batch: dict[str, Any] | None = None

    if config.act.steps is not None:
        for step, raw_batch in bounded_batches(
            lambda: iter(loader), steps=config.act.steps
        ):
            if reload_batch is None:
                reload_batch = raw_batch
            metric = _optimizer_update(
                bundle=bundle,
                optimizer=optimizer,
                raw_batch=raw_batch,
            )
            metric["step"] = step
            metrics.append(metric)
    else:
        assert config.act.epochs is not None
        assert validation_episodes is not None
        validation_dataset = load_act_dataset(
            dataset_root=config.dataset.root,
            repo_id=config.dataset.repo_id,
            episodes=validation_episodes,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        target_epochs = config.act.epochs
        while epochs_completed < target_epochs:
            epoch_rows: list[int] = []
            for raw_batch in loader:
                if reload_batch is None:
                    reload_batch = raw_batch
                metric = _optimizer_update(
                    bundle=bundle,
                    optimizer=optimizer,
                    raw_batch=raw_batch,
                )
                metric["step"] = len(metrics)
                metric["epoch"] = epochs_completed
                metrics.append(metric)
                epoch_rows.extend(int(value) for value in metric["source_row_indices"])
            if not epoch_rows:
                raise ValueError("ACT training epoch contains no source rows")
            epoch_order_hashes.append(row_order_sha256(epoch_rows))
            epochs_completed += 1
            validation_losses.append(
                _validation_loss(bundle=bundle, loader=validation_loader)
            )
            if (
                epochs_completed == target_epochs
                and target_epochs < config.act.maximum_epochs
            ):
                improve = (
                    len(validation_losses) >= 3
                    and validation_losses[-2] <= validation_losses[-3] - 1e-4
                    and validation_losses[-1] <= validation_losses[-2] - 1e-4
                )
                extension_decisions.append(
                    {
                        "after_epoch": epochs_completed,
                        "extended": improve,
                        "recent_validation_losses": validation_losses[-3:],
                    }
                )
                if improve:
                    target_epochs += 1

    summary: dict[str, object] = {
        "steps": len(metrics),
        "epochs": epochs_completed,
        "losses": [metric["loss"] for metric in metrics],
        "metrics": metrics,
        "validation_losses": validation_losses,
        "extension_decisions": extension_decisions,
        "epoch_order_hashes": epoch_order_hashes,
        "initial_state_hash": initial_hash,
        "pretrained_backbone_weights": bundle.config.pretrained_backbone_weights,
        "backbone_weights_sha256": bundle.backbone_weights_sha256,
        "device": str(device),
        "batch_size": batch_size,
        "episode_split": splits,
    }
    if output_dir is not None:
        if reload_batch is None:
            raise ValueError("ACT training produced no checkpoint reload batch")
        checkpoint = Path(output_dir)
        _save_training_checkpoint(
            bundle=bundle,
            output_dir=checkpoint,
            dataset_root=config.dataset.root,
            device=device,
            summary=summary,
            config=config,
        )
        summary["reload_max_abs_error"] = checkpoint_reload_max_abs_error(
            bundle=bundle,
            checkpoint=checkpoint,
            raw_batch=reload_batch,
            device=device,
        )
        (checkpoint / "training_summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _validate_required_smoke_report(config: BridgeConfig) -> None:
    validate_smoke_report_compatibility(config.required_smoke_report)


def train_from_config(
    config_path: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, object]:
    config = load_bridge_config(config_path)
    validate_dataset_root(
        config.dataset.root,
        repo_id=config.dataset.repo_id,
        allow_incomplete=False,
        require_bridge_metadata=True,
        replay=True,
        bridge_config=config,
        require_collection_identity=False,
    )
    _validate_required_smoke_report(config)
    destination = Path(output) if output is not None else config.act.output_dir / "checkpoint"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"ACT training output must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    summary = run_training_with_fallback(
        config,
        batch_size=config.act.batch_size,
        output_dir=destination,
    )
    summary["checkpoint"] = destination
    (destination / "training_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
