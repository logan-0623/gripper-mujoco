from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from interaction_vla.graph_finetune.schema import SCHEMA_VERSION as GRAPH_SCHEMA_VERSION
from interaction_vla.lerobot_bridge.act_smoke import (
    ACTION_CODEC_VERSION,
    STATE_CODEC_VERSION,
    _initial_state_hash,
    _jsonable,
    _optimizer_update,
    _seed_all,
    _validation_loss,
    _write_checkpoint_metadata,
    bounded_batches,
    build_act_bundle_from_dataset,
)
from interaction_vla.lerobot_bridge.config import BridgeConfig
from interaction_vla.lerobot_bridge.provenance import (
    fingerprint_tree,
    sha256_file,
    source_fingerprint,
)
from interaction_vla.lerobot_bridge.rollout import load_act_runtime

from .cache import TokenCache
from .dataset import GraphConditionedDataset
from .schema import CONDITIONS, TOKEN_DIM


ACT_GRAPH_SCHEMA_VERSION = "graph_conditioned_act_v1"
_PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class ControlSplit:
    path: Path
    episodes: dict[str, tuple[int, ...]]
    rows: dict[str, tuple[int, ...]]
    split_seed: int
    sha256: str


def _partition_mapping(value: object, name: str) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(_PARTITIONS):
        raise ValueError(f"split {name} must contain train, validation, and test")
    result: dict[str, tuple[int, ...]] = {}
    combined: list[int] = []
    for partition in _PARTITIONS:
        raw = value[partition]
        if not isinstance(raw, list):
            raise ValueError(f"split {name}.{partition} must be a list")
        items = tuple(int(item) for item in raw)
        if not items or any(item < 0 for item in items):
            raise ValueError(f"split {name}.{partition} must be non-empty and non-negative")
        if len(set(items)) != len(items):
            raise ValueError(f"split {name}.{partition} must be unique")
        result[partition] = items
        combined.extend(items)
    if len(set(combined)) != len(combined):
        raise ValueError(f"split {name} overlaps across partitions")
    return result


def load_control_split(path: str | Path) -> ControlSplit:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Graph fine-tune split manifest: {source}") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "split_seed",
        "episode_indices",
        "row_indices",
    }:
        raise ValueError("Graph fine-tune split manifest fields are incompatible")
    if payload["schema_version"] != GRAPH_SCHEMA_VERSION:
        raise ValueError("Graph fine-tune split schema is incompatible")
    split_seed = int(payload["split_seed"])
    if split_seed < 0:
        raise ValueError("Graph fine-tune split seed must be non-negative")
    return ControlSplit(
        path=source,
        episodes=_partition_mapping(payload["episode_indices"], "episode_indices"),
        rows=_partition_mapping(payload["row_indices"], "row_indices"),
        split_seed=split_seed,
        sha256=sha256_file(source),
    )


def assert_checkpoint_split(
    payload: Mapping[str, Any],
    split: ControlSplit,
    *,
    condition: str,
    seed: int,
) -> None:
    if condition not in CONDITIONS or condition == "flat":
        raise ValueError("checkpoint split validation requires a predicted condition")
    expected_initialization = (
        "random_init" if condition == "predicted_random" else "reflectvlm_init"
    )
    expected = {
        "split_seed": split.split_seed,
        "initialization": expected_initialization,
        "fraction": 1.0,
        "seed": int(seed),
        "selected_train_episodes": list(split.episodes["train"]),
        "train_row_indices": list(split.rows["train"]),
        "validation_row_indices": list(split.rows["validation"]),
        "test_row_indices": list(split.rows["test"]),
    }
    differing = [name for name, value in expected.items() if payload.get(name) != value]
    if differing:
        raise ValueError("Graph checkpoint split mismatch: " + ", ".join(differing))


def _control_source_files(repository: Path) -> tuple[Path, ...]:
    package_root = repository / "interaction_vla"
    files = tuple(
        sorted(
            path
            for path in package_root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    )
    if not files:
        raise FileNotFoundError(f"Graph control source tree is empty: {package_root}")
    return files


def _control_source_fingerprint() -> str:
    repository = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in _control_source_files(repository):
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _graph_bindings(condition: str, seed: int, cache: TokenCache) -> dict[str, object]:
    return {
        "schema_version": ACT_GRAPH_SCHEMA_VERSION,
        "condition": condition,
        "seed": int(seed),
        "token_dim": TOKEN_DIM,
        "cache_sha256": cache.sha256,
        "cache_provenance": asdict(cache.provenance),
        "source_fingerprint": _control_source_fingerprint(),
        "state_codec_version": STATE_CODEC_VERSION,
        "action_codec_version": ACTION_CODEC_VERSION,
    }


def graph_checkpoint_bindings(
    condition: str, seed: int, cache: TokenCache
) -> dict[str, object]:
    return _graph_bindings(condition, seed, cache)


def expected_graph_checkpoint_metadata(
    *,
    dataset_root: str | Path,
    features: Mapping[str, object],
    act_config: object,
    device: torch.device,
    bindings: Mapping[str, object],
) -> dict[str, object]:
    from lerobot.configs import FeatureType
    from lerobot.utils.feature_utils import dataset_to_policy_features

    policy_features = dataset_to_policy_features(dict(features))
    act_config.output_features = {
        key: feature
        for key, feature in policy_features.items()
        if feature.type is FeatureType.ACTION
    }
    act_config.input_features = {
        key: feature
        for key, feature in policy_features.items()
        if key not in act_config.output_features
    }
    return _jsonable(
        {
            "dataset_fingerprint": fingerprint_tree(dataset_root),
            "features": features,
            "state_codec_version": STATE_CODEC_VERSION,
            "action_codec_version": ACTION_CODEC_VERSION,
            "lerobot_version": importlib.metadata.version("lerobot"),
            "act_config": act_config,
            "device": str(device),
            "source_fingerprint": source_fingerprint(),
            "graph_control": dict(bindings),
        }
    )


def load_graph_act_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device,
    expected_metadata: Mapping[str, object],
):
    source = Path(checkpoint)
    metadata_path = source / "bridge_checkpoint.json"
    if not metadata_path.is_file():
        raise ValueError(f"Graph-conditioned ACT metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Graph-conditioned ACT metadata is invalid") from error
    if not isinstance(metadata.get("graph_control"), Mapping):
        raise ValueError("Graph-conditioned ACT binding is missing")
    expected = _jsonable(dict(expected_metadata))
    training_device = metadata.get("device")
    actual_act_config = metadata.get("act_config")
    if training_device not in {"cpu", "mps"} or not isinstance(
        actual_act_config, Mapping
    ):
        raise ValueError("Graph-conditioned ACT checkpoint device metadata is invalid")
    if actual_act_config.get("device") != training_device:
        raise ValueError(
            "Graph-conditioned ACT checkpoint metadata mismatch: act_config, device"
        )
    expected_act_config = expected.get("act_config")
    if not isinstance(expected_act_config, Mapping):
        raise ValueError("expected Graph-conditioned ACT config metadata is invalid")
    expected["device"] = training_device
    expected["act_config"] = {**expected_act_config, "device": training_device}
    differing = sorted(
        name
        for name in set(metadata) | set(expected)
        if metadata.get(name) != expected.get(name)
    )
    if differing:
        raise ValueError(
            "Graph-conditioned ACT checkpoint metadata mismatch: "
            + ", ".join(differing)
        )
    policy, preprocessor, postprocessor = load_act_runtime(source, device=device)
    return policy, preprocessor, postprocessor, metadata


def _save_condition(
    *,
    bundle: Any,
    output_dir: Path,
    dataset_root: Path,
    device: torch.device,
    summary: dict[str, object],
    bindings: dict[str, object],
    reload_batch: dict[str, Any],
) -> float:
    checkpoint = output_dir / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=False)
    bundle.policy.eval()
    with torch.no_grad():
        expected = bundle.policy.predict_action_chunk(
            bundle.preprocessor(reload_batch)
        ).detach().cpu()
    bundle.policy.save_pretrained(checkpoint, push_to_hub=False)
    bundle.preprocessor.save_pretrained(checkpoint, push_to_hub=False)
    bundle.postprocessor.save_pretrained(checkpoint, push_to_hub=False)
    _write_checkpoint_metadata(
        checkpoint,
        bundle=bundle,
        dataset_root=dataset_root,
        device=device,
        extra={"graph_control": bindings},
    )
    expected_metadata = expected_graph_checkpoint_metadata(
        dataset_root=dataset_root,
        features=bundle.dataset.features,
        act_config=bundle.config,
        device=device,
        bindings=bindings,
    )
    policy, preprocessor, _, _ = load_graph_act_checkpoint(
        checkpoint,
        device=device,
        expected_metadata=expected_metadata,
    )
    policy.eval()
    with torch.no_grad():
        actual = policy.predict_action_chunk(preprocessor(reload_batch)).detach().cpu()
    error = float(torch.max(torch.abs(expected - actual)).item())
    if not np.isfinite(error) or error > 1e-5:
        raise ValueError(f"Graph-conditioned ACT reload error is too large: {error}")
    (checkpoint / "training_summary.json").write_text(
        json.dumps(_jsonable({**summary, "reload_max_abs_error": error}), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return error


def _train_condition(
    *,
    condition: str,
    train_dataset: Any,
    validation_dataset: Any,
    cache: TokenCache,
    seed: int,
    output_dir: Path,
    dataset_root: Path,
    device: torch.device,
    architecture: str,
    bridge_config: BridgeConfig | None,
    batch_size: int,
    smoke_steps: int | None,
    initial_epochs: int | None,
) -> dict[str, object]:
    _seed_all(seed)
    conditioned_train = GraphConditionedDataset(train_dataset, cache)
    conditioned_validation = GraphConditionedDataset(validation_dataset, cache)
    bundle = build_act_bundle_from_dataset(
        conditioned_train,
        device=device,
        architecture=architecture,
        bridge_config=bridge_config,
    )
    loader = DataLoader(
        conditioned_train,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        conditioned_validation,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        bundle.policy.get_optim_params(),
        lr=bundle.config.optimizer_lr,
        weight_decay=bundle.config.optimizer_weight_decay,
    )
    initial_hash = _initial_state_hash(bundle.policy)
    parameter_count = sum(parameter.numel() for parameter in bundle.policy.parameters())
    metrics: list[dict[str, object]] = []
    validation_losses: list[float] = []
    epochs_completed = 0
    if smoke_steps is not None:
        for step, raw_batch in bounded_batches(lambda: iter(loader), steps=smoke_steps):
            metric = _optimizer_update(
                bundle=bundle, optimizer=optimizer, raw_batch=raw_batch
            )
            metric["step"] = step
            metrics.append(metric)
    else:
        assert initial_epochs is not None
        for epoch in range(initial_epochs):
            for raw_batch in loader:
                metric = _optimizer_update(
                    bundle=bundle, optimizer=optimizer, raw_batch=raw_batch
                )
                metric["step"] = len(metrics)
                metric["epoch"] = epoch
                metrics.append(metric)
            validation_losses.append(
                _validation_loss(bundle=bundle, loader=validation_loader)
            )
            epochs_completed += 1
    try:
        reload_batch = next(iter(loader))
    except StopIteration as error:
        raise ValueError("Graph-conditioned ACT training loader is empty") from error
    source_rows = [
        int(row)
        for metric in metrics
        for row in metric.get("source_row_indices", [])
    ]
    summary: dict[str, object] = {
        "condition": condition,
        "seed": seed,
        "steps": len(metrics),
        "epochs": epochs_completed,
        "losses": [metric["loss"] for metric in metrics],
        "metrics": metrics,
        "validation_losses": validation_losses,
        "extension_decisions": [],
        "initial_state_hash": initial_hash,
        "parameter_count": parameter_count,
        "source_row_indices": source_rows,
        "cache_sha256": cache.sha256,
        "device": str(device),
        "batch_size": batch_size,
    }
    bindings = _graph_bindings(condition, seed, cache)
    reload_error = _save_condition(
        bundle=bundle,
        output_dir=output_dir,
        dataset_root=dataset_root,
        device=device,
        summary=summary,
        bindings=bindings,
        reload_batch=reload_batch,
    )
    summary["reload_max_abs_error"] = reload_error
    return summary


def assert_paired_summaries(summaries: Mapping[str, object]) -> None:
    if set(summaries) != set(CONDITIONS):
        raise ValueError("paired summaries must contain the exact condition matrix")
    typed: dict[str, Mapping[str, object]] = {}
    for condition in CONDITIONS:
        value = summaries[condition]
        if not isinstance(value, Mapping):
            raise ValueError(f"paired summary for {condition} must be a mapping")
        typed[condition] = value
    reference = typed[CONDITIONS[0]]
    paired_fields = (
        "initial_state_hash",
        "parameter_count",
        "source_row_indices",
        "epochs",
        "extension_decisions",
    )
    differing = [
        field
        for field in paired_fields
        if any(typed[condition].get(field) != reference.get(field) for condition in CONDITIONS[1:])
    ]
    if differing:
        raise ValueError("paired ACT summaries differ: " + ", ".join(differing))


def train_paired_seed(
    *,
    train_dataset: Any,
    validation_dataset: Any,
    caches: Mapping[str, TokenCache],
    seed: int,
    output_dir: str | Path,
    dataset_root: str | Path,
    device: torch.device,
    architecture: str,
    batch_size: int,
    smoke_steps: int | None,
    initial_epochs: int | None,
    maximum_epochs: int | None,
    bridge_config: BridgeConfig | None = None,
) -> dict[str, object]:
    if set(caches) != set(CONDITIONS):
        raise ValueError("paired training requires the exact cache condition matrix")
    if batch_size < 1:
        raise ValueError("paired training batch_size must be positive")
    if (smoke_steps is None) == (initial_epochs is None):
        raise ValueError("choose exactly one of smoke_steps or initial_epochs")
    if smoke_steps is not None and smoke_steps < 1:
        raise ValueError("smoke_steps must be positive")
    if initial_epochs is not None:
        if initial_epochs < 1 or maximum_epochs is None or maximum_epochs < initial_epochs:
            raise ValueError("formal epoch bounds are invalid")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"paired ACT output must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    reference_rows = caches[CONDITIONS[0]].row_indices
    dataset_fingerprints = {
        cache.provenance.dataset_fingerprint for cache in caches.values()
    }
    split_hashes = {cache.provenance.split_manifest_sha256 for cache in caches.values()}
    for condition in CONDITIONS:
        cache = caches[condition]
        if cache.provenance.condition != condition:
            raise ValueError(f"cache provenance condition mismatch: {condition}")
        if not np.array_equal(cache.row_indices, reference_rows):
            raise ValueError("paired ACT caches use different global rows")
        if condition != "flat" and cache.provenance.graph_seed != seed:
            raise ValueError(f"cache Graph seed mismatch: {condition}")
    if len(dataset_fingerprints) != 1 or len(split_hashes) != 1:
        raise ValueError("paired ACT caches disagree on dataset or split provenance")

    summaries: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS:
        condition_dir = destination / condition
        condition_dir.mkdir()
        summaries[condition] = _train_condition(
            condition=condition,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            cache=caches[condition],
            seed=seed,
            output_dir=condition_dir,
            dataset_root=Path(dataset_root),
            device=device,
            architecture=architecture,
            bridge_config=bridge_config,
            batch_size=batch_size,
            smoke_steps=smoke_steps,
            initial_epochs=initial_epochs,
        )
        gc.collect()
    assert_paired_summaries(summaries)
    report: dict[str, object] = {
        "passed": True,
        "schema_version": ACT_GRAPH_SCHEMA_VERSION,
        "seed": seed,
        "conditions": list(CONDITIONS),
        "summaries": summaries,
        "fixed_epochs": initial_epochs,
        "maximum_epochs": maximum_epochs,
    }
    (destination / "paired_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
