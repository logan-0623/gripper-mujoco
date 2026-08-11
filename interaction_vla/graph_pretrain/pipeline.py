from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from interaction_vla.device import resolve_device

from .config import GraphPretrainConfig, ReflectModelConfig, load_graph_pretrain_config
from .model import ReflectGraphEstimator, graph_prediction_loss
from .reflectvlm import (
    PreparedCorpus,
    ReflectTorchDataset,
    Vocabulary,
    prepare_corpus,
)
from .schema import SCHEMA_VERSION


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n").encode()
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_finite_loss(loss: Tensor) -> None:
    if loss.numel() != 1 or not torch.isfinite(loss).item():
        raise FloatingPointError("graph pretraining produced a non-finite scalar loss")


def load_source_dataset(config: GraphPretrainConfig):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "ReflectVLM commands require the LeRobot environment with datasets installed"
        ) from error
    kwargs: dict[str, Any] = {
        "path": config.dataset.repo_id,
        "split": config.dataset.source_split,
    }
    if config.dataset.cache_dir is not None:
        kwargs["cache_dir"] = str(config.dataset.cache_dir)
    return load_dataset(**kwargs)


def _prepare(config: GraphPretrainConfig, source: Any) -> PreparedCorpus:
    return prepare_corpus(
        source,
        split_seed=config.dataset.split_seed,
        ratios=config.dataset.split_ratios,
        max_rows=config.dataset.max_rows,
    )


def _partition_group_counts(corpus: PreparedCorpus) -> dict[str, int]:
    return {
        name: len({corpus.metadata[index].group for index in indices})
        for name, indices in corpus.splits.items()
    }


def inspect_with_source(config: GraphPretrainConfig, source: Any) -> dict[str, Any]:
    corpus = _prepare(config, source)
    groups = {metadata.group for metadata in corpus.metadata}
    partition_groups = {
        name: {corpus.metadata[index].group for index in indices}
        for name, indices in corpus.splits.items()
    }
    overlap = any(
        partition_groups[left] & partition_groups[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    report = {
        "passed": not overlap,
        "schema_version": SCHEMA_VERSION,
        "repo_id": config.dataset.repo_id,
        "source_split": config.dataset.source_split,
        "rows": len(corpus.metadata),
        "groups": len(groups),
        "partition_rows": {name: len(indices) for name, indices in corpus.splits.items()},
        "partition_groups": {
            name: len(values) for name, values in partition_groups.items()
        },
        "vocabulary_size": len(corpus.vocabulary.tokens),
        "group_overlap": overlap,
    }
    if overlap:
        raise ValueError("grouped split leaked a board/environment group")
    return report


def inspect_from_config(path: str | Path) -> dict[str, Any]:
    config = load_graph_pretrain_config(path)
    return inspect_with_source(config, load_source_dataset(config))


def _model(config: ReflectModelConfig, vocab_size: int) -> ReflectGraphEstimator:
    return ReflectGraphEstimator(
        vocab_size=vocab_size,
        image_embedding_dim=config.image_embedding_dim,
        text_embedding_dim=config.text_embedding_dim,
        graph_embedding_dim=config.graph_embedding_dim,
    )


def _loader(
    config: GraphPretrainConfig,
    corpus: PreparedCorpus,
    partition: str,
    *,
    shuffle: bool,
) -> DataLoader[dict[str, Tensor]]:
    dataset = ReflectTorchDataset(
        corpus,
        partition=partition,
        image_size=config.model.image_size,
        max_history_tokens=config.model.max_history_tokens,
    )
    generator = torch.Generator().manual_seed(config.training.seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        generator=generator if shuffle else None,
    )


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _evaluate_loader(
    model: ReflectGraphEstimator,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    examples = 0
    loss_sum = 0.0
    target_correct = in_hand_correct = phase_correct = goal_correct = 0
    state_correct = upright_correct = object_total = 0
    dependency_tp = dependency_fp = dependency_fn = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            outputs = model(
                batch["image"], batch["history_tokens"], batch["history_mask"]
            )
            loss = graph_prediction_loss(outputs, batch)["total"]
            require_finite_loss(loss)
            batch_size = int(batch["image"].shape[0])
            examples += batch_size
            loss_sum += float(loss) * batch_size
            target_correct += int(
                (outputs["target_logits"].argmax(-1) == batch["target_index"]).sum()
            )
            in_hand_correct += int(
                (outputs["in_hand_logits"].argmax(-1) == batch["in_hand_index"]).sum()
            )
            object_mask = batch["object_mask"].bool()
            state_correct += int(
                (
                    (outputs["state_logits"].argmax(-1) == batch["state_ids"])
                    & object_mask
                ).sum()
            )
            upright_correct += int(
                (
                    (outputs["upright_logits"].argmax(-1) == batch["upright_ids"])
                    & object_mask
                ).sum()
            )
            object_total += int(object_mask.sum())
            phase_correct += int(
                (outputs["phase_logits"].argmax(-1) == batch["phase_id"]).sum()
            )
            operator_correct = outputs["operator_logits"].argmax(-1) == batch[
                "goal_operator_id"
            ]
            predicate_correct = outputs["predicate_logits"].argmax(-1) == batch[
                "goal_predicate_id"
            ]
            goal_correct += int((operator_correct & predicate_correct).sum())
            mask = batch["dependency_mask"].bool()
            predicted = outputs["dependency_logits"] > 0.0
            truth = batch["dependency"] > 0.5
            dependency_tp += int((predicted & truth & mask).sum())
            dependency_fp += int((predicted & ~truth & mask).sum())
            dependency_fn += int((~predicted & truth & mask).sum())
    if examples == 0:
        raise ValueError("evaluation partition is empty")
    precision = dependency_tp / max(1, dependency_tp + dependency_fp)
    recall = dependency_tp / max(1, dependency_tp + dependency_fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "examples": examples,
        "mean_loss": loss_sum / examples,
        "target_accuracy": target_correct / examples,
        "in_hand_accuracy": in_hand_correct / examples,
        "state_accuracy": state_correct / object_total,
        "upright_accuracy": upright_correct / object_total,
        "dependency_precision": precision,
        "dependency_recall": recall,
        "dependency_f1": f1,
        "phase_accuracy": phase_correct / examples,
        "goal_exact_accuracy": goal_correct / examples,
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_with_source(config: GraphPretrainConfig, source: Any) -> dict[str, Any]:
    _seed_all(config.training.seed)
    device = resolve_device(config.training.device)
    corpus = _prepare(config, source)
    model = _model(config.model, len(corpus.vocabulary.tokens)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_loader = _loader(config, corpus, "train", shuffle=True)
    validation_loader = _loader(config, corpus, "validation", shuffle=False)
    history: list[dict[str, float | int]] = []
    steps = 0
    best_validation_loss = math.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    for epoch in range(config.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_examples = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            outputs = model(
                batch["image"], batch["history_tokens"], batch["history_mask"]
            )
            loss = graph_prediction_loss(outputs, batch)["total"]
            require_finite_loss(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(batch["image"].shape[0])
            epoch_loss += float(loss.detach()) * count
            epoch_examples += count
            steps += 1
        validation = _evaluate_loader(model, validation_loader, device)
        mean_training_loss = epoch_loss / max(1, epoch_examples)
        history.append(
            {
                "epoch": epoch,
                "training_loss": mean_training_loss,
                "validation_loss": float(validation["mean_loss"]),
            }
        )
        if float(validation["mean_loss"]) < best_validation_loss:
            best_validation_loss = float(validation["mean_loss"])
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None or not math.isfinite(best_validation_loss):
        raise FloatingPointError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    output_dir = config.training.output_dir
    checkpoint = output_dir / "checkpoint.pt"
    checkpoint_payload = {
        "schema_version": SCHEMA_VERSION,
        "repo_id": config.dataset.repo_id,
        "source_split": config.dataset.source_split,
        "split_seed": config.dataset.split_seed,
        "model_config": asdict(config.model),
        "vocabulary": list(corpus.vocabulary.tokens),
        "model_state": best_state,
    }
    _save_checkpoint_atomic(checkpoint, checkpoint_payload)
    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "split_seed": config.dataset.split_seed,
        "partition_rows": {name: len(values) for name, values in corpus.splits.items()},
        "partition_groups": _partition_group_counts(corpus),
        "row_indices": corpus.splits,
    }
    summary = {
        "passed": True,
        "checkpoint": checkpoint.as_posix(),
        "device": device.type,
        "epochs": config.training.epochs,
        "steps": steps,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "history": history,
        "vocabulary_size": len(corpus.vocabulary.tokens),
        "partition_rows": split_manifest["partition_rows"],
    }
    _write_json_atomic(output_dir / "split_manifest.json", split_manifest)
    _write_json_atomic(output_dir / "training_summary.json", summary)
    return summary


def train_from_config(path: str | Path) -> dict[str, Any]:
    config = load_graph_pretrain_config(path)
    return train_with_source(config, load_source_dataset(config))


def load_checkpoint(
    path: str | Path, *, device: str | torch.device
) -> tuple[ReflectGraphEstimator, Vocabulary, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"graph checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("graph checkpoint schema is missing or incompatible")
    required = {"model_config", "vocabulary", "model_state"}
    missing = required - set(payload)
    if missing:
        raise ValueError("graph checkpoint is missing: " + ", ".join(sorted(missing)))
    model_config = ReflectModelConfig(**payload["model_config"])
    vocabulary = Vocabulary(tuple(str(token) for token in payload["vocabulary"]))
    resolved_device = torch.device(device)
    model = _model(model_config, len(vocabulary.tokens)).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, vocabulary, payload


def evaluate_with_source(
    config: GraphPretrainConfig,
    source: Any,
    checkpoint: str | Path,
    *,
    partition: str = "test",
) -> dict[str, Any]:
    if partition not in {"validation", "test"}:
        raise ValueError("evaluation partition must be validation or test")
    device = resolve_device(config.training.device)
    model, vocabulary, payload = load_checkpoint(checkpoint, device=device)
    if payload.get("repo_id") != config.dataset.repo_id or payload.get(
        "source_split"
    ) != config.dataset.source_split or int(payload.get("split_seed", -1)) != int(
        config.dataset.split_seed
    ):
        raise ValueError("graph checkpoint dataset or split provenance is incompatible")
    corpus = replace(_prepare(config, source), vocabulary=vocabulary)
    loader = _loader(config, corpus, partition, shuffle=False)
    metrics = _evaluate_loader(model, loader, device)
    report: dict[str, Any] = {
        "passed": True,
        "schema_version": SCHEMA_VERSION,
        "checkpoint": Path(checkpoint).as_posix(),
        "partition": partition,
        **metrics,
    }
    _write_json_atomic(config.training.output_dir / "evaluation.json", report)
    return report


def evaluate_from_config(
    path: str | Path,
    checkpoint: str | Path,
    *,
    partition: str = "test",
) -> dict[str, Any]:
    config = load_graph_pretrain_config(path)
    return evaluate_with_source(
        config,
        load_source_dataset(config),
        checkpoint,
        partition=partition,
    )
