from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from interaction_vla.lerobot_bridge.provenance import sha256_file

from ..config import ProbeConfig, RepresentationStudyConfig
from ..extraction import _destination as latent_destination
from ..state_bank.io import load_records, load_split, write_json_atomic
from ..state_bank.schema import StateBankRecord
from ..taps.registry import registered_taps
from .targets import ProbeTarget, probe_targets


PROBE_SCHEMA_VERSION = "interaction_frozen_probe_v2"
FORMAL_PRIMARY_TARGETS = (
    "geometry",
    "phase",
    "recovery_state",
    "recovery_type",
    "next_relation",
)
FORMAL_SECONDARY_TARGETS = ("contact", "stable_grasp")


def v2_probe_targets(
    records: Sequence[Mapping[str, object]],
) -> dict[str, ProbeTarget]:
    """Translate State Bank v2 labels into the fixed formal probe ontology."""
    if not records:
        raise ValueError("formal v2 probe targets require records")
    labels: list[Mapping[str, object]] = []
    for record in records:
        value = record.get("labels")
        if not isinstance(value, Mapping):
            raise ValueError("State Bank v2 record has no label mapping")
        labels.append(value)
    geometry = np.asarray([value["geometry"] for value in labels], dtype=np.float32)
    if geometry.shape != (len(labels), 16) or not np.isfinite(geometry).all():
        raise ValueError("formal geometry targets must be finite 16D rows")

    def categorical(name: str, minimum_classes: int) -> ProbeTarget:
        values = np.asarray([value[name] for value in labels], dtype=np.int64)
        if values.shape != (len(labels),) or np.any(values < 0):
            raise ValueError(f"formal {name} targets must be non-negative labels")
        output_dim = max(minimum_classes, int(values.max()) + 1)
        return ProbeTarget(name, "categorical", values, output_dim)

    next_relation = np.asarray(
        [
            (
                value["next_relation"]["relation_id"],
                value["next_relation"]["operator_id"],
                value["next_relation"]["predicate_id"],
            )
            for value in labels
        ],
        dtype=np.int64,
    )
    if (
        next_relation.shape != (len(labels), 3)
        or np.any(next_relation < 0)
        or np.any(next_relation[:, 0] >= 8)
        or np.any(next_relation[:, 1] >= 5)
        or np.any(next_relation[:, 2] >= 7)
    ):
        raise ValueError("formal next-relation targets are outside the registered ontology")

    return {
        "geometry": ProbeTarget(
            "geometry", "continuous", geometry, geometry.shape[1]
        ),
        "phase": categorical("phase", 6),
        "recovery_state": categorical("recovery_state", 3),
        "recovery_type": categorical("recovery_type", 1),
        "next_relation": ProbeTarget(
            "next_relation", "structured", next_relation, 8 + 5 + 7, (8, 5, 7)
        ),
        "contact": ProbeTarget(
            "contact",
            "binary",
            np.asarray([value["contact"] for value in labels], dtype=np.int64),
            2,
        ),
        "stable_grasp": ProbeTarget(
            "stable_grasp",
            "binary",
            np.asarray(
                [value["stable_grasp"] for value in labels], dtype=np.int64
            ),
            2,
        ),
    }


class ProbeModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, kind: str) -> None:
        super().__init__()
        if kind == "linear":
            self.network = nn.Linear(input_dim, output_dim)
        elif kind == "shallow_mlp":
            hidden = min(256, max(32, input_dim // 2))
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, output_dim)
            )
        else:
            raise ValueError("probe model must be linear or shallow_mlp")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _class_weights(values: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(values.astype(np.int64), minlength=classes).astype(np.float64)
    active = counts > 0
    result = np.ones(classes, dtype=np.float32)
    if np.any(active):
        result[active] = float(counts[active].sum()) / (counts[active] * int(active.sum()))
    return torch.from_numpy(result)


def _loss(logits: torch.Tensor, target: torch.Tensor, spec: ProbeTarget, train_values: np.ndarray) -> torch.Tensor:
    if spec.kind == "continuous":
        return nn.functional.mse_loss(logits, target)
    if spec.kind == "multilabel":
        positives = torch.as_tensor(train_values.sum(axis=0), dtype=torch.float32)
        negatives = float(len(train_values)) - positives
        pos_weight = torch.where(positives > 0, negatives / positives.clamp_min(1.0), torch.ones_like(positives))
        return nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    if spec.kind in {"binary", "categorical"}:
        weights = _class_weights(train_values, spec.output_dim).to(logits.device)
        return nn.functional.cross_entropy(logits, target.long(), weight=weights)
    if spec.kind == "structured":
        cursor = 0
        losses = []
        for column, width in enumerate(spec.head_widths):
            head = logits[:, cursor : cursor + width]
            weights = _class_weights(train_values[:, column], width).to(logits.device)
            losses.append(nn.functional.cross_entropy(head, target[:, column].long(), weight=weights))
            cursor += width
        return torch.stack(losses).mean()
    raise ValueError(f"unknown probe target kind: {spec.kind}")


def _balanced_metrics(predicted: np.ndarray, target: np.ndarray, classes: int) -> dict[str, float]:
    recalls: list[float] = []
    f1s: list[float] = []
    for label in range(classes):
        true = target == label
        guessed = predicted == label
        tp = int(np.sum(true & guessed))
        fn = int(np.sum(true & ~guessed))
        fp = int(np.sum(~true & guessed))
        if tp + fn:
            recalls.append(tp / (tp + fn))
            denominator = 2 * tp + fp + fn
            f1s.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return {
        "accuracy": float(np.mean(predicted == target)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _calibration_metrics(
    probabilities: np.ndarray, target: np.ndarray, *, bins: int = 10
) -> dict[str, float]:
    confidence = np.max(probabilities, axis=1)
    predicted = np.argmax(probabilities, axis=1)
    correct = (predicted == target).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if index == 0:
            selected |= confidence == 0.0
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(
                float(np.mean(correct[selected])) - float(np.mean(confidence[selected]))
            )
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[target]
    return {
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error": float(ece),
    }


def evaluate_probe(logits: np.ndarray, target: np.ndarray, spec: ProbeTarget) -> dict[str, float]:
    if spec.kind == "continuous":
        difference = logits - target
        mae = float(np.mean(np.abs(difference)))
        denominator = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
        r2 = 0.0 if denominator <= 1e-12 else 1.0 - float(np.sum(difference**2)) / denominator
        return {"mae": mae, "r2": r2}
    if spec.kind == "multilabel":
        predicted = logits >= 0.0
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        values = [_balanced_metrics(predicted[:, index].astype(np.int64), target[:, index].astype(np.int64), 2) for index in range(target.shape[1])]
        return {
            "accuracy": float(np.mean(predicted == target)),
            "balanced_accuracy": float(np.mean([value["balanced_accuracy"] for value in values])),
            "macro_f1": float(np.mean([value["macro_f1"] for value in values])),
            "brier_score": float(np.mean((probabilities - target) ** 2)),
            "expected_calibration_error": float(
                np.mean(
                    [
                        _calibration_metrics(
                            np.stack((1.0 - probabilities[:, index], probabilities[:, index]), axis=1),
                            target[:, index].astype(np.int64),
                        )["expected_calibration_error"]
                        for index in range(target.shape[1])
                    ]
                )
            ),
        }
    if spec.kind in {"binary", "categorical"}:
        probabilities = _softmax(logits)
        return {
            **_balanced_metrics(np.argmax(logits, axis=1), target, spec.output_dim),
            **_calibration_metrics(probabilities, target.astype(np.int64)),
        }
    cursor = 0
    predictions = []
    head_metrics = []
    for column, width in enumerate(spec.head_widths):
        predicted = np.argmax(logits[:, cursor : cursor + width], axis=1)
        predictions.append(predicted)
        head_metrics.append(_balanced_metrics(predicted, target[:, column], width))
        cursor += width
    stacked = np.stack(predictions, axis=1)
    calibration = [
        _calibration_metrics(
            _softmax(logits[:, sum(spec.head_widths[:column]) : sum(spec.head_widths[: column + 1])]),
            target[:, column].astype(np.int64),
        )
        for column in range(len(spec.head_widths))
    ]
    return {
        "exact_accuracy": float(np.mean(np.all(stacked == target, axis=1))),
        "balanced_accuracy": float(np.mean([value["balanced_accuracy"] for value in head_metrics])),
        "macro_f1": float(np.mean([value["macro_f1"] for value in head_metrics])),
        "brier_score": float(np.mean([value["brier_score"] for value in calibration])),
        "expected_calibration_error": float(
            np.mean([value["expected_calibration_error"] for value in calibration])
        ),
    }


def constant_baseline_logits(
    training_values: np.ndarray, *, rows: int, spec: ProbeTarget
) -> np.ndarray:
    if rows < 1:
        raise ValueError("constant probe baseline requires positive rows")
    values = np.asarray(training_values)
    if spec.kind == "continuous":
        return np.repeat(values.mean(axis=0, keepdims=True), rows, axis=0)
    if spec.kind == "multilabel":
        probabilities = np.clip(values.mean(axis=0), 1.0e-5, 1.0 - 1.0e-5)
        logits = np.log(probabilities / (1.0 - probabilities))
        return np.repeat(logits[None, :], rows, axis=0)
    if spec.kind in {"binary", "categorical"}:
        counts = np.bincount(values.astype(np.int64), minlength=spec.output_dim)
        probabilities = (counts + 1.0) / (counts.sum() + spec.output_dim)
        return np.repeat(np.log(probabilities)[None, :], rows, axis=0)
    heads: list[np.ndarray] = []
    for column, width in enumerate(spec.head_widths):
        counts = np.bincount(values[:, column].astype(np.int64), minlength=width)
        probabilities = (counts + 1.0) / (counts.sum() + width)
        heads.append(np.log(probabilities))
    return np.repeat(np.concatenate(heads)[None, :], rows, axis=0)


def _fit_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    spec: ProbeTarget,
    *,
    model_kind: str,
    weight_decay: float,
    config: ProbeConfig,
) -> tuple[ProbeModel, float, dict[str, np.ndarray]]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x_train.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5)
    train_x = torch.from_numpy(((x_train - mean) / std).astype(np.float32))
    validation_x = torch.from_numpy(((x_validation - mean) / std).astype(np.float32))
    target_dtype = torch.float32 if spec.kind in {"continuous", "multilabel"} else torch.long
    train_y = torch.as_tensor(y_train, dtype=target_dtype)
    validation_y = torch.as_tensor(y_validation, dtype=target_dtype)
    model = ProbeModel(x_train.shape[1], spec.output_dim, kind=model_kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    for _ in range(config.epochs):
        order = torch.randperm(len(train_x), generator=generator)
        model.train()
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(train_x[indices]), train_y[indices], spec, y_train)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        validation_loss = float(_loss(model(validation_x), validation_y, spec, y_train).item())
    return model, validation_loss, {"mean": mean, "std": std}


def train_single_probe(
    features: np.ndarray,
    target: ProbeTarget,
    partitions: Mapping[str, np.ndarray],
    *,
    model_kind: str,
    config: ProbeConfig,
) -> tuple[ProbeModel, dict[str, object], dict[str, np.ndarray]]:
    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]
    candidates = []
    for weight_decay in config.weight_decays:
        model, validation_loss, normalization = _fit_candidate(
            features[train],
            target.values[train],
            features[validation],
            target.values[validation],
            target,
            model_kind=model_kind,
            weight_decay=weight_decay,
            config=config,
        )
        candidates.append((validation_loss, weight_decay, model, normalization))
    validation_loss, weight_decay, model, normalization = min(candidates, key=lambda value: (value[0], value[1]))
    standardized = (features - normalization["mean"]) / normalization["std"]
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(standardized.astype(np.float32))).numpy()
    metrics = {
        name: evaluate_probe(logits[indices], target.values[indices], target)
        for name, indices in partitions.items()
    }
    baseline_logits = constant_baseline_logits(
        target.values[train], rows=len(features), spec=target
    )
    baseline_metrics = {
        name: evaluate_probe(baseline_logits[indices], target.values[indices], target)
        for name, indices in partitions.items()
    }
    return model, {
        "validation_loss": validation_loss,
        "selected_weight_decay": weight_decay,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "sample_counts": {name: int(len(indices)) for name, indices in partitions.items()},
    }, normalization


def _save_model(path: Path, model: ProbeModel, normalization: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "normalization": {name: value.copy() for name, value in normalization.items()},
            "metadata": dict(metadata),
        },
        temporary,
    )
    os.replace(temporary, path)


def train_probe_suite(
    config: RepresentationStudyConfig,
    *,
    backend: str,
    stage: str,
    model_kind: str = "linear",
) -> dict[str, object]:
    latent_root = latent_destination(config, backend=backend, stage=stage, partition="all", limit=None)
    latent_path = latent_root / "latents.npz"
    if not latent_path.is_file():
        raise FileNotFoundError(f"full State Bank latent artifact not found: {latent_path}")
    with np.load(latent_path, allow_pickle=False) as loaded:
        latents = {name: loaded[name].copy() for name in loaded.files}
    state_ids = latents.pop("state_id").astype(str).tolist()
    latents.pop("__action__", None)
    records = load_records(config.state_bank.output_dir / "records.jsonl")
    by_id = {record.state_id: record for record in records}
    if set(state_ids) != set(by_id):
        raise ValueError("latent artifact does not cover the full fixed State Bank")
    ordered: tuple[StateBankRecord, ...] = tuple(by_id[state_id] for state_id in state_ids)
    split = load_split(config.state_bank.output_dir / "split.json")
    partitions = {
        name: np.asarray([index for index, state_id in enumerate(state_ids) if state_id in set(getattr(split, name))], dtype=np.int64)
        for name in ("train", "validation", "test")
    }
    test_indices = partitions["test"]
    for domain in ("expert_support", "policy_shift"):
        values = np.asarray(
            [index for index in test_indices if ordered[int(index)].domain == domain],
            dtype=np.int64,
        )
        if len(values):
            partitions[f"test/{domain}"] = values
    for stratum in ("nominal", "perturbation", "recovery", "terminal"):
        values = np.asarray(
            [index for index in test_indices if ordered[int(index)].stratum == stratum],
            dtype=np.int64,
        )
        if len(values):
            partitions[f"test/{stratum}"] = values
    targets = probe_targets(ordered)
    taps = tuple(tap.tap_id for tap in registered_taps(backend))
    destination = config.probes.output_dir / backend / stage / model_kind
    if destination.exists() and any(destination.iterdir()):
        report_path = destination / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("latent_sha256") == sha256_file(latent_path):
                return report
        raise FileExistsError(f"probe output already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for tap in taps:
        features = np.asarray(latents[tap], dtype=np.float32)
        for name, target in targets.items():
            model, result, normalization = train_single_probe(
                features,
                target,
                partitions,
                model_kind=model_kind,
                config=config.probes,
            )
            result["episode_group_counts"] = {
                name: int(
                    len(
                        {
                            ordered[int(index)].split_group_id
                            for index in indices
                        }
                    )
                )
                for name, indices in partitions.items()
            }
            metadata = {
                "schema_version": PROBE_SCHEMA_VERSION,
                "backend": backend,
                "stage": stage,
                "tap": tap,
                "target": name,
                "model_kind": model_kind,
                "input_dim": features.shape[1],
                "output_dim": target.output_dim,
                "target_kind": target.kind,
                "head_widths": list(target.head_widths),
                "state_bank_manifest_sha256": sha256_file(config.state_bank.output_dir / "manifest.json"),
                "latent_sha256": sha256_file(latent_path),
            }
            checkpoint = destination / tap / f"{name}.pt"
            _save_model(checkpoint, model, normalization, metadata)
            rows.append({**metadata, **result, "checkpoint": checkpoint.as_posix(), "checkpoint_sha256": sha256_file(checkpoint)})
    report = {
        "passed": True,
        "schema_version": PROBE_SCHEMA_VERSION,
        "backend": backend,
        "stage": stage,
        "model_kind": model_kind,
        "latent_sha256": sha256_file(latent_path),
        "rows": rows,
        "primary_split": "test",
        "selection_split": "validation",
    }
    write_json_atomic(destination / "report.json", report)
    return report


def train_v2_probe_suite(
    *,
    latent_path: str | Path,
    records_path: str | Path,
    split_path: str | Path,
    state_bank_manifest: str | Path,
    backend: str,
    condition: str,
    seed_index: int,
    environment_steps: int,
    model_kind: str,
    config: ProbeConfig,
    destination: str | Path,
    targets: Sequence[str],
) -> dict[str, object]:
    """Train frozen formal probes without changing the legacy v1 suite."""
    latent_source = Path(latent_path)
    destination_path = Path(destination)
    requested = tuple(str(value) for value in targets)
    known = set((*FORMAL_PRIMARY_TARGETS, *FORMAL_SECONDARY_TARGETS))
    if not requested or set(requested) - known:
        raise ValueError("formal probe target selection is incompatible")
    if model_kind not in {"linear", "shallow_mlp"}:
        raise ValueError("formal probe model must be linear or shallow_mlp")
    with np.load(latent_source, allow_pickle=False) as loaded:
        latents = {name: loaded[name].copy() for name in loaded.files}
    state_ids = latents.pop("state_id").astype(str).tolist()
    latents.pop("__action__", None)
    records = [
        json.loads(line)
        for line in Path(records_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(record["state_id"]): record for record in records}
    if len(by_id) != len(records) or set(state_ids) != set(by_id):
        raise ValueError("formal latent artifact does not cover State Bank v2")
    ordered = tuple(by_id[state_id] for state_id in state_ids)
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if split.get("schema_version") != "recovery_state_bank_split_v2":
        raise ValueError("formal State Bank v2 split schema is incompatible")
    raw_partitions = split.get("partitions")
    if not isinstance(raw_partitions, Mapping):
        raise ValueError("formal State Bank v2 split partitions are missing")
    partitions = {
        name: np.asarray(
            [
                index
                for index, state_id in enumerate(state_ids)
                if state_id in set(raw_partitions[name])
            ],
            dtype=np.int64,
        )
        for name in ("train", "validation", "test")
    }
    if any(len(indices) == 0 for indices in partitions.values()):
        raise ValueError("formal probe partitions must be non-empty")
    target_map = v2_probe_targets(ordered)
    latent_hash = sha256_file(latent_source)
    bank_hash = sha256_file(state_bank_manifest)
    report_path = destination_path / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("latent_sha256") == latent_hash
            and report.get("state_bank_manifest_sha256") == bank_hash
        ):
            return report
        raise FileExistsError(f"formal probe output binding differs: {destination_path}")
    if destination_path.exists() and any(destination_path.iterdir()):
        raise FileExistsError(f"formal probe output already exists: {destination_path}")
    destination_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for tap in sorted(latents):
        features = np.asarray(latents[tap], dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != len(ordered):
            raise ValueError(f"formal latent tap is not row-aligned: {tap}")
        for name in requested:
            target = target_map[name]
            model, result, normalization = train_single_probe(
                features,
                target,
                partitions,
                model_kind=model_kind,
                config=config,
            )
            result["source_seed_group_counts"] = {
                partition: len(
                    {
                        int(ordered[int(index)]["source_seed"])
                        for index in indices
                    }
                )
                for partition, indices in partitions.items()
            }
            metadata = {
                "schema_version": "recovery_frozen_probe_v2",
                "backend": backend,
                "condition": condition,
                "seed_index": seed_index,
                "environment_steps": environment_steps,
                "tap": tap,
                "target": name,
                "target_role": (
                    "primary"
                    if name in FORMAL_PRIMARY_TARGETS
                    else "secondary"
                ),
                "model_kind": model_kind,
                "input_dim": int(features.shape[1]),
                "output_dim": target.output_dim,
                "target_kind": target.kind,
                "head_widths": list(target.head_widths),
                "state_bank_manifest_sha256": bank_hash,
                "latent_sha256": latent_hash,
            }
            checkpoint = destination_path / tap / f"{name}.pt"
            _save_model(checkpoint, model, normalization, metadata)
            rows.append(
                {
                    **metadata,
                    **result,
                    "checkpoint": checkpoint.as_posix(),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )
    report = {
        "passed": True,
        "schema_version": "recovery_frozen_probe_v2",
        "backend": backend,
        "condition": condition,
        "seed_index": seed_index,
        "environment_steps": environment_steps,
        "model_kind": model_kind,
        "targets": list(requested),
        "latent_sha256": latent_hash,
        "state_bank_manifest_sha256": bank_hash,
        "rows": rows,
        "primary_split": "test",
        "selection_split": "validation",
    }
    write_json_atomic(report_path, report)
    return report
