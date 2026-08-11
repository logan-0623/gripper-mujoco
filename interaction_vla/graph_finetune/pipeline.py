from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from interaction_vla.device import resolve_device
from interaction_vla.graph_pretrain.reflectvlm import Vocabulary
from interaction_vla.graph_pretrain.schema import SCHEMA_VERSION as REFLECT_SCHEMA
from interaction_vla.lerobot_bridge.teacher_schema import SCHEMA_VERSION as TEACHER_SCHEMA

from .config import GraphFinetuneConfig, ModelConfig
from .data import (
    GraphNormalization,
    MuJoCoGraphDataset,
    PreparedMuJoCoCorpus,
    TrainingCorpus,
    prepare_corpus,
)
from .model import (
    MuJoCoGraphEstimator,
    TransferReport,
    graph_finetune_loss,
    initialize_paired_models,
)
from .schema import RELATION_COUNT, SCHEMA_VERSION, SEMANTIC_DIM


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
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
        raise FloatingPointError("graph fine-tuning produced a non-finite scalar loss")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _reflect_vocabulary(path: Path) -> Vocabulary:
    if not path.is_file():
        raise FileNotFoundError(f"ReflectVLM checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != REFLECT_SCHEMA:
        raise ValueError("ReflectVLM checkpoint schema is incompatible")
    values = payload.get("vocabulary")
    if not isinstance(values, (list, tuple)):
        raise ValueError("ReflectVLM checkpoint vocabulary is missing")
    return Vocabulary(tuple(str(value) for value in values))


def _prepare(
    config: GraphFinetuneConfig,
    source: Any,
    records: Sequence[Mapping[str, object]],
    sidecars: Mapping[int, Mapping[str, np.ndarray]],
) -> PreparedMuJoCoCorpus:
    return prepare_corpus(
        source,
        records,
        sidecars,
        split_seed=config.dataset.split_seed,
        split_ratios=config.dataset.split_ratios,
        pretrained_vocabulary=_reflect_vocabulary(config.dataset.reflect_checkpoint),
    )


def _transfer_report_dict(report: TransferReport | None) -> dict[str, Any]:
    if report is None:
        return {
            "copied_modules": [],
            "copied_tensors": [],
            "skipped_tensors": [],
            "copied_token_count": 0,
            "target_token_count": 0,
        }
    return asdict(report)


def _paired_models(
    config: GraphFinetuneConfig,
    training: TrainingCorpus,
    seed: int,
) -> tuple[MuJoCoGraphEstimator, MuJoCoGraphEstimator, TransferReport]:
    model = config.model
    return initialize_paired_models(
        vocab_size=len(training.vocabulary.tokens),
        vocabulary=training.vocabulary,
        reflect_checkpoint=config.dataset.reflect_checkpoint,
        seed=seed,
        image_embedding_dim=model.image_embedding_dim,
        text_embedding_dim=model.text_embedding_dim,
        graph_embedding_dim=model.graph_embedding_dim,
    )


def inspect_with_source(
    config: GraphFinetuneConfig,
    source: Any,
    records: Sequence[Mapping[str, object]],
    sidecars: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    corpus = _prepare(config, source, records, sidecars)
    training = corpus.for_training_fraction(1.0, config.training.seeds[0])
    _, _, transfer = _paired_models(config, training, config.training.seeds[0])
    partition_sets = {name: set(values) for name, values in corpus.splits.items()}
    overlap = any(
        partition_sets[left] & partition_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )
    report = {
        "passed": not overlap,
        "schema_version": SCHEMA_VERSION,
        "teacher_schema_version": TEACHER_SCHEMA,
        "repo_id": config.dataset.repo_id,
        "episodes": len(corpus.records),
        "frames": len(source),
        "partition_episodes": {
            name: len(values) for name, values in corpus.splits.items()
        },
        "partition_frames": {
            name: sum(len(corpus.row_indices[episode]) for episode in values)
            for name, values in corpus.splits.items()
        },
        "episode_overlap": overlap,
        "reflect_checkpoint": config.dataset.reflect_checkpoint.as_posix(),
        "copied_modules": list(transfer.copied_modules),
        "copied_token_count": transfer.copied_token_count,
        "target_token_count": transfer.target_token_count,
        "forbidden_input_keys": [],
    }
    if overlap:
        raise ValueError("episode split leaked across partitions")
    return report


def _dataset(
    config: GraphFinetuneConfig,
    training: TrainingCorpus,
    partition: str,
) -> MuJoCoGraphDataset:
    return MuJoCoGraphDataset(
        training,
        partition=partition,
        image_size=config.model.image_size,
        max_language_tokens=config.model.max_language_tokens,
    )


def _loader(
    config: GraphFinetuneConfig,
    training: TrainingCorpus,
    partition: str,
    *,
    shuffle: bool,
    seed: int,
) -> tuple[MuJoCoGraphDataset, DataLoader[dict[str, Tensor]]]:
    dataset = _dataset(config, training, partition)
    generator = torch.Generator().manual_seed(seed)
    return dataset, DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        generator=generator if shuffle else None,
    )


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _forward(
    model: MuJoCoGraphEstimator, batch: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    return model(
        batch["agent_rgb"],
        batch["wrist_rgb"],
        batch["state"],
        batch["language_tokens"],
        batch["language_mask"],
    )


def _binary_counts(
    predicted: Tensor, truth: Tensor
) -> tuple[int, int, int]:
    predicted_values = predicted.bool()
    truth_values = truth.bool()
    true_positive = int((predicted_values & truth_values).sum().cpu())
    false_positive = int((predicted_values & ~truth_values).sum().cpu())
    false_negative = int((~predicted_values & truth_values).sum().cpu())
    return true_positive, false_positive, false_negative


def _prf(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> tuple[float, float, float]:
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return precision, recall, f1


def _evaluate_loader(
    model: MuJoCoGraphEstimator,
    loader: DataLoader[dict[str, Tensor]],
    *,
    device: torch.device,
    normalization: GraphNormalization,
) -> dict[str, Any]:
    model.eval()
    examples = 0
    loss_sum = 0.0
    entity_counts = [0, 0, 0]
    relation_counts = [0, 0, 0]
    visibility_error = 0.0
    visibility_count = 0
    semantic_error = 0.0
    semantic_count = 0
    relation_error = np.zeros(RELATION_COUNT, dtype=np.float64)
    relation_value_count = np.zeros(RELATION_COUNT, dtype=np.int64)
    relation_correct = operator_correct = predicate_correct = exact_correct = 0
    residual_error = 0.0
    relation_std = torch.as_tensor(
        normalization.relation_std, dtype=torch.float32, device=device
    )
    residual_std = float(normalization.residual_std)
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            outputs = _forward(model, batch)
            loss = graph_finetune_loss(outputs, batch)["total"]
            require_finite_loss(loss)
            count = int(batch["state"].shape[0])
            examples += count
            loss_sum += float(loss.detach().cpu()) * count
            values = _binary_counts(
                outputs["entity_mask_logits"] > 0.0,
                batch["entity_mask"],
            )
            entity_counts = [left + right for left, right in zip(entity_counts, values)]
            values = _binary_counts(
                outputs["relation_mask_logits"] > 0.0,
                batch["relation_mask"],
            )
            relation_counts = [left + right for left, right in zip(relation_counts, values)]
            entity_mask = batch["entity_mask"].bool()
            visibility_error += float(
                torch.abs(
                    outputs["entity_visibility"] - batch["entity_visibility"]
                )[entity_mask].sum().cpu()
            )
            visibility_count += int(entity_mask.sum().cpu()) * 2
            relation_mask = batch["relation_mask"].bool()
            absolute_semantic_error = torch.abs(
                outputs["relation_semantics"] - batch["relation_semantics"]
            ) * relation_std
            semantic_error += float(
                absolute_semantic_error[relation_mask].sum().cpu()
            )
            semantic_count += int(relation_mask.sum().cpu()) * SEMANTIC_DIM
            for relation in range(RELATION_COUNT):
                active = relation_mask[:, relation]
                relation_error[relation] += float(
                    absolute_semantic_error[:, relation][active].sum().cpu()
                )
                relation_value_count[relation] += int(active.sum().cpu()) * SEMANTIC_DIM
            relation_prediction = outputs["goal_relation_logits"].argmax(-1)
            operator_prediction = outputs["goal_operator_logits"].argmax(-1)
            predicate_prediction = outputs["goal_predicate_logits"].argmax(-1)
            relation_match = relation_prediction == batch["goal_relation"]
            operator_match = operator_prediction == batch["goal_operator"]
            predicate_match = predicate_prediction == batch["goal_predicate"]
            relation_correct += int(relation_match.sum().cpu())
            operator_correct += int(operator_match.sum().cpu())
            predicate_correct += int(predicate_match.sum().cpu())
            exact_correct += int(
                (relation_match & operator_match & predicate_match).sum().cpu()
            )
            residual_error += float(
                (
                    torch.abs(outputs["goal_residual"] - batch["goal_residual"])
                    * residual_std
                ).sum().cpu()
            )
    if examples == 0 or visibility_count == 0 or semantic_count == 0:
        raise ValueError("evaluation requires active held-out graph examples")
    entity_precision, entity_recall, entity_f1 = _prf(*entity_counts)
    relation_precision, relation_recall, relation_f1 = _prf(*relation_counts)
    per_relation = [
        float(relation_error[index] / max(1, relation_value_count[index]))
        for index in range(RELATION_COUNT)
    ]
    return {
        "test_examples": examples,
        "mean_loss": loss_sum / examples,
        "entity_mask_precision": entity_precision,
        "entity_mask_recall": entity_recall,
        "entity_mask_f1": entity_f1,
        "relation_mask_precision": relation_precision,
        "relation_mask_recall": relation_recall,
        "relation_mask_f1": relation_f1,
        "entity_visibility_mae": visibility_error / visibility_count,
        "semantic_relation_mae": semantic_error / semantic_count,
        "semantic_relation_mae_by_slot": per_relation,
        "goal_relation_accuracy": relation_correct / examples,
        "goal_operator_accuracy": operator_correct / examples,
        "goal_predicate_accuracy": predicate_correct / examples,
        "goal_exact_accuracy": exact_correct / examples,
        "goal_residual_mae": residual_error / examples,
    }


def _normalization_payload(value: GraphNormalization) -> dict[str, Any]:
    return {
        "state_mean": value.state_mean,
        "state_std": value.state_std,
        "relation_mean": value.relation_mean,
        "relation_std": value.relation_std,
        "residual_mean": value.residual_mean,
        "residual_std": value.residual_std,
    }


def _model_from_config(config: ModelConfig, vocab_size: int) -> MuJoCoGraphEstimator:
    return MuJoCoGraphEstimator(
        vocab_size=vocab_size,
        image_embedding_dim=config.image_embedding_dim,
        text_embedding_dim=config.text_embedding_dim,
        graph_embedding_dim=config.graph_embedding_dim,
    )


def load_finetune_checkpoint(
    path: str | Path, *, device: str | torch.device
) -> tuple[MuJoCoGraphEstimator, Vocabulary, dict[str, Any]]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MuJoCo graph checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("MuJoCo graph checkpoint schema is incompatible")
    required = {"model_config", "vocabulary", "model_state", "normalization"}
    missing = required - set(payload)
    if missing:
        raise ValueError("MuJoCo graph checkpoint is missing: " + ", ".join(sorted(missing)))
    model_config = ModelConfig(**payload["model_config"])
    vocabulary = Vocabulary(tuple(str(value) for value in payload["vocabulary"]))
    model = _model_from_config(model_config, len(vocabulary.tokens)).to(
        torch.device(device)
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, vocabulary, payload


def _train_condition(
    config: GraphFinetuneConfig,
    training: TrainingCorpus,
    *,
    model: MuJoCoGraphEstimator,
    initialization: str,
    fraction: float,
    seed: int,
    transfer: TransferReport | None,
    run_dir: Path,
) -> dict[str, Any]:
    device = resolve_device(config.training.device)
    _seed_all(seed)
    model = model.to(device)
    train_dataset, train_loader = _loader(
        config, training, "train", shuffle=True, seed=seed
    )
    validation_dataset, validation_loader = _loader(
        config, training, "validation", shuffle=False, seed=seed
    )
    test_dataset, test_loader = _loader(
        config, training, "test", shuffle=False, seed=seed
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    steps = 0
    for epoch in range(config.training.epochs):
        model.train()
        total = 0.0
        examples = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            loss = graph_finetune_loss(_forward(model, batch), batch)["total"]
            require_finite_loss(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(batch["state"].shape[0])
            total += float(loss.detach().cpu()) * count
            examples += count
            steps += 1
        validation = _evaluate_loader(
            model,
            validation_loader,
            device=device,
            normalization=training.normalization,
        )
        training_loss = total / max(1, examples)
        validation_loss = float(validation["mean_loss"])
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None or not math.isfinite(best_loss):
        raise FloatingPointError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint_payload = {
        "schema_version": SCHEMA_VERSION,
        "teacher_schema_version": TEACHER_SCHEMA,
        "repo_id": config.dataset.repo_id,
        "split_seed": config.dataset.split_seed,
        "fraction": fraction,
        "seed": seed,
        "initialization": initialization,
        "model_config": asdict(config.model),
        "vocabulary": list(training.vocabulary.tokens),
        "normalization": _normalization_payload(training.normalization),
        "selected_train_episodes": list(training.selected_train_episodes),
        "train_row_indices": list(train_dataset.row_indices),
        "validation_row_indices": list(validation_dataset.row_indices),
        "test_row_indices": list(test_dataset.row_indices),
        "transfer_report": _transfer_report_dict(transfer),
        "model_state": best_state,
    }
    _save_checkpoint_atomic(checkpoint, checkpoint_payload)
    reloaded, _, reloaded_payload = load_finetune_checkpoint(checkpoint, device=device)
    evaluation = _evaluate_loader(
        reloaded,
        test_loader,
        device=device,
        normalization=training.normalization,
    )
    evaluation.update(
        {
            "passed": True,
            "schema_version": SCHEMA_VERSION,
            "initialization": initialization,
            "fraction": fraction,
            "seed": seed,
            "checkpoint": checkpoint.as_posix(),
            "test_row_indices": reloaded_payload["test_row_indices"],
        }
    )
    summary = {
        "passed": True,
        "initialization": initialization,
        "fraction": fraction,
        "seed": seed,
        "device": device.type,
        "epochs": config.training.epochs,
        "steps": steps,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "checkpoint": checkpoint.as_posix(),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "test_examples": len(test_dataset),
        "transfer_report": _transfer_report_dict(transfer),
    }
    _write_json_atomic(run_dir / "training_summary.json", summary)
    _write_json_atomic(run_dir / "evaluation.json", evaluation)
    return evaluation


def _fraction_directory(fraction: float) -> str:
    return "fraction_" + format(float(fraction), ".6g").replace(".", "p")


def _aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "mean_loss",
        "entity_mask_f1",
        "relation_mask_f1",
        "entity_visibility_mae",
        "semantic_relation_mae",
        "goal_relation_accuracy",
        "goal_operator_accuracy",
        "goal_predicate_accuracy",
        "goal_exact_accuracy",
        "goal_residual_mae",
    )
    result: dict[str, Any] = {}
    for condition in ("random_init", "reflectvlm_init"):
        result[condition] = {}
        for name in metric_names:
            values = np.asarray(
                [float(run[condition][name]) for run in runs], dtype=np.float64
            )
            result[condition][name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "count": int(len(values)),
            }
    result["delta"] = {}
    for name in (
        "goal_exact_accuracy",
        "relation_mask_f1",
        "semantic_relation_mae",
    ):
        values = np.asarray(
            [float(run["delta"][name]) for run in runs], dtype=np.float64
        )
        result["delta"][name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "count": int(len(values)),
        }
    return result


def compare_with_source(
    config: GraphFinetuneConfig,
    source: Any,
    records: Sequence[Mapping[str, object]],
    sidecars: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    output_dir = config.training.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"graph fine-tune output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = _prepare(config, source, records, sidecars)
    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "split_seed": config.dataset.split_seed,
        "episode_indices": corpus.splits,
        "row_indices": {
            name: [
                row
                for episode in episodes
                for row in corpus.row_indices[episode]
            ]
            for name, episodes in corpus.splits.items()
        },
    }
    _write_json_atomic(output_dir / "split_manifest.json", split_manifest)
    runs: list[dict[str, Any]] = []
    for fraction in config.training.fractions:
        for seed in config.training.seeds:
            training = corpus.for_training_fraction(fraction, seed)
            random_model, pretrained_model, transfer = _paired_models(
                config, training, seed
            )
            results: dict[str, Any] = {"fraction": fraction, "seed": seed}
            for initialization, model, report in (
                ("random_init", random_model, None),
                ("reflectvlm_init", pretrained_model, transfer),
            ):
                run_dir = (
                    output_dir
                    / initialization
                    / _fraction_directory(fraction)
                    / f"seed_{seed}"
                )
                results[initialization] = _train_condition(
                    config,
                    training,
                    model=model,
                    initialization=initialization,
                    fraction=fraction,
                    seed=seed,
                    transfer=report,
                    run_dir=run_dir,
                )
            if results["random_init"]["test_row_indices"] != results[
                "reflectvlm_init"
            ]["test_row_indices"]:
                raise ValueError("paired conditions evaluated different test rows")
            results["delta"] = {
                name: float(results["reflectvlm_init"][name])
                - float(results["random_init"][name])
                for name in (
                    "goal_exact_accuracy",
                    "relation_mask_f1",
                    "semantic_relation_mae",
                )
            }
            runs.append(results)
    comparison = {
        "passed": True,
        "schema_version": SCHEMA_VERSION,
        "conditions": ["random_init", "reflectvlm_init"],
        "fractions": list(config.training.fractions),
        "seeds": list(config.training.seeds),
        "paired_runs": len(runs),
        "scientific_result": False,
        "aggregate": _aggregate_runs(runs),
        "runs": runs,
    }
    _write_json_atomic(output_dir / "comparison.json", comparison)
    return comparison
