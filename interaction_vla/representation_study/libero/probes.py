from __future__ import annotations

from collections import Counter
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn


STUDY_STAGES = ("pretrained", "sft_25", "sft_50", "sft_100")
STUDY_TAPS = (
    "vision_output",
    "multimodal_fusion",
    "action_expert_input",
    "pre_action",
)
FACTOR_NAMES = ("entity", "geometry", "contact", "stable_grasp", "phase", "next_relation")


def build_stage_tap_factor_grid(
    completed_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    lookup = {
        (str(row["stage"]), str(row["tap"]), str(row["factor"])): row
        for row in completed_rows
    }
    result: list[dict[str, object]] = []
    for stage in STUDY_STAGES:
        for tap in STUDY_TAPS:
            for factor in FACTOR_NAMES:
                row = lookup.get((stage, tap, factor))
                if row is None:
                    result.append(
                        {
                            "stage": stage,
                            "tap": tap,
                            "factor": factor,
                            "status": "not_run",
                            "accessible": None,
                            "primary_metric": None,
                        }
                    )
                else:
                    result.append(dict(row))
    return result

def _confusion(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.unique(np.concatenate((target, prediction)))
    labels = observed if labels is None else np.asarray(labels)
    if not set(observed.tolist()).issubset(set(labels.tolist())):
        raise ValueError("classification label universe omits observed labels")
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    lookup = {label: index for index, label in enumerate(labels)}
    for truth, predicted in zip(target, prediction, strict=True):
        matrix[lookup[truth], lookup[predicted]] += 1
    return labels, matrix


def _binary_auprc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    sorted_target = target[order]
    true_positive = np.cumsum(sorted_target)
    false_positive = np.cumsum(1 - sorted_target)
    recall = true_positive / positives
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def classification_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    score: np.ndarray | None = None,
    binary: bool = False,
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.ndim != 1 or prediction.shape != target.shape or target.size == 0:
        raise ValueError("classification target/prediction must be non-empty vectors")
    _, matrix = _confusion(target, prediction, labels=labels)
    per_class_recall: list[float] = []
    per_class_f1: list[float] = []
    for index in range(matrix.shape[0]):
        true_positive = float(matrix[index, index])
        false_positive = float(matrix[:, index].sum() - matrix[index, index])
        false_negative = float(matrix[index, :].sum() - matrix[index, index])
        recall = true_positive / max(true_positive + false_negative, 1.0)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        per_class_recall.append(recall)
        per_class_f1.append(
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
    result = {
        "accuracy": float(np.mean(target == prediction)),
        "macro_f1": float(np.mean(per_class_f1)),
        "balanced_accuracy": float(np.mean(per_class_recall)),
    }
    if binary:
        if score is None:
            raise ValueError("binary AUPRC requires positive-class scores")
        result["auprc"] = _binary_auprc(target, np.asarray(score))
    return result


def constant_classification_baseline(target: np.ndarray) -> np.ndarray:
    values = np.asarray(target)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("constant baseline requires a non-empty target vector")
    counts = Counter(values.tolist())
    majority = min(counts, key=lambda item: (-counts[item], str(item)))
    return np.full(values.shape, majority, dtype=values.dtype)


def geometry_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    normalization_scale: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    scale = np.asarray(normalization_scale, dtype=np.float64)
    if target.ndim != 2 or prediction.shape != target.shape or scale.shape != (target.shape[1],):
        raise ValueError("geometry inputs have incompatible shapes")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all() or np.any(scale <= 0):
        raise ValueError("geometry metrics require finite values and positive scale")
    normalized_mae = float(np.mean(np.abs(target - prediction) / scale[None, :]))
    residual = float(np.sum((target - prediction) ** 2))
    centered = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    return {
        "normalized_mae": normalized_mae,
        "r2": 1.0 - residual / centered if centered > 0 else float("nan"),
    }


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (values - mean) / scale, mean, scale


def run_linear_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    test_indices: Sequence[int],
    task: Literal["classification", "regression"],
    seed: int,
    l2_grid: tuple[float, ...],
    epochs: int = 300,
    selection_metric: str | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("probe features/targets are invalid")
    train = np.asarray(train_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if any(index.size == 0 for index in (train, validation, test)):
        raise ValueError("probe partitions must be non-empty")
    standardized, x_mean, x_scale = _standardize(x[train], x)
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA probe device requested but CUDA is unavailable")
    best: tuple[float, float, dict[str, torch.Tensor]] | None = None
    if task == "classification":
        classes = np.unique(y[train])
        if len(classes) < 2:
            raise ValueError("classification probe requires at least two training classes")
        class_to_index = {value: index for index, value in enumerate(classes.tolist())}
        encoded = np.asarray([class_to_index.get(value, -1) for value in y], dtype=np.int64)
        if np.any(encoded[validation] < 0) or np.any(encoded[test] < 0):
            raise ValueError("validation/test contains classes absent from training")
        binary = len(classes) == 2
        metric_name = selection_metric or "balanced_accuracy"
        allowed = {"accuracy", "macro_f1", "balanced_accuracy"}
        if binary:
            allowed.add("auprc")
        if metric_name not in allowed:
            raise ValueError(f"unsupported classification selection metric: {metric_name}")
        label_universe = np.arange(len(classes), dtype=np.int64)
        x_tensor = torch.from_numpy(standardized).to(torch_device)
        y_tensor = torch.from_numpy(encoded).to(torch_device)
        train_tensor = torch.as_tensor(train, device=torch_device)
        validation_tensor = torch.as_tensor(validation, device=torch_device)
        test_tensor = torch.as_tensor(test, device=torch_device)
        for l2 in l2_grid:
            torch.manual_seed(seed)
            model = nn.Linear(x.shape[1], len(classes)).to(torch_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
            best_validation = -float("inf")
            best_l2_state: dict[str, torch.Tensor] | None = None
            stale_epochs = 0
            for _ in range(epochs):
                logits = model(x_tensor[train_tensor])
                loss = nn.functional.cross_entropy(logits, y_tensor[train_tensor])
                loss = loss + float(l2) * sum((parameter**2).sum() for parameter in model.parameters())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    validation_logits = model(x_tensor[validation_tensor])
                    validation_prediction = validation_logits.argmax(dim=1).cpu().numpy()
                    validation_probability = torch.softmax(validation_logits, dim=1).cpu().numpy()
                validation_metric = classification_metrics(
                    encoded[validation],
                    validation_prediction,
                    score=validation_probability[:, 1] if binary else None,
                    binary=binary,
                    labels=label_universe,
                )[metric_name]
                if validation_metric > best_validation + 1e-8:
                    best_validation = validation_metric
                    best_l2_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= 20:
                    break
            assert best_l2_state is not None
            candidate = (best_validation, -float(l2), best_l2_state)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        model = nn.Linear(x.shape[1], len(classes)).to(torch_device)
        model.load_state_dict(best[2])
        with torch.no_grad():
            test_logits = model(x_tensor[test_tensor])
            prediction = test_logits.argmax(dim=1).cpu().numpy()
            probability = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_metrics = classification_metrics(
            encoded[test],
            prediction,
            score=probability[:, 1] if binary else None,
            binary=binary,
            labels=label_universe,
        )
        baseline_prediction = constant_classification_baseline(encoded[train])
        majority = int(baseline_prediction[0])
        baseline_test = np.full(len(test), majority, dtype=np.int64)
        prevalence = float(np.mean(encoded[train] == 1)) if binary else None
        baseline_metrics = classification_metrics(
            encoded[test],
            baseline_test,
            score=np.full(len(test), prevalence) if binary else None,
            binary=binary,
            labels=label_universe,
        )
        return {
            "model": "linear",
            "task": task,
            "selection_metric": metric_name,
            "device": torch_device.type,
            "selected_l2": -best[1],
            "test_metrics": test_metrics,
            "baseline_metrics": baseline_metrics,
            "feature_mean": x_mean.tolist(),
            "feature_scale": x_scale.tolist(),
            "classes": classes.tolist(),
            "weight": best[2]["weight"].numpy().tolist(),
            "bias": best[2]["bias"].numpy().tolist(),
            "test_target": encoded[test].tolist(),
            "test_prediction": prediction.tolist(),
            "test_score": probability[:, 1].tolist() if binary else None,
        }
    if task != "regression":
        raise ValueError(f"unknown probe task: {task}")
    y_regression = np.asarray(y, dtype=np.float64)
    if y_regression.ndim == 1:
        y_regression = y_regression[:, None]
    best_regression: tuple[float, float, np.ndarray] | None = None
    metric_name = selection_metric or "mae"
    if metric_name not in {"mae", "normalized_mae"}:
        raise ValueError(f"unsupported regression selection metric: {metric_name}")
    scale = np.ptp(y_regression[train], axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    train_x = np.concatenate((standardized[train], np.ones((len(train), 1))), axis=1)
    gram = train_x.T @ train_x
    rhs = train_x.T @ y_regression[train]
    for l2 in l2_grid:
        regularizer = np.eye(train_x.shape[1]) * float(l2)
        regularizer[-1, -1] = 0.0
        system = gram + regularizer
        weight = (
            np.linalg.pinv(system) @ rhs
            if float(l2) == 0.0
            else np.linalg.solve(system, rhs)
        )
        validation_x = np.concatenate((standardized[validation], np.ones((len(validation), 1))), axis=1)
        validation_prediction = validation_x @ weight
        validation_error = (
            geometry_metrics(
                y_regression[validation],
                validation_prediction,
                normalization_scale=scale,
            )["normalized_mae"]
            if metric_name == "normalized_mae"
            else float(np.mean(np.abs(validation_prediction - y_regression[validation])))
        )
        candidate = (-validation_error, -float(l2), weight)
        if best_regression is None or candidate[:2] > best_regression[:2]:
            best_regression = candidate
    assert best_regression is not None
    test_x = np.concatenate((standardized[test], np.ones((len(test), 1))), axis=1)
    prediction = test_x @ best_regression[2]
    baseline = np.broadcast_to(y_regression[train].mean(axis=0), prediction.shape)
    return {
        "model": "linear",
        "task": task,
        "selection_metric": metric_name,
        "device": "cpu",
        "selected_l2": -best_regression[1],
        "test_metrics": geometry_metrics(y_regression[test], prediction, normalization_scale=scale),
        "baseline_metrics": geometry_metrics(y_regression[test], baseline, normalization_scale=scale),
        "feature_mean": x_mean.tolist(),
        "feature_scale": x_scale.tolist(),
        "weight": best_regression[2][:-1].T.tolist(),
        "bias": best_regression[2][-1].tolist(),
        "test_target": y_regression[test].tolist(),
        "test_prediction": prediction.tolist(),
    }


def run_shallow_mlp_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    test_indices: Sequence[int],
    task: Literal["classification", "regression"],
    seed: int,
    hidden_dim: int,
    l2: float,
    epochs: int = 300,
) -> dict[str, object]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets)
    train = np.asarray(train_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if hidden_dim <= 0 or any(indices.size == 0 for indices in (train, validation, test)):
        raise ValueError("MLP probe dimensions and partitions must be non-empty")
    standardized, x_mean, x_scale = _standardize(x[train], x)
    x_tensor = torch.from_numpy(standardized)
    torch.manual_seed(seed)
    if task == "classification":
        classes = np.unique(y[train])
        class_to_index = {value: index for index, value in enumerate(classes.tolist())}
        encoded = np.asarray([class_to_index.get(value, -1) for value in y], dtype=np.int64)
        if len(classes) < 2 or np.any(encoded[validation] < 0) or np.any(encoded[test] < 0):
            raise ValueError("MLP classification partitions have unsupported classes")
        output_dim = len(classes)
        target_tensor = torch.from_numpy(encoded)
        loss_fn = lambda output, index: nn.functional.cross_entropy(output, target_tensor[index])
    elif task == "regression":
        regression = np.asarray(y, dtype=np.float32)
        if regression.ndim == 1:
            regression = regression[:, None]
        output_dim = regression.shape[1]
        target_tensor = torch.from_numpy(regression)
        loss_fn = lambda output, index: nn.functional.mse_loss(output, target_tensor[index])
    else:
        raise ValueError(f"unknown MLP probe task: {task}")
    model = nn.Sequential(
        nn.Linear(x.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for _ in range(epochs):
        output = model(x_tensor[train])
        loss = loss_fn(output, train)
        loss = loss + l2 * sum((parameter**2).sum() for parameter in model.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_tensor[validation]), validation).item())
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 20:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    with torch.no_grad():
        output = model(x_tensor[test])
    if task == "classification":
        prediction = output.argmax(dim=1).numpy()
        probability = torch.softmax(output, dim=1).numpy()
        binary = output_dim == 2
        metrics = classification_metrics(
            encoded[test],
            prediction,
            score=probability[:, 1] if binary else None,
            binary=binary,
            labels=np.arange(output_dim, dtype=np.int64),
        )
        return {
            "model": "shallow_mlp",
            "task": task,
            "test_metrics": metrics,
            "test_target": encoded[test].tolist(),
            "test_prediction": prediction.tolist(),
            "feature_mean": x_mean.tolist(),
            "feature_scale": x_scale.tolist(),
            "hidden_dim": hidden_dim,
        }
    prediction = output.numpy()
    scale = np.ptp(target_tensor[train].numpy(), axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return {
        "model": "shallow_mlp",
        "task": task,
        "test_metrics": geometry_metrics(
            target_tensor[test].numpy(), prediction, normalization_scale=scale
        ),
        "test_target": target_tensor[test].numpy().tolist(),
        "test_prediction": prediction.tolist(),
        "feature_mean": x_mean.tolist(),
        "feature_scale": x_scale.tolist(),
        "hidden_dim": hidden_dim,
    }
