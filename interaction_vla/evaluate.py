from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from .config import load_config
from .data import episode_paths_from_manifest, load_episode_arrays, split_episode_seeds
from .device import resolve_device
from .env import KinematicTabletopEnv, TerminationReason
from .expert import ScriptedExpert
from .graph.builder import SceneGraphBuilder
from .models.encoders import SceneBatch, scene_graphs_to_batch
from .models.policy import ActionPolicy
from .train import EpisodeFrameDataset, TrainingStatistics, load_training_checkpoint


@dataclass(frozen=True)
class EvaluationCase:
    seed: int
    object_count: int
    split: str
    condition: str = "id_normal"
    layout_mode: str = "normal"


@dataclass(frozen=True)
class EpisodeResult:
    policy: str
    seed: int
    object_count: int
    split: str
    success: bool
    wrong_object: bool
    grasped: bool
    dropped: bool
    steps: int
    termination_reason: str
    on_policy_action_mse: float
    representation: str = ""
    model_seed: int = 0
    ablation: str = "none"
    condition: str = "id_normal"
    layout_mode: str = "normal"


def _result_condition(result: EpisodeResult) -> str:
    """Map legacy results onto the named-condition schema."""

    if result.condition != "id_normal" or result.split == "id":
        return result.condition
    return "crowded_ood" if result.layout_mode == "crowded" else "count_ood"


def shuffle_valid_edge_assignments(batch: SceneBatch, *, seed: int) -> SceneBatch:
    """Break relation-to-edge alignment without changing values or padding."""

    shuffled = batch.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for batch_index in range(batch.edge_features.shape[0]):
        valid_indices = torch.nonzero(
            batch.edge_mask[batch_index], as_tuple=False
        ).flatten()
        if valid_indices.numel() < 2:
            continue
        order = torch.randperm(valid_indices.numel(), generator=generator).to(
            valid_indices.device
        )
        shuffled.edge_features[batch_index, valid_indices] = batch.edge_features[
            batch_index, valid_indices[order]
        ]
    return shuffled


def _metric_summary(results: list[EpisodeResult]) -> dict[str, float | int]:
    if not results:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "wrong_object_rate": 0.0,
            "grasp_rate": 0.0,
            "drop_rate": 0.0,
            "mean_steps": 0.0,
            "on_policy_action_mse": 0.0,
        }
    return {
        "episodes": len(results),
        "success_rate": float(np.mean([result.success for result in results])),
        "wrong_object_rate": float(np.mean([result.wrong_object for result in results])),
        "grasp_rate": float(np.mean([result.grasped for result in results])),
        "drop_rate": float(np.mean([result.dropped for result in results])),
        "mean_steps": float(np.mean([result.steps for result in results])),
        "on_policy_action_mse": float(
            np.mean([result.on_policy_action_mse for result in results])
        ),
    }


def aggregate_results(results: Iterable[EpisodeResult]) -> dict:
    values = list(results)
    by_count: dict[str, dict] = {}
    for object_count in sorted({result.object_count for result in values}):
        group = [result for result in values if result.object_count == object_count]
        splits = {result.split for result in group}
        by_count[str(object_count)] = {
            "split": next(iter(splits)) if len(splits) == 1 else "mixed",
            **_metric_summary(group),
        }
    by_policy = {
        policy: _metric_summary([result for result in values if result.policy == policy])
        for policy in sorted({result.policy for result in values})
    }
    by_policy_and_count = {
        policy: {
            str(object_count): {
                "split": next(
                    result.split
                    for result in values
                    if result.policy == policy and result.object_count == object_count
                ),
                **_metric_summary(
                    [
                        result
                        for result in values
                        if result.policy == policy and result.object_count == object_count
                    ]
                ),
            }
            for object_count in sorted(
                {
                    result.object_count
                    for result in values
                    if result.policy == policy
                }
            )
        }
        for policy in sorted({result.policy for result in values})
    }
    by_condition = {
        condition: _metric_summary(
            [result for result in values if _result_condition(result) == condition]
        )
        for condition in sorted({_result_condition(result) for result in values})
    }
    by_policy_and_condition = {
        policy: {
            condition: _metric_summary(
                [
                    result
                    for result in values
                    if result.policy == policy and _result_condition(result) == condition
                ]
            )
            for condition in sorted(
                {
                    _result_condition(result)
                    for result in values
                    if result.policy == policy
                }
            )
        }
        for policy in sorted({result.policy for result in values})
    }
    by_policy_condition_and_count = {
        policy: {
            condition: {
                str(object_count): _metric_summary(
                    [
                        result
                        for result in values
                        if result.policy == policy
                        and _result_condition(result) == condition
                        and result.object_count == object_count
                    ]
                )
                for object_count in sorted(
                    {
                        result.object_count
                        for result in values
                        if result.policy == policy
                        and _result_condition(result) == condition
                    }
                )
            }
            for condition in sorted(
                {
                    _result_condition(result)
                    for result in values
                    if result.policy == policy
                }
            )
        }
        for policy in sorted({result.policy for result in values})
    }
    return {
        "overall": _metric_summary(values),
        "by_object_count": by_count,
        "by_policy": by_policy,
        "by_policy_and_object_count": by_policy_and_count,
        "by_condition": by_condition,
        "by_policy_and_condition": by_policy_and_condition,
        "by_policy_condition_and_object_count": by_policy_condition_and_count,
        "graph_vs_flat": paired_graph_flat_comparison(values),
    }


def paired_graph_flat_comparison(results: Iterable[EpisodeResult]) -> dict:
    """Join Graph and Flat outcomes on model seed and environment case."""

    values = list(results)
    representations = {
        id(result): result.representation or result.policy.split("_")[0]
        for result in values
    }
    model_seeds = sorted(
        {
            result.model_seed
            for result in values
            if representations[id(result)] in {"flat", "graph"}
            and result.ablation == "none"
        }
    )
    by_seed: dict[str, dict] = {}
    all_pairs: list[tuple[EpisodeResult, EpisodeResult]] = []
    for model_seed in model_seeds:
        flat = {
            (_result_condition(result), result.seed, result.object_count): result
            for result in values
            if representations[id(result)] == "flat"
            and result.model_seed == model_seed
            and result.ablation == "none"
        }
        graph = {
            (_result_condition(result), result.seed, result.object_count): result
            for result in values
            if representations[id(result)] == "graph"
            and result.model_seed == model_seed
            and result.ablation == "none"
        }
        keys = sorted(set(flat) & set(graph))
        pairs = [(flat[key], graph[key]) for key in keys]
        if not pairs:
            continue
        all_pairs.extend(pairs)

        conditions = sorted({_result_condition(pair[0]) for pair in pairs})

        def metric_delta(
            attribute: str,
            condition: str | None = None,
        ) -> float | None:
            selected = [
                pair
                for pair in pairs
                if condition is None or _result_condition(pair[0]) == condition
            ]
            if not selected:
                return None
            return float(
                np.mean(
                    [
                        float(getattr(graph_result, attribute))
                        - float(getattr(flat_result, attribute))
                        for flat_result, graph_result in selected
                    ]
                )
            )

        condition_success_delta = {
            condition: metric_delta("success", condition) for condition in conditions
        }
        condition_wrong_object_delta = {
            condition: metric_delta("wrong_object", condition) for condition in conditions
        }
        criterion_condition = (
            "crowded_ood" if "crowded_ood" in conditions else "count_ood"
        )

        by_seed[str(model_seed)] = {
            "paired_cases": len(pairs),
            "overall_success_delta": metric_delta("success"),
            "id_success_delta": metric_delta("success", "id_normal"),
            "ood_success_delta": metric_delta("success", criterion_condition),
            "condition_success_delta": condition_success_delta,
            "condition_wrong_object_delta": condition_wrong_object_delta,
        }

    ood_deltas = [
        metrics["ood_success_delta"]
        for metrics in by_seed.values()
        if metrics["ood_success_delta"] is not None
    ]
    id_deltas = [
        metrics["id_success_delta"]
        for metrics in by_seed.values()
        if metrics["id_success_delta"] is not None
    ]
    mean_ood_delta = float(np.mean(ood_deltas)) if ood_deltas else None
    criterion_condition = (
        "crowded_ood"
        if any(
            "crowded_ood" in metrics["condition_success_delta"]
            for metrics in by_seed.values()
        )
        else "count_ood"
    )
    criterion = {
        "at_least_three_model_seeds": len(by_seed) >= 3,
        "ood_gain_at_least_10pp": bool(
            mean_ood_delta is not None and mean_ood_delta >= 0.10
        ),
        "all_model_seeds_improve_ood": bool(
            len(by_seed) >= 3
            and ood_deltas
            and len(ood_deltas) == len(by_seed)
            and all(delta > 0 for delta in ood_deltas)
        ),
        "no_material_id_regression": bool(
            not id_deltas or all(delta >= -0.05 for delta in id_deltas)
        ),
    }
    criterion["all_criteria_met"] = all(criterion.values())
    return {
        "paired_cases": len(all_pairs),
        "by_model_seed": by_seed,
        "mean_ood_success_delta": mean_ood_delta,
        "criterion_condition": criterion_condition,
        "criterion": criterion,
    }


def _stage_join_key(result: EpisodeResult) -> tuple[str, int, str, int, int, str]:
    return (
        result.representation or result.policy.split("_")[0],
        result.model_seed,
        _result_condition(result),
        result.seed,
        result.object_count,
        result.ablation,
    )


def _index_stage_results(
    results: Iterable[EpisodeResult],
    *,
    stage_name: str,
) -> dict[tuple[str, int, str, int, int, str], EpisodeResult]:
    indexed: dict[tuple[str, int, str, int, int, str], EpisodeResult] = {}
    for result in results:
        key = _stage_join_key(result)
        if key in indexed:
            raise ValueError(f"duplicate {stage_name} evaluation case: {key}")
        indexed[key] = result
    return indexed


def compare_training_stages(
    baseline_results: Iterable[EpisodeResult],
    recovery_results: Iterable[EpisodeResult],
) -> dict[str, dict[str, dict]]:
    baseline = _index_stage_results(baseline_results, stage_name="baseline")
    recovery = _index_stage_results(recovery_results, stage_name="recovery")
    if set(baseline) != set(recovery):
        missing = len(set(baseline) - set(recovery))
        extra = len(set(recovery) - set(baseline))
        raise ValueError(
            "baseline and recovery must contain identical paired cases "
            f"(missing={missing}, extra={extra})"
        )

    comparison: dict[str, dict[str, dict]] = {}
    representations_and_seeds = sorted(
        {(key[0], key[1]) for key in baseline}
    )
    for representation, model_seed in representations_and_seeds:
        model_metrics: dict[str, object] = {}
        ablations = sorted(
            {
                key[5]
                for key in baseline
                if key[0] == representation and key[1] == model_seed
            }
        )
        by_ablation: dict[str, dict[str, dict[str, float | int]]] = {}
        for ablation in ablations:
            condition_metrics: dict[str, dict[str, float | int]] = {}
            conditions = sorted(
                {
                    key[2]
                    for key in baseline
                    if key[0] == representation
                    and key[1] == model_seed
                    and key[5] == ablation
                }
            )
            for condition in conditions:
                keys = sorted(
                    key
                    for key in baseline
                    if key[0] == representation
                    and key[1] == model_seed
                    and key[2] == condition
                    and key[5] == ablation
                )
                pairs = [(baseline[key], recovery[key]) for key in keys]
                deltas: dict[str, float | int] = {
                    "paired_cases": len(pairs),
                    "success_delta": float(
                        np.mean([float(new.success) - float(old.success) for old, new in pairs])
                    ),
                    "grasp_delta": float(
                        np.mean([float(new.grasped) - float(old.grasped) for old, new in pairs])
                    ),
                    "wrong_object_delta": float(
                        np.mean(
                            [
                                float(new.wrong_object) - float(old.wrong_object)
                                for old, new in pairs
                            ]
                        )
                    ),
                    "steps_delta": float(
                        np.mean([float(new.steps) - float(old.steps) for old, new in pairs])
                    ),
                }
                condition_metrics[condition] = deltas
                if ablation == "none":
                    for metric_name, value in deltas.items():
                        model_metrics[f"{condition}_{metric_name}"] = value
            by_ablation[ablation] = condition_metrics
        model_metrics["by_ablation"] = by_ablation
        comparison.setdefault(representation, {})[str(model_seed)] = model_metrics
    return comparison


def _parse_csv_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean in {field}: {value!r}")


def load_episode_results_csv(path: str | Path) -> tuple[EpisodeResult, ...]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    results: list[EpisodeResult] = []
    for row in rows:
        action_mse = float(row["on_policy_action_mse"])
        if not math.isfinite(action_mse):
            raise ValueError("on_policy_action_mse must be finite")
        split = row["split"]
        layout_mode = row.get("layout_mode") or "normal"
        condition = row.get("condition") or (
            "id_normal"
            if split == "id"
            else ("crowded_ood" if layout_mode == "crowded" else "count_ood")
        )
        results.append(
            EpisodeResult(
                policy=row["policy"],
                seed=int(row["seed"]),
                object_count=int(row["object_count"]),
                split=split,
                success=_parse_csv_bool(row["success"], field="success"),
                wrong_object=_parse_csv_bool(
                    row["wrong_object"], field="wrong_object"
                ),
                grasped=_parse_csv_bool(row["grasped"], field="grasped"),
                dropped=_parse_csv_bool(row["dropped"], field="dropped"),
                steps=int(row["steps"]),
                termination_reason=row["termination_reason"],
                on_policy_action_mse=action_mse,
                representation=row.get("representation") or row["policy"].split("_")[0],
                model_seed=int(row.get("model_seed") or 0),
                ablation=row.get("ablation") or "none",
                condition=condition,
                layout_mode=layout_mode,
            )
        )
    return tuple(results)


@torch.no_grad()
def rollout_policy(
    name: str,
    policy: ActionPolicy,
    statistics: TrainingStatistics,
    case: EvaluationCase,
    *,
    max_objects: int,
    max_steps: int,
    device: torch.device | str,
    edge_shuffle: bool = False,
    min_object_distance: float = 0.12,
    workspace_low: tuple[float, float, float] = (-0.45, -0.35, 0.04),
    workspace_high: tuple[float, float, float] = (0.45, 0.35, 0.55),
    crowded_anchor_min_distance: float = 0.085,
    crowded_anchor_max_distance: float = 0.105,
    representation: str = "",
    model_seed: int = 0,
) -> EpisodeResult:
    resolved_device = torch.device(device)
    env = KinematicTabletopEnv(
        max_objects=max_objects,
        max_steps=max_steps,
        min_object_distance=min_object_distance,
        workspace_low=workspace_low,
        workspace_high=workspace_high,
        crowded_anchor_min_distance=crowded_anchor_min_distance,
        crowded_anchor_max_distance=crowded_anchor_max_distance,
    )
    builder = SceneGraphBuilder(max_objects=max_objects)
    expert = ScriptedExpert()
    expert.reset()
    snapshot = env.reset(
        seed=case.seed,
        object_count=case.object_count,
        layout_mode=case.layout_mode,
    )
    policy.to(resolved_device).eval()
    squared_errors: list[float] = []
    grasped = False
    dropped = False
    reason = TerminationReason.TIMEOUT

    for step_index in range(max_steps):
        expert_action = expert.act(snapshot)
        scene = scene_graphs_to_batch((builder.build(snapshot),)).to(resolved_device)
        scene = statistics.normalize_scene(scene)
        if edge_shuffle:
            scene = shuffle_valid_edge_assignments(
                scene, seed=case.seed * 1009 + step_index
            )
        proprioception = torch.from_numpy(env.proprioception()).float().unsqueeze(0).to(resolved_device)
        proprioception = statistics.normalize_proprioception(proprioception)
        action = policy(
            scene if policy.scene_encoder is not None else None, proprioception
        )[0].cpu().numpy()
        squared_errors.append(float(np.mean(np.square(action - expert_action))))
        transition = env.step(action)
        snapshot = transition.snapshot
        grasped = grasped or bool(transition.info["ever_grasped"])
        dropped = dropped or int(transition.info["drop_count"]) > 0
        reason = transition.reason
        if transition.done:
            break

    return EpisodeResult(
        policy=name,
        seed=case.seed,
        object_count=case.object_count,
        split=case.split,
        success=reason is TerminationReason.SUCCESS,
        wrong_object=reason is TerminationReason.WRONG_OBJECT,
        grasped=grasped,
        dropped=dropped or reason is TerminationReason.DROPPED,
        steps=env.step_count,
        termination_reason=reason.value,
        on_policy_action_mse=float(np.mean(squared_errors)),
        representation=representation or name.split("_")[0],
        model_seed=model_seed,
        ablation="edge_shuffled" if edge_shuffle else "none",
        condition=case.condition,
        layout_mode=case.layout_mode,
    )


@torch.no_grad()
def evaluate_offline_action_mse(
    policy: ActionPolicy,
    statistics: TrainingStatistics,
    episode_paths: Iterable[str | Path],
    *,
    device: torch.device | str,
    edge_shuffle: bool = False,
    shuffle_seed: int = 0,
) -> float:
    """Measure physical action MSE on fixed held-out expert observations."""

    resolved_device = torch.device(device)
    dataset = EpisodeFrameDataset(episode_paths, statistics)
    scene = dataset.scene_batch().to(resolved_device)
    if edge_shuffle:
        scene = shuffle_valid_edge_assignments(scene, seed=shuffle_seed)
    proprioception = dataset.proprioception.to(resolved_device)
    policy.to(resolved_device).eval()
    predictions = policy(
        scene if policy.scene_encoder is not None else None, proprioception
    )
    targets = dataset.actions.to(resolved_device)
    return float(torch.mean((predictions - targets).square()).item())


@torch.no_grad()
def evaluate_offline_normalized_action_mse(
    policy: ActionPolicy,
    statistics: TrainingStatistics,
    episode_paths: Iterable[str | Path],
    *,
    device: torch.device | str,
    edge_shuffle: bool = False,
    shuffle_seed: int = 0,
) -> float:
    resolved_device = torch.device(device)
    dataset = EpisodeFrameDataset(episode_paths, statistics)
    scene = dataset.scene_batch().to(resolved_device)
    if edge_shuffle:
        scene = shuffle_valid_edge_assignments(scene, seed=shuffle_seed)
    proprioception = dataset.proprioception.to(resolved_device)
    policy.to(resolved_device).eval()
    predictions = policy(
        scene if policy.scene_encoder is not None else None, proprioception
    )
    targets = dataset.actions.to(resolved_device)
    normalized_error = (
        statistics.normalize_actions(predictions)
        - statistics.normalize_actions(targets)
    )
    return float(torch.mean(normalized_error.square()).item())


def evaluate_policies(
    policies: Mapping[str, tuple[ActionPolicy, TrainingStatistics]],
    cases: Iterable[EvaluationCase],
    *,
    max_objects: int,
    max_steps: int,
    device: torch.device | str,
    edge_shuffled_policies: Iterable[str] = (),
    min_object_distance: float = 0.12,
    workspace_low: tuple[float, float, float] = (-0.45, -0.35, 0.04),
    workspace_high: tuple[float, float, float] = (0.45, 0.35, 0.55),
    crowded_anchor_min_distance: float = 0.085,
    crowded_anchor_max_distance: float = 0.105,
    representations: Mapping[str, str] | None = None,
    model_seeds: Mapping[str, int] | None = None,
) -> list[EpisodeResult]:
    paired_cases = tuple(cases)
    shuffled_names = set(edge_shuffled_policies)
    representation_by_name = dict(representations or {})
    seed_by_name = dict(model_seeds or {})
    return [
        rollout_policy(
            name,
            policy,
            statistics,
            case,
            max_objects=max_objects,
            max_steps=max_steps,
            device=device,
            edge_shuffle=name in shuffled_names,
            min_object_distance=min_object_distance,
            workspace_low=workspace_low,
            workspace_high=workspace_high,
            crowded_anchor_min_distance=crowded_anchor_min_distance,
            crowded_anchor_max_distance=crowded_anchor_max_distance,
            representation=representation_by_name.get(name, name.split("_")[0]),
            model_seed=seed_by_name.get(name, 0),
        )
        for name, (policy, statistics) in policies.items()
        for case in paired_cases
    ]


def make_evaluation_cases(
    *,
    object_counts: Iterable[int],
    ood_object_counts: Iterable[int],
    episodes_per_count: int,
    base_seed: int,
) -> tuple[EvaluationCase, ...]:
    ood = set(ood_object_counts)
    return tuple(
        EvaluationCase(
            seed=base_seed + object_count * 10_000 + episode_index,
            object_count=object_count,
            split="ood" if object_count in ood else "id",
            condition="count_ood" if object_count in ood else "id_normal",
            layout_mode="normal",
        )
        for object_count in object_counts
        for episode_index in range(episodes_per_count)
    )


def make_conditioned_evaluation_cases(
    *,
    id_object_counts: Iterable[int],
    count_ood_object_counts: Iterable[int],
    crowded_object_counts: Iterable[int],
    episodes_per_count: int,
    base_seed: int,
) -> tuple[EvaluationCase, ...]:
    if episodes_per_count < 1:
        raise ValueError("episodes_per_count must be positive")
    condition_specs = (
        ("id_normal", tuple(id_object_counts), "id", "normal", 0),
        ("count_ood", tuple(count_ood_object_counts), "ood", "normal", 1_000_000),
        ("crowded_ood", tuple(crowded_object_counts), "ood", "crowded", 2_000_000),
    )
    return tuple(
        EvaluationCase(
            seed=base_seed + seed_offset + object_count * 10_000 + episode_index,
            object_count=object_count,
            split=split,
            condition=condition,
            layout_mode=layout_mode,
        )
        for condition, object_counts, split, layout_mode, seed_offset in condition_specs
        for object_count in object_counts
        for episode_index in range(episodes_per_count)
    )


def save_evaluation_report(
    results: Iterable[EpisodeResult],
    output_dir: str | Path,
    *,
    offline_physical_action_mse_by_policy: Mapping[str, float] | None = None,
    offline_normalized_action_mse_by_policy: Mapping[str, float] | None = None,
    recovery_vs_baseline: Mapping[str, object] | None = None,
) -> Path:
    values = list(results)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "episodes.csv"
    if values:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(values[0])))
            writer.writeheader()
            writer.writerows(asdict(result) for result in values)
    report_path = destination / "report.json"
    report = aggregate_results(values)
    if offline_physical_action_mse_by_policy is not None:
        report["offline_physical_action_mse_by_policy"] = dict(
            offline_physical_action_mse_by_policy
        )
    if offline_normalized_action_mse_by_policy is not None:
        report["offline_normalized_action_mse_by_policy"] = dict(
            offline_normalized_action_mse_by_policy
        )
    if recovery_vs_baseline is not None:
        report["recovery_vs_baseline"] = dict(recovery_vs_baseline)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def evaluate_from_config(
    config_path: str | Path,
    checkpoint_paths: Iterable[str | Path],
    baseline_episodes: str | Path | None = None,
) -> Path:
    cfg = load_config(config_path)
    device = resolve_device(cfg.device)
    policies: dict[str, tuple[ActionPolicy, TrainingStatistics]] = {}
    shuffled_names: set[str] = set()
    representations: dict[str, str] = {}
    model_seeds: dict[str, int] = {}
    for checkpoint_path in checkpoint_paths:
        policy, statistics, payload = load_training_checkpoint(checkpoint_path, device)
        name = str(payload["representation"])
        if name in policies:
            name = f"{name}_{Path(checkpoint_path).parent.name}"
        policies[name] = (policy, statistics)
        representations[name] = str(payload["representation"])
        model_seeds[name] = int(payload.get("model_seed", 0))
        if payload["representation"] == "graph":
            shuffled_name = f"{name}_edge_shuffled"
            shuffled_policy, shuffled_statistics, _ = load_training_checkpoint(
                checkpoint_path, device
            )
            policies[shuffled_name] = (shuffled_policy, shuffled_statistics)
            shuffled_names.add(shuffled_name)
            representations[shuffled_name] = "graph"
            model_seeds[shuffled_name] = int(payload.get("model_seed", 0))

    cases = make_conditioned_evaluation_cases(
        id_object_counts=tuple(
            count
            for count in cfg.eval.object_counts
            if count not in cfg.eval.ood_object_counts
        ),
        count_ood_object_counts=cfg.eval.ood_object_counts,
        crowded_object_counts=cfg.eval.crowded_object_counts,
        episodes_per_count=cfg.eval.episodes_per_count,
        base_seed=cfg.seed + 100_000,
    )
    results = evaluate_policies(
        policies,
        cases,
        max_objects=cfg.max_objects,
        max_steps=cfg.eval.max_steps,
        device=device,
        edge_shuffled_policies=shuffled_names,
        min_object_distance=cfg.environment.min_object_distance,
        workspace_low=cfg.environment.workspace_low,
        workspace_high=cfg.environment.workspace_high,
        crowded_anchor_min_distance=cfg.environment.crowded_anchor_min_distance,
        crowded_anchor_max_distance=cfg.environment.crowded_anchor_max_distance,
        representations=representations,
        model_seeds=model_seeds,
    )
    episode_paths = episode_paths_from_manifest(cfg.data_dir)
    seed_by_path = {path: load_episode_arrays(path).seed for path in episode_paths}
    splits = split_episode_seeds(
        seed_by_path.values(), validation_fraction=0.1, test_fraction=0.1, seed=cfg.seed
    )
    test_seeds = set(splits.test)
    test_paths = [path for path, seed in seed_by_path.items() if seed in test_seeds]
    if not test_paths:
        raise ValueError("the episode-level test split is empty")
    offline_physical_metrics = {
        name: evaluate_offline_action_mse(
            policy,
            statistics,
            test_paths,
            device=device,
            edge_shuffle=name in shuffled_names,
            shuffle_seed=cfg.seed,
        )
        for name, (policy, statistics) in policies.items()
    }
    offline_normalized_metrics = {
        name: evaluate_offline_normalized_action_mse(
            policy,
            statistics,
            test_paths,
            device=device,
            edge_shuffle=name in shuffled_names,
            shuffle_seed=cfg.seed,
        )
        for name, (policy, statistics) in policies.items()
    }
    recovery_vs_baseline = None
    if baseline_episodes is not None:
        recovery_vs_baseline = compare_training_stages(
            load_episode_results_csv(baseline_episodes), results
        )
    return save_evaluation_report(
        results,
        Path(cfg.output_dir) / "evaluation",
        offline_physical_action_mse_by_policy=offline_physical_metrics,
        offline_normalized_action_mse_by_policy=offline_normalized_metrics,
        recovery_vs_baseline=recovery_vs_baseline,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired closed-loop policy evaluation")
    parser.add_argument("--config", default="configs/pilot_macos.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--baseline-episodes",
        help="optional Stage A episodes.csv for exact recovery-versus-baseline pairing",
    )
    args = parser.parse_args()
    print(
        evaluate_from_config(
            args.config,
            args.checkpoints,
            baseline_episodes=args.baseline_episodes,
        )
    )


if __name__ == "__main__":
    main()
