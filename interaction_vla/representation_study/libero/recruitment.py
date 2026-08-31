from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from ..statistics import clustered_bootstrap_mean
from .config import LiberoStudyConfig
from .crossfit_probes import (
    CrossFitManifest,
    _crossfit_seed,
    crossfit_partition_indices,
)
from .feature_binding import LIBERO_SMOLVLA_RENAME_MAP
from .latents import (
    _file_sha256,
    _tree_sha256,
    collate_state_bank_observations,
    deterministic_inference_noise,
    load_latent_cache,
)
from .longitudinal import condition_from_plan, load_longitudinal_plan
from .probe_runner import factor_target
from .probes import run_linear_probe
from .schema import StateRecord
from .state_bank import load_state_bank


PRIMARY_CONDITIONS = (
    "pretrained",
    "d25_u16070",
    "d100_u16617",
    "d100_u66470",
)
PRIMARY_FACTOR = "stable_grasp"
PRIMARY_TAP = "action_expert_input"
CONTROL_FACTORS = ("contact", "phase", "geometry")
RECRUITMENT_SCHEMA = "libero_stablegrasp_recruitment_v1"


def _profile_root(protocol_root: Path, max_states: int) -> Path:
    return protocol_root / "recruitment" / PRIMARY_FACTOR / f"n_{max_states:04d}"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("crossfit_probes.py"),
        Path(__file__).with_name("probes.py"),
        Path(__file__).with_name("taps.py"),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(_file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _episode_key(record: StateRecord) -> str:
    return f"{record.suite}:{record.task_id}:{record.source_episode_id}"


def _task_key(record: StateRecord) -> str:
    return f"{record.suite}:{record.task_id}"


def phase_stratum(phase: str | None) -> str:
    mapping = {
        "approach": "pre_contact",
        "align_precontact": "pre_contact",
        "contact": "contact_grasp",
        "secure": "contact_grasp",
        "lift": "post_grasp",
        "transport": "post_grasp",
        "place": "place_release",
        "release_retreat": "place_release",
    }
    if phase not in mapping:
        raise ValueError(f"phase has no recruitment stratum: {phase}")
    return mapping[phase]


def binary_raw_probe(
    *,
    weight: np.ndarray,
    bias: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> tuple[np.ndarray, float]:
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    mean = np.asarray(feature_mean, dtype=np.float64)
    scale = np.asarray(feature_scale, dtype=np.float64)
    if (
        weight.shape != (2, mean.size)
        or bias.shape != (2,)
        or scale.shape != mean.shape
        or np.any(scale <= 0)
        or not all(np.isfinite(value).all() for value in (weight, bias, mean, scale))
    ):
        raise ValueError("binary probe parameters are incompatible")
    direction = (weight[1] - weight[0]) / scale
    raw_bias = float(bias[1] - bias[0] - np.dot(mean, direction))
    if np.linalg.norm(direction) <= 1e-12:
        raise ValueError("binary probe has an empty decision direction")
    return direction, raw_bias


def consensus_direction(directions: np.ndarray) -> np.ndarray:
    values = np.asarray(directions, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("consensus requires finite probe directions")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("consensus contains an empty direction")
    aligned = values / norms[:, None]
    reference = aligned[0]
    aligned = np.where((aligned @ reference)[:, None] < 0, -aligned, aligned)
    _, _, right = np.linalg.svd(aligned, full_matrices=False)
    result = right[0]
    if np.dot(result, reference) < 0:
        result = -result
    return result / np.linalg.norm(result)


def same_norm_random_delta(
    target_delta: np.ndarray, *, target_direction: np.ndarray, seed: int
) -> np.ndarray:
    delta = np.asarray(target_delta, dtype=np.float64)
    direction = np.asarray(target_direction, dtype=np.float64)
    if delta.ndim != 2 or direction.shape != (delta.shape[1],):
        raise ValueError("matched-random delta and direction dimensions differ")
    direction = direction / np.linalg.norm(direction)
    rng = np.random.default_rng(seed)
    random = rng.normal(size=direction.shape)
    random -= np.dot(random, direction) * direction
    norm = np.linalg.norm(random)
    if norm <= 1e-12:
        raise ValueError("matched-random direction is degenerate")
    random /= norm
    return np.linalg.norm(delta, axis=1)[:, None] * random[None, :]


class FinalDenoisingDeltaHook(AbstractContextManager["FinalDenoisingDeltaHook"]):
    """Apply one pooled feature delta to every action token on the final call."""

    def __init__(
        self,
        module: nn.Module,
        *,
        expected_calls: int,
        delta: torch.Tensor,
        expected_pooled: torch.Tensor | None = None,
        zero: bool = False,
        atol: float = 2e-4,
    ) -> None:
        if expected_calls <= 0 or delta.ndim != 2:
            raise ValueError("final-call intervention requires calls and [B,D] delta")
        self.module = module
        self.expected_calls = expected_calls
        self.delta = delta
        self.expected_pooled = expected_pooled
        self.zero = zero
        self.atol = atol
        self.calls = 0
        self.final_pooled: torch.Tensor | None = None
        self._handle = None

    def __enter__(self) -> "FinalDenoisingDeltaHook":
        def hook(_module: nn.Module, _inputs: object, output: torch.Tensor) -> torch.Tensor:
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise ValueError("action_time_mlp_out must return [B,T,D]")
            call = self.calls
            self.calls += 1
            if call != self.expected_calls - 1:
                return output
            pooled = output.mean(dim=1)
            self.final_pooled = pooled.detach().to("cpu", torch.float32)
            if self.expected_pooled is not None and not torch.allclose(
                self.final_pooled,
                self.expected_pooled.detach().to("cpu", torch.float32),
                atol=self.atol,
                rtol=1e-4,
            ):
                error = float(
                    torch.max(
                        torch.abs(
                            self.final_pooled
                            - self.expected_pooled.detach().to("cpu", torch.float32)
                        )
                    ).item()
                )
                raise ValueError(
                    f"live action-expert tensor does not match Protocol-v3 cache: {error}"
                )
            if self.zero:
                return torch.zeros_like(output)
            change = self.delta.to(device=output.device, dtype=output.dtype)
            if change.shape != pooled.shape:
                raise ValueError("intervention delta does not match pooled action-expert shape")
            return output + change[:, None, :]

        self._handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self._handle is not None
        self._handle.remove()
        if exc_type is None and self.calls != self.expected_calls:
            raise ValueError(
                f"SmolVLA denoising calls changed: expected {self.expected_calls}, got {self.calls}"
            )
        return None


def validate_probe_reconstruction(
    *,
    result: Mapping[str, object],
    test_state_ids: Sequence[str],
    archived_result: Mapping[str, object],
    seed_offset: int,
    atol: float = 1e-6,
) -> None:
    payload = archived_result.get("paired_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("archived cross-fit cell has no paired payload")
    archived_ids = tuple(str(value) for value in payload.get("state_ids", ()))
    index = {state_id: row for row, state_id in enumerate(archived_ids)}
    if len(index) != len(archived_ids) or any(value not in index for value in test_state_ids):
        raise ValueError("probe reconstruction state IDs differ from the archive")
    replicates = [
        row
        for row in payload.get("replicates", ())
        if isinstance(row, Mapping) and int(row.get("seed_offset", -1)) == seed_offset
    ]
    if len(replicates) != 1:
        raise ValueError("archived probe replicate is missing or duplicated")
    rows = np.asarray([index[value] for value in test_state_ids], dtype=np.int64)
    expected_prediction = np.asarray(replicates[0]["prediction"])[rows]
    prediction = np.asarray(result["test_prediction"])
    continuous = expected_prediction.ndim == 2
    prediction_matches = (
        np.allclose(prediction, expected_prediction, atol=atol, rtol=0)
        if continuous and prediction.shape == expected_prediction.shape
        else np.array_equal(prediction, expected_prediction)
    )
    if prediction.shape != expected_prediction.shape or not prediction_matches:
        raise ValueError("reconstructed held-out probe prediction changed")
    expected_score = replicates[0].get("score")
    score = result.get("test_score")
    if expected_score is None and score is None:
        return
    if expected_score is None or score is None:
        raise ValueError("reconstructed held-out probe score availability changed")
    expected = np.asarray(expected_score, dtype=np.float64)[rows]
    actual = np.asarray(score, dtype=np.float64)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=atol, rtol=0):
        raise ValueError("reconstructed held-out probe score changed")


def specificity_gate(
    *,
    target_minus_random: Mapping[str, float],
    target_effect: float,
    non_target_effects: Mapping[str, float],
    activation_norm_ratio: float,
    place_target_minus_random: Mapping[str, float] | None,
) -> dict[str, object]:
    failures: list[str] = []
    if float(target_minus_random.get("ci_low", -np.inf)) <= 0:
        failures.append("StableGrasp disruption does not exceed matched random")
    for factor in ("phase", "contact"):
        if target_effect <= float(non_target_effects.get(factor, np.inf)):
            failures.append(f"StableGrasp disruption does not exceed {factor}")
    if place_target_minus_random is None or float(
        place_target_minus_random.get("ci_low", -np.inf)
    ) <= 0:
        failures.append("place-only StableGrasp specificity is not supported")
    if not 0.8 <= activation_norm_ratio <= 1.2:
        failures.append("intervened activation norm is outside the preregistered support band")
    return {"passed": not failures, "failures": failures}


@dataclass(frozen=True)
class _Probe:
    factor: str
    weight: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    output_scale: np.ndarray
    direction: np.ndarray | None
    direction_bias: float | None
    train_mean: np.ndarray

    def disruption(self, source: np.ndarray, changed: np.ndarray) -> np.ndarray:
        delta = np.asarray(changed, dtype=np.float64) - np.asarray(
            source, dtype=np.float64
        )
        raw_weight = self.weight / self.feature_scale[None, :]
        if self.direction is not None:
            return np.abs(delta @ self.direction) / float(self.output_scale[0])
        output = delta @ raw_weight.T
        if self.factor == "geometry":
            return np.mean(np.abs(output) / self.output_scale[None, :], axis=1)
        output -= output.mean(axis=1, keepdims=True)
        return np.linalg.norm(output, axis=1) / float(self.output_scale[0])


def _load_cell(root: Path, condition: str, factor: str) -> dict[str, object]:
    path = (
        root
        / "probes"
        / "crossfit_v1"
        / "cells"
        / condition
        / PRIMARY_TAP
        / "episode_group"
        / f"{factor}.json.gz"
    )
    value = json.loads(gzip.decompress(path.read_bytes()))
    if not isinstance(value, dict) or value.get("result", {}).get("status") != "complete":
        raise ValueError(f"required cross-fit cell is incomplete: {path}")
    return value


def _factor_data(
    records: Sequence[StateRecord], factor: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    applicable = np.asarray(
        [bool(getattr(record.labels.applicability, factor)) for record in records]
    )
    source = np.flatnonzero(applicable)
    raw = [factor_target(records[index], factor) for index in source]
    targets = (
        np.stack(raw).astype(np.float32)
        if factor == "geometry"
        else np.asarray(["|".join(x) if isinstance(x, tuple) else x for x in raw])
    )
    return applicable, source, targets


def _probe_from_result(
    *,
    factor: str,
    result: Mapping[str, object],
    train_features: np.ndarray,
    train_targets: np.ndarray,
) -> _Probe:
    weight = np.asarray(result["weight"], dtype=np.float64)
    bias = np.asarray(result["bias"], dtype=np.float64)
    mean = np.asarray(result["feature_mean"], dtype=np.float64)
    scale = np.asarray(result["feature_scale"], dtype=np.float64)
    raw_weight = weight / scale[None, :]
    raw_bias = bias - raw_weight @ mean
    outputs = train_features @ raw_weight.T + raw_bias
    direction = None
    direction_bias = None
    if factor in {"stable_grasp", "contact"}:
        direction, direction_bias = binary_raw_probe(
            weight=weight, bias=bias, feature_mean=mean, feature_scale=scale
        )
        margin = train_features @ direction + direction_bias
        output_scale = np.asarray([max(float(np.std(margin)), 1e-8)])
    elif factor == "geometry":
        target = np.asarray(train_targets, dtype=np.float64)
        output_scale = np.maximum(np.ptp(target, axis=0), 1e-8)
    else:
        centered = outputs - outputs.mean(axis=1, keepdims=True)
        output_scale = np.asarray(
            [max(float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))), 1e-8)]
        )
    return _Probe(
        factor=factor,
        weight=weight,
        bias=bias,
        feature_mean=mean,
        feature_scale=scale,
        output_scale=output_scale,
        direction=direction,
        direction_bias=direction_bias,
        train_mean=train_features.mean(axis=0),
    )


def _reconstruct_fold_probes(
    *,
    config: LiberoStudyConfig,
    records: Sequence[StateRecord],
    features: np.ndarray,
    manifest: CrossFitManifest,
    condition: str,
    factor: str,
    cell: Mapping[str, object],
    fold: int,
    offsets: Sequence[int],
) -> dict[int, _Probe]:
    applicable, source, targets = _factor_data(records, factor)
    source_to_local = {int(row): index for index, row in enumerate(source)}
    parts = crossfit_partition_indices(records, manifest, fold=fold, applicable=applicable)
    local = {
        name: [source_to_local[index] for index in indices]
        for name, indices in parts.items()
    }
    task = "regression" if factor == "geometry" else "classification"
    archived = cell["result"]
    result: dict[int, _Probe] = {}
    for offset in offsets:
        seed = _crossfit_seed(
            base_seed=config.seed,
            tap=PRIMARY_TAP,
            factor=factor,
            split_name="episode_group",
            fold=fold,
            seed_offset=offset,
        )
        fitted = run_linear_probe(
            features[applicable],
            targets,
            train_indices=local["train"],
            validation_indices=local["validation"],
            test_indices=local["test"],
            task=task,
            seed=seed,
            l2_grid=config.probes.linear_l2,
            epochs=config.probes.linear_epochs,
            selection_metric=(
                "normalized_mae"
                if factor == "geometry"
                else "auprc"
                if factor in {"stable_grasp", "contact"}
                else "macro_f1"
            ),
            device="cpu" if factor == "geometry" else ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        validate_probe_reconstruction(
            result=fitted,
            test_state_ids=tuple(records[index].state_id for index in parts["test"]),
            archived_result=archived,
            seed_offset=offset,
        )
        train_index = np.asarray(local["train"], dtype=np.int64)
        result[offset] = _probe_from_result(
            factor=factor,
            result=fitted,
            train_features=features[applicable][train_index],
            train_targets=targets[train_index],
        )
    return result


def _select_records(records: Sequence[StateRecord], max_states: int) -> tuple[int, ...]:
    if max_states <= 0:
        raise ValueError("recruitment max_states must be positive")
    buckets: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if not record.labels.applicability.stable_grasp:
            continue
        key = f"{_task_key(record)}:{phase_stratum(record.labels.phase)}"
        buckets.setdefault(key, []).append(index)
    for key, values in buckets.items():
        values.sort(
            key=lambda index: hashlib.sha256(
                f"{key}:{records[index].state_id}".encode("utf-8")
            ).digest()
        )
    selected: list[int] = []
    cursor = 0
    keys = sorted(buckets)
    while len(selected) < min(max_states, sum(map(len, buckets.values()))):
        added = False
        for key in keys:
            if cursor < len(buckets[key]):
                selected.append(buckets[key][cursor])
                added = True
                if len(selected) >= max_states:
                    break
        if not added:
            break
        cursor += 1
    return tuple(selected)


def _cluster_interval(
    values: np.ndarray,
    records: Sequence[StateRecord],
    indices: Sequence[int],
    *,
    config: LiberoStudyConfig,
    task: bool = False,
) -> dict[str, float]:
    clusters = [
        _task_key(records[index]) if task else _episode_key(records[index])
        for index in indices
    ]
    return clustered_bootstrap_mean(
        values,
        clusters,
        samples=max(100, config.probes.bootstrap_samples),
        confidence=config.probes.confidence_level,
        seed=config.seed + (1 if task else 0),
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _fold_manifest(root: Path) -> CrossFitManifest:
    value = json.loads((root / "probes" / "crossfit_v1" / "folds.json").read_text())
    row = value["manifests"]["episode_group"]
    return CrossFitManifest(
        split_name=str(row["split_name"]),
        group_unit=str(row["group_unit"]),
        folds=int(row["folds"]),
        seed=int(row["seed"]),
        group_folds={str(key): int(value) for key, value in row["group_folds"].items()},
    )


def _read_accessibility(root: Path) -> dict[str, dict[str, object]]:
    report = json.loads((root / "probes" / "crossfit_v1" / "report.json").read_text())
    result: dict[str, dict[str, object]] = {}
    for split in ("episode_group", "task_group"):
        rows = [
            row
            for row in report["grids"][split]
            if row["condition"] in PRIMARY_CONDITIONS
            and row["tap"] == PRIMARY_TAP
            and row["factor"] == PRIMARY_FACTOR
        ]
        result[split] = {
            str(row["condition"]): {
                "auprc": row.get("primary_metric"),
                "utility": row.get("accessibility_utility"),
                "utility_ci": row.get("accessibility_utility_ci"),
                "accessible": row.get("accessible"),
            }
            for row in rows
        }
    return result


def _matched_training_audit(
    d25: Mapping[str, object],
    d100: Mapping[str, object],
    *,
    d25_updates: int,
    d100_updates: int,
) -> dict[str, object]:
    top_level = (
        "seed",
        "batch_size",
        "num_workers",
        "cudnn_deterministic",
        "optimizer",
        "scheduler",
        "policy",
        "rename_map",
    )
    matched = [key for key in top_level if d25.get(key) == d100.get(key)]
    mismatched = [key for key in top_level if d25.get(key) != d100.get(key)]
    left_dataset = d25.get("dataset")
    right_dataset = d100.get("dataset")
    if not isinstance(left_dataset, Mapping) or not isinstance(right_dataset, Mapping):
        raise ValueError("checkpoint training configs have no dataset mappings")
    dataset_fields = tuple(
        sorted((set(left_dataset) & set(right_dataset)) - {"episodes"})
    )
    dataset_mismatched = [
        key for key in dataset_fields if left_dataset.get(key) != right_dataset.get(key)
    ]
    left_episodes = {int(value) for value in left_dataset.get("episodes", ())}
    right_episodes = {int(value) for value in right_dataset.get("episodes", ())}
    return {
        "passed": not mismatched
        and not dataset_mismatched
        and bool(left_episodes)
        and left_episodes < right_episodes
        and d25_updates > 0
        and d100_updates > 0,
        "matched_top_level": matched,
        "mismatched_top_level": mismatched,
        "mismatched_dataset_metadata": dataset_mismatched,
        "nested_episode_subsets": left_episodes < right_episodes,
        "d25_episodes": len(left_episodes),
        "d100_episodes": len(right_episodes),
        "updates": {"d25": d25_updates, "d100": d100_updates},
        "relative_update_difference": (d100_updates - d25_updates) / d25_updates,
        "planned_training_steps": {
            "d25": int(d25.get("steps", 0)),
            "d100": int(d100.get("steps", 0)),
        },
        "not_matched": ["dataset coverage", "sample repetition", "completed epochs"],
    }


def audit_longitudinal_recruitment(config: LiberoStudyConfig) -> dict[str, object]:
    root = Path(config.output_dir) / "protocol_v3"
    plan = load_longitudinal_plan(config)
    conditions = {
        str(row["condition"]): row
        for row in plan["conditions"]
        if row["condition"] in PRIMARY_CONDITIONS
    }
    if tuple(name for name in PRIMARY_CONDITIONS if name in conditions) != PRIMARY_CONDITIONS:
        raise ValueError("the four recruitment checkpoints are not available")
    latent_reports = {}
    cells = {}
    for condition in PRIMARY_CONDITIONS:
        report_path = root / "latents" / condition / "report.json"
        report = json.loads(report_path.read_text())
        metadata = report["tap_metadata"][PRIMARY_TAP]
        if metadata != {
            "module": "model.action_time_mlp_out",
            "pooling": "action_chunk_mean",
            "call_selection": "final_denoising",
            "shape_semantics": "batch_dimension_excluded",
            "raw_shape": [50, 720],
            "pooled_shape": [720],
        }:
            raise ValueError(f"action-expert intervention tensor changed: {condition}")
        latent_reports[condition] = {
            "report_sha256": _file_sha256(report_path),
            "tap_metadata": metadata,
        }
        cells[condition] = {}
        for factor in (PRIMARY_FACTOR, *CONTROL_FACTORS):
            path = (
                root
                / "probes/crossfit_v1/cells"
                / condition
                / PRIMARY_TAP
                / "episode_group"
                / f"{factor}.json.gz"
            )
            cells[condition][factor] = {
                "path": str(path),
                "sha256": _file_sha256(path),
            }
    records, bank, _, _ = load_state_bank(Path(config.output_dir) / "state_bank")
    d25_config = json.loads(
        (Path(str(conditions["d25_u16070"]["checkpoint"])) / "train_config.json").read_text()
    )
    d100_config = json.loads(
        (Path(str(conditions["d100_u16617"]["checkpoint"])) / "train_config.json").read_text()
    )
    matched_training = _matched_training_audit(
        d25_config,
        d100_config,
        d25_updates=int(conditions["d25_u16070"]["step"]),
        d100_updates=int(conditions["d100_u16617"]["step"]),
    )
    if not matched_training["passed"]:
        raise ValueError(f"D25/D100 matched-update audit failed: {matched_training}")
    phase_counts: dict[str, dict[str, int]] = {}
    for record in records:
        if record.labels.applicability.stable_grasp:
            phase_counts.setdefault(str(record.labels.phase), {"false": 0, "true": 0})[
                str(bool(record.labels.stable_grasp)).lower()
            ] += 1
    report = {
        "schema_version": RECRUITMENT_SCHEMA,
        "passed": True,
        "status": "ready_for_specificity",
        "conditions": conditions,
        "matched_update_audit": matched_training,
        "accessibility": _read_accessibility(root),
        "baseline_policy_success": {condition: "not_run" for condition in PRIMARY_CONDITIONS},
        "tap": latent_reports,
        "intervention_contract": {
            "module": "model.action_time_mlp_out",
            "raw_shape": [50, 720],
            "denoising_call": "final_of_10",
            "token_positions": "all_50_action_tokens",
            "rank": 1,
            "primary_control": "orthogonal_same_rank_same_norm_random",
            "secondary_controls": ["matched_mean", "zero_ood"],
        },
        "crossfit_cells": cells,
        "state_bank_sha256": _file_sha256(Path(config.output_dir) / "state_bank/manifest.json"),
        "state_bank_states": int(bank["states"]),
        "stable_grasp_by_phase": phase_counts,
        "identifiability_warning": "StableGrasp is phase-determined outside place; Phase and place-only specificity are mandatory.",
        "statistics": {"primary_cluster": "episode", "robustness_cluster": "task"},
        "offline_cells": {"primary": 12, "with_secondary_controls": 20},
        "closed_loop": {
            "status": "not_run",
            "authorized": False,
            "estimated_baseline_screen_rollouts": 240,
            "maximum_paired_rollouts_if_all_checkpoints_eligible": 1200,
            "current_rollout_budget": 0,
        },
        "identifiability": {
            "longitudinal_recruitment": "estimable offline with fold-held-out bases",
            "coverage_contrast": "approximately matched updates, not pure causal isolation",
            "end_to_end_vla_scope": "not identifiable because upstream vision is frozen",
            "stablegrasp_specificity": "conditional on passing Phase and place-only gates",
        },
        "implementation_files": [
            "interaction_vla/representation_study/libero/recruitment.py",
            "interaction_vla/representation_study/libero/cli.py",
            "tests/interaction_vla/representation_study/libero/test_recruitment.py",
            "tests/interaction_vla/representation_study/libero/test_cli.py",
            "README.md",
            "SERVER_RUNBOOK.md",
            "ccfa.yaml",
        ],
        "gates": [
            "probe_reconstruction_categorical_exact_continuous_atol_1e-6",
            "stablegrasp_factor_specificity",
            "stablegrasp_longitudinal_action_sensitivity_vs_matched_random",
        ],
    }
    output = root / "recruitment" / PRIMARY_FACTOR / "audit.json"
    write_json_atomic(output, report)
    return report


def _target_delta(
    features: np.ndarray,
    probes: Sequence[_Probe],
    direction: np.ndarray,
    cap: float,
) -> tuple[np.ndarray, float]:
    shifts = []
    for probe in probes:
        assert probe.direction is not None and probe.direction_bias is not None
        denominator = float(np.dot(probe.direction, direction))
        if abs(denominator) <= 1e-10:
            raise ValueError("consensus direction is orthogonal to a seed probe")
        shifts.append(
            -(features @ probe.direction + probe.direction_bias) / denominator
        )
    scalar = np.median(np.stack(shifts), axis=0)
    capped = float(np.mean(np.abs(scalar) > cap))
    scalar = np.clip(scalar, -cap, cap)
    return scalar[:, None] * direction[None, :], capped


def _specificity(
    config: LiberoStudyConfig, *, max_states: int
) -> dict[str, object]:
    root = Path(config.output_dir) / "protocol_v3"
    profile_root = _profile_root(root, max_states)
    report_path = profile_root / "specificity.json"
    binding = _canonical_sha256(
        {
            "audit_sha256": _file_sha256(
                root / "recruitment" / PRIMARY_FACTOR / "audit.json"
            ),
            "config_sha256": _file_sha256(config.source_path),
            "implementation_sha256": _implementation_sha256(),
            "max_states": max_states,
        }
    )
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") != RECRUITMENT_SCHEMA
            or existing.get("max_states") != max_states
            or existing.get("binding_sha256") != binding
        ):
            raise FileExistsError(f"specificity profile has a different binding: {report_path}")
        for row in existing.get("conditions", {}).values():
            artifact = Path(str(row["intervention_artifact"]))
            if _file_sha256(artifact) != row["intervention_sha256"]:
                raise ValueError(f"specificity intervention artifact changed: {artifact}")
        return existing
    records, _, _, _ = load_state_bank(Path(config.output_dir) / "state_bank")
    selected = _select_records(records, max_states)
    manifest = _fold_manifest(root)
    selected_folds = np.asarray(
        [manifest.group_folds[_episode_key(records[index])] for index in selected],
        dtype=np.int64,
    )
    _, pretrained_features, _ = load_latent_cache(
        root / "latents" / "pretrained" / PRIMARY_TAP
    )
    accessibility = _read_accessibility(root)["episode_group"]
    conditions: dict[str, object] = {}
    all_passed = True
    for condition in PRIMARY_CONDITIONS:
        state_ids, features, _ = load_latent_cache(
            root / "latents" / condition / PRIMARY_TAP
        )
        if state_ids != tuple(record.state_id for record in records):
            raise ValueError(f"latent rows differ from State Bank: {condition}")
        cells = {
            factor: _load_cell(root, condition, factor)
            for factor in (PRIMARY_FACTOR, *CONTROL_FACTORS)
        }
        probes: dict[str, dict[int, dict[int, _Probe]]] = {
            factor: {} for factor in cells
        }
        for factor in cells:
            offsets = (
                config.probes.matched_seed_offsets
                if factor == PRIMARY_FACTOR
                else (config.probes.matched_seed_offsets[0],)
            )
            for fold in range(manifest.folds):
                probes[factor][fold] = _reconstruct_fold_probes(
                    config=config,
                    records=records,
                    features=features,
                    manifest=manifest,
                    condition=condition,
                    factor=factor,
                    cell=cells[factor],
                    fold=fold,
                    offsets=offsets,
                )
        source = features[np.asarray(selected)]
        target_delta = np.zeros_like(source, dtype=np.float64)
        random_delta = np.zeros_like(source, dtype=np.float64)
        mean_delta = np.zeros_like(source, dtype=np.float64)
        agreements = []
        support_caps = []
        capped_rates = []
        for fold in range(manifest.folds):
            rows = np.flatnonzero(selected_folds == fold)
            if not len(rows):
                continue
            target_probes = list(probes[PRIMARY_FACTOR][fold].values())
            directions = np.stack([probe.direction for probe in target_probes])
            direction = consensus_direction(directions)
            agreements.append(float(np.min(np.abs(directions @ direction) / np.linalg.norm(directions, axis=1))))
            applicable, source_indices, _ = _factor_data(records, PRIMARY_FACTOR)
            parts = crossfit_partition_indices(
                records, manifest, fold=fold, applicable=applicable
            )
            train = features[np.asarray(parts["train"], dtype=np.int64)]
            rolled = np.roll(train, 1, axis=0)
            cap = float(np.quantile(np.linalg.norm(train - rolled, axis=1), 0.95))
            local_source = source[rows]
            local_target, capped_rate = _target_delta(
                local_source, target_probes, direction, cap
            )
            support_caps.append(cap)
            capped_rates.append(capped_rate)
            target_delta[rows] = local_target
            random_delta[rows] = same_norm_random_delta(
                local_target,
                target_direction=direction,
                seed=config.seed + fold,
            )
            train_mean = target_probes[0].train_mean
            mean_delta[rows] = (
                ((train_mean - local_source) @ direction)[:, None]
                * direction[None, :]
            )
        changed = source + target_delta
        random_changed = source + random_delta
        factor_effect: dict[str, np.ndarray] = {}
        random_effect: dict[str, np.ndarray] = {}
        for factor in (PRIMARY_FACTOR, *CONTROL_FACTORS):
            values = np.zeros(len(selected), dtype=np.float64)
            random_values = np.zeros(len(selected), dtype=np.float64)
            for fold in range(manifest.folds):
                rows = np.flatnonzero(selected_folds == fold)
                if not len(rows):
                    continue
                fold_probes = list(probes[factor][fold].values())
                values[rows] = np.mean(
                    [probe.disruption(source[rows], changed[rows]) for probe in fold_probes],
                    axis=0,
                )
                random_values[rows] = np.mean(
                    [
                        probe.disruption(source[rows], random_changed[rows])
                        for probe in fold_probes
                    ],
                    axis=0,
                )
            factor_effect[factor] = values
            random_effect[factor] = random_values
        factor_contrast = {
            factor: factor_effect[factor] - random_effect[factor]
            for factor in factor_effect
        }
        difference = factor_contrast[PRIMARY_FACTOR]
        target_interval = _cluster_interval(
            difference, records, selected, config=config
        )
        place_rows = np.asarray(
            [row for row, index in enumerate(selected) if records[index].labels.phase == "place"],
            dtype=np.int64,
        )
        place_interval = (
            _cluster_interval(
                difference[place_rows],
                records,
                [selected[row] for row in place_rows],
                config=config,
            )
            if len(place_rows)
            else None
        )
        source_norm = np.linalg.norm(source, axis=1)
        activation_ratio = float(
            np.mean(np.linalg.norm(changed, axis=1)) / max(np.mean(source_norm), 1e-12)
        )
        gate = specificity_gate(
            target_minus_random=target_interval,
            target_effect=float(np.mean(factor_contrast[PRIMARY_FACTOR])),
            non_target_effects={
                factor: float(np.mean(factor_contrast[factor]))
                for factor in CONTROL_FACTORS
            },
            activation_norm_ratio=activation_ratio,
            place_target_minus_random=place_interval,
        )
        all_passed = all_passed and bool(gate["passed"])
        artifact = profile_root / "interventions" / f"{condition}.npz"
        _atomic_npz(
            artifact,
            state_ids=np.asarray([records[index].state_id for index in selected]),
            source=source.astype(np.float32),
            target_delta=target_delta.astype(np.float32),
            random_delta=random_delta.astype(np.float32),
            mean_delta=mean_delta.astype(np.float32),
        )
        conditions[condition] = {
            "passed": gate["passed"],
            "failures": gate["failures"],
            "states": len(selected),
            "place_states": int(len(place_rows)),
            "probe_reconstruction": "categorical_exact_continuous_atol_1e-6",
            "seed_direction_min_cosine": min(agreements),
            "natural_difference_p95_by_fold": support_caps,
            "decision_boundary_cap_rate_by_fold": capped_rates,
            "accessibility": accessibility[condition],
            "raw_drift_from_pretrained": {
                "mean_relative_l2": float(
                    np.mean(
                        np.linalg.norm(features - pretrained_features, axis=1)
                        / np.maximum(np.linalg.norm(pretrained_features, axis=1), 1e-8)
                    )
                ),
                "mean_cosine_distance": float(
                    np.mean(
                        1.0
                        - np.sum(features * pretrained_features, axis=1)
                        / np.maximum(
                            np.linalg.norm(features, axis=1)
                            * np.linalg.norm(pretrained_features, axis=1),
                            1e-8,
                        )
                    )
                ),
            },
            "target_effect": float(np.mean(factor_effect[PRIMARY_FACTOR])),
            "matched_random_effect": float(np.mean(random_effect[PRIMARY_FACTOR])),
            "target_minus_random_effect": float(
                np.mean(factor_contrast[PRIMARY_FACTOR])
            ),
            "target_minus_random_episode_ci": target_interval,
            "target_minus_random_task_ci": _cluster_interval(
                difference, records, selected, config=config, task=True
            ),
            "place_target_minus_random_episode_ci": place_interval,
            "non_target_target_minus_random_effects": {
                factor: float(np.mean(factor_contrast[factor]))
                for factor in CONTROL_FACTORS
            },
            "activation_norm_ratio": activation_ratio,
            "same_norm_max_abs_error": float(
                np.max(
                    np.abs(
                        np.linalg.norm(target_delta, axis=1)
                        - np.linalg.norm(random_delta, axis=1)
                    )
                )
            ),
            "intervention_artifact": str(artifact),
            "intervention_sha256": _file_sha256(artifact),
        }
    report = {
        "schema_version": RECRUITMENT_SCHEMA,
        "gate": "stablegrasp_factor_specificity",
        "passed": all_passed,
        "factor": PRIMARY_FACTOR,
        "tap": PRIMARY_TAP,
        "split": "episode_group_crossfit",
        "sampling": "task_phase_stratum_round_robin",
        "max_states": max_states,
        "binding_sha256": binding,
        "conditions": conditions,
        "action_sensitivity_authorized": all_passed,
    }
    write_json_atomic(report_path, report)
    return report


def _predict(
    *,
    policy: object,
    preprocessor: object,
    postprocessor: object,
    backend: object,
    batch: Mapping[str, object],
    state_ids: Sequence[str],
    checkpoint_id: str,
    delta: np.ndarray,
    expected: np.ndarray,
    zero: bool = False,
) -> np.ndarray:
    processed = preprocessor(backend._raw_batch(batch))
    state = processed["observation.state"]
    flow = policy.model
    original_sample_noise = flow.sample_noise

    def sample_noise(shape, device):
        return deterministic_inference_noise(
            state_ids,
            checkpoint_id=checkpoint_id,
            row_shape=tuple(int(value) for value in shape[1:]),
            device=torch.device(device),
            dtype=state.dtype,
        )

    flow.sample_noise = sample_noise
    policy.reset()
    hook = FinalDenoisingDeltaHook(
        flow.action_time_mlp_out,
        expected_calls=int(flow.config.num_steps),
        delta=torch.as_tensor(delta),
        expected_pooled=torch.as_tensor(expected),
        zero=zero,
    )
    try:
        with torch.no_grad(), hook:
            normalized = policy.predict_action_chunk(processed)
            actions = postprocessor(normalized)
    finally:
        flow.sample_noise = original_sample_noise
    return actions.detach().to("cpu", torch.float32).numpy()


def _action_effect(original: np.ndarray, changed: np.ndarray) -> dict[str, np.ndarray]:
    delta = changed - original
    return {
        "first_action_l2": np.linalg.norm(delta[:, 0, :], axis=1),
        "translation_l2": np.linalg.norm(delta[:, 0, :3], axis=1),
        "rotation_l2": np.linalg.norm(delta[:, 0, 3:6], axis=1),
        "gripper_abs": np.abs(delta[:, 0, 6]),
        "gripper_flip": ((original[:, 0, 6] >= 0) != (changed[:, 0, 6] >= 0)).astype(float),
        "chunk_l2": np.mean(np.linalg.norm(delta, axis=2), axis=1),
    }


def _action_sensitivity(
    config: LiberoStudyConfig, *, max_states: int, batch_size: int
) -> dict[str, object]:
    root = Path(config.output_dir) / "protocol_v3"
    profile_root = _profile_root(root, max_states)
    specificity_path = profile_root / "specificity.json"
    specificity = json.loads(specificity_path.read_text())
    if not specificity.get("passed"):
        return {
            "schema_version": RECRUITMENT_SCHEMA,
            "gate": "stablegrasp_longitudinal_action_sensitivity_vs_matched_random",
            "passed": False,
            "status": "blocked_by_specificity",
            "specificity_report": str(specificity_path),
        }
    report_path = profile_root / "action_sensitivity.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("specificity_sha256") != _file_sha256(specificity_path):
            raise ValueError("action sensitivity is bound to a different specificity report")
        for row in existing.get("conditions", {}).values():
            if _file_sha256(Path(str(row["rows"]))) != row["rows_sha256"]:
                raise ValueError("action-sensitivity row artifact changed")
        return existing
    records, _, _, _ = load_state_bank(Path(config.output_dir) / "state_bank")
    by_id = {record.state_id: record for record in records}
    record_index = {record.state_id: index for index, record in enumerate(records)}
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("LeRobotDataset is required for action sensitivity") from error
    dataset = LeRobotDataset(
        config.sources.lerobot_repo_id,
        root=config.sources.lerobot_root,
        revision=config.sources.lerobot_revision,
        download_videos=True,
    )
    from ..backends.lerobot import SmolVLABackend

    condition_reports: dict[str, object] = {}
    for condition in PRIMARY_CONDITIONS:
        condition_row = condition_from_plan(config, condition)
        checkpoint = Path(str(condition_row["checkpoint"]))
        if _tree_sha256(checkpoint) != condition_row["checkpoint_sha256"]:
            raise ValueError(f"checkpoint changed before action sensitivity: {condition}")
        backend = SmolVLABackend(device="auto")
        backend.load_checkpoint_for_dataset(
            checkpoint,
            repo_id=config.sources.lerobot_repo_id,
            dataset_root=dataset.root,
            rename_map=LIBERO_SMOLVLA_RENAME_MAP,
        )
        policy, preprocessor, postprocessor = backend._loaded()
        artifact = np.load(profile_root / "interventions" / f"{condition}.npz")
        state_ids = tuple(str(value) for value in artifact["state_ids"])
        selected = tuple(by_id[value] for value in state_ids)
        modes = {name: [] for name in ("original", "target", "random", "mean", "zero")}
        for start in tqdm(
            range(0, len(selected), batch_size),
            desc=f"StableGrasp actions {condition}",
            unit="batch",
        ):
            stop = min(start + batch_size, len(selected))
            batch_records = selected[start:stop]
            batch = collate_state_bank_observations(batch_records, dataset)
            expected = artifact["source"][start:stop]
            checkpoint_id = f"{condition}:{str(condition_row['checkpoint_sha256'])[:16]}"
            zero_delta = np.zeros_like(expected)
            original = _predict(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                backend=backend,
                batch=batch,
                state_ids=state_ids[start:stop],
                checkpoint_id=checkpoint_id,
                delta=zero_delta,
                expected=expected,
            )
            modes["original"].append(original)
            for mode, key in (("target", "target_delta"), ("random", "random_delta"), ("mean", "mean_delta")):
                modes[mode].append(
                    _predict(
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        backend=backend,
                        batch=batch,
                        state_ids=state_ids[start:stop],
                        checkpoint_id=checkpoint_id,
                        delta=artifact[key][start:stop],
                        expected=expected,
                    )
                )
            modes["zero"].append(
                _predict(
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    backend=backend,
                    batch=batch,
                    state_ids=state_ids[start:stop],
                    checkpoint_id=checkpoint_id,
                    delta=zero_delta,
                    expected=expected,
                    zero=True,
                )
            )
        actions = {name: np.concatenate(values) for name, values in modes.items()}
        target = _action_effect(actions["original"], actions["target"])
        random = _action_effect(actions["original"], actions["random"])
        mean = _action_effect(actions["original"], actions["mean"])
        zero = _action_effect(actions["original"], actions["zero"])
        metrics = {}
        for name in target:
            usage = target[name] - random[name]
            strata = {}
            for stratum in ("pre_contact", "contact_grasp", "post_grasp", "place_release"):
                rows = np.asarray(
                    [
                        index
                        for index, record in enumerate(selected)
                        if phase_stratum(record.labels.phase) == stratum
                    ],
                    dtype=np.int64,
                )
                if len(rows):
                    strata[stratum] = _cluster_interval(
                        usage[rows],
                        records,
                        [record_index[selected[row].state_id] for row in rows],
                        config=config,
                    )
            indices = [record_index[record.state_id] for record in selected]
            metrics[name] = {
                "target_effect": float(np.mean(target[name])),
                "matched_random_effect": float(np.mean(random[name])),
                "target_minus_random_episode_ci": _cluster_interval(
                    usage, records, indices, config=config
                ),
                "target_minus_random_task_ci": _cluster_interval(
                    usage, records, indices, config=config, task=True
                ),
                "matched_mean_effect": float(np.mean(mean[name])),
                "zero_ood_effect": float(np.mean(zero[name])),
                "strata": strata,
            }
        primary_interval = metrics["first_action_l2"][
            "target_minus_random_episode_ci"
        ]
        functionally_recruited = float(primary_interval["ci_low"]) > 0
        rows = [
            {
                "state_id": record.state_id,
                "episode": _episode_key(record),
                "task": _task_key(record),
                "phase": record.labels.phase,
                "stratum": phase_stratum(record.labels.phase),
                "stable_grasp": record.labels.stable_grasp,
                "effects": {
                    metric: {
                        "target": float(target[metric][index]),
                        "random": float(random[metric][index]),
                        "target_minus_random": float(target[metric][index] - random[metric][index]),
                    }
                    for metric in target
                },
            }
            for index, record in enumerate(selected)
        ]
        rows_path = profile_root / "actions" / f"{condition}.json.gz"
        write_bytes_atomic(
            rows_path,
            gzip.compress(json.dumps(rows, sort_keys=True).encode("utf-8"), compresslevel=6),
        )
        condition_reports[condition] = {
            "states": len(selected),
            "functionally_recruited": functionally_recruited,
            "metrics": metrics,
            "rows": str(rows_path),
            "rows_sha256": _file_sha256(rows_path),
        }
        del backend, policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    recruited = [
        condition
        for condition, row in condition_reports.items()
        if row["functionally_recruited"]
    ]
    accessibility = _read_accessibility(root)["episode_group"]
    specificity_conditions = specificity["conditions"]
    trajectory = {
        condition: {
            "raw_drift": specificity_conditions[condition]["raw_drift_from_pretrained"],
            "accessibility": accessibility[condition],
            "functional_recruitment": condition_reports[condition]["metrics"][
                "first_action_l2"
            ]["target_minus_random_episode_ci"],
        }
        for condition in PRIMARY_CONDITIONS
    }
    report = {
        "schema_version": RECRUITMENT_SCHEMA,
        "gate": "stablegrasp_longitudinal_action_sensitivity_vs_matched_random",
        "passed": bool(recruited),
        "integrity_passed": True,
        "status": "complete",
        "primary_metric": "first_action_l2_target_minus_matched_random",
        "primary_cluster": "episode",
        "robustness_cluster": "task",
        "conditions": condition_reports,
        "functionally_recruited_conditions": recruited,
        "trajectory_R_A_U": trajectory,
        "specificity_sha256": _file_sha256(specificity_path),
        "closed_loop_useful": "not_measured",
    }
    write_json_atomic(report_path, report)
    return report


def run_longitudinal_recruitment(
    config: LiberoStudyConfig,
    *,
    max_states: int,
    batch_size: int,
    specificity_only: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("recruitment batch_size must be positive")
    audit = audit_longitudinal_recruitment(config)
    if dry_run:
        return {
            **audit,
            "status": "dry_run",
            "max_states": max_states,
            "batch_size": batch_size,
            "specificity_only": specificity_only,
        }
    report = _specificity(config, max_states=max_states)
    if specificity_only or not report["passed"]:
        return report
    return _action_sensitivity(config, max_states=max_states, batch_size=batch_size)
