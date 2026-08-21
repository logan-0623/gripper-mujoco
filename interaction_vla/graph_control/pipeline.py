from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from interaction_vla.device import resolve_device
from interaction_vla.graph_finetune.data import fit_normalization, graph_v2_targets
from interaction_vla.graph_finetune.pipeline import load_finetune_checkpoint
from interaction_vla.graph_finetune.schema import GraphV2Normalization, GraphV2Targets
from interaction_vla.lerobot_bridge.act_smoke import (
    _act_config,
    _is_oom,
    load_act_dataset,
)
from interaction_vla.lerobot_bridge.config import BridgeConfig, load_bridge_config
from interaction_vla.lerobot_bridge.provenance import (
    fingerprint_tree,
    sha256_file,
    standard_dataset_fingerprint,
)
from interaction_vla.lerobot_bridge.sidecar import load_teacher_sidecar
from interaction_vla.lerobot_bridge.teacher import TCTIGTeacherExtractor
from interaction_vla.lerobot_bridge.validator import validate_dataset_root

from .cache import (
    CacheProvenance,
    TokenCache,
    build_token_cache,
    load_token_cache,
)
from .config import GraphControlConfig, load_graph_control_config
from .dataset import GraphDatasetMetadata
from .diagnostics import (
    DIAGNOSTICS_SCHEMA_VERSION,
    build_representation_diagnostics,
    validate_episode_layout,
)
from .features import FrozenGraphRuntime
from .failure_analysis import (
    FAILURE_ANALYSIS_SCHEMA_VERSION,
    build_failure_analysis_report,
    episode_error_exposure,
    training_error_thresholds,
)
from .rollout import (
    FlatTokenProvider,
    GraphPolicyRuntime,
    OracleGraphV2TokenProvider,
    PredictedTokenProvider,
    aggregate_rollouts,
    paired_evaluation_cases,
    rollout_case,
)
from .schema import (
    ALL_CONDITIONS,
    ORACLE_CONDITIONS,
    TOKEN_DIM,
    TOKEN_SCHEMA_VERSION,
    TOKEN_SLICES,
)
from .sensitivity import (
    ALL_TOKENS_GROUP,
    CATEGORICAL_GROUPS,
    PREVIOUS_SENSITIVITY_SCHEMA_VERSION,
    SENSITIVITY_SCHEMA_VERSION,
    action_change_metrics,
    build_sensitivity_report,
    finite_difference_interventions,
    make_sensitivity_records,
    mask_token_group,
    predict_first_actions,
    select_episode_balanced_positions,
    standardized_perturbation_magnitude,
    temporally_matched_random_tokens,
    training_action_statistics,
    training_feature_statistics,
)
from .training import (
    ControlSplit,
    assert_checkpoint_split,
    expected_graph_checkpoint_metadata,
    graph_checkpoint_bindings,
    load_control_split,
    load_graph_act_checkpoint,
    load_graph_act_checkpoint_for_analysis,
    train_paired_seed,
)
from .tracing import (
    TRACE_SCHEMA_VERSION,
    load_trace_episode,
    trace_episode_summary,
    write_trace_episode_atomic,
)


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


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _atomic_output_directory(destination: Path):
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"output directory must be empty: {destination}")
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        yield staging
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _validate_formal_epochs(config: GraphControlConfig, bridge: BridgeConfig) -> None:
    expected = config.training.formal_epochs
    if expected != 10 or bridge.act.epochs != expected:
        raise ValueError(
            "formal Graph-conditioned ACT training requires exactly 10 epochs"
        )


def _require_recovery_report(
    config: GraphControlConfig | Any, bridge: BridgeConfig | Any
) -> str:
    recovery = bridge.recovery
    if recovery is None or (
        float(recovery.train_success_threshold) != 0.80
        or float(recovery.heldout_success_threshold) != 0.30
    ):
        raise ValueError("Graph control requires the exact ACT recovery gate 0.80/0.30")
    path = Path(config.required_recovery_report)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"ACT recovery report is invalid: {path}") from error
    try:
        train_rate = float(payload["train_seen"]["success_rate"])
        heldout_rate = float(payload["heldout"]["success_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ACT recovery report did not pass the required gate") from error
    if (
        payload.get("passed") is not True
        or not np.isfinite((train_rate, heldout_rate)).all()
        or train_rate < 0.80
        or heldout_rate < 0.30
    ):
        raise ValueError("ACT recovery report did not pass the required gate")
    return sha256_file(path)


def _require_oracle_report(config: GraphControlConfig | Any) -> str | None:
    path = config.required_oracle_report
    if tuple(config.conditions) == ORACLE_CONDITIONS:
        if path is not None:
            raise ValueError("oracle matrix required_oracle_report must be null")
        return None
    if tuple(config.conditions) != ALL_CONDITIONS or path is None:
        raise ValueError("predicted Graph v2 matrix requires an oracle report")
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Graph v2 oracle report is invalid: {source}") from error
    gate = payload.get("oracle_gate")
    if (
        payload.get("passed") is not True
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
    ):
        raise ValueError("Graph v2 oracle gate did not pass")
    return sha256_file(source)


def _clear_accelerator_memory() -> None:
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


def _train_seed_with_fallback(
    destination: Path,
    *,
    batch_size: int,
    train_attempt: Callable[[int, Path], Mapping[str, object]],
) -> dict[str, object]:
    try:
        report = dict(train_attempt(batch_size, destination))
        report["batch_size"] = batch_size
        return report
    except RuntimeError as error:
        if batch_size != 2 or not _is_oom(error):
            raise
    if destination.exists():
        shutil.rmtree(destination)
    _clear_accelerator_memory()
    report = dict(train_attempt(1, destination))
    report["fallback_from_batch_size"] = 2
    report["batch_size"] = 1
    return report


def _publish_evaluation(
    destination: Path,
    *,
    records: list[dict[str, object]],
    report: Mapping[str, object],
) -> dict[str, object]:
    records_path = destination / "episodes.jsonl"
    report_path = destination / "report.json"
    final = {
        **report,
        "episodes_path": records_path,
        "report_path": report_path,
    }
    with _atomic_output_directory(destination) as staging:
        staging_records = staging / "episodes.jsonl"
        with staging_records.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _write_json_atomic(staging / "report.json", final)
    return final


def _diagnostics_source_fingerprint() -> str:
    digest = hashlib.sha256()
    for source in (Path(__file__), Path(__file__).with_name("diagnostics.py")):
        relative = source.relative_to(Path(__file__).resolve().parents[2])
        content = source.read_bytes()
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _select_diagnostic_cache_rows(
    cache: Any, partition_rows: tuple[int, ...]
) -> np.ndarray:
    rows = np.asarray(cache.row_indices)
    tokens = np.asarray(cache.tokens)
    if rows.ndim != 1 or tokens.shape != (len(rows), TOKEN_DIM):
        raise ValueError("diagnostic cache arrays are incompatible")
    positions_by_row = {int(row): index for index, row in enumerate(rows)}
    if len(positions_by_row) != len(rows):
        raise ValueError("diagnostic cache rows must be unique")
    try:
        positions = np.asarray(
            [positions_by_row[row] for row in partition_rows], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("diagnostic cache rows do not cover the partition") from error
    if len(positions) > 1 and np.any(positions[1:] <= positions[:-1]):
        raise ValueError("diagnostic cache rows differ from split order")
    if not np.array_equal(rows[positions], np.asarray(partition_rows)):
        raise ValueError("diagnostic cache rows differ from the requested partition")
    selected = np.asarray(tokens[positions], dtype=np.float64)
    if selected.shape != (len(partition_rows), TOKEN_DIM) or not np.isfinite(selected).all():
        raise ValueError("selected diagnostic cache tokens are invalid")
    return selected


def _publish_diagnostics(
    destination: Path,
    *,
    episode_records: list[dict[str, object]],
    report: Mapping[str, object],
) -> dict[str, object]:
    report_path = destination / "report.json"
    per_episode_path = destination / "per_episode.jsonl"
    final = {
        **report,
        "report_path": report_path,
        "per_episode_path": per_episode_path,
    }
    with _atomic_output_directory(destination) as staging:
        records_path = staging / "per_episode.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for record in episode_records:
                handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _write_json_atomic(staging / "report.json", final)
    return {
        "passed": bool(final["passed"]),
        "schema_version": final["schema_version"],
        "partition": final["partition"],
        "rows": int(final["rows"]),
        "episodes": int(final["episodes"]),
        "conditions": list(final["conditions"]),
        "estimator_seeds": list(final["estimator_seeds"]),
        "report_path": report_path,
        "per_episode_path": per_episode_path,
    }


def _load_source(bridge: BridgeConfig):
    validate_dataset_root(
        bridge.dataset.root,
        repo_id=bridge.dataset.repo_id,
        allow_incomplete=False,
        require_bridge_metadata=True,
        replay=False,
        bridge_config=None,
    )
    from lerobot.datasets import LeRobotDataset

    return LeRobotDataset(bridge.dataset.repo_id, root=bridge.dataset.root)


def _control_dataset_fingerprint(bridge: BridgeConfig) -> str:
    standard = standard_dataset_fingerprint(bridge.dataset.root)
    teacher_manifest = bridge.dataset.root / "meta" / "teacher_manifest.json"
    if not teacher_manifest.is_file():
        raise FileNotFoundError(f"teacher manifest not found: {teacher_manifest}")
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(standard))
    digest.update(bytes.fromhex(sha256_file(teacher_manifest)))
    return digest.hexdigest()


def _load_graph_payload(
    config: GraphControlConfig,
    split: ControlSplit,
    *,
    condition: str,
    seed: int,
    oracle_report_sha256: str | None,
) -> dict[str, Any]:
    checkpoint = config.graph_checkpoint(condition, seed)
    if checkpoint is None:
        raise ValueError("flat condition has no Graph checkpoint payload")
    model, _, payload = load_finetune_checkpoint(checkpoint, device="cpu")
    del model
    assert_checkpoint_split(payload, split, condition=condition, seed=seed)
    if (
        oracle_report_sha256 is None
        or payload.get("oracle_report_sha256") != oracle_report_sha256
    ):
        raise ValueError("Graph checkpoint oracle report SHA-256 mismatch")
    return payload


def _validate_source_alignment(source: Any, split: ControlSplit) -> None:
    episodes = tuple(
        value for partition in ("train", "validation", "test") for value in split.episodes[partition]
    )
    rows = tuple(
        value for partition in ("train", "validation", "test") for value in split.rows[partition]
    )
    if sorted(episodes) != list(range(int(source.meta.total_episodes))):
        raise ValueError("Graph split episodes do not cover the LeRobot dataset exactly")
    if sorted(rows) != list(range(len(source))):
        raise ValueError("Graph split rows do not cover the LeRobot dataset exactly")


def _context(
    path: str | Path,
) -> tuple[
    GraphControlConfig,
    BridgeConfig,
    ControlSplit,
    Any,
    str,
    str | None,
]:
    config = load_graph_control_config(path)
    bridge = load_bridge_config(config.bridge_config)
    recovery_report_sha256 = _require_recovery_report(config, bridge)
    oracle_report_sha256 = _require_oracle_report(config)
    split = load_control_split(config.split_manifest)
    source = _load_source(bridge)
    _validate_source_alignment(source, split)
    return (
        config,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    )


def diagnose_from_config(
    path: str | Path, *, partition: str | None = None
) -> dict[str, object]:
    preliminary = load_graph_control_config(path)
    diagnostics = preliminary.diagnostics
    if diagnostics is None:
        raise ValueError("graph control diagnostics config is required")
    selected_partition = "test" if partition is None else str(partition)
    if selected_partition not in {"train", "validation", "test"}:
        raise ValueError("diagnostics partition must be train, validation, or test")
    destination = diagnostics.output_dir / selected_partition
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError("Graph diagnostics output already exists")

    (
        config,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _context(path)
    if config.diagnostics is None:
        raise ValueError("graph control diagnostics config is required")
    controls = config.diagnostics
    partition_rows = tuple(int(row) for row in split.rows[selected_partition])
    if not partition_rows:
        raise ValueError("diagnostics partition rows must be non-empty")

    episode_column = np.asarray(source.hf_dataset["episode_index"])
    frame_column = np.asarray(source.hf_dataset["frame_index"])
    if (
        episode_column.ndim != 1
        or frame_column.ndim != 1
        or len(episode_column) != len(frame_column)
        or max(partition_rows) >= len(episode_column)
    ):
        raise ValueError("diagnostic dataset episode/frame columns are incompatible")
    layout = validate_episode_layout(
        row_indices=np.asarray(partition_rows, dtype=np.int64),
        episode_indices=episode_column[np.asarray(partition_rows, dtype=np.int64)],
        frame_indices=frame_column[np.asarray(partition_rows, dtype=np.int64)],
    )

    condition_tokens: dict[tuple[int, str], np.ndarray] = {}
    cache_sha256: dict[str, str] = {}
    dataset_fingerprints: set[str] = set()
    for seed in config.seeds:
        caches = _load_cache_matrix(
            config,
            bridge,
            split,
            seed,
            oracle_report_sha256=oracle_report_sha256,
        )
        if set(caches) != set(config.conditions):
            raise ValueError("diagnostic cache condition matrix is incomplete")
        for condition in config.conditions:
            cache = caches[condition]
            condition_tokens[(seed, condition)] = _select_diagnostic_cache_rows(
                cache, partition_rows
            )
            cache_sha256[f"seed_{seed}/{condition}"] = str(cache.sha256)
            dataset_fingerprints.add(str(cache.provenance.dataset_fingerprint))
    if len(dataset_fingerprints) != 1:
        raise ValueError("diagnostic caches bind different datasets")
    teacher_tokens = condition_tokens[(config.seeds[0], "oracle_graph_v2")]
    report, episode_records = build_representation_diagnostics(
        condition_tokens=condition_tokens,
        teacher_tokens=teacher_tokens,
        layout=layout,
        partition=selected_partition,
        bootstrap_samples=controls.bootstrap_samples,
        bootstrap_seed=controls.bootstrap_seed,
        max_lag=controls.max_lag,
        active_epsilon=controls.active_epsilon,
    )
    final_report = {
        **report,
        "config": config.config_path,
        "split_manifest": split.path,
        "split_manifest_sha256": split.sha256,
        "dataset_fingerprint": next(iter(dataset_fingerprints)),
        "token_schema_version": TOKEN_SCHEMA_VERSION,
        "token_dim": TOKEN_DIM,
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "diagnostics_source_fingerprint": _diagnostics_source_fingerprint(),
        "cache_sha256": cache_sha256,
        "recovery_report_sha256": recovery_report_sha256,
        "oracle_report_sha256": oracle_report_sha256,
    }
    del source
    gc.collect()
    return _publish_diagnostics(
        destination,
        episode_records=episode_records,
        report=final_report,
    )


def _analysis_policy_runtime(
    config: GraphControlConfig,
    bridge: BridgeConfig,
    split: ControlSplit,
    *,
    seed: int,
    condition: str,
    cache: TokenCache,
    recovery_report_sha256: str,
    oracle_report_sha256: str | None,
) -> tuple[Any, Any, Any, dict[str, object], Path, str]:
    bindings = graph_checkpoint_bindings(
        condition,
        seed,
        cache,
        recovery_report_sha256=recovery_report_sha256,
        oracle_report_sha256=oracle_report_sha256,
    )
    checkpoint = (
        config.training.output_dir / f"seed_{seed}" / condition / "checkpoint"
    )
    device = resolve_device(bridge.act.device)
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    base_metadata = LeRobotDatasetMetadata(
        bridge.dataset.repo_id, root=bridge.dataset.root
    )
    expected_metadata = expected_graph_checkpoint_metadata(
        dataset_root=bridge.dataset.root,
        features=GraphDatasetMetadata(base_metadata).features,
        act_config=_act_config(
            device=device,
            architecture="configured",
            bridge_config=bridge,
        ),
        device=device,
        bindings=bindings,
    )
    policy, preprocessor, postprocessor, _, audit = (
        load_graph_act_checkpoint_for_analysis(
            checkpoint,
            device=device,
            expected_metadata=expected_metadata,
        )
    )
    return (
        policy,
        preprocessor,
        postprocessor,
        audit,
        checkpoint,
        fingerprint_tree(checkpoint),
    )


def _sensitivity_source_batches(
    source: Any, *, rows: tuple[int, ...], batch_size: int
) -> list[dict[str, Any]]:
    from torch.utils.data import default_collate

    if not rows or batch_size < 1:
        raise ValueError("sensitivity source rows and batch size must be positive")
    batches: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        samples = [dict(source[row]) for row in rows[start : start + batch_size]]
        if any("observation.environment_state" in sample for sample in samples):
            raise ValueError("base sensitivity sample already contains Graph tokens")
        batch = default_collate(samples)
        if not isinstance(batch, dict):
            raise ValueError("collated sensitivity source batch must be a mapping")
        batches.append(batch)
    return batches


def _publish_sensitivity(
    destination: Path,
    *,
    records: list[dict[str, object]],
    report: Mapping[str, object],
) -> dict[str, object]:
    report_path = destination / "report.json"
    records_path = destination / "records.jsonl"
    final = {**report, "report_path": report_path, "records_path": records_path}
    with _atomic_output_directory(destination) as staging:
        with (staging / "records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _write_json_atomic(staging / "report.json", final)
    return {
        "passed": bool(final["passed"]),
        "schema_version": final["schema_version"],
        "partition": final["partition"],
        "observations": int(final["observations"]),
        "records": int(final["rows"]),
        "policy_seeds": list(final["policy_seeds"]),
        "conditions": list(final["conditions"]),
        "report_path": report_path,
        "records_path": records_path,
    }


def _load_reusable_sensitivity_v2(
    destination: Path,
    *,
    expected: Mapping[str, object],
    expected_records: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, object]], Mapping[str, object]] | None:
    report_path = destination / "report.json"
    records_path = destination / "records.jsonl"
    if not report_path.exists() and not records_path.exists():
        return None
    if not report_path.is_file() or not records_path.is_file():
        raise ValueError("reusable sensitivity v2 requires report.json and records.jsonl")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("reusable sensitivity v2 report is invalid") from error
    if not isinstance(report, Mapping):
        raise ValueError("reusable sensitivity v2 report must be a mapping")
    if report.get("passed") is not True or report.get(
        "schema_version"
    ) != PREVIOUS_SENSITIVITY_SCHEMA_VERSION:
        raise ValueError("reusable sensitivity report is not a passing v2 artifact")
    for key, value in expected.items():
        if _jsonable(report.get(key)) != _jsonable(value):
            raise ValueError(f"reusable sensitivity v2 differs at {key}")
    try:
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("reusable sensitivity v2 records are invalid") from error
    if len(records) != expected_records or int(report.get("rows", -1)) != len(records):
        raise ValueError("reusable sensitivity v2 record count is incompatible")
    if any(
        not isinstance(record, Mapping)
        or record.get("group") == ALL_TOKENS_GROUP
        for record in records
    ):
        raise ValueError("reusable sensitivity v2 contains incompatible controls")
    build_sensitivity_report(
        records,
        partition=str(expected["partition"]),
        bootstrap_samples=1,
        bootstrap_seed=bootstrap_seed,
    )
    provenance: Mapping[str, object] = {
        "schema_version": PREVIOUS_SENSITIVITY_SCHEMA_VERSION,
        "report_path": report_path,
        "report_sha256": sha256_file(report_path),
        "records_path": records_path,
        "records_sha256": sha256_file(records_path),
        "records": len(records),
        "cache_sha256": report.get("cache_sha256"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "dataset_fingerprint": report.get("dataset_fingerprint"),
    }
    return [dict(record) for record in records], provenance


def sensitivity_from_config(
    path: str | Path, *, partition: str | None = None
) -> dict[str, object]:
    preliminary = load_graph_control_config(path)
    controls = preliminary.diagnostics
    if controls is None:
        raise ValueError("graph control diagnostics config is required")
    selected_partition = "test" if partition is None else str(partition)
    if selected_partition not in {"train", "validation", "test"}:
        raise ValueError("sensitivity partition must be train, validation, or test")
    destination = controls.output_dir / selected_partition / "sensitivity_v3"
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError("Graph sensitivity output already exists")

    (
        config,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _context(path)
    if config.diagnostics is None:
        raise ValueError("graph control diagnostics config is required")
    controls = config.diagnostics
    partition_rows = tuple(int(row) for row in split.rows[selected_partition])
    episode_column = np.asarray(source.hf_dataset["episode_index"])
    frame_column = np.asarray(source.hf_dataset["frame_index"])
    if (
        not partition_rows
        or episode_column.ndim != 1
        or frame_column.ndim != 1
        or len(episode_column) != len(frame_column)
        or max(partition_rows) >= len(episode_column)
    ):
        raise ValueError("sensitivity dataset episode/frame columns are incompatible")
    layout = validate_episode_layout(
        row_indices=np.asarray(partition_rows, dtype=np.int64),
        episode_indices=episode_column[np.asarray(partition_rows, dtype=np.int64)],
        frame_indices=frame_column[np.asarray(partition_rows, dtype=np.int64)],
    )
    selected_positions = select_episode_balanced_positions(
        layout, rows_per_episode=controls.sensitivity_rows_per_episode
    )
    selected_rows = tuple(int(value) for value in layout.row_indices[selected_positions])
    action_column = np.asarray(source.hf_dataset["action"], dtype=np.float64)
    if (
        action_column.ndim != 2
        or action_column.shape[1] != 7
        or len(action_column) != len(episode_column)
        or not np.isfinite(action_column).all()
    ):
        raise ValueError("sensitivity dataset action column must be finite [rows, 7]")
    action_statistics = training_action_statistics(
        action_column[np.asarray(split.rows["train"], dtype=np.int64)]
    )
    reusable = _load_reusable_sensitivity_v2(
        controls.output_dir / selected_partition / "sensitivity_v2",
        expected={
            "partition": selected_partition,
            "observations": len(selected_rows),
            "selected_rows": list(selected_rows),
            "rows_per_episode": controls.sensitivity_rows_per_episode,
            "batch_size": controls.sensitivity_batch_size,
            "finite_difference_scale": controls.sensitivity_scale,
            "policy_seeds": list(config.seeds),
            "conditions": list(config.conditions),
            "split_manifest_sha256": split.sha256,
            "token_schema_version": TOKEN_SCHEMA_VERSION,
            "token_dim": TOKEN_DIM,
            "action_statistics": action_statistics,
            "recovery_report_sha256": recovery_report_sha256,
            "oracle_report_sha256": oracle_report_sha256,
        },
        expected_records=(
            len(selected_rows)
            * len(config.seeds)
            * len(config.conditions)
            * 3
            * len(TOKEN_SLICES)
        ),
        bootstrap_seed=controls.bootstrap_seed,
    )
    raw_batches = _sensitivity_source_batches(
        source,
        rows=selected_rows,
        batch_size=controls.sensitivity_batch_size,
    )

    records: list[dict[str, object]] = [] if reusable is None else reusable[0]
    reused_provenance = None if reusable is None else reusable[1]
    cache_sha256: dict[str, str] = {}
    checkpoint_sha256: dict[str, str] = {}
    checkpoint_paths: dict[str, Path] = {}
    compatibility: dict[str, dict[str, object]] = {}
    dataset_fingerprints: set[str] = set()
    probes_per_condition = 3 + (0 if reusable is not None else 3 * len(TOKEN_SLICES))
    progress = tqdm(
        total=len(config.seeds)
        * len(config.conditions)
        * probes_per_condition,
        desc="policy sensitivity",
        unit="probe",
        dynamic_ncols=True,
        disable=None,
    )
    control_provenance: dict[str, dict[str, object]] = {}
    for seed in config.seeds:
        caches = _load_cache_matrix(
            config,
            bridge,
            split,
            seed,
            oracle_report_sha256=oracle_report_sha256,
        )
        for condition in config.conditions:
            cache = caches[condition]
            key = f"seed_{seed}/{condition}"
            cache_sha256[key] = str(cache.sha256)
            if reused_provenance is not None:
                previous_cache = reused_provenance.get("cache_sha256")
                if (
                    not isinstance(previous_cache, Mapping)
                    or previous_cache.get(key) != str(cache.sha256)
                ):
                    raise ValueError(
                        f"reusable sensitivity v2 cache differs at {key}"
                    )
            dataset_fingerprints.add(str(cache.provenance.dataset_fingerprint))
            train_tokens = _select_diagnostic_cache_rows(
                cache, tuple(int(row) for row in split.rows["train"])
            )
            partition_tokens = _select_diagnostic_cache_rows(cache, partition_rows)
            source_tokens = partition_tokens[selected_positions]
            statistics = training_feature_statistics(train_tokens)
            (
                policy,
                preprocessor,
                postprocessor,
                audit,
                checkpoint,
                checkpoint_digest,
            ) = _analysis_policy_runtime(
                config,
                bridge,
                split,
                seed=seed,
                condition=condition,
                cache=cache,
                recovery_report_sha256=recovery_report_sha256,
                oracle_report_sha256=oracle_report_sha256,
            )
            compatibility[key] = audit
            checkpoint_paths[key] = checkpoint
            checkpoint_sha256[key] = checkpoint_digest
            if reused_provenance is not None:
                previous_checkpoint = reused_provenance.get("checkpoint_sha256")
                if (
                    not isinstance(previous_checkpoint, Mapping)
                    or previous_checkpoint.get(key) != checkpoint_digest
                ):
                    raise ValueError(
                        f"reusable sensitivity v2 checkpoint differs at {key}"
                    )
            baseline_actions = predict_first_actions(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                raw_batches=raw_batches,
                tokens=source_tokens,
            )
            progress.update(1)
            matched_partition, matched_provenance = temporally_matched_random_tokens(
                partition_tokens,
                layout,
                seed=controls.bootstrap_seed + seed,
            )
            matched_tokens = matched_partition[selected_positions]
            provenance_key = f"seed_{seed}"
            previous_provenance = control_provenance.setdefault(
                provenance_key, matched_provenance
            )
            if previous_provenance != matched_provenance:
                raise ValueError(
                    "temporally matched control differs across representation conditions"
                )
            for intervention, changed_tokens in (
                ("zero", mask_token_group(source_tokens, ALL_TOKENS_GROUP)),
                ("temporally_matched_random", matched_tokens),
            ):
                changed_actions = (
                    baseline_actions.copy()
                    if np.array_equal(changed_tokens, source_tokens)
                    else predict_first_actions(
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        raw_batches=raw_batches,
                        tokens=changed_tokens,
                    )
                )
                magnitude = standardized_perturbation_magnitude(
                    source_tokens,
                    changed_tokens,
                    ALL_TOKENS_GROUP,
                    training_std=statistics["std"],
                )
                metrics = action_change_metrics(
                    baseline_actions,
                    changed_actions,
                    perturbation_magnitude=magnitude,
                    action_scale=action_statistics["effective_scale"],
                )
                records.extend(
                    make_sensitivity_records(
                        policy_seed=seed,
                        condition=condition,
                        group=ALL_TOKENS_GROUP,
                        intervention=intervention,
                        row_indices=layout.row_indices[selected_positions],
                        episode_indices=layout.episode_indices[selected_positions],
                        frame_indices=layout.frame_indices[selected_positions],
                        metrics=metrics,
                    )
                )
                progress.update(1)
            if reusable is not None:
                del policy, preprocessor, postprocessor
                gc.collect()
                _clear_accelerator_memory()
                continue
            for group in TOKEN_SLICES:
                progress.set_postfix(seed=seed, condition=condition, group=group)
                masked = mask_token_group(source_tokens, group)
                minus, plus = finite_difference_interventions(
                    source_tokens,
                    group,
                    statistics=statistics,
                    scale=controls.sensitivity_scale,
                )
                names = (
                    ("mask", masked),
                    (
                        "toward_uniform"
                        if group in CATEGORICAL_GROUPS
                        else "minus_std",
                        minus,
                    ),
                    (
                        "away_uniform"
                        if group in CATEGORICAL_GROUPS
                        else "plus_std",
                        plus,
                    ),
                )
                for intervention, changed_tokens in names:
                    changed_actions = (
                        baseline_actions.copy()
                        if np.array_equal(changed_tokens, source_tokens)
                        else predict_first_actions(
                            policy=policy,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            raw_batches=raw_batches,
                            tokens=changed_tokens,
                        )
                    )
                    magnitude = standardized_perturbation_magnitude(
                        source_tokens,
                        changed_tokens,
                        group,
                        training_std=statistics["std"],
                    )
                    metrics = action_change_metrics(
                        baseline_actions,
                        changed_actions,
                        perturbation_magnitude=magnitude,
                        action_scale=action_statistics["effective_scale"],
                    )
                    records.extend(
                        make_sensitivity_records(
                            policy_seed=seed,
                            condition=condition,
                            group=group,
                            intervention=intervention,
                            row_indices=layout.row_indices[selected_positions],
                            episode_indices=layout.episode_indices[selected_positions],
                            frame_indices=layout.frame_indices[selected_positions],
                            metrics=metrics,
                        )
                    )
                    progress.update(1)
            del policy, preprocessor, postprocessor
            gc.collect()
            _clear_accelerator_memory()
    progress.close()
    if len(dataset_fingerprints) != 1:
        raise ValueError("sensitivity caches bind different datasets")
    if reused_provenance is not None and reused_provenance.get(
        "dataset_fingerprint"
    ) != next(iter(dataset_fingerprints)):
        raise ValueError("reusable sensitivity v2 dataset fingerprint differs")
    report = build_sensitivity_report(
        records,
        partition=selected_partition,
        bootstrap_samples=controls.bootstrap_samples,
        bootstrap_seed=controls.bootstrap_seed,
    )
    final_report = {
        **report,
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "observations": len(selected_rows),
        "selected_rows": list(selected_rows),
        "rows_per_episode": controls.sensitivity_rows_per_episode,
        "batch_size": controls.sensitivity_batch_size,
        "finite_difference_scale": controls.sensitivity_scale,
        "action_statistics": action_statistics,
        "control_interventions": ["zero", "temporally_matched_random"],
        "control_provenance": control_provenance,
        "reused_sensitivity_v2": reused_provenance,
        "config": config.config_path,
        "split_manifest": split.path,
        "split_manifest_sha256": split.sha256,
        "dataset_fingerprint": next(iter(dataset_fingerprints)),
        "token_schema_version": TOKEN_SCHEMA_VERSION,
        "token_dim": TOKEN_DIM,
        "cache_sha256": cache_sha256,
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_compatibility": compatibility,
        "recovery_report_sha256": recovery_report_sha256,
        "oracle_report_sha256": oracle_report_sha256,
    }
    del source
    gc.collect()
    return _publish_sensitivity(
        destination,
        records=records,
        report=final_report,
    )


def _trace_source_fingerprint() -> str:
    repository = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for source in (
        Path(__file__).with_name("rollout.py"),
        Path(__file__).with_name("tracing.py"),
    ):
        relative = source.relative_to(repository).as_posix().encode("utf-8")
        content = source.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _trace_episode_path(
    root: Path, *, seed: int, condition: str, case_id: str
) -> Path:
    if not case_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in case_id):
        raise ValueError("trace case_id contains unsafe path characters")
    return root / "traces" / f"seed_{seed}" / condition / f"{case_id}.jsonl"


def _read_json_mapping(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is invalid: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _trace_result(
    *,
    destination: Path,
    report: Mapping[str, object],
    resumed: bool,
) -> dict[str, object]:
    return {
        "passed": bool(report["passed"]),
        "episodes": int(report["records"]),
        "policy_seeds": list(report["policy_seeds"]),
        "conditions": list(report["conditions"]),
        "resumed": bool(resumed),
        "manifest_path": destination / "manifest.json",
        "episodes_path": destination / "episodes.jsonl",
        "report_path": destination / "report.json",
    }


def trace_from_config(path: str | Path) -> dict[str, object]:
    preliminary = load_graph_control_config(path)
    trace = preliminary.trace
    if trace is None or not trace.enabled:
        raise ValueError("enabled graph control trace config is required")
    destination = trace.output_dir
    manifest_path = destination / "manifest.json"
    if destination.exists() and not destination.is_dir():
        raise FileExistsError("Graph trace output path is not a directory")
    if destination.exists() and not trace.resume:
        raise FileExistsError("Graph trace output already exists and resume is disabled")
    partials = list(destination.rglob("*.tmp")) if destination.exists() else []
    if partials:
        raise ValueError("Graph trace output contains partial episode files")

    (
        config,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _context(path)
    if config.trace is None or not config.trace.enabled:
        raise ValueError("enabled graph control trace config is required")
    cases = paired_evaluation_cases(
        layouts=config.evaluation.layouts,
        object_counts=config.evaluation.object_counts,
        cases_per_cell=config.evaluation.cases_per_cell,
        master_seed=config.evaluation.master_seed,
    )
    cache_matrices: dict[int, dict[str, TokenCache]] = {}
    cache_sha256: dict[str, str] = {}
    dataset_fingerprints: set[str] = set()
    checkpoint_sha256: dict[str, str] = {}
    checkpoint_paths: dict[str, Path] = {}
    for seed in config.seeds:
        caches = _load_cache_matrix(
            config,
            bridge,
            split,
            seed,
            oracle_report_sha256=oracle_report_sha256,
        )
        cache_matrices[seed] = caches
        for condition in config.conditions:
            key = f"seed_{seed}/{condition}"
            cache_sha256[key] = str(caches[condition].sha256)
            dataset_fingerprints.add(
                str(caches[condition].provenance.dataset_fingerprint)
            )
            checkpoint = (
                config.training.output_dir
                / f"seed_{seed}"
                / condition
                / "checkpoint"
            )
            checkpoint_paths[key] = checkpoint
            checkpoint_sha256[key] = fingerprint_tree(checkpoint)
    if len(dataset_fingerprints) != 1:
        raise ValueError("trace caches bind different datasets")
    expected_manifest: dict[str, object] = {
        "schema_version": "graph_control_trace_manifest_v1",
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "config": config.config_path,
        "config_sha256": sha256_file(config.config_path),
        "split_manifest": split.path,
        "split_manifest_sha256": split.sha256,
        "dataset_fingerprint": next(iter(dataset_fingerprints)),
        "conditions": list(config.conditions),
        "policy_seeds": list(config.seeds),
        "cases": [
            {
                "case_id": case.case_id,
                "environment_seed": case.seed,
                "layout": case.layout,
                "object_count": case.object_count,
            }
            for case in cases
        ],
        "max_steps": config.evaluation.max_steps,
        "cache_sha256": cache_sha256,
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_sha256": checkpoint_sha256,
        "recovery_report_sha256": recovery_report_sha256,
        "oracle_report_sha256": oracle_report_sha256,
        "trace_source_fingerprint": _trace_source_fingerprint(),
    }
    resumed = destination.exists()
    if resumed:
        if not manifest_path.is_file():
            raise ValueError("Graph trace manifest is missing")
        actual_manifest = _read_json_mapping(manifest_path, "Graph trace manifest")
        differing = sorted(
            key
            for key, value in _jsonable(expected_manifest).items()
            if actual_manifest.get(key) != value
        )
        if differing:
            raise ValueError(
                "Graph trace manifest is incompatible: " + ", ".join(differing)
            )
        if actual_manifest.get("complete") is True:
            report = _read_json_mapping(destination / "report.json", "Graph trace report")
            return _trace_result(
                destination=destination,
                report=report,
                resumed=True,
            )
    else:
        destination.mkdir(parents=True, exist_ok=False)
        actual_manifest = {
            **expected_manifest,
            "complete": False,
            "completed_episodes": 0,
            "total_episodes": len(config.seeds) * len(config.conditions) * len(cases),
        }
        _write_json_atomic(manifest_path, actual_manifest)

    _, _, oracle_normalization = _oracle_inputs(bridge, source, split)
    del source
    gc.collect()
    records: list[dict[str, object]] = []
    compatibility: dict[str, object] = {}
    total_episodes = len(config.seeds) * len(config.conditions) * len(cases)
    completed = 0
    progress = tqdm(
        total=total_episodes,
        desc="graph control trace",
        unit="episode",
        dynamic_ncols=True,
        disable=None,
    )
    for seed in config.seeds:
        caches = cache_matrices[seed]
        for condition in config.conditions:
            runtime = _runtime_for_condition(
                config,
                bridge,
                split,
                seed=seed,
                condition=condition,
                cache=caches[condition],
                recovery_report_sha256=recovery_report_sha256,
                oracle_report_sha256=oracle_report_sha256,
                oracle_normalization=oracle_normalization,
                retrospective_analysis=True,
            )
            compatibility[f"seed_{seed}/{condition}"] = (
                runtime.checkpoint_compatibility
            )
            teacher_provider = OracleGraphV2TokenProvider(
                teacher=TCTIGTeacherExtractor(bridge.teacher),
                normalization=oracle_normalization,
            )
            for case in cases:
                progress.set_postfix(
                    seed=seed, condition=condition, case=case.case_id
                )
                trace_path = _trace_episode_path(
                    destination,
                    seed=seed,
                    condition=condition,
                    case_id=case.case_id,
                )
                if trace_path.is_file():
                    trace_records = load_trace_episode(trace_path)
                else:
                    trace_records: list[dict[str, object]] = []
                    rollout_case(
                        bridge,
                        runtime,
                        case,
                        max_steps=config.evaluation.max_steps,
                        trace_callback=trace_records.append,
                        teacher_token_provider=teacher_provider,
                    )
                    write_trace_episode_atomic(trace_path, trace_records)
                records.append(trace_episode_summary(trace_records))
                completed += 1
                progress.update(1)
                _write_json_atomic(
                    manifest_path,
                    {
                        **expected_manifest,
                        "complete": False,
                        "completed_episodes": completed,
                        "total_episodes": total_episodes,
                    },
                )
            del runtime, teacher_provider
            gc.collect()
            _clear_accelerator_memory()
    progress.close()
    aggregate = aggregate_rollouts(records, conditions=config.conditions)
    report = {
        **aggregate,
        "conditions": list(config.conditions),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "trace_manifest": manifest_path,
        "checkpoint_compatibility": compatibility,
        "cache_sha256": cache_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "recovery_report_sha256": recovery_report_sha256,
        "oracle_report_sha256": oracle_report_sha256,
    }
    _write_jsonl_atomic(destination / "episodes.jsonl", records)
    _write_json_atomic(destination / "report.json", report)
    _write_json_atomic(
        manifest_path,
        {
            **expected_manifest,
            "complete": True,
            "completed_episodes": total_episodes,
            "total_episodes": total_episodes,
            "episodes_path": destination / "episodes.jsonl",
            "report_path": destination / "report.json",
        },
    )
    return _trace_result(
        destination=destination,
        report=report,
        resumed=resumed,
    )


def failure_analysis_from_config(
    path: str | Path, *, traces: str | Path
) -> dict[str, object]:
    trace_root = Path(traces)
    manifest_path = trace_root / "manifest.json"
    manifest = _read_json_mapping(manifest_path, "Graph trace manifest")
    if manifest.get("complete") is not True:
        raise ValueError("Graph trace manifest is not complete")
    output_dir = trace_root / "failure_analysis"
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError("Graph failure analysis output already exists")

    (
        config,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _context(path)
    if config.diagnostics is None:
        raise ValueError("graph control diagnostics config is required")
    expected_manifest_fields = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "config_sha256": sha256_file(config.config_path),
        "split_manifest_sha256": split.sha256,
        "conditions": list(config.conditions),
        "policy_seeds": list(config.seeds),
    }
    differing = sorted(
        key
        for key, value in expected_manifest_fields.items()
        if manifest.get(key) != value
    )
    if differing:
        raise ValueError(
            "Graph failure analysis trace manifest mismatch: "
            + ", ".join(differing)
        )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Graph trace manifest cases are invalid")

    condition_tokens: dict[tuple[int, str], np.ndarray] = {}
    teacher_tokens_by_seed: dict[int, np.ndarray] = {}
    cache_sha256: dict[str, str] = {}
    dataset_fingerprints: set[str] = set()
    train_rows = tuple(int(row) for row in split.rows["train"])
    for seed in config.seeds:
        caches = _load_cache_matrix(
            config,
            bridge,
            split,
            seed,
            oracle_report_sha256=oracle_report_sha256,
        )
        teacher_tokens_by_seed[seed] = _select_diagnostic_cache_rows(
            caches["oracle_graph_v2"], train_rows
        )
        for condition in config.conditions:
            cache = caches[condition]
            condition_tokens[(seed, condition)] = _select_diagnostic_cache_rows(
                cache, train_rows
            )
            cache_sha256[f"seed_{seed}/{condition}"] = str(cache.sha256)
            dataset_fingerprints.add(str(cache.provenance.dataset_fingerprint))
    if len(dataset_fingerprints) != 1:
        raise ValueError("failure analysis caches bind different datasets")
    dataset_fingerprint = next(iter(dataset_fingerprints))
    if manifest.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("Graph failure analysis trace dataset fingerprint mismatch")
    thresholds = training_error_thresholds(
        condition_tokens=condition_tokens,
        teacher_tokens_by_seed=teacher_tokens_by_seed,
        quantile=0.75,
    )
    del source, condition_tokens, teacher_tokens_by_seed
    gc.collect()

    exposures: list[dict[str, object]] = []
    for seed in config.seeds:
        for condition in config.conditions:
            key = f"seed_{seed}/{condition}"
            for case_value in cases:
                if not isinstance(case_value, Mapping):
                    raise ValueError("Graph trace manifest case is invalid")
                case_id = str(case_value.get("case_id", ""))
                trace_path = _trace_episode_path(
                    trace_root,
                    seed=seed,
                    condition=condition,
                    case_id=case_id,
                )
                trace_records = load_trace_episode(trace_path)
                if not trace_records:
                    raise ValueError("Graph trace episode is empty")
                first = trace_records[0]
                expected_identity = {
                    "policy_seed": seed,
                    "condition": condition,
                    "case_id": case_id,
                    "environment_seed": int(case_value["environment_seed"]),
                    "layout": str(case_value["layout"]),
                    "object_count": int(case_value["object_count"]),
                }
                identity_differences = sorted(
                    name
                    for name, expected in expected_identity.items()
                    if first.get(name) != expected
                )
                if identity_differences:
                    raise ValueError(
                        "Graph trace episode identity mismatch: "
                        + ", ".join(identity_differences)
                    )
                exposures.append(
                    episode_error_exposure(
                        trace_records,
                        thresholds=thresholds[key],
                    )
                )
    expected_episodes = len(config.seeds) * len(config.conditions) * len(cases)
    if len(exposures) != expected_episodes:
        raise ValueError("Graph failure analysis episode matrix is incomplete")
    report = build_failure_analysis_report(
        exposures,
        thresholds=thresholds,
        bootstrap_samples=config.diagnostics.bootstrap_samples,
        bootstrap_seed=config.diagnostics.bootstrap_seed,
    )
    report = {
        **report,
        "schema_version": FAILURE_ANALYSIS_SCHEMA_VERSION,
        "config": config.config_path,
        "trace_root": trace_root,
        "trace_manifest": manifest_path,
        "trace_manifest_sha256": sha256_file(manifest_path),
        "split_manifest": split.path,
        "split_manifest_sha256": split.sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "cache_sha256": cache_sha256,
        "recovery_report_sha256": recovery_report_sha256,
        "oracle_report_sha256": oracle_report_sha256,
    }
    report_path = output_dir / "report.json"
    exposures_path = output_dir / "episode_exposures.jsonl"
    with _atomic_output_directory(output_dir) as staging:
        _write_jsonl_atomic(staging / "episode_exposures.jsonl", exposures)
        _write_json_atomic(staging / "report.json", report)
    return {
        "passed": bool(report["passed"]),
        "schema_version": report["schema_version"],
        "episodes": int(report["episodes"]),
        "policy_seeds": list(report["policy_seeds"]),
        "conditions": list(report["conditions"]),
        "report_path": report_path,
        "exposures_path": exposures_path,
    }


def inspect_from_config(path: str | Path) -> dict[str, object]:
    (
        config,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _context(path)
    checkpoints: dict[str, object] = {}
    for seed in config.seeds:
        for condition in config.conditions:
            if condition not in {"predicted_random_v2", "predicted_reflect_v2"}:
                continue
            payload = _load_graph_payload(
                config,
                split,
                condition=condition,
                seed=seed,
                oracle_report_sha256=oracle_report_sha256,
            )
            checkpoint = config.graph_checkpoint(condition, seed)
            assert checkpoint is not None
            checkpoints[f"seed_{seed}/{condition}"] = {
                "path": checkpoint,
                "sha256": sha256_file(checkpoint),
                "initialization": payload["initialization"],
                "fraction": payload["fraction"],
                "seed": payload["seed"],
            }
    report: dict[str, object] = {
        "passed": True,
        "conditions": list(config.conditions),
        "seeds": list(config.seeds),
        "token_dim": TOKEN_DIM,
        "dataset_root": bridge.dataset.root,
        "dataset_fingerprint": _control_dataset_fingerprint(bridge),
        "episodes": int(source.meta.total_episodes),
        "rows": len(source),
        "split_manifest": split.path,
        "split_manifest_sha256": split.sha256,
        "recovery_report": config.required_recovery_report,
        "recovery_report_sha256": recovery_report_sha256,
        "oracle_report_sha256": oracle_report_sha256,
        "partition_episodes": {
            name: len(values) for name, values in split.episodes.items()
        },
        "partition_rows": {name: len(values) for name, values in split.rows.items()},
        "graph_checkpoints": checkpoints,
        "future_relation_goal_policy_input": False,
        "oracle_provider": "causal_graph_v2_teacher",
    }
    del source
    gc.collect()
    return report


def _cache_path(root: Path, condition: str, seed: int) -> Path:
    return root / f"seed_{seed}" / f"{condition}.npz"


def _provenance(
    *,
    condition: str,
    dataset_fingerprint: str,
    split: ControlSplit,
    checkpoint: Path | None,
    payload: Mapping[str, Any] | None,
    oracle_report_sha256: str | None,
) -> CacheProvenance:
    if checkpoint is None or payload is None:
        return CacheProvenance(
            condition=condition,
            dataset_fingerprint=dataset_fingerprint,
            split_manifest_sha256=split.sha256,
            graph_checkpoint_sha256=None,
            graph_initialization=None,
            graph_fraction=None,
            graph_seed=None,
            oracle_report_sha256=oracle_report_sha256,
        )
    return CacheProvenance(
        condition=condition,
        dataset_fingerprint=dataset_fingerprint,
        split_manifest_sha256=split.sha256,
        graph_checkpoint_sha256=sha256_file(checkpoint),
        graph_initialization=str(payload["initialization"]),
        graph_fraction=float(payload["fraction"]),
        graph_seed=int(payload["seed"]),
        oracle_report_sha256=oracle_report_sha256,
    )


def _oracle_inputs(
    bridge: BridgeConfig,
    source: Any,
    split: ControlSplit,
) -> tuple[
    dict[int, tuple[int, ...]],
    dict[int, GraphV2Targets],
    GraphV2Normalization,
]:
    manifest_path = bridge.dataset.root / "meta" / "teacher_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("teacher manifest is invalid") from error
    if not isinstance(manifest, list) or not all(
        isinstance(record, Mapping) for record in manifest
    ):
        raise ValueError("teacher manifest must be a list")
    records = {int(record["episode_index"]): record for record in manifest}
    episode_rows: dict[int, tuple[int, ...]] = {}
    targets: dict[int, GraphV2Targets] = {}
    for episode in source.meta.episodes:
        episode_index = int(episode["episode_index"])
        if episode_index not in records:
            raise ValueError(f"teacher manifest is missing episode {episode_index}")
        record = records[episode_index]
        arrays = load_teacher_sidecar(
            bridge.dataset.root / str(record["path"]),
            expected_sha256=str(record["sha256"]),
        )
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        if stop - start != int(record["frames"]):
            raise ValueError("teacher sidecar frame count differs from LeRobot metadata")
        episode_rows[episode_index] = tuple(range(start, stop))
        targets[episode_index] = graph_v2_targets(arrays)
    if set(episode_rows) != set(records):
        raise ValueError("teacher episodes do not match LeRobot metadata")
    states = np.asarray(
        source.hf_dataset["observation.state"], dtype=np.float32
    )
    if states.shape != (len(source), 10) or not np.isfinite(states).all():
        raise ValueError("source end-effector state must be finite with shape [rows, 10]")
    normalization = fit_normalization(
        SimpleNamespace(
            states=states,
            row_indices=episode_rows,
            targets=targets,
        ),
        split.episodes["train"],
    )
    return episode_rows, targets, normalization


def cache_from_config(path: str | Path) -> dict[str, object]:
    (
        config,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _context(path)
    destination = config.cache.directory
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"Graph cache destination must be empty: {destination}")
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    dataset_fingerprint = _control_dataset_fingerprint(bridge)
    episode_rows, oracle_targets, normalization = _oracle_inputs(
        bridge, source, split
    )
    rows = tuple(row for episode in sorted(episode_rows) for row in episode_rows[episode])
    artifacts: dict[str, object] = {}
    try:
        for seed in config.seeds:
            for condition in config.conditions:
                checkpoint = config.graph_checkpoint(condition, seed)
                payload = (
                    None
                    if checkpoint is None
                    else _load_graph_payload(
                        config,
                        split,
                        condition=condition,
                        seed=seed,
                        oracle_report_sha256=oracle_report_sha256,
                    )
                )
                runtime = (
                    None
                    if checkpoint is None
                    else FrozenGraphRuntime(checkpoint, device=bridge.act.device)
                )
                cache = build_token_cache(
                    _cache_path(staging, condition, seed),
                    source=source,
                    episode_rows=episode_rows,
                    condition=condition,
                    runtime=runtime,
                    provenance=_provenance(
                        condition=condition,
                        dataset_fingerprint=dataset_fingerprint,
                        split=split,
                        checkpoint=checkpoint,
                        payload=payload,
                        oracle_report_sha256=oracle_report_sha256,
                    ),
                    oracle_targets=(
                        oracle_targets if condition == "oracle_graph_v2" else None
                    ),
                    normalization=(
                        normalization if condition == "oracle_graph_v2" else None
                    ),
                )
                artifacts[f"seed_{seed}/{condition}"] = cache.sha256
                del runtime
                gc.collect()
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "passed": True,
        "directory": destination,
        "rows": len(rows),
        "token_dim": TOKEN_DIM,
        "artifacts": artifacts,
        "recovery_report_sha256": recovery_report_sha256,
        "oracle_report_sha256": oracle_report_sha256,
        "future_relation_goal_used_for_token": False,
    }


def _expected_cache(
    config: GraphControlConfig,
    split: ControlSplit,
    bridge: BridgeConfig,
    *,
    condition: str,
    seed: int,
    dataset_fingerprint: str | None = None,
    payload: Mapping[str, Any] | None = None,
    oracle_report_sha256: str | None = None,
) -> CacheProvenance:
    if dataset_fingerprint is None:
        dataset_fingerprint = _control_dataset_fingerprint(bridge)
    checkpoint = config.graph_checkpoint(condition, seed)
    if checkpoint is None:
        return _provenance(
            condition=condition,
            dataset_fingerprint=dataset_fingerprint,
            split=split,
            checkpoint=None,
            payload=None,
            oracle_report_sha256=oracle_report_sha256,
        )
    if payload is None:
        payload = _load_graph_payload(
            config,
            split,
            condition=condition,
            seed=seed,
            oracle_report_sha256=oracle_report_sha256,
        )
    return _provenance(
        condition=condition,
        dataset_fingerprint=dataset_fingerprint,
        split=split,
        checkpoint=checkpoint,
        payload=payload,
        oracle_report_sha256=oracle_report_sha256,
    )


def _load_cache_matrix(
    config: GraphControlConfig,
    bridge: BridgeConfig,
    split: ControlSplit,
    seed: int,
    *,
    oracle_report_sha256: str | None,
) -> dict[str, TokenCache]:
    dataset_fingerprint = _control_dataset_fingerprint(bridge)
    payloads: dict[Path, Mapping[str, Any]] = {}
    for condition in config.conditions:
        checkpoint = config.graph_checkpoint(condition, seed)
        if checkpoint is not None and checkpoint not in payloads:
            payloads[checkpoint] = _load_graph_payload(
                config,
                split,
                condition=condition,
                seed=seed,
                oracle_report_sha256=oracle_report_sha256,
            )
    return {
        condition: load_token_cache(
            _cache_path(config.cache.directory, condition, seed),
            expected=_expected_cache(
                config,
                split,
                bridge,
                condition=condition,
                seed=seed,
                dataset_fingerprint=dataset_fingerprint,
                payload=(
                    None
                    if config.graph_checkpoint(condition, seed) is None
                    else payloads[config.graph_checkpoint(condition, seed)]
                ),
                oracle_report_sha256=oracle_report_sha256,
            ),
        )
        for condition in config.conditions
    }


def _train_from_config(path: str | Path, *, smoke: bool) -> dict[str, object]:
    (
        config,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _context(path)
    del source
    if not smoke:
        _validate_formal_epochs(config, bridge)
    destination = config.training.output_dir
    train_dataset = load_act_dataset(
        dataset_root=bridge.dataset.root,
        repo_id=bridge.dataset.repo_id,
        episodes=list(split.episodes["train"]),
    )
    validation_dataset = load_act_dataset(
        dataset_root=bridge.dataset.root,
        repo_id=bridge.dataset.repo_id,
        episodes=list(split.episodes["validation"]),
    )
    with _atomic_output_directory(destination) as staging:
        reports: dict[str, object] = {}
        for seed in config.seeds:
            caches = _load_cache_matrix(
                config,
                bridge,
                split,
                seed,
                oracle_report_sha256=oracle_report_sha256,
            )

            def train_attempt(
                batch_size: int, seed_output: Path
            ) -> Mapping[str, object]:
                return train_paired_seed(
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    caches=caches,
                    seed=seed,
                    output_dir=seed_output,
                    dataset_root=bridge.dataset.root,
                    device=resolve_device(bridge.act.device),
                    architecture="configured",
                    batch_size=batch_size,
                    smoke_steps=config.training.smoke_steps if smoke else None,
                    initial_epochs=None if smoke else config.training.formal_epochs,
                    maximum_epochs=None if smoke else config.training.formal_epochs,
                    conditions=config.conditions,
                    recovery_report_sha256=recovery_report_sha256,
                    oracle_report_sha256=oracle_report_sha256,
                    bridge_config=bridge,
                )

            reports[str(seed)] = _train_seed_with_fallback(
                staging / f"seed_{seed}",
                batch_size=bridge.act.batch_size,
                train_attempt=train_attempt,
            )
        report: dict[str, object] = {
            "passed": True,
            "mode": "smoke" if smoke else "formal",
            "conditions": list(config.conditions),
            "recovery_report_sha256": recovery_report_sha256,
            "oracle_report_sha256": oracle_report_sha256,
            "seeds": list(config.seeds),
            "fixed_epochs": None if smoke else config.training.formal_epochs,
            "reports": reports,
        }
        _write_json_atomic(staging / "comparison.json", report)
    return report


def smoke_from_config(path: str | Path) -> dict[str, object]:
    config = load_graph_control_config(path)
    if len(config.seeds) != 1:
        raise ValueError("Graph-conditioned ACT smoke requires exactly one seed")
    return _train_from_config(path, smoke=True)


def compare_from_config(path: str | Path) -> dict[str, object]:
    return _train_from_config(path, smoke=False)


def _runtime_for_condition(
    config: GraphControlConfig,
    bridge: BridgeConfig,
    split: ControlSplit,
    *,
    seed: int,
    condition: str,
    cache: TokenCache,
    recovery_report_sha256: str,
    oracle_report_sha256: str | None,
    oracle_normalization: GraphV2Normalization,
    retrospective_analysis: bool = False,
) -> GraphPolicyRuntime:
    bindings = graph_checkpoint_bindings(
        condition,
        seed,
        cache,
        recovery_report_sha256=recovery_report_sha256,
        oracle_report_sha256=oracle_report_sha256,
    )
    checkpoint = (
        config.training.output_dir / f"seed_{seed}" / condition / "checkpoint"
    )
    device = resolve_device(bridge.act.device)
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    base_metadata = LeRobotDatasetMetadata(
        bridge.dataset.repo_id, root=bridge.dataset.root
    )
    expected_metadata = expected_graph_checkpoint_metadata(
        dataset_root=bridge.dataset.root,
        features=GraphDatasetMetadata(base_metadata).features,
        act_config=_act_config(
            device=device,
            architecture="configured",
            bridge_config=bridge,
        ),
        device=device,
        bindings=bindings,
    )
    if retrospective_analysis:
        policy, preprocessor, postprocessor, _, compatibility = (
            load_graph_act_checkpoint_for_analysis(
                checkpoint,
                device=device,
                expected_metadata=expected_metadata,
            )
        )
    else:
        policy, preprocessor, postprocessor, _ = load_graph_act_checkpoint(
            checkpoint,
            device=device,
            expected_metadata=expected_metadata,
        )
        compatibility = None
    graph_checkpoint = config.graph_checkpoint(condition, seed)
    if condition == "flat":
        provider: Any = FlatTokenProvider()
    elif condition == "oracle_graph_v2":
        provider = OracleGraphV2TokenProvider(
            teacher=TCTIGTeacherExtractor(bridge.teacher),
            normalization=oracle_normalization,
        )
    else:
        assert graph_checkpoint is not None
        graph_runtime = FrozenGraphRuntime(graph_checkpoint, device=bridge.act.device)
        provider = PredictedTokenProvider(graph_runtime)
    return GraphPolicyRuntime(
        condition=condition,
        policy_seed=seed,
        checkpoint=checkpoint,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        token_provider=provider,
        checkpoint_compatibility=compatibility,
    )


def evaluate_from_config(path: str | Path) -> dict[str, object]:
    preliminary = load_graph_control_config(path)
    evaluation_dir = preliminary.training.output_dir / "evaluation"
    if evaluation_dir.exists() and (
        not evaluation_dir.is_dir() or any(evaluation_dir.iterdir())
    ):
        raise FileExistsError("Graph-conditioned ACT evaluation output already exists")
    (
        config,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _context(path)
    _, _, oracle_normalization = _oracle_inputs(bridge, source, split)
    del source
    cases = paired_evaluation_cases(
        layouts=config.evaluation.layouts,
        object_counts=config.evaluation.object_counts,
        cases_per_cell=config.evaluation.cases_per_cell,
        master_seed=config.evaluation.master_seed,
    )
    records: list[dict[str, object]] = []
    for seed in config.seeds:
        caches = _load_cache_matrix(
            config,
            bridge,
            split,
            seed,
            oracle_report_sha256=oracle_report_sha256,
        )
        for condition in config.conditions:
            runtime = _runtime_for_condition(
                config,
                bridge,
                split,
                seed=seed,
                condition=condition,
                cache=caches[condition],
                recovery_report_sha256=recovery_report_sha256,
                oracle_report_sha256=oracle_report_sha256,
                oracle_normalization=oracle_normalization,
            )
            records.extend(
                rollout_case(
                    bridge,
                    runtime,
                    case,
                    max_steps=config.evaluation.max_steps,
                )
                for case in cases
            )
            del runtime
            gc.collect()
    report = aggregate_rollouts(records, conditions=config.conditions)
    return _publish_evaluation(
        evaluation_dir,
        records=records,
        report={
            **report,
            "cases": len(cases),
            "recovery_report_sha256": recovery_report_sha256,
            "oracle_report_sha256": oracle_report_sha256,
        },
    )
