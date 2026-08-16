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
from typing import Any, Callable, Mapping

import numpy as np
import torch

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
from .features import FrozenGraphRuntime
from .rollout import (
    FlatTokenProvider,
    GraphPolicyRuntime,
    OracleGraphV2TokenProvider,
    PredictedTokenProvider,
    aggregate_rollouts,
    paired_evaluation_cases,
    rollout_case,
)
from .schema import ALL_CONDITIONS, ORACLE_CONDITIONS, TOKEN_DIM
from .training import (
    ControlSplit,
    assert_checkpoint_split,
    expected_graph_checkpoint_metadata,
    graph_checkpoint_bindings,
    load_control_split,
    load_graph_act_checkpoint,
    train_paired_seed,
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
    policy, preprocessor, postprocessor, _ = load_graph_act_checkpoint(
        checkpoint,
        device=device,
        expected_metadata=expected_metadata,
    )
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
