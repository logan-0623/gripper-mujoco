from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from interaction_vla.device import resolve_device
from interaction_vla.graph_finetune.pipeline import load_finetune_checkpoint
from interaction_vla.lerobot_bridge.act_smoke import load_act_dataset
from interaction_vla.lerobot_bridge.config import BridgeConfig, load_bridge_config
from interaction_vla.lerobot_bridge.provenance import (
    sha256_file,
    standard_dataset_fingerprint,
)
from interaction_vla.lerobot_bridge.teacher import TCTIGTeacherExtractor
from interaction_vla.lerobot_bridge.validator import validate_dataset_root

from .cache import (
    CacheProvenance,
    TokenCache,
    build_token_cache,
    current_fields_from_teacher,
    load_token_cache,
    write_token_cache,
)
from .config import GraphControlConfig, load_graph_control_config
from .features import CurrentGraphFields, FrozenGraphRuntime, pack_oracle_current
from .rollout import (
    FlatTokenProvider,
    GraphPolicyRuntime,
    OracleCurrentTokenProvider,
    PredictedTokenProvider,
    aggregate_rollouts,
    paired_evaluation_cases,
    rollout_case,
)
from .schema import CONDITIONS, TOKEN_DIM
from .training import (
    ControlSplit,
    assert_checkpoint_split,
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
) -> dict[str, Any]:
    checkpoint = config.graph_checkpoint(condition, seed)
    if checkpoint is None:
        raise ValueError("flat condition has no Graph checkpoint payload")
    model, _, payload = load_finetune_checkpoint(checkpoint, device="cpu")
    del model
    assert_checkpoint_split(payload, split, condition=condition, seed=seed)
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


def _context(path: str | Path) -> tuple[GraphControlConfig, BridgeConfig, ControlSplit, Any]:
    config = load_graph_control_config(path)
    bridge = load_bridge_config(config.bridge_config)
    split = load_control_split(config.split_manifest)
    source = _load_source(bridge)
    _validate_source_alignment(source, split)
    return config, bridge, split, source


def inspect_from_config(path: str | Path) -> dict[str, object]:
    config, bridge, split, source = _context(path)
    checkpoints: dict[str, object] = {}
    for seed in config.seeds:
        for condition in ("predicted_random", "predicted_reflect"):
            payload = _load_graph_payload(
                config, split, condition=condition, seed=seed
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
        "partition_episodes": {
            name: len(values) for name, values in split.episodes.items()
        },
        "partition_rows": {name: len(values) for name, values in split.rows.items()},
        "graph_checkpoints": checkpoints,
        "future_relation_goal_policy_input": False,
        "teacher_current_fields": [
            "entity_mask",
            "entity_visibility",
            "relation_mask",
            "relation_values[12:22]",
        ],
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
        )
    return CacheProvenance(
        condition=condition,
        dataset_fingerprint=dataset_fingerprint,
        split_manifest_sha256=split.sha256,
        graph_checkpoint_sha256=sha256_file(checkpoint),
        graph_initialization=str(payload["initialization"]),
        graph_fraction=float(payload["fraction"]),
        graph_seed=int(payload["seed"]),
    )


def _current_teacher_fields(
    bridge: BridgeConfig,
    source: Any,
    *,
    selected_rows: set[int],
) -> dict[int, CurrentGraphFields]:
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
    result: dict[int, CurrentGraphFields] = {}
    for episode in source.meta.episodes:
        episode_index = int(episode["episode_index"])
        if episode_index not in records:
            raise ValueError(f"teacher manifest is missing episode {episode_index}")
        record = records[episode_index]
        sidecar = bridge.dataset.root / str(record["path"])
        if sha256_file(sidecar) != str(record["sha256"]):
            raise ValueError(f"teacher sidecar SHA-256 mismatch: {sidecar}")
        causal_keys = (
            "annotation.tc_tig.entity_mask",
            "annotation.tc_tig.entity_visibility",
            "annotation.tc_tig.relation_mask",
            "annotation.tc_tig.relation_values",
        )
        try:
            with np.load(sidecar, allow_pickle=False) as loaded:
                arrays = {name: loaded[name].copy() for name in causal_keys}
        except (KeyError, OSError, ValueError) as error:
            raise ValueError(f"causal teacher fields are invalid: {sidecar}") from error
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        if stop - start != int(record["frames"]):
            raise ValueError("teacher sidecar frame count differs from LeRobot metadata")
        for frame_index, row in enumerate(range(start, stop)):
            if row in selected_rows:
                result[row] = current_fields_from_teacher(
                    arrays, frame_index=frame_index
                )
        del arrays
    if set(result) != selected_rows:
        raise ValueError("causal teacher current fields do not cover cache rows")
    return result


def cache_from_config(path: str | Path) -> dict[str, object]:
    config, bridge, split, source = _context(path)
    destination = config.cache.directory
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(f"Graph cache destination must be empty: {destination}")
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    dataset_fingerprint = _control_dataset_fingerprint(bridge)
    rows = tuple(
        row
        for partition in ("train", "validation", "test")
        for row in split.rows[partition]
    )
    selected_rows = set(rows)
    current_fields: dict[int, CurrentGraphFields] | None = None
    artifacts: dict[str, object] = {}
    try:
        for seed in config.seeds:
            flat_provenance = _provenance(
                condition="flat",
                dataset_fingerprint=dataset_fingerprint,
                split=split,
                checkpoint=None,
                payload=None,
            )
            flat = build_token_cache(
                _cache_path(staging, "flat", seed),
                source=source,
                row_indices=rows,
                condition="flat",
                runtime=None,
                batch_size=config.cache.batch_size,
                provenance=flat_provenance,
            )
            artifacts[f"seed_{seed}/flat"] = flat.sha256

            random_payload = _load_graph_payload(
                config, split, condition="predicted_random", seed=seed
            )
            random_checkpoint = config.graph_checkpoint("predicted_random", seed)
            assert random_checkpoint is not None
            random_runtime = FrozenGraphRuntime(
                random_checkpoint, device=bridge.act.device
            )
            random_cache = build_token_cache(
                _cache_path(staging, "predicted_random", seed),
                source=source,
                row_indices=rows,
                condition="predicted_random",
                runtime=random_runtime,
                batch_size=config.cache.batch_size,
                provenance=_provenance(
                    condition="predicted_random",
                    dataset_fingerprint=dataset_fingerprint,
                    split=split,
                    checkpoint=random_checkpoint,
                    payload=random_payload,
                ),
            )
            artifacts[f"seed_{seed}/predicted_random"] = random_cache.sha256
            del random_runtime

            reflect_payload = _load_graph_payload(
                config, split, condition="predicted_reflect", seed=seed
            )
            reflect_checkpoint = config.graph_checkpoint("predicted_reflect", seed)
            assert reflect_checkpoint is not None
            reflect_runtime = FrozenGraphRuntime(
                reflect_checkpoint, device=bridge.act.device
            )
            reflect_cache = build_token_cache(
                _cache_path(staging, "predicted_reflect", seed),
                source=source,
                row_indices=rows,
                condition="predicted_reflect",
                runtime=reflect_runtime,
                batch_size=config.cache.batch_size,
                provenance=_provenance(
                    condition="predicted_reflect",
                    dataset_fingerprint=dataset_fingerprint,
                    split=split,
                    checkpoint=reflect_checkpoint,
                    payload=reflect_payload,
                ),
            )
            artifacts[f"seed_{seed}/predicted_reflect"] = reflect_cache.sha256
            if current_fields is None:
                current_fields = _current_teacher_fields(
                    bridge, source, selected_rows=selected_rows
                )
            oracle_tokens = np.stack(
                [
                    pack_oracle_current(
                        current_fields[int(row)],
                        reflect_cache.tokens[index],
                        reflect_runtime.normalization,
                    )
                    for index, row in enumerate(reflect_cache.row_indices)
                ],
                axis=0,
            )
            oracle_cache = write_token_cache(
                _cache_path(staging, "oracle_current", seed),
                reflect_cache.row_indices,
                oracle_tokens,
                _provenance(
                    condition="oracle_current",
                    dataset_fingerprint=dataset_fingerprint,
                    split=split,
                    checkpoint=reflect_checkpoint,
                    payload=reflect_payload,
                ),
            )
            artifacts[f"seed_{seed}/oracle_current"] = oracle_cache.sha256
            del reflect_runtime
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
        )
    if payload is None:
        payload = _load_graph_payload(config, split, condition=condition, seed=seed)
    return _provenance(
        condition=condition,
        dataset_fingerprint=dataset_fingerprint,
        split=split,
        checkpoint=checkpoint,
        payload=payload,
    )


def _load_cache_matrix(
    config: GraphControlConfig, bridge: BridgeConfig, split: ControlSplit, seed: int
) -> dict[str, TokenCache]:
    dataset_fingerprint = _control_dataset_fingerprint(bridge)
    payloads: dict[Path, Mapping[str, Any]] = {}
    for condition in CONDITIONS:
        checkpoint = config.graph_checkpoint(condition, seed)
        if checkpoint is not None and checkpoint not in payloads:
            payloads[checkpoint] = _load_graph_payload(
                config, split, condition=condition, seed=seed
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
            ),
        )
        for condition in CONDITIONS
    }


def _train_from_config(path: str | Path, *, smoke: bool) -> dict[str, object]:
    config, bridge, split, source = _context(path)
    del source
    destination = config.training.output_dir
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Graph-conditioned ACT output must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
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
    reports: dict[str, object] = {}
    for seed in config.seeds:
        reports[str(seed)] = train_paired_seed(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            caches=_load_cache_matrix(config, bridge, split, seed),
            seed=seed,
            output_dir=destination / f"seed_{seed}",
            dataset_root=bridge.dataset.root,
            device=resolve_device(bridge.act.device),
            architecture="configured",
            batch_size=bridge.act.batch_size,
            smoke_steps=config.training.smoke_steps if smoke else None,
            initial_epochs=None if smoke else bridge.act.epochs,
            maximum_epochs=None if smoke else bridge.act.epochs,
            bridge_config=bridge,
        )
    report: dict[str, object] = {
        "passed": True,
        "mode": "smoke" if smoke else "formal",
        "conditions": list(CONDITIONS),
        "seeds": list(config.seeds),
        "fixed_epochs": None if smoke else bridge.act.epochs,
        "reports": reports,
    }
    _write_json_atomic(destination / "comparison.json", report)
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
) -> GraphPolicyRuntime:
    bindings = graph_checkpoint_bindings(condition, seed, cache)
    checkpoint = (
        config.training.output_dir / f"seed_{seed}" / condition / "checkpoint"
    )
    device = resolve_device(bridge.act.device)
    policy, preprocessor, postprocessor, _ = load_graph_act_checkpoint(
        checkpoint,
        device=device,
        expected_bindings=bindings,
    )
    graph_checkpoint = config.graph_checkpoint(condition, seed)
    if condition == "flat":
        provider: Any = FlatTokenProvider()
    else:
        assert graph_checkpoint is not None
        graph_runtime = FrozenGraphRuntime(graph_checkpoint, device=bridge.act.device)
        if condition == "oracle_current":
            provider = OracleCurrentTokenProvider(
                graph_runtime, TCTIGTeacherExtractor(bridge.teacher)
            )
        else:
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
    config, bridge, split, source = _context(path)
    del source
    cases = paired_evaluation_cases(
        layouts=config.evaluation.layouts,
        object_counts=config.evaluation.object_counts,
        cases_per_cell=config.evaluation.cases_per_cell,
        master_seed=config.evaluation.master_seed,
    )
    records: list[dict[str, object]] = []
    for seed in config.seeds:
        caches = _load_cache_matrix(config, bridge, split, seed)
        for condition in CONDITIONS:
            runtime = _runtime_for_condition(
                config,
                bridge,
                split,
                seed=seed,
                condition=condition,
                cache=caches[condition],
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
    report = aggregate_rollouts(records)
    evaluation_dir = config.training.output_dir / "evaluation"
    report_path = evaluation_dir / "report.json"
    records_path = evaluation_dir / "episodes.jsonl"
    if report_path.exists() or records_path.exists():
        raise FileExistsError("Graph-conditioned ACT evaluation output already exists")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    temporary = records_path.with_suffix(".jsonl.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, records_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    final = {
        **report,
        "cases": len(cases),
        "episodes_path": records_path,
        "report_path": report_path,
    }
    _write_json_atomic(report_path, final)
    return final
