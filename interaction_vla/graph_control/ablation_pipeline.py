from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from interaction_vla.device import resolve_device
from interaction_vla.lerobot_bridge.act_smoke import _act_config, load_act_dataset
from interaction_vla.lerobot_bridge.provenance import sha256_file

from .ablation import MaskedPredictedTokenProvider, ScheduledShuffledTokenProvider
from .ablation_cache import (
    ABLATION_TRANSFORM_VERSION,
    AblationCacheProvenance,
    build_ablation_cache_matrix,
    load_ablation_token_cache,
)
from .ablation_config import AblationConfig, load_ablation_config
from .cache import TokenCache, load_token_cache
from .config import GraphControlConfig, load_graph_control_config
from .dataset import GraphDatasetMetadata
from .features import FrozenGraphRuntime
from .pipeline import (
    _atomic_output_directory,
    _cache_path,
    _context,
    _expected_cache,
    _load_graph_payload,
    _publish_evaluation,
    _train_seed_with_fallback,
    _validate_formal_epochs,
    _write_json_atomic,
)
from .rollout import (
    FlatTokenProvider,
    GraphPolicyRuntime,
    PredictedTokenProvider,
    aggregate_rollouts,
    paired_evaluation_cases,
    rollout_case,
)
from .schema import ABLATION_CONDITIONS, ALL_CONDITIONS, TOKEN_DIM
from .training import (
    ControlSplit,
    expected_graph_checkpoint_metadata,
    graph_checkpoint_bindings,
    load_graph_act_checkpoint,
    train_paired_seed,
)


ABLATION_REPORT_SCHEMA_VERSION = "control_alignment_ablation_v1"
_PARTITIONS = ("train", "validation", "test")


def partition_episode_rows(
    source: Any, split: ControlSplit
) -> dict[str, dict[int, tuple[int, ...]]]:
    metadata_rows: dict[int, tuple[int, ...]] = {}
    for record in source.meta.episodes:
        episode = int(record["episode_index"])
        start = int(record["dataset_from_index"])
        stop = int(record["dataset_to_index"])
        if episode in metadata_rows or start < 0 or stop <= start:
            raise ValueError("dataset episode metadata is invalid")
        metadata_rows[episode] = tuple(range(start, stop))
    result: dict[str, dict[int, tuple[int, ...]]] = {}
    for partition in _PARTITIONS:
        try:
            result[partition] = {
                episode: metadata_rows[episode]
                for episode in split.episodes[partition]
            }
        except KeyError as error:
            raise ValueError(
                f"{partition} episodes differ from dataset metadata"
            ) from error
        ordered = tuple(
            row
            for episode in split.episodes[partition]
            for row in result[partition][episode]
        )
        if ordered != tuple(split.rows[partition]):
            raise ValueError(f"{partition} row alignment differs from dataset metadata")
    if set(metadata_rows) != {
        episode
        for partition in _PARTITIONS
        for episode in split.episodes[partition]
    }:
        raise ValueError("split episodes do not exactly cover dataset metadata")
    return result


def build_test_reservoir_schedule(
    case_ids: Sequence[str], episode_ids: Sequence[int], *, seed: int
) -> dict[str, int]:
    cases = tuple(str(value) for value in case_ids)
    episodes = np.asarray(tuple(int(value) for value in episode_ids), dtype=np.int64)
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("test reservoir cases must be non-empty and unique")
    if len(episodes) < 2 or len(set(episodes.tolist())) != len(episodes):
        raise ValueError("test reservoir requires at least two unique episodes")
    if seed < 0:
        raise ValueError("test reservoir seed must be non-negative")
    rng = np.random.default_rng(int(seed))
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    return {
        case: int(shuffled[index % len(shuffled)])
        for index, case in enumerate(cases)
    }


def _validate_base_config(
    config: AblationConfig, base: GraphControlConfig
) -> None:
    if base.conditions != ALL_CONDITIONS:
        raise ValueError("ablation base must be the completed full Graph v2 matrix")
    if base.seeds != config.seeds:
        raise ValueError("ablation seeds must exactly match the base Graph v2 seeds")
    old_outputs = {base.cache.directory, base.training.output_dir}
    new_outputs = {
        config.cache_dir,
        config.training_output_dir,
        config.smoke_output_dir,
    }
    if old_outputs & new_outputs:
        raise ValueError("ablation outputs must be isolated from the base experiment")


def _ablation_context(path: str | Path):
    config = load_ablation_config(path)
    preliminary = load_graph_control_config(config.base_graph_control_config)
    _validate_base_config(config, preliminary)
    (
        base,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _context(config.base_graph_control_config)
    if oracle_report_sha256 is None:
        raise ValueError("ablation base must bind a passing Graph oracle report")
    return (
        config,
        base,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    )


def _source_cache(
    base: GraphControlConfig,
    bridge: Any,
    split: ControlSplit,
    *,
    seed: int,
    oracle_report_sha256: str,
) -> TokenCache:
    condition = "predicted_random_v2"
    payload = _load_graph_payload(
        base,
        split,
        condition=condition,
        seed=seed,
        oracle_report_sha256=oracle_report_sha256,
    )
    expected = _expected_cache(
        base,
        split,
        bridge,
        condition=condition,
        seed=seed,
        payload=payload,
        oracle_report_sha256=oracle_report_sha256,
    )
    return load_token_cache(
        _cache_path(base.cache.directory, condition, seed), expected=expected
    )


def _expected_ablation_provenance(
    condition: str,
    source: TokenCache,
    *,
    seed: int,
    shuffle_manifest_sha256: str | None,
) -> AblationCacheProvenance:
    provenance = source.provenance
    assert provenance.oracle_report_sha256 is not None
    assert provenance.graph_checkpoint_sha256 is not None
    return AblationCacheProvenance(
        condition=condition,
        dataset_fingerprint=provenance.dataset_fingerprint,
        split_manifest_sha256=provenance.split_manifest_sha256,
        oracle_report_sha256=provenance.oracle_report_sha256,
        source_condition="predicted_random_v2",
        source_cache_sha256=source.sha256,
        graph_checkpoint_sha256=provenance.graph_checkpoint_sha256,
        graph_initialization="random_init",
        graph_fraction=1.0,
        graph_seed=seed,
        transform_version=ABLATION_TRANSFORM_VERSION,
        shuffle_manifest_sha256=(
            shuffle_manifest_sha256 if condition == "shuffled_graph" else None
        ),
    )


def _load_ablation_caches(
    config: AblationConfig,
    base: GraphControlConfig,
    bridge: Any,
    split: ControlSplit,
    *,
    seed: int,
    oracle_report_sha256: str,
) -> tuple[dict[str, TokenCache], TokenCache]:
    source = _source_cache(
        base,
        bridge,
        split,
        seed=seed,
        oracle_report_sha256=oracle_report_sha256,
    )
    shuffle_manifest = config.cache_dir / f"seed_{seed}" / "shuffle_manifest.json"
    if not shuffle_manifest.is_file():
        raise FileNotFoundError(f"ablation shuffle manifest not found: {shuffle_manifest}")
    shuffle_sha = sha256_file(shuffle_manifest)
    caches = {
        condition: load_ablation_token_cache(
            config.cache_dir / f"seed_{seed}" / f"{condition}.npz",
            expected=_expected_ablation_provenance(
                condition,
                source,
                seed=seed,
                shuffle_manifest_sha256=shuffle_sha,
            ),
        )
        for condition in ABLATION_CONDITIONS
    }
    reference_rows = source.row_indices
    if any(
        not np.array_equal(cache.row_indices, reference_rows)
        for cache in caches.values()
    ):
        raise ValueError("ablation cache rows differ from the source Graph cache")
    return caches, source


def ablation_inspect_from_config(path: str | Path) -> dict[str, object]:
    (
        config,
        base,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _ablation_context(path)
    rows = partition_episode_rows(source, split)
    checkpoints: dict[str, object] = {}
    caches: dict[str, str] = {}
    for seed in config.seeds:
        cache = _source_cache(
            base,
            bridge,
            split,
            seed=seed,
            oracle_report_sha256=oracle_report_sha256,
        )
        checkpoint = base.graph_checkpoint("predicted_random_v2", seed)
        assert checkpoint is not None
        checkpoints[str(seed)] = {
            "path": checkpoint,
            "sha256": sha256_file(checkpoint),
        }
        caches[str(seed)] = cache.sha256
    return {
        "passed": True,
        "schema_version": ABLATION_REPORT_SCHEMA_VERSION,
        "conditions": list(config.conditions),
        "seeds": list(config.seeds),
        "source_condition": "predicted_random_v2",
        "token_dim": TOKEN_DIM,
        "partition_episodes": {
            partition: len(values) for partition, values in rows.items()
        },
        "partition_rows": {
            partition: sum(len(value) for value in rows[partition].values())
            for partition in _PARTITIONS
        },
        "source_checkpoints": checkpoints,
        "source_caches": caches,
        "recovery_report_sha256": recovery_report_sha256,
        "oracle_report_sha256": oracle_report_sha256,
        "output_isolated": True,
    }


def ablation_cache_from_config(path: str | Path) -> dict[str, object]:
    (
        config,
        base,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _ablation_context(path)
    episode_rows = partition_episode_rows(source, split)
    artifacts: dict[str, str] = {}
    with _atomic_output_directory(config.cache_dir) as staging:
        for seed in config.seeds:
            source_cache = _source_cache(
                base,
                bridge,
                split,
                seed=seed,
                oracle_report_sha256=oracle_report_sha256,
            )
            caches, _ = build_ablation_cache_matrix(
                staging / f"seed_{seed}",
                source_cache=source_cache,
                partition_episode_rows=episode_rows,
                seed=seed,
                shuffle_seed=config.shuffle_seed,
            )
            artifacts.update(
                {
                    f"seed_{seed}/{condition}": cache.sha256
                    for condition, cache in caches.items()
                }
            )
        _write_json_atomic(
            staging / "report.json",
            {
                "passed": True,
                "schema_version": ABLATION_REPORT_SCHEMA_VERSION,
                "conditions": list(config.conditions),
                "seeds": list(config.seeds),
                "source_condition": "predicted_random_v2",
                "shuffle_seed": config.shuffle_seed,
                "artifacts": artifacts,
                "recovery_report_sha256": recovery_report_sha256,
                "oracle_report_sha256": oracle_report_sha256,
            },
        )
    return {
        "passed": True,
        "schema_version": ABLATION_REPORT_SCHEMA_VERSION,
        "directory": config.cache_dir,
        "conditions": list(config.conditions),
        "seeds": list(config.seeds),
        "artifacts": artifacts,
    }


def _train_from_config(path: str | Path, *, smoke: bool) -> dict[str, object]:
    (
        config,
        base,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _ablation_context(path)
    del source
    if not smoke:
        _validate_formal_epochs(base, bridge)
    destination = config.smoke_output_dir if smoke else config.training_output_dir
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
    active_seeds = config.seeds[:1] if smoke else config.seeds
    reports: dict[str, object] = {}
    with _atomic_output_directory(destination) as staging:
        for seed in active_seeds:
            caches, source_cache = _load_ablation_caches(
                config,
                base,
                bridge,
                split,
                seed=seed,
                oracle_report_sha256=oracle_report_sha256,
            )

            def train_attempt(batch_size: int, seed_output: Path):
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
                    smoke_steps=config.smoke_steps if smoke else None,
                    initial_epochs=None if smoke else config.formal_epochs,
                    maximum_epochs=None if smoke else config.formal_epochs,
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
            reports[str(seed)]["source_cache_sha256"] = source_cache.sha256
            gc.collect()
        report = {
            "passed": True,
            "schema_version": ABLATION_REPORT_SCHEMA_VERSION,
            "mode": "smoke" if smoke else "formal",
            "conditions": list(config.conditions),
            "seeds": list(active_seeds),
            "fixed_epochs": None if smoke else config.formal_epochs,
            "reports": reports,
            "recovery_report_sha256": recovery_report_sha256,
            "oracle_report_sha256": oracle_report_sha256,
        }
        _write_json_atomic(staging / "comparison.json", report)
    return report


def ablation_smoke_from_config(path: str | Path) -> dict[str, object]:
    return _train_from_config(path, smoke=True)


def ablation_compare_from_config(path: str | Path) -> dict[str, object]:
    return _train_from_config(path, smoke=False)


def _test_sequences(
    source_cache: TokenCache,
    test_episode_rows: Mapping[int, Sequence[int]],
) -> dict[int, np.ndarray]:
    by_row = source_cache.by_row
    return {
        int(episode): np.stack([by_row[int(row)] for row in rows])
        for episode, rows in test_episode_rows.items()
    }


def _ablation_runtime(
    *,
    config: AblationConfig,
    base: GraphControlConfig,
    bridge: Any,
    cache: TokenCache,
    seed: int,
    condition: str,
    recovery_report_sha256: str,
    oracle_report_sha256: str,
    shuffled_sequences: Mapping[int, np.ndarray],
    case_schedule: Mapping[str, int],
) -> GraphPolicyRuntime:
    bindings = graph_checkpoint_bindings(
        condition,
        seed,
        cache,
        recovery_report_sha256=recovery_report_sha256,
        oracle_report_sha256=oracle_report_sha256,
    )
    checkpoint = (
        config.training_output_dir / f"seed_{seed}" / condition / "checkpoint"
    )
    device = resolve_device(bridge.act.device)
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(
        bridge.dataset.repo_id, root=bridge.dataset.root
    )
    expected = expected_graph_checkpoint_metadata(
        dataset_root=bridge.dataset.root,
        features=GraphDatasetMetadata(metadata).features,
        act_config=_act_config(
            device=device,
            architecture="configured",
            bridge_config=bridge,
        ),
        device=device,
        bindings=bindings,
    )
    policy, preprocessor, postprocessor, _ = load_graph_act_checkpoint(
        checkpoint, device=device, expected_metadata=expected
    )
    if condition == "flat":
        provider: Any = FlatTokenProvider()
    elif condition == "shuffled_graph":
        provider = ScheduledShuffledTokenProvider(
            sequences=shuffled_sequences,
            case_schedule=case_schedule,
            max_steps=base.evaluation.max_steps,
        )
    else:
        graph_checkpoint = base.graph_checkpoint("predicted_random_v2", seed)
        assert graph_checkpoint is not None
        provider = MaskedPredictedTokenProvider(
            PredictedTokenProvider(
                FrozenGraphRuntime(graph_checkpoint, device=bridge.act.device)
            ),
            condition=condition,
        )
    return GraphPolicyRuntime(
        condition=condition,
        policy_seed=seed,
        checkpoint=checkpoint,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        token_provider=provider,
    )


def ablation_evaluate_from_config(path: str | Path) -> dict[str, object]:
    preliminary = load_ablation_config(path)
    destination = preliminary.training_output_dir / "evaluation"
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError("ablation evaluation output already exists")
    (
        config,
        base,
        bridge,
        split,
        source,
        recovery_report_sha256,
        oracle_report_sha256,
    ) = _ablation_context(path)
    episode_rows = partition_episode_rows(source, split)
    del source
    cases = paired_evaluation_cases(
        layouts=base.evaluation.layouts,
        object_counts=base.evaluation.object_counts,
        cases_per_cell=base.evaluation.cases_per_cell,
        master_seed=base.evaluation.master_seed,
    )
    schedule = build_test_reservoir_schedule(
        tuple(case.case_id for case in cases),
        split.episodes["test"],
        seed=config.shuffle_seed,
    )
    records: list[dict[str, object]] = []
    source_caches: dict[str, str] = {}
    for seed in config.seeds:
        caches, source_cache = _load_ablation_caches(
            config,
            base,
            bridge,
            split,
            seed=seed,
            oracle_report_sha256=oracle_report_sha256,
        )
        source_caches[str(seed)] = source_cache.sha256
        sequences = _test_sequences(source_cache, episode_rows["test"])
        for condition in config.conditions:
            runtime = _ablation_runtime(
                config=config,
                base=base,
                bridge=bridge,
                cache=caches[condition],
                seed=seed,
                condition=condition,
                recovery_report_sha256=recovery_report_sha256,
                oracle_report_sha256=oracle_report_sha256,
                shuffled_sequences=sequences,
                case_schedule=schedule,
            )
            records.extend(
                rollout_case(
                    bridge,
                    runtime,
                    case,
                    max_steps=base.evaluation.max_steps,
                )
                for case in cases
            )
            del runtime
            gc.collect()
    report = aggregate_rollouts(records, conditions=config.conditions)
    return _publish_evaluation(
        destination,
        records=records,
        report={
            **report,
            "schema_version": ABLATION_REPORT_SCHEMA_VERSION,
            "cases": len(cases),
            "source_condition": "predicted_random_v2",
            "source_cache_sha256": source_caches,
            "test_reservoir_schedule": schedule,
            "shuffle_seed": config.shuffle_seed,
            "recovery_report_sha256": recovery_report_sha256,
            "oracle_report_sha256": oracle_report_sha256,
        },
    )
