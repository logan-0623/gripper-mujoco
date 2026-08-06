from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import interaction_vla.train as train_module
import interaction_vla.physics_data as physics_data_module
from interaction_vla.data import (
    collect_episode,
    collect_recovery_episode,
    save_episode,
    split_episode_seeds,
)
from interaction_vla.env import KinematicTabletopEnv
from interaction_vla.expert import ScriptedExpert
from interaction_vla.models.policy import build_action_policy
from interaction_vla.recovery import make_recovery_spec
from interaction_vla.source_split import (
    deterministic_source_split,
    save_source_split,
    select_training_recovery_sources,
)
from interaction_vla.train import (
    EpisodeFrameDataset,
    TrainingStatistics,
    build_sequence_provenance_fields,
    build_training_provenance,
    evaluate_normalized_mse,
    hash_episode_contents,
    load_training_checkpoint,
    resolve_training_data,
    train_policy,
    train_from_config,
)


MODEL_KWARGS = {
    "max_nodes": 8,
    "max_edges": 56,
    "node_feature_dim": 23,
    "edge_feature_dim": 10,
    "graph_hidden_dim": 16,
    "embedding_dim": 12,
    "policy_hidden_dim": 24,
    "message_rounds": 1,
}


def write_v3_episode(
    path: Path,
    *,
    seed: int,
    action_value: float,
    trajectory_kind: str = "base",
    source_seed: int | None = None,
    source_split: str | None = None,
    variant_id: int | None = None,
) -> Path:
    edge_index = np.asarray(
        tuple(
            (source, target)
            for source in range(8)
            for target in range(8)
            if source != target
        ),
        dtype=np.int64,
    ).T
    metadata = {
        "seed": seed,
        "object_count": 2,
        "target_name": "object_0",
        "reason": "success",
        "trajectory_kind": trajectory_kind,
        "source_seed": source_seed,
        "source_split": source_split,
        "variant_id": variant_id,
        "perturbation_kind": (
            "wrong_way_transport" if trajectory_kind == "recovery" else None
        ),
        "injection_phase": "transport" if trajectory_kind == "recovery" else None,
    }
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        node_features=np.zeros((2, 8, 23), dtype=np.float32),
        edge_index=edge_index,
        edge_features=np.zeros((2, 56, 18), dtype=np.float32),
        node_mask=np.ones((2, 8), dtype=np.bool_),
        edge_mask=np.ones((2, 56), dtype=np.bool_),
        proprioception=np.zeros((2, 23), dtype=np.float32),
        actions=np.full((2, 7), action_value, dtype=np.float32),
        phases=np.asarray(("approach", "transport")),
    )
    return path


def write_v3_source_manifest_fixture(tmp_path: Path) -> SimpleNamespace:
    split = deterministic_source_split(range(100, 110), seed=42)
    training_recovery_sources = select_training_recovery_sources(
        split.train,
        fraction=0.25,
        seed=42,
    )
    benchmark_sources = tuple(sorted(split.validation + split.test))
    save_source_split(
        tmp_path / "source_split.json",
        split,
        training_recovery_sources=training_recovery_sources,
        benchmark_sources=benchmark_sources,
    )
    split_by_seed = {
        source_seed: split_name
        for split_name, source_seeds in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
        for source_seed in source_seeds
    }
    base_records = []
    base_paths = []
    for index, source_seed in enumerate(reversed(range(100, 110))):
        split_name = split_by_seed[source_seed]
        action_value = 0.0 if split_name == "train" else 100.0
        path = write_v3_episode(
            tmp_path / f"shuffled_base_{index:02d}.npz",
            seed=source_seed,
            action_value=action_value,
        )
        base_paths.append(path)
        base_records.append(
            {
                "path": path.name,
                "seed": source_seed,
                "source_seed": source_seed,
                "source_split": split_name,
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(base_records), encoding="utf-8")

    training_records = []
    for variant_id, source_seed in enumerate(training_recovery_sources):
        path = write_v3_episode(
            tmp_path / f"train_recovery_{variant_id:02d}.npz",
            seed=source_seed,
            source_seed=source_seed,
            source_split="train",
            variant_id=variant_id,
            trajectory_kind="recovery",
            action_value=50.0,
        )
        training_records.append(
            {
                "path": path.name,
                "source_seed": source_seed,
                "source_split": "train",
                "variant_id": variant_id,
                "kind": "wrong_way_transport",
            }
        )
    (tmp_path / "recovery_manifest.json").write_text(
        json.dumps(training_records), encoding="utf-8"
    )

    benchmark_records = []
    for variant_id, source_seed in enumerate(benchmark_sources):
        split_name = split_by_seed[source_seed]
        path = write_v3_episode(
            tmp_path / f"benchmark_recovery_{variant_id:02d}.npz",
            seed=source_seed,
            source_seed=source_seed,
            source_split=split_name,
            variant_id=variant_id,
            trajectory_kind="recovery",
            action_value=200.0,
        )
        benchmark_records.append(
            {
                "path": path.name,
                "source_seed": source_seed,
                "source_split": split_name,
                "variant_id": variant_id,
                "kind": "wrong_way_transport",
            }
        )
    benchmark_manifest = tmp_path / "recovery_benchmark_manifest.json"
    benchmark_manifest.write_text(json.dumps(benchmark_records), encoding="utf-8")
    return SimpleNamespace(
        data_dir=tmp_path,
        split=split,
        training_recovery_sources=training_recovery_sources,
        benchmark_sources=benchmark_sources,
        manifest_path=manifest_path,
        base_paths=tuple(base_paths),
    )


def episode_seeds(paths) -> set[int]:
    return {int(train_module.load_episode_arrays(path).seed) for path in paths}


def tiny_episode_files(tmp_path):
    paths = []
    for index, seed in enumerate((11, 12)):
        episode = collect_episode(
            KinematicTabletopEnv(max_objects=5),
            ScriptedExpert(),
            seed=seed,
            object_count=2 + index,
        )
        episode = replace(episode, steps=episode.steps[:4])
        paths.append(save_episode(episode, tmp_path / f"episode_{index}.npz"))
    return paths


def test_statistics_are_fit_only_from_the_supplied_training_episodes(tmp_path) -> None:
    paths = tiny_episode_files(tmp_path)
    statistics = TrainingStatistics.fit(paths[:1])

    with np.load(paths[0], allow_pickle=False) as archive:
        expected = archive["actions"].mean(axis=0)
    np.testing.assert_allclose(statistics.action_mean, expected)


def test_training_selection_obeys_saved_source_manifests(tmp_path: Path) -> None:
    layout = write_v3_source_manifest_fixture(tmp_path)

    selection = resolve_training_data(tmp_path, include_recovery=True)

    assert episode_seeds(selection.base_train_paths) == set(layout.split.train)
    assert episode_seeds(selection.validation_paths) == set(layout.split.validation)
    assert episode_seeds(selection.test_paths) == set(layout.split.test)
    assert episode_seeds(selection.recovery_train_paths) == set(
        layout.training_recovery_sources
    )
    assert episode_seeds(selection.recovery_benchmark_paths) == set(
        layout.benchmark_sources
    )
    assert set(selection.recovery_train_paths).isdisjoint(
        selection.recovery_benchmark_paths
    )


def test_v3_statistics_are_fit_from_training_base_episodes_only(tmp_path: Path) -> None:
    write_v3_source_manifest_fixture(tmp_path)
    selection = resolve_training_data(tmp_path, include_recovery=True)

    statistics = TrainingStatistics.fit(selection.base_train_paths)

    np.testing.assert_allclose(statistics.action_mean, np.zeros(7), atol=1e-6)


def test_training_selection_rejects_split_metadata_mismatch(tmp_path: Path) -> None:
    layout = write_v3_source_manifest_fixture(tmp_path)
    records = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    records[0]["source_split"] = "test"
    layout.manifest_path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="source split mismatch"):
        resolve_training_data(tmp_path, include_recovery=True)


def test_v3_training_provenance_binds_all_source_manifests(tmp_path: Path) -> None:
    write_v3_source_manifest_fixture(tmp_path)
    config = replace(
        train_module.load_config(
            "configs/physics_interaction_chunk_smoke_macos.yaml"
        ),
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "output"),
    )
    selection = resolve_training_data(tmp_path, include_recovery=True)

    provenance, all_paths = build_training_provenance(
        config,
        selection,
        expert_gate_hash="g" * 64,
    )

    assert provenance["source_split_hash"] == selection.source_split_hash
    assert provenance["manifest_hash"] == selection.manifest_hash
    assert provenance["recovery_manifest_hash"] == selection.recovery_manifest_hash
    assert provenance["recovery_benchmark_manifest_hash"] == (
        selection.recovery_benchmark_manifest_hash
    )
    assert set(all_paths) == set(
        selection.base_train_paths
        + selection.validation_paths
        + selection.test_paths
        + selection.recovery_train_paths
        + selection.recovery_benchmark_paths
    )


def test_dataset_content_hash_binds_episode_bytes_and_order(tmp_path) -> None:
    paths = tiny_episode_files(tmp_path)
    original = hash_episode_contents(paths)

    assert hash_episode_contents(reversed(paths)) != original
    paths[0].write_bytes(paths[0].read_bytes() + b"changed")
    assert hash_episode_contents(paths) != original


def test_flat_and_graph_receive_identical_ordered_combined_training_frames(
    tmp_path,
) -> None:
    base_records = []
    seeds = tuple(range(21, 27))
    object_count_by_seed = {}
    for index, seed in enumerate(seeds):
        object_count = 2 + index % 2
        object_count_by_seed[seed] = object_count
        episode = collect_episode(
            KinematicTabletopEnv(max_objects=5),
            ScriptedExpert(),
            seed=seed,
            object_count=object_count,
        )
        path = save_episode(episode, tmp_path / f"episode_{index:03d}.npz")
        base_records.append({"path": path.name, "seed": seed})
    (tmp_path / "manifest.json").write_text(
        json.dumps(base_records), encoding="utf-8"
    )
    splits = split_episode_seeds(
        seeds, validation_fraction=0.1, test_fraction=0.1, seed=5
    )
    recovery_records = []
    for seed in splits.train:
        recovery = collect_recovery_episode(
            KinematicTabletopEnv(max_objects=5),
            ScriptedExpert(),
            source_seed=seed,
            object_count=object_count_by_seed[seed],
            spec=make_recovery_spec(seed, 0),
        )
        path = save_episode(recovery, tmp_path / f"recovery_{seed}.npz")
        recovery_records.append({"path": path.name, "source_seed": seed})
    (tmp_path / "recovery_manifest.json").write_text(
        json.dumps(recovery_records), encoding="utf-8"
    )

    selection = resolve_training_data(
        tmp_path, split_seed=5, include_recovery=True
    )
    statistics = TrainingStatistics.fit(selection.combined_paths)
    flat_dataset = EpisodeFrameDataset(selection.combined_paths, statistics)
    graph_dataset = EpisodeFrameDataset(selection.combined_paths, statistics)

    assert selection.recovery_paths == tuple(
        tmp_path / record["path"] for record in recovery_records
    )
    assert selection.combined_paths == (
        selection.base_training_paths + selection.recovery_paths
    )
    for name in (
        "node_features",
        "edge_features",
        "node_mask",
        "edge_mask",
        "proprioception",
        "actions",
        "sample_weights",
    ):
        torch.testing.assert_close(
            getattr(flat_dataset, name), getattr(graph_dataset, name)
        )
    np.testing.assert_array_equal(flat_dataset.phases, graph_dataset.phases)


def test_phase_balancing_upweights_rare_gripper_transitions(tmp_path) -> None:
    episode = collect_episode(
        KinematicTabletopEnv(max_objects=5), ScriptedExpert(), seed=31, object_count=2
    )
    path = save_episode(episode, tmp_path / "episode.npz")
    statistics = TrainingStatistics.fit((path,))
    dataset = EpisodeFrameDataset((path,), statistics)

    close_weight = dataset.sample_weights[dataset.phases == "close"].mean()
    approach_weight = dataset.sample_weights[dataset.phases == "approach"].mean()

    assert close_weight > approach_weight
    assert dataset.sample_weights.mean().item() == pytest.approx(1.0)


def test_flat_policy_can_overfit_two_tiny_episodes(tmp_path) -> None:
    torch.manual_seed(4)
    paths = tiny_episode_files(tmp_path)
    statistics = TrainingStatistics.fit(paths)
    dataset = EpisodeFrameDataset(paths, statistics)
    policy = build_action_policy(representation="flat", **MODEL_KWARGS)

    result = train_policy(
        policy,
        dataset,
        statistics,
        epochs=350,
        batch_size=len(dataset),
        learning_rate=8e-3,
        seed=9,
        device="cpu",
    )

    assert result.final_normalized_mse < 1e-3
    assert evaluate_normalized_mse(policy, dataset, statistics, "cpu") < 1e-3


def test_train_policy_reports_epoch_progress(tmp_path, monkeypatch) -> None:
    paths = tiny_episode_files(tmp_path)
    statistics = TrainingStatistics.fit(paths)
    dataset = EpisodeFrameDataset(paths, statistics)
    policy = build_action_policy(representation="graph", **MODEL_KWARGS)
    created_progress = []

    class ProgressSpy:
        def __init__(self, iterable, **kwargs) -> None:
            self.iterable = iterable
            self.kwargs = kwargs
            self.postfixes = []

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs) -> None:
            self.postfixes.append(kwargs)

    def fake_tqdm(iterable, **kwargs):
        progress = ProgressSpy(iterable, **kwargs)
        created_progress.append(progress)
        return progress

    monkeypatch.setattr(train_module, "tqdm", fake_tqdm, raising=False)
    train_policy(
        policy,
        dataset,
        statistics,
        epochs=2,
        batch_size=len(dataset),
        learning_rate=1e-3,
        seed=7,
        device="cpu",
        representation="graph",
        show_progress=True,
    )

    assert len(created_progress) == 1
    progress = created_progress[0]
    assert progress.kwargs["desc"] == "graph seed=7"
    assert progress.kwargs["total"] == 2
    assert progress.kwargs["unit"] == "epoch"
    assert len(progress.postfixes) == 2
    assert all("mse" in postfix for postfix in progress.postfixes)


def test_checkpoint_resume_continues_steps_and_optimizer_state(tmp_path) -> None:
    paths = tiny_episode_files(tmp_path)
    statistics = TrainingStatistics.fit(paths)
    dataset = EpisodeFrameDataset(paths, statistics)
    checkpoint = tmp_path / "policy.pt"

    torch.manual_seed(8)
    first_policy = build_action_policy(representation="flat", **MODEL_KWARGS)
    first = train_policy(
        first_policy,
        dataset,
        statistics,
        epochs=2,
        batch_size=len(dataset),
        learning_rate=1e-3,
        seed=3,
        device="cpu",
        checkpoint_path=checkpoint,
        representation="flat",
        model_kwargs=MODEL_KWARGS,
    )
    first_weights = {
        name: value.detach().clone() for name, value in first_policy.state_dict().items()
    }

    resumed_policy, restored_statistics, payload = load_training_checkpoint(checkpoint, "cpu")
    assert payload["global_step"] == first.global_step
    np.testing.assert_allclose(restored_statistics.action_mean, statistics.action_mean)

    resumed = train_policy(
        resumed_policy,
        dataset,
        restored_statistics,
        epochs=1,
        batch_size=len(dataset),
        learning_rate=1e-3,
        seed=3,
        device="cpu",
        checkpoint_path=checkpoint,
        resume_from=checkpoint,
        representation="flat",
        model_kwargs=MODEL_KWARGS,
    )

    assert resumed.global_step == first.global_step + 1
    assert any(
        not torch.equal(first_weights[name], value)
        for name, value in resumed_policy.state_dict().items()
    )


def test_resumed_training_matches_uninterrupted_training_with_recovery_data(tmp_path) -> None:
    paths = tiny_episode_files(tmp_path)
    recovery = collect_recovery_episode(
        KinematicTabletopEnv(max_objects=5),
        ScriptedExpert(),
        source_seed=11,
        object_count=2,
        spec=make_recovery_spec(11, 0),
    )
    paths.append(save_episode(recovery, tmp_path / "recovery_11.npz"))
    statistics = TrainingStatistics.fit(paths)
    dataset = EpisodeFrameDataset(paths, statistics)
    checkpoint = tmp_path / "resume.pt"
    provenance = {
        "base_train_seeds": [11, 12],
        "recovery_filenames": ["recovery_11.npz"],
        "recovery_count": 1,
    }

    torch.manual_seed(21)
    uninterrupted_policy = build_action_policy(representation="flat", **MODEL_KWARGS)
    train_policy(
        uninterrupted_policy,
        dataset,
        statistics,
        epochs=3,
        batch_size=3,
        learning_rate=1e-3,
        seed=15,
        device="cpu",
    )

    torch.manual_seed(21)
    staged_policy = build_action_policy(representation="flat", **MODEL_KWARGS)
    train_policy(
        staged_policy,
        dataset,
        statistics,
        epochs=2,
        batch_size=3,
        learning_rate=1e-3,
        seed=15,
        device="cpu",
        checkpoint_path=checkpoint,
        representation="flat",
        model_kwargs=MODEL_KWARGS,
        training_provenance=provenance,
    )
    resumed_policy, restored_statistics, _ = load_training_checkpoint(checkpoint, "cpu")
    train_policy(
        resumed_policy,
        dataset,
        restored_statistics,
        epochs=1,
        batch_size=3,
        learning_rate=1e-3,
        seed=15,
        device="cpu",
        resume_from=checkpoint,
        training_provenance=provenance,
    )

    for name, expected in uninterrupted_policy.state_dict().items():
        torch.testing.assert_close(resumed_policy.state_dict()[name], expected, atol=0, rtol=0)


def test_fresh_run_replaces_stale_metrics_log(tmp_path) -> None:
    paths = tiny_episode_files(tmp_path)
    statistics = TrainingStatistics.fit(paths)
    dataset = EpisodeFrameDataset(paths, statistics)
    metrics_path = tmp_path / "metrics.jsonl"

    for model_seed in (1, 2):
        torch.manual_seed(model_seed)
        policy = build_action_policy(representation="proprio", embedding_dim=8, policy_hidden_dim=8)
        train_policy(
            policy,
            dataset,
            statistics,
            epochs=1,
            batch_size=len(dataset),
            learning_rate=1e-3,
            seed=model_seed,
            device="cpu",
            metrics_path=metrics_path,
        )

    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "weighted_train_mse" in json.loads(lines[0])


def test_resume_rejects_changed_normalization_statistics(tmp_path) -> None:
    paths = tiny_episode_files(tmp_path)
    statistics = TrainingStatistics.fit(paths)
    dataset = EpisodeFrameDataset(paths, statistics)
    checkpoint = tmp_path / "policy.pt"
    policy = build_action_policy(representation="flat", **MODEL_KWARGS)
    train_policy(
        policy,
        dataset,
        statistics,
        epochs=1,
        batch_size=len(dataset),
        learning_rate=1e-3,
        seed=2,
        device="cpu",
        checkpoint_path=checkpoint,
        representation="flat",
        model_kwargs=MODEL_KWARGS,
    )
    changed_statistics = replace(
        statistics, action_mean=statistics.action_mean + np.float32(1.0)
    )

    with pytest.raises(ValueError, match="statistics"):
        train_policy(
            policy,
            EpisodeFrameDataset(paths, changed_statistics),
            changed_statistics,
            epochs=1,
            batch_size=len(dataset),
            learning_rate=1e-3,
            seed=2,
            device="cpu",
            resume_from=checkpoint,
        )


def test_resume_rejects_changed_training_provenance(tmp_path) -> None:
    paths = tiny_episode_files(tmp_path)
    statistics = TrainingStatistics.fit(paths)
    dataset = EpisodeFrameDataset(paths, statistics)
    checkpoint = tmp_path / "policy.pt"
    policy = build_action_policy(representation="flat", **MODEL_KWARGS)
    provenance = {
        "base_train_seeds": [11],
        "recovery_filenames": ["recovery_11.npz"],
        "recovery_count": 1,
    }
    train_policy(
        policy,
        dataset,
        statistics,
        epochs=1,
        batch_size=len(dataset),
        learning_rate=1e-3,
        seed=2,
        device="cpu",
        checkpoint_path=checkpoint,
        representation="flat",
        model_kwargs=MODEL_KWARGS,
        training_provenance=provenance,
    )

    changed = dict(provenance, recovery_filenames=["different.npz"])
    with pytest.raises(ValueError, match="provenance"):
        train_policy(
            policy,
            dataset,
            statistics,
            epochs=1,
            batch_size=len(dataset),
            learning_rate=1e-3,
            seed=2,
            device="cpu",
            resume_from=checkpoint,
            training_provenance=changed,
        )


@pytest.mark.parametrize("representation", ["flat", "graph"])
def test_v3_trainer_uses_shared_h8_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    representation: str,
) -> None:
    write_v3_source_manifest_fixture(tmp_path / "data")
    base = train_module.load_config(
        "configs/physics_interaction_chunk_smoke_macos.yaml"
    )
    config = replace(
        base,
        device="cpu",
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
    )
    physical_hashes = {
        "expert_gate_hash": "a" * 64,
        "controller_hash": "b" * 64,
        "scene_hash": "c" * 64,
        "config_hash": "d" * 64,
    }
    monkeypatch.setattr(train_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        physics_data_module,
        "expert_gate_provenance",
        lambda *args: physical_hashes,
    )
    monkeypatch.setattr(
        physics_data_module,
        "require_episode_gate_provenance",
        lambda *args: None,
    )

    checkpoint_path = train_from_config("ignored.yaml", representation, model_seed=0)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    summary = json.loads(
        checkpoint_path.with_name("training_summary.json").read_text(encoding="utf-8")
    )

    assert checkpoint["model_kwargs"]["action_horizon"] == 8
    assert checkpoint["temporal_contract"]["contribution"] == "shared_infrastructure"
    assert checkpoint["temporal_contract"]["recovery_loss_fraction"] == 0.25
    assert checkpoint["representation_contract"]["experimental_variable"] == (
        "encoder_only"
    )
    assert checkpoint["data_provenance"]["source_split_hash"]
    assert checkpoint["data_provenance"]["recovery_benchmark_manifest_hash"]
    reconstructed = build_sequence_provenance_fields(
        config,
        resolve_training_data(
            config.data_dir,
            split_seed=config.seed,
            include_recovery=True,
        ),
        model_seed=0,
    )
    assert all(
        checkpoint["training_provenance"][key] == value
        for key, value in reconstructed.items()
    )
    assert summary["effective_loss_mass"] == {"base": 0.75, "recovery": 0.25}
    assert summary["base_training_episodes"] == 8
    assert summary["recovery_training_episodes"] == 2
