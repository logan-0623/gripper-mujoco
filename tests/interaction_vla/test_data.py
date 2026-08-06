from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import yaml

from interaction_vla.data import (
    augment_recovery_from_config,
    collect_episode,
    collect_recovery_episode,
    episode_paths_from_manifest,
    load_episode_arrays,
    recovery_paths_from_manifest,
    save_episode,
    split_episode_seeds,
    collect_from_config,
)
from interaction_vla.env import KinematicTabletopEnv, TerminationReason
from interaction_vla.expert import ScriptedExpert
from interaction_vla.recovery import make_recovery_spec


def test_collector_stores_pre_action_snapshot() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    episode = collect_episode(env, ScriptedExpert(), seed=3, object_count=2)
    first = episode.steps[0]

    assert first.snapshot.gripper.position[0] == np.float32(-0.35)
    assert first.action[0] != 0.0
    np.testing.assert_allclose(first.graph.node_features[0, 4:7], first.snapshot.gripper.position)
    assert episode.reason is TerminationReason.SUCCESS


def test_episode_npz_round_trip(tmp_path) -> None:
    episode = collect_episode(
        KinematicTabletopEnv(max_objects=5), ScriptedExpert(), seed=5, object_count=3
    )
    path = tmp_path / "episode_000005.npz"

    save_episode(episode, path)
    arrays = load_episode_arrays(path)

    assert arrays.seed == 5
    assert arrays.object_count == 3
    assert arrays.reason == TerminationReason.SUCCESS.value
    assert arrays.node_features.shape[0] == len(episode.steps)
    np.testing.assert_allclose(arrays.actions[0], episode.steps[0].action)
    np.testing.assert_allclose(arrays.proprioception[0], episode.steps[0].proprioception)


def test_recovery_metadata_round_trips_and_legacy_metadata_defaults_to_base(tmp_path) -> None:
    base = collect_episode(
        KinematicTabletopEnv(max_objects=5), ScriptedExpert(), seed=18, object_count=2
    )
    recovery = replace(
        base,
        trajectory_kind="recovery",
        source_seed=18,
        variant_id=2,
        perturbation_kind="lift_offset",
        injection_phase="lift",
    )
    recovery_path = save_episode(recovery, tmp_path / "recovery.npz")

    arrays = load_episode_arrays(recovery_path)

    assert arrays.trajectory_kind == "recovery"
    assert arrays.source_seed == 18
    assert arrays.variant_id == 2
    assert arrays.perturbation_kind == "lift_offset"
    assert arrays.injection_phase == "lift"

    with np.load(recovery_path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(payload["metadata"].item()))
    for key in (
        "trajectory_kind",
        "source_seed",
        "variant_id",
        "perturbation_kind",
        "injection_phase",
    ):
        metadata.pop(key)
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    legacy_path = tmp_path / "legacy.npz"
    np.savez_compressed(legacy_path, **payload)

    legacy = load_episode_arrays(legacy_path)

    assert legacy.trajectory_kind == "base"
    assert legacy.source_seed is None
    assert legacy.variant_id is None
    assert legacy.perturbation_kind is None
    assert legacy.injection_phase is None


def test_recovery_collection_saves_only_deterministic_post_perturbation_frames(
    tmp_path,
) -> None:
    spec = make_recovery_spec(42, 0)

    prefix_env = KinematicTabletopEnv(max_objects=5)
    prefix_expert = ScriptedExpert()
    unperturbed = prefix_env.reset(seed=42, object_count=2)
    while True:
        action = prefix_expert.act(unperturbed)
        if prefix_expert.phase is spec.injection_phase:
            break
        transition = prefix_env.step(action)
        unperturbed = transition.snapshot
        assert not transition.done

    first = collect_recovery_episode(
        KinematicTabletopEnv(max_objects=5),
        ScriptedExpert(),
        source_seed=42,
        object_count=2,
        spec=spec,
    )
    second = collect_recovery_episode(
        KinematicTabletopEnv(max_objects=5),
        ScriptedExpert(),
        source_seed=42,
        object_count=2,
        spec=spec,
    )

    assert first.reason is TerminationReason.SUCCESS
    assert first.trajectory_kind == "recovery"
    assert first.steps[0].phase.value == first.injection_phase
    assert not np.array_equal(
        first.steps[0].snapshot.gripper.position,
        unperturbed.gripper.position,
    )

    first_arrays = load_episode_arrays(save_episode(first, tmp_path / "first.npz"))
    second_arrays = load_episode_arrays(save_episode(second, tmp_path / "second.npz"))
    for name in (
        "node_features",
        "edge_features",
        "node_mask",
        "edge_mask",
        "proprioception",
        "actions",
        "phases",
    ):
        np.testing.assert_array_equal(getattr(first_arrays, name), getattr(second_arrays, name))


def test_recovery_generation_uses_only_base_training_seeds(tmp_path) -> None:
    config_path = tmp_path / "recovery.yaml"
    data_dir = tmp_path / "data"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "split_isolation",
                "seed": 7,
                "device": "cpu",
                "max_objects": 5,
                "data_dir": str(data_dir),
                "output_dir": str(tmp_path / "outputs"),
                "train": {
                    "object_counts": [2, 3],
                    "episodes": 6,
                    "batch_size": 8,
                    "epochs": 1,
                    "model_seeds": [0],
                },
                "eval": {
                    "object_counts": [2, 4],
                    "ood_object_counts": [4],
                    "crowded_object_counts": [4],
                    "episodes_per_count": 1,
                    "max_steps": 10,
                },
                "recovery": {"enabled": True, "variants_per_episode": 1},
            }
        ),
        encoding="utf-8",
    )
    collect_from_config(config_path)
    expected = split_episode_seeds(
        range(7, 13), validation_fraction=0.1, test_fraction=0.1, seed=7
    )

    manifest_path = augment_recovery_from_config(config_path)

    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_record = json.loads(
        (data_dir / "recovery_source_split.json").read_text(encoding="utf-8")
    )
    source_seeds = {int(record["source_seed"]) for record in records}
    assert source_seeds == set(expected.train)
    assert source_seeds.isdisjoint(expected.validation)
    assert source_seeds.isdisjoint(expected.test)
    assert set(split_record["train"]) == set(expected.train)
    assert set(split_record["validation"]) == set(expected.validation)
    assert set(split_record["test"]) == set(expected.test)


def test_episode_level_splits_are_deterministic_and_disjoint() -> None:
    first = split_episode_seeds(range(100), validation_fraction=0.2, test_fraction=0.2, seed=11)
    second = split_episode_seeds(range(100), validation_fraction=0.2, test_fraction=0.2, seed=11)

    assert first == second
    assert len(first.train) == 60
    assert len(first.validation) == 20
    assert len(first.test) == 20
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.validation).isdisjoint(first.test)


def test_manifest_is_the_authoritative_episode_list(tmp_path) -> None:
    selected = tmp_path / "episode_000000.npz"
    stale = tmp_path / "episode_999999.npz"
    selected.touch()
    stale.touch()
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"path": selected.name, "seed": 1}]), encoding="utf-8"
    )

    assert episode_paths_from_manifest(tmp_path) == (selected,)


def test_recovery_manifest_resolver_preserves_order_and_rejects_invalid_records(
    tmp_path,
) -> None:
    first = tmp_path / "recovery_first.npz"
    second = tmp_path / "recovery_second.npz"
    first.touch()
    second.touch()
    manifest = tmp_path / "recovery_manifest.json"
    manifest.write_text(
        json.dumps([{"path": second.name}, {"path": first.name}]), encoding="utf-8"
    )
    assert recovery_paths_from_manifest(tmp_path) == (second, first)

    manifest.write_text(json.dumps([{"path": "../escape.npz"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="filename"):
        recovery_paths_from_manifest(tmp_path)

    manifest.write_text(
        json.dumps([{"path": first.name}, {"path": first.name}]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        recovery_paths_from_manifest(tmp_path)

    manifest.write_text(json.dumps([{"path": "missing.npz"}]), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        recovery_paths_from_manifest(tmp_path)

    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        recovery_paths_from_manifest(tmp_path)


def test_episode_split_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        split_episode_seeds(
            (1, 1, 2), validation_fraction=0.2, test_fraction=0.2, seed=0
        )
