from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("lerobot")

from interaction_vla.graph_control.cache import CacheProvenance, write_token_cache
from interaction_vla.graph_control.dataset import GraphConditionedDataset
from interaction_vla.graph_control.schema import CONDITIONS, TOKEN_DIM
from interaction_vla.graph_control.training import (
    _control_source_files,
    assert_checkpoint_split,
    assert_paired_summaries,
    expected_graph_checkpoint_metadata,
    graph_checkpoint_bindings,
    load_control_split,
    load_graph_act_checkpoint,
    train_paired_seed,
)
from interaction_vla.lerobot_bridge.act_smoke import (
    _act_config,
    load_act_dataset,
    pilot_episode_split,
)


def _write_split(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "mujoco_semantic_graph_v1",
                "split_seed": 17,
                "episode_indices": {
                    "train": [0, 1],
                    "validation": [2],
                    "test": [3],
                },
                "row_indices": {
                    "train": [0, 1, 2],
                    "validation": [3],
                    "test": [4, 5],
                },
            }
        ),
        encoding="utf-8",
    )


def test_control_split_is_loaded_exactly_and_rejects_legacy_permutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.json"
    _write_split(path)
    split = load_control_split(path)

    assert split.episodes["train"] == (0, 1)
    assert split.rows["test"] == (4, 5)
    assert len(split.sha256) == 64

    legacy = pilot_episode_split(total_episodes=50, seed=0)
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(
        json.dumps(
            {
                "schema_version": "mujoco_semantic_graph_v1",
                "split_seed": 17,
                "episode_indices": legacy,
                "row_indices": {"train": [0], "validation": [1], "test": [2]},
            }
        ),
        encoding="utf-8",
    )
    pilot = load_control_split(pilot_path)
    with pytest.raises(ValueError, match="checkpoint.*split"):
        assert_checkpoint_split(
            {
                "split_seed": 17,
                "initialization": "reflectvlm_init",
                "fraction": 1.0,
                "seed": 0,
                "selected_train_episodes": [0, 1],
                "train_row_indices": [0, 1, 2],
                "validation_row_indices": [3],
                "test_row_indices": [4, 5],
            },
            pilot,
            condition="predicted_reflect",
            seed=0,
        )


def test_checkpoint_split_binds_seed_initialization_fraction_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    _write_split(path)
    split = load_control_split(path)
    payload = {
        "split_seed": 17,
        "initialization": "reflectvlm_init",
        "fraction": 1.0,
        "seed": 2,
        "selected_train_episodes": [0, 1],
        "train_row_indices": [0, 1, 2],
        "validation_row_indices": [3],
        "test_row_indices": [4, 5],
    }
    assert_checkpoint_split(payload, split, condition="oracle_current", seed=2)
    with pytest.raises(ValueError, match="seed"):
        assert_checkpoint_split(payload, split, condition="oracle_current", seed=1)
    with pytest.raises(ValueError, match="initialization"):
        assert_checkpoint_split(payload, split, condition="predicted_random", seed=2)


def _cache(
    tmp_path: Path, condition: str, rows: list[int], *, offset: float
):
    is_flat = condition == "flat"
    initialization = "random_init" if condition == "predicted_random" else "reflectvlm_init"
    provenance = CacheProvenance(
        condition=condition,
        dataset_fingerprint="d" * 64,
        split_manifest_sha256="a" * 64,
        graph_checkpoint_sha256=None if is_flat else "b" * 64,
        graph_initialization=None if is_flat else initialization,
        graph_fraction=None if is_flat else 1.0,
        graph_seed=None if is_flat else 7,
    )
    tokens = np.full((len(rows), TOKEN_DIM), offset, dtype=np.float32)
    return write_token_cache(tmp_path / f"{condition}.npz", rows, tokens, provenance)


def test_one_update_per_condition_is_paired_and_reloadable(
    tiny_lerobot_dataset, tmp_path: Path
) -> None:
    dataset_root, repo_id = tiny_lerobot_dataset
    base = load_act_dataset(dataset_root=dataset_root, repo_id=repo_id)
    rows = [int(base[index]["index"].item()) for index in range(len(base))]
    caches = {
        condition: _cache(tmp_path, condition, rows, offset=float(index) / 10.0)
        for index, condition in enumerate(CONDITIONS)
    }

    report = train_paired_seed(
        train_dataset=base,
        validation_dataset=base,
        caches=caches,
        seed=7,
        output_dir=tmp_path / "runs",
        dataset_root=dataset_root,
        device=torch.device("cpu"),
        architecture="test",
        batch_size=1,
        smoke_steps=1,
        initial_epochs=None,
        maximum_epochs=None,
    )

    assert report["conditions"] == list(CONDITIONS)
    summaries = report["summaries"]
    assert_paired_summaries(summaries)
    assert len({summary["initial_state_hash"] for summary in summaries.values()}) == 1
    assert len({summary["parameter_count"] for summary in summaries.values()}) == 1
    assert all(summary["steps"] == 1 for summary in summaries.values())
    assert all(summary["reload_max_abs_error"] <= 1e-5 for summary in summaries.values())
    assert len({tuple(summary["source_row_indices"]) for summary in summaries.values()}) == 1

    checkpoint = tmp_path / "runs" / "flat" / "checkpoint"
    bindings = graph_checkpoint_bindings("flat", 7, caches["flat"])
    expected = expected_graph_checkpoint_metadata(
        dataset_root=dataset_root,
        features=GraphConditionedDataset(base, caches["flat"]).features,
        act_config=_act_config(device=torch.device("cpu"), architecture="test"),
        device=torch.device("cpu"),
        bindings=bindings,
    )
    load_graph_act_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
        expected_metadata=expected,
    )

    metadata_path = checkpoint / "bridge_checkpoint.json"
    original = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutations = {
        "dataset_fingerprint": lambda value: "f" * 64,
        "features": lambda value: {**value, "observation.environment_state": {
            **value["observation.environment_state"], "names": ["wrong"] * TOKEN_DIM
        }},
        "state_codec_version": lambda value: "wrong-state-codec",
        "action_codec_version": lambda value: "wrong-action-codec",
        "lerobot_version": lambda value: "0.0.0-tampered",
        "act_config": lambda value: {**value, "optimizer_lr": 0.5},
        "device": lambda value: "mps",
        "source_fingerprint": lambda value: "e" * 64,
        "graph_control": lambda value: {**value, "cache_sha256": "d" * 64},
    }
    for field, mutate in mutations.items():
        altered = dict(original)
        altered[field] = mutate(original[field])
        metadata_path.write_text(json.dumps(altered), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            load_graph_act_checkpoint(
                checkpoint,
                device=torch.device("cpu"),
                expected_metadata=expected,
            )
    metadata_path.write_text(json.dumps(original), encoding="utf-8")


def test_control_source_fingerprint_covers_transitive_graph_inference_code() -> None:
    repository = Path(__file__).resolve().parents[3]
    relative = {
        path.relative_to(repository).as_posix()
        for path in _control_source_files(repository)
    }
    assert "interaction_vla/graph_finetune/data.py" in relative
    assert "interaction_vla/graph_finetune/pipeline.py" in relative
    assert "interaction_vla/graph_pretrain/reflectvlm.py" in relative
    assert "interaction_vla/env.py" in relative
    assert "interaction_vla/franka.py" in relative
    assert "interaction_vla/physics_action_safety.py" in relative
    assert "interaction_vla/physics_env.py" in relative
    assert "interaction_vla/physics_evaluate.py" in relative


def test_pairing_audit_rejects_different_initialization_or_rows() -> None:
    base = {
        condition: {
            "initial_state_hash": "same",
            "parameter_count": 123,
            "source_row_indices": [1, 2],
            "epoch_order_hashes": ["a" * 64],
            "epochs": 1,
            "extension_decisions": [],
        }
        for condition in CONDITIONS
    }
    assert_paired_summaries(base)
    bad = {name: dict(value) for name, value in base.items()}
    bad["predicted_reflect"] = None
    with pytest.raises(ValueError, match="mapping"):
        assert_paired_summaries(bad)

    bad = {name: dict(value) for name, value in base.items()}
    bad["predicted_reflect"]["source_row_indices"] = [2, 1]
    with pytest.raises(ValueError, match="source_row_indices"):
        assert_paired_summaries(bad)

    bad = {name: dict(value) for name, value in base.items()}
    bad["predicted_reflect"]["epoch_order_hashes"] = ["a" * 63 + "b"]
    with pytest.raises(ValueError, match="epoch_order_hashes"):
        assert_paired_summaries(bad)
