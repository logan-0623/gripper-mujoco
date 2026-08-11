from __future__ import annotations

import numpy as np
import pytest
import torch

from interaction_vla.graph_finetune.data import (
    MODEL_BATCH_KEYS,
    MuJoCoGraphDataset,
    prepare_corpus,
    resize_rgb,
    select_training_fraction,
    semantic_targets,
    split_episode_indices,
)
from interaction_vla.graph_pretrain.reflectvlm import Vocabulary
from interaction_vla.graph_finetune.schema import SEMANTIC_CHANNELS


def teacher_arrays(frames: int = 3) -> dict[str, np.ndarray]:
    relation_values = np.arange(frames * 8 * 24, dtype=np.float32).reshape(
        frames, 8, 24
    )
    relation_values[:, :, 16:20] = 0.5
    relation_values[:, :, 20:22] = 0.25
    relation_values[:, :, 22:24] = 1.0
    relation_goal = np.zeros((frames, 5), dtype=np.float32)
    relation_goal[:, 0] = np.arange(frames) % 8
    relation_goal[:, 1] = np.arange(frames) % 5
    relation_goal[:, 2] = np.arange(frames) % 7
    relation_goal[:, 3] = np.linspace(-0.3, 0.0, frames, dtype=np.float32)
    relation_goal[:, 4] = 1.0
    return {
        "annotation.tc_tig.entity_mask": np.ones((frames, 6), dtype=np.bool_),
        "annotation.tc_tig.entity_visibility": np.full(
            (frames, 6, 2), 0.25, dtype=np.float32
        ),
        "annotation.tc_tig.relation_mask": np.ones((frames, 8), dtype=np.bool_),
        "annotation.tc_tig.relation_values": relation_values,
        "annotation.tc_tig.relation_goal": relation_goal,
    }


def manifest_records(count: int = 5) -> list[dict[str, object]]:
    return [
        {
            "episode_index": episode,
            "seed": 1000 + episode,
            "frames": 3,
            "schema_version": "tc_tig_teacher_v1",
            "task": "Pick up the target and place it in the receptacle.",
        }
        for episode in range(count)
    ]


def test_semantic_target_selects_only_coordinate_invariant_channels() -> None:
    arrays = teacher_arrays()

    target = semantic_targets(arrays)

    assert SEMANTIC_CHANNELS == tuple(range(12, 22))
    np.testing.assert_array_equal(
        target.relation_semantics,
        arrays["annotation.tc_tig.relation_values"][:, :, 12:22],
    )
    assert target.relation_semantics.shape == (3, 8, 10)
    assert not hasattr(target, "entity_pose")
    assert not hasattr(target, "depth_agent")
    assert not hasattr(target, "action")


def test_public_rgb_resize_matches_graph_training_contract() -> None:
    image = torch.linspace(0.0, 1.0, 3 * 20 * 24).reshape(3, 20, 24)

    resized = resize_rgb(image, 16, "agent RGB")

    assert resized.shape == (3, 16, 16)
    assert resized.dtype == torch.float32
    assert torch.isfinite(resized).all()

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        resize_rgb(image * 2.0, 16, "agent RGB")


def test_semantic_target_extracts_goal_categories_and_residual() -> None:
    target = semantic_targets(teacher_arrays())

    assert target.goal_relation.tolist() == [0, 1, 2]
    assert target.goal_operator.tolist() == [0, 1, 2]
    assert target.goal_predicate.tolist() == [0, 1, 2]
    np.testing.assert_allclose(target.goal_residual, [-0.3, -0.15, 0.0])


def test_semantic_target_rejects_out_of_range_goal_id() -> None:
    arrays = teacher_arrays()
    arrays["annotation.tc_tig.relation_goal"][0, 0] = 8

    with pytest.raises(ValueError, match="goal_relation"):
        semantic_targets(arrays)


def test_episode_split_is_deterministic_and_never_splits_frames() -> None:
    records = manifest_records()

    first = split_episode_indices(records, seed=7, ratios=(0.6, 0.2, 0.2))
    second = split_episode_indices(records, seed=7, ratios=(0.6, 0.2, 0.2))

    assert first == second
    assert {name: len(value) for name, value in first.items()} == {
        "train": 3,
        "validation": 1,
        "test": 1,
    }
    assert sorted(first["train"] + first["validation"] + first["test"]) == list(
        range(5)
    )
    assert set(first["train"]).isdisjoint(first["validation"] + first["test"])


def test_episode_split_rejects_duplicate_episode_indices() -> None:
    records = manifest_records()
    records[-1]["episode_index"] = 0

    with pytest.raises(ValueError, match="unique"):
        split_episode_indices(records, seed=0, ratios=(0.6, 0.2, 0.2))


def test_training_fraction_selects_whole_episodes_deterministically() -> None:
    episodes = list(range(10))

    first = select_training_fraction(episodes, fraction=0.25, seed=3)
    second = select_training_fraction(episodes, fraction=0.25, seed=3)

    assert first == second
    assert len(first) == 3
    assert set(first).issubset(episodes)


class SyntheticSource:
    def __init__(
        self,
        records: list[dict[str, object]],
        *,
        extreme_episode: int | None = None,
    ) -> None:
        self.samples: list[dict[str, object]] = []
        for record in records:
            episode = int(record["episode_index"])
            for frame in range(int(record["frames"])):
                extreme = episode == extreme_episode
                task = (
                    "heldoutword place the target"
                    if extreme
                    else "pick up the target and place it in the receptacle"
                )
                self.samples.append(
                    {
                        "observation.images.agent": torch.full(
                            (3, 20, 24), episode / 10.0
                        ),
                        "observation.images.wrist": torch.full(
                            (3, 20, 24), frame / 10.0
                        ),
                        "observation.state": torch.full(
                            (10,), 1000.0 if extreme else episode + frame / 10.0
                        ),
                        "action": torch.ones(7),
                        "timestamp": torch.tensor(frame / 20.0),
                        "frame_index": torch.tensor(frame),
                        "episode_index": torch.tensor(episode),
                        "index": torch.tensor(len(self.samples)),
                        "task_index": torch.tensor(1 if extreme else 0),
                        "task": task,
                    }
                )
        self.hf_dataset = {
            key: [
                sample[key].item()
                if isinstance(sample[key], torch.Tensor) and sample[key].ndim == 0
                else sample[key].tolist()
                if isinstance(sample[key], torch.Tensor)
                else sample[key]
                for sample in self.samples
            ]
            for key in (
                "observation.state",
                "frame_index",
                "episode_index",
                "task_index",
            )
        }
        self.tasks = [
            "pick up the target and place it in the receptacle",
            "heldoutword place the target",
        ]
        self.image_accesses = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        self.image_accesses += 1
        return self.samples[index]


def sidecars(records: list[dict[str, object]]) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for record in records:
        episode = int(record["episode_index"])
        arrays = teacher_arrays(int(record["frames"]))
        arrays["annotation.tc_tig.relation_goal"][:, 0] = (
            episode + np.arange(int(record["frames"]))
        ) % 8
        result[episode] = arrays
    return result


def test_prepare_corpus_does_not_decode_images() -> None:
    records = manifest_records()
    source = SyntheticSource(records)

    corpus = prepare_corpus(
        source,
        records,
        sidecars(records),
        split_seed=3,
        split_ratios=(0.6, 0.2, 0.2),
        pretrained_vocabulary=Vocabulary(("<pad>", "<unk>", "pick")),
    )

    assert source.image_accesses == 0
    assert sum(len(values) for values in corpus.splits.values()) == 5


def test_prepare_corpus_accepts_huggingface_dataset_columns() -> None:
    records = manifest_records()
    source = SyntheticSource(records)

    class DatasetColumns:
        def __init__(self, columns):
            self.columns = columns
            self.column_names = list(columns)

        def __len__(self):
            return len(next(iter(self.columns.values())))

        def __getitem__(self, name):
            return self.columns[name]

        def __iter__(self):
            for index in range(len(self)):
                yield {name: values[index] for name, values in self.columns.items()}

    source.hf_dataset = DatasetColumns(source.hf_dataset)

    corpus = prepare_corpus(
        source,
        records,
        sidecars(records),
        split_seed=3,
        split_ratios=(0.6, 0.2, 0.2),
    )

    assert len(corpus.states) == len(source)


def test_dataset_aligns_teacher_by_episode_and_frame_and_drops_action() -> None:
    records = manifest_records()
    source = SyntheticSource(records)
    corpus = prepare_corpus(
        source,
        records,
        sidecars(records),
        split_seed=3,
        split_ratios=(0.6, 0.2, 0.2),
        pretrained_vocabulary=Vocabulary(("<pad>", "<unk>", "pick")),
    )
    training = corpus.for_training_fraction(1.0, seed=9)
    dataset = MuJoCoGraphDataset(
        training,
        partition="train",
        image_size=32,
        max_language_tokens=16,
    )

    sample = dataset[2]
    first_episode = training.selected_train_episodes[0]

    assert set(sample) == MODEL_BATCH_KEYS
    assert "action" not in sample
    assert sample["agent_rgb"].shape == (3, 32, 32)
    assert sample["wrist_rgb"].shape == (3, 32, 32)
    assert sample["state"].shape == (10,)
    assert sample["language_tokens"].shape == (16,)
    assert sample["goal_relation"].item() == (first_episode + 2) % 8


def test_statistics_and_vocabulary_use_selected_training_episodes_only() -> None:
    records = manifest_records()
    split = split_episode_indices(records, seed=3, ratios=(0.6, 0.2, 0.2))
    held_out = split["test"][0]
    source = SyntheticSource(records, extreme_episode=held_out)
    corpus = prepare_corpus(
        source,
        records,
        sidecars(records),
        split_seed=3,
        split_ratios=(0.6, 0.2, 0.2),
        pretrained_vocabulary=Vocabulary(("<pad>", "<unk>", "pick")),
    )

    prepared = corpus.for_training_fraction(1.0, seed=9)

    assert np.max(np.abs(prepared.normalization.state_mean)) < 10.0
    assert "heldoutword" not in prepared.vocabulary.token_to_id
    assert prepared.normalization.relation_mean.shape == (8, 10)
    assert prepared.normalization.relation_std.shape == (8, 10)
    assert np.all(prepared.normalization.relation_std > 0.0)


def test_dataset_rejects_forbidden_teacher_input_key() -> None:
    records = manifest_records()
    source = SyntheticSource(records)
    corpus = prepare_corpus(
        source,
        records,
        sidecars(records),
        split_seed=3,
        split_ratios=(0.6, 0.2, 0.2),
        pretrained_vocabulary=Vocabulary(("<pad>", "<unk>")),
    )
    training = corpus.for_training_fraction(1.0, seed=9)
    first_episode = training.selected_train_episodes[0]
    first_row = corpus.row_indices[first_episode][0]
    source.samples[first_row]["annotation.tc_tig.depth_agent"] = torch.ones(20, 24)
    dataset = MuJoCoGraphDataset(
        training,
        partition="train",
        image_size=32,
        max_language_tokens=16,
    )

    with pytest.raises(ValueError, match="forbidden"):
        dataset[0]
