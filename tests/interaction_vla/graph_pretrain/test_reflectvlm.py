from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
import torch

from interaction_vla.graph_pretrain.reflectvlm import (
    ReflectTorchDataset,
    Vocabulary,
    grouped_split_indices,
    parse_reflect_metadata,
    prepare_corpus,
)
from interaction_vla.graph_pretrain.schema import (
    PHASE_IDS,
    STATE_IDS,
    UPRIGHT_IDS,
)
from interaction_vla.lerobot_bridge.teacher_schema import OPERATOR_IDS, PREDICATE_IDS


def reflect_row(
    *,
    board_id: int = 0,
    env_seed: int = 0,
    action: str = "pick up yellow",
    object_in_hand: str | None = None,
    dependencies: str = "{(2, 3), (3, 4), (4, 5)}",
) -> dict[str, object]:
    return {
        "board_id": board_id,
        "env_seed": env_seed,
        "trajectory_id": board_id * 10,
        "step_id": 0,
        "history": "['pick up green', 'insert green']",
        "oracle_action": action,
        "object_states": str(
            {
                "green block": "DONE",
                "orange block": "READY",
                "purple block": "BLOCKED (by predecessor)",
                "yellow nail": "BAD (is down)",
            }
        ),
        "object_in_hand": object_in_hand,
        "object_is_upright": "{2: True, 3: True, 4: False, 5: False}",
        "object_descriptions": str(
            {
                "brick_1": "gray board",
                "brick_2": "green block",
                "brick_3": "orange block",
                "brick_4": "purple block",
                "brick_5": "yellow nail",
            }
        ),
        "object_dependencies": dependencies,
    }


def test_parse_reflect_metadata_builds_task_conditioned_graph_target() -> None:
    example = parse_reflect_metadata(reflect_row())

    assert example.group == (0, 0)
    assert example.history == ("pick up green", "insert green")
    assert example.target.target_index == 3
    assert example.target.in_hand_index == 6
    assert example.target.state_ids.tolist() == [
        STATE_IDS["done"],
        STATE_IDS["ready"],
        STATE_IDS["blocked"],
        STATE_IDS["bad"],
        STATE_IDS["unknown"],
        STATE_IDS["unknown"],
    ]
    assert example.target.upright_ids.tolist() == [
        UPRIGHT_IDS["true"],
        UPRIGHT_IDS["true"],
        UPRIGHT_IDS["false"],
        UPRIGHT_IDS["false"],
        UPRIGHT_IDS["unknown"],
        UPRIGHT_IDS["unknown"],
    ]
    assert example.target.object_mask.tolist() == [True, True, True, True, False, False]
    expected_dependencies = np.zeros((6, 6), dtype=np.float32)
    expected_dependencies[0, 1] = 1.0
    expected_dependencies[1, 2] = 1.0
    expected_dependencies[2, 3] = 1.0
    np.testing.assert_array_equal(example.target.dependency, expected_dependencies)
    assert example.target.phase_id == PHASE_IDS["pick"]
    assert example.target.goal_operator_id == OPERATOR_IDS["establish"]
    assert example.target.goal_predicate_id == PREDICATE_IDS["co_motion"]


def test_parse_reflect_metadata_accepts_empty_dependency_set() -> None:
    example = parse_reflect_metadata(
        reflect_row(action="insert orange", object_in_hand="brick_3", dependencies="set()")
    )

    assert example.target.target_index == 1
    assert example.target.in_hand_index == 1
    assert not example.target.dependency.any()
    assert example.target.phase_id == PHASE_IDS["insert"]
    assert example.target.goal_predicate_id == PREDICATE_IDS["containment"]


def test_parse_reflect_metadata_masks_variable_object_count_up_to_six() -> None:
    value = reflect_row(action="pick up blue")
    value["object_descriptions"] = str(
        {
            "brick_1": "yellow board",
            "brick_2": "orange block",
            "brick_3": "red block",
            "brick_4": "green block",
            "brick_5": "pink block",
            "brick_6": "blue block",
            "brick_7": "gray block",
        }
    )
    value["object_states"] = str(
        {
            "orange block": "DONE",
            "red block": "READY",
            "green block": "BLOCKED (by predecessor)",
            "pink block": "BAD (is down)",
            "blue block": "READY",
            "gray block": "DONE",
        }
    )
    value["object_is_upright"] = "{2: True, 3: True, 4: True, 5: False, 6: True, 7: True}"
    value["object_dependencies"] = "{(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)}"

    example = parse_reflect_metadata(value)

    assert example.target.target_index == 4
    assert example.target.object_mask.tolist() == [True] * 6
    assert example.target.state_ids.shape == (6,)
    assert example.target.dependency.shape == (6, 6)


def test_parse_reflect_metadata_requires_two_task_objects() -> None:
    value = reflect_row(action="pick up green")
    value["object_descriptions"] = str(
        {"brick_1": "gray board", "brick_2": "green block"}
    )
    value["object_states"] = str({"green block": "READY"})
    value["object_is_upright"] = "{2: True}"
    value["object_dependencies"] = "set()"

    with pytest.raises(ValueError, match="between two and six"):
        parse_reflect_metadata(value)


def test_parse_reflect_metadata_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="unsupported oracle action"):
        parse_reflect_metadata(reflect_row(action="throw yellow"))


def test_grouped_split_is_deterministic_and_has_no_group_overlap() -> None:
    rows = [
        reflect_row(board_id=group, env_seed=100 + group)
        for group in range(12)
        for _ in range(2)
    ]

    first = grouped_split_indices(rows, seed=7, ratios=(0.5, 0.25, 0.25))
    second = grouped_split_indices(rows, seed=7, ratios=(0.5, 0.25, 0.25))

    assert first == second
    assert set(first) == {"train", "validation", "test"}
    assert sorted(first["train"] + first["validation"] + first["test"]) == list(
        range(len(rows))
    )
    groups_by_partition = {
        name: {(rows[index]["board_id"], rows[index]["env_seed"]) for index in indices}
        for name, indices in first.items()
    }
    assert groups_by_partition["train"].isdisjoint(groups_by_partition["validation"])
    assert groups_by_partition["train"].isdisjoint(groups_by_partition["test"])
    assert groups_by_partition["validation"].isdisjoint(groups_by_partition["test"])


def test_grouped_split_requires_at_least_three_groups() -> None:
    rows = [reflect_row(board_id=group, env_seed=group) for group in range(2)]

    with pytest.raises(ValueError, match="at least three groups"):
        grouped_split_indices(rows, seed=0)


class CountingSource:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.image_accesses = 0
        self.column_names = tuple(rows[0])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: int | str):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        self.image_accesses += 1
        return self.rows[key]


def source_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in range(9):
        row = reflect_row(board_id=group, env_seed=100 + group)
        row["history"] = f"['groupword{group}', 'pick up green']"
        row["image"] = Image.new("RGB", (20, 12), color=(group, 20, 40))
        rows.append(row)
    return rows


def test_vocabulary_reserves_pad_and_unknown_and_truncates() -> None:
    vocabulary = Vocabulary.build([("pick up green",), ("insert orange",)])

    token_ids, token_mask = vocabulary.encode(("unseen token", "pick up green"), 3)

    assert vocabulary.tokens[:2] == ("<pad>", "<unk>")
    assert token_ids.tolist() == [1, 1, vocabulary.token_to_id["pick"]]
    assert token_mask.tolist() == [True, True, True]


def test_prepare_corpus_uses_training_history_only_and_does_not_decode_images() -> None:
    source = CountingSource(source_rows())

    corpus = prepare_corpus(source, split_seed=5)

    assert source.image_accesses == 0
    training_words = {
        f"groupword{corpus.metadata[index].group[0]}"
        for index in corpus.splits["train"]
    }
    heldout_words = {
        f"groupword{corpus.metadata[index].group[0]}"
        for partition in ("validation", "test")
        for index in corpus.splits[partition]
    }
    assert training_words <= set(corpus.vocabulary.tokens)
    assert heldout_words.isdisjoint(corpus.vocabulary.tokens)


def test_torch_dataset_lazily_returns_normalized_image_and_graph_tensors() -> None:
    source = CountingSource(source_rows())
    corpus = prepare_corpus(source, split_seed=5)
    dataset = ReflectTorchDataset(
        corpus,
        partition="train",
        image_size=16,
        max_history_tokens=8,
    )

    sample = dataset[0]

    assert source.image_accesses == 1
    assert sample["image"].shape == (3, 16, 16)
    assert sample["image"].dtype == torch.float32
    assert torch.isfinite(sample["image"]).all()
    assert 0.0 <= float(sample["image"].min()) <= float(sample["image"].max()) <= 1.0
    assert sample["history_tokens"].shape == (8,)
    assert sample["history_mask"].dtype == torch.bool
    assert sample["object_mask"].tolist() == [True, True, True, True, False, False]
    assert sample["state_ids"].shape == (6,)
    assert sample["upright_ids"].shape == (6,)
    assert sample["dependency"].shape == (6, 6)
    assert sample["dependency_mask"].sum().item() == 12
