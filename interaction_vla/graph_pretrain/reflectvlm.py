from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from interaction_vla.lerobot_bridge.teacher_schema import OPERATOR_IDS, PREDICATE_IDS

from .schema import (
    NONE_OBJECT_INDEX,
    OBJECT_COUNT,
    PHASE_IDS,
    STATE_IDS,
    UPRIGHT_IDS,
    ReflectGraphTarget,
)


ACTION_GOALS = {
    "pick up": ("pick", "establish", "co_motion"),
    "put down": ("place", "break", "co_motion"),
    "reorient": ("reorient", "increase", "alignment"),
    "insert": ("insert", "establish", "containment"),
}
_BRICK_PATTERN = re.compile(r"^brick_(\d+)$")
_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
METADATA_COLUMNS = (
    "board_id",
    "env_seed",
    "trajectory_id",
    "step_id",
    "history",
    "oracle_action",
    "object_states",
    "object_in_hand",
    "object_is_upright",
    "object_descriptions",
    "object_dependencies",
)


@dataclass(frozen=True)
class ReflectMetadata:
    group: tuple[int, int]
    history: tuple[str, ...]
    target: ReflectGraphTarget


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) < 2 or self.tokens[:2] != ("<pad>", "<unk>"):
            raise ValueError("vocabulary must begin with <pad> and <unk>")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary tokens must be unique")

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def build(cls, histories: Iterable[Sequence[str]]) -> "Vocabulary":
        words = {
            token
            for history in histories
            for action in history
            for token in _TOKEN_PATTERN.findall(str(action).lower())
        }
        return cls(("<pad>", "<unk>", *sorted(words)))

    def encode(
        self, history: Sequence[str], max_tokens: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        mapping = self.token_to_id
        words = [
            token
            for action in history
            for token in _TOKEN_PATTERN.findall(str(action).lower())
        ][:max_tokens]
        token_ids = np.zeros(max_tokens, dtype=np.int64)
        token_mask = np.zeros(max_tokens, dtype=np.bool_)
        if words:
            token_ids[: len(words)] = [mapping.get(word, 1) for word in words]
            token_mask[: len(words)] = True
        return token_ids, token_mask


@dataclass(frozen=True)
class PreparedCorpus:
    source: Any
    metadata: tuple[ReflectMetadata, ...]
    splits: dict[str, list[int]]
    vocabulary: Vocabulary


def _literal(value: Any, name: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text in {"", "null", "None"}:
        return None
    if text == "set()":
        return set()
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"{name} is not a valid literal") from error


def _brick_number(value: Any, name: str) -> int:
    match = _BRICK_PATTERN.fullmatch(str(value))
    if match is None:
        raise ValueError(f"{name} contains an invalid brick identifier: {value}")
    return int(match.group(1))


def _object_descriptions(row: Mapping[str, Any]) -> tuple[dict[int, str], list[int]]:
    raw = _literal(row.get("object_descriptions"), "object_descriptions")
    if not isinstance(raw, Mapping):
        raise ValueError("object_descriptions must be a mapping")
    descriptions: dict[int, str] = {}
    for key, value in raw.items():
        brick = _brick_number(key, "object_descriptions")
        description = str(value).strip().lower()
        if not description or brick in descriptions:
            raise ValueError("object_descriptions contains an empty or duplicate object")
        descriptions[brick] = description
    object_bricks = sorted(brick for brick in descriptions if brick != 1)
    if not 2 <= len(object_bricks) <= OBJECT_COUNT:
        raise ValueError("ReflectVLM rows must contain between two and six task objects")
    return descriptions, object_bricks


def _parse_action(action: Any, descriptions: Mapping[int, str]) -> tuple[str, int]:
    text = str(action).strip().lower()
    for verb in sorted(ACTION_GOALS, key=len, reverse=True):
        prefix = verb + " "
        if not text.startswith(prefix):
            continue
        object_phrase = text[len(prefix) :].strip()
        matches = [
            brick
            for brick, description in descriptions.items()
            if brick != 1
            and (
                object_phrase == description
                or object_phrase in description.split()
                or description.startswith(object_phrase + " ")
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"oracle action object must match exactly one task object: {text}"
            )
        return verb, matches[0]
    raise ValueError(f"unsupported oracle action: {text}")


def _state_id(value: Any) -> int:
    text = str(value).strip().lower()
    for name in ("ready", "blocked", "bad", "done"):
        if text == name or text.startswith(name + " ") or text.startswith(name + "("):
            return STATE_IDS[name]
    return STATE_IDS["unknown"]


def _history(value: Any) -> tuple[str, ...]:
    parsed = _literal(value, "history")
    if parsed is None:
        return ()
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("history must be a list or tuple")
    result = tuple(str(item).strip().lower() for item in parsed)
    if any(not item for item in result):
        raise ValueError("history contains an empty action")
    return result


def _in_hand_index(value: Any, object_bricks: Sequence[int]) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NONE_OBJECT_INDEX
    parsed = str(value).strip() if isinstance(value, str) else value
    if parsed in {"", "null", "None"}:
        return NONE_OBJECT_INDEX
    brick = _brick_number(parsed, "object_in_hand")
    try:
        return object_bricks.index(brick)
    except ValueError as error:
        raise ValueError("object_in_hand is not one of the task objects") from error


def parse_reflect_metadata(row: Mapping[str, Any]) -> ReflectMetadata:
    descriptions, object_bricks = _object_descriptions(row)
    verb, target_brick = _parse_action(row.get("oracle_action"), descriptions)

    raw_states = _literal(row.get("object_states"), "object_states")
    if not isinstance(raw_states, Mapping):
        raise ValueError("object_states must be a mapping")
    states_by_name = {str(key).strip().lower(): value for key, value in raw_states.items()}
    state_ids = np.full(OBJECT_COUNT, STATE_IDS["unknown"], dtype=np.int64)
    state_ids[: len(object_bricks)] = [
        _state_id(states_by_name.get(descriptions[brick], "unknown"))
        for brick in object_bricks
    ]

    raw_upright = _literal(row.get("object_is_upright"), "object_is_upright")
    if not isinstance(raw_upright, Mapping):
        raise ValueError("object_is_upright must be a mapping")
    upright_by_brick = {int(key): value for key, value in raw_upright.items()}
    upright_ids = np.full(OBJECT_COUNT, UPRIGHT_IDS["unknown"], dtype=np.int64)
    upright_ids[: len(object_bricks)] = [
        UPRIGHT_IDS["unknown"]
        if brick not in upright_by_brick
        else UPRIGHT_IDS["true" if bool(upright_by_brick[brick]) else "false"]
        for brick in object_bricks
    ]

    raw_dependencies = _literal(
        row.get("object_dependencies"), "object_dependencies"
    )
    if raw_dependencies is None:
        raw_dependencies = set()
    if not isinstance(raw_dependencies, (set, list, tuple)):
        raise ValueError("object_dependencies must contain pairs")
    dependency = np.zeros((OBJECT_COUNT, OBJECT_COUNT), dtype=np.float32)
    for pair in raw_dependencies:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("object_dependencies must contain two-item pairs")
        source, destination = (int(pair[0]), int(pair[1]))
        if (
            source == destination
            or source not in object_bricks
            or destination not in object_bricks
        ):
            raise ValueError("object_dependencies references an invalid task object")
        dependency[object_bricks.index(source), object_bricks.index(destination)] = 1.0

    phase, operator, predicate = ACTION_GOALS[verb]
    target = ReflectGraphTarget(
        target_index=object_bricks.index(target_brick),
        in_hand_index=_in_hand_index(row.get("object_in_hand"), object_bricks),
        object_mask=np.arange(OBJECT_COUNT) < len(object_bricks),
        state_ids=state_ids,
        upright_ids=upright_ids,
        dependency=dependency,
        phase_id=PHASE_IDS[phase],
        goal_operator_id=OPERATOR_IDS[operator],
        goal_predicate_id=PREDICATE_IDS[predicate],
    )
    return ReflectMetadata(
        group=(int(row["board_id"]), int(row["env_seed"])),
        history=_history(row.get("history")),
        target=target,
    )


def _group_from_row(row: Any) -> tuple[int, int]:
    if isinstance(row, ReflectMetadata):
        return row.group
    if isinstance(row, Mapping):
        return int(row["board_id"]), int(row["env_seed"])
    raise TypeError("rows must be mappings or ReflectMetadata values")


def grouped_split_indices(
    rows: Sequence[Any],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[int]]:
    values = np.asarray(ratios, dtype=np.float64)
    if values.shape != (3,) or np.any(values <= 0.0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("split ratios must be three positive values summing to one")
    group_to_indices: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        group_to_indices.setdefault(_group_from_row(row), []).append(index)
    if len(group_to_indices) < 3:
        raise ValueError("grouped split requires at least three groups")

    def digest(group: tuple[int, int]) -> bytes:
        return hashlib.sha256(f"{int(seed)}:{group[0]}:{group[1]}".encode()).digest()

    ordered_groups = sorted(group_to_indices, key=lambda group: (digest(group), group))
    group_count = len(ordered_groups)
    validation_count = max(1, int(np.floor(group_count * values[1])))
    test_count = max(1, int(np.floor(group_count * values[2])))
    while validation_count + test_count >= group_count:
        if validation_count >= test_count and validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("split ratios leave no training groups")
    test_groups = set(ordered_groups[:test_count])
    validation_groups = set(
        ordered_groups[test_count : test_count + validation_count]
    )
    train_groups = set(ordered_groups) - test_groups - validation_groups
    groups = {
        "train": train_groups,
        "validation": validation_groups,
        "test": test_groups,
    }
    return {
        name: [index for index, row in enumerate(rows) if _group_from_row(row) in selected]
        for name, selected in groups.items()
    }


def _metadata_rows(source: Any) -> list[dict[str, Any]]:
    missing = [
        name
        for name in METADATA_COLUMNS
        if hasattr(source, "column_names") and name not in source.column_names
    ]
    if missing:
        raise ValueError("ReflectVLM source is missing columns: " + ", ".join(missing))
    try:
        columns = {name: source[name] for name in METADATA_COLUMNS}
    except (KeyError, TypeError):
        rows = [source[index] for index in range(len(source))]
        return [{name: row.get(name) for name in METADATA_COLUMNS} for row in rows]
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("ReflectVLM metadata columns have inconsistent lengths")
    row_count = lengths.pop()
    return [
        {name: columns[name][index] for name in METADATA_COLUMNS}
        for index in range(row_count)
    ]


def _limit_splits(
    splits: dict[str, list[int]], max_rows: int | None
) -> dict[str, list[int]]:
    if max_rows is None or max_rows >= sum(len(values) for values in splits.values()):
        return {name: list(values) for name, values in splits.items()}
    if max_rows < 3:
        raise ValueError("max_rows must leave at least one row per partition")
    names = ("train", "validation", "test")
    total = sum(len(splits[name]) for name in names)
    quotas = {
        name: max(1, int(np.floor(max_rows * len(splits[name]) / total)))
        for name in names
    }
    while sum(quotas.values()) > max_rows:
        candidate = max(names, key=lambda name: (quotas[name], len(splits[name])))
        if quotas[candidate] <= 1:
            raise ValueError("max_rows cannot be distributed across partitions")
        quotas[candidate] -= 1
    while sum(quotas.values()) < max_rows:
        candidates = [name for name in names if quotas[name] < len(splits[name])]
        if not candidates:
            break
        candidate = max(
            candidates,
            key=lambda name: (len(splits[name]) - quotas[name], len(splits[name])),
        )
        quotas[candidate] += 1
    return {name: list(splits[name][: quotas[name]]) for name in names}


def prepare_corpus(
    source: Any,
    *,
    split_seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    max_rows: int | None = None,
) -> PreparedCorpus:
    rows = _metadata_rows(source)
    metadata = tuple(parse_reflect_metadata(row) for row in rows)
    splits = _limit_splits(
        grouped_split_indices(metadata, seed=split_seed, ratios=ratios), max_rows
    )
    vocabulary = Vocabulary.build(metadata[index].history for index in splits["train"])
    return PreparedCorpus(
        source=source,
        metadata=metadata,
        splits=splits,
        vocabulary=vocabulary,
    )


def _pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError("image array must have RGB or RGBA channels")
        return Image.fromarray(array.astype(np.uint8, copy=False)).convert("RGB")
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(Path(value["path"])).convert("RGB")
    raise ValueError("image must be a PIL image, NumPy array, or bytes/path mapping")


class ReflectTorchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        corpus: PreparedCorpus,
        *,
        partition: str,
        image_size: int,
        max_history_tokens: int,
    ) -> None:
        if partition not in corpus.splits:
            raise ValueError(f"unknown partition: {partition}")
        if image_size < 8 or max_history_tokens < 1:
            raise ValueError("image_size and max_history_tokens must be positive")
        self.corpus = corpus
        self.partition = partition
        self.indices = tuple(corpus.splits[partition])
        self.image_size = int(image_size)
        self.max_history_tokens = int(max_history_tokens)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        source_index = self.indices[item]
        row = self.corpus.source[source_index]
        image = _pil_image(row["image"]).resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        metadata = self.corpus.metadata[source_index]
        token_ids, token_mask = self.corpus.vocabulary.encode(
            metadata.history, self.max_history_tokens
        )
        target = metadata.target
        object_mask = target.object_mask
        dependency_mask = (
            object_mask[:, None]
            & object_mask[None, :]
            & ~np.eye(OBJECT_COUNT, dtype=np.bool_)
        )
        return {
            "image": image_tensor,
            "history_tokens": torch.from_numpy(token_ids),
            "history_mask": torch.from_numpy(token_mask),
            "target_index": torch.tensor(target.target_index, dtype=torch.long),
            "in_hand_index": torch.tensor(target.in_hand_index, dtype=torch.long),
            "object_mask": torch.from_numpy(object_mask.copy()),
            "state_ids": torch.from_numpy(target.state_ids.copy()),
            "upright_ids": torch.from_numpy(target.upright_ids.copy()),
            "dependency": torch.from_numpy(target.dependency.copy()),
            "dependency_mask": torch.from_numpy(dependency_mask),
            "phase_id": torch.tensor(target.phase_id, dtype=torch.long),
            "goal_operator_id": torch.tensor(
                target.goal_operator_id, dtype=torch.long
            ),
            "goal_predicate_id": torch.tensor(
                target.goal_predicate_id, dtype=torch.long
            ),
        }
