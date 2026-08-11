from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

import numpy as np


CONDITIONS: Final[tuple[str, ...]] = (
    "flat",
    "predicted_random",
    "predicted_reflect",
    "oracle_current",
)

_SLICE_WIDTHS = (
    ("entity_presence", 6),
    ("entity_visibility", 12),
    ("relation_presence", 8),
    ("gripper_target_semantics", 10),
    ("target_receptacle_semantics", 10),
    ("distractor_risks", 8),
    ("next_relation", 8),
    ("relation_operator", 5),
    ("predicate", 7),
    ("goal_residual", 1),
)


def _make_slices() -> Mapping[str, slice]:
    result: dict[str, slice] = {}
    cursor = 0
    for name, width in _SLICE_WIDTHS:
        result[name] = slice(cursor, cursor + width)
        cursor += width
    return MappingProxyType(result)


TOKEN_SLICES: Final[Mapping[str, slice]] = _make_slices()
TOKEN_DIM: Final[int] = TOKEN_SLICES["goal_residual"].stop
TOKEN_FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    f"{name}_{index}"
    for name, value in TOKEN_SLICES.items()
    for index in range(value.stop - value.start)
)


def validate_token(value: object) -> np.ndarray:
    token = np.asarray(value, dtype=np.float32)
    if token.shape != (TOKEN_DIM,) or not np.isfinite(token).all():
        raise ValueError(f"graph control token must be finite with shape [{TOKEN_DIM}]")
    return token


def empty_token() -> np.ndarray:
    return np.zeros(TOKEN_DIM, dtype=np.float32)
