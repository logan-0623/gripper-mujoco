from __future__ import annotations

from typing import Final

import numpy as np

from interaction_vla.graph_finetune.schema import (
    GRAPH_SCHEMA_VERSION,
    TOKEN_DIM,
    TOKEN_FEATURE_NAMES,
    TOKEN_SCHEMA_VERSION,
    TOKEN_SLICES,
)


ORACLE_CONDITIONS: Final[tuple[str, ...]] = ("flat", "oracle_graph_v2")
ALL_CONDITIONS: Final[tuple[str, ...]] = (
    "flat",
    "oracle_graph_v2",
    "predicted_random_v2",
    "predicted_reflect_v2",
)
ABLATION_CONDITIONS: Final[tuple[str, ...]] = (
    "flat",
    "entity_geometry",
    "interaction_state",
    "full_graph",
    "shuffled_graph",
)
CONTROL_CONDITIONS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*ALL_CONDITIONS, *ABLATION_CONDITIONS))
)
CONDITIONS: Final[tuple[str, ...]] = ALL_CONDITIONS


def validate_token(value: object) -> np.ndarray:
    token = np.asarray(value, dtype=np.float32)
    if token.shape != (TOKEN_DIM,) or not np.isfinite(token).all():
        raise ValueError(f"Graph v2 token must be finite with shape [{TOKEN_DIM}]")
    return token.copy()


def empty_token() -> np.ndarray:
    return np.zeros(TOKEN_DIM, dtype=np.float32)
