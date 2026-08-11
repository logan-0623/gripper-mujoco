from __future__ import annotations

import numpy as np
import pytest

from interaction_vla.graph_control.schema import (
    CONDITIONS,
    TOKEN_DIM,
    TOKEN_SLICES,
    empty_token,
    validate_token,
)


def test_token_layout_is_contiguous_and_exactly_75d() -> None:
    assert CONDITIONS == (
        "flat",
        "predicted_random",
        "predicted_reflect",
        "oracle_current",
    )
    cursor = 0
    for value in TOKEN_SLICES.values():
        assert value.start == cursor
        assert value.stop > value.start
        cursor = value.stop
    assert cursor == TOKEN_DIM == 75


def test_empty_token_is_valid_finite_float32() -> None:
    token = empty_token()
    assert token.dtype == np.float32
    assert np.array_equal(token, np.zeros(75, dtype=np.float32))
    assert validate_token(token) is token


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(74, dtype=np.float32),
        np.zeros((1, 75), dtype=np.float32),
        np.full(75, np.nan, dtype=np.float32),
    ],
)
def test_validate_token_rejects_wrong_shape_or_nonfinite(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite.*75"):
        validate_token(value)

