from __future__ import annotations

import numpy as np
import pytest

from interaction_vla.graph_control.schema import (
    ALL_CONDITIONS,
    CONDITIONS,
    ORACLE_CONDITIONS,
    TOKEN_DIM,
    TOKEN_FEATURE_NAMES,
    TOKEN_SLICES,
    empty_token,
    validate_token,
)


def test_control_uses_the_central_graph_v2_contract() -> None:
    assert ORACLE_CONDITIONS == ("flat", "oracle_graph_v2")
    assert ALL_CONDITIONS == (
        "flat",
        "oracle_graph_v2",
        "predicted_random_v2",
        "predicted_reflect_v2",
    )
    assert CONDITIONS == ALL_CONDITIONS
    cursor = 0
    for value in TOKEN_SLICES.values():
        assert value.start == cursor
        assert value.stop > value.start
        cursor = value.stop
    assert cursor == TOKEN_DIM == 89
    assert len(TOKEN_FEATURE_NAMES) == TOKEN_DIM


def test_empty_token_is_valid_finite_float32() -> None:
    token = empty_token()
    assert token.dtype == np.float32
    assert np.array_equal(token, np.zeros(89, dtype=np.float32))
    validated = validate_token(token)
    assert np.array_equal(validated, token)
    assert validated is not token


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(88, dtype=np.float32),
        np.zeros((1, 89), dtype=np.float32),
        np.full(89, np.nan, dtype=np.float32),
    ],
)
def test_validate_token_rejects_wrong_shape_or_nonfinite(value: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite.*89"):
        validate_token(value)
