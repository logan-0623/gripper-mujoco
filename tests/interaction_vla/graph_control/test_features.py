from __future__ import annotations

import numpy as np
import pytest
import torch

from interaction_vla.graph_control.features import (
    CurrentGraphFields,
    normalization_from_payload,
    pack_oracle_current,
    pack_predicted,
)
from interaction_vla.graph_control.schema import TOKEN_SLICES
from interaction_vla.graph_finetune.data import GraphNormalization


def _outputs() -> dict[str, torch.Tensor]:
    semantics = torch.arange(80, dtype=torch.float32).reshape(1, 8, 10)
    return {
        "entity_mask_logits": torch.zeros(1, 6),
        "entity_visibility": torch.full((1, 6, 2), 0.25),
        "relation_mask_logits": torch.zeros(1, 8),
        "relation_semantics": semantics,
        "goal_relation_logits": torch.zeros(1, 8),
        "goal_operator_logits": torch.zeros(1, 5),
        "goal_predicate_logits": torch.zeros(1, 7),
        "goal_residual": torch.tensor([1.5]),
    }


def _normalization() -> GraphNormalization:
    return GraphNormalization(
        state_mean=np.zeros(10, dtype=np.float32),
        state_std=np.ones(10, dtype=np.float32),
        relation_mean=np.arange(80, dtype=np.float32).reshape(8, 10),
        relation_std=np.full((8, 10), 2.0, dtype=np.float32),
        residual_mean=0.25,
        residual_std=0.5,
    )


def test_predicted_token_uses_probabilities_and_normalized_semantics() -> None:
    token = pack_predicted(_outputs())

    assert token.dtype == np.float32
    np.testing.assert_allclose(token[TOKEN_SLICES["entity_presence"]], 0.5)
    np.testing.assert_allclose(token[TOKEN_SLICES["entity_visibility"]], 0.25)
    np.testing.assert_allclose(token[TOKEN_SLICES["relation_presence"]], 0.5)
    np.testing.assert_array_equal(
        token[TOKEN_SLICES["gripper_target_semantics"]], np.arange(10)
    )
    np.testing.assert_array_equal(
        token[TOKEN_SLICES["target_receptacle_semantics"]], np.arange(10, 20)
    )
    np.testing.assert_array_equal(
        token[TOKEN_SLICES["distractor_risks"]],
        np.array([36, 37, 46, 47, 56, 57, 66, 67], dtype=np.float32),
    )
    np.testing.assert_allclose(token[TOKEN_SLICES["next_relation"]], 1.0 / 8.0)
    np.testing.assert_allclose(token[TOKEN_SLICES["relation_operator"]], 1.0 / 5.0)
    np.testing.assert_allclose(token[TOKEN_SLICES["predicate"]], 1.0 / 7.0)
    np.testing.assert_allclose(token[TOKEN_SLICES["goal_residual"]], [1.5])


def test_oracle_current_replaces_only_current_graph_slices() -> None:
    predicted = pack_predicted(_outputs())
    normalization = _normalization()
    normalized_semantics = np.arange(80, dtype=np.float32).reshape(8, 10) / 10.0
    current = CurrentGraphFields(
        entity_mask=np.array([1, 1, 1, 1, 0, 1], dtype=np.bool_),
        entity_visibility=np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(6, 2),
        relation_mask=np.array([1, 1, 0, 1, 1, 0, 1, 1], dtype=np.bool_),
        relation_semantics=(
            normalization.relation_mean
            + normalized_semantics * normalization.relation_std
        ),
    )

    token = pack_oracle_current(current, predicted, normalization)

    np.testing.assert_array_equal(
        token[TOKEN_SLICES["entity_presence"]], current.entity_mask.astype(np.float32)
    )
    np.testing.assert_array_equal(
        token[TOKEN_SLICES["entity_visibility"]], current.entity_visibility.reshape(-1)
    )
    np.testing.assert_array_equal(
        token[TOKEN_SLICES["relation_presence"]], current.relation_mask.astype(np.float32)
    )
    np.testing.assert_allclose(
        token[TOKEN_SLICES["gripper_target_semantics"]],
        normalized_semantics[0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        token[TOKEN_SLICES["target_receptacle_semantics"]],
        normalized_semantics[1],
        atol=1e-6,
    )
    for name in ("next_relation", "relation_operator", "predicate", "goal_residual"):
        np.testing.assert_array_equal(token[TOKEN_SLICES[name]], predicted[TOKEN_SLICES[name]])


def test_oracle_current_contract_cannot_accept_future_relation_goal() -> None:
    kwargs = {
        "entity_mask": np.ones(6, dtype=np.bool_),
        "entity_visibility": np.ones((6, 2), dtype=np.float32),
        "relation_mask": np.ones(8, dtype=np.bool_),
        "relation_semantics": np.ones((8, 10), dtype=np.float32),
        "relation_goal": np.zeros(5, dtype=np.float32),
    }
    with pytest.raises(TypeError, match="relation_goal"):
        CurrentGraphFields(**kwargs)


def test_normalization_payload_is_validated_and_copied() -> None:
    source = _normalization()
    payload = {
        "state_mean": source.state_mean,
        "state_std": source.state_std,
        "relation_mean": source.relation_mean,
        "relation_std": source.relation_std,
        "residual_mean": source.residual_mean,
        "residual_std": source.residual_std,
    }
    restored = normalization_from_payload(payload)
    payload["relation_mean"][0, 0] = 999.0
    assert restored.relation_mean[0, 0] != 999.0


def test_pack_predicted_rejects_missing_or_nonfinite_outputs() -> None:
    outputs = _outputs()
    del outputs["goal_predicate_logits"]
    with pytest.raises(ValueError, match="missing"):
        pack_predicted(outputs)

    outputs = _outputs()
    outputs["goal_residual"][0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        pack_predicted(outputs)
