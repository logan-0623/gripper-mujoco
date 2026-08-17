from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from interaction_vla.graph_control.features import (
    FrozenGraphRuntime,
    normalization_from_payload,
    pack_oracle_current,
    pack_predicted,
)
from interaction_vla.graph_control.schema import TOKEN_DIM, TOKEN_SLICES
from interaction_vla.graph_finetune.config import ModelConfig
from interaction_vla.graph_finetune.schema import (
    GraphV2Normalization,
    GraphV2Targets,
)
from interaction_vla.graph_pretrain.reflectvlm import Vocabulary


def _targets(frames: int = 2) -> GraphV2Targets:
    gripper = np.zeros((frames, 8), dtype=np.float32)
    gripper[:, 5:8] = 0.5
    goal = np.zeros((frames, 10), dtype=np.float32)
    goal[:, 8:10] = 0.5
    distractors = np.zeros((frames, 2, 7), dtype=np.float32)
    distractors[:, :, 4:7] = 0.5
    return GraphV2Targets(
        entity_mask=np.ones((frames, 6), dtype=np.bool_),
        entity_visibility=np.full((frames, 6, 2), 0.5, dtype=np.float32),
        relation_mask=np.ones((frames, 8), dtype=np.bool_),
        gripper_target_geometry=gripper,
        target_receptacle_geometry=goal,
        distractor_geometry=distractors,
        phase=np.zeros(frames, dtype=np.int64),
        relation_trends=np.zeros((frames, 4), dtype=np.float32),
        goal_relation=np.zeros(frames, dtype=np.int64),
        goal_operator=np.zeros(frames, dtype=np.int64),
        goal_predicate=np.zeros(frames, dtype=np.int64),
        goal_residual=np.zeros(frames, dtype=np.float32),
    )


def _outputs() -> dict[str, torch.Tensor]:
    gripper = torch.arange(8, dtype=torch.float32)
    gripper[5:8] = torch.tensor((0.1, 0.2, 0.3))
    target = torch.arange(10, dtype=torch.float32)
    target[8:10] = torch.tensor((0.4, 0.5))
    distractors = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    distractors[:, 4:7] = torch.tensor(((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)))
    return {
        "entity_mask_logits": torch.zeros(1, 6),
        "entity_visibility": torch.full((1, 6, 2), 0.25),
        "relation_mask_logits": torch.zeros(1, 8),
        "gripper_target_geometry": gripper[None],
        "target_receptacle_geometry": target[None],
        "distractor_geometry": distractors[None],
        "phase_logits": torch.zeros(1, 6),
        "relation_trends": torch.arange(4, dtype=torch.float32)[None],
        "goal_relation_logits": torch.zeros(1, 8),
        "goal_operator_logits": torch.zeros(1, 5),
        "goal_predicate_logits": torch.zeros(1, 7),
        "goal_residual": torch.tensor([0.75]),
    }


def _normalization() -> GraphV2Normalization:
    return GraphV2Normalization(
        state_mean=np.zeros(10, dtype=np.float32),
        state_std=np.ones(10, dtype=np.float32),
        workspace_scale=2.0,
        velocity_scale=4.0,
    )


def test_predicted_token_packs_all_graph_v2_groups_and_distributions() -> None:
    token = pack_predicted(_outputs())

    assert token.shape == (TOKEN_DIM,)
    assert token.dtype == np.float32
    assert np.isfinite(token).all()
    np.testing.assert_allclose(token[TOKEN_SLICES["entity_presence"]], 0.5)
    np.testing.assert_allclose(token[TOKEN_SLICES["entity_visibility"]], 0.25)
    np.testing.assert_allclose(token[TOKEN_SLICES["relation_presence"]], 0.5)
    np.testing.assert_array_equal(
        token[TOKEN_SLICES["gripper_target_geometry"]],
        _outputs()["gripper_target_geometry"][0],
    )
    np.testing.assert_array_equal(
        token[TOKEN_SLICES["target_receptacle_geometry"]],
        _outputs()["target_receptacle_geometry"][0],
    )
    np.testing.assert_array_equal(
        token[TOKEN_SLICES["distractor_geometry"]],
        _outputs()["distractor_geometry"][0].reshape(-1),
    )
    np.testing.assert_array_equal(
        token[TOKEN_SLICES["relation_trends"]], np.arange(4)
    )
    for name in ("phase", "next_relation", "relation_operator", "predicate"):
        assert np.isclose(token[TOKEN_SLICES[name]].sum(), 1.0)
    np.testing.assert_allclose(token[TOKEN_SLICES["goal_residual"]], [0.75])


def test_oracle_current_uses_exact_normalized_graph_v2_target() -> None:
    targets = _targets(frames=2)
    gripper = targets.gripper_target_geometry.copy()
    gripper[1] = np.asarray((2, 4, 6, 8, 12, 0.1, 0.2, 0.3), dtype=np.float32)
    goal = targets.target_receptacle_geometry.copy()
    goal[1] = np.asarray((*range(2, 18, 2), 0.4, 0.5), dtype=np.float32)
    distractors = targets.distractor_geometry.copy()
    distractors[1, 0] = np.asarray((2, 4, 6, 8, 0.1, 0.2, 0.3), dtype=np.float32)
    distractors[1, 1] = np.asarray((4, 6, 8, 10, 0.4, 0.5, 0.6), dtype=np.float32)
    trends = targets.relation_trends.copy()
    trends[1] = np.asarray((2, 0.25, 4, 6), dtype=np.float32)
    targets = replace(
        targets,
        gripper_target_geometry=gripper,
        target_receptacle_geometry=goal,
        distractor_geometry=distractors,
        relation_trends=trends,
    )

    token = pack_oracle_current(
        targets,
        frame_index=1,
        normalization=_normalization(),
    )

    np.testing.assert_allclose(
        token[TOKEN_SLICES["gripper_target_geometry"]],
        (1, 2, 3, 4, 3, 0.1, 0.2, 0.3),
    )
    np.testing.assert_allclose(
        token[TOKEN_SLICES["target_receptacle_geometry"]],
        (1, 2, 3, 4, 5, 6, 7, 8, 0.4, 0.5),
    )
    np.testing.assert_allclose(
        token[TOKEN_SLICES["relation_trends"]], (1, 0.25, 2, 3)
    )
    assert token[TOKEN_SLICES["phase"]].sum() == 1.0


def test_normalization_payload_is_validated_and_copied() -> None:
    source = _normalization()
    payload = {
        "state_mean": source.state_mean,
        "state_std": source.state_std,
        "workspace_scale": source.workspace_scale,
        "velocity_scale": source.velocity_scale,
    }

    restored = normalization_from_payload(payload)
    payload["state_mean"][0] = 999.0

    assert restored.state_mean[0] != 999.0


class _RecordingModel:
    def __init__(self) -> None:
        self.previous: list[np.ndarray] = []
        self.language_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def __call__(self, agent, wrist, state, tokens, mask, previous):
        self.previous.append(previous.detach().cpu().numpy().copy())
        self.language_shapes.append((tuple(tokens.shape), tuple(mask.shape)))
        return _outputs()


def test_frozen_runtime_recurrence_uses_previous_prediction_and_reset() -> None:
    runtime = object.__new__(FrozenGraphRuntime)
    runtime.device = torch.device("cpu")
    runtime.model = _RecordingModel()
    runtime.vocabulary = Vocabulary(("<pad>", "<unk>", "place", "target"))
    runtime.model_config = ModelConfig(
        image_size=8,
        max_language_tokens=4,
        image_embedding_dim=8,
        text_embedding_dim=4,
        graph_embedding_dim=8,
    )
    runtime.normalization = _normalization()
    runtime.reset()

    kwargs = {
        "agent_rgb": torch.zeros(3, 8, 8),
        "wrist_rgb": torch.zeros(3, 8, 8),
        "state": np.zeros(10, dtype=np.float32),
        "task": "place target",
    }
    first = runtime.predict_token(**kwargs)
    runtime.predict_token(**kwargs)
    runtime.reset()
    runtime.predict_token(**kwargs)

    assert np.count_nonzero(runtime.model.previous[0]) == 0
    np.testing.assert_array_equal(runtime.model.previous[1][0], first)
    assert np.count_nonzero(runtime.model.previous[2]) == 0
    assert runtime.model.language_shapes == [((1, 4), (1, 4))] * 3


def test_pack_predicted_rejects_missing_nonfinite_and_unbounded_outputs() -> None:
    outputs = _outputs()
    del outputs["goal_predicate_logits"]
    with pytest.raises(ValueError, match="missing"):
        pack_predicted(outputs)

    outputs = _outputs()
    outputs["goal_residual"][0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        pack_predicted(outputs)

    outputs = _outputs()
    outputs["entity_visibility"][0, 0, 0] = 2.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pack_predicted(outputs)
