from __future__ import annotations

import torch

from interaction_vla.graph_pretrain.model import (
    ReflectGraphEstimator,
    graph_prediction_loss,
)


def batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    dependency_mask = ~torch.eye(6, dtype=torch.bool)
    return {
        "image": torch.rand(batch_size, 3, 32, 32),
        "history_tokens": torch.tensor([[2, 3, 0], [3, 1, 4]], dtype=torch.long),
        "history_mask": torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        ),
        "target_index": torch.tensor([0, 3]),
        "in_hand_index": torch.tensor([4, 1]),
        "object_mask": torch.ones(batch_size, 6, dtype=torch.bool),
        "state_ids": torch.tensor(
            [[4, 1, 2, 3, 1, 4], [1, 2, 3, 4, 2, 1]]
        ),
        "upright_ids": torch.tensor(
            [[2, 2, 1, 1, 2, 1], [2, 1, 2, 1, 2, 1]]
        ),
        "dependency": torch.zeros(batch_size, 6, 6),
        "dependency_mask": dependency_mask.unsqueeze(0).expand(batch_size, -1, -1),
        "phase_id": torch.tensor([0, 3]),
        "goal_operator_id": torch.tensor([0, 0]),
        "goal_predicate_id": torch.tensor([3, 4]),
    }


def test_graph_estimator_outputs_all_semantic_graph_heads() -> None:
    model = ReflectGraphEstimator(
        vocab_size=8,
        image_embedding_dim=32,
        text_embedding_dim=16,
        graph_embedding_dim=24,
    )
    values = batch()

    outputs = model(
        values["image"], values["history_tokens"], values["history_mask"]
    )

    assert outputs["target_logits"].shape == (2, 6)
    assert outputs["in_hand_logits"].shape == (2, 7)
    assert outputs["state_logits"].shape == (2, 6, 5)
    assert outputs["upright_logits"].shape == (2, 6, 3)
    assert outputs["dependency_logits"].shape == (2, 6, 6)
    assert outputs["phase_logits"].shape == (2, 4)
    assert outputs["operator_logits"].shape == (2, 5)
    assert outputs["predicate_logits"].shape == (2, 7)
    assert outputs["graph_embedding"].shape == (2, 24)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_graph_prediction_loss_drives_a_finite_optimizer_update() -> None:
    torch.manual_seed(0)
    model = ReflectGraphEstimator(
        vocab_size=8,
        image_embedding_dim=32,
        text_embedding_dim=16,
        graph_embedding_dim=24,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    values = batch()
    before = next(model.parameters()).detach().clone()

    outputs = model(
        values["image"], values["history_tokens"], values["history_mask"]
    )
    losses = graph_prediction_loss(outputs, values)
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    optimizer.step()

    assert set(losses) == {
        "target",
        "in_hand",
        "state",
        "upright",
        "dependency",
        "phase",
        "operator",
        "predicate",
        "total",
    }
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert torch.isfinite(gradient_norm) and float(gradient_norm) > 0.0
    assert not torch.equal(before, next(model.parameters()).detach())
