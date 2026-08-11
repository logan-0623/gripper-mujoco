from __future__ import annotations

from pathlib import Path

import torch

from interaction_vla.graph_finetune.model import (
    MuJoCoGraphEstimator,
    graph_finetune_loss,
    initialize_paired_models,
)
from interaction_vla.graph_pretrain.model import ReflectGraphEstimator
from interaction_vla.graph_pretrain.reflectvlm import Vocabulary
from interaction_vla.graph_pretrain.schema import SCHEMA_VERSION as REFLECT_SCHEMA


def model_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "agent_rgb": torch.rand(batch_size, 3, 32, 32),
        "wrist_rgb": torch.rand(batch_size, 3, 32, 32),
        "state": torch.rand(batch_size, 10),
        "language_tokens": torch.tensor([[2, 3, 0], [3, 1, 4]]),
        "language_mask": torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        ),
        "entity_mask": torch.tensor(
            [[True, True, True, True, True, False], [True] * 6]
        ),
        "entity_visibility": torch.rand(batch_size, 6, 2),
        "relation_mask": torch.tensor(
            [[True, True, True, True, True, False, False, True], [True] * 8]
        ),
        "relation_semantics": torch.rand(batch_size, 8, 10),
        "goal_relation": torch.tensor([0, 2]),
        "goal_operator": torch.tensor([0, 3]),
        "goal_predicate": torch.tensor([3, 4]),
        "goal_residual": torch.tensor([-0.2, 0.0]),
    }


def build_model() -> MuJoCoGraphEstimator:
    return MuJoCoGraphEstimator(
        vocab_size=8,
        image_embedding_dim=32,
        text_embedding_dim=16,
        graph_embedding_dim=24,
    )


def test_estimator_outputs_complete_mujoco_semantic_graph() -> None:
    model = build_model()
    batch = model_batch()

    outputs = model(
        batch["agent_rgb"],
        batch["wrist_rgb"],
        batch["state"],
        batch["language_tokens"],
        batch["language_mask"],
    )

    assert outputs["entity_mask_logits"].shape == (2, 6)
    assert outputs["entity_visibility"].shape == (2, 6, 2)
    assert outputs["relation_mask_logits"].shape == (2, 8)
    assert outputs["relation_semantics"].shape == (2, 8, 10)
    assert outputs["goal_relation_logits"].shape == (2, 8)
    assert outputs["goal_operator_logits"].shape == (2, 5)
    assert outputs["goal_predicate_logits"].shape == (2, 7)
    assert outputs["goal_residual"].shape == (2,)
    assert outputs["graph_embedding"].shape == (2, 24)
    assert all(torch.isfinite(value).all() for value in outputs.values())


def test_masked_loss_ignores_inactive_entity_and_relation_regression() -> None:
    model = build_model()
    batch = model_batch()
    outputs = model(
        batch["agent_rgb"],
        batch["wrist_rgb"],
        batch["state"],
        batch["language_tokens"],
        batch["language_mask"],
    )
    first = graph_finetune_loss(outputs, batch)
    changed = {name: value.clone() for name, value in batch.items()}
    changed["entity_visibility"][0, 5] = 1_000_000.0
    changed["relation_semantics"][0, 5:7] = 1_000_000.0
    second = graph_finetune_loss(outputs, changed)

    assert torch.equal(first["entity_visibility"], second["entity_visibility"])
    assert torch.equal(first["relation_semantics"], second["relation_semantics"])


def test_loss_drives_finite_optimizer_update() -> None:
    torch.manual_seed(0)
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = model_batch()
    before = next(model.parameters()).detach().clone()

    outputs = model(
        batch["agent_rgb"],
        batch["wrist_rgb"],
        batch["state"],
        batch["language_tokens"],
        batch["language_mask"],
    )
    losses = graph_finetune_loss(outputs, batch)
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    assert set(losses) == {
        "entity_mask",
        "entity_visibility",
        "relation_mask",
        "relation_semantics",
        "goal_relation",
        "goal_operator",
        "goal_predicate",
        "goal_residual",
        "total",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    assert torch.isfinite(gradient_norm) and float(gradient_norm) > 0.0
    assert not torch.equal(before, next(model.parameters()).detach())


def save_reflect_checkpoint(path: Path) -> ReflectGraphEstimator:
    torch.manual_seed(101)
    model = ReflectGraphEstimator(
        vocab_size=5,
        image_embedding_dim=32,
        text_embedding_dim=16,
        graph_embedding_dim=24,
    )
    torch.save(
        {
            "schema_version": REFLECT_SCHEMA,
            "repo_id": "test/reflect",
            "source_split": "train",
            "split_seed": 0,
            "model_config": {
                "image_size": 32,
                "max_history_tokens": 8,
                "image_embedding_dim": 32,
                "text_embedding_dim": 16,
                "graph_embedding_dim": 24,
            },
            "vocabulary": ["<pad>", "<unk>", "pick", "target", "place"],
            "model_state": model.state_dict(),
        },
        path,
    )
    return model


def test_reflect_transfer_changes_only_compatible_paired_parameters(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "reflect.pt"
    reflect = save_reflect_checkpoint(checkpoint)
    vocabulary = Vocabulary(("<pad>", "<unk>", "pick", "target", "new"))

    random_model, pretrained_model, transfer = initialize_paired_models(
        vocab_size=len(vocabulary.tokens),
        vocabulary=vocabulary,
        reflect_checkpoint=checkpoint,
        seed=7,
        image_embedding_dim=32,
        text_embedding_dim=16,
        graph_embedding_dim=24,
    )

    assert torch.equal(
        pretrained_model.image_encoder[0].weight,
        reflect.image_encoder[0].weight,
    )
    assert not torch.equal(
        random_model.image_encoder[0].weight,
        reflect.image_encoder[0].weight,
    )
    assert torch.equal(
        random_model.state_encoder[0].weight,
        pretrained_model.state_encoder[0].weight,
    )
    assert torch.equal(
        random_model.relation_head.weight,
        pretrained_model.relation_head.weight,
    )
    assert torch.equal(
        pretrained_model.operator_head.weight,
        reflect.operator_head.weight,
    )
    assert transfer.copied_token_count == 4
    assert "image_encoder" in transfer.copied_modules
    assert "fusion" in transfer.copied_modules
    assert "operator_head" in transfer.copied_modules
    assert "predicate_head" in transfer.copied_modules


def test_reflect_transfer_rejects_incompatible_dimensions(tmp_path: Path) -> None:
    checkpoint = tmp_path / "reflect.pt"
    save_reflect_checkpoint(checkpoint)

    try:
        initialize_paired_models(
            vocab_size=4,
            vocabulary=Vocabulary(("<pad>", "<unk>", "pick", "target")),
            reflect_checkpoint=checkpoint,
            seed=7,
            image_embedding_dim=64,
            text_embedding_dim=16,
            graph_embedding_dim=24,
        )
    except ValueError as error:
        assert "dimension" in str(error)
    else:
        raise AssertionError("incompatible ReflectVLM dimensions were accepted")
