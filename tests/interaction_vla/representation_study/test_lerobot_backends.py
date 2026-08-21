from __future__ import annotations

import pytest
import torch

from interaction_vla.representation_study.backends.lerobot import (
    _TRAINABLE_PREFIXES,
    make_backend,
)
from interaction_vla.representation_study.rl.core import ResidualActorCritic
from interaction_vla.representation_study.taps.registry import registered_taps


@pytest.mark.parametrize("name", ["act", "smolvla", "pi0"])
def test_backend_factory_exposes_fixed_tap_contract_without_loading_weights(name: str) -> None:
    backend = make_backend(name, device="cpu")
    assert backend.backend_name == name
    assert tuple(tap.tap_id for tap in registered_taps(name))


def test_backend_factory_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="unsupported backend"):
        make_backend("unknown", device="cpu")


def test_act_flat_checkpoint_compatibility_injects_only_zero_environment_state() -> None:
    backend = make_backend("act", device="cpu")
    feature = type("Feature", (), {"shape": (89,)})()
    policy = type("Policy", (), {"config": type("Config", (), {"env_state_feature": feature})()})()
    backend.policy = policy
    backend.preprocessor = object()
    backend.postprocessor = object()
    raw = backend._raw_batch({"observation.state": torch.ones(2, 10)})
    assert tuple(raw["observation.environment_state"].shape) == (2, 89)
    assert torch.count_nonzero(raw["observation.environment_state"]) == 0


def test_residual_bundle_changes_only_immediate_action() -> None:
    backend = make_backend("act", device="cpu")
    residual = ResidualActorCritic(8, adapt_representation=False)
    with torch.no_grad():
        residual.actor_mean.bias.fill_(1.0)
    backend.residual_policy = residual
    backend.residual_scale = torch.full((7,), 0.1)
    actions = torch.zeros(2, 8, 7)
    changed = backend._apply_residual(actions, torch.randn(2, 8))
    assert torch.count_nonzero(changed[:, 0]) > 0
    assert torch.count_nonzero(changed[:, 1:]) == 0


def test_residual_bundle_records_clipping_before_returning_bounded_action() -> None:
    backend = make_backend("act", device="cpu")
    residual = ResidualActorCritic(8, adapt_representation=False)
    with torch.no_grad():
        residual.actor_mean.bias.fill_(10.0)
    backend.residual_policy = residual
    backend.residual_scale = torch.full((7,), 0.2)
    actions = torch.full((1, 2, 7), 0.95)

    changed = backend._apply_residual(actions, torch.zeros(1, 8))

    assert backend.last_residual_action_was_clipped is True
    assert torch.all(changed[:, 0, :6] <= 1.0)
    assert torch.all(changed[:, 0, 6] <= 1.0)


def test_modern_vla_fusion_group_does_not_unfreeze_entire_vlm() -> None:
    assert "model.vlm_with_expert" not in _TRAINABLE_PREFIXES["smolvla"]["fusion"]
    assert "model.paligemma_with_expert" not in _TRAINABLE_PREFIXES["pi0"]["fusion"]
    assert any("lm_expert" in value for value in _TRAINABLE_PREFIXES["smolvla"]["fusion"])
