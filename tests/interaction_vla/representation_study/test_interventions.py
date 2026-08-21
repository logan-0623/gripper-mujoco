from __future__ import annotations

import pytest
import torch
from torch import nn

from interaction_vla.representation_study.taps.capture import ModuleTap
from interaction_vla.representation_study.taps.intervene import (
    ForwardTapIntervention,
    intervention_tensor,
)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(3, 4, bias=False)
        self.action = nn.Linear(4, 2, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.action(self.hidden(value))


def test_intervention_tensor_controls_are_shape_preserving() -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.count_nonzero(
        intervention_tensor(value, batch_size=2, mode="zero")
    ) == 0
    mean = intervention_tensor(value, batch_size=2, mode="mean")
    assert torch.allclose(mean[0], mean[1])
    shuffled = intervention_tensor(value, batch_size=2, mode="matched_random")
    assert torch.equal(shuffled[0], value[1])
    with pytest.raises(ValueError, match="at least two"):
        intervention_tensor(value[:1], batch_size=1, mode="matched_random")


def test_forward_intervention_changes_action_without_mutating_weights() -> None:
    model = _Toy()
    inputs = torch.ones(2, 3)
    baseline = model(inputs)
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    intervention = ForwardTapIntervention(
        model, ModuleTap("hidden", "hidden", capture_input=False)
    )
    changed = intervention.run(lambda: model(inputs), batch_size=2, mode="zero")
    assert not torch.allclose(changed, baseline)
    assert torch.count_nonzero(changed) == 0
    assert intervention.last_tensor is not None
    assert torch.count_nonzero(intervention.last_tensor) == 0
    assert all(torch.equal(parameters[name], value) for name, value in model.named_parameters())
