from __future__ import annotations

import torch
from torch import nn

from interaction_vla.representation_study.taps.capture import (
    ForwardTapCapture,
    ModuleTap,
    pool_latent,
    resolve_module,
)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Conv2d(3, 4, 1)
        self.fused = nn.Linear(4, 5)
        self.action = nn.Linear(5, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        visual = self.visual(value).mean(dim=(2, 3))
        fused = self.fused(visual)
        return self.action(fused)


def test_pool_latent_handles_batch_first_sequence_first_and_spatial() -> None:
    assert pool_latent(torch.ones(2, 3, 4), batch_size=2, selector="last").shape == (2, 4)
    assert pool_latent(torch.ones(3, 2, 4), batch_size=2, selector="last").shape == (2, 4)
    assert pool_latent(torch.ones(2, 4, 3, 3), batch_size=2, selector="last").shape == (2, 4)


def test_forward_capture_collects_outputs_and_pre_action_inputs() -> None:
    model = _Toy()
    capture = ForwardTapCapture(
        model,
        (
            ModuleTap("vision", "visual"),
            ModuleTap("fused", "fused"),
            ModuleTap("pre_action", "action", capture_input=True),
        ),
    )
    output, latents = capture.capture(
        lambda: model(torch.ones(2, 3, 4, 4)), batch_size=2
    )
    assert output.shape == (2, 2)
    assert {name: tuple(value.shape) for name, value in latents.items()} == {
        "vision": (2, 4),
        "fused": (2, 5),
        "pre_action": (2, 5),
    }
    assert resolve_module(model, "action") is model.action

