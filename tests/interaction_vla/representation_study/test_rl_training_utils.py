from __future__ import annotations

import torch

from interaction_vla.representation_study.rl.training import (
    _parent_stage,
    collate_packed_observations,
    pack_observation,
    recompute_latents_with_policy_seeds,
)


def test_online_observation_pack_is_lossless_at_uint8_precision() -> None:
    source = {
        "observation.images.agent": torch.rand(1, 3, 8, 8),
        "observation.images.wrist": torch.rand(1, 3, 8, 8),
        "observation.state": torch.randn(1, 10),
        "task": ["pick the green cube"],
    }
    packed = pack_observation(source)
    assert packed["observation.images.agent"].dtype == torch.uint8
    restored = collate_packed_observations([packed, packed], device=torch.device("cpu"))
    assert restored["observation.images.agent"].shape == (2, 3, 8, 8)
    assert torch.max(
        torch.abs(restored["observation.images.agent"][0] - source["observation.images.agent"][0])
    ) <= 1.0 / 255.0
    assert restored["task"] == ["pick the green cube", "pick the green cube"]


def test_rl_branches_share_the_same_sft_parent() -> None:
    assert _parent_stage("rl_head") == "sft"
    assert _parent_stage("rl_representation") == "sft"


def test_representation_update_replays_per_transition_policy_noise() -> None:
    class Runtime:
        device = torch.device("cpu")

        def differentiable_action_and_latent(self, batch, *, tap_id):
            batch_size = batch["observation.state"].shape[0]
            return torch.zeros(batch_size, 1), torch.rand(batch_size, 3)

    rows = [
        pack_observation(
            {
                "observation.images.agent": torch.zeros(1, 3, 2, 2),
                "observation.images.wrist": torch.zeros(1, 3, 2, 2),
                "observation.state": torch.zeros(1, 10),
                "task": ["pick"],
            }
        )
        for _ in range(3)
    ]
    first = recompute_latents_with_policy_seeds(
        Runtime(), rows, [11, 22, 11], tap_id="pre_action"
    )
    second = recompute_latents_with_policy_seeds(
        Runtime(), rows, [11, 22, 11], tap_id="pre_action"
    )

    assert torch.allclose(first, second)
    assert torch.allclose(first[0], first[2])
    assert not torch.allclose(first[0], first[1])
