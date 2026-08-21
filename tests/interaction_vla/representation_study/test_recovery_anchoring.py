from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from interaction_vla.representation_study.rl.anchoring import (
    NominalAnchorCache,
    latent_drift_loss,
    nominal_residual_loss,
)


def test_nominal_anchor_prefers_zero_residual() -> None:
    residual = torch.tensor([[0.2, -0.1]])
    assert nominal_residual_loss(residual).item() == pytest.approx(0.025)


def test_latent_anchor_stops_gradient_on_sft_target() -> None:
    current = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 8, requires_grad=True)
    latent_drift_loss(current, target).backward()
    assert current.grad is not None
    assert target.grad is None


def _cache(tmp_path: Path) -> NominalAnchorCache:
    return NominalAnchorCache.create(
        tmp_path,
        latents=np.arange(40, dtype=np.float32).reshape(10, 4),
        case_ids=tuple(f"case-{index}" for index in range(10)),
        dataset_fingerprint="a" * 64,
        sft_checkpoint="sft/checkpoint",
        tap_id="pre_action",
        seed=7,
    )


def test_nominal_cache_sampling_is_resume_exact(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    state = cache.state_dict()
    expected = cache.sample(6)
    cache.load_state_dict(state)
    repeated = cache.sample(6)
    assert repeated.case_ids == expected.case_ids
    np.testing.assert_array_equal(repeated.latents, expected.latents)


def test_nominal_cache_load_validates_binding(tmp_path: Path) -> None:
    _cache(tmp_path)
    with pytest.raises(ValueError, match="dataset fingerprint"):
        NominalAnchorCache.load(
            tmp_path,
            dataset_fingerprint="b" * 64,
            sft_checkpoint="sft/checkpoint",
            tap_id="pre_action",
            seed=7,
        )
