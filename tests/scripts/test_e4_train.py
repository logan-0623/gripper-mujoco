import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("e4_train", Path(__file__).resolve().parents[2] / "scripts" / "e4_train.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_window_spans_first_close_and_clips_to_episode():
    actions = np.zeros((40, 7))
    actions[:, 6] = -1.0
    actions[5:, 6] = 1.0
    mask = module.window_mask(actions, pre=20, post=3)
    assert mask[:9].all() and not mask[9:].any()
    assert not module.window_mask(-np.ones((10, 7)), 20, 25).any()


def test_window_weighter_upweights_window_frames_and_keeps_mean_one(monkeypatch):
    import lerobot.utils.sample_weighting as sample_weighting

    monkeypatch.setattr(sample_weighting, "make_sample_weighter", sample_weighting.make_sample_weighter)
    module.install_window_weighter(np.array([True, False, False, False]), weight=3.0)
    config = sample_weighting.SampleWeightingConfig(type="window")
    weighter = sample_weighting.make_sample_weighter(config, policy=None, device=torch.device("cpu"))
    weights, stats = weighter.compute_batch_weights({"index": torch.tensor([0, 1, 2, 3])})
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert torch.isclose(weights[0] / weights[1], torch.tensor(3.0))
    assert stats["window_fraction"] == 0.25
