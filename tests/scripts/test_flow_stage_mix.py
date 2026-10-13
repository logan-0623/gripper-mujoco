import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

scripts = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("flow_stage_mix", scripts / "flow_stage_mix.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _flow():
    flow = SimpleNamespace(config=SimpleNamespace(chunk_size=2, max_action_dim=3, num_steps=10))
    flow.sample_noise = lambda shape, device: torch.ones(shape)
    flow.sample_actions = lambda *a, noise=None, **k: noise  # stands in for the integrator
    flow.denoise_step = None
    return flow


def test_noise_scale_applies_everywhere_or_only_in_window():
    for scope, active, expected in (("all", False, 0.5), ("window", False, 1.0)):
        flow = _flow()
        window = SimpleNamespace(active=active)
        module.install_mix(SimpleNamespace(model=flow), SimpleNamespace(model=_flow()), window,
                           frozenset(), "own", {}, noise_scale=0.5, noise_scope=scope)
        out = flow.sample_actions(None, None, None, None, torch.zeros(1, 8))
        assert torch.all(out == expected), (scope, out)


def test_guidance_extrapolates_from_the_weak_model_on_chosen_steps(monkeypatch):
    monkeypatch.setattr(module, "prefix_cache", lambda *a: ("pad", "weak-cache"))
    flow, weak = _flow(), _flow()
    flow.denoise_step = lambda **k: torch.full((1,), 3.0)
    weak.denoise_step = lambda **k: torch.full((1,), 1.0)

    def integrate(*a, noise=None, **k):
        return [float(flow.denoise_step(prefix_pad_masks="own", past_key_values="own-cache", x_t=None,
                                        timestep=None)) for _ in range(10)]
    flow.sample_actions = integrate
    module.install_mix(SimpleNamespace(model=flow), SimpleNamespace(model=weak), SimpleNamespace(active=True),
                       frozenset(range(5)), "own", {}, guide_weight=2.0)
    assert flow.sample_actions(None, None, None, None, torch.zeros(1, 8)) == [5.0] * 5 + [3.0] * 5
