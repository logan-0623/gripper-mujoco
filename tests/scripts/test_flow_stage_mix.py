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
