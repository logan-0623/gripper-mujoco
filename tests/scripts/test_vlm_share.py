import importlib.util
import math
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("vlm_share", Path(__file__).resolve().parents[2] / "scripts" / "vlm_share.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_shares_split_the_loss_increase_between_sides():
    m_v, m_e, r = module.shares(np.array([1.0, 1.0]), np.array([1.2, 1.2]), np.array([1.6, 1.6]))
    assert math.isclose(m_v, 0.2) and math.isclose(m_e, 0.6) and math.isclose(r, 0.25)


def test_switch_point_needs_two_consecutive_below_rho():
    names = ["10k", "15k", "20k", "25k"]
    assert module.switch_point(names, [0.1, 0.4, 0.2, 0.1], 0.25) == "20k"
    assert module.switch_point(names, [0.1, 0.4, 0.2, 0.3], 0.25) is None


def test_bind_cameras_leaves_extra_features_missing():
    batch = {"observation.images.image": 1, "observation.images.image2": 2, "observation.state": 3}
    out = module.bind_cameras(batch, ["observation.images.camera1", "observation.images.camera2",
                                      "observation.images.camera3"])
    assert out == {"observation.state": 3, "observation.images.camera1": 1, "observation.images.camera2": 2}
