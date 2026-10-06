import importlib.util
from pathlib import Path

import numpy as np


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe, analysis = _load("window_entry_probe"), _load("window_entry_analysis")


def test_push_direction_is_horizontal_fixed_per_state_and_keeps_gripper_open():
    a, b = probe.push_action(1, 7, 0.3), probe.push_action(1, 7, 0.3)
    np.testing.assert_array_equal(a, b)
    assert abs(np.hypot(a[0], a[1]) - 0.3) < 1e-12 and np.all(a[2:6] == 0) and a[6] == -1.0
    assert not np.allclose(a, probe.push_action(1, 8, 0.3))


def test_deviation_uses_nearest_demo_translation_and_rotation():
    identity6 = [1, 0, 0, 0, 1, 0]
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    rotated6 = [c, s, 0, -s, c, 0]  # 30 degrees about z
    references = [np.asarray([0.0, 0.0, 0.10, *identity6]), np.asarray([0.05, 0.0, 0.10, *identity6])]
    translation, angle = analysis.deviation([0.04, 0.0, 0.10, *rotated6], references)
    assert abs(translation - 1.0) < 1e-9
    assert abs(angle - 30.0) < 1e-6


def test_auroc_is_undefined_without_both_outcomes():
    assert analysis.auroc([1, 2], [True, True]) is None
    assert analysis.auroc([1, 2, 3, 4], [False, False, True, True]) == 1.0
