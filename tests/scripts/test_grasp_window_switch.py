import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location(
    "grasp_window_switch", Path(__file__).resolve().parents[2] / "scripts" / "grasp_window_switch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _frame(distance, z=0.0, grasped=False):
    pose = np.eye(4)
    pose[2, 3] = z
    return SimpleNamespace(gripper_target_surface_distance=distance, target_pose=pose, grasped=grasped)


def test_grasp_window_opens_near_target_and_stays_closed_after_lift(monkeypatch):
    monkeypatch.setattr(module, "_stable_grasp", lambda frames, i, t: frames[i].grasped)
    window = module.GraspWindow("grasp", approach_distance=0.10, lift=0.02)
    window.reset(_frame(0.30))
    assert not window.active
    window.observe(_frame(0.08))
    assert window.active and window.opened_at == 1
    window.observe(_frame(0.0, z=0.01, grasped=True))
    assert window.active  # grasped but not lifted enough
    window.observe(_frame(0.0, z=0.03, grasped=True))
    assert not window.active and window.closed_at == 3
    window.observe(_frame(0.05))
    assert not window.active


def test_control_modes_ignore_the_window(monkeypatch):
    monkeypatch.setattr(module, "_stable_grasp", lambda frames, i, t: False)
    for mode, expected in (("never", False), ("always", True)):
        window = module.GraspWindow(mode, approach_distance=0.10, lift=0.02)
        window.reset(_frame(0.05))
        assert window.active is expected


def test_summary_joins_events_and_window_metadata(tmp_path):
    root = tmp_path / "task1"
    root.mkdir()
    (root / "physical_events.json").write_text(json.dumps({"episodes": [
        {"initial_state_id": 0, "success": True, "stable_grasp": True},
        {"initial_state_id": 1, "success": False, "stable_grasp": False}]}))
    (root / "window_metadata.json").write_text(json.dumps({"episodes": [
        {"initial_state_id": 0, "steps": 100, "donor_steps": 20, "window_opened_step": 50, "window_closed_step": 70},
        {"initial_state_id": 1, "steps": 280, "donor_steps": 0, "window_opened_step": None,
         "window_closed_step": None}]}))
    summary = module.summarize(tmp_path, [1])
    assert summary["success"] == 1 and summary["episodes"] == 2
    assert abs(summary["mean_donor_fraction"] - 20 / 380) < 1e-9
