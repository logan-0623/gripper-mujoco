import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "demo_restart_rollouts", Path(__file__).resolve().parents[2] / "scripts" / "demo_restart_rollouts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_summary_groups_success_by_restart_fraction(tmp_path):
    root = tmp_path / "task0"
    root.mkdir()
    points = [{"initial_state_id": i, "demo": "demo_0", "frame": f, "fraction": fr}
              for i, (f, fr) in enumerate([(0, 0.0), (50, 0.5)])]
    (root / "restart_points.json").write_text(json.dumps({"points": points}))
    (root / "physical_events.json").write_text(json.dumps({"episodes": [
        {"initial_state_id": 0, "success": False, "stable_grasp": False, "unintended_drop": False},
        {"initial_state_id": 1, "success": True, "stable_grasp": True, "unintended_drop": False}]}))
    summary = module.summarize(tmp_path, [0])
    assert summary["by_fraction"]["0.0"]["success"] == 0
    assert summary["by_fraction"]["0.5"] == {"success": 1, "episodes": 1, "by_task": {"0": 1}}
