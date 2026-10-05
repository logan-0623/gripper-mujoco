import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "demo_restart_rollouts", Path(__file__).resolve().parents[2] / "scripts" / "demo_restart_rollouts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _record(frame, contact=False, grasp=False):
    return SimpleNamespace(frame_index=frame, replay=SimpleNamespace(simulator_state_index=frame),
                           labels=SimpleNamespace(contact=SimpleNamespace(gripper_target=contact),
                                                  stable_grasp=grasp))


def test_event_frames_anchor_on_first_contact_and_grasp_and_clip_to_episode():
    records = [_record(f, contact=f >= 20, grasp=f >= 26) for f in range(40)]
    frames = dict(module.event_frames(records))
    assert frames == {"contact-30": 0, "contact-15": 5, "contact-5": 15, "contact": 20,
                      "grasp": 26, "grasp+10": 36}
    late = dict(module.event_frames([_record(f, contact=f >= 30, grasp=f >= 35) for f in range(40)]))
    assert late["grasp+10"] == 39


def test_fraction_frames_cover_the_demonstration():
    assert module.fraction_frames(11) == [("0.0", 0), ("0.2", 2), ("0.4", 4), ("0.6", 6)]


def test_summary_groups_by_label_and_excludes_goal_satisfied_restores(tmp_path):
    root = tmp_path / "task0"
    root.mkdir()
    points = [{"initial_state_id": i, "demo": "demo_0", "frame": f, "label": label}
              for i, (f, label) in enumerate([(0, "contact-30"), (50, "grasp"), (90, "grasp")])]
    (root / "restart_points.json").write_text(json.dumps({"points": points}))
    (root / "physical_events.json").write_text(json.dumps({"episodes": [
        {"initial_state_id": 0, "success": False, "stable_grasp": False, "unintended_drop": False, "steps": 280},
        {"initial_state_id": 1, "success": True, "stable_grasp": True, "unintended_drop": False, "steps": 40},
        {"initial_state_id": 2, "success": True, "stable_grasp": False, "unintended_drop": False, "steps": 1}]}))
    (root / "restore_checks.json").write_text(json.dumps({"goal_satisfied_at_restore": {"0": False, "1": False, "2": True}}))
    summary = module.summarize(tmp_path, [0])
    assert summary["by_label"]["contact-30"]["success"] == 0
    assert summary["by_label"]["grasp"] == {"success": 1, "episodes": 1, "excluded_goal_already_satisfied": 1,
                                            "by_task": {"0": 1}}
