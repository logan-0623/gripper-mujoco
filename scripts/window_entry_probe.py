"""E3 / E3b: how far from a good grasp approach is the policy when it reaches the object,
and how much deviation there can it correct?

Single-policy closed-loop LIBERO Spatial episodes under the capability-timeline
evaluation contract. The first step whose gripper-target surface distance is at
most ``--approach-distance`` (0.10 m, the E2 window start) is the window entry.

* E3: at entry, the target pose in the gripper frame (translation + rotation6D,
  the StateBank ``gripper_to_target`` convention) is recorded, so entry states
  can be compared with demonstration entries and related to success.
* E3b: with ``--push-steps k > 0``, the next k actions after entry are replaced
  by a horizontal end-effector translation of ``--push-action`` in a direction
  fixed per (task, initial state), with the gripper held open; the policy then
  resumes. The realised displacement is recorded. Comparing success against
  displacement for two checkpoints measures how wide a deviation each corrects.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/gripper-mujoco-hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from interaction_vla.representation_study.libero.acquisition import _evaluation_command
from interaction_vla.representation_study.libero.annotation import relative_pose_9d

PUSH_SEED = 20261006


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def push_action(task: int, state_id: int, magnitude: float) -> np.ndarray:
    """Horizontal translation action, direction fixed per (task, initial state); gripper open."""
    angle = np.random.default_rng(PUSH_SEED + 1000 * task + state_id).uniform(0.0, 2.0 * np.pi)
    return np.asarray([magnitude * np.cos(angle), magnitude * np.sin(angle), 0, 0, 0, 0, -1.0])


def _run_task(args: argparse.Namespace) -> None:
    from lerobot.envs.libero import LiberoEnv
    from interaction_vla.representation_study.libero import capability_events
    from interaction_vla.representation_study.libero.runtime import LiberoOffscreenSimulator

    task = args.task[0]
    episodes, live = [], {}

    def finish(env) -> None:
        item = live.pop(id(env), None)
        if item is not None:
            item.pop("adapter")
            episodes.append(item)
            _write(args.output / "entry_metadata.json", {"push_steps": args.push_steps, "episodes": episodes})

    original_install = capability_events.install_libero_event_recorder

    def install(**kwargs):
        real_reset, real_step, real_close = LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close

        def reset(env, *call_args, **call_kwargs):
            finish(env)
            state_id = int(env.init_state_id) % len(env._init_states)  # same ID the recorder reports
            result = real_reset(env, *call_args, **call_kwargs)
            live[id(env)] = {"initial_state_id": state_id, "steps": 0, "entry_step": None,
                             "entry_pose9d": None, "entry_distance_m": None, "entry_aperture": None,
                             "entry_gripper_xyz": None,
                             "pushed_steps": 0, "push_displacement_m": None, "post_push_pose9d": None,
                             "adapter": LiberoOffscreenSimulator.from_live_env(
                                 env._env, suite="libero_spatial", task_id=int(env.task_id),
                                 task_name=str(env.task), language=str(env.task_description))}
            return result

        def step(env, action):
            item = live[id(env)]
            pushing = item["entry_step"] is not None and item["pushed_steps"] < args.push_steps
            if pushing:
                action = push_action(task, item["initial_state_id"], args.push_action).astype(
                    np.asarray(action).dtype)
                item["pushed_steps"] += 1
            result = real_step(env, action)
            item["steps"] += 1
            frame = item["adapter"]._privileged_frame()
            if item["entry_step"] is None and frame.gripper_target_surface_distance <= args.approach_distance:
                item.update(entry_step=item["steps"], entry_distance_m=frame.gripper_target_surface_distance,
                            entry_pose9d=list(relative_pose_9d(frame.gripper_pose, frame.target_pose)),
                            entry_aperture=frame.gripper_aperture,
                            entry_gripper_xyz=frame.gripper_pose[:3, 3].tolist())
            if pushing and item["pushed_steps"] == args.push_steps:
                item["push_displacement_m"] = float(np.linalg.norm(
                    frame.gripper_pose[:3, 3] - np.asarray(item["entry_gripper_xyz"])))
                item["post_push_pose9d"] = list(relative_pose_9d(frame.gripper_pose, frame.target_pose))
            return result

        def close(env):
            finish(env)
            return real_close(env)

        LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close = reset, step, close
        original_install(**kwargs)

    capability_events.install_libero_event_recorder = install
    forwarded = list(_evaluation_command(args.checkpoint, task, args.output, args.initial_state_offset,
                                         args.episodes, args.initial_state_count))
    separator = forwarded.index("--")
    forwarded[separator:separator] = ["--policy-noise-seed-base", "2057736129"]
    sys.argv = [sys.argv[0], *forwarded[5:]]
    capability_events.main()


def summarize(output: Path, tasks) -> dict:
    rows = []
    for task in tasks:
        root = output / f"task{task}"
        events = {int(e["initial_state_id"]): e
                  for e in json.loads((root / "physical_events.json").read_text())["episodes"]}
        for meta in json.loads((root / "entry_metadata.json").read_text())["episodes"]:
            event = events[int(meta["initial_state_id"])]
            rows.append({"task": task, **meta, "success": bool(event["success"]),
                         "stable_grasp": bool(event["stable_grasp"])})
    summary = {"rows": rows, "success": sum(r["success"] for r in rows), "episodes": len(rows),
               "entered_window": sum(r["entry_step"] is not None for r in rows),
               "by_task": {str(t): sum(r["success"] for r in rows if r["task"] == t) for t in tasks}}
    _write(output / "summary.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=int, action="append", required=True)
    p.add_argument("--initial-state-offset", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--initial-state-count", type=int, default=50)
    p.add_argument("--approach-distance", type=float, default=0.10)
    p.add_argument("--push-steps", type=int, default=0, help="0 records entry only (E3)")
    p.add_argument("--push-action", type=float, default=0.3, help="normalised translation action per push step")
    p.add_argument("--single-task", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.single_task:
        _run_task(args)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    _write(args.output / "contract.json", {
        "checkpoint": str(args.checkpoint.resolve()), "tasks": args.task,
        "initial_state_offset": args.initial_state_offset, "episodes": args.episodes,
        "initial_state_count": args.initial_state_count, "approach_distance_m": args.approach_distance,
        "push_steps": args.push_steps, "push_action": args.push_action, "push_seed": PUSH_SEED,
        "pose_convention": "relative_pose_9d(gripper_pose, target_pose) = StateBank gripper_to_target",
        "seed_base": 2057736129})
    for task in args.task:
        print(f"RUN task={task}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--single-task",
                        "--checkpoint", str(args.checkpoint), "--output", str(args.output / f"task{task}"),
                        "--task", str(task), "--initial-state-offset", str(args.initial_state_offset),
                        "--episodes", str(args.episodes), "--initial-state-count", str(args.initial_state_count),
                        "--approach-distance", str(args.approach_distance),
                        "--push-steps", str(args.push_steps), "--push-action", str(args.push_action)],
                       check=True)
    print(json.dumps({k: v for k, v in summarize(args.output, args.task).items() if k != "rows"}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
