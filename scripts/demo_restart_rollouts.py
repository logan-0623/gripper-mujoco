"""Q2 step 2: restart the policy from demonstration states partway through a task.

For each LIBERO Spatial task, simulator states are taken from the original
demonstrations at fixed fractions of their length (0, 0.25, 0.5, 0.75). The
benchmark reset runs as usual, then the simulator is set to the demonstration
state and the policy runs freely under the native evaluation contract (same
command, seed base, action chunking and step budget as the capability
timeline). If an early checkpoint fails mainly because small errors compound,
its gap to a late checkpoint should shrink as the restart point moves later;
a per-step deficit keeps the gap even from late restarts.

The restart hook sits underneath the physical-event recorder, so recorded
events begin at the restored state. No settling steps are taken after the
restore: the default LeRobot wait action opens the gripper and would drop a
held object.
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

FRACTIONS = (0.0, 0.25, 0.5, 0.75)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def restart_points(raw_root: Path, task: int, demos: int, fractions=FRACTIONS):
    """[(demo_key, frame, fraction, flattened_state)], demo-major order."""
    import h5py
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    path = raw_root / suite.get_task_demonstration(task)
    points = []
    with h5py.File(path, "r") as handle:
        keys = sorted(handle["data"], key=lambda key: int(key.split("_")[-1]))[:demos]
        for key in keys:
            states = np.asarray(handle["data"][key]["states"])
            for fraction in fractions:
                frame = int(round(fraction * (len(states) - 1)))
                points.append((key, frame, fraction, states[frame]))
    return path, points


def _run_task(args: argparse.Namespace) -> None:
    from lerobot.envs.libero import LiberoEnv
    from interaction_vla.representation_study.libero import capability_events

    path, points = restart_points(args.raw_root, args.task[0], args.demos)
    _write(args.output / "restart_points.json", {
        "demonstration_file": str(path),
        "points": [{"initial_state_id": i, "demo": key, "frame": frame, "fraction": fraction}
                   for i, (key, frame, fraction, _) in enumerate(points)]})
    original_install = capability_events.install_libero_event_recorder

    def install(**kwargs):
        real_reset = LiberoEnv.reset

        def reset(env, *call_args, **call_kwargs):
            index = int(env.init_state_id)  # set by the recorder before it calls this reset
            real_reset(env, *call_args, **call_kwargs)
            raw_obs = env._env.set_init_state(points[index % len(points)][3])
            return env._format_raw_obs(raw_obs), {"is_success": False}

        LiberoEnv.reset = reset
        original_install(**kwargs)

    capability_events.install_libero_event_recorder = install
    forwarded = list(_evaluation_command(args.checkpoint, args.task[0], args.output, 0, len(points)))
    task_arg = forwarded.index(f"--env.task_ids=[{args.task[0]}]")
    forwarded[task_arg:task_arg + 1] = ["--env.task_ids", str(args.task[0])]
    separator = forwarded.index("--")
    forwarded[separator:separator] = ["--policy-noise-seed-base", "2057736129"]
    sys.argv = [sys.argv[0], *forwarded[5:]]
    capability_events.main()


def summarize(output: Path, tasks) -> dict:
    rows = []
    for task in tasks:
        root = output / f"task{task}"
        points = json.loads((root / "restart_points.json").read_text())["points"]
        events = json.loads((root / "physical_events.json").read_text())["episodes"]
        by_id = {p["initial_state_id"]: p for p in points}
        for event in events:
            point = by_id[int(event["initial_state_id"])]
            rows.append({"task": task, **point, "success": bool(event["success"]),
                         "stable_grasp": bool(event["stable_grasp"]),
                         "unintended_drop": bool(event["unintended_drop"])})
    by_fraction = {}
    for fraction in sorted({r["fraction"] for r in rows}):
        selected = [r for r in rows if r["fraction"] == fraction]
        by_fraction[str(fraction)] = {"success": sum(r["success"] for r in selected), "episodes": len(selected),
                                      "by_task": {str(t): sum(r["success"] for r in selected if r["task"] == t)
                                                  for t in tasks}}
    summary = {"rows": rows, "by_fraction": by_fraction}
    _write(output / "summary.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=int, action="append", required=True)
    p.add_argument("--demos", type=int, default=10, help="demonstrations per task")
    p.add_argument("--raw-root", type=Path, default=Path("/root/gripper-mujoco/data/libero/raw"))
    p.add_argument("--single-task", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.single_task:
        _run_task(args)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    _write(args.output / "contract.json", {
        "checkpoint": str(args.checkpoint.resolve()), "tasks": args.task, "demos_per_task": args.demos,
        "fractions": list(FRACTIONS), "raw_root": str(args.raw_root), "seed_base": 2057736129,
        "n_action_steps": 10, "denoising_steps": 10, "restore": "benchmark reset, then set_init_state(demo)"})
    for task in args.task:
        print(f"RUN task={task}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--single-task",
                        "--checkpoint", str(args.checkpoint), "--output", str(args.output / f"task{task}"),
                        "--task", str(task), "--demos", str(args.demos), "--raw-root", str(args.raw_root)],
                       check=True)
    print(json.dumps(summarize(args.output, args.task)["by_fraction"], indent=2), flush=True)


if __name__ == "__main__":
    main()
