"""Restart LIBERO Spatial episodes from demonstration states (Q2 step 2 and E1).

The benchmark reset runs as usual, then the simulator is set to a state taken
from an original demonstration and the episode continues under the native
evaluation contract (same command, seed base, action chunking and step budget
as the capability timeline).

Restart points:
* ``fractions``: fixed fractions of demonstration length (0, 0.2, 0.4, 0.6).
* ``events``: aligned to the StateBank's labelled interaction events for its
  annotated demonstrations: 30, 15 and 5 frames before first gripper-target
  contact, first contact, first stable grasp, and 10 frames after it.

Modes:
* ``policy``: the checkpoint acts from the restored state.
* ``demo_replay``: the policy's actions are replaced by the demonstration's own
  remaining actions. This measures restore fidelity: a restored point that the
  demonstration itself cannot finish from says more about the restore than about
  any policy. After the demonstration ends, the last gripper command is held.

The restart hook sits underneath the physical-event recorder, so recorded
events begin at the restored state. No settling steps are taken after the
restore: the default LeRobot wait action opens the gripper and would drop a
held object. Restarts whose demonstration state already satisfies the task goal
are recorded and excluded from the summary.
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

FRACTIONS = (0.0, 0.2, 0.4, 0.6)
EVENT_OFFSETS = (("contact-30", "contact", -30), ("contact-15", "contact", -15), ("contact-5", "contact", -5),
                 ("contact", "contact", 0), ("grasp", "grasp", 0), ("grasp+10", "grasp", 10))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fraction_frames(length: int, fractions=FRACTIONS) -> list[tuple[str, int]]:
    return [(str(fraction), int(round(fraction * (length - 1)))) for fraction in fractions]


def event_frames(records) -> list[tuple[str, int]]:
    """Restart frames for one annotated demonstration (records of a single episode)."""
    rows = sorted(records, key=lambda row: row.frame_index)
    contact = next(r.frame_index for r in rows if r.labels.contact and r.labels.contact.gripper_target)
    grasp = next(r.frame_index for r in rows if r.labels.stable_grasp)
    anchors, last = {"contact": contact, "grasp": grasp}, rows[-1].frame_index
    by_frame = {r.frame_index: r for r in rows}
    return [(label, by_frame[min(max(anchors[anchor] + offset, 0), last)].replay.simulator_state_index)
            for label, anchor, offset in EVENT_OFFSETS]


def restart_points(raw_root: Path, task: int, *, demos: int, points: str, bank: Path | None):
    """Demonstration file and [{demo, frame, label, state, actions}] in demo-major order."""
    import h5py
    from libero.libero import benchmark

    path = raw_root / benchmark.get_benchmark_dict()["libero_spatial"]().get_task_demonstration(task)
    if points == "events":
        from interaction_vla.representation_study.libero.state_bank import load_state_bank

        records, _, _, _ = load_state_bank(bank)
        episodes: dict[str, list] = {}
        for record in records:
            if record.suite == "libero_spatial" and record.task_id == task:
                episodes.setdefault(record.source_episode_id, []).append(record)
        plan = {key: event_frames(rows) for key, rows in sorted(episodes.items())}
    result = []
    with h5py.File(path, "r") as handle:
        if points == "fractions":
            keys = sorted(handle["data"], key=lambda key: int(key.split("_")[-1]))[:demos]
            plan = {key: fraction_frames(len(handle["data"][key]["states"])) for key in keys}
        for key, frames in plan.items():
            states = np.asarray(handle["data"][key]["states"])
            actions = np.asarray(handle["data"][key]["actions"])
            for label, frame in frames:
                result.append({"demo": key, "frame": frame, "label": label,
                               "state": states[frame], "actions": actions[frame:]})
    return path, result


def _run_task(args: argparse.Namespace) -> None:
    from lerobot.envs.libero import LiberoEnv
    from interaction_vla.representation_study.libero import capability_events

    path, points = restart_points(args.raw_root, args.task[0], demos=args.demos, points=args.points,
                                  bank=args.bank)
    _write(args.output / "restart_points.json", {
        "demonstration_file": str(path), "mode": args.mode, "points_source": args.points,
        "points": [{"initial_state_id": i, "demo": p["demo"], "frame": p["frame"], "label": p["label"]}
                   for i, p in enumerate(points)]})
    original_install = capability_events.install_libero_event_recorder
    already_done: dict[int, bool] = {}
    replay: dict[int, list] = {}

    def install(**kwargs):
        real_reset, real_step = LiberoEnv.reset, LiberoEnv.step

        def reset(env, *call_args, **call_kwargs):
            index = int(env.init_state_id) % len(points)  # set by the recorder before this reset
            real_reset(env, *call_args, **call_kwargs)
            raw_obs = env._env.set_init_state(points[index]["state"])
            already_done[index] = bool(env._env.check_success())
            _write(args.output / "restore_checks.json", {"goal_satisfied_at_restore": already_done})
            replay[id(env)] = [index, 0]
            return env._format_raw_obs(raw_obs), {"is_success": False}

        def step(env, action):
            if args.mode == "demo_replay":
                index, cursor = replay[id(env)]
                demo = points[index]["actions"]
                if cursor < len(demo):
                    action = demo[cursor].astype(np.asarray(action).dtype)
                else:
                    action = np.zeros_like(np.asarray(action))
                    action[-1] = demo[-1][-1]
                replay[id(env)][1] = cursor + 1
            return real_step(env, action)

        LiberoEnv.reset, LiberoEnv.step = reset, step
        original_install(**kwargs)

    capability_events.install_libero_event_recorder = install
    forwarded = list(_evaluation_command(args.checkpoint, args.task[0], args.output, 0, len(points)))
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
        done = json.loads((root / "restore_checks.json").read_text())["goal_satisfied_at_restore"]
        by_id = {p["initial_state_id"]: p for p in points}
        for event in events:
            point = by_id[int(event["initial_state_id"])]
            rows.append({"task": task, **point, "goal_satisfied_at_restore": done[str(point["initial_state_id"])],
                         "success": bool(event["success"]), "steps": int(event["steps"]),
                         "stable_grasp": bool(event["stable_grasp"]),
                         "unintended_drop": bool(event["unintended_drop"])})
    labels = list(dict.fromkeys(r["label"] for r in rows))
    by_label = {}
    for label in labels:
        selected = [r for r in rows if r["label"] == label and not r["goal_satisfied_at_restore"]]
        by_label[label] = {"success": sum(r["success"] for r in selected), "episodes": len(selected),
                           "excluded_goal_already_satisfied": sum(
                               r["label"] == label and r["goal_satisfied_at_restore"] for r in rows),
                           "by_task": {str(t): sum(r["success"] for r in selected if r["task"] == t)
                                       for t in tasks}}
    summary = {"rows": rows, "by_label": by_label}
    _write(output / "summary.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="policy checkpoint; in demo_replay mode it is loaded but its actions are ignored")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=int, action="append", required=True)
    p.add_argument("--points", choices=("fractions", "events"), default="fractions")
    p.add_argument("--mode", choices=("policy", "demo_replay"), default="policy")
    p.add_argument("--demos", type=int, default=10, help="demonstrations per task (fractions only)")
    p.add_argument("--bank", type=Path, default=Path(
        "/root/gripper-mujoco/outputs/representation_study/libero_smolvla/state_bank"),
        help="StateBank with labelled demonstrations (events only)")
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
        "checkpoint": str(args.checkpoint.resolve()), "tasks": args.task, "points": args.points,
        "mode": args.mode, "demos_per_task": args.demos if args.points == "fractions" else "all annotated",
        "fractions": list(FRACTIONS), "event_offsets": [list(e) for e in EVENT_OFFSETS],
        "raw_root": str(args.raw_root), "bank": str(args.bank), "seed_base": 2057736129,
        "n_action_steps": 10, "denoising_steps": 10, "restore": "benchmark reset, then set_init_state(demo)"})
    for task in args.task:
        print(f"RUN task={task}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--single-task",
                        "--checkpoint", str(args.checkpoint), "--output", str(args.output / f"task{task}"),
                        "--task", str(task), "--demos", str(args.demos), "--points", args.points,
                        "--mode", args.mode, "--bank", str(args.bank), "--raw-root", str(args.raw_root)],
                       check=True)
    print(json.dumps(summarize(args.output, args.task)["by_label"], indent=2), flush=True)


if __name__ == "__main__":
    main()
