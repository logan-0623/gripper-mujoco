"""E2: is the 10k->25k capability gap confined to the approach-to-grasp window?

A base and a donor checkpoint run in lockstep on every step of a closed-loop
LIBERO Spatial episode: both see the same observation and keep their own action
queues, and the executed action comes from the donor only inside the window.
Running both policies in every arm keeps random-number consumption identical
across arms, so arms differ only in which action is executed.

Window modes:
* ``never``: always execute the base action (base control).
* ``always``: always execute the donor action (donor control).
* ``grasp``: execute the donor action from the first step whose gripper-target
  surface distance is at most ``--approach-distance`` (0.10 m is about 15-20
  frames before first contact in the annotated demonstrations) until the target
  is stably grasped and lifted by ``--lift`` (0.02 m); the window then stays
  closed for the episode.

Evaluation otherwise matches the capability timeline (command, seed base,
initial states, step budget), so outcomes pair by task and initial-state ID.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/gripper-mujoco-hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from interaction_vla.representation_study.libero.acquisition import _evaluation_command
from interaction_vla.representation_study.libero.annotation import AnnotationThresholds, _stable_grasp
from interaction_vla.representation_study.libero.flow_trace import file_hash


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class GraspWindow:
    """Online window state from privileged simulator frames."""

    def __init__(self, mode: str, approach_distance: float, lift: float):
        self.mode, self.approach_distance, self.lift = mode, approach_distance, lift
        self.thresholds = AnnotationThresholds()
        self.reset(None)

    def reset(self, frame) -> None:
        self.frames, self.opened_at, self.closed_at, self.step = [], None, None, 0
        self.initial_z = None if frame is None else float(frame.target_pose[2, 3])
        if frame is not None:
            self._update(frame)

    def observe(self, frame) -> None:
        self.step += 1
        self._update(frame)

    def _update(self, frame) -> None:
        self.frames.append(frame)
        if self.opened_at is None and frame.gripper_target_surface_distance <= self.approach_distance:
            self.opened_at = self.step
        if (self.opened_at is not None and self.closed_at is None
                and _stable_grasp(self.frames, len(self.frames) - 1, self.thresholds) is True
                and float(frame.target_pose[2, 3]) - self.initial_z >= self.lift):
            self.closed_at = self.step

    @property
    def active(self) -> bool:
        if self.mode != "grasp":
            return self.mode == "always"
        return self.opened_at is not None and self.closed_at is None


def _run_task(args: argparse.Namespace) -> None:
    from lerobot.envs.libero import LiberoEnv
    import lerobot.scripts.lerobot_eval as lerobot_eval
    from interaction_vla.representation_study.libero import capability_events
    from interaction_vla.representation_study.libero.runtime import LiberoOffscreenSimulator

    window = GraspWindow(args.window, args.approach_distance, args.lift)
    episodes, live, adapters = [], {}, {}

    def finish(env) -> None:
        item = live.pop(id(env), None)
        if item is not None:
            episodes.append({**item, "window_opened_step": window.opened_at,
                             "window_closed_step": window.closed_at})
            _write(args.output / "window_metadata.json", {"window": args.window, "episodes": episodes})

    original_install = capability_events.install_libero_event_recorder

    def install(**kwargs):
        real_reset, real_step, real_close = LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close

        def reset(env, *call_args, **call_kwargs):
            finish(env)
            state_id = int(env.init_state_id) % len(env._init_states)  # same ID the recorder reports
            result = real_reset(env, *call_args, **call_kwargs)
            adapters[id(env)] = LiberoOffscreenSimulator.from_live_env(
                env._env, suite="libero_spatial", task_id=int(env.task_id),
                task_name=str(env.task), language=str(env.task_description))
            window.reset(adapters[id(env)]._privileged_frame())
            live[id(env)] = {"initial_state_id": state_id, "donor_steps": 0, "steps": 0}
            return result

        def step(env, action):
            result = real_step(env, action)
            window.observe(adapters[id(env)]._privileged_frame())
            return result

        def close(env):
            finish(env)
            return real_close(env)

        LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close = reset, step, close
        original_install(**kwargs)

    real_make_policy = lerobot_eval.make_policy

    def make_policy(cfg, **kwargs):
        base = real_make_policy(cfg=cfg, **kwargs)
        donor_cfg = copy.deepcopy(cfg)
        donor_cfg.pretrained_path = args.donor
        donor = real_make_policy(cfg=donor_cfg, **kwargs).eval()
        base_select, base_reset = base.select_action, base.reset

        def select_action(observation):
            own, other = base_select(observation), donor.select_action(observation)
            use_donor = window.active
            for item in live.values():
                item["steps"] += 1
                item["donor_steps"] += int(use_donor)
            return other if use_donor else own

        def reset():
            base_reset()
            donor.reset()

        base.select_action, base.reset = select_action, reset
        return base

    lerobot_eval.make_policy = make_policy
    capability_events.install_libero_event_recorder = install
    forwarded = list(_evaluation_command(args.base, args.task[0], args.output, args.initial_state_offset,
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
        for meta in json.loads((root / "window_metadata.json").read_text())["episodes"]:
            event = events[int(meta["initial_state_id"])]
            rows.append({"task": task, "initial_state_id": int(event["initial_state_id"]),
                         "success": bool(event["success"]), "stable_grasp": bool(event["stable_grasp"]),
                         "steps": int(meta["steps"]), "donor_steps": int(meta["donor_steps"]),
                         "window_opened_step": meta["window_opened_step"],
                         "window_closed_step": meta["window_closed_step"]})
    summary = {"rows": rows, "success": sum(r["success"] for r in rows), "episodes": len(rows),
               "by_task": {str(t): sum(r["success"] for r in rows if r["task"] == t) for t in tasks},
               "mean_donor_fraction": (sum(r["donor_steps"] for r in rows) / max(sum(r["steps"] for r in rows), 1))}
    _write(output / "summary.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True, help="checkpoint acting outside the window")
    p.add_argument("--donor", type=Path, required=True, help="checkpoint acting inside the window")
    p.add_argument("--window", choices=("never", "grasp", "always"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=int, action="append", required=True)
    p.add_argument("--initial-state-offset", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--initial-state-count", type=int, default=50)
    p.add_argument("--approach-distance", type=float, default=0.10)
    p.add_argument("--lift", type=float, default=0.02)
    p.add_argument("--single-task", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.single_task:
        _run_task(args)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    for checkpoint in (args.base, args.donor):
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint / "model.safetensors")
    processors = {name: [file_hash(c / name) for c in (args.base, args.donor)]
                  for name in sorted(p.name for p in args.base.glob("policy_*processor*"))}
    if any(a != b for a, b in processors.values()):
        raise ValueError("base and donor use different pre/post-processors; actions would not be comparable")
    args.output.mkdir(parents=True)
    _write(args.output / "contract.json", {
        "base": str(args.base.resolve()), "donor": str(args.donor.resolve()), "window": args.window,
        "tasks": args.task, "initial_state_offset": args.initial_state_offset, "episodes": args.episodes,
        "initial_state_count": args.initial_state_count, "approach_distance_m": args.approach_distance,
        "lift_m": args.lift, "seed_base": 2057736129, "lockstep": "both policies act every step",
        "shared_processors_sha256": processors})
    for task in args.task:
        print(f"RUN task={task}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--single-task",
                        "--base", str(args.base), "--donor", str(args.donor), "--window", args.window,
                        "--output", str(args.output / f"task{task}"), "--task", str(task),
                        "--initial-state-offset", str(args.initial_state_offset),
                        "--episodes", str(args.episodes), "--initial-state-count", str(args.initial_state_count),
                        "--approach-distance", str(args.approach_distance), "--lift", str(args.lift)],
                       check=True)
    print(json.dumps({k: v for k, v in summarize(args.output, args.task).items() if k != "rows"}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
