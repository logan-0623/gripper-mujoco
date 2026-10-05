"""Paired LIBERO rollout with a brief gripper-open actuator perturbation."""

from __future__ import annotations

import argparse
import hashlib
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
from interaction_vla.representation_study.libero.annotation import AnnotationThresholds, _stable_grasp
from interaction_vla.representation_study.libero.runtime import LiberoOffscreenSimulator


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _run_branch(args: argparse.Namespace) -> None:
    from lerobot.envs.libero import LiberoEnv
    from interaction_vla.representation_study.libero import capability_events

    original_install = capability_events.install_libero_event_recorder
    rows: list[dict[str, object]] = []
    live: dict[int, dict[str, object]] = {}
    thresholds = AnnotationThresholds()

    def install(**kwargs):
        original_install(**kwargs)
        original_reset, original_step, original_close = (
            LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close
        )

        def finish(env):
            item = live.pop(id(env), None)
            if item is not None:
                rows.append({
                    key: value for key, value in item.items()
                    if key not in {"adapter", "frames", "prefix_actions", "initial_z"}
                })
                _write(args.metadata, {"mode": args.mode, "episodes": rows})

        def reset(env, *call_args, **call_kwargs):
            finish(env)
            result = original_reset(env, *call_args, **call_kwargs)
            adapter = LiberoOffscreenSimulator.from_live_env(
                env._env, suite="libero_spatial", task_id=int(env.task_id),
                task_name=str(env.task), language=str(env.task_description),
            )
            frame = adapter._privileged_frame()
            live[id(env)] = {
                "adapter": adapter, "frames": [frame], "prefix_actions": [],
                "initial_z": float(frame.target_pose[2, 3]),
                "initial_state_id": int(
                    (env.init_state_id - env._reset_stride) % len(env._init_states)
                ),
                "trigger_step": None, "pre_state_sha256": None,
                "prefix_actions_sha256": None, "open_steps": 0,
                "remaining_open_steps": 0, "success": False,
                "steps": 0,
            }
            return result

        def step(env, action):
            item = live[id(env)]
            frames = item["frames"]
            adapter = item["adapter"]
            step_index = int(item["steps"])
            if item["trigger_step"] is None and (
                _stable_grasp(frames, len(frames) - 1, thresholds) is True
                and float(frames[-1].target_pose[2, 3]) - item["initial_z"] >= 0.02
                and not frames[-1].goal_satisfied
            ):
                state = adapter.get_state_flattened()
                prefix = np.asarray(item["prefix_actions"], dtype=np.float64)
                item["trigger_step"] = step_index
                item["pre_state_sha256"] = hashlib.sha256(state.tobytes()).hexdigest()
                item["prefix_actions_sha256"] = hashlib.sha256(prefix.tobytes()).hexdigest()
                item["remaining_open_steps"] = args.open_steps if args.mode == "open" else 0
            requested = np.asarray(action, dtype=np.float64).reshape(-1)
            if item["trigger_step"] is None:
                item["prefix_actions"].append(requested.tolist())
            applied = requested.copy()
            if item["remaining_open_steps"]:
                applied[-1] = -1.0
                item["remaining_open_steps"] -= 1
                item["open_steps"] += 1
            result = original_step(env, applied)
            item["steps"] = step_index + 1
            item["success"] = bool(result[4].get("is_success", False))
            item["frames"].append(adapter._privileged_frame())
            if result[2] or result[3]:
                finish(env)
            return result

        def close(env):
            finish(env)
            return original_close(env)

        LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close = reset, step, close

    capability_events.install_libero_event_recorder = install
    forwarded = list(_evaluation_command(
        args.checkpoint, args.task, args.output, args.initial_state_offset,
        args.episodes, initial_state_count=50,
    ))
    task_arg = forwarded.index(f"--env.task_ids=[{args.task}]")
    forwarded[task_arg:task_arg + 1] = ["--env.task_ids", str(args.task)]
    separator = forwarded.index("--")
    forwarded[separator:separator] = [
        "--record-frame-trace", "--policy-noise-seed-base", "2057736129",
    ]
    sys.argv = [sys.argv[0], *forwarded[5:]]
    capability_events.main()


def _run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.episodes < 1 or args.open_steps < 1:
        raise ValueError("episodes and open_steps must be positive")
    if not (args.checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(args.checkpoint / "model.safetensors")
    args.output.mkdir(parents=True)
    _write(args.output / "contract.json", {
        "checkpoint": str(args.checkpoint.resolve()),
        "tasks": args.task, "initial_state_offset": args.initial_state_offset,
        "episodes_per_task": args.episodes, "open_steps": args.open_steps,
        "n_action_steps": 10, "denoising_steps": 10, "seed": 2057736129,
        "intervention": "override gripper command to -1 after stable grasp and 2cm lift",
        "role": "feasibility_pilot",
    })
    for task in args.task:
        for mode in ("baseline", "open"):
            output = args.output / f"task{task}" / mode
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--checkpoint", str(args.checkpoint), "--output", str(output),
                "--task", str(task), "--initial-state-offset",
                str(args.initial_state_offset), "--episodes", str(args.episodes),
                "--open-steps", str(args.open_steps), "--mode", mode,
                "--metadata", str(output / "branch_metadata.json"),
            ]
            print(f"RUN task={task} mode={mode}", flush=True)
            subprocess.run(command, check=True)
    pairs = []
    for task in args.task:
        items = {}
        for mode in ("baseline", "open"):
            root = args.output / f"task{task}" / mode
            metadata = json.loads((root / "branch_metadata.json").read_text())["episodes"]
            events = json.loads((root / "physical_events.json").read_text())["episodes"]
            items[mode] = {row["initial_state_id"]: (row, event)
                           for row, event in zip(metadata, events, strict=True)}
        if items["baseline"].keys() != items["open"].keys():
            raise ValueError(f"task {task}: initial states differ")
        for state_id in sorted(items["baseline"]):
            base, base_event = items["baseline"][state_id]
            edit, edit_event = items["open"][state_id]
            matched = (
                base["trigger_step"] is not None
                and base["trigger_step"] == edit["trigger_step"]
                and base["pre_state_sha256"] == edit["pre_state_sha256"]
                and base["prefix_actions_sha256"] == edit["prefix_actions_sha256"]
            )
            pairs.append({
                "task": task, "initial_state_id": state_id,
                "pre_event_matched": matched,
                "baseline": {**base, "event_success": base_event["success"],
                             "stable_grasp": base_event["stable_grasp"],
                             "drop": base_event["unintended_drop"],
                             "recovered": base_event["recovered_after_drop"]},
                "open": {**edit, "event_success": edit_event["success"],
                         "stable_grasp": edit_event["stable_grasp"],
                         "drop": edit_event["unintended_drop"],
                         "recovered": edit_event["recovered_after_drop"]},
            })
    _write(args.output / "paired_summary.json", {"pairs": pairs})
    print(f"SUMMARY {args.output / 'paired_summary.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=int, action="append", required=True)
    parser.add_argument("--initial-state-offset", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--open-steps", type=int, default=8)
    parser.add_argument("--mode", choices=("baseline", "open"))
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    if args.mode:
        if len(args.task) != 1 or args.metadata is None:
            raise ValueError("branch needs one task and metadata path")
        _run_branch(args)
    else:
        _run(args)


if __name__ == "__main__":
    main()
