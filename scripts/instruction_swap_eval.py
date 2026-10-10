"""How much of the 5k->25k capability is layout -> trajectory rather than language?

Re-evaluates each lineage checkpoint on LIBERO Spatial with the instruction
handed to the policy replaced; nothing is trained. Conditions:
* ``correct``: the task's own instruction (control, rerun under this harness).
* ``empty``: the empty string.
* ``gibberish``: random lowercase words with the same word lengths, fixed per task.
* ``other``: the instruction of another Spatial task (task k gets task
  (k+1) mod 4 of 0-3), i.e. the same verb and goal but a different bowl.

The override sets ``LiberoEnv.task_description`` after construction, which is
what lerobot_eval puts in ``observation["task"]``; the event recorder logs the
same string as ``language``. Evaluation otherwise matches E1/E2 (command, seed,
per-episode policy-noise seed, initial states), so episodes pair across
conditions and checkpoints by task and initial-state ID.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/gripper-mujoco-hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from interaction_vla.representation_study.libero.acquisition import _evaluation_command

CONDITIONS = ("correct", "empty", "gibberish", "other")
NOISE_SEED_BASE = 2057736129
LINEAGE = Path("/root/autodl-tmp/smolvla-official-reproduction-v2/full_25k_seed1000/checkpoints")


def instruction(condition: str, task: int, language: dict[int, str], tasks: list[int]) -> str:
    own = language[task]
    if condition == "correct":
        return own
    if condition == "empty":
        return ""
    if condition == "gibberish":
        rng = random.Random(task)
        return " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in w) for w in own.split())
    return language[tasks[(tasks.index(task) + 1) % len(tasks)]]


def _install_trace(LiberoEnv, path: Path) -> None:
    """Per episode: end-effector path, gripper commands, and every object's start/end/max-height."""
    episodes, live = [], {}

    def objects(env):
        e = env._env.env
        return {n: e.sim.data.body_xpos[i].tolist() for n, i in e.obj_body_id.items()}

    def eef(env):
        e = env._env.env
        return e.sim.data.site_xpos[e.robots[0].eef_site_id].tolist()

    def finish(env):
        item = live.pop(id(env), None)
        if item is not None:
            item["final_objects"] = objects(env)
            episodes.append(item)
            path.write_text(json.dumps({"episodes": episodes}) + "\n")

    real_reset, real_step, real_close = LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close

    def reset(env, *a, **k):
        finish(env)
        state_id = int(env.init_state_id) % len(env._init_states)  # same ID the recorder reports
        result = real_reset(env, *a, **k)
        start = objects(env)
        live[id(env)] = {"task_id": int(env.task_id), "initial_state_id": state_id,
                         "prompt": env.task_description, "initial_objects": start,
                         "max_z": {n: p[2] for n, p in start.items()}, "eef": [eef(env)], "gripper": []}
        return result

    def step(env, action):
        result = real_step(env, action)
        item = live[id(env)]
        item["eef"].append([round(v, 4) for v in eef(env)])
        item["gripper"].append(round(float(action[-1]), 3))
        for n, p in objects(env).items():
            item["max_z"][n] = max(item["max_z"][n], p[2])
        return result

    def close(env):
        finish(env)
        return real_close(env)

    LiberoEnv.reset, LiberoEnv.step, LiberoEnv.close = reset, step, close


def _run_single(args: argparse.Namespace) -> None:
    from lerobot.envs.libero import LiberoEnv
    from interaction_vla.representation_study.libero import capability_events

    real_init = LiberoEnv.__init__

    def init(env, *a, **k):
        real_init(env, *a, **k)
        env.task_description = args.prompt

    LiberoEnv.__init__ = init
    if args.trace:
        _install_trace(LiberoEnv, args.output / "trace.json")
    forwarded = list(_evaluation_command(args.checkpoint, args.task, args.output, 0,
                                         args.episodes, args.initial_state_count))
    separator = forwarded.index("--")
    forwarded[separator:separator] = ["--policy-noise-seed-base", str(NOISE_SEED_BASE)]
    sys.argv = [sys.argv[0], *forwarded[5:]]
    capability_events.main()


def summarize(output: Path, steps: list[int], tasks: list[int]) -> dict:
    cells = {}
    for step in steps:
        for cond in CONDITIONS:
            for task in tasks:
                path = output / f"{step:06d}" / cond / f"task{task}" / "physical_events.json"
                if path.is_file():
                    for e in json.loads(path.read_text())["episodes"]:
                        cells[step, cond, task, int(e["initial_state_id"])] = e
    table = {}
    for step in steps:
        for cond in CONDITIONS:
            rows = [v for k, v in cells.items() if k[:2] == (step, cond)]
            if not rows:
                continue
            paired = [(bool(cells[(step, "correct") + k[2:]]["success"]), bool(v["success"]))
                      for k, v in cells.items() if k[:2] == (step, cond) and (step, "correct") + k[2:] in cells]
            table[f"{step // 1000}k/{cond}"] = {
                "success": sum(bool(r["success"]) for r in rows),
                "stable_grasp": sum(bool(r["stable_grasp"]) for r in rows),
                "contact": sum(bool(r["contact"]) for r in rows),
                "grasp_then_lift": sum(bool(r["stable_grasp"] and r["supported_lift"]) for r in rows),
                "grasp_then_drop": sum(bool(r["stable_grasp"] and r["drop_onset_step"] is not None) for r in rows),
                "normal_release": sum(bool(r["normal_release"]) for r in rows),
                "contact_onset_median": (sorted(on)[len(on) // 2] if (on := [r["contact_onset_step"] for r in rows
                                         if r["contact_onset_step"] is not None]) else None),
                "episodes": len(rows),
                "by_task": {t: sum(bool(v["success"]) for k, v in cells.items() if k[:3] == (step, cond, t))
                            for t in tasks},
                "paired_vs_correct": {"n": len(paired),
                                      "both": sum(a and b for a, b in paired),
                                      "lost": sum(a and not b for a, b in paired),
                                      "gained": sum(b and not a for a, b in paired)},
                "language": sorted({str(r.get("language")) for r in rows}),
            }
    (output / "summary.json").write_text(json.dumps(table, indent=2) + "\n")
    return table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step", type=int, action="append", help="checkpoint step (default: 5k-25k)")
    p.add_argument("--condition", choices=CONDITIONS, action="append")
    p.add_argument("--task", type=int, action="append")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--initial-state-count", type=int, default=50)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--trace", action="store_true", help="also log end-effector path and object positions")
    p.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--prompt", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.single:
        args.task = args.task[0]
        _run_single(args)
        return
    steps = args.step or [5000, 10000, 15000, 20000, 25000]
    tasks = args.task or [0, 1, 2, 3]
    conditions = args.condition or list(CONDITIONS)
    if not args.summarize_only:
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        language = {t: suite.get_task(t).language for t in tasks}
        jobs = []
        for step in steps:
            checkpoint = LINEAGE / f"{step:06d}" / "pretrained_model"
            if not (checkpoint / "model.safetensors").is_file():
                raise FileNotFoundError(checkpoint / "model.safetensors")
            for cond in conditions:
                for task in tasks:
                    out = args.output / f"{step:06d}" / cond / f"task{task}"
                    if (out / "physical_events.json").is_file() and (out / "eval_info.json").is_file():
                        continue
                    if out.exists():
                        raise FileExistsError(f"incomplete output, remove it first: {out}")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    jobs.append([sys.executable, "-W", "error::RuntimeWarning", str(Path(__file__).resolve()),
                                 "--single", "--output", str(out), "--checkpoint", str(checkpoint),
                                 "--task", str(task), "--prompt", instruction(cond, task, language, tasks),
                                 "--episodes", str(args.episodes),
                                 "--initial-state-count", str(args.initial_state_count)]
                                + (["--trace"] if args.trace else []))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "contract.json").write_text(json.dumps({
            "lineage": str(LINEAGE), "steps": steps, "tasks": tasks, "episodes": args.episodes,
            "initial_state_count": args.initial_state_count, "seed_base": NOISE_SEED_BASE,
            "prompts": {c: {t: instruction(c, t, language, tasks) for t in tasks} for c in CONDITIONS},
        }, indent=2) + "\n")

        def run(cmd):
            log = Path(cmd[cmd.index("--output") + 1]).with_suffix(".log")
            with log.open("w") as fh:
                code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
            print(("DONE " if code == 0 else f"FAIL({code}) ") + str(log), flush=True)
            return code

        print(f"{len(jobs)} jobs", flush=True)
        with ThreadPoolExecutor(args.workers) as pool:
            failed = sum(c != 0 for c in pool.map(run, jobs))
        if failed:
            raise SystemExit(f"{failed} jobs failed")
    for key, row in summarize(args.output, steps, tasks).items():
        print(key, {k: v for k, v in row.items() if k != "language"}, flush=True)


if __name__ == "__main__":
    main()
