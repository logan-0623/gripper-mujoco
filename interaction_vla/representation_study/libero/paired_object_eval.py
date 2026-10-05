"""Paired LIBERO Object evaluation with one scene and two synchronized goals.

Run on the existing Linux/CUDA LeRobot environment. The A and B BDDL files
share every scene and init-state declaration; only language, goal, and the
object-of-interest metadata differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


TARGETS = {"A": ("alphabet_soup_1", "alphabet soup"),
           "B": ("cream_cheese_1", "cream cheese")}


def _replace_once(source: str, pattern: str, replacement: str) -> str:
    result, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected one occurrence of {pattern!r}; found {count}")
    return result


def _paired_bddl(source: str, target: str, label: str) -> str:
    if "alphabet_soup_1 - alphabet_soup" not in source or "cream_cheese_1 - cream_cheese" not in source:
        raise ValueError("source LIBERO Object task must contain both paired objects")
    source = _replace_once(source, r"(?m)^(\s*\(:language ).*(\))$",
                           rf"\g<1>Pick the {label} and place it in the basket\g<2>")
    source = _replace_once(source, r"\(In alphabet_soup_1 basket_1_contain_region\)",
                           f"(In {target} basket_1_contain_region)")
    source = _replace_once(source, r"(\(:obj_of_interest\s+)alphabet_soup_1(\s+basket_1\s+\))",
                           rf"\g<1>{target}\g<2>")
    return source


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def _source_bddl() -> Path:
    from libero.libero import benchmark, get_libero_path

    task = benchmark.get_benchmark_dict()["libero_object"]().get_task(0)
    return Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file


def _run_branch(args: argparse.Namespace) -> None:
    from lerobot.envs.libero import LiberoEnv

    original_init = LiberoEnv.__init__
    bddl = (args.output / f"{args.branch}.bddl").resolve()
    target, label = TARGETS[args.branch]

    def paired_init(env, *call_args, **call_kwargs):
        original_init(env, *call_args, **call_kwargs)
        if env.task_id != 0:
            raise ValueError("paired Object evaluation requires task_id=0")
        env._task_bddl_file = str(bddl)
        env.task_description = f"Pick the {label} and place it in the basket"

    LiberoEnv.__init__ = paired_init

    from . import capability_events

    camera_map = json.dumps({"agentview_image": "camera1",
                             "robot0_eye_in_hand_image": "camera2"}, separators=(",", ":"))
    sys.argv = [sys.argv[0],
                "--events-output", str(args.output / args.branch / "physical_events.json"),
                "--suite", "libero_object",
                "--initial-state-offset", str(args.start_state),
                "--record-frame-trace", "--record-object-selection",
                "--approach-distance-m", str(args.approach_distance_m),
                "--policy-noise-seed-base", str(args.policy_seed),
                "--rendered-episodes", str(args.episodes),
                "--",
                f"--policy.path={args.checkpoint}", "--policy.device=cuda",
                "--policy.use_amp=false", "--policy.n_action_steps=10",
                "--policy.num_steps=10", "--policy.empty_cameras=1",
                "--env.type=libero", "--env.task=libero_object", "--env.task_ids=[0]",
                "--env.obs_type=pixels_agent_pos", "--env.init_states=true",
                "--env.fps=30", "--env.max_parallel_tasks=1",
                f"--env.camera_name_mapping={camera_map}",
                f"--eval.n_episodes={args.episodes}", "--eval.batch_size=1",
                "--eval.use_async_envs=false", "--eval.recording=false",
                f"--seed={args.seed}",
                f"--output_dir={args.output / args.branch / 'eval'}"]
    capability_events.main()


def _compare_pairs(output: Path, episodes: int) -> dict[str, object]:
    rows = {}
    for branch in TARGETS:
        path = output / branch / "physical_events.json"
        rows[branch] = json.loads(path.read_text(encoding="utf-8"))["episodes"]
        if len(rows[branch]) != episodes:
            raise ValueError(f"branch {branch} recorded {len(rows[branch])}/{episodes} episodes")
    pairs = []
    for a, b in zip(rows["A"], rows["B"], strict=True):
        if a["initial_state_id"] != b["initial_state_id"]:
            raise ValueError("paired runs used different LIBERO initial-state IDs")
        for branch, row in (("A", a), ("B", b)):
            target, label = TARGETS[branch]
            if row["target"] != target or row["language"] != f"Pick the {label} and place it in the basket":
                raise ValueError(f"branch {branch} policy text and simulator goal disagree")
        qa, qb = a["initial_qpos"], b["initial_qpos"]
        if len(qa) != len(qb):
            raise ValueError("paired MuJoCo state dimensions differ")
        max_qpos_error = max(abs(x - y) for x, y in zip(qa, qb, strict=True))
        if max_qpos_error > 1e-6:
            raise ValueError(f"paired initial qpos differ by {max_qpos_error:.6g}; scene is not paired")
        pairs.append({"initial_state_id": a["initial_state_id"],
                      "max_initial_qpos_error": max_qpos_error,
                      "A": {key: a[key] for key in ("target", "first_approach", "first_contact",
                                                    "stable_grasp", "supported_lift", "success")},
                      "B": {key: b[key] for key in ("target", "first_approach", "first_contact",
                                                    "stable_grasp", "supported_lift", "success")}})
    result = {"schema": "libero_object_paired_pilot_v1", "pairs": pairs,
              "max_initial_qpos_error": max(p["max_initial_qpos_error"] for p in pairs)}
    (output / "paired_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-state", type=int, default=10)
    parser.add_argument("--approach-distance-m", type=float, default=0.03)
    parser.add_argument("--policy-seed", type=int, default=2057736129)
    parser.add_argument("--seed", type=int, default=2057736129)
    parser.add_argument("--branch", choices=TARGETS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.branch:
        _run_branch(args)
        return
    if not 1 <= args.episodes <= 10 or args.start_state < 0:
        raise ValueError("choose 1–10 paired episodes and a non-negative starting state")
    if not (args.checkpoint / "config.json").is_file():
        raise FileNotFoundError("checkpoint must be a local, complete policy directory")
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite paired evaluation: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    source_path = _source_bddl()
    source = source_path.read_text(encoding="utf-8")
    for branch, (target, label) in TARGETS.items():
        (args.output / f"{branch}.bddl").write_text(
            _paired_bddl(source, target, label), encoding="utf-8"
        )
    contract = {"checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": _checkpoint_hash(args.checkpoint),
                "source_bddl": str(source_path),
                "source_bddl_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "bddl_sha256": {branch: hashlib.sha256((args.output / f"{branch}.bddl").read_bytes()).hexdigest()
                                for branch in TARGETS},
                "episodes": args.episodes, "start_state": args.start_state,
                "policy_seed": args.policy_seed, "seed": args.seed,
                "approach_distance_m": args.approach_distance_m,
                "evaluation": {"suite": "libero_object", "task_id": 0,
                               "action_steps": 10, "denoising_steps": 10,
                               "empty_cameras": 1, "batch_size": 1,
                               "async_envs": False, "max_steps": 280,
                               "control_mode": "relative"},
                "branches": TARGETS}
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    for branch in TARGETS:
        command = [sys.executable, "-m", f"{__package__}.paired_object_eval",
                   "--checkpoint", str(args.checkpoint),
                   "--output", str(args.output), "--episodes", str(args.episodes),
                   "--start-state", str(args.start_state), "--policy-seed", str(args.policy_seed),
                   "--seed", str(args.seed), "--approach-distance-m", str(args.approach_distance_m),
                   "--branch", branch]
        subprocess.run(command, check=True)
    result = _compare_pairs(args.output, args.episodes)
    print(f"Paired LIBERO Object results: {args.output / 'paired_summary.json'}; "
          f"max initial qpos error={result['max_initial_qpos_error']:.6g}")


if __name__ == "__main__":
    main()
