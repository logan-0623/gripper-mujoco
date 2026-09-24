"""Paired exploratory Contact-vs-control closed-loop rollouts."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/gripper-mujoco-hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from interaction_vla.representation_study.libero.acquisition import _evaluation_command, summarize_closed_loop
from interaction_vla.representation_study.libero.flow_trace import _tree_sha256, file_hash


def run(checkpoint: Path, candidates: Path, output: Path, *, tasks: list[int],
        initial_state_offset: int, episodes: int, initial_state_count: int, dose: float) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not tasks or episodes < 1 or initial_state_count < episodes or dose == 0:
        raise ValueError("invalid paired rollout contract")
    output.mkdir(parents=True)
    conditions = ["baseline", "contact_0", "matched_random_0"]
    plan = {
        "schema": "libero_paired_closed_loop_intervention_v1",
        "kind": "paired_closed_loop",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _tree_sha256(checkpoint),
        "candidate_sha256": file_hash(candidates),
        "conditions": conditions,
        "tasks": tasks,
        "dose": dose,
        "stages": list(range(10)),
        "initial_state_offset": initial_state_offset,
        "initial_state_count": initial_state_count,
        "episodes_per_task": episodes,
        "dry_run": False,
        "edit_mode": "suppress",
        "analysis_role": "exploratory_functional_use",
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    total = len(conditions) * len(tasks)
    completed = 0
    for condition in conditions:
        for task in tasks:
            destination = output / condition / f"task{task}"
            command = list(_evaluation_command(
                checkpoint, task, destination, initial_state_offset,
                episodes, initial_state_count))
            if condition != "baseline":
                index = command.index("interaction_vla.representation_study.libero.capability_events")
                command[index] = "interaction_vla.representation_study.libero.flow_intervention_eval"
                separator = command.index("--")
                options = [
                    "--exploratory", "--candidates", str(candidates),
                    "--candidate-id", condition, "--dose", str(dose),
                    "--edit-mode", "suppress" if condition == "contact_0" else "matched_suppress",
                    "--environment-phase", "contact", "--max-edited-chunks", "1",
                    "--allow-control",
                ]
                if condition == "matched_random_0":
                    options += ["--match-candidate-id", "contact_0"]
                for stage in range(10):
                    options += ["--edit-stage", str(stage)]
                command = command[:separator] + options + command[separator:]
            print(f"ROLLOUT {completed + 1}/{total} {condition} task={task}", flush=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
            completed += 1
            (output / "progress.json").write_text(json.dumps({
                "complete": False, "completed": completed, "total": total,
                "condition": condition, "task": task,
            }, indent=2) + "\n")
    report = summarize_closed_loop(output, output / "report.json")
    (output / "progress.json").write_text(json.dumps({
        "complete": True, "completed": total, "total": total,
        "report": str((output / "report.json").resolve()),
    }, indent=2) + "\n")
    print("ALL_DONE FUNCTIONAL_EXPLORATORY", flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=int, action="append", required=True)
    p.add_argument("--initial-state-offset", type=int, default=20)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--initial-state-count", type=int, default=50)
    p.add_argument("--dose", type=float, default=0.5)
    args = p.parse_args()
    values = vars(args)
    values["tasks"] = values.pop("task")
    run(**values)


if __name__ == "__main__":
    main()
