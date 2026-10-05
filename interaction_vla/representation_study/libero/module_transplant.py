"""Q1: where does the 5k->25k capability live? Cross-checkpoint module transplant.

Checkpoints of one lineage share initialisation and parameter names, so a
hybrid can take the VLM side (vision encoder, connector, text model, LM head,
and ``state_proj``, which writes the robot state into the VLM prefix) from one
checkpoint and the action side (``lm_expert`` incl. its cross-attention
projections, ``action_*`` projections) from another. Hybrids are evaluated with
the exact capability-timeline command, so outcomes pair with the timeline by
task and initial-state ID.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..state_bank.io import write_json_atomic
from .acquisition import _evaluation_command
from .flow_trace import file_hash
from .stage_patching import cluster_ci

SCHEMA = "smolvla_module_transplant_v1"
WEIGHTS = "model.safetensors"


def side(key: str) -> str:
    """'prefix' for parameters that build the VLM prefix, 'action' for the flow expert."""
    if key.startswith("model.vlm_with_expert.vlm.") or key.startswith("model.state_proj."):
        return "prefix"
    if key.startswith("model.vlm_with_expert.lm_expert.") or key.startswith("model.action_"):
        return "action"
    raise ValueError(f"unassigned parameter: {key}")


def build(prefix_checkpoint: Path, action_checkpoint: Path, output: Path) -> dict:
    from safetensors.torch import load_file, save_file

    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    prefix, action = load_file(str(prefix_checkpoint / WEIGHTS)), load_file(str(action_checkpoint / WEIGHTS))
    if prefix.keys() != action.keys() or any(prefix[k].shape != action[k].shape for k in prefix):
        raise ValueError("checkpoints do not share one parameter layout")
    merged, counts = {}, {"prefix": 0, "action": 0}
    for key in sorted(prefix):
        group = side(key)
        merged[key] = (prefix if group == "prefix" else action)[key]
        counts[group] += merged[key].numel()
    # Config and normalisation processors are identical within a lineage; take the action side's.
    shutil.copytree(action_checkpoint, output, ignore=shutil.ignore_patterns(WEIGHTS))
    save_file(merged, str(output / WEIGHTS), metadata={"format": "pt"})
    record = {"schema": SCHEMA, "prefix_checkpoint": str(prefix_checkpoint.resolve()),
              "action_checkpoint": str(action_checkpoint.resolve()),
              "prefix_weights_sha256": file_hash(prefix_checkpoint / WEIGHTS),
              "action_weights_sha256": file_hash(action_checkpoint / WEIGHTS),
              "hybrid_weights_sha256": file_hash(output / WEIGHTS), "parameters": counts}
    write_json_atomic(output / "transplant.json", record)
    return record


def evaluate(checkpoint: Path, output: Path, tasks, *, offset: int, episodes: int, state_count: int):
    """Same command, seed, and initial states as the capability timeline."""
    for task in tasks:
        destination = output / f"task{task}"
        if (destination / "physical_events.json").is_file():
            continue
        if destination.exists():
            raise FileExistsError(f"incomplete evaluation output: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(_evaluation_command(checkpoint, task, destination, offset, episodes, state_count),
                       check=True)


def outcomes(root: Path, tasks) -> dict[tuple[int, int], bool]:
    result = {}
    for task in tasks:
        for row in json.loads((root / f"task{task}" / "physical_events.json").read_text())["episodes"]:
            result[(task, int(row["initial_state_id"]))] = bool(row["success"])
    return result


def summarize(arms: dict[str, Path], tasks, output: Path) -> dict:
    """Success per arm and paired differences on shared (task, initial state) cells."""
    data = {name: outcomes(root, tasks) for name, root in arms.items()}
    cells = sorted(set.intersection(*(set(v) for v in data.values())))
    if not cells:
        raise ValueError("arms share no evaluated cells")
    clusters = [f"task{task}" for task, _ in cells]
    success = {name: {"success": int(sum(v[c] for c in cells)), "episodes": len(cells),
                      "by_task": {str(t): int(sum(v[c] for c in cells if c[0] == t)) for t in tasks}}
               for name, v in data.items()}
    names = list(arms)
    paired = {f"{a} - {b}": cluster_ci(np.asarray([data[a][c] for c in cells], float)
                                       - np.asarray([data[b][c] for c in cells], float), clusters)
              for i, a in enumerate(names) for b in names[i + 1:]}
    report = {"schema": SCHEMA + "_report", "cells": len(cells), "success": success,
              "paired_differences": paired, "ci_clusters": "task (4 clusters; wide by design)",
              "arms": {name: str(root.resolve()) for name, root in arms.items()}}
    write_json_atomic(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--prefix-checkpoint", type=Path, required=True)
    b.add_argument("--action-checkpoint", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--arm", nargs=2, action="append", required=True, metavar=("NAME", "EVAL_ROOT"))
    s.add_argument("--output", type=Path, required=True)
    for parser in (e, s):
        parser.add_argument("--task", type=int, action="append")
    e.add_argument("--initial-state-offset", type=int, default=0)
    e.add_argument("--episodes", type=int, default=40)
    e.add_argument("--initial-state-count", type=int, default=50)
    a = p.parse_args()
    tasks = getattr(a, "task", None) or [0, 1, 2, 3]
    if a.command == "build":
        print(json.dumps(build(a.prefix_checkpoint, a.action_checkpoint, a.output), indent=2))
    elif a.command == "evaluate":
        evaluate(a.checkpoint, a.output, tasks, offset=a.initial_state_offset, episodes=a.episodes,
                 state_count=a.initial_state_count)
    else:
        print(json.dumps(summarize({n: Path(r) for n, r in a.arm}, tasks, a.output), indent=2))


if __name__ == "__main__":
    main()
