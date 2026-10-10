"""Where does the policy go under another task's instruction (instruction_swap_eval --trace)?

For each ``other`` episode in scene k with the prompt of task j = (k+1) mod 4:
* lifted: objects whose height rose by more than 5 cm;
* reach point: end-effector position at the first close command (or, if it never
  closes, at its lowest point);
* distances (xy) from the reach point to the current scene's target bowl
  (akita_black_bowl_1), the other bowl (akita_black_bowl_2), and to where task j's
  target bowl sits in task j's own scene with the same initial-state ID
  (the "replay the memorised trajectory" location). Task j's scene positions
  come from any traced episode run in task j's scene (initial positions do not
  depend on the prompt or checkpoint).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

LIFT_M = 0.05


def reach_point(e: dict) -> list[float]:
    closes = [i for i, g in enumerate(e["gripper"]) if g > 0]
    eef = np.asarray(e["eef"])
    return eef[closes[0] + 1].tolist() if closes else eef[int(eef[:, 2].argmin())].tolist()


def xy(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def load(root: Path, step: str, cond: str) -> list[dict]:
    return [e for f in sorted((root / step / cond).glob("task*/trace.json"))
            for e in json.loads(f.read_text())["episodes"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    steps = sorted(d.name for d in args.root.iterdir() if d.is_dir())
    scenes = {(e["task_id"], e["initial_state_id"]): e["initial_objects"]
              for step in steps for cond in ("correct", "other") for e in load(args.root, step, cond)}
    tasks = sorted({t for t, _ in scenes})
    report = {}
    for step in steps:
        for cond in ("correct", "other"):
            for e in load(args.root, step, cond):
                k = e["task_id"]
                j = tasks[(tasks.index(k) + 1) % len(tasks)] if cond == "other" else k
                here, there = e["initial_objects"], scenes.get((j, e["initial_state_id"]))
                r = reach_point(e)
                near = min(here, key=lambda n: xy(r, here[n]))
                row = report.setdefault(f"{int(step) // 1000}k/{cond}/task{k}<-prompt{j}", {
                    "n": 0, "lifted": Counter(), "nearest_object": Counter(), "closed": 0,
                    "d_target_bowl": [], "d_other_bowl": [], "d_replay": [], "replay_vs_target_bowl": [], "replay_vs_other_bowl": []})
                row["n"] += 1
                row["closed"] += any(g > 0 for g in e["gripper"])
                lifted = [n for n in here if e["max_z"][n] - here[n][2] > LIFT_M] or ["none"]
                row["lifted"].update(lifted)
                row["nearest_object"][near] += 1
                row["d_target_bowl"].append(xy(r, here["akita_black_bowl_1"]))
                row["d_other_bowl"].append(xy(r, here["akita_black_bowl_2"]))
                if there is not None:
                    replay = there["akita_black_bowl_1"]
                    row["d_replay"].append(xy(r, replay))
                    row["replay_vs_target_bowl"].append(xy(replay, here["akita_black_bowl_1"]))
                    row["replay_vs_other_bowl"].append(xy(replay, here["akita_black_bowl_2"]))
    for row in report.values():
        for key in ("d_target_bowl", "d_other_bowl", "d_replay", "replay_vs_target_bowl", "replay_vs_other_bowl"):
            row[key] = round(float(np.median(row[key])) * 100, 1) if row[key] else None  # cm
        row["lifted"], row["nearest_object"] = dict(row["lifted"]), dict(row["nearest_object"].most_common(3))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for key, row in report.items():
        print(key, row)


if __name__ == "__main__":
    main()
