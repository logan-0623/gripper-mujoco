"""Where does the policy go under another task's instruction (instruction_swap_eval --trace)?

Implements the criteria preregistered in
docs/results/libero_smolvla_instruction_swap_20261011/README.md.

For each ``other`` episode in scene k with the prompt of task j = (k+1) mod 4:
* reach point: end-effector position one step after the first close command
  (its lowest point if it never closes);
* memorised location: task j's target bowl (akita_black_bowl_1) in task j's own
  scene with the same initial-state ID, taken from any traced episode run in
  scene j (initial positions do not depend on prompt or checkpoint);
* class (xy distances, 5 cm threshold): ``key`` = reach point near the memorised
  location and away from every current bowl; ``bowl`` = reach point near a
  current bowl or a bowl lifted; otherwise ``other``;
* DTW: end-effector path up to the reach point against task j's and task k's
  demonstrations (raw LIBERO HDF5, up to their first close), nearest demonstration,
  both paths resampled to 25 points, mean per-step distance in cm.

Calibration (``correct``): successful episodes should reach within the threshold
of the target bowl in at least 90% of cases. The preregistered 5 cm failed this
(grasps close on the rim), so ``--threshold-cm`` takes the 90th percentile of the
correct successes' reach distances, reported as ``p90_cm``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

LIFT_M, POINTS = 0.05, 25
BOWLS = ("akita_black_bowl_1", "akita_black_bowl_2")


def reach_index(gripper, z: np.ndarray) -> int:
    """Index into the path (which has one more entry than the commands)."""
    closes = [i for i, g in enumerate(gripper) if g > 0]
    return closes[0] + 1 if closes else int(z.argmin())


def xy(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def resample(path: np.ndarray) -> np.ndarray:
    t = np.linspace(0, len(path) - 1, POINTS)
    return np.stack([np.interp(t, np.arange(len(path)), path[:, d]) for d in range(3)], 1)


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    cost = np.linalg.norm(a[:, None] - b[None], axis=2)
    acc = np.full((len(a) + 1, len(b) + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[-1, -1] / (len(a) + len(b)) * 100)


def demo_paths(demos: Path, language: str) -> list[np.ndarray]:
    import h5py

    paths = []
    with h5py.File(demos / (language.replace(" ", "_") + "_demo.hdf5")) as h:
        for demo in h["data"].values():
            pos = np.asarray(demo["obs/ee_pos"])
            paths.append(resample(pos[:reach_index(list(demo["actions"][:, -1]), pos[:, 2]) + 1]))
    return paths


def load(root: Path, step: str, cond: str) -> list[dict]:
    rows = []
    for f in sorted((root / step / cond).glob("task*/trace.json")):
        events = {int(e["initial_state_id"]): e
                  for e in json.loads((f.parent / "physical_events.json").read_text())["episodes"]}
        for e in json.loads(f.read_text())["episodes"]:
            rows.append({**e, "success": bool(events[e["initial_state_id"]]["success"])})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--demos", type=Path, default=Path("data/libero/raw/libero_spatial"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold-cm", type=float, default=5.0,
                   help="reach radius; 'calibrated' runs use the 90th percentile of correct successes")
    args = p.parse_args()
    THRESHOLD_M = args.threshold_cm / 100
    steps = sorted(d.name for d in args.root.iterdir() if d.is_dir())
    data = {(s, c): load(args.root, s, c) for s in steps for c in ("correct", "other")}
    scenes = {(e["task_id"], e["initial_state_id"]): e["initial_objects"] for rows in data.values() for e in rows}
    tasks = sorted({t for t, _ in scenes})
    contract = json.loads((args.root / "contract.json").read_text())
    demos = {t: demo_paths(args.demos, contract["prompts"]["correct"][str(t)]) for t in tasks}

    report = {"calibration": {}, "cells": {}, "decision": {}}
    for (step, cond), rows in data.items():
        for e in rows:
            k = e["task_id"]
            j = tasks[(tasks.index(k) + 1) % len(tasks)] if cond == "other" else k
            here, path = e["initial_objects"], np.asarray(e["eef"])
            stop = reach_index(e["gripper"], path[:, 2])
            r = path[stop]
            lifted = [n for n in here if e["max_z"][n] - here[n][2] > LIFT_M]
            to_bowl = min(xy(r, here[b]) for b in BOWLS)
            memorised = scenes[(j, e["initial_state_id"])]["akita_black_bowl_1"]
            approach = resample(path[:stop + 1])
            d_source = min(dtw(approach, d) for d in demos[j])
            d_current = min(dtw(approach, d) for d in demos[k])
            cell = report["cells"].setdefault(f"{int(step) // 1000}k/{cond}/scene{k}<-prompt{j}", {
                "n": 0, "class": Counter(), "lifted": Counter(), "success": 0,
                "reach_to_memorised_cm": [], "reach_to_target_bowl_cm": [], "reach_to_other_bowl_cm": [],
                "memorised_to_nearest_bowl_cm": [], "dtw_source_cm": [], "dtw_current_cm": [], "source_closer": 0})
            cell["n"] += 1
            cell["success"] += e["success"]
            if xy(r, memorised) <= THRESHOLD_M and to_bowl > THRESHOLD_M:
                cell["class"]["key"] += 1
            elif to_bowl <= THRESHOLD_M or any(b in lifted for b in BOWLS):
                cell["class"]["bowl"] += 1
            else:
                cell["class"]["other"] += 1
            cell["lifted"].update(lifted or ["none"])
            cell["reach_to_memorised_cm"].append(xy(r, memorised) * 100)
            cell["reach_to_target_bowl_cm"].append(xy(r, here[BOWLS[0]]) * 100)
            cell["reach_to_other_bowl_cm"].append(xy(r, here[BOWLS[1]]) * 100)
            cell["memorised_to_nearest_bowl_cm"].append(min(xy(memorised, here[b]) for b in BOWLS) * 100)
            cell["dtw_source_cm"].append(d_source)
            cell["dtw_current_cm"].append(d_current)
            cell["source_closer"] += d_source < d_current
            if cond == "correct" and e["success"]:
                cal = report["calibration"].setdefault(f"{int(step) // 1000}k", {"within": 0, "n": 0, "cm": []})
                cal["within"] += xy(r, here[BOWLS[0]]) <= THRESHOLD_M
                cal["n"] += 1
                cal["cm"].append(xy(r, here[BOWLS[0]]) * 100)

    for step in steps:
        label = f"{int(step) // 1000}k"
        decisive = [c for n, c in report["cells"].items()
                    if n.startswith((f"{label}/other/scene2<", f"{label}/other/scene3<"))]
        if not decisive:
            continue
        n = sum(c["n"] for c in decisive)
        key = sum(c["class"].get("key", 0) for c in decisive) / n
        bowl = sum(c["class"].get("bowl", 0) for c in decisive) / n
        source = float(np.median([d for c in decisive for d in c["dtw_source_cm"]]))
        current = float(np.median([d for c in decisive for d in c["dtw_current_cm"]]))
        report["decision"][label] = {
            "scenes_2_3_episodes": n, "key_fraction": round(key, 2), "bowl_fraction": round(bowl, 2),
            "dtw_source_median_cm": round(source, 1), "dtw_current_median_cm": round(current, 1),
            "supports_key": key >= 0.5 and source < current, "against_key": bowl >= 0.5}
    for cal in report["calibration"].values():
        cal["p90_cm"] = round(float(np.percentile(cal.pop("cm"), 90)), 1)
        cal["passes"] = cal["within"] >= 0.9 * cal["n"]
    report["threshold_cm"] = args.threshold_cm
    for cell in report["cells"].values():
        for key in [k for k, v in cell.items() if isinstance(v, list)]:
            cell[key] = round(float(np.median(cell[key])), 1)
        cell["class"], cell["lifted"] = dict(cell["class"]), dict(cell["lifted"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("threshold_cm", "calibration", "decision")}, indent=2))
    for name, cell in report["cells"].items():
        print(name, cell)


if __name__ == "__main__":
    main()
