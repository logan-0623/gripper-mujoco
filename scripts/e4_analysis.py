"""Summarise E4 (continuation arms from 10k) against the lineage (E3 10k/25k, E5 15k/20k).

Every model is evaluated with window_entry_probe on initial states 0-19:
unpushed on Spatial 0-3 and pushed 8 steps on Spatial 0-1. Reported per model:
unpushed success, pushed success, and retention (pushed success among states
that succeed unpushed, the E3b metric). Paired differences resample (task,
initial state) pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ACQ = Path("/root/autodl-tmp/smolvla-official-reproduction-v2/acquisition")
E3, E4 = ACQ / "e3_e3b_20261007", ACQ / "e4_20261008"


def task_dirs(model: str, push: int, task: int) -> list[Path]:
    """Candidate directories holding one model/push/task cell, first existing wins."""
    if model in ("10k", "25k"):
        return [E3 / f"k{model[:2]}_push{push}" / f"task{task}"]
    if model in ("15k", "20k"):
        name = {"15k": "k01_", "20k": "k02_"}[model]  # run_r2 named them from the first two digits of 015000/020000
        return [E4 / "eval_r2" / f"{name}_push{push}_t{task}" / f"task{task}"]
    arm, step = model.split("@")
    return [E4 / "eval_r2" / f"{arm}_{step}_push{push}_t{task}" / f"task{task}",
            *([E4 / "eval" / f"{arm}_push{push}" / f"task{task}"] if step == "5000" else [])]


def outcomes(model: str, push: int, tasks) -> dict[tuple[int, int], bool]:
    cells = {}
    for task in tasks:
        path = next((d / "physical_events.json" for d in task_dirs(model, push, task)
                     if (d / "eval_info.json").is_file()), None)
        if path is None:
            continue
        for e in json.loads(path.read_text())["episodes"]:
            cells[task, int(e["initial_state_id"])] = bool(e["success"])
    return cells


def paired_ci(a: dict, b: dict, rng: np.random.Generator) -> list[float] | None:
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    d = np.asarray([a[k] - b[k] for k in keys], float)
    boots = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return [round(float(d.mean()), 3), round(float(np.quantile(boots, 0.025)), 3),
            round(float(np.quantile(boots, 0.975)), 3), len(keys)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    models = ["10k", "15k", "20k", "25k"] + [f"{a}@{s}" for s in ("2500", "5000") for a in ("C", "W", "LE", "LA")]
    rng = np.random.default_rng(0)
    data = {m: (outcomes(m, 0, range(4)), outcomes(m, 8, range(2))) for m in models}
    retention = {m: {k: v for k, v in pushed.items() if plain.get(k)} for m, (plain, pushed) in data.items()}
    report = {}
    for m, (plain, pushed) in data.items():
        ref = f"C@{m.split('@')[1]}" if "@" in m else None
        report[m] = {
            "unpushed": [sum(plain.values()), len(plain)],
            "unpushed_by_task": {t: sum(v for k, v in plain.items() if k[0] == t) for t in range(4)},
            "pushed8": [sum(pushed.values()), len(pushed)],
            "retention8": [sum(retention[m].values()), len(retention[m])],
            "unpushed_minus_15k": paired_ci(plain, data["15k"][0], rng) if m != "15k" else None,
            "pushed8_minus_15k": paired_ci(pushed, data["15k"][1], rng) if m != "15k" else None,
            "unpushed_minus_C": paired_ci(plain, data[ref][0], rng) if ref and ref != m else None,
            "pushed8_minus_C": paired_ci(pushed, data[ref][1], rng) if ref and ref != m else None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for m, r in report.items():
        print(m, r)


if __name__ == "__main__":
    main()
