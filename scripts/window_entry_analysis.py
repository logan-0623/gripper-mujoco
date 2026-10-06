"""Analyse E3 (window-entry deviation) and E3b (push robustness) runs of window_entry_probe.py.

Entry deviation is measured against the demonstrations' own window entries
(first frame with gripper-target surface distance <= 0.10 m) in the StateBank,
per task: translation is the distance to the nearest demonstration entry, and
rotation the smallest geodesic angle to any demonstration entry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

APPROACH_DISTANCE = 0.10


def rotation_from_6d(values) -> np.ndarray:
    """Rotation matrix whose first two columns are the stored rotation6D (Gram-Schmidt)."""
    a, b = np.asarray(values[:3], float), np.asarray(values[3:6], float)
    x = a / np.linalg.norm(a)
    y = b - x * np.dot(x, b)
    y /= np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def demo_entries(bank: Path, tasks) -> dict[int, list[np.ndarray]]:
    from interaction_vla.representation_study.libero.state_bank import load_state_bank

    records, _, _, _ = load_state_bank(bank)
    episodes: dict[tuple[int, str], list] = {}
    for r in records:
        if r.suite == "libero_spatial" and r.task_id in tasks:
            episodes.setdefault((r.task_id, r.source_episode_id), []).append(r)
    entries: dict[int, list[np.ndarray]] = {t: [] for t in tasks}
    for (task, _), rows in episodes.items():
        rows.sort(key=lambda r: r.frame_index)
        entry = next(r for r in rows if r.labels.geometry.gripper_target_distance <= APPROACH_DISTANCE)
        entries[task].append(np.asarray(entry.labels.geometry.gripper_to_target, float))
    return entries


def deviation(pose9d, references) -> tuple[float, float]:
    """(translation cm to nearest demo entry, smallest rotation angle in degrees)."""
    pose = np.asarray(pose9d, float)
    translation = min(np.linalg.norm(pose[:3] - ref[:3]) for ref in references) * 100.0
    rotation = rotation_from_6d(pose[3:])
    angle = min(geodesic_deg(rotation, rotation_from_6d(ref[3:])) for ref in references)
    return float(translation), float(angle)


def auroc(scores, failures) -> float | None:
    scores, failures = np.asarray(scores, float), np.asarray(failures, bool)
    if failures.all() or not failures.any():
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(failures, scores))


def analyze(runs: dict[str, Path], bank: Path) -> dict:
    summaries = {name: json.loads((path / "summary.json").read_text()) for name, path in runs.items()}
    tasks = sorted({row["task"] for s in summaries.values() for row in s["rows"]})
    references = demo_entries(bank, tasks)
    report = {}
    for name, summary in summaries.items():
        rows = [r for r in summary["rows"] if r["entry_pose9d"] is not None]
        devs = [deviation(r["entry_pose9d"], references[r["task"]]) for r in rows]
        failures = [not r["success"] for r in rows]
        pushed = [r["push_displacement_m"] for r in rows if r["push_displacement_m"] is not None]
        report[name] = {
            "success": summary["success"], "episodes": summary["episodes"],
            "entered_window": len(rows),
            "entry_translation_cm_median": float(np.median([d[0] for d in devs])) if devs else None,
            "entry_rotation_deg_median": float(np.median([d[1] for d in devs])) if devs else None,
            "auroc_translation_predicts_failure": auroc([d[0] for d in devs], failures),
            "auroc_rotation_predicts_failure": auroc([d[1] for d in devs], failures),
            "push_displacement_cm_median": float(np.median(pushed) * 100) if pushed else None,
            "rows": [{"task": r["task"], "initial_state_id": r["initial_state_id"], "success": r["success"],
                      "entry_translation_cm": d[0], "entry_rotation_deg": d[1]}
                     for r, d in zip(rows, devs)],
        }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", nargs=2, action="append", required=True, metavar=("NAME", "OUTPUT_DIR"))
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = analyze({name: Path(path) for name, path in a.run}, a.bank)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, values in report.items():
        print(name, {k: v for k, v in values.items() if k != "rows"})


if __name__ == "__main__":
    main()
