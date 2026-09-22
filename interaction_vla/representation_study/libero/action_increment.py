"""Paired Z+A versus A analysis of saved readouts; no fitting or policy execution."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .predictive_states import Protocol, TARGETS, file_hash, sequence_index, target
from .state_bank import load_state_bank

ACTION = "policy_action_chunk"


def paired_summary(scores, y, tasks, episodes, mask):
    """Equal-task primary metric; overlapping frames are not independent trials."""
    if not mask.any():
        return {"status": "not_estimable", "n": 0, "reason": "empty_subset"}
    errors = {name: (score[mask].astype(np.float64) - y[mask]) ** 2
              for name, score in scores.items()}
    tasks, episodes = tasks[mask], episodes[mask]

    def metrics(selected):
        values = {f"brier_{name}": float(error[selected].mean()) for name, error in errors.items()}
        return {"n": int(selected.sum()), **values,
                "gain_a_minus_za": values["brier_a"] - values["brier_za"]}

    per_task = [{"task": str(t), **metrics(tasks == t)} for t in np.unique(tasks)]
    per_episode = [{"task": str(t), "episode": str(e), **metrics((tasks == t) & (episodes == e))}
                   for t in np.unique(tasks) for e in np.unique(episodes[tasks == t])]
    macro = {key: float(np.mean([row[key] for row in per_task]))
             for key in ("brier_a", "brier_z", "brier_za", "gain_a_minus_za")}
    return {"status": "measured", "n": int(mask.sum()), "tasks": len(per_task),
            "episodes": len(per_episode), "task_macro": macro,
            "sample_weighted": metrics(np.ones(len(tasks), dtype=bool)),
            "tasks_positive_gain": sum(row["gain_a_minus_za"] > 0 for row in per_task),
            "per_task": per_task, "per_episode": per_episode}


def checked_prediction(predictions, key, expected, source_row):
    for field in ("state_id", "task", "episode", "target"):
        if not np.array_equal(predictions[f"{key}/{field}"], expected[field]):
            raise ValueError(f"Prediction alignment/labels differ: {key}/{field}")
    score = predictions[f"{key}/score"]
    if (score.shape != expected["target"].shape or not np.isfinite(score).all()
            or np.any((score < 0) | (score > 1))):
        raise ValueError(f"Invalid saved scores: {key}")
    error = (score.astype(np.float64) - expected["target"]) ** 2
    macro = np.mean([error[expected["task"] == t].mean() for t in np.unique(expected["task"])])
    if (source_row["n"] != len(score)
            or not np.isclose(error.mean(), source_row["brier"], rtol=1e-6, atol=1e-8)
            or not np.isclose(macro, source_row["task_macro_brier"], rtol=1e-6, atol=1e-8)):
        raise ValueError(f"Saved predictions disagree with source report: {key}")
    return score


def analyze(run_dir: Path, bank: Path):
    source = json.loads((run_dir / "report.json").read_text())
    if source.get("schema") != "smolvla_token_readouts_v1":
        raise ValueError("Expected completed token readout report")
    if not source["features"]:
        raise ValueError("No source representations to compare")
    if source["state_bank_sha256"] != file_hash(bank / "manifest.json"):
        raise ValueError("Source report/StateBank mismatch")
    cfg = Protocol(**source["protocol"])
    records, _, split, _ = load_state_bank(bank)
    index, partitions, _ = sequence_index(records, split, cfg)
    anchors = [records[i] for i in index[:, cfg.history - 1]]
    identity = {"state_id": np.array([r.state_id for r in anchors]),
                "task": np.array([f"{r.suite}/{r.task_id}" for r in anchors]),
                "episode": np.array([f"{r.suite}/{r.task_id}/{r.source_episode_id}" for r in anchors])}
    if len(set(identity["state_id"])) != len(anchors):
        raise ValueError("Duplicate anchor state IDs")
    rows = {(r["method"], r["target"], r["horizon"]): r
            for r in source["results"] if "method" in r}
    if len(rows) != sum("method" in r for r in source["results"]):
        raise ValueError("Duplicate source result rows")
    comparisons = []
    with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as predictions:
        for label in TARGETS:
            now = np.array([np.nan if target(r, label) is None else target(r, label) for r in anchors])
            for horizon in cfg.horizons:
                future = [records[i] for i in index[:, cfg.history - 1 + horizon]]
                y = np.array([np.nan if target(r, label) is None else target(r, label) for r in future])
                valid = (partitions == "test") & np.isfinite(now) & np.isfinite(y)
                expected = {k: v[valid] for k, v in identity.items()}
                expected["target"] = y[valid]
                changed = y[valid] != now[valid]
                for feature, shape in source["features"].items():
                    methods = {"a": ACTION, "z": feature, "za": f"{feature}+{ACTION}"}
                    entry = {"target": label, "horizon": horizon, "representation": feature,
                             "dimensions_a": source["controls"][ACTION][1], "dimensions_z": shape[1],
                             "dimensions_za": source["controls"][methods["za"]][1]}
                    if not valid.any():
                        comparisons.append({**entry, "status": "not_estimable", "reason": "no_valid_test_labels"})
                        continue
                    if any((m, label, horizon) not in rows for m in methods.values()):
                        raise ValueError(f"Missing A/Z/Z+A comparison: {feature}/{label}/{horizon}")
                    if any(rows[(m, label, horizon)]["status"] != "measured" for m in methods.values()):
                        comparisons.append({**entry, "status": "not_estimable", "reason": "source_readout_not_estimable"})
                        continue
                    scores = {name: checked_prediction(predictions, f"{method}/{label}/{horizon}",
                                                      expected, rows[(method, label, horizon)])
                              for name, method in methods.items()}
                    entry.update(status="measured", alphas={name: rows[(m, label, horizon)]["alpha"]
                                                           for name, m in methods.items()}, subsets={})
                    for subset, mask in {"all": np.ones(len(changed), dtype=bool),
                                         "changed": changed, "unchanged": ~changed}.items():
                        entry["subsets"][subset] = paired_summary(
                            scores, expected["target"], expected["task"], expected["episode"], mask)
                    comparisons.append(entry)
    return {"schema": "smolvla_action_increment_v1", "status": "exploratory_posthoc_paired_analysis",
            "source_sha256": {name: file_hash(run_dir / name) for name in ("report.json", "test_predictions.npz")},
            "analysis_sha256": file_hash(Path(__file__)), "state_bank_sha256": source["state_bank_sha256"],
            "source_cache_binding_sha256": source["cache_binding_sha256"], "protocol": source["protocol"],
            "action": "current-observation predicted 50-action chunk; NOT realized demonstration future",
            "primary_metric": "task_macro Brier(A) - Brier(Z+A); positive is improvement",
            "horizon_zero": "current-label decoding diagnostic, not future prediction",
            "changed_subset": "current and future endpoint labels differ; not all intervening transitions",
            "selection": "all source representations and horizons retained; no test-set winner selection",
            "inference": "descriptive paired gains only; no independent-frame CI, p-value or significance claim",
            "limitations": source["limitations"] + [
                "Z+A and A have different input dimensions; no matched nuisance-feature control yet",
                "gain measures this readout family's performance, not conditional mutual information or causal use",
                "no gain does not establish absence of information; negative gain may reflect estimation/generalization"],
            "training": "not_run", "rl": "frozen", "closed_loop": "not_run", "comparisons": comparisons}


def run(run_dir: Path, bank: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    report = analyze(run_dir, bank)
    flattened = []
    for entry in report["comparisons"]:
        subsets = entry.get("subsets", {"all": {"status": entry["status"], "n": 0,
                                              "reason": entry.get("reason")}})
        for subset, values in subsets.items():
            flattened.append({"target": entry["target"], "horizon": entry["horizon"],
                              "representation": entry["representation"], "subset": subset,
                              "status": values["status"], "reason": values.get("reason"),
                              "n": values["n"], "tasks": values.get("tasks"),
                              "tasks_positive_gain": values.get("tasks_positive_gain"),
                              **{key: values.get("task_macro", {}).get(key)
                                 for key in ("brier_a", "brier_z", "brier_za", "gain_a_minus_za")}})
    buffer = io.StringIO()
    if flattened:
        writer = csv.DictWriter(buffer, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    output.mkdir(parents=True, exist_ok=False)
    write_bytes_atomic(output / "summary.csv", buffer.getvalue().encode("utf-8"))
    write_json_atomic(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("outputs/representation_study/libero_smolvla/state_bank"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.run_dir, args.bank, args.output)
    print(f"Complete: {len(report['comparisons'])} paired configurations; no training. Results: {args.output}")


if __name__ == "__main__":
    main()
