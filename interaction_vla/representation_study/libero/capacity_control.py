"""Matched-dimension random-control experiment for Z+A versus A."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..state_bank.io import write_json_atomic
from .predictive_states import Protocol, TARGETS, file_hash, fit_readout, readout_scores, sequence_index, target
from .state_bank import load_state_bank
from .token_cache import load_summaries
from .token_readouts import build_features


def random_features(features, window_count, seed):
    rng = np.random.default_rng(seed)
    # Each control has exactly the same per-window width as its real Z branch.
    return {name: rng.standard_normal((window_count, values.shape[1]), dtype=np.float32)
            for name, values in features.items()}


def compare(features, randoms, actions, records, index, partitions, cfg):
    anchors = index[:, cfg.history - 1]
    tasks = np.array([f"{r.suite}/{r.task_id}" for r in (records[i] for i in anchors)])
    rows = []
    for label in TARGETS:
        now = np.array([np.nan if target(records[i], label) is None else target(records[i], label)
                        for i in anchors])
        for horizon in cfg.horizons:
            future = index[:, cfg.history - 1 + horizon]
            y = np.array([np.nan if target(records[i], label) is None else target(records[i], label)
                          for i in future])
            valid = (partitions == "test") & np.isfinite(now) & np.isfinite(y)
            if not valid.any():
                continue
            action = actions[anchors].reshape(len(index), -1)
            for name, z in features.items():
                random_z = randoms[name]
                candidates = {"A": action, "Z": z, "Z+A": np.column_stack((z, action)),
                              "R": random_z, "R+A": np.column_stack((random_z, action))}
                for method, x in candidates.items():
                    fitted = fit_readout(x, y, partitions)
                    if fitted is None:
                        continue
                    model, alpha = fitted
                    score = readout_scores(model, x[valid])
                    error = (score - y[valid]) ** 2
                    test_tasks = tasks[valid]
                    task_brier = {t: float(error[test_tasks == t].mean()) for t in np.unique(test_tasks)}
                    row = {"representation": name, "target": label, "horizon": horizon,
                           "method": method, "alpha": alpha, "n": int(valid.sum()),
                           "task_macro_brier": float(np.mean(list(task_brier.values()))),
                           "brier": float(error.mean()), "task_brier": task_brier,
                           "random_seed": cfg.seed, "z_width": int(z.shape[1]),
                           "action_width": int(action.shape[1]), "status": "measured"}
                    changed = y[valid] != now[valid]
                    if changed.any():
                        changed_error = error[changed]
                        changed_tasks = test_tasks[changed]
                        row["changed_n"] = int(changed.sum())
                        row["changed_task_macro_brier"] = float(np.mean(
                            [changed_error[changed_tasks == t].mean() for t in np.unique(changed_tasks)]))
                    else:
                        row["changed_n"] = 0
                        row["changed_task_macro_brier"] = None
                    rows.append(row)
    # Add paired gains after all A/Z+A/R+A rows exist.
    lookup = {(r["representation"], r["target"], r["horizon"], r["method"]): r for r in rows}
    for row in rows:
        if row["method"] in {"Z+A", "R+A"}:
            base = lookup[(row["representation"], row["target"], row["horizon"], "A")]
            row["gain_over_A"] = base["task_macro_brier"] - row["task_macro_brier"]
            if row["changed_task_macro_brier"] is not None:
                row["changed_gain_over_A"] = base["changed_task_macro_brier"] - row["changed_task_macro_brier"]
            else:
                row["changed_gain_over_A"] = None
    return rows


def run(bank: Path, cache: Path, output: Path, cfg=Protocol()):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    records, _, split, _ = load_state_bank(bank)
    index, partitions, audit = sequence_index(records, split, cfg)
    mean, first, actions, binding = load_summaries(cache, bank)
    features, _, fitted = build_features(mean, first, actions, records, split, index, cfg)
    randoms = random_features(features, len(index), cfg.seed + 100003)
    rows = compare(features, randoms, actions, records, index, partitions, cfg)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "report.json", {
        "schema": "smolvla_capacity_control_v1", "status": "exploratory_matched_dimension",
        "protocol": cfg.__dict__, "sequence_audit": audit,
        "cache_binding_sha256": file_hash(cache / "binding.json"),
        "cache_manifest_sha256": file_hash(cache / "manifest.json"),
        "state_bank_sha256": binding["state_bank_sha256"],
        "random_control": "independent deterministic Gaussian features with exactly matched Z width; seed=cfg.seed+100003",
        "primary_metric": "task_macro_brier; gain_over_A = Brier(A)-Brier(condition)",
        "interpretation": "R+A tests dimension/estimation advantage; Z+A-R+A is descriptive evidence beyond that control",
        "limitations": ["two held-out tasks", "finite Ridge readout not conditional mutual information",
                        "random features are a capacity control, not a semantic negative control",
                        "no significance claim under overlapping windows", "closed_loop_not_run", "rl_frozen"],
        "features": {k: list(v.shape) for k, v in features.items()},
        "preprocessing": "train-only StandardScaler/PCA reused from current protocol",
        "rows": rows, "sae": "not_run", "ret": "not_run", "closed_loop": "not_run", "rl": "frozen"})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=Path("outputs/representation_study/libero_smolvla/state_bank"))
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = run(args.bank, args.cache, args.output)
    print(f"Complete: {len(rows)} matched-control readouts; results: {args.output}")


if __name__ == "__main__":
    main()
