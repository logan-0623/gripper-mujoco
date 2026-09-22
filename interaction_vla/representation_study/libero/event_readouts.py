"""G2b prospective onset/offset readouts on the frozen 480-D token cache."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ..state_bank.io import write_json_atomic
from .predictive_states import Protocol, file_hash, fit_readout, readout_scores, sequence_index, target
from .state_bank import load_state_bank
from .token_cache import load_summaries

ARMS = ("B0_AC", "B1_AC_pca_mean_current", "B2_AC_pca_first_current", "B3_AC_pca_first_history")
EVENTS = ("onset", "offset")
TARGETS = ("contact", "stable_grasp")
HORIZONS = (5, 10)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def pca_transform(pca, values):
    """PCA projection without Apple's BLAS matmul warning path."""
    centered = np.asarray(values) - pca.mean_
    return np.einsum("ij,kj->ik", centered, pca.components_, optimize=True)


def event_labels(records, index, label, horizon, event):
    values = np.array([[np.nan if target(records[i], label) is None else target(records[i], label)
                        for i in row] for row in index], dtype=np.float32)
    current = values[:, 3]
    future = values[:, 4:4 + horizon]
    complete = np.isfinite(current) & np.isfinite(future).all(axis=1)
    if event == "onset":
        risk = complete & (current == 0)
        y = (future == 1).any(axis=1)
    else:
        risk = complete & (current == 1)
        y = (future == 0).any(axis=1)
    return risk, y.astype(np.float32), current


def build_event_features(mean, first, actions, records, split, index, cfg):
    train_states = np.array([split.assignments[r.state_id] == "train" for r in records])
    history = index[:, :cfg.history]
    anchor = history[:, -1]
    fitted = {}
    scaled = {}
    for name, values in (("mean", mean), ("first", first)):
        scaler = StandardScaler().fit(values[train_states])
        # sklearn's dense float32 PCA path raises a spurious macOS FP warning;
        # keep preprocessing float64 and cast only the final readout features.
        scaled[name] = scaler.transform(values).astype(np.float64, copy=False)
        fitted[name] = {"scaler": scaler}
    pca_mean = PCA(n_components=cfg.z_dim, svd_solver="full").fit(scaled["mean"][train_states])
    pca_first = PCA(n_components=cfg.z_dim, svd_solver="full").fit(scaled["first"][train_states])
    train_windows = np.array([
        split.assignments[records[i].state_id] == "train" for i in anchor
    ])
    pca_history = PCA(n_components=cfg.z_dim, svd_solver="full").fit(
        scaled["first"][history[train_windows]].reshape(train_windows.sum(), -1))
    fitted["pca_mean"] = pca_mean
    fitted["pca_first"] = pca_first
    fitted["pca_history"] = pca_history
    pca_mean_all = pca_transform(pca_mean, scaled["mean"]).astype(np.float32)
    pca_first_all = pca_transform(pca_first, scaled["first"]).astype(np.float32)
    pca_hist_all = pca_transform(
        pca_history, scaled["first"][history].reshape(len(index), -1)
    ).astype(np.float32)
    state = np.array([r.observation.robot_state for r in records], dtype=np.float32)
    elapsed = np.zeros(len(records), dtype=np.float32)
    starts = {}
    for r in records:
        key = (r.suite, r.task_id, r.source_episode_id)
        starts[key] = min(starts.get(key, r.observation.timestamp), r.observation.timestamp)
    for i, r in enumerate(records):
        elapsed[i] = r.observation.timestamp - starts[(r.suite, r.task_id, r.source_episode_id)]
    action = actions[anchor].reshape(len(index), -1)
    proprio = state[history].reshape(len(index), -1)
    context = np.column_stack((action, proprio, elapsed[anchor]))
    features = {
        "B0_AC": context,
        "B1_AC_pca_mean_current": np.column_stack((context, pca_mean_all[anchor])),
        "B2_AC_pca_first_current": np.column_stack((context, pca_first_all[anchor])),
        "B3_AC_pca_first_history": np.column_stack((context, pca_hist_all)),
    }
    return features, fitted


def episode_swap(meta, seed):
    rng = np.random.default_rng(seed)
    donors = np.full(len(meta["task"]), -1, dtype=np.int64)
    for partition in np.unique(meta["partition"]):
        partition_rows = np.flatnonzero(meta["partition"] == partition)
        for task in np.unique(meta["task"][partition_rows]):
            rows = partition_rows[meta["task"][partition_rows] == task]
            episodes = np.unique(meta["episode"][rows])
            if len(episodes) < 2:
                raise ValueError(f"episode swap requires two episodes for {task}/{partition}")
            order = rng.permutation(episodes)
            for recipient, donor in zip(order, np.roll(order, 1)):
                src = rows[meta["episode"][rows] == donor]
                dst = rows[meta["episode"][rows] == recipient]
                src = src[np.argsort(meta["elapsed"][src], kind="stable")]
                times = meta["elapsed"][src]
                right = np.searchsorted(times, meta["elapsed"][dst]).clip(0, len(src) - 1)
                left = np.maximum(right - 1, 0)
                pick = np.abs(times[left] - meta["elapsed"][dst]) <= np.abs(times[right] - meta["elapsed"][dst])
                donors[dst] = src[np.where(pick, left, right)]
    if (np.any(donors < 0) or np.any(meta["episode"] == meta["episode"][donors]) or
            np.any(meta["partition"] != meta["partition"][donors])):
        raise ValueError("invalid episode swap")
    return donors


def fit_mlp(x, y, partitions, seed, weight_decay):
    train = (partitions == "train") & np.isfinite(y)
    validation = (partitions == "validation") & np.isfinite(y)
    if train.sum() == 0 or validation.sum() == 0 or np.unique(y[train]).size < 2:
        return None
    model = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam",
                          learning_rate_init=0.001, batch_size=256, max_iter=200,
                          early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                          alpha=weight_decay, random_state=seed)
    # macOS Accelerate can raise a false divide-by-zero RuntimeWarning from
    # finite dense BLAS operands. Keep the suppression at sklearn's boundary,
    # then validate the learned parameters and probabilities explicitly.
    with np.errstate(all="ignore"):
        model.fit(x[train], y[train].astype(int))
        score = model.predict_proba(x)[:, 1]
    if not all(np.isfinite(w).all() and np.isfinite(b).all()
               for w, b in zip(model.coefs_, model.intercepts_)):
        raise ValueError("MLP fit produced non-finite parameters")
    if not np.isfinite(score).all():
        raise ValueError("MLP produced non-finite probabilities")
    return model, score


def task_metrics(score, y, mask, tasks):
    """Per-task metrics; never collapse the two held-out tasks prematurely."""
    output = {}
    for task in np.unique(tasks[mask]):
        selected = mask & (tasks == task)
        positives = int((y[selected] == 1).sum())
        negatives = int((y[selected] == 0).sum())
        output[str(task)] = {
            "n": int(selected.sum()),
            "positive": positives,
            "negative": negatives,
            "positive_rate": float(positives / selected.sum()),
            "brier": float(np.mean((score[selected] - y[selected]) ** 2)),
            "auprc": (float(average_precision_score(y[selected], score[selected]))
                      if positives and negatives else None),
        }
    return output


def run(bank, cache, output, *, smoke=False, mlp=False):
    bank, cache, output = Path(bank), Path(cache), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    cfg = Protocol(history=4, horizons=(5, 10))
    records, _, split, _ = load_state_bank(bank)
    index, partitions, audit = sequence_index(records, split, cfg)
    mean, first, actions, binding = load_summaries(cache, bank)
    features, preprocessing = build_event_features(mean, first, actions, records, split, index, cfg)
    anchors = index[:, cfg.history - 1]
    meta = {"task": np.array([f"{records[i].suite}/{records[i].task_id}" for i in anchors]),
            "episode": np.array([f"{records[i].suite}/{records[i].task_id}/{records[i].source_episode_id}" for i in anchors]),
            "elapsed": np.array([records[i].observation.timestamp for i in anchors], dtype=np.float32),
            "partition": partitions.copy()}
    jobs = [(label, event, horizon) for label in TARGETS for event in EVENTS for horizon in HORIZONS]
    if smoke:
        jobs = jobs[:1]
    rows = []
    for label, event, horizon in jobs:
        risk, y, current = event_labels(records, index, label, horizon, event)
        active = risk & np.isfinite(y)
        part = partitions[active]
        y_active = y[active]
        meta_active = {k: v[active] for k, v in meta.items()}
        counts = {p: {"positive": int(((part == p) & (y_active == 1)).sum()),
                      "negative": int(((part == p) & (y_active == 0)).sum())} for p in ("train", "validation", "test")}
        swap_maps = {seed: episode_swap(meta_active, seed) for seed in (0, 1, 2)}
        for arm in ARMS:
            x = features[arm][active]
            candidates = [("aligned", None, x)]
            if arm != "B0_AC":
                for seed, donors in swap_maps.items():
                    swapped = x.copy()
                    start = features["B0_AC"].shape[1]
                    swapped[:, start:] = x[donors, start:]
                    candidates.append(("episode_swap", seed, swapped))
            for condition, seed, candidate in candidates:
                fitted = fit_readout(candidate, y_active, part)
                if fitted is None:
                    rows.append({"label": label, "event": event, "horizon": horizon, "arm": arm,
                                 "condition": condition, "seed": seed, "status": "not_estimable", "counts": counts})
                    continue
                model, alpha = fitted
                score = readout_scores(model, candidate)
                test = part == "test"
                metrics = task_metrics(score, y_active, test, meta_active["task"])
                task_scores = [v["brier"] for v in metrics.values()]
                row = {"label": label, "event": event, "horizon": horizon, "arm": arm,
                             "condition": condition, "seed": seed, "status": "measured", "counts": counts,
                             "readout": "ridge", "alpha": alpha, "n_iter": getattr(model[-1], "n_iter_", None),
                             "task_macro_brier": float(np.mean(task_scores)),
                             "task_metrics": metrics,
                             "brier": float(np.mean((score[test] - y_active[test]) ** 2)),
                             "auprc": float(average_precision_score(y_active[test], score[test])) if np.unique(y_active[test]).size == 2 else None,
                             "parameters": int(sum(np.prod(w.shape) + np.prod(b.shape) for w, b in zip(model.coefs_, model.intercepts_)))
                             if hasattr(model, "coefs_") else int(candidate.shape[1])}
                if condition == "episode_swap":
                    mismatch = np.abs(meta_active["elapsed"] - meta_active["elapsed"][swap_maps[seed]])
                    row.update({"donor_coverage": float(np.mean(np.isfinite(mismatch))),
                                "elapsed_mismatch_mean": float(np.mean(mismatch)),
                                "elapsed_mismatch_p95": float(np.quantile(mismatch, 0.95))})
                rows.append(row)
                # §11 reserves episode-swap nulls for the Ridge gate; do not
                # multiply the MLP run into a second null grid.
                if mlp and condition == "aligned":
                    for wd in (0.0, 0.0001, 0.001, 0.01):
                        for seed_mlp in (0, 1, 2):
                            fitted_mlp = fit_mlp(candidate, y_active, part, seed_mlp, wd)
                            if fitted_mlp is not None:
                                m, s = fitted_mlp
                                val = part == "validation"
                                val_metrics = task_metrics(s, y_active, val, meta_active["task"])
                                test_metrics = task_metrics(s, y_active, test, meta_active["task"])
                                rows.append({"label": label, "event": event, "horizon": horizon, "arm": arm,
                                             "condition": condition, "seed": seed, "readout": "mlp",
                                             "mlp_seed": seed_mlp, "weight_decay": wd, "status": "measured",
                                             "counts": counts, "validation_task_macro_brier": float(np.mean([
                                                 v["brier"] for v in val_metrics.values()])),
                                             "validation_task_metrics": val_metrics,
                                             "task_macro_brier": float(np.mean([
                                                 v["brier"] for v in test_metrics.values()])),
                                             "task_metrics": test_metrics,
                                             "brier": float(np.mean((s[test] - y_active[test]) ** 2)),
                                             "auprc": float(average_precision_score(y_active[test], s[test])) if np.unique(y_active[test]).size == 2 else None,
                                             "parameters": int(sum(np.prod(w.shape) + np.prod(b.shape) for w, b in zip(m.coefs_, m.intercepts_)))})
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "report.json", jsonable({"schema": "smolvla_event_readouts_v1", "complete": True,
        "status": "offline_exploratory", "protocol": asdict(cfg), "smoke": smoke, "mlp": mlp,
        "sequence_audit": audit, "cache_binding_sha256": file_hash(cache / "binding.json"),
        "cache_manifest_sha256": file_hash(cache / "manifest.json"), "state_bank_sha256": binding["state_bank_sha256"],
        "risk_set": "current Y=0/1; event is any opposite label in next K frames; full K=10 eligibility shared",
        "arms": ARMS, "primary_metric": "task_macro_brier; B0 minus Bi paired gain",
        "limitations": ["two previously inspected test tasks", "overlapping windows clustered by episode",
                        "risk-set current label is privileged", "window event is not a hazard model", "rl_frozen", "closed_loop_not_run"],
        "rows": rows}))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=Path("outputs/representation_study/libero_smolvla/state_bank"))
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--mlp", action="store_true")
    args = p.parse_args()
    rows = run(args.bank, args.cache, args.output, smoke=args.smoke, mlp=args.mlp)
    print(f"Complete: {len(rows)} event readout rows; results: {args.output}")


if __name__ == "__main__":
    main()
