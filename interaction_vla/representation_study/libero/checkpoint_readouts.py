"""Exploratory CKA and frozen cross-checkpoint readouts on paired flow caches.

Consumes train/validation episode caches, never the confirmation partition.
No policy training, candidate selection, or causal gate is performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ..state_bank.io import write_json_atomic
from .flow_diff import load_trace
from .flow_trace import file_hash
from .predictive_states import SparseReadoutRidge
from .state_bank import load_state_bank

TAPS = ("expert_middle", "expert_late")
CONTRACT = ("state_bank_sha256", "dataset_revision", "dataset_scientific_sha256",
            "contract_tree_sha256", "metadata_tree_sha256", "image_binding",
            "num_steps", "chunk_size", "max_action_dim", "expert_middle_index",
            "expert_late_index", "query_mode")


def episode(row):
    return row.suite, row.task_id, row.source_episode_id


def weights(records):
    """Task equal, episode equal within task, state equal within episode."""
    groups = [episode(row) for row in records]
    tasks = {key[:2] for key in groups}
    counts = {key: groups.count(key) for key in set(groups)}
    return np.asarray([1 / (len(tasks) * sum(g[:2] == key[:2] for g in counts)
                           * counts[key]) for key in groups])


def linear_cka(x, y, weight):
    """Weighted centered linear CKA; a constant representation is undefined."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("CKA needs paired two-dimensional features")
    weight = np.asarray(weight, dtype=float)
    if weight.shape != (len(x),) or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError("invalid CKA weights")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite CKA features")
    weight = weight / weight.sum()
    centered = []
    for value in (x, y):
        value = value - np.sum(value * weight[:, None], axis=0)
        value = value * np.sqrt(weight[:, None])
        # Sparse multiplication uses the project's macOS-safe numerical path.
        centered.append((csr_matrix(value) @ csr_matrix(value).T).toarray())
    a, b = centered
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / denominator) if denominator > 0 else None


def targets(records):
    return {
        "contact": np.asarray([float(r.labels.contact.gripper_target)
                                if r.labels.contact is not None else np.nan for r in records]),
        "stable_grasp": np.asarray([float(r.labels.stable_grasp)
                                     if r.labels.stable_grasp is not None else np.nan for r in records]),
        "gripper_target_distance": np.asarray([r.labels.geometry.gripper_target_distance
                                               if r.labels.geometry else np.nan for r in records]),
        "target_goal_distance": np.asarray([r.labels.geometry.target_goal_distance
                                            if r.labels.geometry else np.nan for r in records]),
        "demonstration_gripper_action": np.asarray([r.observation.action[6] for r in records]),
    }


def probe_matrix(features, train_y, validation_y, train_records, validation_records, alpha):
    """Fit source scaler/readout once; never refit on a destination checkpoint."""
    train = np.isfinite(train_y)
    valid = np.isfinite(validation_y)
    if train.sum() < 2 or not valid.any() or np.unique(train_y[train]).size < 2:
        return {"status": "not_estimable", "reason": "missing targets or constant training target"}
    tr = [r for r, keep in zip(train_records, train) if keep]
    va = [r for r, keep in zip(validation_records, valid) if keep]
    w = weights(tr) * len(tr)
    baseline = float(np.average(train_y[train], weights=w))
    rows = []
    for source, (x_train, _) in features.items():
        model = make_pipeline(StandardScaler(), FunctionTransformer(csr_matrix),
                              SparseReadoutRidge(alpha=alpha, solver="lsqr"))
        model.fit(x_train[train], train_y[train], standardscaler__sample_weight=w,
                  sparsereadoutridge__sample_weight=w)
        for destination, (_, x_valid) in features.items():
            prediction = model.predict(x_valid[valid])
            if not np.isfinite(prediction).all():
                raise ValueError("non-finite readout predictions")
            error = (prediction - validation_y[valid]) ** 2
            null_error = (baseline - validation_y[valid]) ** 2
            for task in [None] + sorted({(r.suite, r.task_id) for r in va}):
                mask = np.asarray([task is None or (r.suite, r.task_id) == task for r in va])
                selected = [r for r, keep in zip(va, mask) if keep]
                ww = weights(selected)
                mse = float(np.sum(ww * error[mask]))
                null_mse = float(np.sum(ww * null_error[mask]))
                rows.append({"source": source, "destination": destination,
                             "task": list(task) if task else "macro", "mse": mse,
                             "training_mean_mse": null_mse, "mse_gain": null_mse - mse,
                             "episodes": len({episode(r) for r in selected}), "states": len(selected)})
    return {"status": "complete", "rows": rows}


def run(bank, checkpoints, output, *, alpha=10.0):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if len(checkpoints) < 2 or len({c[0] for c in checkpoints}) != len(checkpoints):
        raise ValueError("supply at least two uniquely named checkpoints in training order")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    records, manifest, _, split = load_state_bank(bank)
    if not manifest.get("audit_passed"):
        raise ValueError("StateBank audit has not passed")
    by_id = {r.state_id: r for r in records}
    anchors, selected, pooled, provenance = {}, {}, {}, {}
    common_contract = None
    hashes = set()
    for name, train_path, validation_path in checkpoints:
        pooled[name], provenance[name] = {}, {}
        checkpoint_hash = None
        for partition, path in (("train", train_path), ("validation", validation_path)):
            ids, arrays, binding, trace_manifest = load_trace(Path(path))
            if binding.get("flow_edit") is not None or binding.get("split_group") != "episode":
                raise ValueError("only unedited episode-split traces are supported")
            if binding.get("partition") != partition or binding.get("state_bank_sha256") != file_hash(bank / "manifest.json"):
                raise ValueError("StateBank or partition binding mismatch")
            if not len(ids) or len(set(ids.tolist())) != len(ids) or any(split.assignments.get(i) != partition for i in ids):
                raise ValueError("empty, duplicated, or incorrectly partitioned state IDs")
            if any(key not in binding for key in CONTRACT):
                raise ValueError("missing trace comparison contract")
            contract = {key: binding[key] for key in CONTRACT}
            if common_contract is None:
                common_contract = contract
            if contract != common_contract or binding["query_mode"] not in {"natural_integration", "fixed_reference_points"}:
                raise ValueError("incompatible trace comparison contracts")
            if checkpoint_hash is not None and checkpoint_hash != binding["checkpoint_tree_sha256"]:
                raise ValueError("train/validation checkpoint mismatch")
            checkpoint_hash = binding["checkpoint_tree_sha256"]
            paired = {key: arrays[key] for key in ("epsilon", "sigma")}
            if binding["query_mode"] == "fixed_reference_points":
                paired["x_sigma"] = arrays["x_sigma"]
            if any(not np.isfinite(value).all() for value in paired.values()):
                raise ValueError("non-finite noise or flow points")
            if partition not in anchors:
                anchors[partition] = (ids, paired)
                selected[partition] = [by_id[i] for i in ids]
            else:
                ref_ids, ref = anchors[partition]
                if not np.array_equal(ids, ref_ids) or any(not np.array_equal(paired[k], ref[k]) for k in paired):
                    raise ValueError("unpaired state IDs, noise, or flow points")
            pooled[name][partition] = {}
            for tap in TAPS:
                values = arrays[tap]
                if values.ndim != 5 or values.shape[0] != len(ids) or min(values.shape) < 1 or not np.isfinite(values).all():
                    raise ValueError("expected finite state/repeat/stage/token/hidden arrays")
                pooled[name][partition][tap] = values.mean(axis=(1, 3), dtype=np.float64)
            provenance[name][partition] = {"path": str(Path(path).resolve()),
                                           "checkpoint_tree_sha256": checkpoint_hash,
                                           "binding_sha256": trace_manifest["binding_sha256"]}
            del arrays
        if checkpoint_hash in hashes:
            raise ValueError("duplicate checkpoint content")
        hashes.add(checkpoint_hash)
    if {episode(r) for r in selected["train"]} & {episode(r) for r in selected["validation"]}:
        raise ValueError("train/validation episode leakage")
    for tap in TAPS:
        shapes = {pooled[name][part][tap].shape[1:] for name in pooled for part in selected}
        if len(shapes) != 1:
            raise ValueError("incompatible tap shapes")
    labels = {p: targets(rs) for p, rs in selected.items()}
    controls = {
        "frame_index": tuple(np.asarray([[r.frame_index] for r in selected[p]], dtype=float)
                             for p in ("train", "validation")),
        "robot_state": tuple(np.asarray([r.observation.robot_state for r in selected[p]], dtype=float)
                             for p in ("train", "validation")),
    }
    if any(not np.isfinite(x).all() for pair in controls.values() for x in pair):
        raise ValueError("non-finite nuisance controls")
    control_rows = [{"control": name, "target": target,
                     **probe_matrix({name: pair}, labels["train"][target], labels["validation"][target],
                                    selected["train"], selected["validation"], alpha)}
                    for name, pair in controls.items() for target in labels["train"]]
    names = list(pooled)
    cka_rows, readouts = [], []
    for tap in TAPS:
        for stage in range(pooled[names[0]]["train"][tap].shape[1]):
            features = {n: tuple(pooled[n][p][tap][:, stage] for p in ("train", "validation")) for n in names}
            for left in range(len(names)):
                for right in range(left, len(names)):
                    for part_index, part in enumerate(("train", "validation")):
                        cka_rows.append({"tap": tap, "flow_stage": stage, "partition": part,
                                         "before": names[left], "after": names[right],
                                         "cka": linear_cka(features[names[left]][part_index],
                                                           features[names[right]][part_index], weights(selected[part]))})
            for target in labels["train"]:
                readouts.append({"tap": tap, "flow_stage": stage, "target": target,
                                 **probe_matrix(features, labels["train"][target], labels["validation"][target],
                                                selected["train"], selected["validation"], alpha)})
            print(f"Complete {tap} flow stage {stage}", flush=True)
    report = {"schema": "smolvla_checkpoint_readouts_v1", "exploration_only": True,
              "source_sha256": file_hash(Path(__file__)),
              "inputs": provenance, "comparison_contract": common_contract,
              "pooling": "mean over noise repeats and action tokens; flow stages separate",
              "alpha": alpha, "alpha_selection": "fixed; no validation tuning",
              "weighting": "task equal / episode equal / state equal",
              "probe_metric": "unclipped MSE (including binary targets), not calibrated probability",
              "limitations": ["single lineage must be verified separately", "no causal-use claim",
                              "validation is development, not independent confirmation",
                              "natural-flow comparisons include evolving x_sigma",
                              "pooled expert taps only; no VLM or token-level attribution",
                              "no confidence intervals or conditional nuisance-adjusted probes"],
              "controls": control_rows, "cka": cka_rows, "readouts": readouts}
    write_json_atomic(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", nargs=3, action="append", required=True,
                        metavar=("NAME", "TRAIN_TRACE", "VALIDATION_TRACE"))
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.bank, args.checkpoint, args.output, alpha=args.alpha)


if __name__ == "__main__":
    main()
