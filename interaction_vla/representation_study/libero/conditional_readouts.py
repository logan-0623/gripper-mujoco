"""Resumable conditional-risk experiment; frozen VLA, label-free matched controls."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
from tqdm import tqdm

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .predictive_states import Protocol, TARGETS, file_hash, fit_readout, readout_scores, sequence_index, target
from .state_bank import load_state_bank
from .token_cache import load_summaries
from .token_readouts import build_features

REPRESENTATIONS = tuple(f"{kind}_{token}_{history}" for kind in ("hidden", "pca")
                        for token in ("mean", "first_token") for history in ("current", "history"))
CONTEXTS = ("A", "C", "AC", "ACY")
NULLS = ("partition_permute", "episode_swap")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def window_data(records, split, cfg):
    index, partitions, audit = sequence_index(records, split, cfg)
    anchors = [records[i] for i in index[:, cfg.history - 1]]
    starts = {}
    for r in records:
        key = (r.suite, r.task_id, r.source_episode_id)
        starts[key] = min(starts.get(key, r.observation.timestamp), r.observation.timestamp)
    return index, {"partitions": partitions,
                   "state_id": np.array([r.state_id for r in anchors]),
                   "task": np.array([f"{r.suite}/{r.task_id}" for r in anchors]),
                   "episode": np.array([f"{r.suite}/{r.task_id}/{r.source_episode_id}" for r in anchors]),
                   "elapsed": np.array([r.observation.timestamp - starts[(r.suite, r.task_id, r.source_episode_id)]
                                        for r in anchors], dtype=np.float32)}, audit


def donor_map(meta, kind, seed):
    """Move whole representation/history rows; never inspect labels or actions."""
    rng = np.random.default_rng(seed)
    donors = np.full(len(meta["state_id"]), -1, dtype=np.int64)
    for partition in np.unique(meta["partitions"]):
        part = np.flatnonzero(meta["partitions"] == partition)
        if kind == "partition_permute":
            if len(part) < 2:
                raise ValueError("Permutation requires >=2 windows per partition")
            order = rng.permutation(part)
            donors[order] = np.roll(order, 1)
        elif kind == "episode_swap":
            for task in np.unique(meta["task"][part]):
                group = part[meta["task"][part] == task]
                episodes = np.unique(meta["episode"][group])
                if len(episodes) < 2:
                    raise ValueError(f"Episode swap needs >=2 episodes: {partition}/{task}")
                order = rng.permutation(episodes)
                for recipient, donor in zip(order, np.roll(order, 1)):
                    src = group[meta["episode"][group] == donor]
                    dst = group[meta["episode"][group] == recipient]
                    src = src[np.argsort(meta["elapsed"][src], kind="stable")]
                    times = meta["elapsed"][src]
                    right = np.searchsorted(times, meta["elapsed"][dst]).clip(0, len(src)-1)
                    left = (right - 1).clip(0)
                    use_left = abs(times[left] - meta["elapsed"][dst]) <= abs(times[right] - meta["elapsed"][dst])
                    donors[dst] = src[np.where(use_left, left, right)]
        else:
            raise ValueError(f"Unknown null: {kind}")
    if np.any(donors < 0) or np.any(donors == np.arange(len(donors))):
        raise ValueError("Incomplete or self-matched donor map")
    if not np.array_equal(meta["partitions"], meta["partitions"][donors]):
        raise ValueError("Donor map crosses partitions")
    if kind == "episode_swap" and (not np.array_equal(meta["task"], meta["task"][donors])
                                  or np.any(meta["episode"] == meta["episode"][donors])):
        raise ValueError("Episode swap must preserve task and change episode")
    return donors


def experiment_jobs(representations, contexts, horizons, seeds):
    jobs = []
    for label in TARGETS:
        for horizon in horizons:
            for context in contexts:
                common = {"target": label, "horizon": horizon, "context": context}
                jobs.append({**common, "representation": None, "condition": "baseline", "seed": None})
                for representation in representations:
                    jobs.append({**common, "representation": representation, "condition": "aligned", "seed": None})
                    for condition in NULLS:
                        for seed in seeds:
                            jobs.append({**common, "representation": representation, "condition": condition, "seed": seed})
    return jobs


def labels_for(records, index, label, horizon, cfg):
    def values(column):
        return np.array([np.nan if target(records[i], label) is None else target(records[i], label)
                         for i in index[:, column]], dtype=np.float32)
    now = values(cfg.history - 1)
    future = values(cfg.history - 1 + horizon)
    # Identical eligibility for ALL contexts, including privileged current-label controls.
    future[~np.isfinite(now)] = np.nan
    return now, future


def context_features(context, action, controls, now):
    if context == "A":
        return action
    if context == "C":
        return controls
    if context == "AC":
        return np.column_stack((action, controls))
    if context == "ACY":
        # Missing-current-label rows are excluded from fitting/scoring by labels_for.
        return np.column_stack((action, controls, np.where(np.isfinite(now), now, 0)))
    raise ValueError(context)


def encode_npz(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def fit_checkpoint(output, spec, binding_hash, x, y, partitions):
    """Receipt is the commit point: interrupted unreceipted fits may be repeated."""
    stem = output / "fits" / digest(spec)
    receipt_path = stem.with_suffix(".json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["binding_sha256"] != binding_hash or receipt["spec"] != spec:
            raise ValueError(f"Fit binding mismatch: {stem.name}")
        expected_files = {".npz", ".joblib"} if receipt["status"] == "measured" else set()
        if receipt["status"] not in ("measured", "not_estimable") or set(receipt["files"]) != expected_files:
            raise ValueError(f"Invalid fit receipt: {stem.name}")
        for suffix, sha in receipt["files"].items():
            if file_hash(stem.with_suffix(suffix)) != sha:
                raise ValueError(f"Corrupt fit: {stem.name}{suffix}")
        if receipt["status"] == "measured":
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as saved:
                score = saved["score"]
                if score.shape != y.shape or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
                    raise ValueError(f"Invalid saved scores: {stem.name}")
        return receipt, True
    started = time.perf_counter()
    fitted = fit_readout(x, y, partitions)
    receipt = {"spec": spec, "binding_sha256": binding_hash, "dimensions": x.shape[1], "files": {}}
    if fitted is None:
        receipt.update(status="not_estimable", reason="train_or_validation_class_support")
    else:
        model, alpha = fitted
        score = readout_scores(model, x)
        buffer = io.BytesIO()
        joblib.dump(model, buffer, compress=3)
        write_bytes_atomic(stem.with_suffix(".joblib"), buffer.getvalue())
        write_bytes_atomic(stem.with_suffix(".npz"), encode_npz(score=score))
        receipt.update(status="measured", alpha=alpha, risk={})
        for partition in ("train", "validation", "test"):
            mask = (partitions == partition) & np.isfinite(y)
            receipt["risk"][partition] = float(np.mean((score[mask].astype(np.float64) - y[mask])**2)) if mask.any() else None
        receipt["files"] = {suffix: file_hash(stem.with_suffix(suffix)) for suffix in (".npz", ".joblib")}
    receipt["seconds"] = time.perf_counter() - started
    write_json_atomic(receipt_path, receipt)
    return receipt, False


def paired_risks(base, aligned, candidate, y, meta, mask):
    if not mask.any():
        return {"status": "not_estimable", "n": 0}
    errors = {name: (score[mask].astype(np.float64) - y[mask])**2
              for name, score in (("base", base), ("aligned", aligned), ("candidate", candidate))}
    task, episode = meta["task"][mask], meta["episode"][mask]
    def metrics(selected):
        result = {"brier_"+name: float(error[selected].mean()) for name, error in errors.items()}
        return {"n": int(selected.sum()), **result,
                "gain_candidate_vs_base": result["brier_base"] - result["brier_candidate"],
                "gain_aligned_vs_candidate": result["brier_candidate"] - result["brier_aligned"]}
    per_task = [{"task": str(t), **metrics(task == t)} for t in np.unique(task)]
    per_episode = [{"episode": str(e), **metrics(episode == e)} for e in np.unique(episode)]
    macro = {key: float(np.mean([row[key] for row in per_task])) for key in per_task[0] if key not in ("task", "n")}
    return {"status": "measured", "n": int(mask.sum()), "tasks": len(per_task),
            "task_macro": macro, "sample_weighted": metrics(np.ones(mask.sum(), dtype=bool)),
            "per_task": per_task, "per_episode": per_episode}


def summarize(output, jobs, records, index, meta, cfg):
    rows = []
    for job in jobs:
        if job["condition"] == "baseline":
            continue
        common = {k: job[k] for k in ("target", "horizon", "context")}
        base = {**common, "representation": None, "condition": "baseline", "seed": None}
        aligned = {**job, "condition": "aligned", "seed": None}
        specs = (base, aligned, job)
        receipts = [json.loads((output / "fits" / (digest(s)+".json")).read_text()) for s in specs]
        if any(r["status"] != "measured" for r in receipts):
            rows.append({**job, "status": "not_estimable", "reason": "readout_class_support"})
            continue
        scores = []
        for spec in specs:
            with np.load(output / "fits" / (digest(spec)+".npz"), allow_pickle=False) as f:
                scores.append(f["score"])
        now, y = labels_for(records, index, job["target"], job["horizon"], cfg)
        valid = (meta["partitions"] == "test") & np.isfinite(y)
        masks = {"all": valid, "changed": valid & (y != now), "unchanged": valid & (y == now),
                 "onset": valid & (now == 0) & (y == 1), "offset": valid & (now == 1) & (y == 0)}
        rows.append({**job, "status": "measured", "dimensions": receipts[-1]["dimensions"],
                     "alpha": receipts[-1]["alpha"], "partition_risk": receipts[-1]["risk"],
                     "subsets": {name: paired_risks(*scores, y, meta, mask) for name, mask in masks.items()}})
    return rows


def _execute(output, jobs, binding, data, *, resume=False, max_fits=None):
    records, index, meta, cfg, features, action, controls, donors, preprocessing = data
    binding = json.loads(json.dumps(binding))
    if output.exists():
        if not resume:
            raise FileExistsError("Output exists; use --resume with the identical protocol")
        if json.loads((output / "binding.json").read_text()) != binding:
            raise ValueError("Resume binding mismatch; use a new output directory")
    else:
        output.mkdir(parents=True)
        write_json_atomic(output / "binding.json", binding)
        buffer = io.BytesIO()
        joblib.dump(preprocessing, buffer, compress=3)
        write_bytes_atomic(output / "preprocessing.joblib", buffer.getvalue())
        write_bytes_atomic(output / "windows.npz", encode_npz(**meta, index=index, **donors))
        write_json_atomic(output / "setup.json", {name: file_hash(output / name) for name in ("preprocessing.joblib", "windows.npz")})
    for name, sha in json.loads((output / "setup.json").read_text()).items():
        if file_hash(output / name) != sha:
            raise ValueError(f"Corrupt setup artifact: {name}")
    binding_hash = file_hash(output / "binding.json")
    fitted_count = 0
    started = time.perf_counter()
    receipts = []
    with tqdm(total=len(jobs), desc="Conditional readouts", unit="fit") as progress:
        for job in jobs:
            now, y = labels_for(records, index, job["target"], job["horizon"], cfg)
            x = context_features(job["context"], action, controls, now)
            if job["representation"]:
                z = features[job["representation"]]
                if job["condition"] in NULLS:
                    z = z[donors[f"{job['condition']}_{job['seed']}"]]
                x = np.column_stack((x, z))
            progress.set_postfix_str(f"{job['target']} +{job['horizon']} {job['context']} {job['representation']} {job['condition']} seed={job['seed']}")
            receipt, reused = fit_checkpoint(output, job, binding_hash, x, y, meta["partitions"])
            receipts.append(receipt)
            fitted_count += not reused
            progress.update()
            state = {"complete": False, "completed": len(receipts), "expected": len(jobs),
                     "new_fits_this_invocation": fitted_count, "elapsed_seconds_this_invocation": time.perf_counter()-started}
            write_json_atomic(output / "progress.json", state)
            if max_fits is not None and fitted_count >= max_fits and len(receipts) < len(jobs):
                return state
    rows = summarize(output, jobs, records, index, meta, cfg)
    report = {"schema": "conditional_readouts_v1", "complete": True, "status": "offline_exploratory",
              "binding_sha256": binding_hash, "fits": len(jobs),
              "measured_fits": sum(r["status"] == "measured" for r in receipts),
              "total_fit_seconds": sum(r["seconds"] for r in receipts), "results": rows,
              "interpretation": "finite Ridge-family conditional Brier risk; not MI, causal use, or do(action)",
              "limitations": ["two previously inspected test tasks in the current bank; no confirmatory holdout",
                              "randomization seeds are not independent datasets or policy seeds",
                              "matched input dimension does not guarantee identical covariance/effective capacity",
                              "episode swaps approximate task/time matching, not samples from p(Z|C)",
                              "no gain is not evidence of no information; no significance or winner selection"],
              "rl": "frozen", "vla": "frozen", "sae_ret": "not_trained", "closed_loop": "not_run"}
    import csv
    buffer = io.StringIO()
    flat = []
    for row in rows:
        for subset, metrics in row.get("subsets", {"all": {"status": row["status"], "n": 0}}).items():
            flat.append({**{k: row[k] for k in ("target", "horizon", "context", "representation", "condition", "seed")},
                         "subset": subset, "status": metrics["status"], "n": metrics["n"],
                         **{k: metrics.get("task_macro", {}).get(k) for k in
                            ("brier_base", "brier_aligned", "brier_candidate", "gain_candidate_vs_base", "gain_aligned_vs_candidate")}})
    writer = csv.DictWriter(buffer, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
    write_bytes_atomic(output / "summary.csv", buffer.getvalue().encode())
    write_json_atomic(output / "report.json", report)
    write_json_atomic(output / "progress.json", {**state, "complete": True})
    return {"complete": True, "fits": len(jobs), "report": str(output / "report.json")}


def execute(output, jobs, binding, data, *, resume=False, max_fits=None):
    # OS releases this advisory lock on exit/crash; no stale PID cleanup needed.
    import fcntl
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another writer is using this output directory") from error
        return _execute(output, jobs, binding, data, resume=resume, max_fits=max_fits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--bank", type=Path, default=Path("outputs/representation_study/libero_smolvla/state_bank"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contexts", nargs="+", choices=CONTEXTS, default=list(CONTEXTS))
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    parser.add_argument("--horizons", type=int, nargs="+", choices=(0, 1, 5, 10), default=[1, 5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-fits", type=int, help="Stop after this many new fits; --resume completes the same bound protocol")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for values in (args.contexts, args.representations, args.horizons, args.seeds):
        if len(set(values)) != len(values):
            parser.error("Duplicate protocol entries are not allowed")
    if min(args.seeds) < 0 or (args.max_fits is not None and args.max_fits < 1):
        parser.error("Seeds must be nonnegative and max-fits positive")
    cfg = Protocol()
    records, _, split, _ = load_state_bank(args.bank)
    index, meta, audit = window_data(records, split, cfg)
    jobs = experiment_jobs(args.representations, args.contexts, args.horizons, args.seeds)
    donors = {f"{kind}_{seed}": donor_map(meta, kind, seed) for kind in NULLS for seed in args.seeds}
    donor_audit = {name: {"sha256": array_digest(d), "unique_donor_windows": len(np.unique(d)),
                          "same_episode_fraction": float(np.mean(meta["episode"] == meta["episode"][d])),
                          "elapsed_mismatch_p95_seconds": float(np.quantile(abs(meta["elapsed"]-meta["elapsed"][d]), .95))}
                   for name, d in donors.items()}
    manifest = json.loads((args.cache / "manifest.json").read_text())
    cache_binding = json.loads((args.cache / "binding.json").read_text())
    if (not manifest["complete"] or not manifest["full_state_bank"]
            or manifest["binding_sha256"] != file_hash(args.cache / "binding.json")
            or cache_binding["state_bank_sha256"] != file_hash(args.bank / "manifest.json")):
        raise ValueError("Complete matching token cache required")
    plan = {"fits": len(jobs), "max_ridge_fits": 4*len(jobs), "windows": audit["windows"],
            "contexts": args.contexts, "representations": args.representations, "horizons": args.horizons,
            "randomization_seeds": args.seeds, "donor_audit": donor_audit,
            "eligibility": "current and future target both finite in every partition and every arm",
            "C": "elapsed observed seconds + 4-frame proprioception; no task one-hot or future actions",
            "ACY": "privileged current label of the target, not a deployable policy input",
            "nulls": "whole-row permutation within partition; same-task different-episode nearest elapsed-time donor",
            "scientific_success_gate": "none; preserve negative, null and mixed results",
            "resource_note": "CPU Ridge only; fit count is exact, runtime unknown until smoke; no VLA/SAE/RET training"}
    print(json.dumps(plan, indent=2), flush=True)
    if args.command == "plan":
        return
    if args.output is None:
        parser.error("run requires --output")
    if args.output.exists() and not args.resume:
        parser.error("Output exists; use --resume or a new output directory")
    mean, first, actions, _ = load_summaries(args.cache, args.bank)
    features, controls, preprocessing = build_features(mean, first, actions, records, split, index, cfg)
    features = {name: features[name] for name in args.representations}
    action = controls["policy_action_chunk"]
    observed = np.column_stack((meta["elapsed"], controls["robot_state_history"]))
    del controls, mean, first, actions
    binding = {"schema": "conditional_readouts_v1", "plan": plan, "protocol": asdict(cfg),
               "bank_sha256": file_hash(args.bank / "manifest.json"),
               "cache_sha256": {name: file_hash(args.cache / name) for name in ("manifest.json", "binding.json")},
               "source_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in
                                 ("conditional_readouts.py", "predictive_states.py", "token_readouts.py", "token_cache.py", "state_bank.py")},
               "feature_sha256": {name: array_digest(x) for name, x in {**features, "A": action, "C": observed}.items()},
               "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                           "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")}}}
    result = execute(args.output, jobs, binding, (records, index, meta, cfg, features, action, observed, donors, preprocessing),
                     resume=args.resume, max_fits=args.max_fits)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
