"""Frozen-VLA sequence diagnostics; official SAE/RET, shared held-out readouts.

No policy training or rollout is performed here. See research/predictive-states.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch.utils.data import DataLoader, Dataset

from ..state_bank.io import write_json_atomic
from ..statistics import clustered_bootstrap_mean
from .latents import load_latent_cache
from .schema import StateRecord
from .splits import PARTITIONS, SplitManifest, validate_split
from .state_bank import load_state_bank

ROOT = Path(__file__).resolve().parents[3]
TARGETS = ("contact", "stable_grasp")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upstream(name: str) -> dict:
    path = ROOT / "research" / name
    if not path.is_dir():
        raise FileNotFoundError(f"Missing official checkout: research/{name}")
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD"], text=True)
    untracked = subprocess.check_output(
        ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"], text=True
    ).strip()
    if untracked:
        raise ValueError(f"Untracked upstream source must be accounted for: {name}: {untracked}")
    sys.path.insert(0, str(path))
    return {"commit": commit, "patch_sha256": hashlib.sha256(diff.encode()).hexdigest()}


@dataclass(frozen=True)
class Protocol:
    history: int = 4
    horizons: tuple[int, ...] = (0, 1, 5, 10)
    dt: float = 0.1
    seed: int = 42
    z_dim: int = 32
    ret_steps: int = 2000
    sae_epochs: int = 30
    sae_expansion: float = 2.0
    sae_k: int = 32
    batch_size: int = 64
    device: str = "cpu"

    def __post_init__(self):
        if self.history < 2 or not self.horizons or min(self.horizons) < 0:
            raise ValueError("history >= 2 and non-negative horizons required")
        if max(self.horizons) < 1 or len(set(self.horizons)) != len(self.horizons):
            raise ValueError("Include a future horizon; horizons must be unique")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if min(self.z_dim, self.ret_steps, self.sae_epochs, self.sae_k, self.batch_size) <= 0:
            raise ValueError("Model dimensions and training budgets must be positive")
        if self.z_dim % 8 or not np.isfinite(self.sae_expansion) or self.sae_expansion <= 0:
            raise ValueError("RET z_dim must be divisible by 8; SAE expansion must be positive")


def sequence_index(records: list[StateRecord], split: SplitManifest, cfg: Protocol):
    """Index complete windows only; no interpolation, padding, or split recreation."""
    validate_split(records, split)
    if split.group_unit != "task":
        raise ValueError("Cross-task evaluation requires the existing task-group split")
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[(record.suite, record.task_id, record.source_episode_id)].append(i)
    length = cfg.history + max(cfg.horizons)
    windows, dropped, gaps = [], Counter(), Counter()
    for indices in groups.values():
        indices.sort(key=lambda i: records[i].frame_index)
        frames = np.array([records[i].frame_index for i in indices])
        times = np.array([records[i].observation.timestamp for i in indices])
        if len(np.unique(frames)) != len(frames) or np.any(np.diff(times) <= 0):
            raise ValueError("Duplicate frames or non-increasing episode timestamps")
        gaps.update(map(int, np.diff(frames)))
        for start in range(max(0, len(indices) - length + 1)):
            stop = start + length
            if np.any(np.diff(frames[start:stop]) != 1):
                dropped["frame_gap"] += 1
            elif not np.allclose(np.diff(times[start:stop]), cfg.dt, rtol=0, atol=1e-6):
                dropped["time_gap"] += 1
            else:
                windows.append(indices[start:stop])
    if not windows:
        raise ValueError("No complete, correctly timed windows; check history/horizon/dt")
    index = np.asarray(windows, dtype=np.int64)
    anchors = [records[i] for i in index[:, cfg.history - 1]]
    partitions = np.array([split.assignments[r.state_id] for r in anchors])
    counts = Counter(partitions)
    if any(counts[p] == 0 for p in PARTITIONS):
        raise ValueError(f"Every split needs complete windows, found: {dict(counts)}")
    report = {
        "states": len(records), "episodes": len(groups), "windows": len(index),
        "partition_windows": dict(counts), "discarded_windows": dict(dropped),
        "frame_gaps": {str(k): v for k, v in sorted(gaps.items())},
        "history": cfg.history, "horizons": list(cfg.horizons), "dt_seconds": cfg.dt,
        "targets": {name: sum(target(r, name) is not None for r in records) for name in TARGETS},
    }
    return index, partitions, report


def target(record: StateRecord, name: str):
    if name == "contact":
        return None if record.labels.contact is None else float(record.labels.contact.gripper_target)
    if name == "stable_grasp":
        return None if record.labels.stable_grasp is None else float(record.labels.stable_grasp)
    raise ValueError(name)


def bound_latents(bank: Path, records, cache: Path):
    ids, values, manifest = load_latent_cache(cache)
    if manifest["state_bank_sha256"] != file_hash(bank / "manifest.json"):
        raise ValueError("Latents belong to another StateBank")
    if set(ids) != {r.state_id for r in records}:
        raise ValueError("Latents must cover exactly these StateBank IDs")
    lookup = {state_id: i for i, state_id in enumerate(ids)}
    return values[[lookup[r.state_id] for r in records]], manifest


class RETWindows(Dataset):
    """Robot frames occupy causal sequence positions, not language tokens."""
    def __init__(self, values, index):
        self.values = values
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        h = torch.from_numpy(self.values[self.index[i]])
        # Autonomous RET ignores token IDs; no tokenizer, token loss, or rewards.
        return {"hidden_states": h, "input_ids": torch.zeros(len(h), dtype=torch.long),
                "attention_mask": torch.ones(len(h), dtype=torch.long)}


def train_representations(values, records, split, index, partitions, cfg, output):
    from accelerate import Accelerator

    provenance = {name: upstream(name) for name in ("action-atlas", "ret")}
    from experiments.train_sae import TrainSAEConfig, train_sae_on_activations
    from experiments.sae_hooks import TopKSAE
    from ret.config import ExperimentConfig
    from ret.training.trainer import JEPATrainer

    train_rows = np.array([split.assignments[r.state_id] == "train" for r in records])
    if cfg.sae_k > int(values.shape[1] * cfg.sae_expansion):
        raise ValueError("SAE k exceeds dictionary width")
    if cfg.z_dim > min(train_rows.sum(), values.shape[1]):
        raise ValueError("z_dim exceeds train-only PCA rank")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    scaler = StandardScaler().fit(values[train_rows])
    scaled = scaler.transform(values).astype(np.float32)
    pca = PCA(n_components=cfg.z_dim, svd_solver="full").fit(scaled[train_rows])
    pca_values = pca.transform(scaled).astype(np.float32)
    history = index[:, :cfg.history]
    features = {
        "hidden_current": scaled[history[:, -1]],
        "hidden_history": scaled[history].reshape(len(index), -1),
        "pca_history": pca_values[history].reshape(len(index), -1),
    }
    # Same already-published training implementation; only TRAIN rows are supplied.
    sae_result = train_sae_on_activations(torch.from_numpy(values[train_rows]), TrainSAEConfig(
        expansion=cfg.sae_expansion, k=cfg.sae_k, epochs=cfg.sae_epochs,
        batch_size=cfg.batch_size, patience=cfg.sae_epochs, device=cfg.device,
    ))
    torch.save(sae_result, output / "sae.pt")
    sae = TopKSAE(values.shape[1], sae_result["config"]["hidden_dim"], cfg.sae_k).to(cfg.device)
    sae.load_state_dict(sae_result["sae_state_dict"])
    sae.eval()
    codes = []
    with torch.no_grad():
        for batch in torch.from_numpy(values).split(cfg.batch_size):
            normalized = (batch - sae_result["activation_mean"]) / sae_result["activation_std"]
            codes.append(sae.encode(normalized.to(cfg.device)).cpu().numpy())
    codes = np.concatenate(codes)
    features["sae_current"] = codes[history[:, -1]]
    features["sae_history"] = codes[history].reshape(len(index), -1)

    ret_cfg = ExperimentConfig()
    ret_cfg.llm.name = "frozen-smolvla-activation-cache"
    ret_cfg.llm.hidden_dim = values.shape[1]
    ret_cfg.data.mode = "cached"
    ret_cfg.data.hf_dataset = None
    ret_cfg.data.cached_hidden_states_dir = "bound-by-run-report"
    ret_cfg.encoder.type = "transformer"
    ret_cfg.encoder.z_dim = cfg.z_dim
    ret_cfg.encoder.num_heads = 4
    ret_cfg.loss.prediction_type = "l1"
    ret_cfg.training.max_steps = cfg.ret_steps
    ret_cfg.training.batch_size = cfg.batch_size
    ret_cfg.training.seed = cfg.seed
    ret_cfg.training.eval_every = max(1, cfg.ret_steps // 10)
    ret_cfg.logging.log_every = ret_cfg.training.eval_every
    ret_cfg.logging.output_dir = str(output / "ret")
    torch.manual_seed(cfg.seed)
    accelerator = Accelerator(cpu=cfg.device == "cpu", mixed_precision="no")
    if accelerator.device.type != torch.device(cfg.device).type:
        raise ValueError(f"RET requested {cfg.device}, accelerate selected {accelerator.device}")
    trainer = JEPATrainer(ret_cfg, accelerator)
    loaders = [DataLoader(
        RETWindows(scaled, index[partitions == p, :cfg.history + 1]),
        batch_size=cfg.batch_size, shuffle=p == "train", num_workers=0,
        generator=torch.Generator().manual_seed(cfg.seed),
    ) for p in ("train", "validation")]
    trainer.train(*loaders, use_wandb=False)
    encoder = accelerator.unwrap_model(trainer.encoder).eval()
    ret_values = []
    with torch.no_grad():
        for start in range(0, len(index), cfg.batch_size):
            h = torch.from_numpy(scaled[history[start:start + cfg.batch_size]]).to(cfg.device)
            z, _ = encoder(h)
            ret_values.append(z[:, -1].cpu().numpy())
    features["ret_history"] = np.concatenate(ret_values)
    np.savez_compressed(output / "preprocessing.npz", mean=scaler.mean_, scale=scaler.scale_,
                        pca_components=pca.components_, pca_mean=pca.mean_)
    diagnostics = {
        "sae_validation_dead_fraction": float((np.abs(codes[np.array([
            split.assignments[r.state_id] == "validation" for r in records
        ])]).sum(axis=0) == 0).mean()),
        "ret_validation_mean_std": float(features["ret_history"][partitions == "validation"].std(axis=0).mean()),
        "ret_steps": trainer.global_step,
    }
    return features, provenance, diagnostics


def paired_action_effects(forward, layer, deltas: dict[str, np.ndarray], *, seed: int,
                          reference: np.ndarray, atol: float = 1e-5):
    """Replay an actual frozen policy with official Atlas hooks, last call only.

    forward() must reset policy queues/history, use identical observations, and
    return one postprocessed 7D action. Callers must report its coordinate system:
    postprocessing alone does not establish physical units or environment execution.
    An empty deltas dict requests only the zero-replay check. Deltas are in ORIGINAL
    activation units, broadcast across tokens. Caller supplies target/random
    controls and selects features on train/validation, never test outcomes.
    """
    upstream("action-atlas")
    from experiments.hooks import ActivationCaptureHook, ActivationInjectionHook

    if not np.isfinite(atol) or atol <= 0 or "zero" in deltas:
        raise ValueError("Positive tolerance required; zero control is added automatically")
    reference = np.asarray(reference)
    if reference.ndim != 1 or not np.isfinite(reference).all():
        raise ValueError("Reference must be a finite pooled activation vector")
    for delta in deltas.values():
        if np.shape(delta) != reference.shape or not np.isfinite(delta).all():
            raise ValueError("Every delta must match the reference vector")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    def infer(hook):
        calls = 0
        def counted_hook(module, inputs, result):
            nonlocal calls
            calls += 1
            if isinstance(hook, ActivationInjectionHook):
                actual = result[0] if isinstance(result, tuple) else result
                hook.device = actual.device
            return hook(module, inputs, result)
        handle = layer.register_forward_hook(counted_hook)
        try:
            # External/environment RNGs must be reset by the forward callback.
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.manual_seed(seed)
                result = torch.as_tensor(forward()).detach().cpu().numpy()
            if result.shape != (7,) or not np.isfinite(result).all():
                raise ValueError("forward must return one finite postprocessed 7D action")
            return result, calls
        finally:
            handle.remove()

    capture = ActivationCaptureHook()
    baseline, baseline_calls = infer(capture)
    if not capture.activations:
        raise ValueError("Selected policy layer was never called")
    actual = capture.activations[-1]
    if actual.ndim != 3 or actual.shape[0] != 1:
        raise ValueError("Expected one-state activation [1, tokens, features]")
    pooled = actual.float().mean(dim=(0, 1)).numpy()
    if not np.allclose(pooled, reference, atol=atol, rtol=0):
        raise ValueError("Live policy activation differs from the bound cached reference")
    output = {"original_action": baseline.tolist(), "hook_calls": len(capture.activations),
              "scope": "offline_first_action_sensitivity_not_closed_loop", "conditions": {}}
    for name, delta in {"zero": np.zeros_like(reference), **deltas}.items():
        stored = list(capture.activations)
        stored[-1] = actual + torch.as_tensor(delta, dtype=actual.dtype)
        hook = ActivationInjectionHook(stored)
        action, calls = infer(hook)
        if calls != baseline_calls or hook.injection_count != len(stored) or hook.shape_mismatches:
            raise ValueError("Policy replay hook call count/shape differs")
        if name == "zero" and not np.allclose(action, baseline, atol=atol, rtol=0):
            raise ValueError("Zero-delta replay is not equivalent to the original policy")
        diff = action - baseline
        output["conditions"][name] = {
            "action": action.tolist(), "translation_l2": float(np.linalg.norm(diff[:3])),
            "rotation_l2": float(np.linalg.norm(diff[3:6])), "gripper_abs": float(abs(diff[6])),
            "activation_delta_l2": float(np.linalg.norm(delta)),
        }
    return output


class SparseReadoutRidge(Ridge):
    """Keep sklearn's Ridge solver, avoiding dense BLAS in its final intercept."""

    def _set_intercept(self, X_offset, y_offset, X_scale=None):
        if not self.fit_intercept:
            self.intercept_ = 0.0
            return
        self.coef_ = self.coef_.astype(X_offset.dtype, copy=False)
        if X_scale is not None:
            self.coef_ = self.coef_ / X_scale
        # Same offset dot product for one or multiple targets. Elementwise
        # reduction avoids the false macOS matmul FP flags without hiding warnings.
        self.intercept_ = y_offset - np.sum(self.coef_ * X_offset, axis=-1)


def readout_scores(model, x):
    """Reject invalid raw predictions before clipping can hide an infinity."""
    score = model.predict(np.asarray(x, dtype=np.float32))
    if not np.isfinite(score).all():
        raise ValueError("Readout produced non-finite scores before clipping")
    return np.clip(score, 0, 1)


def fit_readout(x, y, partitions):
    """Same linear family; select regularization on validation only."""
    # Keep cache, time and action readouts at the same numerical precision.
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Readout features must be finite")
    valid = np.isfinite(y)
    masks = {p: valid & (partitions == p) for p in PARTITIONS}
    if any(m.sum() == 0 for m in masks.values()):
        return None
    if any(np.unique(y[masks[p]]).size < 2 for p in ("train", "validation")):
        return None
    best = None
    for alpha in (0.1, 1.0, 10.0, 100.0):
        # Use SciPy's sparse matvec path: local macOS dense matmul produces false
        # FP flags even for finite float32 inputs. Same LSQR objective, no filtering.
        model = make_pipeline(StandardScaler(), FunctionTransformer(csr_matrix),
                              SparseReadoutRidge(alpha=alpha, solver="lsqr"))
        model.fit(x[masks["train"]], y[masks["train"]])
        if not np.isfinite(model[-1].coef_).all() or not np.isfinite(model[-1].intercept_).all():
            raise ValueError("Readout fit produced non-finite parameters")
        pred = readout_scores(model, x[masks["validation"]])
        loss = float(np.mean((pred - y[masks["validation"]]) ** 2))
        if best is None or loss < best[0]:
            best = loss, alpha, model
    return best[2], best[1]


def evaluate(features, records, index, partitions, cfg, output, *, controls=None, progress=False):
    import joblib

    anchors = [records[i] for i in index[:, cfg.history - 1]]
    tasks = np.array([f"{r.suite}/{r.task_id}" for r in anchors])
    episodes = np.array([f"{r.suite}/{r.task_id}/{r.source_episode_id}" for r in anchors])
    test = partitions == "test"
    # Only already-observed elapsed time, never episode-normalized time (future duration).
    starts = {}
    for r in records:
        key = (r.suite, r.task_id, r.source_episode_id)
        starts[key] = min(starts.get(key, r.observation.timestamp), r.observation.timestamp)
    elapsed = np.array([r.observation.timestamp - starts[(r.suite, r.task_id, r.source_episode_id)]
                        for r in anchors], dtype=np.float32)[:, None]
    actions = np.array([[records[j].observation.action for j in row[cfg.history - 1:-1]] for row in index])
    rows, predictions, readouts = [], {}, {}
    for name in TARGETS:
        now = np.array([np.nan if target(r, name) is None else target(r, name) for r in anchors])
        for horizon in cfg.horizons:
            future = [records[i] for i in index[:, cfg.history - 1 + horizon]]
            y = np.array([np.nan if target(r, name) is None else target(r, name) for r in future])
            mask = test & np.isfinite(y) & np.isfinite(now)
            if not mask.any():
                rows.append({"target": name, "horizon": horizon, "status": "not_estimable"})
                continue
            train = (partitions == "train") & np.isfinite(y)
            if not train.any():
                rows.append({"target": name, "horizon": horizon, "status": "not_estimable"})
                continue
            candidates = {**features, **(controls or {}), "elapsed_time": elapsed}
            if horizon:
                realized_actions = actions[:, :horizon].reshape(len(index), -1)
                candidates["realized_actions_only"] = realized_actions
                candidates.update({f"{method}+realized_actions": np.column_stack((x, realized_actions))
                                   for method, x in features.items()})
            baseline_scores = {"train_prevalence": np.full(mask.sum(), y[train].mean())}
            if horizon:
                baseline_scores["privileged_persistence"] = now[mask]
            for method, x in candidates.items():
                if progress:
                    print(f"Readout {name} +{horizon}: {method}", flush=True)
                fitted = fit_readout(x, y, partitions)
                if fitted is None:
                    rows.append({"method": method, "target": name, "horizon": horizon,
                                 "status": "not_estimable", "reason": "train/validation class support"})
                    continue
                model, alpha = fitted
                score = readout_scores(model, x[mask])
                baseline_scores[method] = score
                readouts[f"{method}/{name}/{horizon}"] = model
                # Store alpha with the row, not an implicit test-set choice.
                rows.append({"method": method, "target": name, "horizon": horizon,
                             "alpha": alpha, "status": "measured"})
            for method, score in baseline_scores.items():
                key = f"{method}/{name}/{horizon}"
                row = next((r for r in rows if (r.get("method"), r["target"], r["horizon"]) ==
                            (method, name, horizon)), None)
                if row is None:
                    row = {"method": method, "target": name, "horizon": horizon, "status": "measured"}
                    rows.append(row)
                error = (score - y[mask]) ** 2
                changed = y[mask] != now[mask]
                per_task = [float(error[tasks[mask] == t].mean()) for t in np.unique(tasks[mask])]
                row.update(n=int(mask.sum()), test_tasks=len(per_task), brier=float(error.mean()),
                           task_macro_brier=float(np.mean(per_task)), positives=int(y[mask].sum()),
                           auprc=float(average_precision_score(y[mask], score)) if np.unique(y[mask]).size == 2 else None,
                           changed_n=int(changed.sum()), changed_brier=float(error[changed].mean()) if changed.any() else None)
                reference = baseline_scores["privileged_persistence" if horizon else "train_prevalence"]
                difference = (reference - y[mask]) ** 2 - error
                row["brier_gain_over_reference"] = clustered_bootstrap_mean(
                    difference, tasks[mask], samples=1000, confidence=0.95, seed=cfg.seed
                ) if len(per_task) >= 2 else None
                predictions[key + "/score"] = score
                predictions[key + "/target"] = y[mask]
                predictions[key + "/state_id"] = np.array([r.state_id for r in anchors])[mask]
                predictions[key + "/episode"] = episodes[mask]
                predictions[key + "/task"] = tasks[mask]
    np.savez_compressed(output / "test_predictions.npz", **predictions)
    joblib.dump(readouts, output / "readouts.joblib")
    return rows


def run(bank: Path, cache: Path | None, output: Path, cfg: Protocol, methods: str = "all"):
    if methods not in {"all", "hidden-only", "controls-only"}:
        raise ValueError("methods must be all, hidden-only, or controls-only")
    if methods != "controls-only" and cache is None:
        raise ValueError("Representation comparisons require a bound latent cache")
    if methods == "controls-only" and cache is not None:
        raise ValueError("controls-only does not consume a latent cache; omit --cache")
    records, _, split, _ = load_state_bank(bank)
    index, partitions, audit = sequence_index(records, split, cfg)
    values, binding = bound_latents(bank, records, cache) if cache is not None else (None, None)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    output.mkdir(parents=True)
    write_json_atomic(output / "sequence_audit.json", audit)
    if methods == "all":
        features, source, diagnostics = train_representations(values, records, split, index, partitions, cfg, output)
    elif methods == "hidden-only":
        # Explicit cheap diagnostic, never labeled as the full SAE/RET comparison.
        history = index[:, :cfg.history]
        features = {"hidden_current": values[history[:, -1]],
                    "hidden_history": values[history].reshape(len(index), -1)}
        source, diagnostics = {}, {}
    else:
        features, source, diagnostics = {}, {}, {}
    rows = evaluate(features, records, index, partitions, cfg, output)
    report = {
        "schema": "predictive_interaction_states_v1", "status": "offline_diagnostic",
        "protocol": asdict(cfg), "latent_binding": binding, "upstream": source,
        "mode": methods, "state_bank_sha256": file_hash(bank / "manifest.json"),
        "representation_comparison": "not_run" if methods == "controls-only" else "offline_readouts_only",
        "runner_sha256": file_hash(Path(__file__)), "diagnostics": diagnostics,
        "methods": list(features), "results": rows,
        "readout": {"family": "StandardScaler + Ridge(lsqr), clipped scores",
                    "matrix_backend": "scipy_csr_after_standardization",
                    "dtype": "float32", "alpha_grid": [0.1, 1.0, 10.0, 100.0],
                    "selection": "validation Brier only", "calibrated_probabilities": False},
        "policy_capability": "not_verified_by_this_run", "causal_control": "not_run",
        "event_sae": "not_run", "rl": "frozen",
        "interpretation": "Encoding/future-label readouts only; no control utility or novelty claim.",
        "action_conditioning": "Realized demonstration actions, not counterfactual or autonomous forecasting.",
    }
    write_json_atomic(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "run"))
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizons", type=int, nargs="+", default=[0, 1, 5, 10])
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--ret-steps", type=int, default=2000)
    parser.add_argument("--sae-epochs", type=int, default=30)
    parser.add_argument("--sae-k", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--methods", choices=("all", "hidden-only", "controls-only"), default="all")
    args = parser.parse_args()
    cfg = Protocol(**{k: getattr(args, k) for k in (
        "history", "dt", "seed", "z_dim", "ret_steps", "sae_epochs", "sae_k", "batch_size", "device"
    )}, horizons=tuple(args.horizons))
    if args.command == "audit":
        records, _, split, _ = load_state_bank(args.bank)
        _, _, report = sequence_index(records, split, cfg)
        args.output.mkdir(parents=True, exist_ok=False)
        write_json_atomic(args.output / "sequence_audit.json", report)
        print(json.dumps(report, indent=2))
    else:
        if args.cache is None and args.methods != "controls-only":
            parser.error("run requires --cache from the SAME frozen checkpoint and StateBank")
        report = run(args.bank, args.cache, args.output, cfg, args.methods)
        print(f"Saved {len(report['results'])} diagnostic rows to {args.output}/report.json; causal control NOT RUN")


if __name__ == "__main__":
    main()
