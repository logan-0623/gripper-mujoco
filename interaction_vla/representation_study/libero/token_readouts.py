"""Hidden/PCA and action-plan controls on one complete token/action cache."""
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..state_bank.io import write_json_atomic
from .predictive_states import Protocol, evaluate, file_hash, sequence_index
from .state_bank import load_state_bank
from .token_cache import load_summaries


def build_features(mean, first, actions, records, split, index, cfg):
    train = np.array([split.assignments[r.state_id] == "train" for r in records])
    history = index[:, :cfg.history]
    anchor = history[:, -1]
    features, fitted = {}, {}
    for name, values in (("mean", mean), ("first_token", first)):
        scaler = StandardScaler().fit(values[train])
        scaled = scaler.transform(values).astype(np.float32)
        pca = PCA(n_components=cfg.z_dim, svd_solver="full").fit(scaled[train])
        # Same fitted PCA transform, using the project's SciPy path to avoid
        # known false dense-matmul floating-point flags on this macOS setup.
        compressed = (csr_matrix(scaled - pca.mean_) @ pca.components_.T).astype(np.float32)
        fitted[name] = {"scaler": scaler, "pca": pca}
        for kind, x in (("hidden", scaled), ("pca", compressed)):
            features[f"{kind}_{name}_current"] = x[anchor]
            features[f"{kind}_{name}_history"] = x[history].reshape(len(index), -1)
    state = np.array([r.observation.robot_state for r in records], dtype=np.float32)
    plan = actions[anchor].reshape(len(index), -1)
    controls = {"policy_first_action": actions[anchor, 0], "policy_action_chunk": plan,
                "robot_state_current": state[anchor],
                "robot_state_history": state[history].reshape(len(index), -1)}
    controls.update({f"{name}+policy_action_chunk": np.column_stack((x, plan))
                     for name, x in features.items()})
    return features, controls, fitted


def run(bank: Path, cache: Path, output: Path, cfg=Protocol()):
    import joblib
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    started = time.perf_counter()
    records, _, split, _ = load_state_bank(bank)
    index, partitions, audit = sequence_index(records, split, cfg)
    mean, first, actions, binding = load_summaries(cache, bank)
    features, controls, fitted = build_features(mean, first, actions, records, split, index, cfg)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "sequence_audit.json", audit)
    joblib.dump(fitted, output / "preprocessing.joblib")
    rows = evaluate(features, records, index, partitions, cfg, output, controls=controls, progress=True)
    report = {"schema": "smolvla_token_readouts_v1", "status": "offline_exploratory",
              "protocol": asdict(cfg), "cache_binding_sha256": file_hash(cache / "binding.json"),
              "cache_manifest_sha256": file_hash(cache / "manifest.json"),
              "state_bank_sha256": binding["state_bank_sha256"],
              "source_sha256": {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("predictive_states.py"))},
              "features": {k: list(v.shape) for k, v in features.items()},
              "controls": {k: list(v.shape) for k, v in controls.items()}, "results": rows,
              "elapsed_seconds": time.perf_counter()-started,
              "readout": "train-only scaling; Ridge LSQR; validation Brier alpha selection [0.1,1,10,100]",
              "primary_metric": "task_macro_brier", "history_tokens": "robot_observation_time",
              "action_plan": "current-observation predicted 50-action chunk, not realized future actions",
              "realized_actions": "privileged demonstration future, separate information condition",
              "generalization": "probe/representation task holdout; policy training task exposure not established",
              "limitations": ["two held-out tasks", "existing test controls previously inspected",
                              "mean vs first token only, not an all-token readout", "not capacity-matched",
                              "single extraction noise seed; pilot noise sensitivity reported separately"],
              "sae": "not_run", "ret": "not_run", "rl": "frozen", "closed_loop": "not_run"}
    write_json_atomic(output / "report.json", report)
    return report
