"""Q6a: structure measures that keep moving after probe accuracy saturates.

Consumes natural flow traces of several checkpoints of one lineage (same
states, noise, and solver contract) and reports, per checkpoint, expert tap,
flow stage, and token selection:

* conditional score: grouped-CV gain of [observed state + hidden PCA] over the
  observed state alone (frame index + robot state). Targets are image-derived
  or future quantities, not proprioceptive inputs: gripper-to-target distance,
  target-to-goal distance, their change K frames ahead, and stable grasp now
  and K frames ahead. Future labels come from the dense StateBank.
* fragility: Gaussian noise level (in standardized PCA units) at which the
  hidden-only probe loses half of its clean advantage over chance
  (Reblitz-Richardson 2026). It captures margin and redundancy after accuracy
  plateaus.
* transition consistency (ALAM-style, Tang et al. 2026): with states at
  t, t+K, t+2K of one episode, fit linear maps A_K: h_t -> h_{t+K}; report the
  composition error of A_K(A_K h_t) against h_{t+2K} relative to the direct map
  A_{2K}, and the reversal error B_K A_K h_t vs h_t. Raw differences are
  trivially additive, so only fitted dynamics are tested. Sparse traces rarely
  contain such triplets; the measure is then reported as not estimable.

Instruction-verb decodability (SALT, Li et al. 2026) is not computed: LIBERO
Spatial instructions share a single verb.

Token selections avoid the full 50-token mean that Action Atlas showed destroys
action-relevant information; ``all_mean`` is kept only for comparison with the
earlier pooled readouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ..state_bank.io import write_json_atomic
from .flow_diff import load_trace
from .flow_trace import file_hash
from .state_bank import load_state_bank

SCHEMA = "smolvla_representation_structure_v1"
TAPS = ("expert_middle", "expert_late")
TOKEN_SELECTIONS = {"token0": slice(0, 1), "executed_mean": slice(0, 10), "all_mean": slice(0, 50)}
NOISE_GRID = np.geomspace(0.05, 20.0, 24)


def select_tokens(values: np.ndarray, selection: str) -> np.ndarray:
    """[state, repeat, token, hidden] -> [state, hidden], averaging repeats and chosen tokens."""
    return values[:, :, TOKEN_SELECTIONS[selection]].mean(axis=(1, 2), dtype=np.float64)


def build_targets(rows, bank_by_frame, horizon: int) -> dict[str, tuple[str, np.ndarray]]:
    """Image-derived and future labels; NaN where the future frame is outside the episode."""
    def geometry(record, name):
        return getattr(record.labels.geometry, name) if record.labels.geometry else np.nan

    def future(record):
        return bank_by_frame.get((record.suite, record.task_id, record.source_episode_id,
                                  record.frame_index + horizon))

    targets = {}
    for name in ("gripper_target_distance", "target_goal_distance"):
        now = np.asarray([geometry(r, name) for r in rows], dtype=float)
        later = np.asarray([geometry(f, name) if (f := future(r)) else np.nan for r in rows], dtype=float)
        targets[name] = ("regression", now)
        targets[f"delta_{name}_k{horizon}"] = ("regression", later - now)

    def grasp(record):
        return float(record.labels.stable_grasp) if record and record.labels.stable_grasp is not None else np.nan

    targets["stable_grasp"] = ("classification", np.asarray([grasp(r) for r in rows]))
    targets[f"stable_grasp_k{horizon}"] = ("classification", np.asarray([grasp(future(r)) for r in rows]))
    return targets


def _model(kind: str):
    if kind == "regression":
        return Ridge(alpha=10.0)
    return LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")


def _score(kind: str, y, prediction) -> float:
    if kind == "regression":
        return float(r2_score(y, prediction))
    return float(balanced_accuracy_score(y, prediction))


def grouped_probe(features, controls, y, kind, groups, *, folds: int, pca_dim: int, seed: int):
    """Grouped-CV scores for controls, controls+hidden, hidden-only, and hidden fragility."""
    keep = np.isfinite(y)
    features, controls, y, groups = features[keep], controls[keep], y[keep], np.asarray(groups)[keep]
    if kind == "classification":
        y = y.astype(int)
        if np.unique(y).size < 2:
            return {"status": "not_estimable", "reason": "single class"}
    unique_groups = np.unique(groups)
    if unique_groups.size < folds:
        return {"status": "not_estimable", "reason": f"{unique_groups.size} episodes < {folds} folds"}
    chance = 0.0 if kind == "regression" else 0.5
    predictions = {name: np.empty(len(y), dtype=float) for name in ("controls", "joint", "hidden")}
    noisy = np.empty((len(NOISE_GRID), len(y)), dtype=float)
    rng = np.random.default_rng(seed)
    for train, test in GroupKFold(n_splits=folds).split(features, y, groups):
        if kind == "classification" and np.unique(y[train]).size < 2:
            return {"status": "not_estimable", "reason": "single class in a training fold"}
        pca = PCA(n_components=min(pca_dim, len(train) - 1), random_state=seed).fit(features[train])
        hidden_scaler = StandardScaler().fit(pca.transform(features[train]))
        hidden_train = hidden_scaler.transform(pca.transform(features[train]))
        hidden_test = hidden_scaler.transform(pca.transform(features[test]))
        control_scaler = StandardScaler().fit(controls[train])
        control_train, control_test = control_scaler.transform(controls[train]), control_scaler.transform(controls[test])
        for name, (fit_x, test_x) in {
            "controls": (control_train, control_test),
            "joint": (np.column_stack((control_train, hidden_train)), np.column_stack((control_test, hidden_test))),
            "hidden": (hidden_train, hidden_test),
        }.items():
            model = _model(kind).fit(fit_x, y[train])
            predictions[name][test] = model.predict(test_x)
            if name == "hidden":
                for index, sigma in enumerate(NOISE_GRID):
                    noisy[index, test] = model.predict(test_x + rng.normal(scale=sigma, size=test_x.shape))
    if not all(np.isfinite(value).all() for value in predictions.values()) or not np.isfinite(noisy).all():
        raise ValueError("non-finite probe predictions")
    scores = {name: _score(kind, y, value) for name, value in predictions.items()}
    noise_scores = [_score(kind, y, row) for row in noisy]
    advantage = scores["hidden"] - chance
    fragility = None
    if advantage > 0:
        threshold = chance + advantage / 2
        fragility = next((float(sigma) for sigma, score in zip(NOISE_GRID, noise_scores) if score < threshold),
                         float("inf"))
    return {"status": "complete", "kind": kind, "states": int(len(y)), "episodes": int(unique_groups.size),
            "chance": chance, **{f"{name}_score": value for name, value in scores.items()},
            "conditional_gain": scores["joint"] - scores["controls"],
            "fragility_sigma": fragility,
            "noise_curve": [{"sigma": float(s), "score": v} for s, v in zip(NOISE_GRID, noise_scores)]}


def transition_triplets(rows, horizon: int):
    """Indices (i, j, k) whose frames are t, t+K, t+2K of one episode."""
    index = {(r.suite, r.task_id, r.source_episode_id, r.frame_index): i for i, r in enumerate(rows)}
    triplets = []
    for i, r in enumerate(rows):
        key = (r.suite, r.task_id, r.source_episode_id)
        j, k = index.get((*key, r.frame_index + horizon)), index.get((*key, r.frame_index + 2 * horizon))
        if j is not None and k is not None:
            triplets.append((i, j, k))
    return triplets


def transition_consistency(features, rows, horizon: int, *, minimum: int, pca_dim: int, seed: int):
    triplets = transition_triplets(rows, horizon)
    if len(triplets) < minimum:
        return {"status": "not_estimable",
                "reason": f"{len(triplets)} (t, t+K, t+2K) triplets < {minimum}; needs a dense trace"}
    i, j, k = (np.asarray(column) for column in zip(*triplets))
    z = StandardScaler().fit_transform(PCA(n_components=min(pca_dim, len(features) - 1),
                                           random_state=seed).fit_transform(features))
    episodes = [f"{rows[a].suite}:{rows[a].task_id}:{rows[a].source_episode_id}" for a in i]
    group_ids = np.unique(episodes, return_inverse=True)[1]
    if np.unique(group_ids).size < 2:
        return {"status": "not_estimable", "reason": "transition triplets come from one episode"}
    errors = {"composition": [], "direct": [], "reversal": [], "identity_baseline": []}
    for train, test in GroupKFold(n_splits=min(5, np.unique(group_ids).size)).split(i, groups=group_ids):
        forward = Ridge(alpha=1.0).fit(np.vstack((z[i[train]], z[j[train]])), np.vstack((z[j[train]], z[k[train]])))
        direct = Ridge(alpha=1.0).fit(z[i[train]], z[k[train]])
        backward = Ridge(alpha=1.0).fit(np.vstack((z[j[train]], z[k[train]])), np.vstack((z[i[train]], z[j[train]])))
        composed = forward.predict(forward.predict(z[i[test]]))
        errors["composition"].append(np.mean((composed - z[k[test]]) ** 2))
        errors["direct"].append(np.mean((direct.predict(z[i[test]]) - z[k[test]]) ** 2))
        errors["reversal"].append(np.mean((backward.predict(forward.predict(z[i[test]])) - z[i[test]]) ** 2))
        errors["identity_baseline"].append(np.mean((z[i[test]] - z[k[test]]) ** 2))
    mean = {name: float(np.mean(values)) for name, values in errors.items()}
    return {"status": "complete", "triplets": len(triplets),
            "composition_over_direct": mean["composition"] / mean["direct"],
            "reversal_error": mean["reversal"], "errors": mean}


def success_series(path: Path | None, tasks: set[tuple[str, int]]):
    if path is None:
        return None
    # The capability timeline evaluates LIBERO Spatial tasks only.
    rows = json.loads(path.read_text())["rows"]
    series = {}
    for row in rows:
        if ("libero_spatial", int(row["task"])) in tasks:
            totals = series.setdefault(int(row["step"]), [0, 0])
            totals[0] += int(row["success"])
            totals[1] += int(row["episodes"])
    if not series:
        return {"status": "no_overlap", "reason": "traced tasks do not include the timeline's Spatial tasks"}
    return {str(step): {"success": s, "episodes": n, "rate": s / n} for step, (s, n) in sorted(series.items())}


def run(bank: Path, checkpoints: Sequence[tuple[str, Path]], output: Path, *, horizon: int,
        folds: int, pca_dim: int, stages: Sequence[int] | None, selections: Sequence[str],
        success: Path | None = None, minimum_triplets: int = 30, seed: int = 0):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if len(checkpoints) < 2 or len({name for name, _ in checkpoints}) != len(checkpoints):
        raise ValueError("supply at least two uniquely named checkpoints in training order")
    if set(selections) - set(TOKEN_SELECTIONS) or horizon <= 0:
        raise ValueError("unknown token selection or non-positive horizon")
    records, manifest, _, _ = load_state_bank(bank)
    if not manifest.get("audit_passed"):
        raise ValueError("StateBank audit has not passed")
    by_id = {r.state_id: r for r in records}
    by_frame = {(r.suite, r.task_id, r.source_episode_id, r.frame_index): r for r in records}
    anchor, rows, provenance, report_rows = None, None, {}, []
    for name, path in checkpoints:
        ids, arrays, binding, trace_manifest = load_trace(path)
        if binding.get("query_mode") != "natural_integration" or binding.get("flow_edit") is not None:
            raise ValueError(f"{name}: only unedited natural traces are supported")
        if binding["state_bank_sha256"] != file_hash(bank / "manifest.json"):
            raise ValueError(f"{name}: StateBank binding mismatch")
        if anchor is None:
            anchor = (ids, arrays["epsilon"])
            rows = [by_id[i] for i in ids.tolist()]
            targets = build_targets(rows, by_frame, horizon)
            controls = np.column_stack(([r.frame_index for r in rows],
                                        np.asarray([r.observation.robot_state for r in rows], dtype=float)))
            groups = [f"{r.suite}:{r.task_id}:{r.source_episode_id}" for r in rows]
        elif not np.array_equal(ids, anchor[0]) or not np.array_equal(arrays["epsilon"], anchor[1]):
            raise ValueError(f"{name}: states or noise are not paired with the first checkpoint")
        provenance[name] = {"path": str(path.resolve()), "binding_sha256": trace_manifest["binding_sha256"],
                            "checkpoint_tree_sha256": binding["checkpoint_tree_sha256"]}
        for tap in TAPS:
            values = arrays[tap]  # [state, repeat, stage, token, hidden]
            for stage in (stages if stages is not None else range(values.shape[2])):
                for selection in selections:
                    features = select_tokens(values[:, :, stage], selection)
                    row = {"checkpoint": name, "tap": tap, "stage": int(stage), "tokens": selection,
                           "targets": {target: grouped_probe(features, controls, y, kind, groups,
                                                             folds=folds, pca_dim=pca_dim, seed=seed)
                                       for target, (kind, y) in targets.items()},
                           "transition": transition_consistency(features, rows, horizon,
                                                                minimum=minimum_triplets,
                                                                pca_dim=pca_dim, seed=seed)}
                    report_rows.append(row)
                print(json.dumps({"checkpoint": name, "tap": tap, "stage": int(stage)}), flush=True)
        del arrays
    report = {"schema": SCHEMA, "exploration_only": True, "source_sha256": file_hash(Path(__file__)),
              "inputs": provenance, "states": len(rows),
              "episodes": len({(r.suite, r.task_id, r.source_episode_id) for r in rows}),
              "horizon_frames": horizon, "folds": folds, "pca_dim": pca_dim,
              "controls": "frame index + robot state (observed inputs)",
              "fragility_definition": "smallest Gaussian sigma (standardized PCA units) at which the hidden-only "
                                      "probe loses half its clean advantage over chance; inf if never",
              "noise_grid": NOISE_GRID.tolist(),
              "success": success_series(success, {(r.suite, r.task_id) for r in rows}),
              "not_computed": {"instruction_verb_decodability": "LIBERO Spatial instructions share one verb"},
              "rows": report_rows}
    write_json_atomic(output / "report.json", report)
    return {"schema": SCHEMA, "rows": len(report_rows), "output": str(output / "report.json")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--checkpoint", nargs=2, action="append", required=True, metavar=("NAME", "TRACE"),
                   help="checkpoint name and natural trace, in training order")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=10, help="future offset K in frames")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--stage", type=int, action="append", help="default: every flow stage")
    p.add_argument("--tokens", choices=tuple(TOKEN_SELECTIONS), action="append")
    p.add_argument("--success", type=Path, help="capability timeline report.json for the same lineage")
    p.add_argument("--minimum-triplets", type=int, default=30)
    a = p.parse_args()
    print(json.dumps(run(a.bank, [(name, Path(path)) for name, path in a.checkpoint], a.output,
                         horizon=a.horizon, folds=a.folds, pca_dim=a.pca_dim, stages=a.stage,
                         selections=a.tokens or list(TOKEN_SELECTIONS), success=a.success,
                         minimum_triplets=a.minimum_triplets), indent=2))


if __name__ == "__main__":
    main()
