"""Q2: is an early checkpoint wrong per step, or only in closed loop?

Compares each checkpoint's executed action prefix with the demonstration's
next actions on the same StateBank states, noise, and solver contract (natural
flow traces). If early and late checkpoints differ little here while their
closed-loop success differs a lot, the gap is compounding error rather than
single-step action quality. Demonstrations are one valid behaviour among
several, so absolute errors are a lower-bound proxy; paired differences between
checkpoints are the quantity of interest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..state_bank.io import write_json_atomic
from .flow_trace import file_hash
from .stage_patching import cluster_ci
from .state_bank import load_state_bank

SCHEMA = "smolvla_offline_action_error_v1"


def load_actions(trace: Path):
    """State IDs, executed actions [state, repeat, chunk, 7], and binding, without hidden arrays."""
    manifest = json.loads((trace / "manifest.json").read_text())
    binding = json.loads((trace / "binding.json").read_text())
    if not manifest.get("complete") or binding.get("query_mode") != "natural_integration" \
            or binding.get("flow_edit") is not None:
        raise ValueError(f"need a complete unedited natural trace: {trace}")
    ids, actions = [], []
    for relative in manifest["shards"]:
        with np.load(trace / relative, allow_pickle=False) as shard:
            ids.extend(shard["state_ids"].tolist())
            actions.append(shard["action_postprocessed"])
    if ids != binding["state_ids"]:
        raise ValueError(f"trace state order mismatch: {trace}")
    return ids, np.concatenate(actions), binding


def per_state_errors(policy: np.ndarray, demo: np.ndarray) -> dict[str, np.ndarray]:
    """policy [state, repeat, H, 7], demo [state, H, 7] -> per-state means over repeats and steps."""
    delta = policy - demo[:, None]
    return {
        "translation_l2": np.linalg.norm(delta[..., :3], axis=-1).mean(axis=(1, 2)),
        "rotation_l2": np.linalg.norm(delta[..., 3:6], axis=-1).mean(axis=(1, 2)),
        "gripper_agreement": (np.sign(policy[..., 6]) == np.sign(demo[:, None, :, 6])).mean(axis=(1, 2)),
        "noise_spread": policy.std(axis=1).mean(axis=(1, 2)),
    }


def run(bank: Path, checkpoints, output: Path, *, horizon: int):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    records, manifest, _, _ = load_state_bank(bank)
    if not manifest.get("audit_passed"):
        raise ValueError("StateBank audit has not passed")
    by_id = {r.state_id: r for r in records}
    by_frame = {(r.suite, r.task_id, r.source_episode_id, r.frame_index): r for r in records}
    reference_ids, results = None, {}
    for name, trace in checkpoints:
        ids, actions, binding = load_actions(trace)
        if binding["state_bank_sha256"] != file_hash(bank / "manifest.json"):
            raise ValueError(f"{name}: StateBank binding mismatch")
        if reference_ids is None:
            reference_ids = ids
            rows = [by_id[i] for i in ids]
            demo, keep = [], []
            for row in rows:
                chunk = [by_frame.get((row.suite, row.task_id, row.source_episode_id, row.frame_index + k))
                         for k in range(horizon)]
                keep.append(all(c is not None for c in chunk))
                demo.append([c.observation.action if c else [np.nan] * 7 for c in chunk])
            keep, demo = np.asarray(keep), np.asarray(demo, dtype=float)
            clusters = [f"{r.suite}:{r.task_id}:{r.source_episode_id}" for r in rows]
            phases = np.asarray([r.labels.phase for r in rows])
        elif ids != reference_ids:
            raise ValueError(f"{name}: states are not paired with the first checkpoint")
        results[name] = per_state_errors(actions[keep, :, :horizon], demo[keep])
    kept_clusters = [c for c, k in zip(clusters, keep) if k]
    kept_phases = phases[keep]
    names = [name for name, _ in checkpoints]
    summary = {name: {metric: cluster_ci(values, kept_clusters) for metric, values in metrics.items()}
               for name, metrics in results.items()}
    by_phase = {name: {phase: {metric: float(values[kept_phases == phase].mean())
                               for metric, values in metrics.items()}
                       for phase in sorted(set(kept_phases))}
                for name, metrics in results.items()}
    last = names[-1]
    paired = {name: {metric: cluster_ci(results[name][metric] - results[last][metric], kept_clusters)
                     for metric in results[name]}
              for name in names[:-1]}
    report = {"schema": SCHEMA, "exploration_only": True, "source_sha256": file_hash(Path(__file__)),
              "horizon": horizon, "states": int(keep.sum()), "dropped_near_episode_end": int((~keep).sum()),
              "episodes": len(set(kept_clusters)),
              "inputs": {name: str(Path(trace).resolve()) for name, trace in checkpoints},
              "summary": summary, "by_phase": by_phase,
              "paired_minus_last": paired,
              "phase_counts": {phase: int((kept_phases == phase).sum()) for phase in sorted(set(kept_phases))},
              "notes": ["demonstration states (in-distribution), not the policy's own closed-loop states",
                        "demonstration is one valid action sequence; compare checkpoints, not absolutes"]}
    write_json_atomic(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--checkpoint", nargs=2, action="append", required=True, metavar=("NAME", "TRACE"),
                   help="in training order; the last one is the paired reference")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=10, help="executed actions per chunk")
    a = p.parse_args()
    report = run(a.bank, [(n, Path(t)) for n, t in a.checkpoint], a.output, horizon=a.horizon)
    print(json.dumps({k: report[k] for k in ("states", "episodes", "summary")}, indent=2))


if __name__ == "__main__":
    main()
