"""Paired E2 summary of two complete SmolVLA conditional-flow traces."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .flow_trace import SCHEMA as TRACE_SCHEMA, file_hash


SCHEMA = "smolvla_paired_flow_diff_v1"
STAGE_KEYS = ("x_sigma", "velocity", "expert_middle", "expert_late")


def load_trace(cache: Path):
    manifest = json.loads((cache / "manifest.json").read_text())
    binding = json.loads((cache / "binding.json").read_text())
    if manifest.get("schema") != TRACE_SCHEMA or not manifest.get("complete"):
        raise ValueError(f"incomplete or incompatible trace: {cache}")
    if manifest["binding_sha256"] != file_hash(cache / "binding.json"):
        raise ValueError(f"trace binding hash mismatch: {cache}")
    arrays = {}
    ids = []
    for relative in manifest["shards"]:
        path = cache / relative
        receipt = json.loads(path.with_suffix(".json").read_text())
        if receipt["binding_sha256"] != manifest["binding_sha256"] or receipt["sha256"] != file_hash(path):
            raise ValueError(f"trace shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as shard:
            ids.extend(shard["state_ids"].tolist())
            for key in shard.files:
                if key not in {"state_ids", "sigma"}:
                    arrays.setdefault(key, []).append(shard[key].copy())
            arrays.setdefault("sigma", []).append(shard["sigma"].copy())
    result = {key: values[0] if key == "sigma" else np.concatenate(values)
              for key, values in arrays.items()}
    if ids != binding["state_ids"] or len(ids) != manifest["states"]:
        raise ValueError("trace state coverage/order mismatch")
    if any(not np.array_equal(result["sigma"], item) for item in arrays["sigma"][1:]):
        raise ValueError("sigma differs across shards")
    return np.asarray(ids), result, binding, manifest


def _rms(values, axes):
    return np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=axes))


def _stage_summary(delta, before, after):
    axes = tuple(range(3, delta.ndim))
    paired = _rms(delta, axes)
    mean_delta = delta.mean(axis=1, keepdims=True)
    signal = _rms(mean_delta, (1,) + axes)
    residual = _rms(delta - mean_delta, axes).mean(axis=1)
    before_noise = _rms(before - before.mean(axis=1, keepdims=True), axes).mean(axis=1)
    after_noise = _rms(after - after.mean(axis=1, keepdims=True), axes).mean(axis=1)
    return {
        "paired_rms": paired.mean(axis=(0, 1)).tolist(),
        "repeat_mean_signal_rms": signal.mean(axis=0).tolist(),
        "repeat_residual_rms": residual.mean(axis=0).tolist(),
        "before_repeat_sensitivity_rms": before_noise.mean(axis=0).tolist(),
        "after_repeat_sensitivity_rms": after_noise.mean(axis=0).tolist(),
    }, paired


def compare(before, after):
    ids_a, a, binding_a, manifest_a = before
    ids_b, b, binding_b, manifest_b = after
    if not np.array_equal(ids_a, ids_b):
        raise ValueError("state IDs/order differ")
    for key in ("epsilon", "sigma"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"paired traces have different {key}")
    for key in ("state_bank_sha256", "dataset_revision", "dataset_scientific_sha256",
                "contract_tree_sha256", "num_steps", "chunk_size", "max_action_dim"):
        if key not in binding_a or key not in binding_b or binding_a[key] != binding_b[key]:
            raise ValueError(f"paired trace contract differs: {key}")
    required = STAGE_KEYS + ("action_postprocessed",)
    if any(key not in a or key not in b or a[key].shape != b[key].shape for key in required):
        raise ValueError("paired trace arrays are missing or shape-incompatible")

    sigma = a["sigma"][0]
    evidence = {"state_ids": ids_a, "sigma": sigma}
    stages = {}
    for key in STAGE_KEYS:
        summary, paired = _stage_summary(b[key] - a[key], a[key], b[key])
        stages[key] = summary
        evidence[f"{key}_paired_rms"] = paired

    velocity_delta = b["velocity"] - a["velocity"]
    action_dim = a["action_postprocessed"].shape[-1]
    groups = {"translation": slice(0, min(3, action_dim)),
              "rotation": slice(3, min(6, action_dim)),
              "gripper": slice(6, min(7, action_dim)),
              "executed": slice(0, action_dim),
              "padding": slice(action_dim, velocity_delta.shape[-1])}
    velocity_groups = {}
    for name, columns in groups.items():
        values = velocity_delta[..., columns]
        if values.shape[-1]:
            paired = _rms(values, (-2, -1))
            velocity_groups[name] = paired.mean(axis=(0, 1)).tolist()
            evidence[f"velocity_{name}_paired_rms"] = paired

    action_delta = b["action_postprocessed"] - a["action_postprocessed"]
    action = {}
    for name, columns in groups.items():
        if name == "padding" or columns.stop > action_dim:
            continue
        chunk = action_delta[..., columns]
        if chunk.shape[-1]:
            action[name] = {
                "first_action_rms": float(_rms(chunk[:, :, 0], (-1,)).mean()),
                "chunk_rms": float(_rms(chunk, (-2, -1)).mean()),
            }
    evidence["first_action_delta"] = action_delta[:, :, 0]
    evidence["action_chunk_delta_rms"] = _rms(action_delta, (-2, -1))
    report = {
        "schema": SCHEMA, "states": len(ids_a), "noise_repeats": int(a["epsilon"].shape[1]),
        "sigma": sigma.tolist(), "stage_metrics": stages,
        "velocity_component_paired_rms": velocity_groups, "action_metrics": action,
        "before_binding_sha256": manifest_a["binding_sha256"],
        "after_binding_sha256": manifest_b["binding_sha256"],
        "interpretation": "Exploratory paired model diff; differences do not establish capability or causal use.",
    }
    return report, evidence


def run(before: Path, after: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    report, evidence = compare(load_trace(before), load_trace(after))
    buffer = io.BytesIO()
    np.savez(buffer, **evidence)
    write_bytes_atomic(output / "paired_evidence.npz", buffer.getvalue())
    report["paired_evidence_sha256"] = file_hash(output / "paired_evidence.npz")
    report["inputs"] = {"before": str(before.resolve()), "after": str(after.resolve())}
    write_json_atomic(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.before, args.after, args.output), indent=2))


if __name__ == "__main__":
    main()
