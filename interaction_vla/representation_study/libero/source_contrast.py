"""Fixed-flow source contrasts for one frozen SmolVLA candidate direction."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .flow_diff import load_trace
from .flow_trace import (action_atlas_provenance, bind_policy_images, file_hash,
                         load_frozen_policy, query_action_flow_at_points)
from .latents import collate_state_bank_observations
from .state_bank import load_state_bank


def _geometry(record) -> np.ndarray:
    value = record.labels.geometry
    if value is None:
        return np.asarray((np.nan, np.nan), dtype=float)
    return np.asarray((value.gripper_target_distance, value.target_goal_distance), dtype=float)


def matched_donors(records) -> dict[str, np.ndarray]:
    """Freeze cross-episode donors using observed state, time, and geometry only."""
    robot = np.asarray([row.observation.robot_state for row in records], dtype=float)
    geometry = np.asarray([_geometry(row) for row in records])
    time = np.asarray([row.frame_index for row in records], dtype=float)[:, None]
    features = np.column_stack((robot, geometry, time))
    mean, scale = np.nanmean(features, axis=0), np.nanstd(features, axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = np.where(np.isfinite(features), (features - mean) / scale, 0.0)
    visual, proprio = [], []
    for index, row in enumerate(records):
        pool = np.asarray([
            other for other, candidate in enumerate(records)
            if candidate.task_id == row.task_id
            and candidate.source_episode_id != row.source_episode_id
        ], dtype=int)
        if not len(pool):
            raise ValueError("source contrast requires a cross-episode donor per task")
        robot_time = np.linalg.norm(
            standardized[pool][:, np.r_[0:8, 10]]
            - standardized[index, np.r_[0:8, 10]], axis=1)
        geometry_distance = np.linalg.norm(
            standardized[pool][:, 8:10] - standardized[index, 8:10], axis=1)
        threshold = np.median(geometry_distance)
        eligible = geometry_distance >= threshold
        visual.append(pool[np.argmin(np.where(eligible, robot_time, np.inf))])
        threshold = np.median(geometry_distance)
        eligible = geometry_distance <= threshold
        robot_distance = np.linalg.norm(
            standardized[pool][:, :8] - standardized[index, :8], axis=1)
        proprio.append(pool[np.argmax(np.where(eligible, robot_distance, -np.inf))])
    return {"visual_geometry": np.asarray(visual), "robot_state": np.asarray(proprio)}


def donor_diagnostics(records, donors: dict[str, np.ndarray]) -> dict[str, object]:
    """Report coverage and standardized residual distances for frozen donors."""
    robot = np.asarray([row.observation.robot_state for row in records], dtype=float)
    geometry = np.asarray([_geometry(row) for row in records], dtype=float)
    time = np.asarray([row.frame_index for row in records], dtype=float)[:, None]
    features = np.column_stack((robot, geometry, time))
    mean, scale = np.nanmean(features, axis=0), np.nanstd(features, axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = np.nan_to_num((features - mean) / scale)
    report = {}
    for mode, indices in donors.items():
        valid = np.asarray(indices, dtype=int)
        if len(valid) != len(records) or np.any(valid < 0) or np.any(valid >= len(records)):
            raise ValueError(f"invalid donor coverage for {mode}")
        delta = standardized - standardized[valid]
        report[mode] = {
            "coverage": float(len(valid) / max(len(records), 1)),
            "mean_distance": float(np.linalg.norm(delta, axis=1).mean()),
            "p95_distance": float(np.quantile(np.linalg.norm(delta, axis=1), 0.95)),
            "mean_robot_distance": float(np.linalg.norm(delta[:, :8], axis=1).mean()),
            "mean_geometry_distance": float(np.linalg.norm(delta[:, 8:10], axis=1).mean()),
            "mean_time_distance": float(np.abs(delta[:, 10]).mean()),
        }
    return report


def _swap(batch, donor, mode: str):
    result = dict(batch)
    if mode == "visual_geometry":
        keys = [key for key in batch if key.startswith("observation.images.")]
    elif mode == "robot_state":
        keys = ["observation.state"]
    elif mode == "no_op":
        return result
    else:
        raise ValueError(f"unknown source contrast: {mode}")
    for key in keys:
        if key not in batch or key not in donor:
            raise ValueError(f"source contrast batch is missing {key}")
        result[key] = donor[key]
    return result


def run(bank: Path, dataset_root: Path, checkpoint: Path, contract: Path,
        metadata: Path, reference_trace: Path, candidates: Path, candidate_id: str,
        output: Path, *, device: str = "cuda", batch_size: int = 4,
        max_states: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"source contrast output exists: {output}")
    if batch_size <= 0 or (max_states is not None and max_states <= 0):
        raise ValueError("source contrast batch/max-state settings are invalid")
    state_ids, reference, reference_binding, reference_manifest = load_trace(reference_trace)
    if reference_binding.get("query_mode") != "natural_integration":
        raise ValueError("source contrast requires a natural reference trace")
    records, manifest, _, _ = load_state_bank(bank)
    if not manifest.get("audit_passed"):
        raise ValueError("StateBank audit has not passed")
    by_id = {row.state_id: row for row in records}
    if max_states is not None and max_states < len(state_ids):
        groups = {}
        for index, state_id in enumerate(state_ids.tolist()):
            row = by_id[state_id]
            groups.setdefault((row.task_id, row.source_episode_id), []).append(index)
        chosen = []
        while len(chosen) < max_states and any(groups.values()):
            for key in sorted(groups):
                if groups[key] and len(chosen) < max_states:
                    chosen.append(groups[key].pop(0))
        indices = np.asarray(chosen, dtype=int)
        reference = {name: (value[indices] if value.shape[:1] == (len(state_ids),) else value)
                     for name, value in reference.items()}
        state_ids = state_ids[indices]
    selected = [by_id[state_id] for state_id in state_ids.tolist()]
    donors = matched_donors(selected)
    donor_report = donor_diagnostics(selected, donors)
    artifact = json.loads(candidates.read_text(encoding="utf-8"))
    row = next((item for item in artifact["candidates"] if item["id"] == candidate_id), None)
    if row is None:
        raise ValueError(f"candidate not found: {candidate_id}")
    direction = np.asarray(row["direction"], dtype=np.float32)
    tap = row["tap"]

    action_atlas_provenance()
    from experiments.model_adapters import SmolVLAAdapter
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    revision = next(iter({record.source_revision for record in selected}))
    dataset = LeRobotDataset("lerobot/libero", root=dataset_root, revision=revision,
                             episodes=sorted({row.lerobot_episode_index for row in selected}),
                             video_backend="torchcodec")
    policy, pre, _ = load_frozen_policy(checkpoint, contract, metadata, device)
    adapter = SmolVLAAdapter(); adapter.policy = policy
    layers = adapter.get_layer_groups()["expert"]
    middle, late = layers[len(layers) // 2], layers[-1]
    modes = ("no_op", "visual_geometry", "robot_state")
    values = {mode: {"projection": [], "velocity_rms": []} for mode in modes}
    for start in range(0, len(selected), batch_size):
        stop = min(start + batch_size, len(selected))
        receiver_rows = selected[start:stop]
        receiver, _ = bind_policy_images(
            collate_state_bank_observations(receiver_rows, dataset), policy.config.input_features)
        for mode in modes:
            if mode == "no_op":
                raw = receiver
            else:
                donor_rows = [selected[index] for index in donors[mode][start:stop]]
                donor, _ = bind_policy_images(
                    collate_state_bank_observations(donor_rows, dataset),
                    policy.config.input_features)
                raw = _swap(receiver, donor, mode)
            processed = pre(raw)
            projections, velocity = [], []
            for repeat in range(reference["epsilon"].shape[1]):
                noise = torch.from_numpy(reference["epsilon"][start:stop, repeat]).to(device)
                trace = query_action_flow_at_points(
                    policy, processed, noise,
                    reference["x_sigma"][start:stop, repeat],
                    reference["sigma"][repeat], middle, late)
                hidden = trace[tap]
                projections.append(np.einsum("nstd,d->ns", hidden, direction) / hidden.shape[2])
                delta = trace["velocity"] - reference["velocity"][start:stop, repeat]
                velocity.append(np.sqrt(np.mean(delta ** 2, axis=(2, 3))))
            values[mode]["projection"].append(np.stack(projections, axis=1))
            values[mode]["velocity_rms"].append(np.stack(velocity, axis=1))
        print(json.dumps({"states": stop, "expected": len(selected)}), flush=True)
    arrays = {f"{mode}_{name}": np.concatenate(parts).astype(np.float32)
              for mode, metrics in values.items() for name, parts in metrics.items()}
    buffer = io.BytesIO()
    np.savez(buffer, state_ids=state_ids,
             visual_donor_state_ids=state_ids[donors["visual_geometry"]],
             robot_donor_state_ids=state_ids[donors["robot_state"]], **arrays)
    write_bytes_atomic(output / "contrasts.npz", buffer.getvalue())
    summary = {}
    natural_projection = np.einsum(
        "nrstd,d->nrs", reference[tap], direction
    ) / reference[tap].shape[3]
    no_op_projection = arrays["no_op_projection"]
    for mode in modes:
        projection_delta = arrays[f"{mode}_projection"] - natural_projection
        source_increment = arrays[f"{mode}_projection"] - no_op_projection
        summary[mode] = {
            "projection_delta_vs_natural_rms_by_stage": np.sqrt(
                np.mean(projection_delta ** 2, axis=(0, 1))).tolist(),
            "projection_increment_over_no_op_rms_by_stage": np.sqrt(
                np.mean(source_increment ** 2, axis=(0, 1))).tolist(),
            "velocity_delta_rms_by_stage": arrays[f"{mode}_velocity_rms"].mean(
                axis=(0, 1)).tolist(),
        }
    report = {
        "schema": "smolvla_source_contrast_v1", "complete": True,
        "candidate_id": candidate_id, "candidate_sha256": file_hash(candidates),
        "tap": tap, "states": len(selected),
        "independent_episodes": len({(r.task_id, r.source_episode_id) for r in selected}),
        "reference_binding_sha256": reference_manifest["binding_sha256"],
        "donor_rule": {
            "shared": "same task, different source episode; no future outcome used",
            "visual_geometry": "geometry-different half, minimum standardized robot+time distance",
            "robot_state": "geometry-near half, maximum standardized robot-state distance",
        },
        "donor_diagnostics": donor_report,
        "fixed_inputs": ["state_ids", "epsilon", "x_sigma", "sigma", "language"],
        "contrasts_sha256": file_hash(output / "contrasts.npz"), "summary": summary,
    }
    write_json_atomic(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract-checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-id", default="formation_0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-states", type=int)
    args = parser.parse_args()
    result = run(args.bank, args.dataset_root, args.checkpoint,
                 args.contract_checkpoint, args.metadata, args.reference_trace,
                 args.candidates, args.candidate_id, args.output,
                 device=args.device, batch_size=args.batch_size,
                 max_states=args.max_states)
    print(json.dumps(result, indent=2))
    print(f"ALL_DONE output={args.output} states={result['states']}", flush=True)


if __name__ == "__main__":
    main()
