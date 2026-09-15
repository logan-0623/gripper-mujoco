"""Capability timeline and label-blind model-diff pipeline for one SmolVLA lineage."""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .flow_diff import load_trace
from .flow_trace import file_hash
from .latents import _tree_sha256
from .state_bank import load_state_bank


SCHEMA = "smolvla_acquisition_v1"
EVENT_KEYS = (
    "contact", "stable_grasp", "geometric_lift", "supported_lift",
    "normal_release", "unintended_drop", "recovered_after_drop", "success",
)


def checkpoint_lineage(checkpoint_root: Path, base_checkpoint: Path,
                       steps: Sequence[int], output: Path) -> dict:
    if not steps or list(steps) != sorted(set(steps)) or min(steps) <= 0:
        raise ValueError("checkpoint steps must be unique, positive, and increasing")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen lineage: {output}")
    if not (base_checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"base checkpoint is incomplete: {base_checkpoint}")
    rows, contracts = [], []
    for step in steps:
        checkpoint = checkpoint_root / f"{step:06d}" / "pretrained_model"
        state = checkpoint.parent / "training_state" / "training_step.json"
        config = checkpoint / "train_config.json"
        if not checkpoint.is_dir() or not state.is_file() or not config.is_file():
            raise FileNotFoundError(f"incomplete checkpoint at step {step}: {checkpoint}")
        recorded = json.loads(state.read_text(encoding="utf-8"))
        observed = int(recorded.get("step", recorded.get("training_step", -1)))
        if observed != step:
            raise ValueError(f"checkpoint directory/recorded step mismatch: {step} != {observed}")
        train_config = json.loads(config.read_text(encoding="utf-8"))
        contracts.append({key: train_config.get(key) for key in (
            "seed", "batch_size", "steps", "optimizer", "scheduler", "rename_map"
        )} | {"dataset": train_config.get("dataset"), "policy_training": {
            key: train_config.get("policy", {}).get(key) for key in (
                "pretrained_path", "freeze_vision_encoder", "train_expert_only",
                "train_state_proj", "load_vlm_weights",
            )
        }})
        rows.append({
            "step": step,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _tree_sha256(checkpoint),
            "train_config_sha256": file_hash(config),
            "training_state_sha256": file_hash(state),
        })
    if any(value != contracts[0] for value in contracts[1:]):
        raise ValueError("training contract changes across checkpoints")
    report = {"schema": SCHEMA, "kind": "immutable_lineage",
              "base_checkpoint": str(base_checkpoint.resolve()),
              "base_checkpoint_sha256": _tree_sha256(base_checkpoint),
              "training_contract": contracts[0], "checkpoints": rows}
    write_json_atomic(output, report)
    return report


def _evaluation_command(checkpoint: Path, task: int, output: Path,
                        initial_state_offset: int, episodes: int) -> tuple[str, ...]:
    return (
        sys.executable, "-W", "error::RuntimeWarning", "-m",
        "interaction_vla.representation_study.libero.capability_events",
        "--events-output", str(output / "physical_events.json"),
        "--suite", "libero_spatial", "--initial-state-offset", str(initial_state_offset),
        "--rendered-episodes", "0", "--",
        f"--policy.path={checkpoint}", "--policy.device=cuda", "--policy.use_amp=false",
        "--policy.n_action_steps=10", "--policy.num_steps=10", "--policy.empty_cameras=1",
        "--env.type=libero", "--env.task=libero_spatial", f"--env.task_ids=[{task}]",
        "--env.obs_type=pixels_agent_pos", "--env.init_states=true", "--env.fps=30",
        "--env.max_parallel_tasks=1",
        '--env.camera_name_mapping={"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}',
        f"--eval.n_episodes={episodes}", "--eval.batch_size=1",
        "--eval.use_async_envs=false", "--eval.recording=false", "--seed=2057736129",
        f"--output_dir={output}",
    )


def evaluate_timeline(lineage: Path, output: Path, tasks: Sequence[int], *,
                      initial_state_offset: int, episodes: int, dry_run: bool) -> dict:
    if not tasks or min(tasks) < 0 or initial_state_offset < 0 or episodes <= 0:
        raise ValueError("tasks, initial-state offset, and episode count are invalid")
    manifest = json.loads(lineage.read_text(encoding="utf-8"))
    if manifest.get("kind") != "immutable_lineage":
        raise ValueError("lineage manifest is incompatible")
    commands = []
    for row in manifest["checkpoints"]:
        checkpoint = Path(row["checkpoint"])
        if _tree_sha256(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"checkpoint changed: {checkpoint}")
        for task in tasks:
            destination = output / f"step_{int(row['step']):06d}" / f"task{task}"
            command = _evaluation_command(
                checkpoint, task, destination, initial_state_offset, episodes
            )
            commands.append(list(command))
            if dry_run:
                continue
            if (destination / "eval_info.json").is_file() and (
                destination / "physical_events.json"
            ).is_file():
                continue
            if destination.exists():
                raise FileExistsError(f"incomplete evaluation output: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
    plan = {
        "schema": SCHEMA, "kind": "timeline_evaluation",
        "lineage_sha256": file_hash(lineage), "tasks": list(tasks),
        "initial_state_offset": initial_state_offset, "episodes_per_task": episodes,
        "dry_run": dry_run, "commands": commands,
    }
    write_json_atomic(output / "evaluation_plan.json", plan)
    return plan


def summarize_timeline(lineage: Path, root: Path, output: Path) -> dict:
    manifest = json.loads(lineage.read_text(encoding="utf-8"))
    plan = json.loads((root / "evaluation_plan.json").read_text(encoding="utf-8"))
    if plan["lineage_sha256"] != file_hash(lineage):
        raise ValueError("timeline evaluation belongs to another lineage")
    rows = []
    for checkpoint in manifest["checkpoints"]:
        step = int(checkpoint["step"])
        for task in plan["tasks"]:
            directory = root / f"step_{step:06d}" / f"task{task}"
            events_path, eval_path = directory / "physical_events.json", directory / "eval_info.json"
            events = json.loads(events_path.read_text(encoding="utf-8"))["episodes"]
            if [int(row["initial_state_id"]) for row in events] != list(range(
                int(plan["initial_state_offset"]),
                int(plan["initial_state_offset"]) + int(plan["episodes_per_task"]),
            )):
                raise ValueError(f"initial-state contract differs: {directory}")
            successes = json.loads(eval_path.read_text(encoding="utf-8"))[
                "per_task"
            ][0]["metrics"]["successes"]
            row = {"step": step, "task": int(task), "episodes": len(events)}
            row.update({key: sum(bool(item.get(key)) for item in events) for key in EVENT_KEYS})
            if row["success"] != sum(bool(value) for value in successes):
                raise ValueError(f"success records disagree: {directory}")
            row["eval_info_sha256"] = file_hash(eval_path)
            row["physical_events_sha256"] = file_hash(events_path)
            rows.append(row)
    aggregate = []
    for step in sorted({row["step"] for row in rows}):
        selected = [row for row in rows if row["step"] == step]
        aggregate.append({"step": step, **{
            key: sum(int(row[key]) for row in selected) for key in ("episodes", *EVENT_KEYS)
        }})
    report = {"schema": SCHEMA, "kind": "capability_timeline", "rows": rows,
              "aggregate": aggregate, "lineage_sha256": file_hash(lineage)}
    write_json_atomic(output / "report.json", report)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    (output / "results.csv").parent.mkdir(parents=True, exist_ok=True)
    (output / "results.csv").write_text(buffer.getvalue(), encoding="utf-8")
    return report


def _orthogonal_random(existing: list[np.ndarray], dimension: int,
                       count: int, seed: int) -> list[np.ndarray]:
    rng, result = np.random.default_rng(seed), []
    basis = [row / np.linalg.norm(row) for row in existing]
    while len(result) < count:
        value = rng.normal(size=dimension)
        for direction in (*basis, *result):
            value -= np.dot(value, direction) * direction
        norm = np.linalg.norm(value)
        if norm > 1e-8:
            result.append(value / norm)
    return result


def discover_change_subspaces(traces: Mapping[str, Path], *, before: str, after: str,
                              tap: str, rank: int, output: Path) -> dict:
    if len(traces) < 2 or rank <= 0:
        raise ValueError("discovery requires at least two traces and positive rank")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen candidates: {output}")
    if tap not in {"expert_middle", "expert_late"} or before == after:
        raise ValueError("tap and distinct endpoints are required")
    loaded = {name: load_trace(path) for name, path in traces.items()}
    if before not in loaded or after not in loaded:
        raise ValueError("before/after labels must name supplied traces")
    reference_ids = loaded[before][0]
    shapes = {value[1][tap].shape for value in loaded.values()}
    if len(shapes) != 1 or any(not np.array_equal(value[0], reference_ids) for value in loaded.values()):
        raise ValueError("traces must have aligned states and activation shapes")
    query_modes = {value[2].get("query_mode", "natural_integration")
                   for value in loaded.values()}
    if len(query_modes) != 1:
        raise ValueError("discovery traces must use one query mode")
    reference_arrays = loaded[before][1]
    paired_keys = ("epsilon", "sigma", "x_sigma") if query_modes == {
        "fixed_reference_points"
    } else ("epsilon", "sigma")
    if any(not np.array_equal(value[1][key], reference_arrays[key])
           for value in loaded.values() for key in paired_keys):
        raise ValueError("discovery traces do not share fixed flow inputs")
    # State and solver stage are the sampling units; token/noise replicates are averaged.
    state_stage = {
        name: value[1][tap].mean(axis=(1, 3)) for name, value in loaded.items()
    }
    pooled = np.concatenate([value.reshape(-1, value.shape[-1]) for value in state_stage.values()])
    mean, scale = pooled.mean(0), pooled.std(0)
    scale[scale < 1e-8] = 1.0
    standardized = (pooled - mean) / scale
    _, singular, vh = np.linalg.svd(standardized, full_matrices=False)
    effective_rank = min(rank, vh.shape[0])
    components = vh[:effective_rank]
    delta = (state_stage[after] - state_stage[before]) / scale
    coefficients = delta.reshape(-1, delta.shape[-1]) @ components.T
    _, change_singular, change_vh = np.linalg.svd(coefficients, full_matrices=False)

    formation = []
    for index in range(min(4, len(change_vh))):
        raw = scale * (change_vh[index] @ components)
        raw /= np.linalg.norm(raw)
        formation.append(raw)
    energy = np.mean(np.square(coefficients), axis=0)
    controls = []
    for component_index in np.argsort(energy)[:min(4, effective_rank)]:
        raw = scale * components[component_index]
        raw /= np.linalg.norm(raw)
        controls.append(raw)
    randoms = _orthogonal_random(formation + controls, pooled.shape[1], 4, 2057736129)
    candidates = []
    for role, directions in (("formation", formation), ("low_change", controls),
                             ("matched_random", randoms)):
        for index, direction in enumerate(directions):
            candidates.append({"id": f"{role}_{index}", "role": role, "tap": tap,
                               "direction": direction.astype(float).tolist()})
    basis = io.BytesIO()
    np.savez(basis, mean=mean, scale=scale, components=components,
             pca_singular_values=singular[:effective_rank],
             change_singular_values=change_singular, state_ids=reference_ids)
    write_bytes_atomic(output / "shared_basis.npz", basis.getvalue())
    report = {
        "schema": SCHEMA, "kind": "frozen_change_candidates", "before": before,
        "after": after, "tap": tap, "rank": effective_rank,
        "selection_uses_physical_labels": False,
        "query_mode": next(iter(query_modes)),
        "sampling_unit": "state x solver_stage; token/noise averaged",
        "trace_bindings": {name: value[3]["binding_sha256"] for name, value in loaded.items()},
        "shared_basis_sha256": file_hash(output / "shared_basis.npz"),
        "candidates": candidates,
    }
    write_json_atomic(output / "candidates.json", report)
    return report


def candidate_trajectory(traces: Mapping[str, Path], candidates: Path,
                         output: Path) -> dict:
    """Project frozen candidates through checkpoints without refitting them."""
    if len(traces) < 2 or output.exists():
        raise ValueError("trajectory requires at least two traces and a new output directory")
    artifact = json.loads(candidates.read_text(encoding="utf-8"))
    rows = artifact.get("candidates", [])
    if not rows:
        raise ValueError("candidate artifact is empty")
    tap = artifact["tap"]
    directions = np.asarray([row["direction"] for row in rows], dtype=np.float64)
    loaded = {name: load_trace(path) for name, path in traces.items()}
    reference_ids = next(iter(loaded.values()))[0]
    query_modes = {value[2].get("query_mode", "natural_integration")
                   for value in loaded.values()}
    if query_modes != {artifact.get("query_mode", next(iter(query_modes)))}:
        raise ValueError("trajectory query mode differs from frozen discovery")
    reference_arrays = next(iter(loaded.values()))[1]
    paired_keys = ("epsilon", "sigma", "x_sigma") if query_modes == {
        "fixed_reference_points"
    } else ("epsilon", "sigma")
    arrays = {}
    for name, (ids, values, _, _) in loaded.items():
        if not np.array_equal(ids, reference_ids) or tap not in values:
            raise ValueError(f"unaligned or incomplete trajectory trace: {name}")
        if any(not np.array_equal(values[key], reference_arrays[key]) for key in paired_keys):
            raise ValueError(f"trajectory trace does not share flow inputs: {name}")
        activations = values[tap]
        if activations.ndim != 5 or activations.shape[-1] != directions.shape[1]:
            raise ValueError("candidate direction and activation widths differ")
        arrays[name] = np.einsum(
            "nrsd,cd->ncrs", activations.mean(axis=3), directions, optimize=True
        )
    before, after = artifact["before"], artifact["after"]
    if before not in arrays or after not in arrays:
        raise ValueError("trajectory must include the frozen discovery endpoints")
    endpoint_delta = arrays[after] - arrays[before]
    summaries = []
    for name, values in arrays.items():
        mean = values.mean(axis=(0, 2))
        shift = values - arrays[before]
        dot = np.einsum("ncrs,ncrs->c", shift, endpoint_delta)
        endpoint_energy = np.einsum("ncrs,ncrs->c", endpoint_delta, endpoint_delta)
        shift_energy = np.einsum("ncrs,ncrs->c", shift, shift)
        coefficient = np.divide(dot, endpoint_energy,
                                out=np.full_like(dot, np.nan), where=endpoint_energy > 1e-12)
        cosine = np.divide(dot, np.sqrt(shift_energy * endpoint_energy),
                           out=np.full_like(dot, np.nan),
                           where=(shift_energy * endpoint_energy) > 1e-12)
        relative_rms = np.sqrt(np.divide(
            shift_energy, endpoint_energy, out=np.full_like(dot, np.nan),
            where=endpoint_energy > 1e-12,
        ))
        summaries.append({
            "checkpoint": name,
            "candidate_stage_mean": mean.tolist(),
            "candidate_stage_std": values.std(axis=(0, 2)).tolist(),
            "endpoint_projection_coefficient": coefficient.tolist(),
            "endpoint_cosine": cosine.tolist(),
            "relative_endpoint_rms": relative_rms.tolist(),
        })
    buffer = io.BytesIO()
    np.savez(buffer, state_ids=reference_ids,
             checkpoint_names=np.asarray(list(arrays)),
             candidate_ids=np.asarray([row["id"] for row in rows]),
             **{f"projection_{name}": value.astype(np.float32)
                for name, value in arrays.items()})
    write_bytes_atomic(output / "projections.npz", buffer.getvalue())
    report = {
        "schema": SCHEMA, "kind": "frozen_candidate_trajectory",
        "candidate_sha256": file_hash(candidates), "tap": tap,
        "selection_uses_physical_labels": False,
        "state_ids": reference_ids.tolist(), "summaries": summaries,
        "trace_bindings": {name: value[3]["binding_sha256"]
                           for name, value in loaded.items()},
        "projections_sha256": file_hash(output / "projections.npz"),
    }
    write_json_atomic(output / "report.json", report)
    return report


def _incremental_r2(y: np.ndarray, nuisance: np.ndarray, target: np.ndarray) -> float:
    keep = np.isfinite(y) & np.isfinite(target) & np.isfinite(nuisance).all(axis=1)
    y, nuisance, target = y[keep], nuisance[keep], target[keep]
    if len(y) < nuisance.shape[1] + 3 or np.var(y) < 1e-12:
        return float("nan")
    base = np.column_stack((np.ones(len(y)), nuisance))
    full = np.column_stack((base, target))
    error_base = np.square(y - base @ np.linalg.lstsq(base, y, rcond=None)[0]).sum()
    error_full = np.square(y - full @ np.linalg.lstsq(full, y, rcond=None)[0]).sum()
    return float((error_base - error_full) / error_base) if error_base > 1e-12 else float("nan")


def interpret_candidates(trajectory: Path, state_bank: Path, checkpoint: str,
                         output: Path) -> dict:
    """Interpret already-frozen candidates; never select candidates here."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite interpretation: {output}")
    report = json.loads((trajectory / "report.json").read_text(encoding="utf-8"))
    if report.get("kind") != "frozen_candidate_trajectory" or (
        report["projections_sha256"] != file_hash(trajectory / "projections.npz")
    ):
        raise ValueError("trajectory artifact is incomplete or changed")
    records, _, _, _ = load_state_bank(state_bank)
    by_id = {row.state_id: row for row in records}
    with np.load(trajectory / "projections.npz", allow_pickle=False) as data:
        ids = data["state_ids"].tolist()
        key = f"projection_{checkpoint}"
        if key not in data.files or any(item not in by_id for item in ids):
            raise ValueError("checkpoint or StateBank coverage differs from trajectory")
        values = data[key].mean(axis=2)  # state x candidate x solver stage
        candidate_ids = data["candidate_ids"].tolist()
    selected = [by_id[item] for item in ids]
    task_values = sorted({row.task_id for row in selected})
    nuisance = np.column_stack((
        np.asarray([row.frame_index for row in selected], dtype=float),
        *[np.asarray([row.task_id == task for row in selected], dtype=float)
          for task in task_values[1:]],
    ))
    labels = {
        "contact": np.asarray([
            float(row.labels.contact.gripper_target)
            if row.labels.contact is not None else np.nan for row in selected
        ]),
        "stable_grasp": np.asarray([
            float(row.labels.stable_grasp)
            if row.labels.stable_grasp is not None else np.nan for row in selected
        ]),
        "gripper_target_distance": np.asarray([
            row.labels.geometry.gripper_target_distance
            if row.labels.geometry is not None else np.nan for row in selected
        ]),
        "target_goal_distance": np.asarray([
            row.labels.geometry.target_goal_distance
            if row.labels.geometry is not None else np.nan for row in selected
        ]),
        "action_norm": np.asarray([np.linalg.norm(row.observation.action) for row in selected]),
        "action_gripper": np.asarray([row.observation.action[6] for row in selected]),
    }
    phases = sorted({row.labels.phase for row in selected if row.labels.phase is not None})
    for phase in phases:
        labels[f"phase:{phase}"] = np.asarray([
            float(row.labels.phase == phase) if row.labels.phase is not None else np.nan
            for row in selected
        ])
    rows = []
    for candidate_index, candidate_id in enumerate(candidate_ids):
        for stage in range(values.shape[2]):
            y = values[:, candidate_index, stage]
            rows.append({
                "candidate_id": candidate_id, "stage": stage,
                "conditional_incremental_r2": {
                    name: _incremental_r2(y, nuisance, target)
                    for name, target in labels.items()
                },
            })
    result = {
        "schema": SCHEMA, "kind": "post_freeze_candidate_interpretation",
        "trajectory_sha256": file_hash(trajectory / "report.json"),
        "state_bank_sha256": file_hash(state_bank / "manifest.json"),
        "checkpoint": checkpoint, "selection_performed": False,
        "nuisance": ["frame_index", "task_identity"],
        "action_fields_are_leakage_diagnostics": True,
        "unavailable_state_labels": ["geometric_lift", "supported_lift", "success"],
        "closed_loop_metrics_are_reported_only_after_intervention": True,
        "rows": rows,
    }
    write_json_atomic(output, result)
    return result


def intervention_smoke_command(bank: Path, dataset_root: Path, checkpoint: Path,
                               contract: Path, metadata: Path, candidates: Path,
                               candidate_id: str, output: Path, *, device: str,
                               max_states: int, dose: float, stages: Sequence[int],
                               partition: str = "validation", split_group: str = "episode",
                               suite: str | None = None, task_ids: Sequence[int] = ()) -> dict:
    """Run one baseline plus signed, stage-local edits through the real trace path."""
    if max_states <= 0 or dose <= 0 or not stages or min(stages) < 0 or max(stages) >= 10:
        raise ValueError("smoke states, dose, and stages are invalid")
    groups = {"early": tuple(x for x in stages if x <= 2),
              "middle": tuple(x for x in stages if 3 <= x <= 6),
              "late": tuple(x for x in stages if x >= 7),
              "all": tuple(stages)}
    groups = {name: value for name, value in groups.items() if value}
    common = [sys.executable, "-W", "error::RuntimeWarning", "-m",
              "interaction_vla.representation_study.libero.flow_trace", "run",
              "--bank", str(bank), "--dataset-root", str(dataset_root),
              "--checkpoint", str(checkpoint), "--contract-checkpoint", str(contract),
              "--metadata", str(metadata), "--device", device, "--batch-size", "1",
              "--max-states", str(max_states), "--noise-repeats", "1",
              "--partition", partition, "--split-group", split_group]
    if suite is not None:
        common.extend(("--suite", suite))
    for task_id in task_ids:
        common.extend(("--task-id", str(task_id)))
    commands = [common + ["--output", str(output / "baseline")]]
    for name, selected_stages in groups.items():
        for sign in (-1, 1):
            destination = output / f"{name}_{sign:+d}"
            command = common + ["--output", str(destination), "--candidates", str(candidates),
                                "--candidate-id", candidate_id, "--dose", str(sign * dose)]
            for stage in selected_stages:
                command.extend(("--edit-stage", str(stage)))
            commands.append(command)
    for command in commands:
        destination = Path(command[command.index("--output") + 1])
        if (destination / "manifest.json").is_file():
            continue
        subprocess.run(command, check=True)
    result = {"schema": SCHEMA, "kind": "real_policy_intervention_smoke",
              "candidate_sha256": file_hash(candidates), "candidate_id": candidate_id,
              "dose": dose, "stage_groups": {key: list(value) for key, value in groups.items()},
              "max_states": max_states, "partition": partition, "split_group": split_group,
              "suite": suite, "task_ids": list(task_ids),
              "conditions": [Path(command[command.index("--output") + 1]).name
                                                         for command in commands]}
    write_json_atomic(output / "report.json", result)
    return result


def offline_action_gate(baseline: Path, edited: Mapping[str, Path], candidates: Path,
                        output: Path, *, bootstrap_samples: int = 10000) -> dict:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite offline gate: {output}")
    candidate_rows = {row["id"]: row for row in json.loads(
        candidates.read_text(encoding="utf-8")
    )["candidates"]}
    base_ids, base, _, _ = load_trace(baseline)
    effects = {}
    for candidate_id, path in edited.items():
        ids, arrays, _, _ = load_trace(path)
        if candidate_id not in candidate_rows or not np.array_equal(ids, base_ids):
            raise ValueError(f"edited trace is unbound: {candidate_id}")
        delta = arrays["action_postprocessed"] - base["action_postprocessed"]
        effects[candidate_id] = np.sqrt(np.mean(np.square(delta), axis=(1, 2, 3)))
    controls = [value for key, value in effects.items()
                if candidate_rows[key]["role"] in {"low_change", "matched_random"}]
    if not controls:
        raise ValueError("offline gate requires edited matched controls")
    control = np.mean(np.stack(controls), axis=0)
    rng = np.random.default_rng(2057736129)
    rows = []
    for candidate_id, effect in effects.items():
        if candidate_rows[candidate_id]["role"] != "formation":
            continue
        difference = effect - control
        boot = np.asarray([rng.choice(difference, len(difference), replace=True).mean()
                           for _ in range(bootstrap_samples)])
        interval = np.quantile(boot, [0.025, 0.975])
        rows.append({"candidate_id": candidate_id, "mean_action_rms": float(effect.mean()),
                     "mean_control_rms": float(control.mean()),
                     "target_minus_control_ci95": interval.tolist(),
                     "passed": bool(interval[0] > 0)})
    report = {"schema": SCHEMA, "kind": "offline_action_gate",
              "candidate_sha256": file_hash(candidates), "state_ids": base_ids.tolist(),
              "bootstrap_samples": bootstrap_samples, "rows": rows,
              "passed_candidate_ids": [row["candidate_id"] for row in rows if row["passed"]]}
    write_json_atomic(output, report)
    return report


def _intervention_command(checkpoint: Path, task: int, output: Path, candidates: Path,
                          gate: Path, candidate_id: str, dose: float,
                          stages: Sequence[int], offset: int, episodes: int,
                          control: bool) -> tuple[str, ...]:
    base = list(_evaluation_command(checkpoint, task, output, offset, episodes))
    module_index = base.index("interaction_vla.representation_study.libero.capability_events")
    base[module_index] = "interaction_vla.representation_study.libero.flow_intervention_eval"
    separator = base.index("--")
    prefix = ["--candidates", str(candidates), "--gate", str(gate),
              "--candidate-id", candidate_id, "--dose", str(dose)]
    for stage in stages:
        prefix.extend(("--edit-stage", str(stage)))
    if control:
        prefix.append("--allow-control")
    return tuple(base[:separator] + prefix + base[separator:])


def run_closed_loop(checkpoint: Path, candidates: Path, gate: Path, output: Path,
                    tasks: Sequence[int], candidate_ids: Sequence[str], *, dose: float,
                    stages: Sequence[int], initial_state_offset: int, episodes: int,
                    dry_run: bool) -> dict:
    if not tasks or min(tasks) < 0 or episodes <= 0 or initial_state_offset < 0:
        raise ValueError("closed-loop task and episode contract is invalid")
    if not stages or min(stages) < 0 or max(stages) >= 10 or dose == 0:
        raise ValueError("closed-loop flow stages/dose are invalid")
    artifact = json.loads(candidates.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in artifact["candidates"]}
    gate_value = json.loads(gate.read_text(encoding="utf-8"))
    if gate_value.get("candidate_sha256") != file_hash(candidates):
        raise ValueError("offline gate and candidates differ")
    selected = list(candidate_ids) or [
        *gate_value["passed_candidate_ids"],
        *[key for key, row in rows.items() if row["role"] == "low_change"][:2],
        *[key for key, row in rows.items() if row["role"] == "matched_random"][:2],
    ]
    if not selected or any(key not in rows for key in selected):
        raise ValueError("closed-loop candidate selection is empty or unknown")
    commands = []
    conditions = ("baseline", *selected)
    for condition in conditions:
        for task in tasks:
            destination = output / condition / f"task{task}"
            command = (_evaluation_command(checkpoint, task, destination,
                                            initial_state_offset, episodes)
                       if condition == "baseline" else
                       _intervention_command(checkpoint, task, destination, candidates, gate,
                                             condition, dose, stages, initial_state_offset,
                                             episodes, rows[condition]["role"] != "formation"))
            commands.append(list(command))
            if dry_run:
                continue
            if (destination / "eval_info.json").is_file() and (
                destination / "physical_events.json"
            ).is_file():
                continue
            if destination.exists():
                raise FileExistsError(f"incomplete closed-loop output: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
    plan = {"schema": SCHEMA, "kind": "paired_closed_loop", "checkpoint": str(checkpoint),
            "checkpoint_sha256": _tree_sha256(checkpoint),
            "candidate_sha256": file_hash(candidates), "gate_sha256": file_hash(gate),
            "conditions": list(conditions), "tasks": list(tasks), "dose": dose,
            "stages": list(stages), "initial_state_offset": initial_state_offset,
            "episodes_per_task": episodes, "dry_run": dry_run, "commands": commands}
    write_json_atomic(output / "plan.json", plan)
    return plan


def summarize_closed_loop(root: Path, output: Path, *, bootstrap_samples: int = 10000) -> dict:
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if plan.get("dry_run"):
        raise ValueError("cannot summarize a dry-run plan")
    baseline = {}
    for task in plan["tasks"]:
        values = json.loads((root / "baseline" / f"task{task}" /
                             "physical_events.json").read_text(encoding="utf-8"))["episodes"]
        baseline.update({(int(task), int(row["initial_state_id"])): row for row in values})
    rng, summaries = np.random.default_rng(2057736129), []
    for condition in plan["conditions"][1:]:
        differences, action_groups = [], {"translation": [], "rotation": [], "gripper": []}
        event_delta = {key: [] for key in EVENT_KEYS}
        for task in plan["tasks"]:
            values = json.loads((root / condition / f"task{task}" /
                                 "physical_events.json").read_text(encoding="utf-8"))["episodes"]
            for row in values:
                key = (int(task), int(row["initial_state_id"])); original = baseline[key]
                differences.append(int(bool(row["success"])) - int(bool(original["success"])))
                for event in EVENT_KEYS:
                    event_delta[event].append(int(bool(row[event])) - int(bool(original[event])))
                first, second = np.asarray(original["executed_actions"]), np.asarray(row["executed_actions"])
                length = min(len(first), len(second))
                if length:
                    delta = second[:length] - first[:length]
                    action_groups["translation"].append(float(np.sqrt(np.mean(delta[:, :3] ** 2))))
                    action_groups["rotation"].append(float(np.sqrt(np.mean(delta[:, 3:6] ** 2))))
                    action_groups["gripper"].append(float(np.sqrt(np.mean(delta[:, 6:7] ** 2))))
        differences = np.asarray(differences, dtype=float)
        boot = np.asarray([rng.choice(differences, len(differences), replace=True).mean()
                           for _ in range(bootstrap_samples)])
        summaries.append({"condition": condition, "pairs": len(differences),
                          "delta_success": float(differences.mean()),
                          "delta_success_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                          "event_rate_deltas": {key: float(np.mean(value))
                                                for key, value in event_delta.items()},
                          "mean_executed_action_rms": {key: float(np.mean(value))
                                                       for key, value in action_groups.items()}})
    report = {"schema": SCHEMA, "kind": "paired_closed_loop_report",
              "plan_sha256": file_hash(root / "plan.json"),
              "bootstrap_samples": bootstrap_samples, "conditions": summaries}
    write_json_atomic(output, report)
    return report


def _assignments(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError("assignments must be unique NAME=PATH values")
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lineage = commands.add_parser("lineage")
    lineage.add_argument("--checkpoint-root", type=Path, required=True)
    lineage.add_argument("--base-checkpoint", type=Path, required=True)
    lineage.add_argument("--step", type=int, action="append", required=True)
    lineage.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--lineage", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--task", type=int, action="append", default=[])
    evaluate.add_argument("--initial-state-offset", type=int, default=10)
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--dry-run", action="store_true")
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--lineage", type=Path, required=True)
    summarize.add_argument("--root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    discover = commands.add_parser("discover")
    discover.add_argument("--trace", action="append", default=[], required=True)
    discover.add_argument("--before", required=True); discover.add_argument("--after", required=True)
    discover.add_argument("--tap", choices=("expert_middle", "expert_late"), required=True)
    discover.add_argument("--rank", type=int, default=32); discover.add_argument("--output", type=Path, required=True)
    trajectory = commands.add_parser("candidate-trajectory")
    trajectory.add_argument("--trace", action="append", default=[], required=True)
    trajectory.add_argument("--candidates", type=Path, required=True)
    trajectory.add_argument("--output", type=Path, required=True)
    interpret = commands.add_parser("candidate-interpret")
    interpret.add_argument("--trajectory", type=Path, required=True)
    interpret.add_argument("--state-bank", type=Path, required=True)
    interpret.add_argument("--checkpoint", required=True)
    interpret.add_argument("--output", type=Path, required=True)
    smoke = commands.add_parser("intervention-smoke")
    smoke.add_argument("--bank", type=Path, required=True)
    smoke.add_argument("--dataset-root", type=Path, required=True)
    smoke.add_argument("--checkpoint", type=Path, required=True)
    smoke.add_argument("--contract-checkpoint", type=Path, required=True)
    smoke.add_argument("--metadata", type=Path, required=True)
    smoke.add_argument("--candidates", type=Path, required=True)
    smoke.add_argument("--candidate-id", required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--device", default="cuda")
    smoke.add_argument("--max-states", type=int, default=2)
    smoke.add_argument("--dose", type=float, default=1.0)
    smoke.add_argument("--edit-stage", type=int, action="append", default=[])
    smoke.add_argument("--partition", choices=("train", "validation", "test"), default="validation")
    smoke.add_argument("--split-group", choices=("task", "episode"), default="episode")
    smoke.add_argument("--suite")
    smoke.add_argument("--task-id", type=int, action="append", default=[])
    gate = commands.add_parser("gate")
    gate.add_argument("--baseline", type=Path, required=True)
    gate.add_argument("--edited", action="append", default=[], required=True)
    gate.add_argument("--candidates", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    closed = commands.add_parser("closed-loop")
    closed.add_argument("--checkpoint", type=Path, required=True)
    closed.add_argument("--candidates", type=Path, required=True)
    closed.add_argument("--gate", type=Path, required=True)
    closed.add_argument("--output", type=Path, required=True)
    closed.add_argument("--task", type=int, action="append", default=[])
    closed.add_argument("--candidate-id", action="append", default=[])
    closed.add_argument("--dose", type=float, default=1.0)
    closed.add_argument("--edit-stage", type=int, action="append", default=[])
    closed.add_argument("--initial-state-offset", type=int, default=20)
    closed.add_argument("--episodes", type=int, default=10)
    closed.add_argument("--dry-run", action="store_true")
    closed_summary = commands.add_parser("closed-loop-summary")
    closed_summary.add_argument("--root", type=Path, required=True)
    closed_summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lineage": result = checkpoint_lineage(args.checkpoint_root, args.base_checkpoint, args.step, args.output)
    elif args.command == "evaluate": result = evaluate_timeline(args.lineage, args.output, args.task or range(4), initial_state_offset=args.initial_state_offset, episodes=args.episodes, dry_run=args.dry_run)
    elif args.command == "summarize": result = summarize_timeline(args.lineage, args.root, args.output)
    elif args.command == "discover": result = discover_change_subspaces(_assignments(args.trace), before=args.before, after=args.after, tap=args.tap, rank=args.rank, output=args.output)
    elif args.command == "candidate-trajectory": result = candidate_trajectory(_assignments(args.trace), args.candidates, args.output)
    elif args.command == "candidate-interpret": result = interpret_candidates(args.trajectory, args.state_bank, args.checkpoint, args.output)
    elif args.command == "intervention-smoke": result = intervention_smoke_command(args.bank, args.dataset_root, args.checkpoint, args.contract_checkpoint, args.metadata, args.candidates, args.candidate_id, args.output, device=args.device, max_states=args.max_states, dose=args.dose, stages=args.edit_stage or tuple(range(10)), partition=args.partition, split_group=args.split_group, suite=args.suite, task_ids=args.task_id)
    elif args.command == "gate": result = offline_action_gate(args.baseline, _assignments(args.edited), args.candidates, args.output)
    elif args.command == "closed-loop": result = run_closed_loop(args.checkpoint, args.candidates, args.gate, args.output, args.task or range(4), args.candidate_id, dose=args.dose, stages=args.edit_stage or (0, 5, 9), initial_state_offset=args.initial_state_offset, episodes=args.episodes, dry_run=args.dry_run)
    else: result = summarize_closed_loop(args.root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
