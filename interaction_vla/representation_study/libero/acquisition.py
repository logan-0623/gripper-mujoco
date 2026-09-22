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


SCHEMA = "smolvla_acquisition_v2"
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
                        initial_state_offset: int, episodes: int,
                        initial_state_count: int | None = None) -> tuple[str, ...]:
    command = [
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
    ]
    if initial_state_count is not None:
        command[command.index("--rendered-episodes") + 2:command.index("--rendered-episodes") + 2] = [
            "--initial-state-count", str(initial_state_count)
        ]
    return tuple(command)


def _actual_state_ids(offset: int, episodes: int, state_count: int | None) -> list[int]:
    if state_count is not None and (state_count <= 0 or episodes > state_count):
        raise ValueError("initial_state_count must be positive and actual states must not repeat")
    requested = range(offset, offset + episodes)
    return [int(state % state_count) for state in requested] if state_count else list(requested)


def _plan_actual_cells(path: Path, plan: Mapping[str, object]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    incomplete = False
    commands = plan.get("commands", [])
    for command in commands:
        if not isinstance(command, list) or "--events-output" not in command:
            incomplete = True
            continue
        event_path = Path(command[command.index("--events-output") + 1])
        if not event_path.is_absolute():
            event_path = path.parent / event_path
        if not event_path.is_file():
            incomplete = True
            continue
        payload = json.loads(event_path.read_text(encoding="utf-8"))
        episodes = payload.get("episodes", [])
        if len(episodes) != int(plan["episodes_per_task"]):
            incomplete = True
        for row in episodes:
            if "task_id" in row and "initial_state_id" in row:
                cells.add((int(row["task_id"]), int(row["initial_state_id"])))
            else:
                incomplete = True
    if cells and not incomplete and {task for task, _ in cells} == set(plan["tasks"]):
        return cells
    offset = int(plan["initial_state_offset"])
    count = int(plan["episodes_per_task"])
    state_count = plan.get("initial_state_count")
    if not state_count:
        raise ValueError(f"historical initial_state_count or complete actual records required: {path}")
    actual = _actual_state_ids(offset, count, int(state_count))
    return cells | {(int(task), state) for task in plan["tasks"] for state in actual}


def evaluate_timeline(lineage: Path, output: Path, tasks: Sequence[int], *,
                      initial_state_offset: int, episodes: int, dry_run: bool,
                      initial_state_count: int | None = None) -> dict:
    if not tasks or min(tasks) < 0 or initial_state_offset < 0 or episodes <= 0:
        raise ValueError("tasks, initial-state offset, and episode count are invalid")
    if initial_state_count is None or initial_state_count <= 0:
        raise ValueError("timeline evaluation requires initial_state_count")
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
                checkpoint, task, destination, initial_state_offset, episodes,
                initial_state_count
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
        "initial_state_count": initial_state_count, "dry_run": dry_run, "commands": commands,
    }
    write_json_atomic(output / "evaluation_plan.json", plan)
    return plan


def _load_event_rows(path: Path, plan: Mapping[str, object], task: int) -> dict[int, dict[str, object]]:
    """Load one task's events and fail closed on missing or duplicated cells."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError(f"event file has no episode list: {path}")
    try:
        episodes = int(plan["episodes_per_task"])
        offset = int(plan["initial_state_offset"])
        state_count = int(plan["initial_state_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"event contract lacks initial_state_count: {path}") from exc
    expected_ids = _actual_state_ids(offset, episodes, state_count)
    requested_ids = list(range(offset, offset + episodes))
    expected_requested = dict(zip(expected_ids, requested_ids, strict=True))
    if len(rows) != episodes:
        raise ValueError(f"episode count differs: {path}: {len(rows)} != {episodes}")
    result: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"invalid event row: {path}")
        if int(row.get("task_id", -1)) != int(task):
            raise ValueError(f"event task identity differs: {path}")
        state_id = int(row.get("initial_state_id", -1))
        if state_id in result:
            raise ValueError(f"duplicate initial_state_id {state_id}: {path}")
        if int(row.get("initial_state_count", -1)) != state_count:
            raise ValueError(f"event state-count identity differs: {path}")
        requested = row.get("requested_initial_state_id")
        if requested is None or int(requested) != expected_requested.get(state_id, -1):
            raise ValueError(f"requested/actual state identity differs: {path}")
        result[state_id] = dict(row)
    if set(result) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(result))
        extra = sorted(set(result) - set(expected_ids))
        raise ValueError(f"missing or unexpected initial states in {path}: missing={missing}, extra={extra}")
    return result


def summarize_timeline(lineage: Path, root: Path, output: Path) -> dict:
    manifest = json.loads(lineage.read_text(encoding="utf-8"))
    plan = json.loads((root / "evaluation_plan.json").read_text(encoding="utf-8"))
    if plan["lineage_sha256"] != file_hash(lineage):
        raise ValueError("timeline evaluation belongs to another lineage")
    if not isinstance(plan.get("initial_state_count"), int) or plan["initial_state_count"] <= 0:
        raise ValueError("timeline summary requires explicit initial_state_count")
    rows = []
    task_cells: dict[int, set[int]] = {}
    for checkpoint in manifest["checkpoints"]:
        step = int(checkpoint["step"])
        for task in plan["tasks"]:
            directory = root / f"step_{step:06d}" / f"task{task}"
            events_path, eval_path = directory / "physical_events.json", directory / "eval_info.json"
            event_rows = _load_event_rows(events_path, plan, int(task))
            expected_ids = _actual_state_ids(
                int(plan["initial_state_offset"]), int(plan["episodes_per_task"]),
                int(plan["initial_state_count"]),
            )
            current = set(event_rows)
            previous = task_cells.setdefault(int(task), current)
            if current != previous:
                raise ValueError(f"paired initial-state set differs: {directory}")
            eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
            successes = eval_payload["per_task"][0]["metrics"]["successes"]
            if len(successes) != len(expected_ids):
                raise ValueError(f"evaluation count differs: {directory}")
            event_list = [event_rows[state_id] for state_id in expected_ids]
            success_count = sum(bool(value) for value in successes)
            event_success_count = sum(bool(item.get("success")) for item in event_list)
            if event_success_count != success_count:
                raise ValueError(f"success records disagree: {directory}")
            row = {
                "step": step, "task": int(task), "episodes": len(event_list),
                "initial_state_ids": expected_ids,
                "eval_info_sha256": file_hash(eval_path),
                "physical_events_sha256": file_hash(events_path),
            }
            row.update({key: sum(bool(item.get(key)) for item in event_list) for key in EVENT_KEYS})
            rows.append(row)
    expected_steps = [int(row["step"]) for row in manifest["checkpoints"]]
    if sorted({int(row["step"]) for row in rows}) != expected_steps:
        raise ValueError("timeline is missing checkpoint rows")
    aggregate = []
    for step in expected_steps:
        selected = [row for row in rows if row["step"] == step]
        aggregate.append({"step": step, **{
            key: sum(int(row[key]) for row in selected) for key in ("episodes", *EVENT_KEYS)
        }})
    report = {
        "schema": SCHEMA, "kind": "capability_timeline", "complete": True,
        "initial_state_count": int(plan["initial_state_count"]),
        "paired_initial_state_ids": {str(task): sorted(values) for task, values in task_cells.items()},
        "rows": rows, "aggregate": aggregate, "lineage_sha256": file_hash(lineage),
    }
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


def expand_random_controls(candidates: Path, output: Path, *, count: int = 8) -> dict:
    """Freeze additional deterministic random controls without changing target directions."""
    if output.exists():
        raise FileExistsError(f"control output exists: {output}")
    if count < 1:
        raise ValueError("random control count must be positive")
    source = json.loads(candidates.read_text(encoding="utf-8"))
    rows = source.get("candidates", [])
    retained = [row for row in rows if row["role"] != "matched_random"]
    if not retained:
        raise ValueError("candidate artifact has no retained directions")
    existing = [np.asarray(row["direction"], dtype=float) for row in retained]
    randoms = _orthogonal_random(existing, len(existing[0]), count, 2057736129)
    expanded = [*retained, *[
        {"id": f"matched_random_{index}", "role": "matched_random",
         "tap": retained[0]["tap"], "direction": direction.tolist()}
        for index, direction in enumerate(randoms)
    ]]
    basis_source = candidates.with_name("shared_basis.npz")
    if not basis_source.is_file():
        raise FileNotFoundError(f"shared basis is missing: {basis_source}")
    output.mkdir(parents=True)
    write_bytes_atomic(output / "shared_basis.npz", basis_source.read_bytes())
    report = {**source, "kind": "frozen_change_candidates_with_random_panel",
              "source_candidate_sha256": file_hash(candidates),
              "random_control_count": count, "candidates": expanded,
              "shared_basis_sha256": file_hash(output / "shared_basis.npz")}
    write_json_atomic(output / "candidates.json", report)
    return report


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
        "trace_checkpoint_sha256": {
            name: (value[2].get("checkpoint_tree_sha256")
                   or value[2].get("checkpoint_sha256"))
            for name, value in loaded.items()
        },
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


def candidate_context_analysis(trace: Path, candidates: Path, state_bank: Path,
                               output: Path, candidate_ids: Sequence[str] = ()) -> dict:
    """Episode-held-out description of frozen candidates; never selects directions."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate context analysis: {output}")
    artifact = json.loads(candidates.read_text(encoding="utf-8"))
    tap = artifact["tap"]
    rows = [row for row in artifact["candidates"]
            if (not candidate_ids and row["role"] == "formation")
            or row["id"] in candidate_ids]
    if not rows:
        raise ValueError("candidate context analysis has no selected candidates")
    state_ids, arrays, binding, _ = load_trace(trace)
    records, _, _, _ = load_state_bank(state_bank)
    by_id = {row.state_id: row for row in records}
    if tap not in arrays or any(state_id not in by_id for state_id in state_ids.tolist()):
        raise ValueError("trace tap or StateBank coverage is incomplete")
    selected = [by_id[state_id] for state_id in state_ids.tolist()]
    directions = np.asarray([row["direction"] for row in rows], dtype=np.float64)
    activations = arrays[tap]
    if activations.ndim != 5 or activations.shape[-1] != directions.shape[1]:
        raise ValueError("candidate direction and trace activation shapes differ")
    projections = np.einsum(
        "nrsd,cd->ncrs", activations.mean(axis=3), directions, optimize=True
    ).mean(axis=2)
    tasks = np.asarray([row.task_id for row in selected], dtype=int)
    episodes = np.asarray([
        f"{row.suite}/{row.task_id}/{row.source_episode_id}" for row in selected
    ])
    task_values = sorted(set(tasks.tolist()))
    phases = sorted({row.labels.phase for row in selected if row.labels.phase is not None})
    base = np.column_stack((
        np.asarray([row.frame_index for row in selected], dtype=float),
        *[tasks == task for task in task_values[1:]],
    ))
    robot = np.asarray([row.observation.robot_state for row in selected], dtype=float)
    action = np.asarray([row.observation.action for row in selected], dtype=float)
    physical = np.column_stack((
        np.asarray([row.labels.geometry.gripper_target_distance
                    if row.labels.geometry is not None else np.nan for row in selected]),
        np.asarray([row.labels.geometry.target_goal_distance
                    if row.labels.geometry is not None else np.nan for row in selected]),
        np.asarray([float(row.labels.contact.gripper_target)
                    if row.labels.contact is not None else np.nan for row in selected]),
        np.asarray([float(row.labels.stable_grasp)
                    if row.labels.stable_grasp is not None else np.nan for row in selected]),
        *[np.asarray([float(row.labels.phase == phase)
                     if row.labels.phase is not None else np.nan for row in selected])
          for phase in phases],
    ))
    groups = {
        "task_time": base,
        "task_time_robot": np.column_stack((base, robot)),
        "task_time_action": np.column_stack((base, action)),
        "task_time_physical": np.column_stack((base, physical)),
        "task_time_robot_action": np.column_stack((base, robot, action)),
        "task_time_robot_action_physical": np.column_stack(
            (base, robot, action, physical)),
    }

    def cross_validated_prediction(features: np.ndarray, target: np.ndarray) -> np.ndarray:
        prediction = np.full(len(target), np.nan, dtype=float)
        for episode in np.unique(episodes):
            test = episodes == episode
            train = ~test
            x_train, x_test = features[train].copy(), features[test].copy()
            mean = np.nanmean(x_train, axis=0)
            mean[~np.isfinite(mean)] = 0.0
            x_train = np.where(np.isfinite(x_train), x_train, mean)
            x_test = np.where(np.isfinite(x_test), x_test, mean)
            scale = x_train.std(axis=0)
            scale[scale < 1e-8] = 1.0
            x_train, x_test = (x_train - mean) / scale, (x_test - mean) / scale
            y_mean = float(target[train].mean())
            gram = x_train.T @ x_train + np.eye(x_train.shape[1])
            coefficient = np.linalg.solve(gram, x_train.T @ (target[train] - y_mean))
            prediction[test] = y_mean + x_test @ coefficient
        if not np.isfinite(prediction).all():
            raise ValueError("candidate context cross-validation produced non-finite predictions")
        return prediction

    def task_episode_state_mean(values: np.ndarray) -> float:
        per_task = []
        for task in task_values:
            task_episodes = np.unique(episodes[tasks == task])
            per_task.append(np.mean([
                values[(tasks == task) & (episodes == episode)].mean()
                for episode in task_episodes
            ]))
        return float(np.mean(per_task))

    results = []
    for candidate_index, row in enumerate(rows):
        for stage in range(projections.shape[2]):
            target = projections[:, candidate_index, stage]
            null_error = task_episode_state_mean(np.square(target - target.mean()))
            metrics = {}
            for name, features in groups.items():
                prediction = cross_validated_prediction(features, target)
                error = task_episode_state_mean(np.square(target - prediction))
                metrics[name] = {"risk": error, "predictive_r2": (
                    float(1.0 - error / null_error) if null_error > 1e-12 else float("nan")
                )}
            base_risk = metrics["task_time"]["risk"]
            for metric in metrics.values():
                metric["risk_reduction_over_task_time"] = float(base_risk - metric["risk"])
            results.append({"candidate_id": row["id"], "stage": stage,
                            "metrics": metrics})
    report = {
        "schema": SCHEMA, "kind": "episode_held_out_candidate_context",
        "candidate_sha256": file_hash(candidates),
        "trace_binding_sha256": binding.get("binding_sha256"),
        "state_bank_sha256": file_hash(state_bank / "manifest.json"),
        "selection_performed": False, "tap": tap,
        "states": len(selected), "independent_episodes": len(np.unique(episodes)),
        "tasks": task_values,
        "cross_validation_unit": "source_episode",
        "aggregation": "task equal, episode equal within task, state equal within episode",
        "ridge_alpha": 1.0,
        "action_fields_are_leakage_diagnostics": True,
        "results": results,
    }
    write_json_atomic(output, report)
    return report



def audit_longitudinal_evidence(
    lineage: Path,
    timeline: Path,
    candidates: Path,
    trajectory: Path,
    functional_reports: Mapping[str, Path],
    output: Path,
) -> dict:
    """Audit that R_k, U_k and S_k artifacts cover the same frozen lineage."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite longitudinal audit: {output}")
    lineage_value = json.loads(lineage.read_text(encoding="utf-8"))
    if lineage_value.get("kind") != "immutable_lineage":
        raise ValueError("lineage artifact is incompatible")
    checkpoints = lineage_value.get("checkpoints", [])
    if len(checkpoints) < 2:
        raise ValueError("longitudinal audit requires at least two checkpoints")
    checkpoint_steps = [int(row["step"]) for row in checkpoints]
    checkpoint_hashes = {str(row["checkpoint_sha256"]) for row in checkpoints}
    if checkpoint_steps != sorted(set(checkpoint_steps)):
        raise ValueError("lineage checkpoint steps are not unique and ordered")

    timeline_value = json.loads(timeline.read_text(encoding="utf-8"))
    timeline_ok = (
        timeline_value.get("kind") == "capability_timeline"
        and timeline_value.get("complete") is True
        and timeline_value.get("lineage_sha256") == file_hash(lineage)
    )
    timeline_rows = timeline_value.get("rows", [])
    timeline_steps = sorted({int(row["step"]) for row in timeline_rows})
    if timeline_ok and timeline_steps != checkpoint_steps:
        timeline_ok = False
    timeline_cells = timeline_value.get("paired_initial_state_ids", {})
    if timeline_ok and not timeline_cells:
        timeline_ok = False
    timeline_seen: set[tuple[int, int]] = set()
    timeline_by_step: dict[int, dict[str, list[int]]] = {}
    for row in timeline_rows:
        try:
            step, task = int(row["step"]), int(row["task"])
            ids = [int(value) for value in row["initial_state_ids"]]
            episodes = int(row["episodes"])
        except (KeyError, TypeError, ValueError):
            timeline_ok = False
            continue
        key = (step, task)
        if key in timeline_seen or len(ids) != len(set(ids)) or episodes != len(ids):
            timeline_ok = False
        timeline_seen.add(key)
        timeline_by_step.setdefault(step, {})[str(task)] = sorted(ids)
    if timeline_ok:
        first_cells = timeline_by_step.get(checkpoint_steps[0], {})
        if any(timeline_by_step.get(step) != first_cells for step in checkpoint_steps):
            timeline_ok = False

    candidate_value = json.loads(candidates.read_text(encoding="utf-8"))
    candidate_hash = file_hash(candidates)
    candidate_ids = [str(row["id"]) for row in candidate_value.get("candidates", [])]
    trajectory_value = json.loads((trajectory / "report.json").read_text(encoding="utf-8"))
    projection_path = trajectory / "projections.npz"
    representation_reasons = []
    if trajectory_value.get("kind") != "frozen_candidate_trajectory":
        representation_reasons.append("trajectory kind is incompatible")
    if trajectory_value.get("candidate_sha256") != candidate_hash:
        representation_reasons.append("trajectory candidate hash differs")
    if trajectory_value.get("selection_uses_physical_labels") is not False:
        representation_reasons.append("trajectory selection was not label-blind")
    if not projection_path.is_file() or trajectory_value.get("projections_sha256") != file_hash(projection_path):
        representation_reasons.append("trajectory projection artifact is stale or missing")
    trajectory_names = [str(row.get("checkpoint")) for row in trajectory_value.get("summaries", [])]
    if len(trajectory_names) < 2 or len(set(trajectory_names)) != len(trajectory_names):
        representation_reasons.append("trajectory lacks unique cross-checkpoint summaries")
    trace_hashes = {
        str(value) for value in trajectory_value.get("trace_checkpoint_sha256", {}).values()
        if value
    }
    if trace_hashes != checkpoint_hashes:
        representation_reasons.append("trajectory traces do not cover the frozen lineage")
    state_ids = trajectory_value.get("state_ids", [])
    if not state_ids or len(state_ids) != len(set(state_ids)):
        representation_reasons.append("trajectory state IDs are missing or duplicated")
    try:
        with np.load(projection_path, allow_pickle=False) as values:
            projected_ids = [str(item) for item in values["candidate_ids"].tolist()]
            if projected_ids != candidate_ids:
                representation_reasons.append("trajectory candidate IDs differ")
            for name in trajectory_names:
                key = f"projection_{name}"
                if key not in values.files or not np.isfinite(values[key]).all():
                    representation_reasons.append(f"trajectory projection is missing or non-finite: {name}")
    except (KeyError, OSError, ValueError) as exc:
        representation_reasons.append(f"trajectory projection cannot be read: {exc}")

    functional_seen: set[str] = set()
    functional_rows = []
    functional_reasons = []
    for name, path in functional_reports.items():
        value = json.loads(path.read_text(encoding="utf-8"))
        report_hash = value.get("checkpoint_sha256")
        if value.get("kind") != "offline_action_gate":
            functional_reasons.append(f"{name}: incompatible gate kind")
        if value.get("candidate_sha256") != candidate_hash:
            functional_reasons.append(f"{name}: candidate hash differs")
        if report_hash not in checkpoint_hashes:
            functional_reasons.append(f"{name}: checkpoint hash is not in lineage")
        if int(value.get("independent_episodes", 0)) <= 0:
            functional_reasons.append(f"{name}: no independent episodes")
        consistency = value.get("numerical_self_consistency", {})
        if consistency.get("all_arrays_finite") is not True or consistency.get("shared_state_noise_sigma") is not True:
            functional_reasons.append(f"{name}: numerical self-consistency failed")
        if report_hash:
            functional_seen.add(str(report_hash))
        functional_rows.append({
            "name": name, "path": str(path), "checkpoint_sha256": report_hash,
            "independent_episodes": value.get("independent_episodes"),
        })
    missing_functional = sorted(checkpoint_hashes - functional_seen)
    if missing_functional:
        functional_reasons.append("functional reports do not cover every lineage checkpoint")

    result = {
        "schema": SCHEMA,
        "kind": "longitudinal_evidence_audit",
        "lineage_sha256": file_hash(lineage),
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_sha256s": [str(row["checkpoint_sha256"]) for row in checkpoints],
        "representation_alignment": {
            "passed": not representation_reasons,
            "candidate_sha256": candidate_hash,
            "trajectory_sha256": file_hash(trajectory / "report.json"),
            "reasons": representation_reasons,
        },
        "capability_comparison": {
            "passed": timeline_ok,
            "timeline_sha256": file_hash(timeline),
            "steps": timeline_steps,
            "paired_initial_state_ids": timeline_cells,
        },
        "functional_dependence": {
            "passed": not functional_reasons,
            "reports": functional_rows,
            "missing_checkpoint_sha256s": missing_functional,
            "reasons": functional_reasons,
        },
        "primary_comparison": ["R_k", "U_k", "S_k"],
        "protocol_ready": bool(
            not representation_reasons and timeline_ok and not functional_reasons
        ),
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
                        state_bank: Path, output: Path, *, executed_steps: int = 10,
                        minimum_episodes: int = 8,
                        bootstrap_samples: int = 10000) -> dict:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if executed_steps <= 0 or minimum_episodes < 2:
        raise ValueError("executed steps and minimum episodes are invalid")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite offline gate: {output}")
    candidate_rows = {row["id"]: row for row in json.loads(
        candidates.read_text(encoding="utf-8")
    )["candidates"]}
    records, _, _, _ = load_state_bank(state_bank)
    by_id = {row.state_id: row for row in records}
    base_ids, base, base_binding, _ = load_trace(baseline)
    if any(state_id not in by_id for state_id in base_ids.tolist()):
        raise ValueError("StateBank does not cover every trace state")
    episodes = np.asarray([
        f"{by_id[state_id].suite}/{by_id[state_id].task_id}/"
        f"{by_id[state_id].source_episode_id}" for state_id in base_ids.tolist()
    ])
    unique_episodes = np.unique(episodes)
    effects, action_deltas, conditions = {}, {}, {}
    for condition, path in edited.items():
        ids, arrays, binding, _ = load_trace(path)
        edit = binding.get("flow_edit") or {}
        candidate_id = edit.get("candidate_id")
        if candidate_id not in candidate_rows or not np.array_equal(ids, base_ids):
            raise ValueError(f"edited trace is unbound: {condition}")
        if any(not np.array_equal(arrays[key], base[key]) for key in ("epsilon", "sigma")):
            raise ValueError(f"edited trace does not share baseline noise/sigma: {condition}")
        action = arrays["action_postprocessed"]
        baseline_action = base["action_postprocessed"]
        if action.shape != baseline_action.shape or executed_steps > action.shape[2]:
            raise ValueError("action trace shape or executed prefix is invalid")
        delta = action - baseline_action
        action_deltas[condition] = delta[:, :, :executed_steps]

        def rms(value: np.ndarray, slc: slice) -> np.ndarray:
            return np.sqrt(np.mean(np.square(value[..., slc]), axis=(1, 2, 3)))

        effects[condition] = {
            "executed_full": rms(delta[:, :, :executed_steps], slice(None)),
            "executed_translation": rms(delta[:, :, :executed_steps], slice(0, 3)),
            "executed_rotation": rms(delta[:, :, :executed_steps], slice(3, 6)),
            "executed_gripper": rms(delta[:, :, :executed_steps], slice(6, 7)),
            "planned_full": rms(delta, slice(None)),
        }
        conditions[condition] = {
            "candidate_id": candidate_id,
            "role": candidate_rows[candidate_id]["role"],
            "stages": tuple(edit.get("stages", ())),
            "dose": float(edit.get("dose", np.nan)),
        }

    def episode_means(values: np.ndarray) -> np.ndarray:
        return np.asarray([values[episodes == episode].mean() for episode in unique_episodes])

    rng = np.random.default_rng(2057736129)
    rows = []
    for condition, metadata in conditions.items():
        if metadata["role"] != "formation":
            continue
        matched = [name for name, row in conditions.items()
                   if row["role"] in {"low_change", "matched_random"}
                   and row["stages"] == metadata["stages"]
                   and row["dose"] == metadata["dose"]]
        if not matched:
            rows.append({"condition": condition, **metadata, "passed": False,
                         "reason": "missing stage-and-dose-matched controls"})
            continue
        control = np.mean(np.stack([
            effects[name]["executed_full"] for name in matched
        ]), axis=0)
        target = effects[condition]["executed_full"]
        difference = episode_means(target - control)
        boot = np.asarray([rng.choice(difference, len(difference), replace=True).mean()
                           for _ in range(bootstrap_samples)])
        interval = np.quantile(boot, [0.025, 0.975])
        enough = len(unique_episodes) >= minimum_episodes
        rows.append({"condition": condition, **metadata,
                     "matched_controls": matched,
                     "independent_episodes": int(len(unique_episodes)),
                     "mean_executed_action_rms": {
                         key.removeprefix("executed_"): float(value.mean())
                         for key, value in effects[condition].items()
                         if key.startswith("executed_")
                     },
                     "mean_planned_full_rms": float(
                         effects[condition]["planned_full"].mean()),
                     "mean_control_executed_full_rms": float(control.mean()),
                     "target_minus_control_ci95": interval.tolist(),
                     "passed": bool(enough and interval[0] > 0),
                     "reason": ("passed" if enough and interval[0] > 0 else
                                "too few independent episodes" if not enough else
                                "episode-clustered interval includes zero")})
    signed_pairs = []
    for name, row in conditions.items():
        if row["dose"] <= 0:
            continue
        opposite = next((other for other, candidate in conditions.items()
                         if candidate["candidate_id"] == row["candidate_id"]
                         and candidate["stages"] == row["stages"]
                         and candidate["dose"] == -row["dose"]), None)
        if opposite is None:
            continue
        first, second = action_deltas[name], action_deltas[opposite]
        denominator = 0.5 * (np.sqrt(np.mean(first ** 2)) + np.sqrt(np.mean(second ** 2)))
        signed_pairs.append({
            "positive": name, "negative": opposite,
            "antisymmetry_residual_ratio": (
                float(np.sqrt(np.mean((first + second) ** 2)) / denominator)
                if denominator > 0 else 0.0),
        })
    direction_norm_error = max((abs(np.linalg.norm(row["direction"]) - 1.0)
                                for row in candidate_rows.values()
                                if "direction" in row), default=0.0)
    report = {"schema": SCHEMA, "kind": "offline_action_gate",
              "candidate_sha256": file_hash(candidates), "state_ids": base_ids.tolist(),
              "state_bank_sha256": file_hash(state_bank / "manifest.json"),
              "checkpoint_sha256": (base_binding.get("checkpoint_sha256")
                                     or base_binding.get("checkpoint_tree_sha256")),
              "baseline_binding_sha256": base_binding.get("binding_sha256"),
              "primary_metric": f"first_{executed_steps}_actions_full_rms",
              "secondary_metrics": ["translation", "rotation", "gripper", "planned_full"],
              "bootstrap_unit": "source_episode",
              "independent_episodes": int(len(unique_episodes)),
              "minimum_episodes": minimum_episodes,
              "numerical_self_consistency": {
                  "all_arrays_finite": True,
                  "shared_state_noise_sigma": True,
                  "maximum_candidate_unit_norm_error": float(direction_norm_error),
                  "signed_pair_checks": signed_pairs,
              },
              "bootstrap_samples": bootstrap_samples, "rows": rows,
              "passed_candidate_ids": sorted({row["candidate_id"] for row in rows
                                                if row.get("passed")})}
    write_json_atomic(output, report)
    return report


def _intervention_command(checkpoint: Path, task: int, output: Path, candidates: Path,
                          gate: Path, candidate_id: str, dose: float,
                          stages: Sequence[int], offset: int, episodes: int,
                          control: bool, edit_mode: str = "additive",
                          match_candidate_id: str | None = None,
                          initial_state_count: int | None = None) -> tuple[str, ...]:
    base = list(_evaluation_command(checkpoint, task, output, offset, episodes,
                                    initial_state_count))
    module_index = base.index("interaction_vla.representation_study.libero.capability_events")
    base[module_index] = "interaction_vla.representation_study.libero.flow_intervention_eval"
    separator = base.index("--")
    prefix = ["--candidates", str(candidates), "--gate", str(gate),
              "--candidate-id", candidate_id, "--dose", str(dose),
              "--edit-mode", edit_mode]
    if match_candidate_id is not None:
        prefix.extend(("--match-candidate-id", match_candidate_id))
    for stage in stages:
        prefix.extend(("--edit-stage", str(stage)))
    if control:
        prefix.append("--allow-control")
    return tuple(base[:separator] + prefix + base[separator:])


def freeze_confirmation_contract(output: Path, tasks: Sequence[int], *,
                                 initial_state_offset: int, episodes: int,
                                 used_plans: Sequence[Path] = (),
                                 binding: Mapping[str, object] | None = None,
                                 initial_state_count: int | None = None) -> dict:
    """Freeze unseen simulator cells and reject overlap with recorded plans."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite confirmation contract: {output}")
    if not tasks or min(tasks) < 0 or initial_state_offset < 0 or episodes <= 0:
        raise ValueError("confirmation task and initial-state contract is invalid")
    if initial_state_count is None or initial_state_count <= 0:
        raise ValueError("confirmation requires positive initial_state_count")
    requested_ids = list(range(initial_state_offset, initial_state_offset + episodes))
    actual_ids = _actual_state_ids(initial_state_offset, episodes, initial_state_count)
    requested = {(int(task), state) for task in tasks for state in actual_ids}
    used, bindings = set(), {}
    for path in used_plans:
        plan = json.loads(path.read_text(encoding="utf-8"))
        used.update(_plan_actual_cells(path, plan))
        bindings[str(path.resolve())] = file_hash(path)
    overlap = sorted(requested & used)
    if overlap:
        raise ValueError(f"confirmation cells were used previously: {overlap}")
    contract_binding = dict(binding or {})
    contract_binding.update({
        "tasks": sorted(set(int(task) for task in tasks)),
        "initial_state_offset": int(initial_state_offset),
        "episodes_per_task": int(episodes),
        "requested_initial_state_ids": requested_ids,
        "actual_initial_state_ids": actual_ids,
        "initial_state_count": initial_state_count,
    })
    required_binding = (
        "checkpoint_sha256", "processor_sha256", "candidate_sha256", "tap",
        "gate_sha256", "initial_state_count",
        "stages", "dose", "edit_mode", "candidate_ids", "executed_steps",
        "noise_rule", "primary_metric", "secondary_metrics", "controls",
        "estimator", "confidence_level", "comparison_family", "budget",
        "history_usage",
        # Longitudinal §0.6 binding: R_k/U_k/S_k share one frozen lineage.
        "lineage_sha256", "checkpoint_steps", "training_contract_sha256",
        "candidate_trajectory_sha256", "candidate_alignment",
        "readout_protocol_sha256", "readout_protocol",
        "dose_calibration_sha256", "primary_comparison",
        "longitudinal_audit_sha256",
    )
    def present(key: str) -> bool:
        value = contract_binding.get(key)
        return value is not None and value != "" and value != [] and value != {} and value != "TBD"
    missing_binding = [key for key in required_binding if not present(key)]
    report = {
        "schema": SCHEMA, "kind": "frozen_confirmation_contract",
        "tasks": contract_binding["tasks"],
        "initial_state_offset": initial_state_offset,
        "episodes_per_task": episodes,
        "requested_initial_state_ids": requested_ids,
        "actual_initial_state_ids": actual_ids,
        "initial_state_count": initial_state_count,
        "selection_uses_confirmation_results": False,
        "used_plan_bindings": bindings,
        "binding": contract_binding,
        "contract_complete": not missing_binding,
        "missing_binding": missing_binding,
    }
    write_json_atomic(output, report)
    return report


def run_closed_loop(checkpoint: Path, candidates: Path, gate: Path, output: Path,
                    tasks: Sequence[int], candidate_ids: Sequence[str], *, dose: float,
                    stages: Sequence[int], initial_state_offset: int, episodes: int,
                    dry_run: bool, confirmation_contract: Path | None = None,
                    edit_mode: str = "additive",
                    initial_state_count: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to relabel existing closed-loop output: {output}")
    if not tasks or min(tasks) < 0 or episodes <= 0 or initial_state_offset < 0:
        raise ValueError("closed-loop task and episode contract is invalid")
    if not stages or min(stages) < 0 or max(stages) >= 10 or dose == 0:
        raise ValueError("closed-loop flow stages/dose are invalid")
    if initial_state_count is None or initial_state_count <= 0:
        raise ValueError("closed-loop runs require initial_state_count")
    _actual_state_ids(initial_state_offset, episodes, initial_state_count)
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
    confirmation_sha256 = None
    if confirmation_contract is not None:
        confirmation = json.loads(confirmation_contract.read_text(encoding="utf-8"))
        expected = (sorted(set(int(task) for task in tasks)), initial_state_offset, episodes)
        observed = (confirmation.get("tasks"), confirmation.get("initial_state_offset"),
                    confirmation.get("episodes_per_task"))
        if confirmation.get("kind") != "frozen_confirmation_contract" or observed != expected:
            raise ValueError("closed-loop run differs from frozen confirmation contract")
        if not confirmation.get("contract_complete"):
            raise ValueError(
                "confirmation contract is incomplete; missing "
                + ", ".join(confirmation.get("missing_binding", []))
            )
        binding = confirmation["binding"]
        if binding.get("checkpoint_sha256") != _tree_sha256(checkpoint):
            raise ValueError("confirmation checkpoint hash differs")
        if binding.get("candidate_sha256") != file_hash(candidates):
            raise ValueError("confirmation candidate hash differs")
        if binding.get("gate_sha256") not in (None, file_hash(gate)):
            raise ValueError("confirmation gate hash differs")
        expected_binding = {
            "tasks": sorted(set(int(task) for task in tasks)),
            "initial_state_offset": int(initial_state_offset),
            "episodes_per_task": int(episodes),
            "initial_state_count": initial_state_count,
            "stages": list(stages), "dose": float(dose),
            "edit_mode": edit_mode,
            "candidate_ids": list(selected),
            "executed_steps": 10,
        }
        for key, value in expected_binding.items():
            if value is not None and binding.get(key) != value:
                raise ValueError(f"confirmation binding differs: {key}")
        confirmation_sha256 = file_hash(confirmation_contract)
    formation_ids = [key for key in selected if rows[key]["role"] == "formation"]
    match_target = formation_ids[0] if len(formation_ids) == 1 else None
    if edit_mode != "additive" and match_target is None:
        raise ValueError("suppression closed loop requires exactly one formation target")
    commands = []
    conditions = ("baseline", *selected)
    for condition in conditions:
        for task in tasks:
            destination = output / condition / f"task{task}"
            command = (_evaluation_command(checkpoint, task, destination,
                                            initial_state_offset, episodes,
                                            initial_state_count)
                       if condition == "baseline" else
                       _intervention_command(checkpoint, task, destination, candidates, gate,
                                             condition, dose, stages, initial_state_offset,
                                             episodes, rows[condition]["role"] != "formation",
                                             edit_mode,
                                             match_target if rows[condition]["role"] != "formation" else None,
                                             initial_state_count))
            commands.append(list(command))
            if dry_run:
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
            "initial_state_count": initial_state_count,
            "episodes_per_task": episodes, "dry_run": dry_run,
            "edit_mode": edit_mode, "match_target": match_target,
            "analysis_role": "confirmation" if confirmation_sha256 else "development",
            "confirmation_contract_sha256": confirmation_sha256, "commands": commands}
    write_json_atomic(output / "plan.json", plan)
    return plan


def summarize_closed_loop(root: Path, output: Path, *, bootstrap_samples: int = 10000) -> dict:
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if plan.get("dry_run"):
        raise ValueError("cannot summarize a dry-run plan")
    if not isinstance(plan.get("initial_state_count"), int) or plan["initial_state_count"] <= 0:
        raise ValueError("closed-loop summary requires explicit initial_state_count")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    baseline: dict[tuple[int, int], dict[str, object]] = {}
    task_cells: dict[int, set[int]] = {}
    for task in plan["tasks"]:
        path = root / "baseline" / f"task{task}" / "physical_events.json"
        rows = _load_event_rows(path, plan, int(task))
        task_cells[int(task)] = set(rows)
        baseline.update({(int(task), state_id): row for state_id, row in rows.items()})

    rng, summaries = np.random.default_rng(2057736129), []
    for condition in plan["conditions"][1:]:
        differences, task_ids = [], []
        action_groups = {"full": [], "translation": [], "rotation": [], "gripper": []}
        action_task_ids = {key: [] for key in action_groups}
        event_delta = {key: [] for key in EVENT_KEYS}
        condition_cells: dict[int, set[int]] = {}
        for task in plan["tasks"]:
            path = root / condition / f"task{task}" / "physical_events.json"
            rows = _load_event_rows(path, plan, int(task))
            condition_cells[int(task)] = set(rows)
            if condition_cells[int(task)] != task_cells[int(task)]:
                raise ValueError(f"paired initial-state set differs: {path}")
            for state_id in sorted(rows):
                row = rows[state_id]
                original = baseline[(int(task), state_id)]
                differences.append(int(bool(row["success"])) - int(bool(original["success"])))
                task_ids.append(int(task))
                for event in EVENT_KEYS:
                    event_delta[event].append(
                        int(bool(row.get(event))) - int(bool(original.get(event)))
                    )
                first = np.asarray(original.get("executed_actions", []), dtype=float)
                second = np.asarray(row.get("executed_actions", []), dtype=float)
                if first.ndim != 2 or second.ndim != 2 or first.shape[1:] != second.shape[1:]:
                    raise ValueError(f"paired action shape differs: {path}")
                length = min(len(first), len(second))
                if length:
                    delta = second[:length] - first[:length]
                    action_groups["full"].append(float(np.sqrt(np.mean(delta ** 2))))
                    action_groups["translation"].append(float(np.sqrt(np.mean(delta[:, :3] ** 2))))
                    action_groups["rotation"].append(float(np.sqrt(np.mean(delta[:, 3:6] ** 2))))
                    action_groups["gripper"].append(float(np.sqrt(np.mean(delta[:, 6:7] ** 2))))
                    for metric in action_groups:
                        action_task_ids[metric].append(int(task))
        differences = np.asarray(differences, dtype=float)
        task_ids = np.asarray(task_ids, dtype=int)
        if len(differences) != sum(len(cells) for cells in task_cells.values()):
            raise ValueError(f"paired count differs for condition: {condition}")
        task_values = np.unique(task_ids)

        def task_macro(values, ids):
            values, ids = np.asarray(values, dtype=float), np.asarray(ids, dtype=int)
            return float(np.mean([
                values[ids == task].mean() for task in np.unique(ids) if np.any(ids == task)
            ]))

        boot = np.asarray([
            np.mean([rng.choice(differences[task_ids == task], np.sum(task_ids == task), replace=True).mean()
                     for task in task_values])
            for _ in range(bootstrap_samples)
        ])
        per_task = {}
        for task in task_values:
            mask = task_ids == task
            per_task[str(int(task))] = {
                "pairs": int(mask.sum()),
                "initial_state_ids": sorted(task_cells[int(task)]),
                "delta_success": float(differences[mask].mean()),
                "event_rate_deltas": {
                    key: float(np.asarray(value)[mask].mean())
                    for key, value in event_delta.items()
                },
            }
        summaries.append({
            "condition": condition,
            "pairs": len(differences),
            "paired_initial_state_ids": {str(task): sorted(values) for task, values in condition_cells.items()},
            "delta_success": task_macro(differences, task_ids),
            "delta_success_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "event_rate_deltas": {key: task_macro(value, task_ids) for key, value in event_delta.items()},
            "mean_executed_action_rms": {
                key: task_macro(value, action_task_ids[key])
                for key, value in action_groups.items() if value
            },
            "action_pairs": {key: len(value) for key, value in action_groups.items()},
            "per_task": per_task,
        })
    report = {
        "schema": SCHEMA, "kind": "paired_closed_loop_report", "complete": True,
        "initial_state_count": int(plan["initial_state_count"]),
        "paired_initial_state_ids": {str(task): sorted(values) for task, values in task_cells.items()},
        "plan_sha256": file_hash(root / "plan.json"),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "task-stratified paired initial condition",
        "primary_action_metric": "actually executed actions",
        "conditions": summaries,
    }
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
    evaluate.add_argument("--initial-state-count", type=int)
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
    context = commands.add_parser("candidate-context")
    context.add_argument("--trace", type=Path, required=True)
    context.add_argument("--candidates", type=Path, required=True)
    context.add_argument("--state-bank", type=Path, required=True)
    context.add_argument("--candidate-id", action="append", default=[])
    context.add_argument("--output", type=Path, required=True)
    expand = commands.add_parser("expand-random-controls")
    expand.add_argument("--candidates", type=Path, required=True)
    expand.add_argument("--output", type=Path, required=True)
    expand.add_argument("--count", type=int, default=8)
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
    gate.add_argument("--state-bank", type=Path, required=True)
    gate.add_argument("--executed-steps", type=int, default=10)
    gate.add_argument("--minimum-episodes", type=int, default=8)
    gate.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit-longitudinal")
    audit.add_argument("--lineage", type=Path, required=True)
    audit.add_argument("--timeline", type=Path, required=True)
    audit.add_argument("--candidates", type=Path, required=True)
    audit.add_argument("--trajectory", type=Path, required=True)
    audit.add_argument("--functional-report", action="append", default=[])
    audit.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze-confirmation")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--task", type=int, action="append", required=True)
    freeze.add_argument("--initial-state-offset", type=int, required=True)
    freeze.add_argument("--episodes", type=int, required=True)
    freeze.add_argument("--initial-state-count", type=int)
    freeze.add_argument("--used-plan", type=Path, action="append", default=[])
    freeze.add_argument("--binding", type=Path)
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
    closed.add_argument("--initial-state-count", type=int)
    closed.add_argument("--episodes", type=int, default=10)
    closed.add_argument("--dry-run", action="store_true")
    closed.add_argument("--edit-mode", choices=("additive", "suppress", "matched_suppress"),
                        default="additive")
    closed.add_argument("--confirmation-contract", type=Path)
    closed_summary = commands.add_parser("closed-loop-summary")
    closed_summary.add_argument("--root", type=Path, required=True)
    closed_summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lineage": result = checkpoint_lineage(args.checkpoint_root, args.base_checkpoint, args.step, args.output)
    elif args.command == "evaluate": result = evaluate_timeline(args.lineage, args.output, args.task or range(4), initial_state_offset=args.initial_state_offset, episodes=args.episodes, dry_run=args.dry_run, initial_state_count=args.initial_state_count)
    elif args.command == "summarize": result = summarize_timeline(args.lineage, args.root, args.output)
    elif args.command == "discover": result = discover_change_subspaces(_assignments(args.trace), before=args.before, after=args.after, tap=args.tap, rank=args.rank, output=args.output)
    elif args.command == "candidate-trajectory": result = candidate_trajectory(_assignments(args.trace), args.candidates, args.output)
    elif args.command == "candidate-interpret": result = interpret_candidates(args.trajectory, args.state_bank, args.checkpoint, args.output)
    elif args.command == "candidate-context": result = candidate_context_analysis(
        args.trace, args.candidates, args.state_bank, args.output, args.candidate_id)
    elif args.command == "expand-random-controls": result = expand_random_controls(
        args.candidates, args.output, count=args.count)
    elif args.command == "intervention-smoke": result = intervention_smoke_command(args.bank, args.dataset_root, args.checkpoint, args.contract_checkpoint, args.metadata, args.candidates, args.candidate_id, args.output, device=args.device, max_states=args.max_states, dose=args.dose, stages=args.edit_stage or tuple(range(10)), partition=args.partition, split_group=args.split_group, suite=args.suite, task_ids=args.task_id)
    elif args.command == "gate": result = offline_action_gate(
        args.baseline, _assignments(args.edited), args.candidates, args.state_bank,
        args.output, executed_steps=args.executed_steps,
        minimum_episodes=args.minimum_episodes)
    elif args.command == "audit-longitudinal": result = audit_longitudinal_evidence(
        args.lineage, args.timeline, args.candidates, args.trajectory,
        _assignments(args.functional_report), args.output)
    elif args.command == "freeze-confirmation": result = freeze_confirmation_contract(
        args.output, args.task, initial_state_offset=args.initial_state_offset,
        episodes=args.episodes, used_plans=args.used_plan,
        initial_state_count=args.initial_state_count,
        binding=(json.loads(args.binding.read_text(encoding="utf-8"))
                 if args.binding else None))
    elif args.command == "closed-loop": result = run_closed_loop(
        args.checkpoint, args.candidates, args.gate, args.output,
        args.task or range(4), args.candidate_id, dose=args.dose,
        stages=args.edit_stage or (0, 5, 9),
        initial_state_offset=args.initial_state_offset, episodes=args.episodes,
        dry_run=args.dry_run, confirmation_contract=args.confirmation_contract,
        edit_mode=args.edit_mode, initial_state_count=args.initial_state_count)
    else: result = summarize_closed_loop(args.root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
