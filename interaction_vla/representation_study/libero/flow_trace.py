"""Resumable E1 trace of SmolVLA's complete conditional action-flow path."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .latents import (_canonical_sha256, _runtime_provenance, _tree_sha256,
                      collate_state_bank_observations)
from .state_bank import load_state_bank


SCHEMA = "smolvla_conditional_flow_trace_v1"


@dataclass(frozen=True)
class FlowEdit:
    tap: str
    stages: tuple[int, ...]
    direction: torch.Tensor
    dose: float

    def __post_init__(self) -> None:
        if self.tap not in {"expert_middle", "expert_late"}:
            raise ValueError("flow edit tap must be expert_middle or expert_late")
        if not self.stages or min(self.stages) < 0 or self.direction.ndim != 1:
            raise ValueError("flow edit requires stages and a one-dimensional direction")
        if not np.isfinite(self.dose) or self.dose == 0:
            raise ValueError("flow edit dose must be finite and non-zero")


def _replace_tensor_output(output, changed: torch.Tensor):
    if isinstance(output, torch.Tensor):
        return changed
    if isinstance(output, tuple):
        return (changed, *output[1:])
    if isinstance(output, list):
        return [changed, *output[1:]]
    raise TypeError("expert tap must return a tensor or tensor-first tuple/list")


def _capture_hook(values: list[torch.Tensor], *, tap: str, edit: FlowEdit | None):
    def hook(_module, _inputs, output):
        tensor = _tensor_output(output)
        stage = len(values)
        if edit is not None and edit.tap == tap and stage in edit.stages:
            direction = edit.direction.to(device=tensor.device, dtype=tensor.dtype)
            if direction.shape != (tensor.shape[-1],):
                raise ValueError("flow edit direction does not match expert hidden width")
            tensor = tensor + edit.dose * direction
            output = _replace_tensor_output(output, tensor)
        values.append(tensor.detach().cpu())
        return output
    return hook


def install_flow_edit(policy, layer, edit: FlowEdit):
    """Install the same stage edit used by trace/query on a live policy."""
    if max(edit.stages) >= int(policy.config.num_steps):
        raise ValueError("flow edit stage exceeds the policy solver length")
    counter = 0

    def hook(_module, _inputs, output):
        nonlocal counter
        stage = counter % int(policy.config.num_steps)
        counter += 1
        if stage not in edit.stages:
            return output
        tensor = _tensor_output(output)
        direction = edit.direction.to(device=tensor.device, dtype=tensor.dtype)
        if direction.shape != (tensor.shape[-1],):
            raise ValueError("flow edit direction does not match expert hidden width")
        return _replace_tensor_output(output, tensor + edit.dose * direction)

    return layer.register_forward_hook(hook)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_atlas_provenance() -> dict[str, str]:
    path = Path(__file__).resolve().parents[3] / "research" / "action-atlas"
    if not path.is_dir():
        raise FileNotFoundError("Missing official checkout: research/action-atlas")
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD"], text=True)
    untracked = subprocess.check_output(
        ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"], text=True
    ).strip()
    if untracked:
        raise ValueError(f"Untracked Action Atlas files: {untracked}")
    sys.path.insert(0, str(path))
    return {"commit": commit, "patch_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def load_frozen_policy(checkpoint: Path, contract: Path, metadata: Path, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(str(contract.resolve()))
    config.device = device
    config.vlm_model_name = str(metadata.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(
        str(checkpoint.resolve()), config=config, local_files_only=True, strict=True,
    ).to(device).eval().requires_grad_(False)
    pre, post = make_pre_post_processors(
        policy.config, str(contract.resolve()),
        preprocessor_overrides={"device_processor": {"device": device},
                                "tokenizer_processor": {"tokenizer_name": str(metadata.resolve())}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def paired_inference_noise(state_ids: Sequence[str], repeat: int, row_shape: tuple[int, ...]) -> torch.Tensor:
    """CPU noise keyed only by state/repeat, so checkpoints receive identical epsilon."""
    if not state_ids or repeat < 0 or not row_shape or any(size <= 0 for size in row_shape):
        raise ValueError("state IDs, non-negative repeat, and positive row shape required")
    rows = []
    for state_id in state_ids:
        digest = hashlib.sha256(f"flow-trace:v1:{repeat}:{state_id}".encode()).digest()
        generator = torch.Generator(device="cpu").manual_seed(
            int.from_bytes(digest[:8], "little") % (2**63 - 1)
        )
        rows.append(torch.randn(row_shape, generator=generator, dtype=torch.float32))
    return torch.stack(rows)


def bind_policy_images(batch: dict[str, object], input_features) -> tuple[dict[str, object], dict[str, str]]:
    """Bind LIBERO's two cameras to either native or legacy SmolVLA feature names."""
    expected = sorted(key for key in input_features if key.startswith("observation.images."))
    if not expected:
        raise ValueError("checkpoint declares no image features")
    if all(key in batch for key in expected):
        return batch, {key: key for key in expected}
    sources = [key for key in ("observation.images.image", "observation.images.image2") if key in batch]
    if not sources:
        raise ValueError("LIBERO batch contains no bindable image features")
    result, binding = dict(batch), {}
    for index, target in enumerate(expected):
        if index < len(sources):
            result[target] = batch[sources[index]]
            binding[target] = sources[index]
        else:
            source = batch[sources[0]]
            if not isinstance(source, torch.Tensor):
                raise TypeError("image features must be tensors")
            result[target] = torch.zeros_like(source)
            binding[target] = "zero_like:" + sources[0]
    return result, binding


def select_records(records, split, *, partition: str, max_states: int, seed: int = 42,
                   suite: str | None = None, task_ids: Sequence[int] = ()):
    """Label-blind, task-balanced sampling; episode diversity is favored within each task."""
    if max_states <= 0:
        raise ValueError("max_states must be positive")
    groups = {}
    for record in records:
        if (split.assignments[record.state_id] == partition
                and (suite is None or record.suite == suite)
                and (not task_ids or record.task_id in task_ids)):
            groups.setdefault((record.suite, record.task_id), {}).setdefault(
                record.lerobot_episode_index, []
            ).append(record)
    if not groups:
        raise ValueError(f"No states in partition {partition}")
    rng = np.random.default_rng(seed)
    task_queues = {}
    for task, episodes in sorted(groups.items()):
        queues = []
        for episode, rows in sorted(episodes.items()):
            rows = sorted(rows, key=lambda row: row.state_id)
            queues.append([rows[i] for i in rng.permutation(len(rows))])
        queue = []
        while any(queues):
            for rows in queues:
                if rows:
                    queue.append(rows.pop())
        task_queues[task] = queue
    selected = []
    while len(selected) < max_states and any(task_queues.values()):
        for task in sorted(task_queues):
            if task_queues[task] and len(selected) < max_states:
                selected.append(task_queues[task].pop())
    return selected


def _tensor_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("expert tap must return a tensor or a tuple whose first item is a tensor")


def trace_action_flow(policy, processed, noise: torch.Tensor, middle_layer, late_layer,
                      edit: FlowEdit | None = None):
    """Capture every actual denoise call while leaving native integration untouched."""
    flow = policy.model
    original = flow.denoise_step
    middle, late, noisy, velocity, sigma = [], [], [], [], []
    handles = [
        middle_layer.register_forward_hook(_capture_hook(middle, tap="expert_middle", edit=edit)),
        late_layer.register_forward_hook(_capture_hook(late, tap="expert_late", edit=edit)),
    ]

    def traced(*args, **kwargs):
        x_t = kwargs.get("x_t", args[2] if len(args) > 2 else None)
        timestep = kwargs.get("timestep", args[3] if len(args) > 3 else None)
        if x_t is None or timestep is None:
            raise RuntimeError("SmolVLA denoise_step signature changed")
        before = len(middle), len(late)
        result = original(*args, **kwargs)
        if (len(middle), len(late)) != (before[0] + 1, before[1] + 1):
            raise RuntimeError("expert taps did not fire exactly once in a denoise step")
        noisy.append(x_t.detach().cpu())
        velocity.append(result.detach().cpu())
        sigma.append(float(torch.as_tensor(timestep).flatten()[0].cpu()))
        return result

    flow.denoise_step = traced
    try:
        policy.reset()
        with torch.inference_mode():
            raw_actions = policy.predict_action_chunk(processed, noise=noise)
    finally:
        flow.denoise_step = original
        for handle in handles:
            handle.remove()
    steps = int(policy.config.num_steps)
    if not (len(noisy) == len(velocity) == len(middle) == len(late) == steps):
        raise RuntimeError(f"expected {steps} complete denoise stages")
    expected_sigma = np.arange(steps, 0, -1, dtype=np.float32) / steps
    np.testing.assert_allclose(np.asarray(sigma), expected_sigma, atol=1e-6, rtol=0)
    final_x = noisy[-1] - velocity[-1] / steps
    np.testing.assert_allclose(
        final_x[..., : raw_actions.shape[-1]].float().numpy(),
        raw_actions.detach().cpu().float().numpy(), atol=2e-5, rtol=0,
    )
    return {
        "epsilon": noise.detach().cpu().float().numpy(),
        "sigma": expected_sigma,
        "x_sigma": torch.stack(noisy, dim=1).float().numpy(),
        "velocity": torch.stack(velocity, dim=1).float().numpy(),
        "expert_middle": torch.stack(middle, dim=1).float().numpy(),
        "expert_late": torch.stack(late, dim=1).float().numpy(),
        "final_x0": final_x.float().numpy(),
        "action_normalized": raw_actions.detach().cpu().float().numpy(),
    }


def query_action_flow_at_points(policy, processed, noise: torch.Tensor, reference_x_sigma,
                                reference_sigma, middle_layer, late_layer,
                                edit: FlowEdit | None = None):
    """Query one checkpoint at another trace's fixed (observation, x_sigma, sigma) points."""
    points = torch.as_tensor(reference_x_sigma, device=noise.device, dtype=noise.dtype)
    sigmas = np.asarray(reference_sigma, dtype=np.float32)
    steps = int(policy.config.num_steps)
    if points.shape != (len(noise), steps, *noise.shape[1:]) or sigmas.shape != (steps,):
        raise ValueError("reference flow points do not match batch/noise/solver contract")
    expected = np.arange(steps, 0, -1, dtype=np.float32) / steps
    np.testing.assert_allclose(sigmas, expected, atol=1e-6, rtol=0)

    flow = policy.model
    original = flow.denoise_step
    middle, late, velocity = [], [], []
    handles = [
        middle_layer.register_forward_hook(_capture_hook(middle, tap="expert_middle", edit=edit)),
        late_layer.register_forward_hook(_capture_hook(late, tap="expert_late", edit=edit)),
    ]

    def queried(*args, **kwargs):
        index = len(velocity)
        timestep = kwargs.get("timestep", args[3] if len(args) > 3 else None)
        if timestep is None or index >= steps:
            raise RuntimeError("SmolVLA denoise_step signature/count changed")
        observed = float(torch.as_tensor(timestep).flatten()[0].detach().cpu())
        if not np.isclose(observed, sigmas[index], atol=1e-6, rtol=0):
            raise RuntimeError("solver sigma differs from frozen reference")
        fixed = points[:, index]
        if "x_t" in kwargs:
            kwargs = dict(kwargs)
            kwargs["x_t"] = fixed
        else:
            args = list(args)
            args[2] = fixed
        before = len(middle), len(late)
        result = original(*args, **kwargs)
        if (len(middle), len(late)) != (before[0] + 1, before[1] + 1):
            raise RuntimeError("expert taps did not fire exactly once in a fixed-point query")
        velocity.append(result.detach().cpu())
        return result

    flow.denoise_step = queried
    try:
        policy.reset()
        with torch.inference_mode():
            policy.predict_action_chunk(processed, noise=noise)
    finally:
        flow.denoise_step = original
        for handle in handles:
            handle.remove()
    if not (len(velocity) == len(middle) == len(late) == steps):
        raise RuntimeError(f"expected {steps} complete fixed-point queries")
    return {
        "sigma": sigmas,
        "x_sigma": points.detach().cpu().float().numpy(),
        "velocity": torch.stack(velocity, dim=1).float().numpy(),
        "expert_middle": torch.stack(middle, dim=1).float().numpy(),
        "expert_late": torch.stack(late, dim=1).float().numpy(),
    }


def _read_shard(path: Path, ids, binding_hash: str, repeats: int, binding):
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["binding_sha256"] != binding_hash or receipt["sha256"] != file_hash(path):
        raise ValueError(f"corrupt or differently bound shard: {path}")
    with np.load(path, allow_pickle=False) as data:
        if data["state_ids"].tolist() != list(ids):
            raise ValueError("shard state IDs/order mismatch")
        required = {"epsilon", "sigma", "x_sigma", "velocity", "expert_middle", "expert_late"}
        if binding.get("query_mode") == "natural_integration":
            required.update({"final_x0", "action_normalized", "action_postprocessed"})
        if not required.issubset(data.files):
            raise ValueError("incomplete flow trace shard")
        if data["epsilon"].shape[:2] != (len(ids), repeats):
            raise ValueError("unexpected epsilon batch/repeat shape")
        if data["x_sigma"].shape[:3] != (len(ids), repeats, binding["num_steps"]):
            raise ValueError("unexpected flow-stage shape")
        if data["sigma"].shape != (repeats, binding["num_steps"]):
            raise ValueError("unexpected sigma repeat/stage shape")
        if any(not np.isfinite(data[name]).all() for name in required):
            raise ValueError("non-finite flow trace")


def plan(bank: Path, *, partition: str, max_states: int, repeats: int,
         suite: str | None = None, task_ids: Sequence[int] = (),
         split_group: str = "task"):
    if split_group not in {"task", "episode"}:
        raise ValueError("split_group must be task or episode")
    records, manifest, task_split, episode_split = load_state_bank(bank)
    split = task_split if split_group == "task" else episode_split
    if not manifest.get("audit_passed"):
        raise ValueError("StateBank audit has not passed")
    selected = select_records(records, split, partition=partition, max_states=max_states,
                              suite=suite, task_ids=task_ids)
    # Current SmolVLA expert width; run records observed shapes in its manifest.
    expert_hidden_dim = 720
    bytes_per_state_repeat = 4 * (50 * 32 * (1 + 2 * 10 + 1) + 2 * 10 * 50 * expert_hidden_dim + 2 * 50 * 7)
    return {"schema": SCHEMA, "partition": partition, "split_group": split_group,
            "suite": suite,
            "task_ids": list(task_ids), "requested_states": max_states,
            "selected_states": len(selected), "noise_repeats": repeats,
            "action_chunk_generations_per_checkpoint": len(selected) * repeats,
            "denoise_stage_records_per_checkpoint": len(selected) * repeats * 10,
            "storage_estimate_expert_hidden_dim": expert_hidden_dim,
            "estimated_uncompressed_bytes_per_checkpoint": len(selected) * repeats * bytes_per_state_repeat,
            "state_ids": [row.state_id for row in selected]}


def load_flow_edit(path: Path, candidate_id: str, dose: float,
                   stages: Sequence[int]) -> tuple[FlowEdit, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("candidates", [])
    row = next((item for item in rows if item.get("id") == candidate_id), None)
    if row is None:
        raise ValueError(f"candidate not found: {candidate_id}")
    direction = torch.tensor(row["direction"], dtype=torch.float32)
    edit = FlowEdit(str(row["tap"]), tuple(sorted(set(int(x) for x in stages))), direction, dose)
    return edit, file_hash(path)


def run(bank: Path, dataset_root: Path, checkpoint: Path, contract: Path, metadata: Path, output: Path, *,
        device: str, batch_size: int, partition: str, max_states: int, repeats: int,
        reference_trace: Path | None = None, candidate_path: Path | None = None,
        candidate_id: str | None = None, dose: float = 1.0,
        edit_stages: Sequence[int] = (0, 5, 9), suite: str | None = None,
        task_ids: Sequence[int] = (), split_group: str = "task"):
    if batch_size <= 0 or repeats <= 0:
        raise ValueError("batch_size and repeats must be positive")
    if split_group not in {"task", "episode"}:
        raise ValueError("split_group must be task or episode")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("Set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
    records, manifest, task_split, episode_split = load_state_bank(bank)
    split = task_split if split_group == "task" else episode_split
    if not manifest.get("audit_passed"):
        raise ValueError("StateBank audit has not passed")
    selected = select_records(records, split, partition=partition, max_states=max_states,
                              suite=suite, task_ids=task_ids)
    reference = None
    if reference_trace is not None:
        from .flow_diff import load_trace

        reference_ids, reference_arrays, reference_binding, reference_manifest = load_trace(reference_trace)
        if reference_binding.get("query_mode", "natural_integration") != "natural_integration":
            raise ValueError("reference trace must be a natural integration trace")
        if reference_ids.tolist() != [row.state_id for row in selected]:
            raise ValueError("reference trace state IDs/order differ from this selection")
        if reference_arrays["epsilon"].shape[1] != repeats:
            raise ValueError("reference trace noise repeats differ")
        reference = (reference_arrays, reference_binding, reference_manifest)
    edit = None
    candidate_hash = None
    if candidate_path is not None or candidate_id is not None:
        if candidate_path is None or candidate_id is None:
            raise ValueError("candidate path and ID must be supplied together")
        edit, candidate_hash = load_flow_edit(candidate_path, candidate_id, dose, edit_stages)
    revisions = {row.source_revision for row in selected}
    if len(revisions) != 1:
        raise ValueError("mixed dataset revisions")
    revision = revisions.pop()
    source = action_atlas_provenance()
    from experiments.model_adapters import SmolVLAAdapter
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    started = time.perf_counter()
    dataset = LeRobotDataset("lerobot/libero", root=dataset_root, revision=revision,
                             episodes=sorted({row.lerobot_episode_index for row in selected}),
                             video_backend="torchcodec")
    checkpoint_hash = _tree_sha256(checkpoint)
    policy, pre, post = load_frozen_policy(checkpoint, contract, metadata, device)
    if edit is not None and max(edit.stages) >= int(policy.config.num_steps):
        raise ValueError("flow edit stage exceeds the policy solver length")
    adapter = SmolVLAAdapter()
    adapter.policy = policy
    layers = adapter.get_layer_groups()["expert"]
    middle_index, late_index = len(layers) // 2, len(layers) - 1
    _, image_binding = bind_policy_images(
        collate_state_bank_observations(selected[:1], dataset), policy.config.input_features
    )
    binding = {
        "schema": SCHEMA, "state_bank_sha256": file_hash(bank / "manifest.json"),
        "checkpoint_tree_sha256": checkpoint_hash, "metadata_tree_sha256": _tree_sha256(metadata),
        "contract_tree_sha256": _tree_sha256(contract),
        "dataset_revision": revision, "dataset_scientific_sha256": _canonical_sha256({
            str(path.relative_to(dataset_root)): file_hash(path)
            for pattern in ("meta/**/*.json", "meta/**/*.parquet", "data/**/*.parquet", "videos/**/*.mp4")
            for path in sorted(dataset_root.glob(pattern))}),
        "upstream": source, "partition": partition, "split_group": split_group,
        "selection_seed": 42,
        "suite_filter": suite, "task_id_filter": list(task_ids),
        "state_ids": [row.state_id for row in selected], "batch_size": batch_size,
        "noise_repeats": repeats, "noise": "flow-trace:v1 keyed by state_id and repeat; checkpoint-independent CPU float32",
        "num_steps": int(policy.config.num_steps), "chunk_size": int(policy.config.chunk_size),
        "max_action_dim": int(policy.config.max_action_dim), "expert_layer_count": len(layers),
        "expert_middle_index": middle_index, "expert_late_index": late_index,
        "query_mode": "fixed_reference_points" if reference is not None else "natural_integration",
        "reference_trace": (
            {"path": str(reference_trace.resolve()),
             "binding_sha256": reference[2]["binding_sha256"]}
            if reference is not None else None
        ),
        "flow_edit": (
            {"candidate_path": str(candidate_path.resolve()), "candidate_sha256": candidate_hash,
             "candidate_id": candidate_id, "tap": edit.tap, "stages": list(edit.stages),
             "dose": edit.dose}
            if edit is not None else None
        ),
        "image_binding": image_binding,
        "runtime": _runtime_provenance(policy, batch_size=batch_size),
        "source_sha256": {path.name: file_hash(path) for path in
                          (Path(__file__), Path(__file__).with_name("latents.py"))},
    }
    if reference is not None:
        reference_binding = reference[1]
        for key in ("state_bank_sha256", "dataset_revision", "dataset_scientific_sha256",
                    "contract_tree_sha256", "num_steps", "chunk_size", "max_action_dim"):
            if reference_binding.get(key) != binding.get(key):
                raise ValueError(f"reference trace contract differs: {key}")
    if (output / "binding.json").exists():
        if json.loads((output / "binding.json").read_text()) != binding:
            raise ValueError("resume binding differs; use a new output directory")
    elif output.exists():
        raise FileExistsError(f"unbound existing output: {output}")
    else:
        write_json_atomic(output / "binding.json", binding)
    binding_hash = file_hash(output / "binding.json")
    shards, measured, resumed = [], [], 0
    for start in range(0, len(selected), batch_size):
        rows = selected[start:start + batch_size]
        ids = [row.state_id for row in rows]
        path = output / "shards" / f"{start:06d}.npz"
        if path.exists() and path.with_suffix(".json").exists():
            _read_shard(path, ids, binding_hash, repeats, binding)
            resumed += len(rows)
        else:
            tick = time.perf_counter()
            batch, observed_binding = bind_policy_images(
                collate_state_bank_observations(rows, dataset), policy.config.input_features
            )
            if observed_binding != image_binding:
                raise ValueError("image binding changed within one trace run")
            processed = pre(batch)
            traces = []
            for repeat in range(repeats):
                if reference is None:
                    noise = paired_inference_noise(
                        ids, repeat, (binding["chunk_size"], binding["max_action_dim"])
                    ).to(device)
                    trace = trace_action_flow(
                        policy, processed, noise, layers[middle_index], layers[late_index], edit
                    )
                    action_dim = trace["action_normalized"].shape[-1]
                    trace["action_postprocessed"] = post(
                        torch.from_numpy(trace["action_normalized"]).to(device).reshape(-1, action_dim)
                    ).reshape(len(rows), binding["chunk_size"], action_dim).detach().cpu().float().numpy()
                else:
                    reference_arrays = reference[0]
                    stop = start + len(rows)
                    noise = torch.from_numpy(reference_arrays["epsilon"][start:stop, repeat]).to(device)
                    trace = query_action_flow_at_points(
                        policy, processed, noise,
                        reference_arrays["x_sigma"][start:stop, repeat],
                        reference_arrays["sigma"][repeat],
                        layers[middle_index], layers[late_index], edit,
                    )
                    trace["epsilon"] = noise.detach().cpu().float().numpy()
                traces.append(trace)
            arrays = {
                name: np.stack([trace[name] for trace in traces], axis=0 if name == "sigma" else 1)
                for name in traces[0]
            }
            buffer = io.BytesIO()
            np.savez(buffer, state_ids=np.asarray(ids), **arrays)
            write_bytes_atomic(path, buffer.getvalue())
            receipt = {"sha256": file_hash(path), "binding_sha256": binding_hash,
                       "states": len(rows), "seconds": time.perf_counter() - tick}
            write_json_atomic(path.with_suffix(".json"), receipt)
            _read_shard(path, ids, binding_hash, repeats, binding)
            measured.append(receipt)
        shards.append(str(path.relative_to(output)))
        progress = {"complete": False, "states": min(start + batch_size, len(selected)),
                    "expected": len(selected), "resumed_states": resumed}
        write_json_atomic(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    result = {"schema": SCHEMA, "complete": True, "binding_sha256": binding_hash,
              "states": len(selected), "noise_repeats": repeats, "shards": shards,
              "measured_states": sum(item["states"] for item in measured),
              "resumed_states": resumed, "elapsed_seconds": time.perf_counter() - started,
              "batch_seconds": [item["seconds"] for item in measured],
              "closed_loop": "not_run"}
    with np.load(output / shards[0], allow_pickle=False) as first:
        result["array_contract"] = {
            name: {"shape": list(first[name].shape), "dtype": str(first[name].dtype)}
            for name in first.files if name != "state_ids"
        }
    write_json_atomic(output / "manifest.json", result)
    write_json_atomic(output / "progress.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--bank", type=Path, default=Path("outputs/representation_study/libero_smolvla/state_bank"))
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/datasets/libero"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--contract-checkpoint", type=Path,
                        help="Checkpoint supplying the dataset-bound config/processors; defaults to --checkpoint")
    parser.add_argument("--metadata", type=Path, default=Path("outputs/pretrained/SmolVLM2-500M-Instruct"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--split-group", choices=("task", "episode"), default="task")
    parser.add_argument("--suite")
    parser.add_argument("--task-id", type=int, action="append", default=[])
    parser.add_argument("--max-states", type=int, default=512)
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--reference-trace", type=Path,
                        help="Natural trace whose x_sigma/sigma points are held fixed for this checkpoint")
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--edit-stage", type=int, action="append")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(args.bank, partition=args.partition, max_states=args.max_states,
                              repeats=args.noise_repeats, suite=args.suite,
                              task_ids=args.task_id, split_group=args.split_group), indent=2))
        return
    if args.checkpoint is None or args.output is None:
        parser.error("run requires --checkpoint and --output")
    print(json.dumps(run(args.bank, args.dataset_root, args.checkpoint,
                         args.contract_checkpoint or args.checkpoint, args.metadata, args.output,
                         device=args.device, batch_size=args.batch_size, partition=args.partition,
                         max_states=args.max_states, repeats=args.noise_repeats,
                         reference_trace=args.reference_trace, candidate_path=args.candidates,
                         candidate_id=args.candidate_id, dose=args.dose,
                         edit_stages=args.edit_stage or (0, 5, 9), suite=args.suite,
                         task_ids=args.task_id, split_group=args.split_group), indent=2))


if __name__ == "__main__":
    main()
