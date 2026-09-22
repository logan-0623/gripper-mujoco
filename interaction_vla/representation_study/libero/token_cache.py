"""Resumable frozen SmolVLA token/action cache and train-only noise pilot."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .latents import (_canonical_sha256, _runtime_provenance, _tree_sha256,
                      collate_state_bank_observations, deterministic_inference_noise)
from .predictive_states import file_hash, upstream
from .smolvla_smoke import load_frozen_policy
from .state_bank import load_state_bank


def pilot_records(records, split):
    """Four lexically selected TRAIN tasks, 16 spaced states per first episode."""
    train = [r for r in records if split.assignments[r.state_id] == "train"]
    tasks = sorted({(r.suite, r.task_id) for r in train})[:4]
    chosen = []
    for task in tasks:
        rows = [r for r in train if (r.suite, r.task_id) == task]
        episode = min(r.lerobot_episode_index for r in rows)
        rows = sorted((r for r in rows if r.lerobot_episode_index == episode), key=lambda r: r.frame_index)
        chosen.extend(rows[i] for i in np.linspace(0, len(rows)-1, min(16, len(rows)), dtype=int))
    if not chosen:
        raise ValueError("No train pilot states")
    return chosen


def read_shard(path, ids, binding_hash):
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["binding_sha256"] != binding_hash or file_hash(path) != receipt["sha256"]:
        raise ValueError(f"Corrupt or differently bound shard: {path}")
    with np.load(path, allow_pickle=False) as data:
        if data["state_ids"].tolist() != list(ids):
            raise ValueError("Shard state IDs/order mismatch")
        tokens, actions = data["tokens"].copy(), data["actions"].copy()
    if tokens.shape != (len(ids), 50, 480) or actions.shape != (len(ids), 50, 7):
        raise ValueError("Unexpected token/action shape")
    if any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (tokens, actions)):
        raise ValueError("Token/action arrays must be finite float32")
    return tokens, actions, receipt


def extract(bank, dataset_root, checkpoint, metadata, output, *, device="cpu", batch_size=4,
            noise_seed=0, pilot=False):
    if batch_size <= 0 or noise_seed < 0:
        raise ValueError("Positive batch size and non-negative noise seed required")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("Set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
    records, manifest, split, _ = load_state_bank(bank)
    if not manifest.get("audit_passed"):
        raise ValueError("StateBank audit has not passed")
    selected = pilot_records(records, split) if pilot else records
    revisions = {r.source_revision for r in records}
    if len(revisions) != 1:
        raise ValueError("Mixed dataset revisions")
    revision = revisions.pop()
    source = upstream("action-atlas")
    from experiments.hooks import ActivationCaptureHook
    from experiments.model_adapters import SmolVLAAdapter
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    started = time.perf_counter()
    dataset = LeRobotDataset("lerobot/libero", root=dataset_root, revision=revision,
                            episodes=sorted({r.lerobot_episode_index for r in selected}),
                            video_backend="torchcodec")
    checkpoint_hash = _tree_sha256(checkpoint)
    policy, pre, post = load_frozen_policy(checkpoint, metadata, device)
    adapter = SmolVLAAdapter()
    adapter.policy = policy
    layer = adapter.get_layer_groups()["expert"][-1]
    binding = {
        "schema": "smolvla_tokens_actions_v1", "state_bank_sha256": file_hash(bank / "manifest.json"),
        "checkpoint_tree_sha256": checkpoint_hash, "metadata_tree_sha256": _tree_sha256(metadata),
        "dataset_revision": revision, "upstream": source, "pilot": pilot,
        "dataset_scientific_sha256": _canonical_sha256({
            str(p.relative_to(dataset_root)): file_hash(p)
            for pattern in ("meta/**/*.json", "meta/**/*.parquet", "data/**/*.parquet", "videos/**/*.mp4")
            for p in sorted(dataset_root.glob(pattern))}),
        "state_ids": [r.state_id for r in selected], "batch_size": batch_size,
        "noise_seed": noise_seed, "noise": "state-keyed CPU float32 normal, explicit predict_action_chunk input",
        "tap": "action_atlas/expert/31/mlp/output", "denoising_call": "last_of_10",
        "token_shape": [50, 480], "action_shape": [50, 7], "dtype": "float32",
        "action_coordinates": "checkpoint_postprocessed_before_environment_clipping_or_scaling",
        "runtime": _runtime_provenance(policy, batch_size=batch_size),
        "source_sha256": {p.name: file_hash(p) for p in
                          (Path(__file__), Path(__file__).with_name("smolvla_smoke.py"),
                           Path(__file__).with_name("latents.py"))},
    }
    if (output / "binding.json").exists():
        if json.loads((output / "binding.json").read_text()) != binding:
            raise ValueError("Resume binding differs; use a new output directory")
    elif output.exists():
        raise FileExistsError(f"Unbound existing output: {output}")
    else:
        write_json_atomic(output / "binding.json", binding)
    binding_hash = file_hash(output / "binding.json")
    if (output / "manifest.json").exists():
        load_summaries(output, bank, require_full=not pilot)
        return json.loads((output / "manifest.json").read_text())
    setup_seconds = time.perf_counter() - started
    measured, resumed, shards = [], 0, []
    for start in range(0, len(selected), batch_size):
        rows = selected[start:start+batch_size]
        ids = [r.state_id for r in rows]
        path = output / "shards" / f"{start:06d}.npz"
        if path.exists() and path.with_suffix(".json").exists():
            read_shard(path, ids, binding_hash)
            resumed += len(rows)
        else:
            tick = time.perf_counter()
            # Validate the non-image source table once per frame, without decoding twice.
            for r in rows:
                sample = dataset.hf_dataset[dataset.absolute_to_relative_idx[r.observation.dataset_index]]
                for key, expected in (("index", r.observation.dataset_index),
                                      ("frame_index", r.frame_index), ("episode_index", r.lerobot_episode_index)):
                    if int(sample[key]) != expected:
                        raise ValueError(f"StateBank/source mismatch: {key}")
                for key, expected in (("observation.state", r.observation.robot_state),
                                      ("action", r.observation.action), ("timestamp", r.observation.timestamp)):
                    np.testing.assert_allclose(np.asarray(sample[key]), expected, rtol=0, atol=1e-6)
                task = dataset.meta.tasks.iloc[int(sample["task_index"])].name
                if task != r.language:
                    raise ValueError("StateBank/source language mismatch")
            batch = collate_state_bank_observations(rows, dataset)
            for key in (rows[0].observation.global_rgb_key, rows[0].observation.wrist_rgb_key):
                x = batch[key]
                if x.shape != (len(rows), 3, 256, 256) or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                    raise ValueError("Invalid RGB batch")
            processed = pre(batch)
            noise = deterministic_inference_noise(
                ids, checkpoint_id=f"{checkpoint_hash}:noise={noise_seed}",
                row_shape=(policy.config.chunk_size, policy.config.max_action_dim),
                device=torch.device("cpu"), dtype=torch.float32,
            ).to(device)
            capture = ActivationCaptureHook()
            handle = layer.register_forward_hook(capture)
            try:
                policy.reset()
                with torch.inference_mode():
                    raw_actions = policy.predict_action_chunk(processed, noise=noise)
                    actions = post(raw_actions.reshape(-1, 7)).reshape(len(rows), 50, 7).float().cpu().numpy()
                if len(capture.activations) != 10:
                    raise ValueError("Expected 10 expert MLP calls")
                tokens = capture.activations[-1].float().numpy()
            finally:
                handle.remove()
            # Check the new chunk path against native select_action, once per run.
            if not measured and resumed == 0:
                policy.reset()
                repeated = ActivationCaptureHook()
                handle = layer.register_forward_hook(repeated)
                try:
                    with torch.inference_mode():
                        first = post(policy.select_action(processed, noise=noise)).float().cpu().numpy()
                finally:
                    handle.remove()
                np.testing.assert_allclose(actions[:, 0], first, atol=1e-5, rtol=0)
                np.testing.assert_allclose(tokens, repeated.activations[-1].float().numpy(), atol=1e-5, rtol=0)
            buffer = io.BytesIO()
            np.savez(buffer, state_ids=np.array(ids), tokens=tokens, actions=actions)
            write_bytes_atomic(path, buffer.getvalue())
            receipt = {"sha256": file_hash(path), "binding_sha256": binding_hash,
                       "states": len(rows), "seconds": time.perf_counter()-tick,
                       "repeat_and_select_action_check": start == 0}
            write_json_atomic(path.with_suffix(".json"), receipt)
            read_shard(path, ids, binding_hash)
            measured.append(receipt)
        shards.append(str(path.relative_to(output)))
        done = start + len(rows)
        if done % 32 == 0 or done == len(selected):
            elapsed = time.perf_counter() - started
            progress = {"complete": False, "states": done, "expected": len(selected),
                        "resumed_states": resumed, "elapsed_seconds": elapsed}
            write_json_atomic(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    result = {"schema": binding["schema"], "complete": True, "full_state_bank": not pilot,
              "binding_sha256": binding_hash, "states": len(selected), "shards": shards,
              "setup_seconds": setup_seconds, "elapsed_seconds": time.perf_counter()-started,
              "measured_states": sum(x["states"] for x in measured), "resumed_states": resumed,
              "batch_seconds": [x["seconds"] for x in measured],
              "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "mps_driver_bytes": torch.mps.driver_allocated_memory() if device == "mps" else None,
              "rl": "frozen", "closed_loop": "not_run"}
    first_receipt = json.loads((output / shards[0]).with_suffix(".json").read_text())
    result["repeat_and_select_action_check"] = "passed" if first_receipt["repeat_and_select_action_check"] else "not_run"
    write_json_atomic(output / "manifest.json", result)
    write_json_atomic(output / "progress.json", result)
    return result


def load_summaries(cache, bank, *, require_full=True):
    manifest = json.loads((cache / "manifest.json").read_text())
    binding = json.loads((cache / "binding.json").read_text())
    if manifest.get("schema") != "smolvla_tokens_actions_v1" or binding.get("schema") != manifest["schema"]:
        raise ValueError("Unknown token cache schema")
    if not manifest["complete"] or (require_full and not manifest["full_state_bank"]):
        raise ValueError("Complete full-StateBank cache required")
    if manifest["binding_sha256"] != file_hash(cache / "binding.json") or binding["state_bank_sha256"] != file_hash(bank / "manifest.json"):
        raise ValueError("Cache binding/StateBank mismatch")
    if binding["batch_size"] <= 0 or not binding["state_ids"]:
        raise ValueError("Empty cache or invalid batch size")
    expected = [f"shards/{i:06d}.npz" for i in range(0, len(binding["state_ids"]), binding["batch_size"])]
    if manifest["shards"] != expected:
        raise ValueError("Shard list/order/coverage mismatch")
    means, firsts, actions = [], [], []
    offset = 0
    for relative in manifest["shards"]:
        ids = binding["state_ids"][offset:offset+binding["batch_size"]]
        tokens, chunk, _ = read_shard(cache / relative, ids, manifest["binding_sha256"])
        means.append(tokens.mean(axis=1))
        firsts.append(tokens[:, 0].copy())  # Do not retain the full token tensor through a view.
        actions.append(chunk)
        offset += len(ids)
    if offset != len(binding["state_ids"]) or offset != manifest["states"]:
        raise ValueError("Incomplete shard coverage")
    ids = binding["state_ids"]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate state IDs")
    if require_full:
        records, _, _, _ = load_state_bank(bank)
        if ids != [r.state_id for r in records]:
            raise ValueError("Full cache state IDs/order differ from StateBank")
    return np.concatenate(means), np.concatenate(firsts), np.concatenate(actions), binding


def noise_report(root, bank):
    loaded = [load_summaries(root / f"noise_{seed}", bank, require_full=False) for seed in range(3)]
    comparable = [{k: v for k, v in item[3].items() if k != "noise_seed"} for item in loaded]
    if not all(x == comparable[0] for x in comparable):
        raise ValueError("Noise pilot bindings may differ only in noise seed")
    metrics = {}
    for label, column in (("token_mean", 0), ("first_token", 1), ("action_chunk", 2)):
        x = np.stack([item[column] for item in loaded]).astype(np.float64)
        within = float(x.var(axis=0).mean())
        between = float(x.mean(axis=0).var(axis=0).mean())
        metrics[label] = {"within_state_noise_variance": within, "between_state_mean_variance": between,
                          "noise_to_state_variance_ratio": within / between if between > 0 else None}
    report = {"states": len(loaded[0][3]["state_ids"]), "noise_seeds": [0, 1, 2],
              "partition": "train", "metrics": metrics,
              "interpretation": "Sensitivity diagnostic; neither physical semantics nor predictive robustness established",
              "runs": [json.loads((root / f"noise_{seed}" / "manifest.json").read_text()) for seed in range(3)]}
    write_json_atomic(root / "noise_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "extract", "evaluate"))
    parser.add_argument("--bank", type=Path, default=Path("outputs/representation_study/libero_smolvla/state_bank"))
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/datasets/libero"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/pretrained/smolvla_libero"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/pretrained/SmolVLM2-500M-Instruct"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.command == "evaluate":
        if args.cache is None:
            parser.error("evaluate requires --cache")
        from .token_readouts import run
        run(args.bank, args.cache, args.output)
    else:
        for seed in (range(3) if args.command == "pilot" else [0]):
            dest = args.output / f"noise_{seed}" if args.command == "pilot" else args.output
            extract(args.bank, args.dataset_root, args.checkpoint, args.metadata, dest,
                    device=args.device, batch_size=args.batch_size, noise_seed=seed, pilot=args.command == "pilot")
        if args.command == "pilot":
            print(json.dumps(noise_report(args.output, args.bank)["metrics"], indent=2))


if __name__ == "__main__":
    main()
