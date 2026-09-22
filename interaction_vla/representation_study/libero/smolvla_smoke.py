"""Bounded real-observation smoke, not a full cache or a LIBERO rollout.

Uses native LeRobot preprocessing and official Action Atlas layer access/hooks.
Run with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 after downloading inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ..state_bank.io import write_json_atomic
from .latents import _runtime_provenance, _tree_sha256, collate_state_bank_observations
from .predictive_states import file_hash, paired_action_effects, upstream
from .state_bank import load_state_bank


def load_frozen_policy(checkpoint: Path, metadata: Path, device: str = "cpu"):
    """Native offline loader; keep checkpoint processors/statistics unchanged."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(str(checkpoint.resolve()))
    config.device = device
    config.vlm_model_name = str(metadata.resolve())
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(
        str(checkpoint.resolve()), config=config, local_files_only=True, strict=True,
    ).to(device).eval().requires_grad_(False)
    pre, post = make_pre_post_processors(
        policy.config, str(checkpoint.resolve()),
        preprocessor_overrides={"device_processor": {"device": device},
                                "tokenizer_processor": {"tokenizer_name": str(metadata.resolve())}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def run(bank: Path, dataset_root: Path, checkpoint: Path, metadata: Path,
        output: Path, max_states: int = 4):
    if not 1 <= max_states <= 16:
        raise ValueError("Smoke budget is 1–16 train frames; full extraction is a separate gate")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("Set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; download separately")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    for path in (bank, dataset_root, checkpoint, metadata):
        if not path.is_dir():
            raise FileNotFoundError(path)
    records, manifest, split, _ = load_state_bank(bank)
    if not manifest.get("audit_passed"):
        raise ValueError("StateBank audit must pass")
    revisions = {r.source_revision for r in records}
    if len(revisions) != 1:
        raise ValueError("Expected exactly one StateBank source revision")
    revision = revisions.pop()
    train = [r for r in records if split.assignments[r.state_id] == "train"]
    if not train:
        raise ValueError("No train states")
    episode = train[0].lerobot_episode_index
    selected = sorted((r for r in train if r.lerobot_episode_index == episode),
                      key=lambda r: r.frame_index)[:max_states]

    source = upstream("action-atlas")
    from experiments.hooks import ActivationCaptureHook
    from experiments.model_adapters import SmolVLAAdapter
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    started = time.perf_counter()
    dataset = LeRobotDataset("lerobot/libero", root=dataset_root, revision=revision,
                            episodes=[episode], video_backend="torchcodec")
    checkpoint_hash = _tree_sha256(checkpoint)
    policy, pre, post = load_frozen_policy(checkpoint, metadata)
    adapter = SmolVLAAdapter()
    adapter.policy = policy
    layers = adapter.get_layer_groups()["expert"]
    layer = layers[-1]  # Predeclared last expert MLP; no layer/feature search in smoke.
    setup_seconds = time.perf_counter() - started
    rows, values = [], []
    for record in selected:
        started_frame = time.perf_counter()
        sample = dataset[dataset.absolute_to_relative_idx[record.observation.dataset_index]]
        for key, expected in (("index", record.observation.dataset_index),
                              ("episode_index", record.lerobot_episode_index),
                              ("frame_index", record.frame_index)):
            if int(sample[key]) != expected:
                raise ValueError(f"StateBank/data mismatch: {key}")
        if sample["task"] != record.language:
            raise ValueError("StateBank/data language mismatch")
        for key, expected in (("observation.state", record.observation.robot_state),
                              ("action", record.observation.action),
                              ("timestamp", record.observation.timestamp)):
            np.testing.assert_allclose(sample[key].numpy(), expected, atol=1e-6, rtol=0)
        batch = collate_state_bank_observations([record], dataset)
        image_ranges = {}
        for key in (record.observation.global_rgb_key, record.observation.wrist_rgb_key):
            image = batch[key]
            if image.shape != (1, 3, 256, 256) or not torch.isfinite(image).all():
                raise ValueError(f"Invalid RGB input: {key}")
            low, high = float(image.min()), float(image.max())
            if not 0 <= low < high <= 1:
                raise ValueError(f"Blank or out-of-range RGB input: {key}")
            image_ranges[key] = [low, high]
        processed = pre(batch)
        # State-keyed CPU RNG is independent of frame processing order.
        seed = int.from_bytes(hashlib.sha256(
            f"{checkpoint_hash}:{record.state_id}".encode()).digest()[:8], "little") % (2**63 - 1)

        def forward():
            policy.reset()
            return post(policy.select_action(processed))[0]

        capture = ActivationCaptureHook()
        handle = layer.register_forward_hook(capture)
        try:
            with torch.random.fork_rng(devices=[]), torch.inference_mode():
                torch.manual_seed(seed)
                action = forward().detach().cpu().numpy()
        finally:
            handle.remove()
        activation = capture.activations[-1]
        if activation.ndim != 3 or activation.shape[0] != 1 or not torch.isfinite(activation).all():
            raise ValueError("Expected finite [1, tokens, features] activation")
        pooled = activation.float().mean(dim=(0, 1)).numpy()
        replay = paired_action_effects(forward, layer, {}, seed=seed, reference=pooled)
        np.testing.assert_allclose(action, replay["original_action"], atol=1e-5, rtol=0)
        values.append(pooled)
        rows.append({"state_id": record.state_id, "frame_index": record.frame_index,
                     "dataset_index": record.observation.dataset_index, "seed": seed,
                     "activation_shape": list(activation.shape), "image_ranges": image_ranges,
                     "seconds_including_three_forwards": time.perf_counter() - started_frame,
                     "replay": replay})
        print(f"Verified real frame {record.frame_index}: {tuple(activation.shape)}, zero replay passed", flush=True)

    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "smoke_activations.npz",
                        state_ids=np.array([r.state_id for r in selected]), values=np.stack(values))
    report = {
        "schema": "smolvla_real_observation_smoke_v1", "status": "passed",
        "dataset": {"repo_id": "lerobot/libero", "revision": revision, "episode": episode,
                    "root": str(dataset_root.resolve()), "episode_frames": len(dataset)},
        "checkpoint": str(checkpoint.resolve()), "checkpoint_tree_sha256": checkpoint_hash,
        "metadata_tree_sha256": _tree_sha256(metadata),
        "state_bank_sha256": file_hash(bank / "manifest.json"), "upstream": source,
        "implementation_sha256": {p.name: file_hash(p) for p in
                                  (Path(__file__), Path(__file__).with_name("latents.py"),
                                   Path(__file__).with_name("predictive_states.py"))},
        "data_sha256": {str(p.relative_to(dataset_root)): file_hash(p)
                        for p in sorted(dataset_root.rglob("*")) if p.is_file()
                        and ".cache" not in p.relative_to(dataset_root).parts},
        "runtime": _runtime_provenance(policy, batch_size=1), "setup_seconds": setup_seconds,
        "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "tap": f"action_atlas/expert/{len(layers)-1}/mlp/output",
        "pooling": "all_action_tokens_mean_at_last_denoising_call",
        "action_coordinates": "checkpoint_postprocessed_policy_output_before_environment_clipping_or_scaling",
        "action_distance_units": "policy_action_coordinates_not_meters_or_radians",
        "states": len(rows), "partition": "train", "rows": rows,
        "complete_state_bank_cache": False, "closed_loop": "not_run", "rl": "frozen",
        "policy_capability": "not_established_by_offline_smoke",
        "scope": "real_observations_and_zero_replay_only_not_predictive_or_causal_evidence",
    }
    write_json_atomic(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("outputs/representation_study/libero_smolvla/state_bank"))
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/datasets/libero"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/pretrained/smolvla_libero"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/pretrained/SmolVLM2-500M-Instruct"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=4)
    args = parser.parse_args()
    report = run(args.bank, args.dataset_root, args.checkpoint, args.metadata, args.output, args.max_states)
    print(json.dumps({"states": report["states"], "status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
