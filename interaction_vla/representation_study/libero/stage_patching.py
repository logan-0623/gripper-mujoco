"""Q0b: at which flow stage is SmolVLA's grasp/release decision committed?

Pairs a recipient state with a matched donor whose natural gripper decision is
the opposite, then replaces one denoising stage of the recipient with the
donor's value and integrates the rest of the flow under the recipient's own
observation. Three patch sites are swept over every stage k:

* ``x_t``: the flow state itself. The integrator owns ``x_t``, so the stage-k
  velocity is rewritten as ``(x_donor + dt * v(x_donor) - x_current) / dt``;
  the integrator then lands exactly on the donor's next point.
* ``expert_middle`` / ``expert_late``: the per-token output of that expert
  layer's MLP sublayer (the same module the natural trace records). The
  residual stream and attention outputs are left untouched.

Donor values come from an existing natural trace of the same checkpoint, so
noise, solver stages, and tokens are paired exactly. Same-decision donors are a
disruption control: a patch that flips decisions for opposite donors but not for
same-decision donors carries decision information rather than generic damage.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .flow_diff import load_trace
from .flow_trace import (bind_policy_images, file_hash, load_frozen_policy,
                         _replace_tensor_output, _tensor_output)
from .latents import _tree_sha256, collate_state_bank_observations
from .state_bank import load_state_bank

SCHEMA = "smolvla_flow_stage_patching_v1"
SITES = ("x_t", "expert_middle", "expert_late")
GRIPPER = 6


def decisions(actions: np.ndarray, executed: int) -> np.ndarray:
    """Mean executed gripper command per state/repeat; LIBERO uses -1 open, +1 close."""
    if actions.ndim != 4 or actions.shape[-1] <= GRIPPER or not 0 < executed <= actions.shape[2]:
        raise ValueError("expected [state, repeat, chunk, action] with a gripper dimension")
    return actions[:, :, :executed, GRIPPER].mean(axis=2)


def match_pairs(rows, gripper: np.ndarray, *, margin: float, max_pairs: int, seed: int):
    """Match each confident recipient to the nearest confident donor in the same task.

    Distance uses standardized robot state and frame index only; labels and
    patched outcomes are never used. Returns opposite-decision and
    same-decision pairs with disjoint donor choices per recipient.
    """
    mean = gripper.mean(axis=1)
    consistent = np.all(np.sign(gripper) == np.sign(mean)[:, None], axis=1)
    confident = consistent & (np.abs(mean) >= margin)
    features = np.column_stack(([r.frame_index for r in rows],
                                np.asarray([r.observation.robot_state for r in rows], dtype=float)))
    scale = features.std(axis=0)
    features = (features - features.mean(axis=0)) / np.where(scale < 1e-8, 1.0, scale)
    order = np.random.default_rng(seed).permutation(len(rows))
    opposite, same = [], []
    for recipient in order:
        if not confident[recipient]:
            continue
        task = (rows[recipient].suite, rows[recipient].task_id)
        episode = rows[recipient].source_episode_id
        candidates = [j for j in range(len(rows)) if j != recipient and confident[j]
                      and (rows[j].suite, rows[j].task_id) == task
                      and rows[j].source_episode_id != episode]
        distance = {j: float(np.linalg.norm(features[recipient] - features[j])) for j in candidates}
        for target, sign in ((opposite, -1.0), (same, 1.0)):
            pool = [j for j in candidates if np.sign(mean[j]) == sign * np.sign(mean[recipient])]
            if pool and len(target) < max_pairs:
                donor = min(pool, key=distance.get)
                target.append((int(recipient), int(donor), distance[donor]))
    return opposite, same


def expert_mlps(policy) -> list:
    """Same modules as the natural trace (Action Atlas SmolVLAAdapter 'expert' group)."""
    return [layer.mlp for layer in policy.model.vlm_with_expert.lm_expert.layers]


def patched_actions(policy, processed, noise: torch.Tensor, *, site: str | None, stage: int,
                    donor: torch.Tensor | None) -> torch.Tensor:
    """Run one action chunk, replacing ``site`` at denoising call ``stage`` with ``donor``."""
    flow = policy.model
    steps = int(policy.config.num_steps)
    dt = -1.0 / steps
    original = flow.denoise_step
    calls, handles = [0], []

    def denoise(*args, **kwargs):
        index = calls[0]
        calls[0] += 1
        if site != "x_t" or index != stage:
            return original(*args, **kwargs)
        x_current = kwargs.get("x_t", args[2] if len(args) > 2 else None)
        if x_current is None:
            raise RuntimeError("SmolVLA denoise_step signature changed")
        target = donor.to(device=x_current.device, dtype=x_current.dtype)
        if "x_t" in kwargs:
            kwargs = {**kwargs, "x_t": target}
        else:
            args = (*args[:2], target, *args[3:])
        velocity = original(*args, **kwargs)
        return (target + dt * velocity - x_current) / dt

    if site in {"expert_middle", "expert_late"}:
        layers = expert_mlps(policy)
        layer = layers[len(layers) // 2] if site == "expert_middle" else layers[-1]
        seen = [0]

        def hook(_module, _inputs, output):
            index = seen[0]
            seen[0] += 1
            if index != stage:
                return output
            tensor = _tensor_output(output)
            replacement = donor.to(device=tensor.device, dtype=tensor.dtype)
            if replacement.shape != tensor.shape:
                raise ValueError(f"donor {tuple(replacement.shape)} != live {tuple(tensor.shape)}")
            return _replace_tensor_output(output, replacement)

        handles.append(layer.register_forward_hook(hook))
    flow.denoise_step = denoise
    try:
        policy.reset()
        with torch.inference_mode():
            actions = policy.predict_action_chunk(processed, noise=noise)
    finally:
        flow.denoise_step = original
        for handle in handles:
            handle.remove()
    if calls[0] != steps:
        raise RuntimeError(f"expected {steps} denoise calls, saw {calls[0]}")
    return actions


def run(bank: Path, dataset_root: Path, checkpoint: Path, contract: Path, metadata: Path,
        trace: Path, output: Path, *, device: str, batch_size: int, executed: int,
        margin: float, max_pairs: int, repeats: Sequence[int], stages: Sequence[int],
        sites: Sequence[str], seed: int = 0):
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("Set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    ids, arrays, binding, manifest = load_trace(trace)
    if binding.get("query_mode") != "natural_integration" or binding.get("flow_edit") is not None:
        raise ValueError("donors must come from an unedited natural trace")
    checkpoint_hash = _tree_sha256(checkpoint)
    if checkpoint_hash != binding["checkpoint_tree_sha256"]:
        raise ValueError("trace was recorded with a different checkpoint")
    if any(r < 0 or r >= arrays["epsilon"].shape[1] for r in repeats):
        raise ValueError("requested noise repeat is absent from the trace")
    steps = int(binding["num_steps"])
    if not stages or min(stages) < 0 or max(stages) >= steps or set(sites) - set(SITES):
        raise ValueError("invalid stages or sites")
    records, bank_manifest, _, _ = load_state_bank(bank)
    if not bank_manifest.get("audit_passed") or binding["state_bank_sha256"] != file_hash(bank / "manifest.json"):
        raise ValueError("StateBank audit or binding mismatch")
    by_id = {r.state_id: r for r in records}
    rows = [by_id[i] for i in ids.tolist()]
    natural = decisions(arrays["action_postprocessed"], executed)
    opposite, same = match_pairs(rows, natural, margin=margin, max_pairs=max_pairs, seed=seed)
    if not opposite:
        raise ValueError("no confident opposite-decision pairs; lower --margin or use more states")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("lerobot/libero", root=dataset_root, revision=binding["dataset_revision"],
                             episodes=sorted({r.lerobot_episode_index for r in rows}),
                             video_backend="torchcodec")
    policy, pre, post = load_frozen_policy(checkpoint, contract, metadata, device)
    layers = expert_mlps(policy)
    if (len(layers), len(layers) // 2, len(layers) - 1) != (
            binding["expert_layer_count"], binding["expert_middle_index"], binding["expert_late_index"]):
        raise ValueError("expert layer indexing differs from the trace")
    report_binding = {"schema": SCHEMA, "trace": str(trace.resolve()),
                      "trace_binding_sha256": manifest["binding_sha256"],
                      "checkpoint_tree_sha256": checkpoint_hash, "executed_steps": executed,
                      "decision_margin": margin, "repeats": list(repeats), "stages": list(stages),
                      "sites": list(sites), "pair_seed": seed, "source_sha256": file_hash(Path(__file__))}
    output.mkdir(parents=True)
    write_json_atomic(output / "binding.json", report_binding)

    def postprocess(raw):
        dim = raw.shape[-1]
        return post(raw.reshape(-1, dim)).reshape(raw.shape).detach().cpu().float().numpy()

    results = {}
    for kind, pairs in (("opposite", opposite), ("same", same)):
        if not pairs:
            continue
        recipients = np.asarray([p[0] for p in pairs])
        donors = np.asarray([p[1] for p in pairs])
        out = {"natural": np.zeros((len(pairs), len(repeats), executed), np.float32),
               "patched": np.zeros((len(sites), len(stages), len(pairs), len(repeats), executed), np.float32),
               "noop_max_abs": np.zeros((len(pairs), len(repeats)), np.float32)}
        for start in range(0, len(pairs), batch_size):
            part = slice(start, start + batch_size)
            batch_rows = [rows[i] for i in recipients[part]]
            batch, _ = bind_policy_images(collate_state_bank_observations(batch_rows, dataset),
                                          policy.config.input_features)
            processed = pre(batch)
            for r_index, repeat in enumerate(repeats):
                noise = torch.from_numpy(arrays["epsilon"][recipients[part], repeat]).to(device)
                live = postprocess(patched_actions(policy, processed, noise, site=None, stage=-1, donor=None))
                stored = arrays["action_postprocessed"][recipients[part], repeat]
                out["noop_max_abs"][part, r_index] = np.abs(live - stored).max(axis=(1, 2))
                out["natural"][part, r_index] = live[:, :executed, GRIPPER]
                for s_index, site in enumerate(sites):
                    key = "x_sigma" if site == "x_t" else site
                    for k_index, stage in enumerate(stages):
                        donor = torch.from_numpy(arrays[key][donors[part], repeat, stage])
                        actions = postprocess(patched_actions(policy, processed, noise, site=site,
                                                              stage=stage, donor=donor))
                        out["patched"][s_index, k_index, part, r_index] = actions[:, :executed, GRIPPER]
            print(json.dumps({"kind": kind, "pairs_done": min(start + batch_size, len(pairs)),
                              "pairs": len(pairs)}), flush=True)
        donor_gripper = natural[donors][:, list(repeats)]
        results[kind] = {**out, "recipients": recipients, "donors": donors,
                         "donor_gripper": donor_gripper,
                         "distance": np.asarray([p[2] for p in pairs])}
    payload = {f"{kind}_{name}": value for kind, values in results.items() for name, value in values.items()}
    buffer = io.BytesIO()
    np.savez(buffer, state_ids=ids, **payload)
    write_bytes_atomic(output / "patching.npz", buffer.getvalue())
    report = summarize(output, rows=rows)
    return report


def flip_metrics(natural, patched, donor_gripper):
    """Per-pair decision flip and fractional shift toward the donor.

    natural, donor_gripper: [pair, repeat]; patched: [..., pair, repeat].
    """
    shift = (patched - natural) / np.where(np.abs(donor_gripper - natural) < 1e-6, np.nan,
                                           donor_gripper - natural)
    flip = (np.sign(patched) == np.sign(donor_gripper)) & (np.sign(natural) != np.sign(donor_gripper))
    return flip.mean(axis=-1), np.nanmean(shift, axis=-1)


def cluster_ci(values, clusters, *, draws: int = 2000, seed: int = 0):
    """Percentile CI of the mean, resampling recipient source episodes."""
    values = np.asarray(values, dtype=float)
    labels = [str(cluster) for cluster in clusters]
    keys = sorted(set(labels))
    groups = [np.flatnonzero([label == key for label in labels]) for key in keys]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        pick = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        means.append(np.nanmean(values[pick]))
    low, high = np.nanpercentile(means, [2.5, 97.5])
    return {"mean": float(np.nanmean(values)), "ci_low": float(low), "ci_high": float(high),
            "clusters": len(keys), "pairs": int(len(values))}


def summarize(output: Path, *, rows) -> dict:
    binding = json.loads((output / "binding.json").read_text())
    data = np.load(output / "patching.npz", allow_pickle=False)
    report = {"schema": SCHEMA + "_report", "binding_sha256": file_hash(output / "binding.json"),
              "exploration_only": True, "curves": {}}
    for kind in ("opposite", "same"):
        if f"{kind}_patched" not in data.files:
            continue
        natural = data[f"{kind}_natural"].mean(axis=-1)
        patched = data[f"{kind}_patched"].mean(axis=-1)
        donor = data[f"{kind}_donor_gripper"]
        recipients = data[f"{kind}_recipients"]
        clusters = [(rows[i].suite, rows[i].task_id, rows[i].source_episode_id) for i in recipients]
        flip, shift = flip_metrics(natural, patched, donor)
        if kind == "same":
            # A same-decision donor cannot "flip"; count any sign change as disruption.
            flip = (np.sign(patched) != np.sign(natural)).mean(axis=-1)
        report["noop_max_abs_error"] = max(report.get("noop_max_abs_error", 0.0),
                                           float(data[f"{kind}_noop_max_abs"].max()))
        report["curves"][kind] = {
            site: [{"stage": stage, "flip_rate": cluster_ci(flip[s, k], clusters),
                    "shift_toward_donor": cluster_ci(shift[s, k], clusters) if kind == "opposite" else None}
                   for k, stage in enumerate(binding["stages"])]
            for s, site in enumerate(binding["sites"])}
    curves = report["curves"].get("opposite", {})
    reliable = {site: [row["stage"] for row in points if row["flip_rate"]["ci_low"] > 0.5]
                for site, points in curves.items()}
    stages = binding["stages"]
    trajectory = reliable.get("x_t", [])
    report["x_t_commitment_stage"] = next(
        (stage for stage in stages if all(later in trajectory for later in stages if later >= stage)), None)
    report["decisive_stages"] = {site: values for site, values in reliable.items() if site != "x_t"}
    same = report["curves"].get("same", {})
    report["same_donor_max_disruption"] = {
        site: max(row["flip_rate"]["mean"] for row in points) for site, points in same.items()}
    report["decision_rules"] = {
        "x_t_commitment_stage": "earliest stage from which every later x_t patch transfers the donor decision "
                                "(flip-rate CI lower bound > 0.5); None if no such stage",
        "decisive_stages": "stages where a single-stage MLP-sublayer patch transfers the donor decision",
        "same_donor_max_disruption": "sign changes caused by same-decision donors; should stay near 0"}
    report["limits"] = ["offline demo states, not closed-loop", "MLP-sublayer patch, not full residual",
                        "matching uses robot state and frame index only"]
    write_json_atomic(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--contract-checkpoint", type=Path)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True, help="natural flow trace of the same checkpoint")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--executed", type=int, default=10, help="executed actions per chunk (n_action_steps)")
    p.add_argument("--margin", type=float, default=0.5, help="minimum |mean gripper| for a confident decision")
    p.add_argument("--max-pairs", type=int, default=64)
    p.add_argument("--repeat", type=int, action="append", help="trace noise repeat(s); default 0")
    p.add_argument("--stage", type=int, action="append", help="default: every solver stage")
    p.add_argument("--site", choices=SITES, action="append")
    a = p.parse_args()
    print(json.dumps(run(a.bank, a.dataset_root, a.checkpoint, a.contract_checkpoint or a.checkpoint,
                         a.metadata, a.trace, a.output, device=a.device, batch_size=a.batch_size,
                         executed=a.executed, margin=a.margin, max_pairs=a.max_pairs,
                         repeats=a.repeat or [0], stages=a.stage or list(range(10)),
                         sites=a.site or list(SITES)), indent=2))


if __name__ == "__main__":
    main()
