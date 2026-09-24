"""Bounded frozen-policy suppression and explicitly annotated matched masking."""
from __future__ import annotations

import argparse
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .checkpoint_readouts import episode, weights
from .flow_diff import load_trace
from .flow_trace import (action_atlas_provenance, bind_policy_images, file_hash,
                         load_flow_edit, load_frozen_policy, query_action_flow_at_points,
                         trace_action_flow)
from .latents import _canonical_sha256, _tree_sha256, collate_state_bank_observations
from .state_bank import load_state_bank


def postprocess_actions(post, actions, device):
    value = torch.from_numpy(actions).to(device)
    return post(value.reshape(-1, value.shape[-1])).reshape(value.shape).detach().cpu().float().numpy()


def mask_batch(batch, state_ids, spec, region, fill):
    """Pixel rectangles on raw, bound policy images; never infer semantic labels."""
    if fill not in {"mean", "blur"}:
        raise ValueError("unknown mask fill")
    result = dict(batch)
    keys = {spec["states"][state]["camera"] for state in state_ids}
    for key in keys:
        if key not in batch or key.endswith("camera3"):
            raise ValueError("mask camera missing or synthetic")
        result[key] = batch[key].clone()
    for index, state in enumerate(state_ids):
        row = spec["states"][state]
        image = batch[row["camera"]][index]
        if image.ndim != 3 or not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError("mask expects finite CHW float images")
        h, w = image.shape[-2:]
        boxes = row["regions"]
        if len(boxes) < 2 or region not in boxes:
            raise ValueError("at least two matched regions required")
        areas = []
        for box in boxes.values():
            if len(box) != 4 or any(type(v) is not int for v in box):
                raise ValueError("boxes must be integer [y0,x0,y1,x1] pixels")
            y0, x0, y1, x1 = box
            if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
                raise ValueError("mask rectangle outside image")
            areas.append((y1-y0)*(x1-x0))
        if len(set(areas)) != 1:
            raise ValueError("control mask areas differ")
        y0, x0, y1, x1 = boxes[region]
        replacement = (image.mean(dim=(-2, -1), keepdim=True).expand_as(image) if fill == "mean"
                       else F.avg_pool2d(F.pad(image[None], (5,5,5,5), mode="replicate"), 11, stride=1)[0])
        result[row["camera"]][index, :, y0:y1, x0:x1] = replacement[:, y0:y1, x0:x1]
    return result


def action_metrics(delta, prefix):
    if delta.ndim != 3 or delta.shape[-1] != 7 or not 1 <= prefix <= delta.shape[1]:
        raise ValueError("expected batch x plan x 7 actions and a valid deployed prefix")
    out = {}
    for window, length in (("deployed", prefix), ("plan", delta.shape[1])):
        for name, columns in (("full", slice(None)), ("translation", slice(0,3)),
                              ("rotation", slice(3,6)), ("gripper", slice(6,7))):
            out[f"{window}_{name}_rms"] = np.sqrt(np.mean(np.square(delta[:, :length, columns], dtype=float), axis=(1,2)))
    out["first_action_delta"] = delta[:, 0]
    out["deployed_gripper_signed_mean"] = delta[:, :prefix, 6].mean(axis=1)
    return out


def run(bank, dataset_root, checkpoint, contract, metadata, train_trace, reference_trace,
        candidates, output, *, max_states=32, repeats=3, dose=.5, mask_spec=None, batch_size=4,
        target_id="formation_0", control_ids=("low_change_0", "matched_random_0", "matched_random_1")):
    if output.exists():
        raise FileExistsError(output)
    if not 0 < dose <= 1 or max_states < 4 or repeats < 1 or batch_size < 1:
        raise ValueError("invalid bounded experiment settings")
    ids, reference, binding, manifest = load_trace(reference_trace)
    if binding.get("partition") != "validation" or binding.get("split_group") != "episode" or binding.get("query_mode") != "natural_integration" or binding.get("flow_edit") is not None:
        raise ValueError("requires unedited episode-validation natural trace")
    for key, actual in (("checkpoint_tree_sha256", _tree_sha256(checkpoint)),
                        ("contract_tree_sha256", _tree_sha256(contract)),
                        ("metadata_tree_sha256", _tree_sha256(metadata)),
                        ("state_bank_sha256", file_hash(bank / "manifest.json"))):
        if binding[key] != actual:
            raise ValueError(f"reference identity mismatch: {key}")
    if repeats > reference["epsilon"].shape[1] or max_states > len(ids):
        raise ValueError("requested repeats/states exceed reference")
    dataset_hash = _canonical_sha256({str(p.relative_to(dataset_root)):file_hash(p)
                                     for pattern in ("meta/**/*.json","meta/**/*.parquet","data/**/*.parquet","videos/**/*.mp4")
                                     for p in sorted(dataset_root.glob(pattern))})
    if dataset_hash != binding["dataset_scientific_sha256"]:
        raise ValueError("dataset contents differ from reference")
    records, bank_manifest, _, split = load_state_bank(bank)
    if not bank_manifest.get("audit_passed"):
        raise ValueError("unaudited StateBank")
    by_id = {r.state_id:r for r in records}
    selected = [by_id[i] for i in ids[:max_states]]
    if any(split.assignments[r.state_id] != "validation" for r in selected):
        raise ValueError("validation partition mismatch")
    train_ids, train, train_binding, train_manifest = load_trace(train_trace)
    if train_binding.get("partition") != "train" or train_binding.get("split_group") != "episode" or train_binding.get("checkpoint_tree_sha256") != binding["checkpoint_tree_sha256"]:
        raise ValueError("training center cache differs from checkpoint/partition")
    if train_binding.get("query_mode") != "natural_integration" or train_binding.get("flow_edit") is not None:
        raise ValueError("training centers require unedited natural traces")
    for key in ("state_bank_sha256","dataset_scientific_sha256","contract_tree_sha256","metadata_tree_sha256","image_binding","num_steps"):
        if train_binding.get(key) != binding.get(key):
            raise ValueError(f"train/validation contract mismatch: {key}")
    if any(split.assignments.get(i) != "train" for i in train_ids):
        raise ValueError("training partition mismatch")
    if {episode(by_id[i]) for i in train_ids} & {episode(r) for r in selected}:
        raise ValueError("episode leakage")
    centers = {tap: train[tap].mean(axis=(0,1,2,3)) for tap in ("expert_middle", "expert_late")}
    del train
    edits = {}
    for candidate_path in candidates:
        artifact = json.loads(candidate_path.read_text())
        basis_path = candidate_path.with_name("shared_basis.npz")
        if artifact.get("shared_basis_sha256") != file_hash(basis_path):
            raise ValueError("candidate basis hash mismatch")
        with np.load(basis_path, allow_pickle=False) as basis:
            if set(basis["state_ids"].tolist()) != set(train_ids.tolist()):
                raise ValueError("candidate discovery samples differ from training cache")
        candidate_ids = (target_id, *control_ids)
        for candidate in candidate_ids:
            edit, _ = load_flow_edit(candidate_path, candidate, dose, tuple(range(binding["num_steps"])),
                                     mode="suppress" if candidate == target_id else "matched_suppress",
                                     match_candidate_id=target_id)
            for direction in (edit.direction, edit.coefficient_direction):
                if not torch.isfinite(direction).all() or not torch.isclose(direction.norm(), torch.tensor(1.), atol=1e-4):
                    raise ValueError("candidate/control must have unit norm")
            edit = replace(edit, center=torch.tensor(centers[edit.tap]))
            edits[f"{edit.tap}:{candidate}"] = edit
    spec = json.loads(mask_spec.read_text()) if mask_spec else None
    if {e.tap for e in edits.values()} != {"expert_middle", "expert_late"} or len(edits) != 8:
        raise ValueError("requires exactly one candidate artifact per middle/late tap")
    mask_cases = []
    if spec:
        if not spec.get("annotation_source") or spec.get("reference_binding_sha256") != manifest["binding_sha256"]:
            raise ValueError("mask provenance/reference mismatch")
        if any(i not in spec["states"] for i in ids[:max_states]):
            raise ValueError("missing state mask annotations")
        regions = sorted(spec["states"][ids[0]]["regions"])
        if any(sorted(spec["states"][i]["regions"]) != regions for i in ids[:max_states]):
            raise ValueError("inconsistent mask arm coverage")
        mask_cases = [(region, fill) for region in regions for fill in ("mean", "blur")]
    action_atlas_provenance()
    from experiments.model_adapters import SmolVLAAdapter
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset("lerobot/libero", root=dataset_root, revision=binding["dataset_revision"],
                             episodes=sorted({r.lerobot_episode_index for r in selected}), video_backend="torchcodec")
    policy, pre, post = load_frozen_policy(checkpoint, contract, metadata, "cuda")
    # Deployment eval explicitly overrides n_action_steps=10; checkpoint stores 50.
    # predict_action_chunk still generates the full plan, irrespective of execution prefix.
    prefix = 10
    if int(policy.config.chunk_size) < prefix:
        raise ValueError("generated plan shorter than the declared deployment prefix")
    adapter = SmolVLAAdapter(); adapter.policy = policy
    layers = adapter.get_layer_groups()["expert"]
    middle, late = layers[len(layers)//2], layers[-1]
    collected = {}
    for start in range(0, len(selected), batch_size):
        stop = min(start + batch_size, len(selected))
        raw, image_binding = bind_policy_images(collate_state_bank_observations(selected[start:stop], dataset), policy.config.input_features)
        if image_binding != binding["image_binding"]:
            raise ValueError("image preprocessing mismatch")
        for repeat in range(repeats):
            noise = torch.from_numpy(reference["epsilon"][start:stop, repeat]).to("cuda")
            processed = pre(raw)
            baseline = trace_action_flow(policy, processed, noise, middle, late)
            baseline_action = postprocess_actions(post, baseline["action_normalized"], "cuda")
            np.testing.assert_allclose(baseline_action, reference["action_postprocessed"][start:stop,repeat], atol=2e-5, rtol=1e-5)
            cases = [("no_op", None, None), *[(k, v, None) for k,v in edits.items()],
                     *[(f"mask:{region}:{fill}", None, (region,fill)) for region,fill in mask_cases]]
            fixed_noop = query_action_flow_at_points(policy, processed, noise,
                                                     baseline["x_sigma"], baseline["sigma"], middle, late)
            for name, edit, mask in cases:
                observed = pre(mask_batch(raw, ids[start:stop].tolist(), spec, *mask)) if mask else processed
                natural = trace_action_flow(policy, observed, noise, middle, late, edit)
                action = postprocess_actions(post, natural["action_normalized"], "cuda")
                fixed = query_action_flow_at_points(policy, observed, noise, baseline["x_sigma"], baseline["sigma"], middle, late, edit)
                metrics = action_metrics(action-baseline_action, prefix)
                fixed_velocity_delta = fixed["velocity"] - (fixed_noop["velocity"] if edit is not None else baseline["velocity"])
                metrics["fixed_velocity_rms_by_stage"] = np.sqrt(np.mean(np.square(fixed_velocity_delta,dtype=float), axis=(2,3)))
                if edit:
                    tap = edit.tap
                    metrics["fixed_hidden_delta_rms_by_stage"] = np.sqrt(np.mean(np.square(fixed[tap]-fixed_noop[tap],dtype=float),axis=(2,3)))
                for tap in ("expert_middle", "expert_late"):
                    target = next(e for e in edits.values() if e.tap == tap and e is not edit)
                    delta = fixed[tap] - (fixed_noop[tap] if edit is not None else baseline[tap])
                    metrics[f"{tap}_projection_delta_by_stage"] = np.einsum("nstd,d->ns",delta,target.coefficient_direction.numpy()) / delta.shape[2]
                for metric,value in metrics.items():
                    if not np.isfinite(value).all():
                        raise ValueError("non-finite intervention result")
                    collected.setdefault(f"{name}/{metric}",[]).append((start,stop,repeat,value))
            print(json.dumps({"states":stop,"total":len(selected),"repeat":repeat+1,"arms":len(cases)}),flush=True)
    arrays = {}
    for key, chunks in collected.items():
        result = np.empty((len(selected), repeats, *chunks[0][3].shape[1:]))
        for start,stop,repeat,value in chunks:
            result[start:stop,repeat] = value
        arrays[key] = result
    # Same-call-path no-op must pass independently of candidate effect magnitude.
    if np.max(arrays["no_op/deployed_full_rms"]) > 2e-5:
        raise ValueError("natural no-op replay failed")
    buffer = io.BytesIO(); np.savez(buffer, state_ids=ids[:max_states], **arrays)
    write_bytes_atomic(output / "effects.npz", buffer.getvalue())
    for path in candidates:
        artifact = json.loads(path.read_text())
        tap = artifact["tap"]
        buffer = io.BytesIO()
        with np.load(path.with_name("shared_basis.npz"), allow_pickle=False) as source:
            basis = {k:source[k] for k in source.files}
        basis["mean"] = centers[tap]
        np.savez(buffer, **basis)
        write_bytes_atomic(output/tap/"shared_basis.npz",buffer.getvalue())
        write_json_atomic(output/tap/"candidates.json",{
            **artifact, "center_checkpoint_sha256":binding["checkpoint_tree_sha256"],
            "center_train_binding_sha256":train_manifest["binding_sha256"],
            "source_candidate_sha256":file_hash(path),
            "shared_basis_sha256":file_hash(output/tap/"shared_basis.npz")})
    summary = {}
    for key, value in arrays.items():
        state_mean = value.mean(axis=1)
        summary[key] = {"task_macro": np.einsum("n,n...->...",weights(selected),state_mean).tolist(),
                        "by_task": {str(task): np.mean(state_mean[[r.task_id==task for r in selected]],axis=0).tolist() for task in sorted({r.task_id for r in selected})}}
    report = {"schema":"smolvla_candidate_controls_v1", "complete":True,"exploratory":True,
              "checkpoint_sha256":binding["checkpoint_tree_sha256"], "reference_binding_sha256":manifest["binding_sha256"],
              "train_binding_sha256":train_manifest["binding_sha256"],"candidate_hashes":{str(p):file_hash(p) for p in candidates},
              "mask_spec_sha256":file_hash(mask_spec) if mask_spec else None,
              "mask_annotation_source":spec["annotation_source"] if spec else None,
              "states":len(selected),"episodes":len({episode(r) for r in selected}),"noise_repeats":repeats,
              "dose":dose,"deployed_prefix":prefix,"stages":list(range(binding["num_steps"])),
              "checkpoint_n_action_steps":int(policy.config.n_action_steps),
              "deployed_prefix_source":"paired rollout command --policy.n_action_steps=10; not checkpoint default",
              "center":"own-checkpoint training mean; shared frozen raw-space candidate directions",
              "target_id":target_id,"control_ids":list(control_ids),
              "controls":"target coefficient redirected into unit random/low-change directions; same nominal dose",
              "effects_sha256":file_hash(output/"effects.npz"),"summary":summary,
              "limits":["four validation episodes; descriptive only","raw shared directions do not guarantee semantic alignment",
                        "fixed queries now report candidate-minus-no-op local effects; natural integration measures action-plan effects",
                        "no new closed-loop success measurement", "fixed no-op vs natural numerical floor reported, not subtracted"]}
    write_json_atomic(output/"report.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("bank","dataset-root","checkpoint","contract","metadata","train-trace","reference-trace","output"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--candidates",type=Path,action="append",required=True)
    p.add_argument("--target-id",default="formation_0")
    p.add_argument("--control-id",action="append",default=None)
    p.add_argument("--mask-spec",type=Path)
    p.add_argument("--max-states",type=int,default=32)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--dose",type=float,default=.5)
    p.add_argument("--batch-size",type=int,default=4)
    a=p.parse_args(); values=vars(a); values["control_ids"]=tuple(values.pop("control_id") or ("low_change_0","matched_random_0","matched_random_1")); result=run(**values)
    print(f"ALL_DONE states={result['states']}",flush=True)


if __name__=="__main__":
    main()
