"""Bind exploratory Contact readouts, offline action effects, and behavior by checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..state_bank.io import write_json_atomic
from .flow_trace import file_hash


TARGETS = ("contact", "stable_grasp", "gripper_target_distance")
COMPONENTS = ("full", "translation", "rotation", "gripper")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(lineage: Path, timeline: Path, readouts: Path, effects: dict[str, Path], output: Path,
        *, aligned_readouts: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    lineage_data, timeline_data, readout_data = map(_read, (lineage, timeline, readouts))
    checkpoints = lineage_data["checkpoints"]
    names = [f'{row["step"]:06d}' for row in checkpoints]
    if (lineage_data.get("kind") != "immutable_lineage"
            or names != ["005000", "010000", "015000", "020000", "025000"]
            or set(effects) != set(names)):
        raise ValueError("effects must cover the ordered 5k–25k training lineage")
    if (timeline_data.get("kind") != "capability_timeline" or not timeline_data.get("complete")
            or timeline_data.get("lineage_sha256") != file_hash(lineage)):
        raise ValueError("capability timeline is incomplete or from another lineage")
    if (readout_data.get("schema") != "smolvla_checkpoint_readouts_v1"
            or set(readout_data["inputs"]) != set(names)
            or readout_data["comparison_contract"].get("query_mode") != "natural_integration"):
        raise ValueError("readouts do not cover natural traces for this lineage")
    aligned_data = _read(aligned_readouts) if aligned_readouts else None
    if aligned_data and (aligned_data.get("schema") != readout_data["schema"]
                         or aligned_data.get("inputs") != readout_data["inputs"]
                         or aligned_data.get("comparison_contract") != readout_data["comparison_contract"]
                         or aligned_data.get("alpha") != readout_data.get("alpha")
                         or aligned_data.get("transfer_moment_alignment") is not True):
        raise ValueError("moment-aligned readouts use a different comparison contract")

    behavior = {(f'{row["step"]:06d}', int(row["task"])): row for row in timeline_data["rows"]}
    tasks = sorted({task for _, task in behavior})
    if (len(behavior) != len(timeline_data["rows"]) or len(behavior) != len(names) * len(tasks)
            or tasks != sorted(int(task) for task in timeline_data.get("paired_initial_state_ids", {}))
            or any((name, task) not in behavior for name in names for task in tasks)):
        raise ValueError("timeline task/checkpoint cells are incomplete")
    for task in tasks:
        identities = [behavior[name, task]["initial_state_ids"] for name in names]
        if any(ids != identities[0] or len(ids) != len(set(ids))
               or len(ids) != behavior[name, task]["episodes"]
               for ids, name in zip(identities, names, strict=True)):
            raise ValueError("behavior initial states are not paired across checkpoints")

    lookup, task_lookup = {}, {}
    for block in readout_data["readouts"]:
        if block["tap"] != "expert_late" or block["target"] not in TARGETS:
            continue
        if block["status"] != "complete":
            raise ValueError("required expert-late readout is not estimable")
        for row in block["rows"]:
            if row["task"] == "macro":
                lookup[block["target"], block["flow_stage"], row["source"], row["destination"]] = row
            else:
                task_lookup[block["target"], block["flow_stage"], row["source"],
                            row["destination"], int(row["task"][1])] = row
    stages = sorted({stage for _, stage, _, _ in lookup})
    if stages != list(range(10)):
        raise ValueError("readouts must cover all ten flow stages")
    aligned_lookup = {}
    if aligned_data:
        for block in aligned_data["readouts"]:
            if block["tap"] == "expert_late" and block["target"] in TARGETS:
                if block["status"] != "complete":
                    raise ValueError("required moment-aligned readout is not estimable")
                for row in block["rows"]:
                    if row["task"] == "macro":
                        aligned_lookup[block["target"], block["flow_stage"],
                                       row["source"], row["destination"]] = row

    reference_ids = None
    contract = None
    rows = []
    for checkpoint, name in zip(checkpoints, names, strict=True):
        provenance = readout_data["inputs"][name]
        if any(provenance[part]["checkpoint_tree_sha256"] != checkpoint["checkpoint_sha256"]
               for part in ("train", "validation")):
            raise ValueError(f"readout checkpoint differs from lineage: {name}")
        effect_path = effects[name]
        effect = _read(effect_path)
        if (effect.get("schema") != "smolvla_candidate_controls_v1" or not effect.get("complete")
                or effect["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
                or effect["train_binding_sha256"] != provenance["train"]["binding_sha256"]
                or effect["reference_binding_sha256"] != provenance["validation"]["binding_sha256"]):
            raise ValueError(f"offline effect and paired readout traces differ: {name}")
        candidate_taps = set()
        for candidate_path, candidate_hash in effect.get("candidate_hashes", {}).items():
            path = Path(candidate_path)
            candidate = _read(path)
            if (file_hash(path) != candidate_hash
                    or candidate.get("train_trace_binding_sha256") != provenance["train"]["binding_sha256"]):
                raise ValueError(f"candidate was not fitted on this checkpoint's train trace: {name}")
            candidate_taps.add(candidate["tap"])
        if candidate_taps != {"expert_middle", "expert_late"}:
            raise ValueError(f"candidate tap coverage differs: {name}")
        current_contract = {key: effect[key] for key in ("states", "episodes", "noise_repeats", "dose", "stages", "deployed_prefix", "target_id", "control_ids")}
        if (current_contract["stages"] != stages or current_contract["deployed_prefix"] != 10
                or current_contract["target_id"] != "contact_0"
                or current_contract["control_ids"] != ["low_change_0", "matched_random_0", "matched_random_1"]):
            raise ValueError(f"offline intervention protocol differs: {name}")
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise ValueError(f"offline intervention budgets differ: {name}")
        array_path = effect_path.with_name("effects.npz")
        if file_hash(array_path) != effect["effects_sha256"]:
            raise ValueError(f"offline effects array hash differs: {name}")
        with np.load(array_path, allow_pickle=False) as arrays:
            ids = arrays["state_ids"].tolist()
            paired_effects = {}
            for component in COMPONENTS:
                metric = f"deployed_{component}_rms"
                target_values = arrays[f"expert_late:contact_0/{metric}"]
                random_values = np.mean([arrays[f"expert_late:matched_random_{index}/{metric}"]
                                         for index in range(2)], axis=0)
                if (target_values.shape != (effect["states"], effect["noise_repeats"])
                        or random_values.shape != target_values.shape
                        or not np.isfinite(target_values).all()
                        or not np.isfinite(random_values).all()):
                    raise ValueError(f"invalid paired offline action arrays: {name}")
                paired_effects[component] = (target_values - random_values).mean(axis=1)
        for partition in ("train", "validation"):
            path = Path(provenance[partition]["path"]) / "binding.json"
            if file_hash(path) != provenance[partition]["binding_sha256"]:
                raise ValueError(f"readout trace binding changed: {name}")
        binding_path = Path(provenance["validation"]["path"]) / "binding.json"
        binding = _read(binding_path)
        if binding["noise_repeats"] != effect["noise_repeats"]:
            raise ValueError(f"noise repeats differ from readout trace: {name}")
        if ids != binding["state_ids"][:effect["states"]] or (reference_ids is not None and ids != reference_ids):
            raise ValueError(f"offline effects use different StateBank states: {name}")
        reference_ids = ids
        summary = effect["summary"]
        action = {}
        for component in COMPONENTS:
            metric = f"deployed_{component}_rms"
            value = lambda arm: float(summary[f"expert_late:{arm}/{metric}"]["task_macro"])
            target = value("contact_0")
            random = [value(f"matched_random_{index}") for index in range(2)]
            by_task = {}
            for task in tasks:
                measured = lambda arm: float(summary[f"expert_late:{arm}/{metric}"]["by_task"][str(task)])
                by_task[str(task)] = measured("contact_0") - float(np.mean([
                    measured(f"matched_random_{index}") for index in range(2)]))
            action[component] = {"target_rms": target, "matched_random_rms": random,
                                 "target_minus_random_mean_rms": target - float(np.mean(random)),
                                 "target_minus_random_mean_rms_by_task": by_task,
                                 "low_change_rms": value("low_change_0"),
                                 "paired_state_median_rms": float(np.median(paired_effects[component])),
                                 "paired_state_positive_fraction": float(np.mean(paired_effects[component] > 0)),
                                 "largest_absolute_paired_state": {
                                     "state_id": ids[int(np.argmax(np.abs(paired_effects[component])))],
                                     "difference_rms": float(paired_effects[component][
                                         np.argmax(np.abs(paired_effects[component]))])}}
            if not np.isclose(action[component]["target_minus_random_mean_rms"],
                              paired_effects[component].mean(), atol=1e-9):
                raise ValueError(f"offline aggregate and paired arrays differ: {name}")
        if float(summary["no_op/deployed_full_rms"]["task_macro"]) > 2e-5:
            raise ValueError(f"no-op replay is not self-consistent: {name}")
        readable = {}
        for target in TARGETS:
            readable[target] = {
                "self_mse_gain_by_stage": [float(lookup[target, stage, name, name]["mse_gain"]) for stage in stages],
                "from_5k_mse_gain_by_stage": [float(lookup[target, stage, names[0], name]["mse_gain"]) for stage in stages],
                "from_25k_mse_gain_by_stage": [float(lookup[target, stage, names[-1], name]["mse_gain"]) for stage in stages],
                "self_mse_gain_by_task_at_stage_9": {
                    str(task): float(task_lookup[target, 9, name, name, task]["mse_gain"])
                    for task in tasks},
            }
            if aligned_data:
                readable[target]["moment_aligned_from_5k_mse_gain_by_stage"] = [
                    float(aligned_lookup[target, stage, names[0], name]["mse_gain"]) for stage in stages]
                readable[target]["moment_aligned_from_25k_mse_gain_by_stage"] = [
                    float(aligned_lookup[target, stage, names[-1], name]["mse_gain"]) for stage in stages]
        rows.append({"step": checkpoint["step"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                     "readability": readable, "offline_action_response": action,
                     "behavior_by_task": {str(task): {key: behavior[name, task][key] / behavior[name, task]["episodes"]
                                                      for key in ("contact", "stable_grasp", "supported_lift", "success")}
                                          for task in tasks},
                     "effect_report": str(effect_path.resolve())})
    result = {"schema": "smolvla_minimal_longitudinal_contact_v1", "complete": True,
              "analysis_role": "exploratory", "lineage_sha256": file_hash(lineage),
              "timeline_sha256": file_hash(timeline), "readouts_sha256": file_hash(readouts),
              "moment_aligned_readouts_sha256": file_hash(aligned_readouts) if aligned_readouts else None,
              "offline_contract": contract, "offline_state_ids": reference_ids,
              "readability_metric": "validation training-mean MSE minus Ridge MSE; positive is better, no CI",
              "offline_metric": "postprocessed first ten planned actions vs no-op; RMS target minus mean of two matched random controls",
              "alignment": "R and U use paired StateBank observations/noise; S shares checkpoint/task but uses different simulator initial states",
              "candidate_rule": "Contact direction refit on each checkpoint train trace; not one aligned cross-checkpoint feature",
              "limits": ["validation contains only four independent demonstration episodes",
                         "offline action response is not closed-loop functional use",
                         "cross-checkpoint changes cannot alone separate reorganization from recruitment",
                         "selected Contact direction and existing behavior data are exploratory"],
              "rows": rows}
    write_json_atomic(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lineage", "timeline", "readouts", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--effect", nargs=2, action="append", required=True,
                        metavar=("STEP", "REPORT"))
    parser.add_argument("--aligned-readouts", type=Path)
    args = parser.parse_args()
    effects = dict(args.effect)
    if len(effects) != len(args.effect):
        raise ValueError("duplicate effect checkpoint")
    result = run(args.lineage, args.timeline, args.readouts,
                 {step: Path(path) for step, path in effects.items()}, args.output,
                 aligned_readouts=args.aligned_readouts)
    print(f"ALL_DONE_LONGITUDINAL rows={len(result['rows'])}", flush=True)


if __name__ == "__main__":
    main()
