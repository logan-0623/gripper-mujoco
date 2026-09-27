import json

import numpy as np
import pytest

from interaction_vla.representation_study.libero.flow_trace import file_hash
from interaction_vla.representation_study.libero.longitudinal_contact_summary import run


def test_summary_binds_readouts_effects_and_behavior(tmp_path):
    names = ["005000", "010000", "015000", "020000", "025000"]
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({"kind": "immutable_lineage", "checkpoints": [
        {"step": int(name), "checkpoint_sha256": f"model-{name}"} for name in names]}))
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps({"kind": "capability_timeline", "complete": True,
        "lineage_sha256": file_hash(lineage), "paired_initial_state_ids": {"0": [20], "1": [20]},
        "rows": [
            {"step": int(name), "task": task, "initial_state_ids": [20], "episodes": 1,
             "contact": 1, "stable_grasp": 1, "supported_lift": 1, "success": 1}
            for name in names for task in (0, 1)]}))
    inputs, effects = {}, {}
    for name in names:
        inputs[name] = {}
        for partition in ("train", "validation"):
            path = tmp_path / name / partition
            path.mkdir(parents=True)
            (path / "binding.json").write_text(json.dumps({"state_ids": [1, 2], "noise_repeats": 3}))
            inputs[name][partition] = {"path": str(path), "binding_sha256": file_hash(path / "binding.json"),
                                       "checkpoint_tree_sha256": f"model-{name}"}
        path = tmp_path / name / "effects"
        path.mkdir()
        arrays = {"state_ids": np.asarray([1, 2])}
        for component in ("full", "translation", "rotation", "gripper"):
            for arm, value in (("contact_0", .2), ("matched_random_0", .1),
                               ("matched_random_1", .1)):
                arrays[f"expert_late:{arm}/deployed_{component}_rms"] = np.full((2, 3), value)
        np.savez(path / "effects.npz", **arrays)
        candidate_hashes = {}
        for tap in ("expert_middle", "expert_late"):
            candidate = tmp_path / name / tap / "candidates.json"
            candidate.parent.mkdir()
            candidate.write_text(json.dumps({"tap": tap,
                "train_trace_binding_sha256": inputs[name]["train"]["binding_sha256"]}))
            candidate_hashes[str(candidate)] = file_hash(candidate)
        summary = {}
        for component in ("full", "translation", "rotation", "gripper"):
            for arm, value in (("contact_0", .2), ("matched_random_0", .1),
                               ("matched_random_1", .1), ("low_change_0", .05)):
                summary[f"expert_late:{arm}/deployed_{component}_rms"] = {
                    "task_macro": value, "by_task": {"0": value, "1": value}}
        summary["no_op/deployed_full_rms"] = {"task_macro": 0.0}
        report = path / "report.json"
        report.write_text(json.dumps({"schema": "smolvla_candidate_controls_v1", "complete": True,
            "checkpoint_sha256": f"model-{name}",
            "train_binding_sha256": inputs[name]["train"]["binding_sha256"],
            "reference_binding_sha256": inputs[name]["validation"]["binding_sha256"],
            "candidate_hashes": candidate_hashes,
            "effects_sha256": file_hash(path / "effects.npz"), "states": 2, "episodes": 2,
            "noise_repeats": 3, "dose": .5, "stages": list(range(10)), "deployed_prefix": 10,
            "target_id": "contact_0",
            "control_ids": ["low_change_0", "matched_random_0", "matched_random_1"],
            "summary": summary}))
        effects[name] = report
    readouts = tmp_path / "readouts.json"
    blocks = []
    for target in ("contact", "stable_grasp", "gripper_target_distance"):
        for stage in range(10):
            blocks.append({"tap": "expert_late", "flow_stage": stage, "target": target,
                           "status": "complete", "rows": [
                               {"source": source, "destination": dest, "task": task,
                                "mse_gain": .1} for source in names for dest in names
                               for task in ("macro", ["libero_spatial", 0], ["libero_spatial", 1])]})
    readouts.write_text(json.dumps({"schema": "smolvla_checkpoint_readouts_v1",
                                    "comparison_contract": {"query_mode": "natural_integration"},
                                    "inputs": inputs, "readouts": blocks}))
    result = run(lineage, timeline, readouts, effects, tmp_path / "summary.json")
    assert len(result["rows"]) == 5
    assert result["rows"][0]["offline_action_response"]["full"]["target_minus_random_mean_rms"] == pytest.approx(.1)
    assert result["rows"][0]["offline_action_response"]["full"]["paired_state_median_rms"] == pytest.approx(.1)
    assert result["rows"][0]["offline_action_response"]["full"]["paired_state_positive_fraction"] == 1.0
    assert result["rows"][0]["readability"]["contact"]["self_mse_gain_by_stage"] == [.1] * 10

    aligned = tmp_path / "aligned.json"
    aligned_data = json.loads(readouts.read_text())
    aligned_data["transfer_moment_alignment"] = True
    aligned.write_text(json.dumps(aligned_data))
    aligned_result = run(lineage, timeline, readouts, effects, tmp_path / "aligned_summary.json",
                         aligned_readouts=aligned)
    assert aligned_result["rows"][0]["readability"]["contact"]["moment_aligned_from_25k_mse_gain_by_stage"] == [.1] * 10

    broken = json.loads(effects[names[0]].read_text())
    broken["reference_binding_sha256"] = "other-trace"
    effects[names[0]].write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="offline effect and paired readout traces differ"):
        run(lineage, timeline, readouts, effects, tmp_path / "invalid.json")
