import json
from pathlib import Path

import numpy as np

from interaction_vla.representation_study.libero import acquisition


def _trace(offset: float):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(6, 2, 3, 4, 8)).astype("float32")
    values[..., 0] += offset
    arrays = {
        "expert_middle": values,
        "expert_late": values,
        "action_postprocessed": rng.normal(size=(6, 2, 4, 7)).astype("float32"),
    }
    return np.array([f"s{i}" for i in range(6)]), arrays, {}, {"binding_sha256": str(offset)}


def test_lineage_binds_base_checkpoints_and_training_contract(tmp_path: Path):
    base = tmp_path / "base"; base.mkdir(); (base / "config.json").write_text("{}")
    checkpoints = tmp_path / "checkpoints"
    for step in (5000, 10000):
        model = checkpoints / f"{step:06d}/pretrained_model"; model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (model / "train_config.json").write_text(json.dumps({
            "seed": 1000, "batch_size": 32, "steps": 25000,
            "policy": {"train_expert_only": False},
        }))
        state = model.parent / "training_state"; state.mkdir()
        (state / "training_step.json").write_text(json.dumps({"step": step}))
    output = tmp_path / "lineage.json"
    result = acquisition.checkpoint_lineage(checkpoints, base, (5000, 10000), output)
    assert result["base_checkpoint_sha256"]
    assert [row["step"] for row in result["checkpoints"]] == [5000, 10000]
    assert result["training_contract"]["batch_size"] == 32


def test_label_blind_discovery_freezes_formation_and_controls(tmp_path: Path, monkeypatch):
    traces = {"early": _trace(0.0), "late": _trace(2.0)}
    monkeypatch.setattr(acquisition, "load_trace", lambda path: traces[path.name])
    output = tmp_path / "discovery"
    result = acquisition.discover_change_subspaces(
        {"early": Path("early"), "late": Path("late")}, before="early", after="late",
        tap="expert_middle", rank=4, output=output,
    )
    roles = {row["role"] for row in result["candidates"]}
    assert roles == {"formation", "low_change", "matched_random"}
    assert result["selection_uses_physical_labels"] is False
    assert (output / "shared_basis.npz").is_file()
    assert (output / "candidates.json").is_file()


def test_timeline_summary_checks_paired_state_ids(tmp_path: Path):
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({"kind": "immutable_lineage", "checkpoints": [{"step": 5000}]}))
    root = tmp_path / "eval"
    root.mkdir()
    (root / "evaluation_plan.json").write_text(json.dumps({
        "lineage_sha256": acquisition.file_hash(lineage), "tasks": [0],
        "initial_state_offset": 10, "episodes_per_task": 2,
    }))
    cell = root / "step_005000/task0"; cell.mkdir(parents=True)
    events = [{"initial_state_id": state, **{key: True for key in acquisition.EVENT_KEYS}}
              for state in (10, 11)]
    (cell / "physical_events.json").write_text(json.dumps({"episodes": events}))
    (cell / "eval_info.json").write_text(json.dumps({
        "per_task": [{"metrics": {"successes": [True, True]}}]
    }))
    result = acquisition.summarize_timeline(lineage, root, tmp_path / "summary")
    assert result["aggregate"][0]["success"] == 2
    assert (tmp_path / "summary/results.csv").is_file()


def test_closed_loop_plan_only_admits_gated_target_and_explicit_controls(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"candidates": [
        {"id": "formation_0", "role": "formation", "tap": "expert_middle", "direction": [1.0]},
        {"id": "low_change_0", "role": "low_change", "tap": "expert_middle", "direction": [1.0]},
        {"id": "matched_random_0", "role": "matched_random", "tap": "expert_middle", "direction": [1.0]},
    ]}))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "candidate_sha256": acquisition.file_hash(candidates),
        "passed_candidate_ids": ["formation_0"],
    }))
    result = acquisition.run_closed_loop(
        checkpoint, candidates, gate, tmp_path / "rollouts", [0], [], dose=1.0,
        stages=(0, 5, 9), initial_state_offset=20, episodes=2, dry_run=True,
    )
    assert result["conditions"] == [
        "baseline", "formation_0", "low_change_0", "matched_random_0"
    ]
    assert any("flow_intervention_eval" in item for item in result["commands"][1])
