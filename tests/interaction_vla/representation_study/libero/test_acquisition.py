import json
from pathlib import Path
from types import SimpleNamespace

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
        "epsilon": np.zeros((6, 2, 4, 8), dtype="float32"),
        "sigma": np.zeros((2, 3), dtype="float32"),
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


def test_frozen_candidate_trajectory_does_not_refit(tmp_path: Path, monkeypatch):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({
        "tap": "expert_middle", "before": "5k", "after": "10k",
        "candidates": [{"id": "formation_0", "direction": [1.0, 0, 0, 0, 0, 0, 0, 0]}],
    }))
    traces = {"5k": _trace(0.0), "10k": _trace(2.0), "15k": _trace(3.0)}
    monkeypatch.setattr(acquisition, "load_trace", lambda path: traces[path.name])
    output = tmp_path / "trajectory"
    result = acquisition.candidate_trajectory(
        {name: Path(name) for name in traces}, candidates, output
    )
    assert result["selection_uses_physical_labels"] is False
    assert [row["checkpoint"] for row in result["summaries"]] == ["5k", "10k", "15k"]
    assert abs(result["summaries"][0]["endpoint_projection_coefficient"][0]) < 1e-12
    assert abs(result["summaries"][1]["endpoint_projection_coefficient"][0] - 1) < 1e-12
    with np.load(output / "projections.npz", allow_pickle=False) as values:
        assert values["projection_15k"].shape == (6, 1, 2, 3)


def test_intervention_smoke_runs_signed_stage_groups(tmp_path: Path, monkeypatch):
    candidates = tmp_path / "candidates.json"; candidates.write_text("{}")
    calls = []

    def run(command, check):
        assert check is True
        calls.append(command)

    monkeypatch.setattr(acquisition.subprocess, "run", run)
    result = acquisition.intervention_smoke_command(
        tmp_path / "bank", tmp_path / "dataset", tmp_path / "checkpoint",
        tmp_path / "contract", tmp_path / "metadata", candidates, "formation_0",
        tmp_path / "smoke", device="cuda", max_states=2, dose=1.0,
        stages=tuple(range(10)),
    )
    assert result["stage_groups"] == {
        "early": [0, 1, 2], "middle": [3, 4, 5, 6],
        "late": [7, 8, 9], "all": list(range(10)),
    }
    assert len(calls) == 9
    assert any("-1.0" in command for command in calls)
    assert any("1.0" in command for command in calls)


def test_incremental_r2_detects_signal_after_nuisance():
    nuisance = np.arange(20, dtype=float)[:, None]
    target = np.tile((0.0, 1.0), 10)
    y = nuisance[:, 0] + 5 * target
    assert acquisition._incremental_r2(y, nuisance, target) > 0.5


def test_candidate_interpretation_runs_only_after_freeze(tmp_path: Path, monkeypatch):
    trajectory = tmp_path / "trajectory"; trajectory.mkdir()
    projections = trajectory / "projections.npz"
    ids = np.asarray([f"s{i}" for i in range(8)])
    values = np.arange(8, dtype="float32")[:, None, None, None]
    np.savez(projections, state_ids=ids, candidate_ids=np.asarray(["formation_0"]),
             projection_25k=np.repeat(values, 3, axis=3))
    (trajectory / "report.json").write_text(json.dumps({
        "kind": "frozen_candidate_trajectory",
        "projections_sha256": acquisition.file_hash(projections),
    }))
    bank = tmp_path / "bank"; bank.mkdir(); (bank / "manifest.json").write_text("{}")
    records = []
    for i, state_id in enumerate(ids.tolist()):
        labels = SimpleNamespace(
            contact=SimpleNamespace(gripper_target=bool(i % 2)), stable_grasp=bool(i % 2),
            geometry=SimpleNamespace(gripper_target_distance=float(i), target_goal_distance=float(7 - i)),
            phase="grasp" if i % 2 else "approach",
        )
        records.append(SimpleNamespace(
            state_id=state_id, task_id=i % 2, frame_index=i, labels=labels,
            observation=SimpleNamespace(action=(0, 0, 0, 0, 0, 0, i % 2)),
        ))
    monkeypatch.setattr(acquisition, "load_state_bank", lambda _: (records, {}, None, None))
    output = tmp_path / "interpret.json"
    result = acquisition.interpret_candidates(trajectory, bank, "25k", output)
    assert result["selection_performed"] is False
    assert result["action_fields_are_leakage_diagnostics"] is True
    assert output.is_file()


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
