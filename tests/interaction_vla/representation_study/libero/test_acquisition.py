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
        "initial_state_count": 50,
    }))
    cell = root / "step_005000/task0"; cell.mkdir(parents=True)
    events = [{"task_id": 0, "initial_state_id": state,
               "requested_initial_state_id": state, "initial_state_count": 50,
               **{key: True for key in acquisition.EVENT_KEYS}}
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
        stages=(0, 5, 9), initial_state_offset=20, initial_state_count=50,
        episodes=2, dry_run=True,
    )
    assert result["conditions"] == [
        "baseline", "formation_0", "low_change_0", "matched_random_0"
    ]
    assert any("flow_intervention_eval" in item for item in result["commands"][1])


def test_offline_gate_uses_executed_prefix_episode_clusters_and_matched_controls(
    tmp_path: Path, monkeypatch,
):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"candidates": [
        {"id": "formation_0", "role": "formation"},
        {"id": "low_change_0", "role": "low_change"},
        {"id": "matched_random_0", "role": "matched_random"},
    ]}))
    bank = tmp_path / "bank"; bank.mkdir(); (bank / "manifest.json").write_text("{}")
    ids = np.asarray([f"s{i}" for i in range(8)])
    records = [SimpleNamespace(state_id=state_id, suite="spatial", task_id=i % 2,
                               source_episode_id=f"episode-{i}")
               for i, state_id in enumerate(ids)]
    monkeypatch.setattr(acquisition, "load_state_bank",
                        lambda _: (records, {}, None, None))
    baseline_action = np.zeros((8, 1, 50, 7), dtype="float32")
    shared = {"action_postprocessed": baseline_action,
              "epsilon": np.zeros((8, 1, 50, 32), dtype="float32"),
              "sigma": np.zeros((1, 10), dtype="float32")}

    def trace(candidate_id=None, amount=0.0, stages=(0, 1, 2), dose=1.0):
        arrays = {key: value.copy() for key, value in shared.items()}
        arrays["action_postprocessed"][:, :, :10] += amount
        return ids, arrays, {"binding_sha256": "base", "flow_edit": (
            None if candidate_id is None else {
                "candidate_id": candidate_id, "stages": list(stages), "dose": dose,
            })}, {}

    traces = {
        "baseline": trace(), "formation": trace("formation_0", 2.0),
        "low": trace("low_change_0", 0.1),
        "random": trace("matched_random_0", 0.2),
    }
    monkeypatch.setattr(acquisition, "load_trace", lambda path: traces[path.name])
    output = tmp_path / "gate.json"
    result = acquisition.offline_action_gate(
        Path("baseline"), {"target_early_pos": Path("formation"),
                           "low_early_pos": Path("low"),
                           "random_early_pos": Path("random")},
        candidates, bank, output, bootstrap_samples=100,
    )
    row = result["rows"][0]
    assert result["primary_metric"] == "first_10_actions_full_rms"
    assert result["bootstrap_unit"] == "source_episode"
    assert row["independent_episodes"] == 8
    assert row["matched_controls"] == ["low_early_pos", "random_early_pos"]
    assert row["mean_executed_action_rms"]["gripper"] == 2.0
    assert row["passed"] is True


def test_confirmation_contract_rejects_previously_used_cells(tmp_path: Path):
    used = tmp_path / "used.json"
    used.write_text(json.dumps({"tasks": [0], "initial_state_offset": 20,
                                "episodes_per_task": 5, "initial_state_count": 50}))
    with np.testing.assert_raises_regex(ValueError, "used previously"):
        acquisition.freeze_confirmation_contract(
            tmp_path / "confirmation.json", [0], initial_state_offset=24,
            episodes=3, used_plans=[used], initial_state_count=50,
        )
    result = acquisition.freeze_confirmation_contract(
        tmp_path / "fresh.json", [0, 1], initial_state_offset=30,
        episodes=5, used_plans=[used], initial_state_count=50,
    )
    assert result["selection_uses_confirmation_results"] is False
    assert result["contract_complete"] is False
    assert "checkpoint_sha256" in result["missing_binding"]


def test_candidate_context_is_episode_held_out_and_does_not_select(tmp_path: Path,
                                                                   monkeypatch):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({
        "tap": "expert_late",
        "candidates": [{"id": "formation_0", "role": "formation",
                        "direction": [1.0, 0.0]}],
    }))
    bank = tmp_path / "bank"; bank.mkdir(); (bank / "manifest.json").write_text("{}")
    ids = np.asarray([f"s{i}" for i in range(8)])
    records = []
    for i, state_id in enumerate(ids):
        records.append(SimpleNamespace(
            state_id=state_id, suite="spatial", task_id=i % 2,
            source_episode_id=f"episode-{i}", frame_index=i,
            observation=SimpleNamespace(robot_state=tuple(float(i + j) for j in range(8)),
                                        action=tuple(float(i % 3) for _ in range(7))),
            labels=SimpleNamespace(
                phase="approach" if i % 2 else "align_precontact",
                geometry=SimpleNamespace(gripper_target_distance=float(i),
                                         target_goal_distance=float(8 - i)),
                contact=SimpleNamespace(gripper_target=bool(i % 2)),
                stable_grasp=bool(i % 2),
            ),
        ))
    activations = np.zeros((8, 2, 3, 4, 2), dtype="float32")
    activations[..., 0] = np.arange(8)[:, None, None, None]
    monkeypatch.setattr(acquisition, "load_trace", lambda _: (
        ids, {"expert_late": activations}, {"binding_sha256": "trace"}, {}
    ))
    monkeypatch.setattr(acquisition, "load_state_bank",
                        lambda _: (records, {}, None, None))
    output = tmp_path / "context.json"
    result = acquisition.candidate_context_analysis(
        Path("trace"), candidates, bank, output, ["formation_0"]
    )
    assert result["selection_performed"] is False
    assert result["cross_validation_unit"] == "source_episode"
    assert result["independent_episodes"] == 8
    assert len(result["results"]) == 3


def test_confirmation_rejects_overlap_after_simulator_state_modulo(tmp_path: Path):
    events = tmp_path / "events.json"
    events.write_text(json.dumps({"episodes": [{"task_id": 0, "initial_state_id": 10}]}))
    used = tmp_path / "used.json"
    used.write_text(json.dumps({
        "tasks": [0], "initial_state_offset": 50, "episodes_per_task": 1,
        "commands": [["--events-output", str(events)]],
    }))
    with np.testing.assert_raises_regex(ValueError, "used previously"):
        acquisition.freeze_confirmation_contract(
            tmp_path / "confirmation.json", [0], initial_state_offset=60,
            episodes=1, initial_state_count=50, used_plans=[used],
        )


def test_actual_state_ids_record_modulo_contract():
    assert acquisition._actual_state_ids(60, 8, 50) == list(range(10, 18))
    command = acquisition._evaluation_command(
        Path("checkpoint"), 0, Path("output"), 60, 8, 50,
    )
    assert "--initial-state-count" in command


def test_history_without_actual_identity_fails_closed(tmp_path):
    plan = {"tasks": [0], "initial_state_offset": 60, "episodes_per_task": 8}
    with np.testing.assert_raises_regex(ValueError, "initial_state_count"):
        acquisition._plan_actual_cells(tmp_path / "old.json", plan)


def test_confirmation_requires_count_and_unique_actual_states(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "initial_state_count"):
        acquisition.freeze_confirmation_contract(
            tmp_path / "missing.json", [0], initial_state_offset=0, episodes=1)
    with np.testing.assert_raises_regex(ValueError, "repeat"):
        acquisition.freeze_confirmation_contract(
            tmp_path / "repeat.json", [0], initial_state_offset=0,
            episodes=51, initial_state_count=50)


def test_closed_loop_never_relabels_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with np.testing.assert_raises_regex(FileExistsError, "existing"):
        acquisition.run_closed_loop(
            tmp_path / "checkpoint", tmp_path / "candidates", tmp_path / "gate",
            output, [0], [], dose=1., stages=[0], initial_state_offset=0,
            episodes=1, dry_run=True)



def test_timeline_summary_rejects_duplicate_or_missing_cells(tmp_path: Path):
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({"kind": "immutable_lineage", "checkpoints": [{"step": 5000}]}))
    root = tmp_path / "eval"; root.mkdir()
    (root / "evaluation_plan.json").write_text(json.dumps({
        "lineage_sha256": acquisition.file_hash(lineage), "tasks": [0],
        "initial_state_offset": 10, "episodes_per_task": 2, "initial_state_count": 50,
    }))
    cell = root / "step_005000/task0"; cell.mkdir(parents=True)
    events = [{"task_id": 0, "initial_state_id": 10,
               "requested_initial_state_id": 10, "initial_state_count": 50,
               **{key: True for key in acquisition.EVENT_KEYS}},
              {"task_id": 0, "initial_state_id": 10,
               "requested_initial_state_id": 10, "initial_state_count": 50,
               **{key: True for key in acquisition.EVENT_KEYS}}]
    (cell / "physical_events.json").write_text(json.dumps({"episodes": events}))
    (cell / "eval_info.json").write_text(json.dumps({"per_task": [{"metrics": {"successes": [True, True]}}]}))
    with np.testing.assert_raises_regex(ValueError, "duplicate"):
        acquisition.summarize_timeline(lineage, root, tmp_path / "summary")


def test_longitudinal_audit_requires_all_r_u_s_artifacts(tmp_path: Path):
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({"kind": "immutable_lineage", "checkpoints": [
        {"step": 5000, "checkpoint_sha256": "a" * 64},
        {"step": 10000, "checkpoint_sha256": "b" * 64},
    ]}))
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps({
        "kind": "capability_timeline", "complete": True,
        "lineage_sha256": acquisition.file_hash(lineage),
        "rows": [
            {"step": 5000, "task": 0, "episodes": 1, "initial_state_ids": [10]},
            {"step": 10000, "task": 0, "episodes": 1, "initial_state_ids": [10]},
        ],
        "paired_initial_state_ids": {"0": [10]},
    }))
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"candidates": [{"id": "formation_0"}]}))
    trajectory = tmp_path / "trajectory"; trajectory.mkdir()
    projection = trajectory / "projections.npz"
    np.savez(projection, candidate_ids=np.asarray(["formation_0"]), projection_5k=np.zeros((1, 1)), projection_10k=np.zeros((1, 1)))
    (trajectory / "report.json").write_text(json.dumps({
        "kind": "frozen_candidate_trajectory", "candidate_sha256": acquisition.file_hash(candidates),
        "selection_uses_physical_labels": False,
        "projections_sha256": acquisition.file_hash(projection),
        "summaries": [{"checkpoint": "5k"}, {"checkpoint": "10k"}],
        "trace_checkpoint_sha256": {"5k": "a" * 64, "10k": "b" * 64},
        "state_ids": ["s0"],
    }))
    report = acquisition.audit_longitudinal_evidence(
        lineage, timeline, candidates, trajectory, {}, tmp_path / "audit.json"
    )
    assert report["representation_alignment"]["passed"] is True
    assert report["capability_comparison"]["passed"] is True
    assert report["functional_dependence"]["passed"] is False
    assert report["protocol_ready"] is False


def test_longitudinal_audit_passes_only_with_functional_reports(tmp_path: Path):
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({"kind": "immutable_lineage", "checkpoints": [
        {"step": 5000, "checkpoint_sha256": "a" * 64},
        {"step": 10000, "checkpoint_sha256": "b" * 64},
    ]}))
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps({
        "kind": "capability_timeline", "complete": True,
        "lineage_sha256": acquisition.file_hash(lineage),
        "rows": [
            {"step": 5000, "task": 0, "episodes": 1, "initial_state_ids": [10]},
            {"step": 10000, "task": 0, "episodes": 1, "initial_state_ids": [10]},
        ],
        "paired_initial_state_ids": {"0": [10]},
    }))
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"candidates": [{"id": "formation_0"}]}))
    trajectory = tmp_path / "trajectory"; trajectory.mkdir()
    projection = trajectory / "projections.npz"
    np.savez(projection, candidate_ids=np.asarray(["formation_0"]),
             projection_5k=np.zeros((1, 1)), projection_10k=np.zeros((1, 1)))
    (trajectory / "report.json").write_text(json.dumps({
        "kind": "frozen_candidate_trajectory", "candidate_sha256": acquisition.file_hash(candidates),
        "selection_uses_physical_labels": False,
        "projections_sha256": acquisition.file_hash(projection),
        "summaries": [{"checkpoint": "5k"}, {"checkpoint": "10k"}],
        "trace_checkpoint_sha256": {"5k": "a" * 64, "10k": "b" * 64},
        "state_ids": ["s0"],
    }))
    gates = {}
    for name, checkpoint_hash in (("5k", "a" * 64), ("10k", "b" * 64)):
        gate = tmp_path / f"gate_{name}.json"
        gate.write_text(json.dumps({
            "kind": "offline_action_gate", "candidate_sha256": acquisition.file_hash(candidates),
            "checkpoint_sha256": checkpoint_hash, "independent_episodes": 8,
            "numerical_self_consistency": {
                "all_arrays_finite": True, "shared_state_noise_sigma": True,
            },
        }))
        gates[name] = gate
    report = acquisition.audit_longitudinal_evidence(
        lineage, timeline, candidates, trajectory, gates, tmp_path / "audit.json"
    )
    assert report["functional_dependence"]["passed"] is True
    assert report["protocol_ready"] is True
