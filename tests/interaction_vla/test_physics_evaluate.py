from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

import interaction_vla.physics_evaluate as physics_evaluate_module
import interaction_vla.physics_data as physics_data_module
from interaction_vla.config import load_config
from interaction_vla.contact_physics import GraspState, InteractionSubstepEvent
from interaction_vla.franka_controller import ControllerDiagnostics
from interaction_vla.models.encoders import SceneBatch
from interaction_vla.physics_action_safety import IKProjectionResult
from interaction_vla.physics_env import FrankaContactEnv
from interaction_vla.physics_evaluate import (
    InteractionRolloutTracker,
    PhysicsEpisodeResult,
    PhysicsEvaluationCase,
    aggregate_physics_results,
    build_parser,
    default_evaluation_conditions,
    evaluate_from_config,
    initial_case_fingerprint,
    make_physics_evaluation_cases,
    preload_evaluation_checkpoints,
    resolve_evaluation_model_seeds,
    rollout_physics_policy,
    shuffle_valid_physics_edges,
    validate_physics_checkpoint,
)
from interaction_vla.train import TrainingStatistics


def test_physics_cases_are_unique_and_cover_id_count_and_crowded() -> None:
    config = load_config("configs/physics_smoke_macos.yaml")
    cases = make_physics_evaluation_cases(config)

    assert {case.condition for case in cases} == {
        "id_normal",
        "count_ood",
        "crowded_ood",
        "controlled_randomization",
    }
    assert len({(case.condition, case.seed, case.object_count) for case in cases}) == len(cases)
    assert all(
        case.layout_mode == ("crowded" if case.condition == "crowded_ood" else "normal")
        for case in cases
    )
    assert all(
        case.randomized == (case.condition == "controlled_randomization")
        for case in cases
    )


def test_v3_default_evaluation_is_interaction_id_plus_heldout_recovery() -> None:
    v3 = load_config("configs/physics_interaction_chunk_pilot_macos.yaml")
    legacy = load_config("configs/physics_recovery_pilot_macos.yaml")

    assert default_evaluation_conditions(v3) == (
        "id_normal",
        "heldout_recovery",
    )
    assert default_evaluation_conditions(legacy) is None


def test_physics_case_count_can_be_overridden_without_changing_groups() -> None:
    config = load_config("configs/physics_recovery_pilot_macos.yaml")

    full = make_physics_evaluation_cases(config)
    quick = make_physics_evaluation_cases(config, episodes_per_count=5)

    assert len(full) == 160
    assert len(quick) == 40
    assert {case.condition for case in quick} == {
        "id_normal",
        "count_ood",
        "crowded_ood",
        "controlled_randomization",
    }
    assert quick == make_physics_evaluation_cases(config, episodes_per_count=5)


@pytest.mark.parametrize("episodes_per_count", [0, -1])
def test_physics_case_count_override_must_be_positive(
    episodes_per_count: int,
) -> None:
    config = load_config("configs/physics_recovery_pilot_macos.yaml")

    with pytest.raises(ValueError, match="episodes_per_count must be positive"):
        make_physics_evaluation_cases(
            config,
            episodes_per_count=episodes_per_count,
        )


def test_model_seed_override_selects_seed_zero_without_changing_config() -> None:
    config = load_config("configs/physics_recovery_pilot_macos.yaml")

    assert resolve_evaluation_model_seeds(config, None) == (0, 1, 2)
    assert resolve_evaluation_model_seeds(config, (0,)) == (0,)
    with pytest.raises(ValueError, match="unique"):
        resolve_evaluation_model_seeds(config, (0, 0))
    with pytest.raises(ValueError, match="configured"):
        resolve_evaluation_model_seeds(config, (9,))


def test_physics_evaluate_parser_accepts_scope_overrides() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "configs/physics_recovery_pilot_macos.yaml",
            "--model-seeds",
            "0",
            "--episodes-per-count",
            "5",
            "--conditions",
            "id_normal",
            "crowded_ood",
            "--output",
            "id_sanity.json",
            "--disable-ik-projection",
            "--include-edge-shuffle",
        ]
    )

    assert args.model_seeds == [0]
    assert args.episodes_per_count == 5
    assert args.conditions == ["id_normal", "crowded_ood"]
    assert args.output == "id_sanity.json"
    assert args.disable_ik_projection is True
    assert args.include_edge_shuffle is True


def test_physics_evaluate_main_forwards_scope_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    captured: dict[str, object] = {}

    def fake_evaluate(config_path: str, **kwargs: object) -> Path:
        captured.update({"config_path": config_path, **kwargs})
        return report

    monkeypatch.setattr(physics_evaluate_module, "evaluate_from_config", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "physics_evaluate",
            "--config",
            "configs/physics_recovery_pilot_macos.yaml",
            "--model-seeds",
            "0",
            "--episodes-per-count",
            "5",
            "--conditions",
            "id_normal",
            "--output",
            "id_sanity.json",
            "--disable-ik-projection",
            "--include-edge-shuffle",
        ],
    )

    physics_evaluate_module.main()

    assert captured == {
        "config_path": "configs/physics_recovery_pilot_macos.yaml",
        "model_seeds": [0],
        "include_edge_shuffle": True,
        "episodes_per_count": 5,
        "conditions": ["id_normal"],
        "output": "id_sanity.json",
        "ik_projection": False,
        "show_progress": True,
    }


def test_condition_filter_preserves_generated_case_order() -> None:
    cases = (
        PhysicsEvaluationCase("id:2:1", 1, 2, "id_normal", "normal"),
        PhysicsEvaluationCase("count:4:2", 2, 4, "count_ood", "normal"),
        PhysicsEvaluationCase("crowded:4:3", 3, 4, "crowded_ood", "crowded"),
        PhysicsEvaluationCase(
            "random:4:4", 4, 4, "controlled_randomization", "normal", True
        ),
    )

    selected = physics_evaluate_module.resolve_evaluation_conditions(
        cases,
        ("crowded_ood", "id_normal"),
    )

    assert selected == (cases[0], cases[2])
    assert physics_evaluate_module.resolve_evaluation_conditions(cases, None) == cases


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        ((), "at least one"),
        (("id_normal", "id_normal"), "unique"),
        (("unknown",), "unknown"),
    ],
)
def test_condition_filter_rejects_invalid_selection(
    requested: tuple[str, ...],
    message: str,
) -> None:
    cases = (PhysicsEvaluationCase("id:2:1", 1, 2, "id_normal", "normal"),)

    with pytest.raises(ValueError, match=message):
        physics_evaluate_module.resolve_evaluation_conditions(cases, requested)


def test_evaluation_output_paths_allow_isolated_report_in_evaluation_dir(
    tmp_path: Path,
) -> None:
    config = replace(
        load_config("configs/physics_smoke_macos.yaml"),
        output_dir=str(tmp_path / "experiment"),
    )
    report = tmp_path / "experiment" / "evaluation" / "id_sanity.json"

    report_path, csv_path = physics_evaluate_module.resolve_evaluation_output_paths(
        config,
        report,
    )

    assert report_path == report
    assert csv_path == report.with_name("id_sanity_episodes.csv")


@pytest.mark.parametrize(
    "relative_output",
    [
        "expert_gate.json",
        "evaluation/report.json",
        "evaluation/not_json.txt",
        "../outside.json",
    ],
)
def test_evaluation_output_paths_reject_unsafe_or_invalid_destinations(
    tmp_path: Path,
    relative_output: str,
) -> None:
    output_dir = tmp_path / "experiment"
    config = replace(
        load_config("configs/physics_smoke_macos.yaml"),
        output_dir=str(output_dir),
    )

    with pytest.raises(ValueError, match=r"evaluation|\.json|default"):
        physics_evaluate_module.resolve_evaluation_output_paths(
            config,
            output_dir / relative_output,
        )


def test_selected_checkpoint_preflight_ignores_unselected_seeds_and_fails_before_load(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        load_config("configs/physics_recovery_pilot_macos.yaml"),
        output_dir=str(tmp_path),
    )
    checkpoint_paths = {}
    for representation in ("flat", "graph"):
        checkpoint = tmp_path / representation / "seed_0" / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        checkpoint_paths[representation] = checkpoint
    physical_hashes = {
        "expert_gate_hash": "gate",
        "controller_hash": "controller",
        "scene_hash": "scene",
        "config_hash": "config",
    }
    training_provenance = {"dataset_content_hash": "dataset"}
    load_calls: list[Path] = []

    def fake_load_training_checkpoint(checkpoint, device):
        checkpoint_path = Path(checkpoint)
        load_calls.append(checkpoint_path)
        representation = checkpoint_path.parents[1].name
        payload = {
            "backend": "franka_contact",
            "feature_schema": "physics_v2",
            "action_mode": "cartesian_7d",
            "action_dim": 7,
            "proprioception_dim": 23,
            "model_kwargs": {"edge_feature_dim": 18},
            "representation": representation,
            "model_seed": 0,
            "training_provenance": training_provenance,
            **physical_hashes,
        }
        return object(), object(), payload

    monkeypatch.setattr(
        physics_evaluate_module,
        "load_training_checkpoint",
        fake_load_training_checkpoint,
    )
    loaded = preload_evaluation_checkpoints(
        config,
        model_seeds=(0,),
        device="cpu",
        physical_hashes=physical_hashes,
        expected_training_provenance=training_provenance,
    )

    assert set(loaded) == {(0, "flat"), (0, "graph")}
    checkpoint_paths["graph"].unlink()
    load_calls.clear()
    with pytest.raises(FileNotFoundError, match=r"graph/seed_0/checkpoint\.pt"):
        preload_evaluation_checkpoints(
            config,
            model_seeds=(0,),
            device="cpu",
            physical_hashes=physical_hashes,
            expected_training_provenance=training_provenance,
        )
    assert load_calls == []


def test_edge_shuffle_changes_only_valid_edge_row_assignment() -> None:
    features = torch.arange(2 * 6 * 18, dtype=torch.float32).reshape(2, 6, 18)
    batch = SceneBatch(
        node_features=torch.zeros((2, 4, 23)),
        edge_index=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
        edge_features=features,
        node_mask=torch.ones((2, 4), dtype=torch.bool),
        edge_mask=torch.tensor(
            [[True, True, True, True, False, False], [True, True, True, False, False, False]]
        ),
    )

    shuffled = shuffle_valid_physics_edges(batch, seed=5)

    torch.testing.assert_close(shuffled.node_features, batch.node_features)
    torch.testing.assert_close(shuffled.edge_index, batch.edge_index)
    torch.testing.assert_close(shuffled.node_mask, batch.node_mask)
    torch.testing.assert_close(shuffled.edge_mask, batch.edge_mask)
    for row in range(2):
        valid = batch.edge_mask[row]
        original_rows = sorted(tuple(value.tolist()) for value in batch.edge_features[row, valid])
        shuffled_rows = sorted(tuple(value.tolist()) for value in shuffled.edge_features[row, valid])
        assert shuffled_rows == original_rows
        torch.testing.assert_close(
            shuffled.edge_features[row, ~valid], batch.edge_features[row, ~valid]
        )


def test_same_physics_case_has_identical_initial_fingerprint() -> None:
    kwargs = dict(
        max_steps=3,
        physics=load_config("configs/physics_smoke_macos.yaml").physics,
        workspace_low=(0.25, -0.35, 0.23),
        workspace_high=(0.78, 0.35, 0.75),
        crowded_anchor_min_distance=0.055,
        crowded_anchor_max_distance=0.075,
    )
    first = FrankaContactEnv(**kwargs)
    second = FrankaContactEnv(**kwargs)
    first.reset(seed=123, object_count=4, layout_mode="crowded")
    second.reset(seed=123, object_count=4, layout_mode="crowded")

    assert initial_case_fingerprint(first) == initial_case_fingerprint(second)
    np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
    np.testing.assert_array_equal(first.data.qvel, second.data.qvel)


def test_checkpoint_contract_rejects_legacy_or_wrong_dimensions() -> None:
    valid = {
        "backend": "franka_contact",
        "feature_schema": "physics_v2",
        "action_mode": "cartesian_7d",
        "action_dim": 7,
        "proprioception_dim": 23,
        "model_kwargs": {"edge_feature_dim": 18},
    }
    validate_physics_checkpoint(valid)
    invalid = dict(valid, action_dim=4)
    with pytest.raises(ValueError, match="7D"):
        validate_physics_checkpoint(invalid)
    physical_hashes = {
        "expert_gate_hash": "a",
        "controller_hash": "b",
        "scene_hash": "c",
        "config_hash": "d",
    }
    with pytest.raises(ValueError, match="stale"):
        validate_physics_checkpoint(valid, expected_provenance=physical_hashes)
    identity = {
        **valid,
        **physical_hashes,
        "representation": "graph",
        "model_seed": 3,
    }
    validate_physics_checkpoint(
        identity,
        expected_provenance=physical_hashes,
        expected_representation="graph",
        expected_model_seed=3,
    )
    with pytest.raises(ValueError, match="identity"):
        validate_physics_checkpoint(
            identity,
            expected_representation="flat",
            expected_model_seed=3,
        )


def test_physics_aggregation_reports_interaction_metrics_and_paired_delta() -> None:
    common = dict(
        case_id="crowded_ood:4:77",
        seed=77,
        object_count=4,
        condition="crowded_ood",
        layout_mode="crowded",
        model_seed=0,
        ablation="none",
        bilateral_contact=True,
        wrong_object_stable_grasp=False,
        dropped=False,
        placement=False,
        ik_failure=False,
        physics_failure=False,
        steps=20,
        termination_reason="timeout",
        physics_hash="abc",
        initial_state_hash="xyz",
    )
    flat = PhysicsEpisodeResult(
        policy="flat_seed_0",
        representation="flat",
        success=False,
        stable_lift=False,
        transport_progress=0.05,
        transport_progress_rate=0.2,
        premature_open=True,
        action_saturation_rate=1.0,
        ik_projection_rate=0.5,
        zero_pose_projection_rate=0.25,
        mean_ik_projection_scale=0.75,
        post_placement_reclose=True,
        **common,
    )
    graph = PhysicsEpisodeResult(
        policy="graph_seed_0", representation="graph", success=True, stable_lift=True,
        placement=True, termination_reason="success", transport_progress=0.20,
        transport_progress_rate=0.8, premature_open=False,
        action_saturation_rate=0.0, ik_projection_rate=0.25,
        zero_pose_projection_rate=0.0, mean_ik_projection_scale=0.875, **{
            key: value for key, value in common.items()
            if key not in {"placement", "termination_reason"}
        }
    )
    shuffled = PhysicsEpisodeResult(
        policy="graph_edge_shuffle_seed_0",
        representation="graph",
        success=False,
        stable_lift=False,
        ablation="edge_shuffle",
        transport_progress=0.02,
        transport_progress_rate=0.1,
        premature_open=True,
        **{key: value for key, value in common.items() if key != "ablation"},
    )

    report = aggregate_physics_results((flat, graph, shuffled))

    assert report["by_policy"]["graph_seed_0"]["stable_lift_rate"] == 1.0
    assert report["by_policy"]["graph_seed_0"]["mean_transport_progress"] == 0.20
    assert report["by_policy"]["graph_seed_0"]["premature_open_rate"] == 0.0
    assert report["by_policy"]["flat_seed_0"]["action_saturation_rate"] == 1.0
    assert report["by_policy"]["flat_seed_0"]["ik_projection_rate"] == 0.5
    assert report["by_policy"]["flat_seed_0"]["zero_pose_projection_rate"] == 0.25
    assert report["by_policy"]["flat_seed_0"]["mean_ik_projection_scale"] == 0.75
    assert report["by_policy"]["flat_seed_0"][
        "post_placement_reclose_rate"
    ] == 1.0
    assert report["by_policy"]["graph_seed_0"][
        "post_placement_reclose_rate"
    ] == 0.0
    assert report["overall"]["termination_reason_counts"] == {
        "success": 1,
        "timeout": 2,
    }
    assert report["graph_vs_flat"]["by_model_seed"]["0"]["success_delta"] == 1.0
    assert report["graph_vs_flat"]["by_model_seed"]["0"]["stable_lift_delta"] == 1.0
    assert report["graph_vs_flat"]["by_model_seed"]["0"][
        "transport_progress_rate_delta"
    ] == pytest.approx(0.6)
    assert report["graph_vs_flat"]["by_model_seed"]["0"][
        "premature_open_delta"
    ] == -1.0
    assert (
        report["graph_vs_flat"]["by_model_seed"]["0"]["by_condition"]
        ["crowded_ood"]["success_delta"]
        == 1.0
    )
    assert (
        report["graph_vs_edge_shuffle"]["by_model_seed"]["0"]["success_delta"]
        == 1.0
    )
    assert report["learned_policy_sanity"] == {}


def test_interaction_tracker_does_not_credit_expert_prefix() -> None:
    baseline = {
        "tracker_substep": 50,
        "ever_bilateral_target_contact": True,
        "ever_stable_target": True,
        "ever_bilateral_wrong_object": False,
        "ever_stable_wrong_object": False,
        "total_stable_target_substeps": 5,
        "dropped_target": False,
    }
    tracker = InteractionRolloutTracker(
        target_name="object_0",
        baseline=baseline,
    )
    prefix_only = GraspState(
        bilateral_object=None,
        stable_object=None,
        stable_frames=0,
        ever_stable_target=True,
        dropped_target=False,
        ever_bilateral_target_contact=True,
        total_stable_target_substeps=5,
        tracker_substep=50,
    )

    tracker.observe(prefix_only)
    metrics = tracker.metrics()

    assert metrics["target_bilateral_contact"] is False
    assert metrics["stable_target_grasp"] is False
    assert metrics["stable_target_substeps"] == 0


def test_interaction_tracker_preserves_substep_order_and_transient_wrong_grasp() -> None:
    tracker = InteractionRolloutTracker(target_name="object_0", baseline={})
    final_state = GraspState(
        bilateral_object=None,
        stable_object=None,
        stable_frames=0,
        ever_stable_target=False,
        dropped_target=False,
        ever_stable_wrong_object=True,
        total_bilateral_target_substeps=1,
        total_bilateral_wrong_substeps=1,
        tracker_substep=3,
    )
    events = (
        InteractionSubstepEvent(
            substep=1,
            bilateral_objects=("object_1",),
        ),
        InteractionSubstepEvent(
            substep=2,
            bilateral_objects=("object_0",),
        ),
        InteractionSubstepEvent(
            substep=3,
            stable_objects=("object_1",),
        ),
    )

    tracker.observe(final_state, events=events)
    metrics = tracker.metrics()

    assert metrics["first_bilateral_object"] == "object_1"
    assert metrics["target_first_contact"] is False
    assert metrics["first_bilateral_target_substep"] == 2
    assert metrics["wrong_object_interaction"] is True
    assert metrics["wrong_object_stable_grasp"] is True


def test_v3_aggregation_groups_primary_interaction_and_secondary_task() -> None:
    result = PhysicsEpisodeResult(
        policy="graph_seed_0",
        representation="graph",
        model_seed=0,
        ablation="none",
        case_id="id_normal:2:1",
        seed=1,
        object_count=2,
        condition="id_normal",
        layout_mode="normal",
        success=True,
        bilateral_contact=True,
        stable_lift=True,
        wrong_object_stable_grasp=False,
        dropped=False,
        placement=True,
        ik_failure=False,
        physics_failure=False,
        steps=20,
        termination_reason="success",
        physics_hash="physics",
        initial_state_hash="state",
        first_bilateral_object="object_0",
        target_first_contact=True,
        target_bilateral_contact=True,
        stable_target_grasp=True,
        stable_target_substeps=17,
        wrong_object_interaction=False,
        dropped_target=False,
        strict_containment=True,
        receptacle_base_contact=True,
        strict_placement=True,
        gripper_released=True,
        tcp_retreated=True,
        strict_task_success=True,
    )

    metrics = aggregate_physics_results((result,))["by_policy"]["graph_seed_0"]

    assert metrics["primary_interaction"]["target_first_contact_rate"] == 1.0
    assert metrics["primary_interaction"]["grasp_given_contact_rate"] == 1.0
    assert metrics["primary_interaction"]["mean_stable_target_substeps"] == 17.0
    assert metrics["secondary_task"]["strict_task_success_rate"] == 1.0


def test_physics_aggregation_reports_id_control_and_manipulation_sanity() -> None:
    common = dict(
        case_id="id_normal:2:77",
        seed=77,
        object_count=2,
        condition="id_normal",
        layout_mode="normal",
        model_seed=0,
        ablation="none",
        bilateral_contact=True,
        wrong_object_stable_grasp=False,
        dropped=False,
        placement=False,
        ik_failure=False,
        steps=20,
        termination_reason="timeout",
        physics_hash="abc",
        initial_state_hash="xyz",
        success=False,
    )
    healthy = PhysicsEpisodeResult(
        policy="graph_seed_0",
        representation="graph",
        stable_lift=True,
        physics_failure=False,
        **common,
    )
    unhealthy = PhysicsEpisodeResult(
        policy="flat_seed_0",
        representation="flat",
        stable_lift=False,
        physics_failure=True,
        **common,
    )

    sanity = aggregate_physics_results((healthy, unhealthy))["learned_policy_sanity"]

    assert sanity["graph_seed_0"] == {
        "episodes": 1,
        "control_passed": True,
        "manipulation_passed": True,
        "passed": True,
        "physics_failure_rate": 0.0,
        "stable_lift_rate": 1.0,
    }
    assert sanity["flat_seed_0"] == {
        "episodes": 1,
        "control_passed": False,
        "manipulation_passed": False,
        "passed": False,
        "physics_failure_rate": 1.0,
        "stable_lift_rate": 0.0,
    }


class ConstantPhysicalPolicy(torch.nn.Module):
    scene_encoder = None

    def __init__(self, action: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("action", torch.from_numpy(action.astype(np.float32)))

    def forward(self, _scene, proprioception: torch.Tensor) -> torch.Tensor:
        return self.action.unsqueeze(0).expand(proprioception.shape[0], -1)


def _unit_physics_statistics() -> TrainingStatistics:
    return TrainingStatistics(
        node_mean=np.zeros(23, dtype=np.float32),
        node_std=np.ones(23, dtype=np.float32),
        edge_mean=np.zeros(18, dtype=np.float32),
        edge_std=np.ones(18, dtype=np.float32),
        proprio_mean=np.zeros(23, dtype=np.float32),
        proprio_std=np.ones(23, dtype=np.float32),
        action_mean=np.zeros(7, dtype=np.float32),
        action_std=np.ones(7, dtype=np.float32),
    )


def _one_step_case() -> PhysicsEvaluationCase:
    return PhysicsEvaluationCase(
        case_id="id_normal:2:123",
        seed=123,
        object_count=2,
        condition="id_normal",
        layout_mode="normal",
    )


def test_rollout_records_ik_projection_and_action_saturation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/physics_smoke_macos.yaml")
    config = replace(config, eval=replace(config.eval, max_steps=1))
    projection_calls: list[np.ndarray] = []

    def fake_projection(_controller, action: np.ndarray) -> IKProjectionResult:
        raw = np.asarray(action, dtype=np.float32).copy()
        projected = raw.copy()
        projected[:6] *= 0.25
        projection_calls.append(raw)
        raw_diagnostics = ControllerDiagnostics(
            ik_limited=True,
            position_error=0.02,
            orientation_error=0.04,
            iterations=20,
            joint_target=np.zeros(7),
        )
        projected_diagnostics = replace(
            raw_diagnostics,
            ik_limited=False,
            position_error=0.001,
            orientation_error=0.001,
        )
        return IKProjectionResult(
            raw_action=raw,
            action=projected,
            scale=0.25,
            raw_diagnostics=raw_diagnostics,
            projected_diagnostics=projected_diagnostics,
        )

    monkeypatch.setattr(
        physics_evaluate_module,
        "project_cartesian_action",
        fake_projection,
        raising=False,
    )
    result = rollout_physics_policy(
        policy_name="flat_seed_0",
        representation="flat",
        model_seed=0,
        policy=ConstantPhysicalPolicy(
            np.asarray((1.0, 0, 0, 0, 0, 0, 1), dtype=np.float32)
        ),
        statistics=_unit_physics_statistics(),
        config=config,
        case=_one_step_case(),
        device="cpu",
        ik_projection=True,
    )

    assert len(projection_calls) == 1
    assert result.steps == 1
    assert result.action_saturation_rate == 1.0
    assert result.ik_projection_rate == 1.0
    assert result.zero_pose_projection_rate == 0.0
    assert result.mean_ik_projection_scale == 0.25


def test_rollout_can_reproduce_raw_policy_without_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/physics_smoke_macos.yaml")
    config = replace(config, eval=replace(config.eval, max_steps=1))

    def unexpected_projection(*_args, **_kwargs):
        raise AssertionError("projection must be disabled")

    monkeypatch.setattr(
        physics_evaluate_module,
        "project_cartesian_action",
        unexpected_projection,
        raising=False,
    )
    result = rollout_physics_policy(
        policy_name="graph_seed_0",
        representation="graph",
        model_seed=0,
        policy=ConstantPhysicalPolicy(
            np.asarray((1.0, 0, 0, 0, 0, 0, 1), dtype=np.float32)
        ),
        statistics=_unit_physics_statistics(),
        config=config,
        case=_one_step_case(),
        device="cpu",
        ik_projection=False,
    )

    assert result.action_saturation_rate == 1.0
    assert result.ik_projection_rate == 0.0
    assert result.zero_pose_projection_rate == 0.0
    assert result.mean_ik_projection_scale == 1.0


def test_rollout_records_reclose_after_stable_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/physics_smoke_macos.yaml")
    config = replace(config, eval=replace(config.eval, max_steps=2))
    inner_env = physics_evaluate_module._make_env(config)

    class StablePlacementEnv:
        def __getattr__(self, name: str):
            return getattr(inner_env, name)

        def reset(self, **kwargs):
            return inner_env.reset(**kwargs)

        def step(self, action: np.ndarray):
            transition = inner_env.step(action)
            return replace(
                transition,
                info={**transition.info, "stable_placement": True},
            )

    monkeypatch.setattr(
        physics_evaluate_module,
        "_make_env",
        lambda _config, *, randomized=False: StablePlacementEnv(),
    )

    result = rollout_physics_policy(
        policy_name="graph_seed_0",
        representation="graph",
        model_seed=0,
        policy=ConstantPhysicalPolicy(np.zeros(7, dtype=np.float32)),
        statistics=_unit_physics_statistics(),
        config=config,
        case=_one_step_case(),
        device="cpu",
        ik_projection=False,
    )

    assert result.placement is True
    assert result.post_placement_reclose is True


def test_rollout_does_not_count_reclose_on_first_placement_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/physics_smoke_macos.yaml")
    config = replace(config, eval=replace(config.eval, max_steps=1))
    inner_env = physics_evaluate_module._make_env(config)

    class FirstStepPlacementEnv:
        def __getattr__(self, name: str):
            return getattr(inner_env, name)

        def reset(self, **kwargs):
            return inner_env.reset(**kwargs)

        def step(self, action: np.ndarray):
            transition = inner_env.step(action)
            return replace(
                transition,
                info={**transition.info, "stable_placement": True},
            )

    monkeypatch.setattr(
        physics_evaluate_module,
        "_make_env",
        lambda _config, *, randomized=False: FirstStepPlacementEnv(),
    )

    result = rollout_physics_policy(
        policy_name="graph_seed_0",
        representation="graph",
        model_seed=0,
        policy=ConstantPhysicalPolicy(np.zeros(7, dtype=np.float32)),
        statistics=_unit_physics_statistics(),
        config=config,
        case=_one_step_case(),
        device="cpu",
        ik_projection=False,
    )

    assert result.placement is True
    assert result.post_placement_reclose is False


@pytest.mark.parametrize(
    ("include_edge_shuffle", "expected_variants"),
    [
        (False, ["flat", "graph"]),
        (True, ["flat", "graph", "graph_edge_shuffle"]),
    ],
)
def test_subset_evaluation_reports_scope_and_rollout_progress(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    include_edge_shuffle: bool,
    expected_variants: list[str],
) -> None:
    config = replace(
        load_config("configs/physics_smoke_macos.yaml"),
        output_dir=str(tmp_path),
        data_dir=str(tmp_path / "data"),
    )
    cases = (
        PhysicsEvaluationCase(
            case_id="id_normal:2:11",
            seed=11,
            object_count=2,
            condition="id_normal",
            layout_mode="normal",
        ),
        PhysicsEvaluationCase(
            case_id="crowded_ood:4:12",
            seed=12,
            object_count=4,
            condition="crowded_ood",
            layout_mode="crowded",
        ),
    )
    progress_instances = []
    rollout_calls: list[dict[str, object]] = []

    class ProgressSpy:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.updates = 0
            self.postfixes = []
            self.closed = False

        def update(self, amount: int) -> None:
            self.updates += amount

        def set_postfix(self, **kwargs) -> None:
            self.postfixes.append(kwargs)

        def close(self) -> None:
            self.closed = True

    def fake_tqdm(**kwargs):
        progress = ProgressSpy(**kwargs)
        progress_instances.append(progress)
        return progress

    def fake_rollout(**kwargs) -> PhysicsEpisodeResult:
        rollout_calls.append(kwargs)
        case = kwargs["case"]
        edge_shuffle = bool(kwargs.get("edge_shuffle", False))
        return PhysicsEpisodeResult(
            policy=kwargs["policy_name"],
            representation=kwargs["representation"],
            model_seed=kwargs["model_seed"],
            ablation="edge_shuffle" if edge_shuffle else "none",
            case_id=case.case_id,
            seed=case.seed,
            object_count=case.object_count,
            condition=case.condition,
            layout_mode=case.layout_mode,
            success=not edge_shuffle,
            bilateral_contact=False,
            stable_lift=False,
            wrong_object_stable_grasp=False,
            dropped=False,
            placement=False,
            ik_failure=False,
            physics_failure=False,
            steps=1,
            termination_reason="timeout",
            physics_hash=f"physics-{case.case_id}",
            initial_state_hash=f"state-{case.case_id}",
        )

    monkeypatch.setattr(physics_evaluate_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        physics_evaluate_module,
        "make_physics_evaluation_cases",
        lambda _config, *, episodes_per_count=None: cases,
    )
    monkeypatch.setattr(
        physics_data_module,
        "expert_gate_provenance",
        lambda *_args, **_kwargs: {"expert_gate_hash": "gate"},
    )
    monkeypatch.setattr(
        physics_evaluate_module,
        "resolve_training_data",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        physics_evaluate_module,
        "build_training_provenance",
        lambda *_args, **_kwargs: ({"dataset_content_hash": "dataset"}, ()),
    )
    monkeypatch.setattr(
        physics_evaluate_module,
        "preload_evaluation_checkpoints",
        lambda *_args, **_kwargs: {
            (0, "flat"): (object(), object()),
            (0, "graph"): (object(), object()),
        },
    )
    monkeypatch.setattr(physics_evaluate_module, "rollout_physics_policy", fake_rollout)
    monkeypatch.setattr(physics_evaluate_module, "tqdm", fake_tqdm, raising=False)

    default_report = tmp_path / "evaluation" / "report.json"
    default_report.parent.mkdir(parents=True)
    custom_report = tmp_path / "evaluation" / "id_sanity_report.json"
    output = None
    if include_edge_shuffle:
        default_report.write_text("official full report", encoding="utf-8")
        output = custom_report

    report_path = evaluate_from_config(
        "ignored.yaml",
        model_seeds=(0,),
        include_edge_shuffle=include_edge_shuffle,
        episodes_per_count=5,
        conditions=("id_normal",),
        output=output,
        ik_projection=False,
        show_progress=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(progress_instances) == 1
    progress = progress_instances[0]
    expected_rollouts = len(expected_variants)
    assert progress.kwargs["total"] == expected_rollouts
    assert progress.updates == expected_rollouts
    assert progress.closed
    assert {postfix["policy"] for postfix in progress.postfixes} == set(
        expected_variants
    )
    assert {call["case"].condition for call in rollout_calls} == {"id_normal"}
    assert all(call["ik_projection"] is False for call in rollout_calls)
    if include_edge_shuffle:
        assert report_path == custom_report
        assert (tmp_path / "evaluation" / "id_sanity_report_episodes.csv").is_file()
        assert default_report.read_text(encoding="utf-8") == "official full report"
    else:
        assert report_path == default_report
        assert (tmp_path / "evaluation" / "episodes.csv").is_file()
    assert report["evaluation_scope"] == {
        "model_seeds": [0],
        "episodes_per_count": 5,
        "case_count": 1,
        "rollout_count": expected_rollouts,
        "policy_variants": expected_variants,
        "conditions": ["id_normal"],
        "ik_projection_enabled": False,
        "ik_projection_scales": [1.0, 0.5, 0.25, 0.125, 0.0],
    }
    if include_edge_shuffle:
        assert set(report["graph_vs_edge_shuffle"]["by_model_seed"]) == {"0"}
    else:
        assert report["graph_vs_edge_shuffle"] == {"by_model_seed": {}}
