from __future__ import annotations

import numpy as np
import pytest
import torch

from interaction_vla.data import collect_episode, save_episode
from interaction_vla.env import KinematicTabletopEnv
from interaction_vla.evaluate import (
    EpisodeResult,
    EvaluationCase,
    aggregate_results,
    compare_training_stages,
    evaluate_offline_action_mse,
    evaluate_policies,
    load_episode_results_csv,
    make_conditioned_evaluation_cases,
    paired_graph_flat_comparison,
    save_evaluation_report,
    shuffle_valid_edge_assignments,
)
from interaction_vla.expert import ScriptedExpert
from interaction_vla.graph.builder import SceneGraphBuilder
from interaction_vla.models.encoders import scene_graphs_to_batch
from interaction_vla.models.policy import build_action_policy
from interaction_vla.train import TrainingStatistics


def test_report_separates_id_and_ood_counts() -> None:
    results = [
        EpisodeResult("flat", 1, 2, "id", True, False, True, False, 20, "success", 0.1),
        EpisodeResult("flat", 2, 4, "ood", False, True, False, False, 8, "wrong_object", 0.3),
    ]

    report = aggregate_results(results)

    assert report["by_object_count"]["4"]["split"] == "ood"
    assert report["overall"]["success_rate"] == 0.5
    assert report["overall"]["wrong_object_rate"] == 0.5
    assert report["overall"]["on_policy_action_mse"] == 0.2
    assert report["by_policy_and_object_count"]["flat"]["2"]["success_rate"] == 1.0
    assert report["by_policy_and_object_count"]["flat"]["4"]["wrong_object_rate"] == 1.0


def test_conditioned_cases_use_distinct_deterministic_namespaces() -> None:
    cases = make_conditioned_evaluation_cases(
        id_object_counts=(2, 3),
        count_ood_object_counts=(4, 5),
        crowded_object_counts=(4, 5),
        episodes_per_count=2,
        base_seed=100,
    )

    assert {case.condition for case in cases} == {
        "id_normal",
        "count_ood",
        "crowded_ood",
    }
    assert all(
        case.layout_mode == "crowded"
        for case in cases
        if case.condition == "crowded_ood"
    )
    assert len({(case.condition, case.seed) for case in cases}) == len(cases)


def test_report_groups_metrics_by_condition() -> None:
    results = [
        EpisodeResult(
            "flat",
            1,
            2,
            "id",
            True,
            False,
            True,
            False,
            20,
            "success",
            0.1,
            condition="id_normal",
        ),
        EpisodeResult(
            "flat",
            2,
            4,
            "ood",
            False,
            True,
            False,
            False,
            8,
            "wrong_object",
            0.3,
            condition="crowded_ood",
            layout_mode="crowded",
        ),
    ]

    report = aggregate_results(results)

    assert report["by_condition"]["crowded_ood"]["wrong_object_rate"] == 1.0
    assert report["by_policy_and_condition"]["flat"]["id_normal"]["success_rate"] == 1.0
    assert (
        report["by_policy_condition_and_object_count"]["flat"]["crowded_ood"]["4"][
            "episodes"
        ]
        == 1
    )


def test_edge_shuffle_changes_only_valid_edge_assignments() -> None:
    env = KinematicTabletopEnv(max_objects=5)
    graph = SceneGraphBuilder(max_objects=5).build(env.reset(seed=3, object_count=3))
    batch = scene_graphs_to_batch((graph,))

    shuffled = shuffle_valid_edge_assignments(batch, seed=7)
    valid = batch.edge_mask[0]

    torch.testing.assert_close(shuffled.edge_index, batch.edge_index)
    torch.testing.assert_close(shuffled.edge_features[~batch.edge_mask], batch.edge_features[~batch.edge_mask])
    original_rows = sorted(map(tuple, batch.edge_features[0, valid].numpy()))
    shuffled_rows = sorted(map(tuple, shuffled.edge_features[0, valid].numpy()))
    assert original_rows == shuffled_rows
    assert not torch.equal(shuffled.edge_features[0, valid], batch.edge_features[0, valid])


def test_two_policies_receive_identical_paired_cases(tmp_path) -> None:
    episode = collect_episode(
        KinematicTabletopEnv(max_objects=5),
        ScriptedExpert(),
        seed=19,
        object_count=2,
    )
    path = save_episode(episode, tmp_path / "episode.npz")
    statistics = TrainingStatistics.fit((path,))
    policies = {
        name: (build_action_policy(representation="proprio", embedding_dim=8, policy_hidden_dim=8), statistics)
        for name in ("first", "second")
    }
    cases = (
        EvaluationCase(seed=101, object_count=2, split="id"),
        EvaluationCase(seed=202, object_count=4, split="ood"),
    )

    results = evaluate_policies(
        policies,
        cases,
        max_objects=5,
        max_steps=1,
        device="cpu",
    )

    first_cases = [(result.seed, result.object_count) for result in results if result.policy == "first"]
    second_cases = [(result.seed, result.object_count) for result in results if result.policy == "second"]
    assert first_cases == second_cases == [(101, 2), (202, 4)]


def test_offline_action_error_is_computed_on_saved_expert_frames(tmp_path) -> None:
    episode = collect_episode(
        KinematicTabletopEnv(max_objects=5),
        ScriptedExpert(),
        seed=29,
        object_count=2,
    )
    path = save_episode(episode, tmp_path / "held_out.npz")
    statistics = TrainingStatistics.fit((path,))
    policy = build_action_policy(
        representation="proprio", embedding_dim=8, policy_hidden_dim=8
    )

    error = evaluate_offline_action_mse(policy, statistics, (path,), device="cpu")

    assert np.isfinite(error)
    assert error >= 0.0


def test_paired_graph_flat_comparison_reports_each_model_seed() -> None:
    results = []
    for model_seed in (0, 1, 2):
        results.extend(
            (
                EpisodeResult(
                    "flat",
                    100,
                    4,
                    "ood",
                    False,
                    False,
                    False,
                    False,
                    20,
                    "timeout",
                    0.2,
                    representation="flat",
                    model_seed=model_seed,
                ),
                EpisodeResult(
                    "graph",
                    100,
                    4,
                    "ood",
                    True,
                    False,
                    True,
                    False,
                    10,
                    "success",
                    0.1,
                    representation="graph",
                    model_seed=model_seed,
                ),
            )
        )

    comparison = paired_graph_flat_comparison(results)

    assert comparison["by_model_seed"]["0"]["ood_success_delta"] == 1.0
    assert comparison["by_model_seed"]["1"]["ood_success_delta"] == 1.0
    assert comparison["by_model_seed"]["0"]["condition_success_delta"]["count_ood"] == 1.0
    assert comparison["criterion"]["ood_gain_at_least_10pp"] is True
    assert comparison["criterion"]["all_model_seeds_improve_ood"] is True
    assert comparison["criterion"]["at_least_three_model_seeds"] is True
    assert comparison["criterion"]["all_criteria_met"] is True


def test_training_stage_comparison_pairs_exact_cases_and_reports_condition_deltas() -> None:
    baseline = []
    recovery = []
    for representation in ("flat", "graph"):
        for case_index in range(20):
            common = {
                "policy": representation,
                "seed": 1000 + case_index,
                "object_count": 4,
                "split": "ood",
                "wrong_object": False,
                "dropped": False,
                "termination_reason": "timeout",
                "on_policy_action_mse": 0.1,
                "representation": representation,
                "model_seed": 0,
                "condition": "crowded_ood",
                "layout_mode": "crowded",
            }
            baseline_success = case_index < 2
            recovery_success = case_index < (4 if representation == "flat" else 7)
            baseline.append(
                EpisodeResult(
                    **common,
                    success=baseline_success,
                    grasped=baseline_success,
                    steps=100,
                )
            )
            recovery.append(
                EpisodeResult(
                    **common,
                    success=recovery_success,
                    grasped=recovery_success,
                    steps=80,
                )
            )

    comparison = compare_training_stages(baseline, recovery)

    assert comparison["graph"]["0"]["crowded_ood_success_delta"] == pytest.approx(0.25)
    assert comparison["flat"]["0"]["crowded_ood_success_delta"] == pytest.approx(0.10)
    assert comparison["graph"]["0"]["crowded_ood_grasp_delta"] == pytest.approx(0.25)
    assert comparison["graph"]["0"]["crowded_ood_steps_delta"] == pytest.approx(-20.0)

    with pytest.raises(ValueError, match="duplicate"):
        compare_training_stages(baseline + [baseline[0]], recovery)
    with pytest.raises(ValueError, match="paired cases"):
        compare_training_stages(baseline, recovery[:-1])


def test_episode_result_csv_loader_restores_explicit_scalar_types(tmp_path) -> None:
    expected = EpisodeResult(
        "graph",
        91,
        5,
        "ood",
        True,
        False,
        True,
        False,
        42,
        "success",
        0.125,
        representation="graph",
        model_seed=2,
        condition="crowded_ood",
        layout_mode="crowded",
    )
    save_evaluation_report((expected,), tmp_path)

    loaded = load_episode_results_csv(tmp_path / "episodes.csv")

    assert loaded == (expected,)
    assert isinstance(loaded[0].success, bool)
    assert isinstance(loaded[0].seed, int)
    assert isinstance(loaded[0].on_policy_action_mse, float)
