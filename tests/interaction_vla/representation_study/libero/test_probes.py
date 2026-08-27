from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from interaction_vla.representation_study.libero.probes import (
    FACTOR_NAMES,
    STUDY_STAGES,
    STUDY_TAPS,
    build_stage_tap_factor_grid,
    classification_metrics,
    constant_classification_baseline,
    geometry_metrics,
    run_linear_probe,
)
from interaction_vla.representation_study.libero.probe_runner import (
    _accessibility_decision,
    _assert_stage_invariant_probe_results,
    _bootstrap_ci,
    _matched_probe_seed,
    _paired_stage_delta,
    _probe_artifact_root,
    _replicated_bootstrap_ci,
    _run_cell,
    _shortcut_baseline_metrics,
    factor_target,
    partition_indices,
)

from .helpers import make_record
from interaction_vla.representation_study.libero.config import load_libero_study_config
from interaction_vla.representation_study.libero.splits import SplitManifest


def test_classification_metrics_include_macro_f1_balanced_accuracy_and_auprc() -> None:
    target = np.asarray([0, 0, 0, 1, 1, 1])
    prediction = np.asarray([0, 0, 1, 1, 1, 0])
    score = np.asarray([0.1, 0.2, 0.6, 0.9, 0.8, 0.4])
    metrics = classification_metrics(target, prediction, score=score, binary=True)
    assert 0 < metrics["macro_f1"] < 1
    assert 0 < metrics["balanced_accuracy"] < 1
    assert 0 < metrics["auprc"] <= 1


def test_constant_baseline_is_explicit_and_geometry_is_normalized() -> None:
    target = np.asarray([0, 0, 1, 2])
    baseline = constant_classification_baseline(target)
    assert np.array_equal(baseline, [0, 0, 0, 0])
    geometry = geometry_metrics(
        np.asarray([[0.0, 0.0], [2.0, 4.0]]),
        np.asarray([[0.0, 0.0], [1.0, 2.0]]),
        normalization_scale=np.asarray([2.0, 4.0]),
    )
    assert geometry["normalized_mae"] == 0.25
    assert geometry["r2"] == 0.5


def test_linear_probe_is_deterministic_and_beats_majority_on_separable_data() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 4)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.int64)
    train = np.arange(80)
    validation = np.arange(80, 100)
    test = np.arange(100, 120)
    first = run_linear_probe(
        x,
        y,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        task="classification",
        seed=11,
        l2_grid=(0.0, 1e-3),
        epochs=100,
    )
    second = run_linear_probe(
        x,
        y,
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        task="classification",
        seed=11,
        l2_grid=(0.0, 1e-3),
        epochs=100,
    )
    assert first["test_metrics"] == second["test_metrics"]
    assert first["test_metrics"]["balanced_accuracy"] > first["baseline_metrics"]["balanced_accuracy"]


def test_linear_probe_macro_f1_uses_the_complete_training_class_universe() -> None:
    features = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 3.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 0])
    result = run_linear_probe(
        features,
        targets,
        train_indices=np.arange(0, 6),
        validation_indices=np.arange(6, 9),
        test_indices=np.arange(9, 11),
        task="classification",
        seed=3,
        l2_grid=(0.0,),
        epochs=100,
    )

    assert result["test_metrics"]["accuracy"] == 1.0
    assert result["test_metrics"]["macro_f1"] == pytest.approx(1.0 / 3.0)


def test_report_grid_never_confuses_missing_experiments_with_results() -> None:
    rows = build_stage_tap_factor_grid(())
    assert len(rows) == len(STUDY_STAGES) * len(STUDY_TAPS) * len(FACTOR_NAMES)
    assert all(row["status"] == "not_run" for row in rows)
    assert all(row["accessible"] is None for row in rows)


def test_factor_targets_use_one_fixed_interaction_vocabulary() -> None:
    record = make_record(task_id=2, episode=3, frame=4, contact=True, stable=True)
    assert factor_target(record, "entity") == "object_2"
    assert factor_target(record, "contact") is True
    assert factor_target(record, "stable_grasp") is True
    assert factor_target(record, "phase") == "approach"
    assert factor_target(record, "next_relation") == (
        "gripper",
        "near",
        "target",
        "establish",
    )
    geometry = factor_target(record, "geometry")
    assert isinstance(geometry, np.ndarray)
    assert geometry.shape == (20,)


def test_partition_indices_preserve_state_bank_group_assignments() -> None:
    records = tuple(make_record(task_id=task, episode=task, frame=0) for task in range(3))
    assignments = {
        records[0].state_id: "train",
        records[1].state_id: "validation",
        records[2].state_id: "test",
    }
    partitions = partition_indices(records, assignments, applicable=np.ones(3, dtype=bool))
    assert partitions == {"train": [0], "validation": [1], "test": [2]}


def test_shortcut_controls_include_instruction_task_and_time() -> None:
    records = tuple(
        make_record(task_id=task, episode=episode, frame=frame, phase=("approach", "place")[frame])
        for task in range(2)
        for episode in range(3)
        for frame in range(2)
    )
    source_indices = np.arange(len(records), dtype=np.int64)
    targets = np.asarray([record.labels.phase for record in records])
    local_parts = {
        "train": list(range(0, 8)),
        "validation": [8, 9],
        "test": [10, 11],
    }
    baselines = _shortcut_baseline_metrics(
        records=records,
        source_indices=source_indices,
        targets=targets,
        local_parts=local_parts,
        factor="phase",
    )
    assert set(baselines) == {"task_id", "instruction", "normalized_time_bin"}


def test_accessibility_requires_interval_to_clear_strongest_baseline() -> None:
    assert _accessibility_decision(
        confidence_interval={"low": 0.71, "high": 0.82},
        threshold=0.70,
        higher_is_better=True,
    ) is True
    assert _accessibility_decision(
        confidence_interval={"low": 0.65, "high": 0.82},
        threshold=0.70,
        higher_is_better=True,
    ) is False
    assert _accessibility_decision(
        confidence_interval={"low": None, "high": None},
        threshold=0.70,
        higher_is_better=True,
    ) is None


def test_classification_bootstrap_uses_target_prediction_union_as_label_universe() -> None:
    interval = _bootstrap_ci(
        factor="phase",
        target=np.asarray(["approach", "approach", "approach", "approach"]),
        prediction=np.asarray(["approach", "transport", "approach", "transport"]),
        score=None,
        clusters=("episode_0", "episode_0", "episode_1", "episode_1"),
        samples=50,
        confidence=0.95,
        seed=9,
    )

    assert interval["samples"] == 50
    assert interval["low"] is not None
    assert interval["high"] is not None


def test_probe_seed_is_stage_invariant_and_identity_specific() -> None:
    first = _matched_probe_seed(
        base_seed=17,
        tap="vision_output",
        factor="contact",
        split_name="task_group",
        replicate_offset=0,
    )
    repeated = _matched_probe_seed(
        base_seed=17,
        tap="vision_output",
        factor="contact",
        split_name="task_group",
        replicate_offset=0,
    )
    different_replicate = _matched_probe_seed(
        base_seed=17,
        tap="vision_output",
        factor="contact",
        split_name="task_group",
        replicate_offset=1,
    )
    different_factor = _matched_probe_seed(
        base_seed=17,
        tap="vision_output",
        factor="phase",
        split_name="task_group",
        replicate_offset=0,
    )

    assert first == repeated
    assert len({first, different_replicate, different_factor}) == 3


def _paired_row(*, prediction: list[int], factor: str = "phase") -> dict[str, object]:
    return {
        "status": "complete",
        "primary_metric_name": "macro_f1" if factor == "phase" else "normalized_mae",
        "paired_payload": {
            "state_ids": ["s0", "s1", "s2", "s3"],
            "clusters": ["episode_0", "episode_0", "episode_1", "episode_1"],
            "target": [0, 0, 1, 1] if factor == "phase" else [[0.0], [0.0], [1.0], [1.0]],
            "normalization_scale": None if factor == "phase" else [1.0],
            "replicates": [
                {
                    "seed": 101,
                    "prediction": prediction,
                    "score": None,
                }
            ],
        },
    }


def test_paired_stage_delta_uses_matched_states_groups_and_probe_seeds() -> None:
    delta = _paired_stage_delta(
        factor="phase",
        reference=_paired_row(prediction=[0, 0, 0, 0]),
        destination=_paired_row(prediction=[0, 0, 1, 1]),
        samples=100,
        confidence=0.95,
        seed=4,
    )

    assert delta["status"] == "complete"
    assert delta["metric"] == "macro_f1"
    assert delta["higher_is_better"] is True
    assert delta["destination_minus_reference"] == pytest.approx(2.0 / 3.0)
    assert delta["improvement"] == pytest.approx(2.0 / 3.0)
    assert delta["delta_low"] == delta["improvement_low"]
    assert delta["delta_high"] == delta["improvement_high"]
    assert delta["groups"] == 2


def test_paired_geometry_delta_flips_sign_for_improvement() -> None:
    reference = _paired_row(prediction=[[0.5], [0.5], [0.5], [0.5]], factor="geometry")
    destination = _paired_row(prediction=[[0.0], [0.0], [1.0], [1.0]], factor="geometry")
    delta = _paired_stage_delta(
        factor="geometry",
        reference=reference,
        destination=destination,
        samples=50,
        confidence=0.95,
        seed=4,
    )

    assert delta["higher_is_better"] is False
    assert delta["destination_minus_reference"] == pytest.approx(-0.5)
    assert delta["improvement"] == pytest.approx(0.5)
    assert delta["improvement_low"] == pytest.approx(-delta["delta_high"])
    assert delta["improvement_high"] == pytest.approx(-delta["delta_low"])


def test_binary_bootstrap_fails_gate_when_valid_resamples_are_too_sparse() -> None:
    interval = _replicated_bootstrap_ci(
        factor="contact",
        target=np.asarray([1, 1, 0, 0]),
        predictions=(np.asarray([1, 1, 0, 0]),),
        scores=(np.asarray([0.9, 0.8, 0.2, 0.1]),),
        clusters=("positive_only", "positive_only", "negative_only", "negative_only"),
        samples=400,
        confidence=0.95,
        seed=5,
        normalization_scale=None,
        labels=np.asarray([0, 1]),
        minimum_valid_rate=0.9,
    )

    assert interval["requested_samples"] == 400
    assert interval["valid_rate"] < 0.9
    assert interval["low"] is None
    assert interval["high"] is None
    assert "valid rate" in interval["gate_reason"]


def test_identical_latents_require_identical_probe_results() -> None:
    row = _paired_row(prediction=[0, 0, 1, 1])
    _assert_stage_invariant_probe_results(row, dict(row))

    changed = _paired_row(prediction=[0, 0, 0, 0])
    with pytest.raises(ValueError, match="identical latent matrices produced different"):
        _assert_stage_invariant_probe_results(row, changed)


def test_probe_v2_artifacts_use_a_versioned_root() -> None:
    root = _probe_artifact_root(Path("outputs/representation_study/example"))
    assert root == Path("outputs/representation_study/example/probes/protocol_v2")


def test_run_cell_uses_all_matched_probe_seeds_and_emits_pairing_payload() -> None:
    config = load_libero_study_config(
        "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    )
    config = replace(
        config,
        probes=replace(config.probes, matched_seed_offsets=(0, 1), linear_epochs=20),
    )
    partitions = ("train", "train", "validation", "validation", "test", "test")
    records = tuple(
        make_record(
            task_id=task,
            episode=task,
            frame=frame,
            phase=("approach", "place")[frame],
        )
        for task in range(6)
        for frame in range(2)
    )
    assignments = {
        record.state_id: partitions[record.task_id] for record in records
    }
    split = SplitManifest(
        name="task_group",
        group_unit="task",
        seed=3,
        ratios=(1 / 3, 1 / 3, 1 / 3),
        assignments=assignments,
    )
    features = np.asarray(
        [[-1.0, float(record.task_id)] if record.labels.phase == "approach" else [1.0, float(record.task_id)] for record in records],
        dtype=np.float32,
    )

    result = _run_cell(
        config=config,
        records=records,
        features=features,
        split=split,
        split_name="task_group",
        tap="vision_output",
        factor="phase",
        run_capacity_check=False,
    )

    expected_seeds = [
        _matched_probe_seed(
            base_seed=config.seed,
            tap="vision_output",
            factor="phase",
            split_name="task_group",
            replicate_offset=offset,
        )
        for offset in (0, 1)
    ]
    assert result["probe_seeds"] == expected_seeds
    assert len(result["linear_replicates"]) == 2
    assert [item["seed"] for item in result["paired_payload"]["replicates"]] == expected_seeds
    assert result["paired_payload"]["state_ids"] == [
        record.state_id for record in records if assignments[record.state_id] == "test"
    ]


def test_run_cell_propagates_sparse_binary_bootstrap_to_failed_gate() -> None:
    config = load_libero_study_config(
        "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    )
    config = replace(config, probes=replace(config.probes, linear_epochs=20))
    partitions = ("train", "train", "validation", "validation", "test", "test")
    records = tuple(
        make_record(
            task_id=task,
            episode=task,
            frame=frame,
            contact=task % 2 == 0,
        )
        for task in range(6)
        for frame in range(2)
    )
    assignments = {
        record.state_id: partitions[record.task_id] for record in records
    }
    split = SplitManifest(
        name="task_group",
        group_unit="task",
        seed=3,
        ratios=(1 / 3, 1 / 3, 1 / 3),
        assignments=assignments,
    )
    features = np.asarray(
        [[float(record.labels.contact.gripper_target), float(record.frame_index)] for record in records],
        dtype=np.float32,
    )

    result = _run_cell(
        config=config,
        records=records,
        features=features,
        split=split,
        split_name="task_group",
        tap="vision_output",
        factor="contact",
        run_capacity_check=False,
    )

    assert result["status"] == "failed_gate"
    assert result["primary_metric"] is not None
    assert result["accessible"] is None
    assert "valid rate" in result["reason"]
