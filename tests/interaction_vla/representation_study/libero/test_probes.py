import numpy as np

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
    _shortcut_baseline_metrics,
    factor_target,
    partition_indices,
)

from .helpers import make_record


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
