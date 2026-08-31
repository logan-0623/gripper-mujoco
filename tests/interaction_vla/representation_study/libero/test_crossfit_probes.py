from collections import Counter
from dataclasses import replace
import json

import numpy as np
import pytest
import interaction_vla.representation_study.libero.crossfit_probes as crossfit_module

from interaction_vla.representation_study.libero.crossfit_probes import (
    CONDITION_CONTRASTS,
    CROSSFIT_REPORT_SCHEMA,
    _current_protocol_binding,
    _file_sha256,
    _bootstrap_accessibility,
    _paired_delta_gate,
    _time_bins,
    _fold_baseline_predictions,
    _write_immutable_json,
    build_crossfit_manifest,
    build_crossfit_report_grid,
    crossfit_partition_indices,
    inspect_crossfit_probe_report,
    paired_crossfit_delta,
    run_crossfit_cell,
    run_crossfit_probe_study,
)
from interaction_vla.representation_study.libero.longitudinal import CONDITION_SPECS
from interaction_vla.representation_study.libero.probes import FACTOR_NAMES, STUDY_TAPS
from interaction_vla.representation_study.libero.config import load_libero_study_config

from .helpers import make_record


def test_task_crossfit_keeps_groups_intact_and_tests_each_group_once() -> None:
    records = tuple(
        make_record(task_id=task, episode=episode, frame=frame)
        for task in range(9)
        for episode in range(2)
        for frame in range(2)
    )
    manifest = build_crossfit_manifest(
        records, split_name="task_group", folds=3, seed=17
    )

    tested: Counter[str] = Counter()
    for fold in range(3):
        parts = crossfit_partition_indices(
            records, manifest, fold=fold, applicable=np.ones(len(records), dtype=bool)
        )
        groups = {
            name: {
                f"{records[index].suite}:{records[index].task_id}"
                for index in indices
            }
            for name, indices in parts.items()
        }
        assert not groups["train"] & groups["validation"]
        assert not groups["train"] & groups["test"]
        assert not groups["validation"] & groups["test"]
        tested.update(groups["test"])

    assert set(tested.values()) == {1}
    assert len(tested) == 9


def test_episode_crossfit_is_blocked_within_task_and_has_no_episode_leakage() -> None:
    records = tuple(
        make_record(task_id=task, episode=episode, frame=frame)
        for task in range(4)
        for episode in range(3)
        for frame in range(2)
    )
    manifest = build_crossfit_manifest(
        records, split_name="episode_group", folds=3, seed=23
    )

    tested: Counter[str] = Counter()
    for fold in range(3):
        parts = crossfit_partition_indices(
            records, manifest, fold=fold, applicable=np.ones(len(records), dtype=bool)
        )
        episode_sets = {
            name: {
                f"{records[index].suite}:{records[index].task_id}:"
                f"{records[index].source_episode_id}"
                for index in indices
            }
            for name, indices in parts.items()
        }
        assert not episode_sets["train"] & episode_sets["validation"]
        assert not episode_sets["train"] & episode_sets["test"]
        assert not episode_sets["validation"] & episode_sets["test"]
        test_tasks = {records[index].task_id for index in parts["test"]}
        assert test_tasks == {0, 1, 2, 3}
        tested.update(episode_sets["test"])

    assert set(tested.values()) == {1}
    assert len(tested) == 12


def _smoke_probe_config():
    config = load_libero_study_config(
        "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    )
    return replace(
        config,
        probes=replace(
            config.probes,
            bootstrap_samples=50,
            matched_seed_offsets=(0,),
            linear_l2=(0.0,),
            linear_epochs=30,
        ),
    )


def test_crossfit_cell_emits_one_out_of_fold_prediction_per_applicable_state() -> None:
    records = tuple(
        make_record(
            task_id=task,
            episode=task,
            frame=frame,
            phase=("approach", "place")[frame],
        )
        for task in range(9)
        for frame in range(2)
    )
    features = np.asarray(
        [[-2.0, float(record.task_id)] if record.labels.phase == "approach" else [2.0, float(record.task_id)] for record in records],
        dtype=np.float32,
    )
    manifest = build_crossfit_manifest(
        records, split_name="task_group", folds=3, seed=17
    )

    row = run_crossfit_cell(
        config=_smoke_probe_config(),
        records=records,
        features=features,
        manifest=manifest,
        tap="pre_action",
        factor="phase",
    )

    assert row["status"] == "complete"
    assert row["folds_completed"] == 3
    assert row["selection_metric"] == "macro_f1"
    assert row["probe_device"] in {"cpu", "cuda"}
    assert row["oof_states"] == len(records)
    assert len(set(row["paired_payload"]["state_ids"])) == len(records)
    assert row["primary_metric"] > row["baseline_metric"]
    assert row["accessibility_utility"] == pytest.approx(
        row["primary_metric"] - row["accessibility_threshold"]
    )
    assert set(row["shortcut_baselines"]) == {
        "task_id",
        "instruction",
        "normalized_time_bin",
    }


def test_crossfit_cell_marks_unseen_task_bound_entity_classes_not_estimable() -> None:
    records = tuple(
        make_record(task_id=task, episode=task, frame=frame)
        for task in range(6)
        for frame in range(2)
    )
    features = np.eye(len(records), dtype=np.float32)
    manifest = build_crossfit_manifest(
        records, split_name="task_group", folds=3, seed=17
    )

    row = run_crossfit_cell(
        config=_smoke_probe_config(),
        records=records,
        features=features,
        manifest=manifest,
        tap="vision_output",
        factor="entity",
    )

    assert row["status"] == "not_estimable"
    assert row["primary_metric"] is None
    assert "absent from training" in row["reason"]


def _geometry_payload(prediction: list[list[float]]) -> dict[str, object]:
    return {
        "status": "complete",
        "paired_payload": {
            "state_ids": ["s0", "s1", "s2", "s3"],
            "clusters": ["task0", "task0", "task1", "task1"],
            "target": [[0.0], [0.0], [1.0], [1.0]],
            "row_normalization_scale": [[1.0], [1.0], [1.0], [1.0]],
            "labels": None,
            "replicates": [
                {"seed_offset": 0, "prediction": prediction, "score": None}
            ],
        },
    }


def test_paired_crossfit_geometry_delta_flips_error_sign_for_improvement() -> None:
    delta = paired_crossfit_delta(
        factor="geometry",
        reference=_geometry_payload([[0.5], [0.5], [0.5], [0.5]]),
        destination=_geometry_payload([[0.0], [0.0], [1.0], [1.0]]),
        samples=50,
        confidence=0.95,
        minimum_valid_rate=0.9,
        seed=9,
    )

    assert delta["status"] == "complete"
    assert delta["destination_minus_reference"] == pytest.approx(-0.5)
    assert delta["improvement"] == pytest.approx(0.5)


def test_geometry_crossfit_records_every_fold_seed_for_every_offset() -> None:
    config = replace(
        _smoke_probe_config(),
        probes=replace(_smoke_probe_config().probes, matched_seed_offsets=(0, 1)),
    )
    records = tuple(
        make_record(task_id=task, episode=task, frame=frame)
        for task in range(6)
        for frame in range(2)
    )
    features = np.asarray(
        [[float(record.task_id), float(record.frame_index)] for record in records],
        dtype=np.float32,
    )
    row = run_crossfit_cell(
        config=config,
        records=records,
        features=features,
        manifest=build_crossfit_manifest(
            records, split_name="task_group", folds=3, seed=17
        ),
        tap="pre_action",
        factor="geometry",
    )

    assert {key: len(value) for key, value in row["fold_seeds"].items()} == {
        "0": 3,
        "1": 3,
    }


def test_binary_shortcut_uses_the_group_negative_class_below_half() -> None:
    records = tuple(
        replace(
            make_record(task_id=index, episode=index, frame=0),
            language="shared" if index < 3 or index == 7 else f"other-{index}",
        )
        for index in range(8)
    )
    targets = np.asarray([False, False, True, True, True, True, True, False])
    result = _fold_baseline_predictions(
        records=records,
        source_indices=np.arange(len(records)),
        targets=targets,
        local_parts={"train": list(range(7)), "validation": [], "test": [7]},
        factor="contact",
    )

    assert result["instruction"][0].tolist() == [False]


def test_normalized_time_bins_use_only_training_fold_scale() -> None:
    records = tuple(
        make_record(task_id=index, episode=index, frame=frame)
        for index, frame in enumerate((0, 9, 5, 100))
    )
    bins = _time_bins(records, np.arange(4), np.asarray([0, 1]))

    assert bins.tolist() == [0, 9, 5, 9]


def test_crossfit_report_grid_has_every_condition_tap_factor_cell() -> None:
    rows = build_crossfit_report_grid([], split_name="task_group")

    assert len(rows) == len(CONDITION_SPECS) * len(STUDY_TAPS) * len(FACTOR_NAMES)
    assert {
        (row["condition"], row["tap"], row["factor"])
        for row in rows
    } == {
        (spec.condition, tap, factor)
        for spec in CONDITION_SPECS
        for tap in STUDY_TAPS
        for factor in FACTOR_NAMES
    }
    assert {row["status"] for row in rows} == {"not_run"}
    assert CONDITION_CONTRASTS == (
        ("pretrained", "d100_u16617", "early_d100"),
        ("d25_u16070", "d50_u16324", "data_25_to_50_at_16k"),
        ("d50_u16324", "d100_u16617", "data_50_to_100_at_16k"),
        ("d50_u32650", "d100_u33234", "data_50_to_100_at_32k"),
        ("d100_u16617", "d100_u33234", "d100_16_to_32k"),
        ("d100_u33234", "d100_u49851", "d100_32_to_50k"),
        ("d100_u49851", "d100_u66470", "d100_50_to_66k"),
    )


def test_accessibility_bootstrap_reselects_strongest_shortcut_per_resample() -> None:
    target = np.zeros((4, 1), dtype=np.float64)
    _, utility = _bootstrap_accessibility(
        factor="geometry",
        target=target,
        predictions=[np.full((4, 1), 0.4)],
        scores=[None],
        baselines={
            "left": (np.asarray([[0.0], [0.0], [1.0], [1.0]]), None),
            "right": (np.asarray([[1.0], [1.0], [0.0], [0.0]]), None),
        },
        clusters=("a", "a", "b", "b"),
        labels=None,
        row_scale=np.ones((4, 1)),
        samples=1000,
        confidence=0.95,
        minimum_valid_rate=0.9,
        seed=3,
    )

    assert utility["point"] == pytest.approx(0.1)
    assert utility["high"] <= 0.1000001


def test_crossfit_cells_are_immutable_under_changed_binding(tmp_path) -> None:
    path = tmp_path / "cell.json"
    _write_immutable_json(path, {"binding": {"condition": "pretrained"}})

    _write_immutable_json(path, {"binding": {"condition": "pretrained"}})
    with pytest.raises(FileExistsError, match="scientific binding changed"):
        _write_immutable_json(path, {"binding": {"condition": "d25_u16070"}})
    assert json.loads(path.read_text())["binding"]["condition"] == "pretrained"


def test_crossfit_report_inspection_rejects_a_stale_latent_gate(tmp_path) -> None:
    config = replace(
        _smoke_probe_config(),
        output_dir=tmp_path / "study",
        source_path=tmp_path / "config.yaml",
    )
    paths = (
        config.output_dir / "state_bank/manifest.json",
        config.output_dir / "protocol_v3/conditions/manifest.json",
        config.output_dir / "protocol_v3/latent_gate/report.json",
        config.output_dir / "timelines/report.json",
        config.source_path,
    )
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"index": index}))
    root = config.output_dir / "protocol_v3/probes/crossfit_v1"
    cell = root / "cells/cell.json.gz"
    cell.parent.mkdir(parents=True, exist_ok=True)
    cell.write_bytes(b"cell")
    folds = root / "folds.json"
    folds.write_text("{}")
    rows = [
        {
            **row,
            "cell_artifact": str(cell),
            "cell_artifact_sha256": _file_sha256(cell),
        }
        for row in build_crossfit_report_grid([], split_name="task_group")
    ]
    episode_rows = [
        {**row, "split": "episode_group"}
        for row in rows
    ]
    report = {
        "schema_version": CROSSFIT_REPORT_SCHEMA,
        "passed": True,
        "protocol_binding": {
            **_current_protocol_binding(config),
            "folds_sha256": _file_sha256(folds),
        },
        "conditions": [spec.condition for spec in CONDITION_SPECS],
        "taps": list(STUDY_TAPS),
        "factors": list(FACTOR_NAMES),
        "splits": ["task_group", "episode_group"],
        "status_counts": {"complete": len(rows) * 2, "not_estimable": 0, "failed_gate": 0, "not_run": 0},
        "grids": {"task_group": rows, "episode_group": episode_rows},
        "paired_deltas": [],
        "paired_delta_gate": {"passed": True, "failures": []},
        "identical_latent_sanity": {"passed": True, "failures": []},
        "interpretation_boundary": {},
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report))

    assert inspect_crossfit_probe_report(config)["passed"] is True
    paths[2].write_text('{"changed": true}')
    with pytest.raises(ValueError, match="stale binding"):
        inspect_crossfit_probe_report(config)


def test_crossfit_study_writes_resumes_and_rejects_stale_cache_binding(
    tmp_path, monkeypatch
) -> None:
    config = replace(
        _smoke_probe_config(),
        output_dir=tmp_path / "study",
        source_path=tmp_path / "config.yaml",
    )
    records = tuple(
        make_record(task_id=task, episode=task * 10 + episode, frame=frame)
        for task in range(3)
        for episode in range(3)
        for frame in range(2)
    )
    for path in (
        config.output_dir / "state_bank/manifest.json",
        config.output_dir / "protocol_v3/conditions/manifest.json",
        config.output_dir / "protocol_v3/latent_gate/report.json",
        config.output_dir / "timelines/report.json",
        config.source_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    bank_hash = _file_sha256(config.output_dir / "state_bank/manifest.json")
    plan = {
        "conditions": [
            {
                "condition": spec.condition,
                "stage": spec.stage,
                "data_fraction": spec.data_fraction,
                "step": spec.step,
                "checkpoint_sha256": str(index) * 64,
            }
            for index, spec in enumerate(CONDITION_SPECS, start=1)
        ]
    }
    for spec in CONDITION_SPECS:
        for tap in STUDY_TAPS:
            path = config.output_dir / "protocol_v3/latents" / spec.condition / tap / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"condition": spec.condition, "tap": tap}))

    monkeypatch.setattr(crossfit_module, "validate_annotation_timeline_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(crossfit_module, "load_state_bank", lambda _root: (records, {}, None, None))
    monkeypatch.setattr(crossfit_module, "load_longitudinal_plan", lambda _config: plan)
    monkeypatch.setattr(
        crossfit_module,
        "inspect_longitudinal_latents",
        lambda _config: {"passed": True, "identical_cache_pairs": []},
    )

    def load_cache(_root):
        return (
            tuple(record.state_id for record in records),
            np.ones((len(records), 2), dtype=np.float32),
            {"values_sha256": "f" * 64, "state_bank_sha256": bank_hash},
        )

    monkeypatch.setattr(crossfit_module, "load_latent_cache", load_cache)
    calls = 0

    def run_cell(**_kwargs):
        nonlocal calls
        calls += 1
        return {
            "status": "complete",
            "accessible": True,
            "primary_metric": 1.0,
            "paired_payload": {},
        }

    monkeypatch.setattr(crossfit_module, "run_crossfit_cell", run_cell)
    monkeypatch.setattr(
        crossfit_module,
        "paired_crossfit_delta",
        lambda **_kwargs: {"status": "complete"},
    )

    first = run_crossfit_probe_study(config)
    assert first["passed"] is True
    assert first["cells"] == 384
    assert calls == len(STUDY_TAPS) * 2 * len(FACTOR_NAMES)
    run_crossfit_probe_study(config)
    assert calls == len(STUDY_TAPS) * 2 * len(FACTOR_NAMES)

    monkeypatch.setattr(
        crossfit_module,
        "load_latent_cache",
        lambda _root: (
            tuple(record.state_id for record in records),
            np.ones((len(records), 2), dtype=np.float32),
            {"values_sha256": "f" * 64, "state_bank_sha256": "0" * 64},
        ),
    )
    with pytest.raises(ValueError, match="State Bank binding is stale"):
        run_crossfit_probe_study(config)
    monkeypatch.setattr(crossfit_module, "load_latent_cache", load_cache)

    stale = config.output_dir / "protocol_v3/latents/pretrained/vision_output/manifest.json"
    stale.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="stale binding"):
        run_crossfit_probe_study(config)


def test_paired_delta_gate_requires_available_deltas_for_complete_cells() -> None:
    keys = {
        (condition, "pre_action", "task_group", "phase"): {
            "result": {"status": status}
        }
        for condition, status in (("left", "complete"), ("right", "complete"))
    }
    delta = {
        "contrast": "left_to_right",
        "reference": "left",
        "destination": "right",
        "tap": "pre_action",
        "split": "task_group",
        "factor": "phase",
        "status": "not_available",
    }

    assert _paired_delta_gate([delta], keys)["passed"] is False
    keys[("left", "pre_action", "task_group", "phase")]["result"][
        "status"
    ] = "not_estimable"
    assert _paired_delta_gate([delta], keys)["passed"] is True
