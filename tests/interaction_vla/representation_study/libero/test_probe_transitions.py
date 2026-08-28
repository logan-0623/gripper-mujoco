from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path

import interaction_vla.representation_study.libero.probe_transitions as transitions
import pytest
from interaction_vla.representation_study.libero.config import (
    load_libero_study_config,
)
from interaction_vla.representation_study.libero.probe_transitions import (
    ADJACENT_STAGE_PAIRS,
    _load_full_probe_rows,
    build_adjacent_stage_delta_grid,
    enrich_adjacent_stage_deltas,
)
from interaction_vla.representation_study.libero.probes import (
    FACTOR_NAMES,
    STUDY_TAPS,
)


def _config():
    config = load_libero_study_config(
        "configs/representation_study/libero_smolvla_smoke_linux_cuda.yaml"
    )
    return replace(
        config,
        probes=replace(config.probes, bootstrap_samples=50),
    )


def _paired_row(
    *,
    stage: str,
    factor: str,
    prediction: list[object],
) -> dict[str, object]:
    geometry = factor == "geometry"
    return {
        "stage": stage,
        "tap": "action_expert_input",
        "factor": factor,
        "split": "episode_group",
        "status": "complete",
        "paired_payload": {
            "state_ids": ["s0", "s1", "s2", "s3"],
            "clusters": ["episode_0", "episode_0", "episode_1", "episode_1"],
            "target": (
                [[0.0], [0.0], [1.0], [1.0]] if geometry else [0, 0, 1, 1]
            ),
            "normalization_scale": [1.0] if geometry else None,
            "labels": None if geometry else [0, 1],
            "replicates": [
                {
                    "seed": 101,
                    "prediction": prediction,
                    "score": None,
                }
            ],
        },
    }


def _existing_pretrained_to_sft25(
    split_name: str = "episode_group",
) -> list[dict[str, object]]:
    return [
        {
            "reference_stage": "pretrained",
            "destination_stage": "sft_25",
            "tap": tap,
            "factor": factor,
            "split": split_name,
            "status": "not_available",
            "reason": "fixture reference row",
        }
        for tap in STUDY_TAPS
        for factor in FACTOR_NAMES
    ]


def test_adjacent_stage_pairs_are_preregistered_in_training_order() -> None:
    assert ADJACENT_STAGE_PAIRS == (
        ("pretrained", "sft_25"),
        ("sft_25", "sft_50"),
        ("sft_50", "sft_100"),
    )


def test_adjacent_grid_reuses_reference_and_computes_sft25_to_sft50() -> None:
    existing = _existing_pretrained_to_sft25()
    marker = next(
        row
        for row in existing
        if row["tap"] == "action_expert_input" and row["factor"] == "phase"
    )
    rows = [
        _paired_row(
            stage="sft_25",
            factor="phase",
            prediction=[0, 0, 0, 0],
        ),
        _paired_row(
            stage="sft_50",
            factor="phase",
            prediction=[0, 0, 1, 1],
        ),
    ]

    result = build_adjacent_stage_delta_grid(
        rows,
        existing_reference_deltas=existing,
        split_name="episode_group",
        config=_config(),
    )

    assert len(result) == len(ADJACENT_STAGE_PAIRS) * len(STUDY_TAPS) * len(
        FACTOR_NAMES
    )
    copied = next(
        row
        for row in result
        if row["reference_stage"] == "pretrained"
        and row["destination_stage"] == "sft_25"
        and row["tap"] == "action_expert_input"
        and row["factor"] == "phase"
    )
    assert copied == marker
    assert copied is not marker

    adjacent = next(
        row
        for row in result
        if row["reference_stage"] == "sft_25"
        and row["destination_stage"] == "sft_50"
        and row["tap"] == "action_expert_input"
        and row["factor"] == "phase"
    )
    assert adjacent["status"] == "complete"
    assert adjacent["improvement"] > 0
    assert adjacent["improvement_low"] >= 0

    missing = next(
        row
        for row in result
        if row["reference_stage"] == "sft_50"
        and row["destination_stage"] == "sft_100"
        and row["tap"] == "action_expert_input"
        and row["factor"] == "phase"
    )
    assert missing["status"] == "not_available"


def test_adjacent_geometry_delta_preserves_lower_is_better_direction() -> None:
    rows = [
        _paired_row(
            stage="sft_25",
            factor="geometry",
            prediction=[[0.5], [0.5], [0.5], [0.5]],
        ),
        _paired_row(
            stage="sft_50",
            factor="geometry",
            prediction=[[0.0], [0.0], [1.0], [1.0]],
        ),
    ]

    result = build_adjacent_stage_delta_grid(
        rows,
        existing_reference_deltas=_existing_pretrained_to_sft25(),
        split_name="episode_group",
        config=_config(),
    )
    adjacent = next(
        row
        for row in result
        if row["reference_stage"] == "sft_25"
        and row["destination_stage"] == "sft_50"
        and row["tap"] == "action_expert_input"
        and row["factor"] == "geometry"
    )

    assert adjacent["destination_minus_reference"] == -0.5
    assert adjacent["improvement"] == 0.5
    assert adjacent["improvement_low"] == -adjacent["delta_high"]
    assert adjacent["improvement_high"] == -adjacent["delta_low"]


def test_enrichment_preserves_report_and_is_deterministic(
    monkeypatch,
) -> None:
    config = _config()
    original = {
        "schema_version": "libero_stage_tap_factor_probe_report_v2",
        "probe_protocol": "protocol_v2",
        "passed": True,
        "complete": False,
        "stage_deltas": _existing_pretrained_to_sft25("task_group"),
        "secondary_stage_deltas": _existing_pretrained_to_sft25("episode_group"),
        "preserved_marker": {"value": 7},
    }
    holder = {"report": deepcopy(original), "writes": []}
    primary_rows = [
        _paired_row(
            stage="sft_25",
            factor="phase",
            prediction=[0, 0, 0, 0],
        ),
        _paired_row(
            stage="sft_50",
            factor="phase",
            prediction=[0, 0, 1, 1],
        ),
    ]

    monkeypatch.setattr(
        transitions,
        "inspect_probe_report",
        lambda _config: deepcopy(holder["report"]),
    )
    monkeypatch.setattr(
        transitions,
        "_load_full_probe_rows",
        lambda _root, *, report, split_name: deepcopy(primary_rows),
    )

    def capture_write(path: Path, value: dict[str, object]) -> None:
        holder["writes"].append(path)
        holder["report"] = deepcopy(value)

    monkeypatch.setattr(transitions, "write_json_atomic", capture_write)

    first = enrich_adjacent_stage_deltas(config)
    second = enrich_adjacent_stage_deltas(config)

    assert all(first[key] == value for key, value in original.items())
    assert first == second
    assert first["adjacent_stage_pairs"] == [
        {"reference_stage": "pretrained", "destination_stage": "sft_25"},
        {"reference_stage": "sft_25", "destination_stage": "sft_50"},
        {"reference_stage": "sft_50", "destination_stage": "sft_100"},
    ]
    assert len(first["adjacent_stage_delta_analysis_sha256"]) == 64
    assert len(first["adjacent_stage_deltas"]) == 72
    assert len(first["secondary_adjacent_stage_deltas"]) == 72
    assert holder["writes"] == [
        config.output_dir / "probes/protocol_v2/report.json",
        config.output_dir / "probes/protocol_v2/report.json",
    ]


def _binding_report() -> dict[str, object]:
    key = "sft_25/action_expert_input"
    return {
        "probe_protocol": "protocol_v2",
        "state_bank_manifest_sha256": "state-bank",
        "split_manifest_sha256": {
            "task_group": "task-split",
            "episode_group": "episode-split",
        },
        "latent_cache_manifest_sha256": {key: "latent-manifest"},
        "latent_content_sha256": {key: "latent-content"},
        "config_sha256": "config",
        "implementation_sha256": "implementation",
    }


def _write_bound_cells(
    root: Path,
    report: dict[str, object],
    *,
    stage: str = "sft_25",
    split_name: str = "episode_group",
) -> None:
    tap = "action_expert_input"
    key = f"{stage}/{tap}"
    for factor in FACTOR_NAMES:
        identity = {
            "stage": stage,
            "tap": tap,
            "factor": factor,
            "split": split_name,
        }
        binding = {
            **identity,
            "probe_protocol": report["probe_protocol"],
            "state_bank_manifest_sha256": report["state_bank_manifest_sha256"],
            "split_manifest_sha256": report["split_manifest_sha256"][split_name],
            "latent_manifest_sha256": report["latent_cache_manifest_sha256"][key],
            "latent_content_sha256": report["latent_content_sha256"][key],
            "config_sha256": report["config_sha256"],
            "implementation_sha256": report["implementation_sha256"],
        }
        path = root / ".cells" / stage / tap / split_name / f"{factor}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"binding": binding, "row": {**identity, "status": "complete"}}),
            encoding="utf-8",
        )


def test_full_row_loader_rejects_orphaned_and_stale_cells(tmp_path: Path) -> None:
    report = _binding_report()
    _write_bound_cells(tmp_path, report)
    loaded = _load_full_probe_rows(
        tmp_path,
        report=report,
        split_name="episode_group",
    )
    assert any(
        row["stage"] == "sft_25"
        and row["tap"] == "action_expert_input"
        and row["factor"] == "phase"
        and row["status"] == "complete"
        for row in loaded
    )

    orphan_report = deepcopy(report)
    orphan_key = "sft_100/action_expert_input"
    orphan_report["latent_cache_manifest_sha256"][orphan_key] = "orphan-manifest"
    orphan_report["latent_content_sha256"][orphan_key] = "orphan-content"
    _write_bound_cells(tmp_path, orphan_report, stage="sft_100")
    with pytest.raises(ValueError, match="orphaned probe cell"):
        _load_full_probe_rows(
            tmp_path,
            report=report,
            split_name="episode_group",
        )

    stale_path = (
        tmp_path
        / ".cells/sft_25/action_expert_input/episode_group/phase.json"
    )
    stale = json.loads(stale_path.read_text(encoding="utf-8"))
    stale["binding"]["config_sha256"] = "stale-config"
    stale_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="binding is stale"):
        _load_full_probe_rows(
            tmp_path,
            report=report,
            split_name="episode_group",
        )


def test_enrichment_loads_bound_cells_and_atomically_replaces_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = replace(_config(), output_dir=tmp_path)
    report = {
        **_binding_report(),
        "schema_version": "libero_stage_tap_factor_probe_report_v2",
        "passed": True,
        "complete": False,
        "stage_deltas": _existing_pretrained_to_sft25("task_group"),
        "secondary_stage_deltas": _existing_pretrained_to_sft25("episode_group"),
        "preserved_marker": {"value": 11},
    }
    artifact_root = tmp_path / "probes/protocol_v2"
    artifact_root.mkdir(parents=True)
    report_path = artifact_root / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _write_bound_cells(artifact_root, report, split_name="task_group")
    _write_bound_cells(artifact_root, report, split_name="episode_group")
    monkeypatch.setattr(
        transitions,
        "inspect_probe_report",
        lambda _config: json.loads(report_path.read_text(encoding="utf-8")),
    )

    first = enrich_adjacent_stage_deltas(config)
    first_on_disk = json.loads(report_path.read_text(encoding="utf-8"))
    second = enrich_adjacent_stage_deltas(config)

    assert first_on_disk == first
    assert second == first
    assert second["preserved_marker"] == {"value": 11}
    assert len(second["adjacent_stage_deltas"]) == 72
    assert len(second["secondary_adjacent_stage_deltas"]) == 72
