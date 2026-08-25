import hashlib

import pytest

from interaction_vla.representation_study.libero import collector
from interaction_vla.representation_study.libero.alignment import AlignmentRow

from .helpers import make_record


def _alignment_row(episode: int) -> AlignmentRow:
    episode_id = f"demo_{episode}"
    return AlignmentRow(
        suite="libero_spatial",
        task_id=0,
        raw_episode_id=episode_id,
        lerobot_episode_id=episode_id,
        frames=10,
        raw_relative_path="task_0.hdf5",
        raw_demo_key=episode_id,
        model_xml_sha256="a" * 64,
        action_sha256="b" * 64,
        max_action_error=0.0,
    )


def _replay_row(
    *, task_id: int, episode: int, passed: bool, l2_p95: float, max_abs: float
) -> dict[str, object]:
    return {
        "suite": "libero_spatial",
        "task_id": task_id,
        "episode_id": f"demo_{episode}",
        "passed": passed,
        "replay_mode": "restore_each_recorded_state_then_step_action",
        "validation_vector": "robot_qpos",
        "replay_protocol": "libero_deterministic_replay_v2",
        "validated_transitions": 9,
        "l2_p95": l2_p95,
        "max_abs": max_abs,
    }


def test_replay_selection_backfills_rejected_candidates_deterministically() -> None:
    select_passing_rows = getattr(collector, "_select_passing_rows", None)
    assert select_passing_rows is not None, "collector must support replay backfill"
    candidates = tuple(_alignment_row(index) for index in range(6))
    ranked = tuple(
        sorted(
            candidates,
            key=lambda row: hashlib.sha256(
                f"7:{row.suite}:{row.task_id}:{row.raw_episode_id}".encode()
            ).hexdigest(),
        )
    )
    rejected_ids = {ranked[0].raw_episode_id, ranked[1].raw_episode_id}

    selection = select_passing_rows(
        candidates,
        required=3,
        seed=7,
        is_acceptable=lambda row: row.raw_episode_id not in rejected_ids,
    )

    assert selection.attempted == ranked[:5]
    assert selection.rejected == ranked[:2]
    assert selection.selected == ranked[2:5]


def test_replay_selection_refuses_a_task_without_enough_passing_episodes() -> None:
    select_passing_rows = getattr(collector, "_select_passing_rows", None)
    assert select_passing_rows is not None

    with pytest.raises(ValueError, match="only 1 replay-valid episodes"):
        select_passing_rows(
            tuple(_alignment_row(index) for index in range(3)),
            required=2,
            seed=7,
            is_acceptable=lambda row: row.raw_episode_id == "demo_0",
        )


def test_replay_statistics_gate_selected_bank_and_preserve_candidate_rejections() -> None:
    build_statistics = getattr(collector, "_build_replay_statistics", None)
    assert build_statistics is not None, "collector must separate bank and candidate stats"
    selected = (
        _replay_row(task_id=0, episode=0, passed=True, l2_p95=0.002, max_abs=0.01),
        _replay_row(task_id=1, episode=1, passed=True, l2_p95=0.004, max_abs=0.02),
    )
    rejected = _replay_row(
        task_id=0, episode=9, passed=False, l2_p95=0.03, max_abs=0.08
    )

    report = build_statistics(
        selected_rows=selected,
        attempted_rows=(rejected, *selected),
        expected_task_keys={("libero_spatial", 0), ("libero_spatial", 1)},
        required_episodes_per_task=1,
    )

    assert report["episodes"] == 2
    assert report["accepted"] == 2
    assert report["acceptance_rate"] == 1.0
    assert report["l2_p95"] < 0.01
    assert report["max_abs"] == 0.02
    assert report["candidate_attempts"] == 3
    assert report["candidate_rejected"] == 1
    assert report["candidate_acceptance_rate"] == pytest.approx(2 / 3)
    rows = report["rows"]
    assert isinstance(rows, list)
    assert [row["selected"] for row in rows] == [False, True, True]


def test_source_binding_changes_with_pipeline_hash() -> None:
    source_binding = getattr(collector, "_source_binding_sha256", None)
    assert source_binding is not None

    common = {
        "catalog_sha256": "a" * 64,
        "alignment_sha256": "b" * 64,
        "config_sha256": "c" * 64,
    }
    first = source_binding(**common, pipeline_sha256="d" * 64)
    second = source_binding(**common, pipeline_sha256="e" * 64)

    assert len(first) == 64
    assert first != second


def test_compatible_cached_records_must_match_current_ontology() -> None:
    validate_cached = getattr(collector, "_validate_cached_candidate", None)
    assert validate_cached is not None
    record = make_record(task_id=0, episode=0, frame=0)

    with pytest.raises(ValueError, match="annotation ontology"):
        validate_cached(
            (record,),
            {"passed": True},
            ontology_sha256="d" * 64,
            episode_key="libero_spatial:0:demo_0",
        )
