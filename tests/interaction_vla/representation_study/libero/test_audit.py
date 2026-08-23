from interaction_vla.representation_study.libero.audit import build_state_bank_audit

from .helpers import make_record


def test_audit_reports_factor_support_imbalance_ranges_and_replay_gate() -> None:
    records = tuple(
        make_record(
            task_id=task,
            episode=task,
            frame=frame,
            phase="contact" if frame == 1 else "approach",
            contact=frame == 1,
            stable=frame == 1,
        )
        for task in range(3)
        for frame in range(2)
    )
    report = build_state_bank_audit(
        records,
        replay_statistics={
            "episodes": 3,
            "accepted": 3,
            "acceptance_rate": 1.0,
            "l2_p95": 0.001,
            "max_abs": 0.002,
        },
        minimum_acceptance_rate=0.95,
        l2_p95_tolerance=0.01,
        max_abs_tolerance=0.05,
    )
    assert report["passed"]
    assert report["tasks"] == 3
    assert report["episodes"] == 3
    assert report["states"] == 6
    assert report["phase_distribution"] == {"approach": 3, "contact": 3}
    assert report["contact"]["positive"] == 3
    assert report["contact"]["subfields"]["gripper_target"]["positive"] == 3
    assert report["contact"]["isolated_positive_frames"] == 3
    assert report["entity_distribution"]["object_0"] == 2
    assert report["next_relation_distribution"]["gripper|near|target|establish"] == 6
    assert report["trajectory_outcomes"]["source_outcome_available"] is False
    assert report["trajectory_outcomes"]["terminal_goal_relation_not_observed"] == 3
    assert report["geometry"]["gripper_target_distance"]["max"] == 0.02
    assert report["missing_or_invalid"] == 0


def test_audit_fails_on_replay_mismatch() -> None:
    report = build_state_bank_audit(
        (make_record(task_id=0, episode=0, frame=0),),
        replay_statistics={
            "episodes": 1,
            "accepted": 0,
            "acceptance_rate": 0.0,
            "l2_p95": 0.2,
            "max_abs": 0.3,
        },
        minimum_acceptance_rate=0.95,
        l2_p95_tolerance=0.01,
        max_abs_tolerance=0.05,
    )
    assert not report["passed"]
    assert {
        "replay acceptance rate is below threshold",
        "replay l2 p95 exceeds tolerance",
        "replay maximum absolute error exceeds tolerance",
    }.issubset(report["gate_reasons"])


def test_audit_fails_when_contact_or_stable_grasp_has_no_class_support() -> None:
    report = build_state_bank_audit(
        tuple(make_record(task_id=task, episode=task, frame=0) for task in range(3)),
        replay_statistics={
            "episodes": 3,
            "accepted": 3,
            "acceptance_rate": 1.0,
            "l2_p95": 0.0,
            "max_abs": 0.0,
        },
        minimum_acceptance_rate=0.95,
        l2_p95_tolerance=0.01,
        max_abs_tolerance=0.05,
    )
    assert report["passed"] is False
    assert "contact labels do not contain both classes" in report["gate_reasons"]
    assert "stable-grasp labels do not contain both classes" in report["gate_reasons"]
