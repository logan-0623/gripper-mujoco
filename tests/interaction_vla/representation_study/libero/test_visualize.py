import hashlib
import json

from interaction_vla.representation_study.libero.visualize import (
    approve_annotation_timelines,
    render_annotation_timelines,
    validate_annotation_timeline_report,
)

from .helpers import make_record


def test_timeline_sampling_and_rendering_are_seeded_and_deterministic(tmp_path) -> None:
    records = tuple(
        make_record(task_id=task, episode=task, frame=frame, contact=frame == 1)
        for task in range(5)
        for frame in range(3)
    )
    first = render_annotation_timelines(
        records, output_dir=tmp_path / "first", count=3, seed=17
    )
    second = render_annotation_timelines(
        records, output_dir=tmp_path / "second", count=3, seed=17
    )
    assert len(first) == 3
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]


def test_timeline_gate_requires_explicit_review_and_validates_artifact_hashes(tmp_path) -> None:
    bank_manifest = tmp_path / "state_bank" / "manifest.json"
    bank_manifest.parent.mkdir(parents=True)
    bank_manifest.write_text('{"schema_version":"bank"}\n', encoding="utf-8")
    image = tmp_path / "timelines" / "episode.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"timeline")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report_path = image.parent / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "libero_annotation_timeline_report_v1",
                "passed": True,
                "manual_review_passed": False,
                "state_bank_manifest_sha256": digest(bank_manifest),
                "timelines": [{"path": str(image), "sha256": digest(image)}],
            }
        ),
        encoding="utf-8",
    )

    assert validate_annotation_timeline_report(
        report_path, state_bank_manifest=bank_manifest, require_approved=False
    )["passed"]
    try:
        validate_annotation_timeline_report(
            report_path, state_bank_manifest=bank_manifest, require_approved=True
        )
    except ValueError as error:
        assert "manual review" in str(error)
    else:
        raise AssertionError("probe gate must require explicit timeline review")
    approved = approve_annotation_timelines(
        report_path, state_bank_manifest=bank_manifest
    )
    assert approved["manual_review_passed"] is True
    assert validate_annotation_timeline_report(
        report_path, state_bank_manifest=bank_manifest, require_approved=True
    )["manual_review_passed"] is True
