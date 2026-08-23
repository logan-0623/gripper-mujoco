import json
from pathlib import Path

import pytest

from interaction_vla.representation_study.libero.state_bank import (
    EpisodeShardWriter,
    finalize_state_bank,
    load_state_bank,
)

from .helpers import make_record


def test_state_bank_shards_resume_only_when_binding_matches(tmp_path: Path) -> None:
    writer = EpisodeShardWriter(tmp_path / "bank", source_binding_sha256="a" * 64)
    records = (make_record(task_id=0, episode=0, frame=0),)
    path = writer.write(
        "libero_spatial:0:demo_0", records, metadata={"replay": {"passed": True}}
    )
    assert writer.write("libero_spatial:0:demo_0", records) == path
    loaded = writer.load("libero_spatial:0:demo_0")
    assert loaded is not None
    loaded_records, metadata = loaded
    assert loaded_records == records
    assert metadata["replay"]["passed"] is True

    stale = EpisodeShardWriter(tmp_path / "bank", source_binding_sha256="b" * 64)
    with pytest.raises(ValueError, match="stale episode shard"):
        stale.write("libero_spatial:0:demo_0", records)


def test_finalize_builds_two_splits_and_rejects_incompatible_overwrite(tmp_path: Path) -> None:
    records = tuple(
        make_record(task_id=task, episode=task * 10 + episode, frame=0)
        if episode == 0
        else make_record(
            task_id=task,
            episode=task * 10 + episode,
            frame=0,
            contact=True,
            stable=episode == 2,
        )
        for task in range(6)
        for episode in range(3)
    )
    root = tmp_path / "state_bank"
    report = finalize_state_bank(
        records,
        output_dir=root,
        source_binding_sha256="a" * 64,
        ontology_sha256="b" * 64,
        split_seed=7,
        task_ratios=(0.5, 0.25, 0.25),
        episode_ratios=(0.5, 0.25, 0.25),
        replay_statistics={"episodes": 18, "accepted": 18, "l2_p95": 0.0, "max_abs": 0.0},
    )
    assert report["passed"]
    loaded, manifest, task_split, episode_split = load_state_bank(root)
    assert loaded == records
    assert manifest["states"] == 18
    assert task_split.group_unit == "task"
    assert episode_split.group_unit == "episode"

    with pytest.raises(FileExistsError, match="different scientific binding"):
        finalize_state_bank(
            records,
            output_dir=root,
            source_binding_sha256="d" * 64,
            ontology_sha256="b" * 64,
            split_seed=7,
            task_ratios=(0.5, 0.25, 0.25),
            episode_ratios=(0.5, 0.25, 0.25),
            replay_statistics={"episodes": 18, "accepted": 18},
        )

    manifest["audit_passed"] = False
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="audit is not passing"):
        load_state_bank(root)


def test_failed_audit_is_preserved_before_state_bank_refuses_finalization(tmp_path: Path) -> None:
    records = tuple(make_record(task_id=task, episode=task, frame=0) for task in range(3))
    root = tmp_path / "bank"
    with pytest.raises(ValueError, match="audit gate failed"):
        finalize_state_bank(
            records,
            output_dir=root,
            source_binding_sha256="a" * 64,
            ontology_sha256="b" * 64,
            split_seed=1,
            task_ratios=(0.6, 0.2, 0.2),
            episode_ratios=(0.6, 0.2, 0.2),
            replay_statistics={
                "episodes": 3,
                "accepted": 0,
                "acceptance_rate": 0.0,
                "l2_p95": 1.0,
                "max_abs": 1.0,
            },
        )
    report = json.loads((root / "audit/report.json").read_text())
    assert not report["passed"]
    assert not (root / "manifest.json").exists()
