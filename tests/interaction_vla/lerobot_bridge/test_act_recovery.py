from pathlib import Path
from types import SimpleNamespace

import pytest

from interaction_vla.lerobot_bridge import act_recovery as recovery_module
from interaction_vla.lerobot_bridge.act_recovery import (
    RecoveryCase,
    aggregate_recovery,
    evaluate_recovery,
    heldout_candidates,
    require_verified_training_summary,
    train_seen_cases,
)


def test_train_seen_cases_use_only_training_normal_two_object_episodes() -> None:
    manifest = [
        {
            "episode_index": index,
            "seed": 100 + index,
            "object_count": 2 if index % 2 == 0 else 3,
        }
        for index in range(20)
    ]
    cases = train_seen_cases(manifest, train_episodes=range(20), count=5)

    assert len(cases) == 5
    assert all(case.partition == "train_seen" for case in cases)
    assert all(case.layout == "normal" and case.object_count == 2 for case in cases)
    assert len({case.seed for case in cases}) == 5


def test_heldout_candidate_schedule_is_deterministic_and_disjoint() -> None:
    first = heldout_candidates(master_seed=17, count=200, forbidden_seeds={1, 2, 3})
    second = heldout_candidates(master_seed=17, count=200, forbidden_seeds={1, 2, 3})

    assert first == second
    assert len(first) == 200
    assert len({case.seed for case in first}) == 200
    assert not ({case.seed for case in first} & {1, 2, 3})


def test_recovery_requires_both_prespecified_gates() -> None:
    records = [
        {
            "partition": "train_seen",
            "success": index < 8,
            "termination_reason": "success" if index < 8 else "timeout",
        }
        for index in range(10)
    ] + [
        {
            "partition": "heldout",
            "success": index < 6,
            "termination_reason": "success" if index < 6 else "timeout",
        }
        for index in range(20)
    ]

    report = aggregate_recovery(
        records,
        train_threshold=0.8,
        heldout_threshold=0.3,
    )

    assert report["passed"] is True
    assert report["train_seen"]["success_rate"] == pytest.approx(0.8)
    assert report["heldout"]["success_rate"] == pytest.approx(0.3)


def test_recovery_rejects_unverified_checkpoint_summary(tmp_path: Path) -> None:
    summary = tmp_path / "training_summary.json"
    summary.write_text('{"reload_max_abs_error": 0.01}', encoding="utf-8")

    with pytest.raises(ValueError, match="reload"):
        require_verified_training_summary(summary)


def test_recovery_qualifies_cases_before_loading_policy(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "dataset"
    metadata = dataset_root / "meta"
    metadata.mkdir(parents=True)
    (metadata / "teacher_manifest.json").write_text(
        "["
        '{"episode_index": 0, "seed": 100, "object_count": 2},'
        '{"episode_index": 1, "seed": 101, "object_count": 3}'
        "]",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "training_summary.json").write_text(
        '{"reload_max_abs_error": 0.0}', encoding="utf-8"
    )
    config = SimpleNamespace(
        dataset=SimpleNamespace(root=dataset_root, repo_id="local/test", episodes=50),
        act=SimpleNamespace(seed=0, device="cpu"),
        recovery=SimpleNamespace(
            output_dir=tmp_path / "evaluation",
            train_seen_cases=1,
            heldout_cases=2,
            heldout_attempt_multiplier=3,
            heldout_master_seed=17,
            max_steps=12,
            train_success_threshold=1.0,
            heldout_success_threshold=1.0,
        ),
    )
    monkeypatch.setattr(recovery_module, "load_bridge_config", lambda path: config)
    monkeypatch.setattr(
        recovery_module,
        "pilot_episode_split",
        lambda **kwargs: {"train": [0], "validation": [1], "test": [2]},
    )
    monkeypatch.setattr(
        recovery_module, "validate_dataset_root", lambda *args, **kwargs: None
    )
    events: list[str] = []

    def expert(config, case, *, max_steps):
        events.append(f"expert:{case.case_id}")
        return {
            "case_id": case.case_id,
            "success": True,
            "termination_reason": "success",
            "steps": max_steps,
        }

    monkeypatch.setattr(recovery_module, "rollout_expert_case", expert)

    def load_bundle(**kwargs):
        events.append("load_policy")
        return object(), object(), object(), {"dataset_fingerprint": "d" * 64}

    monkeypatch.setattr(recovery_module, "_load_checkpoint_bundle", load_bundle)
    monkeypatch.setattr(
        recovery_module, "resolve_device", lambda requested: "cpu"
    )

    def policy_rollout(config, runtime, **kwargs):
        events.append(f"policy:{kwargs['seed']}")
        return {
            "success": True,
            "termination_reason": "success",
            "steps": kwargs["max_steps"],
            "mean_ik_projection_scale": 1.0,
            "action_clipping_rate": 0.0,
            "gripper_switch_count": 2,
        }

    monkeypatch.setattr(recovery_module, "rollout_loaded_policy", policy_rollout)

    report = evaluate_recovery("config.yaml", checkpoint)

    assert events[:3] == [
        "expert:train_seen_000",
        "expert:heldout_000",
        "expert:heldout_001",
    ]
    assert events[3] == "load_policy"
    assert report["passed"] is True
    assert report["heldout_candidates_inspected"] == 2
    assert report["heldout_selected_case_ids"] == ["heldout_000", "heldout_001"]
    assert len(report["records"]) == 3
    assert (config.recovery.output_dir / "recovery_report.json").is_file()
