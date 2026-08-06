from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

import interaction_vla.physics_data as physics_data_module
from interaction_vla.config import PhysicsConfig, load_config
from interaction_vla.physics_data import (
    PhysicsEpisode,
    PhysicsRecoveryRejected,
    collect_physics_episode,
    expert_gate_provenance,
    expected_gate_hashes,
    prepare_physics_recovery_start,
    require_expert_gate,
    require_episode_gate_provenance,
    save_physics_episode,
)
from interaction_vla.physics_env import FrankaContactEnv, PhysicsInterventionResult
from interaction_vla.physics_expert import PhysicsExpertPhase, PhysicsScriptedExpert
from interaction_vla.physics_recovery import (
    PhysicsRecoveryKind,
    make_physics_recovery_spec,
)
from interaction_vla.physics_recording import MultiViewFrame, RGBDFrame
from interaction_vla.train import inspect_episode_dimensions
from interaction_vla.data import load_episode_arrays


class ProgressSpy:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.updates: list[int] = []
        self.postfixes: list[dict[str, object]] = []
        self.closed = False

    def update(self, amount: int) -> None:
        self.updates.append(amount)

    def set_postfix(self, **kwargs: object) -> None:
        self.postfixes.append(kwargs)

    def close(self) -> None:
        self.closed = True


def fake_episode(reason: str) -> PhysicsEpisode:
    return PhysicsEpisode(
        seed=1,
        object_count=2,
        target_name="object_0",
        reason=reason,
        trajectory_source="scripted",
        metadata={},
        node_features=np.zeros((1, 1, 1), dtype=np.float32),
        edge_index=np.zeros((2, 1), dtype=np.int64),
        edge_features=np.zeros((1, 1, 1), dtype=np.float32),
        node_mask=np.ones((1, 1), dtype=np.bool_),
        edge_mask=np.ones((1, 1), dtype=np.bool_),
        proprioception=np.zeros((1, 23), dtype=np.float32),
        actions=np.zeros((1, 7), dtype=np.float32),
        phases=np.asarray(("transport",)),
        contact_state=np.zeros((1, 2, 2), dtype=np.bool_),
        contact_force=np.zeros((1, 2, 2), dtype=np.float32),
        relative_pose=np.zeros((1, 2, 6), dtype=np.float32),
        stable_grasp=np.zeros((1, 2), dtype=np.bool_),
    )


def make_env() -> FrankaContactEnv:
    return FrankaContactEnv(
        max_steps=180,
        physics=PhysicsConfig(settle_steps=100),
        workspace_low=(0.25, -0.35, 0.23),
        workspace_high=(0.78, 0.35, 0.75),
        crowded_anchor_min_distance=0.055,
        crowded_anchor_max_distance=0.075,
    )


class TransitioningExpert:
    def __init__(self) -> None:
        self.phase = PhysicsExpertPhase.TRANSPORT

    def act(self, snapshot, contacts, grasp) -> np.ndarray:
        self.phase = PhysicsExpertPhase.RELEASE
        return np.zeros(7, dtype=np.float32)


def test_recorded_phase_is_the_phase_that_generated_the_action() -> None:
    expert = TransitioningExpert()

    phase, action = physics_data_module._expert_action_with_phase(
        expert, object(), object(), object()
    )

    assert phase == "transport"
    assert expert.phase is PhysicsExpertPhase.RELEASE
    np.testing.assert_array_equal(action, np.zeros(7, dtype=np.float32))


def test_physics_episode_saves_pre_action_7d_18d_contact_diagnostics(tmp_path) -> None:
    env = make_env()
    expert = PhysicsScriptedExpert(env.physics)

    episode = collect_physics_episode(
        env,
        expert,
        seed=11,
        object_count=2,
        trajectory_source="scripted",
        expert_gate_hash="a" * 64,
    )
    path = save_physics_episode(episode, tmp_path / "episode.npz")

    assert episode.reason == "success"
    assert episode.actions.shape[1] == 7
    assert episode.edge_features.shape[2] == 18
    assert episode.proprioception.shape[1] == 23
    release_actions = episode.actions[episode.phases == "release"]
    retreat_actions = episode.actions[episode.phases == "retreat"]
    assert len(release_actions) > 0
    assert len(retreat_actions) > 0
    assert np.all(release_actions[:, 6] == 1.0)
    assert np.all(retreat_actions[:, 6] == 1.0)
    assert float(np.mean(retreat_actions[:, 2])) > 0.5
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        assert metadata["trajectory_source"] == "scripted"
        assert metadata["feature_schema"] == "physics_v2"
        assert archive["actions"].shape[1] == 7
        assert archive["edge_features"].shape[2] == 18
        assert archive["contact_state"].shape == (len(episode.actions), 2, 2)
        assert archive["contact_force"].shape == (len(episode.actions), 2, 2)
        assert archive["relative_pose"].shape == (len(episode.actions), 2, 6)
        assert archive["stable_grasp"].shape == (len(episode.actions), 2)

    dimensions = inspect_episode_dimensions((path,))
    assert dimensions == {
        "node_feature_dim": 23,
        "edge_feature_dim": 18,
        "proprioception_dim": 23,
        "action_dim": 7,
    }
    require_episode_gate_provenance((path,), "a" * 64)
    with pytest.raises(ValueError, match="gate provenance"):
        require_episode_gate_provenance((path,), "b" * 64)


def test_expert_gate_must_pass_and_match_config_scene_and_controller_hashes(tmp_path) -> None:
    config_path = "configs/physics_smoke_macos.yaml"
    gate_path = tmp_path / "expert_gate.json"
    with pytest.raises(FileNotFoundError, match="expert gate"):
        require_expert_gate(config_path, gate_path)

    hashes = expected_gate_hashes(config_path)
    report = {
        "passed": True,
        **hashes,
    }
    gate_path.write_text(json.dumps(report), encoding="utf-8")
    artifact_hash = require_expert_gate(config_path, gate_path)
    assert len(artifact_hash) == 64
    provenance = expert_gate_provenance(config_path, gate_path)
    assert provenance == {
        "expert_gate_hash": artifact_hash,
        **hashes,
    }

    report["controller_hash"] = "0" * 64
    gate_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        require_expert_gate(config_path, gate_path)


@pytest.mark.parametrize("variant_id", [0, 1, 2])
def test_post_grasp_recovery_saves_only_corrective_suffix(
    tmp_path, variant_id: int
) -> None:
    env = make_env()
    spec = make_physics_recovery_spec(11, variant_id)
    episode = collect_physics_episode(
        env,
        PhysicsScriptedExpert(env.physics),
        seed=11,
        object_count=2,
        recovery=spec,
    )
    path = save_physics_episode(episode, tmp_path / f"recovery_{variant_id}.npz")

    loaded = load_episode_arrays(path)
    assert episode.reason == "success"
    assert episode.metadata["injection_phase"] == "transport"
    assert episode.phases[0] in {"lift", "transport"}
    assert episode.actions[0, 6] == 0.0
    assert "approach" not in set(episode.phases)
    assert loaded.trajectory_kind == "recovery"
    assert loaded.source_seed == 11
    assert loaded.variant_id == variant_id
    assert loaded.perturbation_kind == spec.kind.value
    assert loaded.injection_phase == "transport"
    assert episode.metadata["recovery"] == spec.metadata()

    if spec.translation_steps:
        target_rows = np.flatnonzero(episode.node_features[0, :, 22] == 1.0)
        assert len(target_rows) == 1
        target_xy = episode.node_features[0, int(target_rows[0]), 4:6]
        receptacle_xy = episode.node_features[0, env.max_objects + 1, 4:6]
        assert float(np.dot(episode.actions[0, :2], receptacle_xy - target_xy)) > 0.0
    else:
        assert episode.metadata["recovery"]["open_substeps"] == 1


def test_recovery_start_is_deterministic_and_records_handoff_baseline() -> None:
    spec = make_physics_recovery_spec(11, 0)

    def prepare_once():
        env = make_env()
        expert = PhysicsScriptedExpert(env.physics)
        prepared = prepare_physics_recovery_start(
            env,
            expert,
            spec=spec,
            object_count=2,
            source_split="test",
        )
        return env, prepared

    first_env, first = prepare_once()
    second_env, second = prepare_once()

    np.testing.assert_array_equal(first_env.data.qpos, second_env.data.qpos)
    np.testing.assert_array_equal(first_env.data.qvel, second_env.data.qvel)
    assert first.snapshot.target_object.name == second.snapshot.target_object.name
    assert first.source_seed == second.source_seed == 11
    assert first.source_split == second.source_split == "test"
    assert first.variant_id == second.variant_id == 0
    assert first.kind == second.kind == spec.kind.value
    assert first.interaction_baseline == second.interaction_baseline
    assert first.interaction_baseline["tracker_substep"] > 0
    assert first.interaction_baseline["ever_bilateral_target_contact"] is True

    qpos_at_handoff = first_env.data.qpos.copy()
    first_env.step(np.zeros(7, dtype=np.float32))
    assert not np.array_equal(first_env.data.qpos, qpos_at_handoff)


def test_terminal_reclose_recovery_teaches_open_retreat(monkeypatch) -> None:
    env = make_env()
    spec = make_physics_recovery_spec(11, 3, kind_index=3)
    intervention_steps: list[dict[str, object]] = []
    advance_intervention = env.advance_intervention

    def track_intervention(action, *, substeps):
        tcp_before, _ = env.controller.tcp_pose()
        finger_before = float(np.mean(env.proprioception()[13:15]))
        result = advance_intervention(action, substeps=substeps)
        tcp_after, _ = env.controller.tcp_pose()
        intervention_steps.append(
            {
                "action": np.asarray(action).copy(),
                "substeps": substeps,
                "tcp_before": tcp_before.copy(),
                "tcp_after": tcp_after.copy(),
                "finger_before": finger_before,
                "finger_after": float(np.mean(env.proprioception()[13:15])),
                "supported": (
                    env.target_name
                    in env.contact_diagnostics.object_receptacle
                ),
                "physics_failure": result.physics_failure,
            }
        )
        return result

    monkeypatch.setattr(env, "advance_intervention", track_intervention)

    episode = collect_physics_episode(
        env,
        PhysicsScriptedExpert(env.physics),
        seed=11,
        object_count=2,
        recovery=spec,
    )

    assert episode.reason == "success"
    assert episode.metadata["injection_phase"] == "retreat"
    assert set(episode.phases) == {"retreat"}
    assert episode.actions[0, 6] == 1.0
    assert episode.actions[0, 2] > 0.5
    assert episode.metadata["recovery"]["close_descent_steps"] == 5
    assert len(intervention_steps) == 5
    for step in intervention_steps:
        np.testing.assert_array_equal(
            step["action"],
            np.asarray((0, 0, -1, 0, 0, 0, 0), dtype=np.float32),
        )
        assert step["substeps"] == env.physics.substeps
        assert step["supported"] is True
        assert step["physics_failure"] is None
    assert intervention_steps[-1]["tcp_after"][2] < intervention_steps[0][
        "tcp_before"
    ][2]
    assert intervention_steps[-1]["finger_after"] < intervention_steps[0][
        "finger_before"
    ]


def test_terminal_intervention_rejects_reported_physics_failure(
    monkeypatch,
) -> None:
    env = make_env()
    snapshot = env.reset(seed=11, object_count=2)
    spec = make_physics_recovery_spec(11, 3, kind_index=3)

    monkeypatch.setattr(
        env,
        "advance_intervention",
        lambda action, *, substeps: PhysicsInterventionResult(
            snapshot=snapshot,
            controller_diagnostics=None,
            physics_failure="severe_penetration",
        ),
    )

    with pytest.raises(
        PhysicsRecoveryRejected,
        match=(
            "physics_failure_during_terminal_intervention:"
            "severe_penetration"
        ),
    ):
        physics_data_module._apply_recovery_intervention(env, snapshot, spec)


def test_recovery_quality_requires_every_kind_and_minimum_rate() -> None:
    attempted = {
        PhysicsRecoveryKind.WRONG_WAY_TRANSPORT: 10,
        PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE: 10,
    }
    accepted = {
        PhysicsRecoveryKind.WRONG_WAY_TRANSPORT: 8,
        PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE: 7,
    }

    failed_summary = physics_data_module.recovery_quality_summary(
        attempted,
        accepted,
        minimum_rate=0.8,
        expected_kinds=tuple(PhysicsRecoveryKind),
    )
    assert failed_summary["premature_open"] == {
        "attempted": 0,
        "accepted": 0,
        "acceptance_rate": 0.0,
        "passed": False,
    }
    with pytest.raises(RuntimeError, match="premature_open"):
        physics_data_module.require_recovery_quality(failed_summary)

    attempted[PhysicsRecoveryKind.PREMATURE_OPEN] = 10
    attempted[PhysicsRecoveryKind.RECEPTACLE_MISALIGNMENT] = 10
    accepted[PhysicsRecoveryKind.PREMATURE_OPEN] = 8
    accepted[PhysicsRecoveryKind.RECEPTACLE_MISALIGNMENT] = 8
    accepted[PhysicsRecoveryKind.POST_PLACEMENT_RECLOSE] = 8
    summary = physics_data_module.recovery_quality_summary(
        attempted,
        accepted,
        minimum_rate=0.8,
        expected_kinds=tuple(PhysicsRecoveryKind),
    )
    physics_data_module.require_recovery_quality(summary)
    assert summary["post_placement_reclose"] == {
        "attempted": 10,
        "accepted": 8,
        "acceptance_rate": 0.8,
        "passed": True,
    }


def test_optional_rgbd_sidecar_is_pre_action_aligned_and_named_in_metadata(tmp_path) -> None:
    class FakeRecorder:
        def capture(self, env):
            rgb = np.zeros((2, 3, 3), dtype=np.uint8)
            depth = np.ones((2, 3), dtype=np.float32)
            return MultiViewFrame(
                policy_step=env.step_count,
                simulation_time=float(env.data.time),
                state_hash=f"step-{env.step_count}",
                views={
                    name: RGBDFrame(rgb=rgb, depth=depth)
                    for name in ("agent", "wrist", "side", "top")
                },
            )

    episode = collect_physics_episode(
        make_env(),
        PhysicsScriptedExpert(PhysicsConfig(settle_steps=100)),
        seed=11,
        object_count=2,
        recorder=FakeRecorder(),
    )
    episode_path = save_physics_episode(
        episode,
        tmp_path / "episode.npz",
        rgbd_path=tmp_path / "episode_rgbd.npz",
    )

    assert len(episode.rgbd_frames) == len(episode.actions)
    assert [frame.policy_step for frame in episode.rgbd_frames] == list(
        range(len(episode.actions))
    )
    with np.load(episode_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    assert metadata["rgbd_sidecar"] == "episode_rgbd.npz"
    with np.load(tmp_path / "episode_rgbd.npz", allow_pickle=False) as archive:
        assert archive["observation_agent_rgb"].shape[0] == len(episode.actions)


def test_collection_reports_base_acceptance_and_every_recovery_attempt(
    monkeypatch, tmp_path: Path
) -> None:
    original = load_config("configs/physics_recovery_smoke_macos.yaml")
    config = replace(
        original,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
        train=replace(original.train, episodes=1),
    )
    base_attempts = 0
    recovery_attempts = 0
    progress_instances: list[ProgressSpy] = []

    def fake_tqdm(**kwargs: object) -> ProgressSpy:
        progress = ProgressSpy(**kwargs)
        progress_instances.append(progress)
        return progress

    def fake_collect(*args: object, **kwargs: object) -> PhysicsEpisode:
        nonlocal base_attempts, recovery_attempts
        if kwargs.get("recovery") is None:
            base_attempts += 1
            return fake_episode("timeout" if base_attempts == 1 else "success")
        recovery_attempts += 1
        if recovery_attempts == 1:
            return fake_episode("success")
        if recovery_attempts == 2:
            raise PhysicsRecoveryRejected("trigger_not_reached:timeout")
        return fake_episode("timeout")

    monkeypatch.setattr(physics_data_module, "load_config", lambda path: config)
    monkeypatch.setattr(physics_data_module, "require_expert_gate", lambda *args: "g" * 64)
    monkeypatch.setattr(physics_data_module, "FrankaContactEnv", lambda **kwargs: object())
    monkeypatch.setattr(physics_data_module, "PhysicsScriptedExpert", lambda config: object())
    monkeypatch.setattr(physics_data_module, "collect_physics_episode", fake_collect)
    monkeypatch.setattr(
        physics_data_module,
        "require_recovery_quality",
        lambda _summary: None,
    )
    monkeypatch.setattr(
        physics_data_module,
        "save_physics_episode",
        lambda episode, path, **kwargs: Path(path),
    )
    monkeypatch.setattr(physics_data_module, "tqdm", fake_tqdm)

    manifest = physics_data_module.collect_from_config(
        "ignored.yaml",
        show_progress=True,
    )

    assert manifest == tmp_path / "data" / "manifest.json"
    assert len(progress_instances) == 2
    base_progress, recovery_progress = progress_instances
    assert base_progress.kwargs == {
        "total": 1,
        "desc": "base data",
        "unit": "episode",
        "dynamic_ncols": True,
    }
    assert base_progress.updates == [1]
    assert base_progress.postfixes == [
        {
            "attempts": 1,
            "accepted": 0,
            "rejected": 1,
            "objects": 2,
            "reason": "timeout",
        },
        {
            "attempts": 2,
            "accepted": 1,
            "rejected": 1,
            "objects": 3,
            "reason": "success",
        },
    ]
    assert base_progress.closed
    assert recovery_progress.kwargs == {
        "total": 3,
        "desc": "recovery",
        "unit": "attempt",
        "dynamic_ncols": True,
    }
    assert recovery_progress.updates == [1, 1, 1]
    assert [postfix["kind"] for postfix in recovery_progress.postfixes] == [
        "wrong_way_transport",
        "premature_open",
        "receptacle_misalignment",
    ]
    assert [postfix["reason"] for postfix in recovery_progress.postfixes] == [
        "success",
        "trigger_not_reached:timeout",
        "timeout",
    ]
    assert recovery_progress.postfixes[-1]["accepted"] == 1
    assert recovery_progress.postfixes[-1]["rejected"] == 2
    assert recovery_progress.closed

    base_attempts = 0
    recovery_attempts = 0
    physics_data_module.collect_from_config("ignored.yaml")
    assert len(progress_instances) == 2


def test_collection_cli_enables_progress(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    captured: dict[str, object] = {}

    def fake_collect(config_path: str, **kwargs: object) -> Path:
        captured.update({"config_path": config_path, **kwargs})
        return output

    monkeypatch.setattr(physics_data_module, "collect_from_config", fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "physics_data",
            "collect",
            "--config",
            "configs/physics_recovery_smoke_macos.yaml",
        ],
    )

    physics_data_module.main()

    assert captured == {
        "config_path": "configs/physics_recovery_smoke_macos.yaml",
        "expert_gate": None,
        "show_progress": True,
    }


def test_collection_progress_closes_and_labels_unexpected_recovery_exception(
    monkeypatch, tmp_path: Path
) -> None:
    original = load_config("configs/physics_recovery_smoke_macos.yaml")
    config = replace(
        original,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
        train=replace(original.train, episodes=1),
    )
    progress_instances: list[ProgressSpy] = []

    def fake_collect(*args: object, **kwargs: object) -> PhysicsEpisode:
        if kwargs.get("recovery") is None:
            return fake_episode("success")
        raise RuntimeError("boom")

    def fake_tqdm(**kwargs: object) -> ProgressSpy:
        progress = ProgressSpy(**kwargs)
        progress_instances.append(progress)
        return progress

    monkeypatch.setattr(physics_data_module, "load_config", lambda path: config)
    monkeypatch.setattr(physics_data_module, "require_expert_gate", lambda *args: "g" * 64)
    monkeypatch.setattr(physics_data_module, "FrankaContactEnv", lambda **kwargs: object())
    monkeypatch.setattr(physics_data_module, "PhysicsScriptedExpert", lambda config: object())
    monkeypatch.setattr(physics_data_module, "collect_physics_episode", fake_collect)
    monkeypatch.setattr(
        physics_data_module,
        "save_physics_episode",
        lambda episode, path, **kwargs: Path(path),
    )
    monkeypatch.setattr(physics_data_module, "tqdm", fake_tqdm)

    with pytest.raises(RuntimeError, match="boom"):
        physics_data_module.collect_from_config(
            "ignored.yaml",
            show_progress=True,
        )

    assert len(progress_instances) == 2
    base_progress, recovery_progress = progress_instances
    assert base_progress.closed
    assert recovery_progress.closed
    assert recovery_progress.updates == [1]
    assert recovery_progress.postfixes[0]["reason"] == "exception:RuntimeError"


def test_v3_collection_routes_recovery_by_source_split(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original = load_config("configs/physics_interaction_chunk_smoke_macos.yaml")
    config = replace(
        original,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
    )
    progress_instances: list[ProgressSpy] = []

    def fake_tqdm(**kwargs: object) -> ProgressSpy:
        progress = ProgressSpy(**kwargs)
        progress_instances.append(progress)
        return progress

    monkeypatch.setattr(physics_data_module, "load_config", lambda path: config)
    monkeypatch.setattr(
        physics_data_module,
        "require_expert_gate",
        lambda *args: "g" * 64,
    )
    monkeypatch.setattr(
        physics_data_module,
        "FrankaContactEnv",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        physics_data_module,
        "PhysicsScriptedExpert",
        lambda config: object(),
    )
    monkeypatch.setattr(
        physics_data_module,
        "collect_physics_episode",
        lambda *args, **kwargs: fake_episode("success"),
    )
    monkeypatch.setattr(
        physics_data_module,
        "save_physics_episode",
        lambda episode, path, **kwargs: Path(path),
    )
    monkeypatch.setattr(physics_data_module, "tqdm", fake_tqdm)

    physics_data_module.collect_from_config("ignored.yaml", show_progress=True)

    data_dir = tmp_path / "data"
    source_payload = json.loads(
        (data_dir / "source_split.json").read_text(encoding="utf-8")
    )
    assert len(source_payload["train"]) == 8
    assert len(source_payload["validation"]) == 1
    assert len(source_payload["test"]) == 1
    assert len(source_payload["training_recovery_sources"]) == 2

    base_records = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert {record["source_split"] for record in base_records} == {
        "train",
        "validation",
        "test",
    }
    training_records = json.loads(
        (data_dir / "recovery_manifest.json").read_text(encoding="utf-8")
    )
    benchmark_records = json.loads(
        (data_dir / "recovery_benchmark_manifest.json").read_text(encoding="utf-8")
    )
    assert {record["source_seed"] for record in training_records}.issubset(
        source_payload["training_recovery_sources"]
    )
    assert {record["source_seed"] for record in benchmark_records}.issubset(
        source_payload["validation"] + source_payload["test"]
    )
    assert all(record["source_split"] == "train" for record in training_records)
    assert {record["source_split"] for record in benchmark_records} == {
        "validation",
        "test",
    }
    assert [progress.kwargs["desc"] for progress in progress_instances] == [
        "base data",
        "training recovery",
        "held-out recovery",
    ]
