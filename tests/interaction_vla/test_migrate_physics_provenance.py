from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

import interaction_vla.migrate_physics_provenance as migration
from interaction_vla.config import load_config
from interaction_vla.physics_data import expected_gate_hashes
from interaction_vla.train import build_training_provenance, resolve_training_data


def test_migration_module_exists() -> None:
    assert importlib.util.find_spec("interaction_vla.migrate_physics_provenance") is not None


def _write_gate(path: Path, *, controller_hash: str, success: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "controller_hash": controller_hash,
                "scene_hash": "scene",
                "config_hash": "config",
                "success_rate": 0.9,
                "results": [
                    {
                        "case_id": "normal-0",
                        "success": success,
                        "reason": "success" if success else "timeout",
                        "steps": 10,
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_gate_equivalence_allows_only_controller_hash_change(tmp_path: Path) -> None:
    old_gate = _write_gate(tmp_path / "old.json", controller_hash="old")
    new_gate = _write_gate(tmp_path / "new.json", controller_hash="new")

    migration.assert_gate_reports_equivalent(old_gate, new_gate)


def test_gate_equivalence_rejects_any_outcome_change(tmp_path: Path) -> None:
    old_gate = _write_gate(tmp_path / "old.json", controller_hash="old")
    new_gate = _write_gate(
        tmp_path / "new.json", controller_hash="new", success=False
    )

    with pytest.raises(ValueError, match="differ outside controller_hash"):
        migration.assert_gate_reports_equivalent(old_gate, new_gate)


def _write_episode(
    path: Path,
    *,
    seed: int,
    gate_hash: str,
    trajectory_kind: str = "base",
) -> None:
    metadata = {
        "seed": seed,
        "object_count": 2,
        "target_name": "object_0",
        "reason": "success",
        "trajectory_kind": trajectory_kind,
        "source_seed": seed if trajectory_kind == "recovery" else None,
        "variant_id": 0 if trajectory_kind == "recovery" else None,
        "perturbation_kind": "translation" if trajectory_kind == "recovery" else None,
        "injection_phase": "transport" if trajectory_kind == "recovery" else None,
        "backend": "franka_contact",
        "feature_schema": "physics_v2",
        "expert_gate_hash": gate_hash,
    }
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        node_features=np.zeros((1, 6, 23), dtype=np.float32),
        edge_index=np.zeros((2, 30), dtype=np.int64),
        edge_features=np.zeros((1, 30, 18), dtype=np.float32),
        node_mask=np.ones((1, 6), dtype=np.bool_),
        edge_mask=np.zeros((1, 30), dtype=np.bool_),
        proprioception=np.zeros((1, 23), dtype=np.float32),
        actions=np.zeros((1, 7), dtype=np.float32),
        phases=np.asarray(["transport"]),
    )


def _make_preflight_fixture(tmp_path: Path) -> dict[str, Path]:
    output_dir = tmp_path / "output"
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True)
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "migration_test",
                "seed": 42,
                "device": "cpu",
                "backend": "franka_contact",
                "max_objects": 3,
                "data_dir": str(data_dir),
                "output_dir": str(output_dir),
                "train": {
                    "object_counts": [2],
                    "episodes": 1,
                    "model_seeds": [0],
                },
                "eval": {
                    "object_counts": [2, 3],
                    "ood_object_counts": [3],
                    "crowded_object_counts": [3],
                    "episodes_per_count": 1,
                },
                "model": {"action_dim": 7},
                "recovery": {"enabled": True, "variants_per_episode": 1},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    current = expected_gate_hashes(config_path)
    old_gate = _write_gate(output_dir / "expert_gate.json", controller_hash="old")
    verified_gate = _write_gate(
        tmp_path / "verified_gate.json",
        controller_hash=current["controller_hash"],
    )
    for gate_path in (old_gate, verified_gate):
        report = json.loads(gate_path.read_text(encoding="utf-8"))
        report.update(
            scene_hash=current["scene_hash"],
            config_hash=current["config_hash"],
        )
        if gate_path == verified_gate:
            report["rollout_integrity_hash"] = current[
                "rollout_integrity_hash"
            ]
        gate_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    old_gate_hash = hashlib.sha256(old_gate.read_bytes()).hexdigest()

    base_path = data_dir / "episode_000000.npz"
    recovery_path = data_dir / "recovery_000000.npz"
    _write_episode(base_path, seed=42, gate_hash=old_gate_hash)
    _write_episode(
        recovery_path,
        seed=42,
        gate_hash=old_gate_hash,
        trajectory_kind="recovery",
    )
    (data_dir / "manifest.json").write_text(
        json.dumps([{"path": base_path.name}]), encoding="utf-8"
    )
    (data_dir / "recovery_manifest.json").write_text(
        json.dumps([{"path": recovery_path.name}]), encoding="utf-8"
    )

    config = load_config(config_path)
    selection = resolve_training_data(
        config.data_dir,
        split_seed=config.seed,
        include_recovery=config.recovery.enabled,
    )
    training_provenance, _ = build_training_provenance(
        config,
        selection,
        expert_gate_hash=old_gate_hash,
    )
    for representation in ("flat", "graph"):
        checkpoint = output_dir / representation / "seed_0" / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save(
            {
                "representation": representation,
                "model_seed": 0,
                "backend": "franka_contact",
                "feature_schema": "physics_v2",
                "expert_gate_hash": old_gate_hash,
                "controller_hash": "old",
                "scene_hash": current["scene_hash"],
                "config_hash": current["config_hash"],
                "training_provenance": training_provenance,
                "sentinel": torch.tensor([1.0, 2.0]),
            },
            checkpoint,
        )
    return {
        "config": config_path,
        "output": output_dir,
        "verified_gate": verified_gate,
        "base": base_path,
        "recovery": recovery_path,
    }


def test_prepare_migration_discovers_old_chain_without_writing(tmp_path: Path) -> None:
    fixture = _make_preflight_fixture(tmp_path)
    backup_dir = fixture["output"] / "provenance_backups" / "chosen"
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in fixture["output"].rglob("*")
        if path.is_file()
    }

    plan = migration.prepare_migration(
        fixture["config"],
        verified_gate=fixture["verified_gate"],
        model_seeds=(0,),
        backup_dir=backup_dir,
    )

    assert plan.episode_paths == (fixture["base"], fixture["recovery"])
    assert tuple(path.parents[1].name for path in plan.checkpoint_paths) == (
        "flat",
        "graph",
    )
    assert plan.backup_dir == backup_dir
    assert not backup_dir.exists()
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in fixture["output"].rglob("*")
        if path.is_file()
    }
    assert after == before


def test_prepare_migration_rejects_backup_inside_data_tree(tmp_path: Path) -> None:
    fixture = _make_preflight_fixture(tmp_path)
    unsafe_backup = fixture["output"] / "data" / "nested-backup"

    with pytest.raises(ValueError, match="backup directory must not be inside data"):
        migration.prepare_migration(
            fixture["config"],
            verified_gate=fixture["verified_gate"],
            model_seeds=(0,),
            backup_dir=unsafe_backup,
        )

    assert not unsafe_backup.exists()


def test_prepare_migration_rejects_symbolic_links_in_data_tree(tmp_path: Path) -> None:
    fixture = _make_preflight_fixture(tmp_path)
    outside = tmp_path / "outside-data"
    outside.mkdir()
    (outside / "payload.bin").write_bytes(b"outside")
    link = fixture["output"] / "data" / "linked-directory"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic links"):
        migration.prepare_migration(
            fixture["config"],
            verified_gate=fixture["verified_gate"],
            model_seeds=(0,),
        )


def test_restore_rejects_manifest_parent_traversal(tmp_path: Path) -> None:
    fixture = _make_preflight_fixture(tmp_path)
    plan = migration.prepare_migration(
        fixture["config"],
        verified_gate=fixture["verified_gate"],
        model_seeds=(0,),
        backup_dir=fixture["output"] / "provenance_backups" / "tampered",
    )
    manifest_path = migration._create_backup(plan)
    crafted_backup = plan.backup_dir.parent / "escape.bin"
    crafted_backup.write_bytes(b"escape")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        {
            "relative_path": "../escape.bin",
            "sha256": migration._sha256(crafted_backup),
            "will_modify": True,
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    escaped_destination = fixture["output"].parent / "escape.bin"

    with pytest.raises(ValueError, match="unsafe backup manifest path"):
        migration._restore_backup(plan)

    assert not escaped_destination.exists()


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def test_rewrite_episode_changes_only_metadata_gate_hash(tmp_path: Path) -> None:
    episode = tmp_path / "episode.npz"
    metadata = {
        "backend": "franka_contact",
        "feature_schema": "physics_v2",
        "expert_gate_hash": "old-gate",
        "seed": 42,
    }
    np.savez_compressed(
        episode,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        floats=np.asarray([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32),
        indices=np.asarray([3, 1, 4], dtype=np.int64),
        mask=np.asarray([True, False], dtype=np.bool_),
    )
    before = _load_archive(episode)

    migration._rewrite_episode_gate_hash(episode, "old-gate", "new-gate")

    after = _load_archive(episode)
    after_metadata = json.loads(str(after.pop("metadata").item()))
    before_metadata = json.loads(str(before.pop("metadata").item()))
    assert after_metadata == {**before_metadata, "expert_gate_hash": "new-gate"}
    assert set(after) == set(before)
    for name in before:
        assert after[name].dtype == before[name].dtype
        assert after[name].shape == before[name].shape
        assert np.array_equal(after[name], before[name], equal_nan=True)


def test_rewrite_episode_rejects_unexpected_old_hash_without_mutation(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode.npz"
    np.savez_compressed(
        episode,
        metadata=np.asarray(json.dumps({"expert_gate_hash": "actual"})),
        values=np.asarray([1.0], dtype=np.float32),
    )
    before = hashlib.sha256(episode.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="unexpected expert_gate_hash"):
        migration._rewrite_episode_gate_hash(episode, "expected", "new")

    assert hashlib.sha256(episode.read_bytes()).hexdigest() == before


def _checkpoint_payload() -> dict[str, object]:
    return {
        "representation": "graph",
        "model_seed": 0,
        "expert_gate_hash": "old-gate",
        "controller_hash": "old-controller",
        "scene_hash": "scene",
        "config_hash": "config",
        "training_provenance": {"expert_gate_hash": "old-gate", "content": "old"},
        "model_state": {
            "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        },
        "optimizer_state": {
            "state": {0: {"step": torch.tensor(4), "exp_avg": torch.tensor([0.5])}},
            "param_groups": [{"lr": 0.001, "params": [0]}],
        },
        "statistics": {
            "mean": np.asarray([1.0, np.nan], dtype=np.float32),
            "labels": ("x", "y"),
        },
        "completed_epochs": 80,
        "global_step": 123,
    }


def test_rewrite_checkpoint_changes_only_approved_provenance_fields(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(_checkpoint_payload(), checkpoint)
    before = torch.load(checkpoint, map_location="cpu", weights_only=False)
    physical_hashes = {
        "expert_gate_hash": "new-gate",
        "controller_hash": "new-controller",
        "rollout_integrity_hash": "new-rollout",
        "scene_hash": "new-scene",
        "config_hash": "new-config",
    }
    training_provenance = {
        "expert_gate_hash": "new-gate",
        "content": "new",
    }

    migration._rewrite_checkpoint(
        checkpoint,
        new_physical_hashes=physical_hashes,
        new_training_provenance=training_provenance,
    )

    after = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for name, value in physical_hashes.items():
        assert after[name] == value
    assert after["training_provenance"] == training_provenance
    approved = {*physical_hashes, "training_provenance"}
    migration._assert_nested_equal(
        {key: value for key, value in before.items() if key not in approved},
        {key: value for key, value in after.items() if key not in approved},
        path="checkpoint",
    )


def test_recursive_invariant_reports_changed_tensor_path() -> None:
    expected = _checkpoint_payload()
    actual = _checkpoint_payload()
    actual["model_state"]["weight"][0, 1] = 9.0

    with pytest.raises(ValueError, match=r"checkpoint\.model_state\.weight"):
        migration._assert_nested_equal(expected, actual, path="checkpoint")


def test_recursive_invariant_supports_nested_exact_values() -> None:
    expected = {
        "tensor": torch.tensor([1.0, 2.0]),
        "array": np.asarray([1.0, np.nan]),
        "items": [1, ("x", None)],
    }
    actual = {
        "tensor": torch.tensor([1.0, 2.0]),
        "array": np.asarray([1.0, np.nan]),
        "items": [1, ("x", None)],
    }

    migration._assert_nested_equal(expected, actual, path="root")


def test_create_backup_copies_complete_recovery_surface(tmp_path: Path) -> None:
    fixture = _make_preflight_fixture(tmp_path)
    sentinel = fixture["output"] / "data" / "unreferenced-sentinel.bin"
    sentinel.write_bytes(b"keep this too")
    backup_dir = fixture["output"] / "provenance_backups" / "chosen"
    plan = migration.prepare_migration(
        fixture["config"],
        verified_gate=fixture["verified_gate"],
        model_seeds=(0,),
        backup_dir=backup_dir,
    )

    manifest_path = migration._create_backup(plan)

    assert manifest_path == backup_dir / "backup_manifest.json"
    assert (backup_dir / "expert_gate.json").is_file()
    assert (backup_dir / "data" / sentinel.name).read_bytes() == sentinel.read_bytes()
    assert (backup_dir / "flat" / "seed_0" / "checkpoint.pt").is_file()
    assert (backup_dir / "graph" / "seed_0" / "checkpoint.pt").is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    records = {record["relative_path"]: record for record in manifest["files"]}
    expected = {
        "expert_gate.json",
        *{
            path.relative_to(fixture["output"]).as_posix()
            for path in (fixture["output"] / "data").rglob("*")
            if path.is_file()
        },
        "flat/seed_0/checkpoint.pt",
        "graph/seed_0/checkpoint.pt",
    }
    assert set(records) == expected
    for relative_path, record in records.items():
        source = fixture["output"] / relative_path
        backup = backup_dir / relative_path
        assert migration._sha256(source) == record["sha256"]
        assert migration._sha256(backup) == record["sha256"]
    assert records["data/unreferenced-sentinel.bin"]["will_modify"] is False
    assert records[fixture["base"].relative_to(fixture["output"]).as_posix()][
        "will_modify"
    ] is True


def test_migration_failure_restores_all_original_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_preflight_fixture(tmp_path)
    sentinel = fixture["output"] / "data" / "unreferenced-sentinel.bin"
    sentinel.write_bytes(b"preserve")
    backup_dir = fixture["output"] / "provenance_backups" / "rollback-case"
    protected = [fixture["output"] / "expert_gate.json"]
    protected.extend(
        sorted(path for path in (fixture["output"] / "data").rglob("*") if path.is_file())
    )
    protected.extend(
        fixture["output"] / representation / "seed_0" / "checkpoint.pt"
        for representation in ("flat", "graph")
    )
    original_digests = {path: migration._sha256(path) for path in protected}

    def fail_checkpoint(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected checkpoint failure")

    monkeypatch.setattr(migration, "_rewrite_checkpoint", fail_checkpoint)
    with pytest.raises(RuntimeError, match="rollback completed"):
        migration.migrate_from_config(
            fixture["config"],
            verified_gate=fixture["verified_gate"],
            model_seeds=(0,),
            backup_dir=backup_dir,
        )

    assert backup_dir.is_dir()
    assert (backup_dir / "backup_manifest.json").is_file()
    assert {path: migration._sha256(path) for path in protected} == original_digests


def test_successful_migration_validates_and_writes_audit_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_preflight_fixture(tmp_path)
    backup_dir = fixture["output"] / "provenance_backups" / "success-case"
    preload_call: dict[str, object] = {}

    def fake_preload(config, **kwargs: object) -> dict:
        preload_call.update({"config": config, **kwargs})
        return {(0, "flat"): (object(), object()), (0, "graph"): (object(), object())}

    monkeypatch.setattr(migration, "preload_evaluation_checkpoints", fake_preload)

    report_path = migration.migrate_from_config(
        fixture["config"],
        verified_gate=fixture["verified_gate"],
        model_seeds=(0,),
        backup_dir=backup_dir,
    )

    assert report_path == fixture["output"] / "provenance_migration.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["migrated_episode_count"] == 2
    assert report["migrated_checkpoint_count"] == 2
    assert report["backup_dir"] == str(backup_dir.resolve())
    assert all(report["invariants"].values())
    assert report["old_dataset_content_hash"] != report["new_dataset_content_hash"]
    assert preload_call["model_seeds"] == (0,)
    assert preload_call["device"] == "cpu"
    assert backup_dir.is_dir()
    new_gate_hash = migration._sha256(fixture["output"] / "expert_gate.json")
    assert new_gate_hash == migration._sha256(fixture["verified_gate"])
    for episode in (fixture["base"], fixture["recovery"]):
        with np.load(episode, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
        assert metadata["expert_gate_hash"] == new_gate_hash


def test_migration_progress_covers_every_mutation_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_preflight_fixture(tmp_path)
    instances: list[object] = []

    class FakeProgress:
        def __init__(self, **kwargs: object) -> None:
            self.total = kwargs["total"]
            self.count = 0
            self.closed = False
            self.postfixes: list[dict[str, object]] = []
            instances.append(self)

        def set_postfix(self, **kwargs: object) -> None:
            self.postfixes.append(kwargs)

        def update(self, amount: int) -> None:
            self.count += amount

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(migration, "tqdm", FakeProgress, raising=False)
    monkeypatch.setattr(
        migration,
        "preload_evaluation_checkpoints",
        lambda *_args, **_kwargs: {
            (0, "flat"): (object(), object()),
            (0, "graph"): (object(), object()),
        },
    )

    migration.migrate_from_config(
        fixture["config"],
        verified_gate=fixture["verified_gate"],
        model_seeds=(0,),
        backup_dir=fixture["output"] / "provenance_backups" / "progress-case",
        show_progress=True,
    )

    assert len(instances) == 1
    progress = instances[0]
    assert progress.total == 6
    assert progress.count == 6
    assert progress.closed is True
    assert {value["phase"] for value in progress.postfixes} == {
        "episodes",
        "gate",
        "checkpoints",
        "validation",
    }


def test_progress_close_failure_rolls_back_completed_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_preflight_fixture(tmp_path)
    protected = [fixture["output"] / "expert_gate.json"]
    protected.extend(
        sorted(path for path in (fixture["output"] / "data").rglob("*") if path.is_file())
    )
    protected.extend(
        fixture["output"] / representation / "seed_0" / "checkpoint.pt"
        for representation in ("flat", "graph")
    )
    original_digests = {path: migration._sha256(path) for path in protected}

    class CloseFailureProgress:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def set_postfix(self, **_kwargs: object) -> None:
            pass

        def update(self, _amount: int) -> None:
            pass

        def close(self) -> None:
            raise RuntimeError("injected progress close failure")

    monkeypatch.setattr(migration, "tqdm", CloseFailureProgress)
    monkeypatch.setattr(
        migration,
        "preload_evaluation_checkpoints",
        lambda *_args, **_kwargs: {
            (0, "flat"): (object(), object()),
            (0, "graph"): (object(), object()),
        },
    )

    with pytest.raises(RuntimeError, match="rollback completed"):
        migration.migrate_from_config(
            fixture["config"],
            verified_gate=fixture["verified_gate"],
            model_seeds=(0,),
            backup_dir=fixture["output"] / "provenance_backups" / "close-failure",
            show_progress=True,
        )

    assert {path: migration._sha256(path) for path in protected} == original_digests


def test_keyboard_interrupt_after_mutation_restores_original_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_preflight_fixture(tmp_path)
    protected = [fixture["output"] / "expert_gate.json"]
    protected.extend(
        sorted(path for path in (fixture["output"] / "data").rglob("*") if path.is_file())
    )
    protected.extend(
        fixture["output"] / representation / "seed_0" / "checkpoint.pt"
        for representation in ("flat", "graph")
    )
    original_digests = {path: migration._sha256(path) for path in protected}

    def interrupt_checkpoint(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt("injected interrupt")

    monkeypatch.setattr(migration, "_rewrite_checkpoint", interrupt_checkpoint)

    with pytest.raises(KeyboardInterrupt, match="injected interrupt"):
        migration.migrate_from_config(
            fixture["config"],
            verified_gate=fixture["verified_gate"],
            model_seeds=(0,),
            backup_dir=fixture["output"] / "provenance_backups" / "interrupt",
        )

    assert {path: migration._sha256(path) for path in protected} == original_digests


def test_migration_parser_and_main_forward_cli_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = migration.build_parser().parse_args(
        [
            "--config",
            "pilot.yaml",
            "--verified-gate",
            "verified.json",
            "--model-seeds",
            "0",
            "2",
            "--backup-dir",
            "backup",
        ]
    )
    assert args.model_seeds == [0, 2]
    captured: dict[str, object] = {}
    report = tmp_path / "report.json"

    def fake_migrate(config_path: str, **kwargs: object) -> Path:
        captured.update({"config_path": config_path, **kwargs})
        return report

    monkeypatch.setattr(migration, "migrate_from_config", fake_migrate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_physics_provenance",
            "--config",
            "pilot.yaml",
            "--verified-gate",
            "verified.json",
            "--model-seeds",
            "0",
            "--backup-dir",
            "backup",
        ],
    )

    migration.main()

    assert captured == {
        "config_path": "pilot.yaml",
        "verified_gate": "verified.json",
        "model_seeds": [0],
        "backup_dir": "backup",
        "show_progress": True,
    }
    assert capsys.readouterr().out.strip() == str(report)
