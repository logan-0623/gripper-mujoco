from __future__ import annotations

"""Verified migration of physical pilot provenance metadata."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import load_config
from .data import episode_paths_from_manifest, recovery_paths_from_manifest
from .physics_data import (
    expected_gate_hashes,
    expert_gate_provenance,
    require_episode_gate_provenance,
)
from .physics_evaluate import preload_evaluation_checkpoints
from .train import build_training_provenance, resolve_training_data


@dataclass(frozen=True)
class MigrationPlan:
    config_path: Path
    output_dir: Path
    data_dir: Path
    official_gate: Path
    verified_gate: Path
    backup_dir: Path
    episode_paths: tuple[Path, ...]
    checkpoint_paths: tuple[Path, ...]
    model_seeds: tuple[int, ...]
    old_gate_hash: str
    new_gate_hash: str
    old_hashes: Mapping[str, str]
    new_hashes: Mapping[str, str]
    old_training_provenance: Mapping[str, object]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_no_symlinks(root: Path, *, label: str) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError(f"{label} must not contain symbolic links")


def _artifact_relative(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"artifact is outside configured output directory: {path}")
    return resolved_path.relative_to(resolved_root)


def _safe_manifest_relative(value: object) -> Path:
    relative = Path(str(value))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe backup manifest path: {value}")
    return relative


def _load_gate(path: str | Path) -> dict[str, object]:
    gate_path = Path(path)
    report = json.loads(gate_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"expert gate must contain a JSON object: {gate_path}")
    if report.get("passed") is not True:
        raise ValueError(f"expert gate did not pass: {gate_path}")
    return report


def assert_gate_reports_equivalent(
    old_gate: str | Path, verified_gate: str | Path
) -> None:
    old_report = _load_gate(old_gate)
    verified_report = _load_gate(verified_gate)
    normalized_old = dict(old_report)
    normalized_old["controller_hash"] = verified_report.get("controller_hash")
    if "rollout_integrity_hash" in verified_report:
        normalized_old["rollout_integrity_hash"] = verified_report.get(
            "rollout_integrity_hash"
        )
    if normalized_old != verified_report:
        raise ValueError(
            "expert gate reports differ outside controller_hash/rollout_integrity_hash"
        )


def _episode_paths(data_dir: Path, *, recovery_enabled: bool) -> tuple[Path, ...]:
    paths = episode_paths_from_manifest(data_dir)
    if recovery_enabled:
        paths += recovery_paths_from_manifest(data_dir)
    if len(paths) != len(set(paths)):
        raise ValueError("base and recovery manifests contain duplicate episode paths")
    return paths


def _checkpoint_paths(
    output_dir: Path,
    *,
    model_seeds: Iterable[int],
    configured_seeds: tuple[int, ...],
) -> tuple[tuple[int, str, Path], ...]:
    selected = tuple(int(seed) for seed in model_seeds)
    if not selected:
        raise ValueError("at least one model seed is required")
    if len(selected) != len(set(selected)):
        raise ValueError("model seeds must be unique")
    unknown = tuple(seed for seed in selected if seed not in configured_seeds)
    if unknown:
        raise ValueError(f"model seeds are not configured: {unknown}")
    paths = tuple(
        (
            seed,
            representation,
            output_dir / representation / f"seed_{seed}" / "checkpoint.pt",
        )
        for seed in selected
        for representation in ("flat", "graph")
    )
    missing = tuple(path for _, _, path in paths if not path.is_file())
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"missing selected checkpoints:\n{formatted}")
    return paths


def _default_backup_dir(output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return output_dir / "provenance_backups" / timestamp


def prepare_migration(
    config_path: str | Path,
    *,
    verified_gate: str | Path,
    model_seeds: Iterable[int] = (0,),
    backup_dir: str | Path | None = None,
) -> MigrationPlan:
    resolved_config_path = Path(config_path)
    config = load_config(resolved_config_path)
    if config.backend != "franka_contact":
        raise ValueError("provenance migration requires backend=franka_contact")

    output_dir = Path(config.output_dir)
    data_dir = Path(config.data_dir)
    official_gate = output_dir / "expert_gate.json"
    resolved_verified_gate = Path(verified_gate)
    _assert_no_symlinks(data_dir, label="data directory")
    for gate_path in (official_gate, resolved_verified_gate):
        if gate_path.is_symlink():
            raise ValueError(f"expert gate must not be a symbolic link: {gate_path}")
    old_report = _load_gate(official_gate)
    verified_report = _load_gate(resolved_verified_gate)
    assert_gate_reports_equivalent(official_gate, resolved_verified_gate)

    current_hashes = expected_gate_hashes(resolved_config_path)
    if any(verified_report.get(name) != value for name, value in current_hashes.items()):
        raise ValueError(
            "verified expert gate is stale for the current config, scene, or controller"
        )
    for name in ("scene_hash", "config_hash"):
        if old_report.get(name) != current_hashes[name]:
            raise ValueError(f"old expert gate has unexpected {name}")
    old_hashes = {
        name: str(old_report[name])
        for name in (
            "controller_hash",
            "rollout_integrity_hash",
            "scene_hash",
            "config_hash",
        )
        if name in old_report
    }
    old_gate_hash = _sha256(official_gate)
    new_gate_hash = _sha256(resolved_verified_gate)

    episode_paths = _episode_paths(
        data_dir,
        recovery_enabled=config.recovery.enabled,
    )
    for episode_path in episode_paths:
        with np.load(episode_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
        if (
            metadata.get("backend") != "franka_contact"
            or metadata.get("feature_schema") != "physics_v2"
            or metadata.get("expert_gate_hash") != old_gate_hash
        ):
            raise ValueError(
                f"episode does not match the old physical provenance chain: {episode_path}"
            )

    selection = resolve_training_data(
        data_dir,
        split_seed=config.seed,
        include_recovery=config.recovery.enabled,
    )
    old_training_provenance, provenance_paths = build_training_provenance(
        config,
        selection,
        expert_gate_hash=old_gate_hash,
    )
    if provenance_paths != episode_paths:
        raise ValueError("training provenance paths differ from migration episode paths")

    checkpoint_records = _checkpoint_paths(
        output_dir,
        model_seeds=model_seeds,
        configured_seeds=config.train.model_seeds,
    )
    for _, _, checkpoint_path in checkpoint_records:
        if checkpoint_path.is_symlink():
            raise ValueError(
                f"checkpoint must not be a symbolic link: {checkpoint_path}"
            )
    for seed, representation, checkpoint_path in checkpoint_records:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_top_level: dict[str, object] = {
            "representation": representation,
            "model_seed": seed,
            "backend": "franka_contact",
            "feature_schema": "physics_v2",
            "expert_gate_hash": old_gate_hash,
            **old_hashes,
        }
        if any(payload.get(name) != value for name, value in expected_top_level.items()):
            raise ValueError(
                f"checkpoint does not match the old physical provenance chain: {checkpoint_path}"
            )
        if dict(payload.get("training_provenance", {})) != old_training_provenance:
            raise ValueError(
                f"checkpoint training provenance does not match old data: {checkpoint_path}"
            )

    resolved_backup_dir = (
        _default_backup_dir(output_dir)
        if backup_dir is None
        else Path(backup_dir)
    )
    if resolved_backup_dir.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError("backup directory must not be inside data directory")
    if resolved_backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {resolved_backup_dir}")
    return MigrationPlan(
        config_path=resolved_config_path,
        output_dir=output_dir,
        data_dir=data_dir,
        official_gate=official_gate,
        verified_gate=resolved_verified_gate,
        backup_dir=resolved_backup_dir,
        episode_paths=episode_paths,
        checkpoint_paths=tuple(path for _, _, path in checkpoint_records),
        model_seeds=tuple(seed for seed, _, _ in checkpoint_records[::2]),
        old_gate_hash=old_gate_hash,
        new_gate_hash=new_gate_hash,
        old_hashes=old_hashes,
        new_hashes=current_hashes,
        old_training_provenance=old_training_provenance,
    )


@contextmanager
def _atomic_path(destination: Path):
    destination = Path(destination)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=destination.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left, right))


def _rewrite_episode_gate_hash(
    path: str | Path,
    old_gate_hash: str,
    new_gate_hash: str,
) -> None:
    episode_path = Path(path)
    with np.load(episode_path, allow_pickle=False) as archive:
        before = {name: archive[name].copy() for name in archive.files}
    if "metadata" not in before:
        raise ValueError(f"episode metadata is missing: {episode_path}")
    old_metadata = json.loads(str(before["metadata"].item()))
    if old_metadata.get("expert_gate_hash") != old_gate_hash:
        raise ValueError(f"episode has unexpected expert_gate_hash: {episode_path}")
    new_metadata = {**old_metadata, "expert_gate_hash": new_gate_hash}
    rewritten = dict(before)
    rewritten["metadata"] = np.asarray(json.dumps(new_metadata, sort_keys=True))

    with _atomic_path(episode_path) as temporary:
        np.savez_compressed(temporary, **rewritten)
        with np.load(temporary, allow_pickle=False) as archive:
            after = {name: archive[name].copy() for name in archive.files}
        if set(after) != set(before):
            raise ValueError(f"episode array keys changed during migration: {episode_path}")
        for name, original in before.items():
            if name == "metadata":
                continue
            migrated = after[name]
            if (
                migrated.dtype != original.dtype
                or migrated.shape != original.shape
                or not _array_equal(migrated, original)
            ):
                raise ValueError(
                    f"episode array changed during provenance migration: {episode_path}:{name}"
                )
        reloaded_metadata = json.loads(str(after["metadata"].item()))
        if reloaded_metadata != new_metadata:
            raise ValueError(f"episode metadata changed unexpectedly: {episode_path}")


def _invariant_error(path: str, detail: str) -> ValueError:
    return ValueError(f"provenance invariant failed at {path}: {detail}")


def _assert_nested_equal(expected: object, actual: object, *, path: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise _invariant_error(path, "tensor type changed")
        expected_cpu = expected.detach().cpu()
        actual_cpu = actual.detach().cpu()
        if (
            expected_cpu.dtype != actual_cpu.dtype
            or expected_cpu.shape != actual_cpu.shape
            or not torch.equal(expected_cpu, actual_cpu)
        ):
            raise _invariant_error(path, "tensor value, dtype, or shape changed")
        return
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray):
            raise _invariant_error(path, "array type changed")
        if (
            expected.dtype != actual.dtype
            or expected.shape != actual.shape
            or not _array_equal(expected, actual)
        ):
            raise _invariant_error(path, "array value, dtype, or shape changed")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or type(expected) is not type(actual):
            raise _invariant_error(path, "mapping type changed")
        if set(expected) != set(actual):
            raise _invariant_error(path, "mapping keys changed")
        for key in expected:
            _assert_nested_equal(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, Sequence) and not isinstance(
        expected, (str, bytes, bytearray)
    ):
        if type(expected) is not type(actual):
            raise _invariant_error(path, "sequence type changed")
        if len(expected) != len(actual):
            raise _invariant_error(path, "sequence length changed")
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_nested_equal(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
            )
        return
    if type(expected) is not type(actual):
        raise _invariant_error(path, "value type changed")
    if isinstance(expected, float) and math.isnan(expected) and math.isnan(actual):
        return
    if expected != actual:
        raise _invariant_error(path, "value changed")


def _rewrite_checkpoint(
    path: str | Path,
    *,
    new_physical_hashes: Mapping[str, str],
    new_training_provenance: Mapping[str, object],
) -> None:
    checkpoint_path = Path(path)
    required_physical_keys = {
        "expert_gate_hash",
        "controller_hash",
        "rollout_integrity_hash",
        "scene_hash",
        "config_hash",
    }
    if set(new_physical_hashes) != required_physical_keys:
        raise ValueError("new physical hashes must contain the complete provenance key set")
    original = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    migrated = dict(original)
    migrated.update(dict(new_physical_hashes))
    migrated["training_provenance"] = dict(new_training_provenance)
    approved_keys = required_physical_keys | {"training_provenance"}

    with _atomic_path(checkpoint_path) as temporary:
        torch.save(migrated, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        _assert_nested_equal(
            {key: value for key, value in original.items() if key not in approved_keys},
            {key: value for key, value in reloaded.items() if key not in approved_keys},
            path="checkpoint",
        )
        for name, value in new_physical_hashes.items():
            if reloaded.get(name) != value:
                raise _invariant_error(f"checkpoint.{name}", "provenance value changed")
        if dict(reloaded.get("training_provenance", {})) != dict(
            new_training_provenance
        ):
            raise _invariant_error(
                "checkpoint.training_provenance",
                "training provenance changed",
            )


def _create_backup(plan: MigrationPlan) -> Path:
    if plan.backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {plan.backup_dir}")
    _assert_no_symlinks(plan.data_dir, label="data directory")
    plan.backup_dir.mkdir(parents=True)
    backup_root = plan.backup_dir.resolve()
    shutil.copy2(
        plan.official_gate,
        backup_root / "expert_gate.json",
        follow_symlinks=False,
    )
    shutil.copytree(plan.data_dir, backup_root / "data", symlinks=True)
    _assert_no_symlinks(backup_root / "data", label="backed-up data directory")
    for checkpoint in plan.checkpoint_paths:
        if checkpoint.is_symlink():
            raise ValueError(f"checkpoint must not be a symbolic link: {checkpoint}")
        relative = _artifact_relative(checkpoint, plan.output_dir)
        destination = backup_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, destination, follow_symlinks=False)

    previous_report = plan.output_dir / "provenance_migration.json"
    if previous_report.is_file():
        if previous_report.is_symlink():
            raise ValueError(
                f"previous migration report must not be a symbolic link: {previous_report}"
            )
        shutil.copy2(
            previous_report,
            backup_root / previous_report.name,
            follow_symlinks=False,
        )

    episode_set = set(plan.episode_paths)
    sources = [plan.official_gate]
    sources.extend(
        sorted(path for path in plan.data_dir.rglob("*") if path.is_file())
    )
    sources.extend(plan.checkpoint_paths)
    if previous_report.is_file():
        sources.append(previous_report)
    records: list[dict[str, object]] = []
    for source in sources:
        if source.is_symlink():
            raise ValueError(f"backup source must not be a symbolic link: {source}")
        relative = _artifact_relative(source, plan.output_dir)
        backup = backup_root / relative
        digest = _sha256(source)
        if not backup.is_file() or _sha256(backup) != digest:
            raise RuntimeError(f"backup digest verification failed: {source}")
        records.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": digest,
                "will_modify": (
                    source == plan.official_gate
                    or source in episode_set
                    or source in plan.checkpoint_paths
                    or source == previous_report
                ),
            }
        )
    manifest_payload = {
        "schema_version": 1,
        "source_output_dir": str(plan.output_dir.resolve()),
        "files": records,
    }
    manifest_path = backup_root / "backup_manifest.json"
    with _atomic_path(manifest_path) as temporary:
        temporary.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return manifest_path


def _restore_backup(plan: MigrationPlan) -> None:
    manifest_path = plan.backup_dir / "backup_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("files"), list
    ):
        raise RuntimeError(f"invalid backup manifest: {manifest_path}")
    output_root = plan.output_dir.resolve()
    backup_root = plan.backup_dir.resolve()
    validated: list[tuple[dict[str, object], Path, Path]] = []
    for record in manifest["files"]:
        relative = _safe_manifest_relative(record.get("relative_path"))
        backup = (backup_root / relative).resolve()
        destination = (output_root / relative).resolve(strict=False)
        if not backup.is_relative_to(backup_root) or not destination.is_relative_to(
            output_root
        ):
            raise ValueError(f"unsafe backup manifest path: {relative}")
        if backup.is_symlink() or destination.is_symlink():
            raise ValueError(f"unsafe symbolic link in backup manifest path: {relative}")
        validated.append((record, backup, destination))
    for record, backup, destination in validated:
        if not backup.is_file():
            raise RuntimeError(f"backup file is missing: {backup}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, destination, follow_symlinks=False)
    for record, _, destination in validated:
        if not destination.is_file() or _sha256(destination) != record["sha256"]:
            raise RuntimeError(f"rollback digest verification failed: {destination}")


def _promote_verified_gate(plan: MigrationPlan) -> None:
    verified_bytes = plan.verified_gate.read_bytes()
    with _atomic_path(plan.official_gate) as temporary:
        temporary.write_bytes(verified_bytes)
        if _sha256(temporary) != plan.new_gate_hash:
            raise RuntimeError("verified gate digest changed during promotion")


def _post_validate(
    plan: MigrationPlan,
    new_training_provenance: Mapping[str, object],
) -> dict[str, bool]:
    expected_physical_hashes = {
        "expert_gate_hash": plan.new_gate_hash,
        **plan.new_hashes,
    }
    observed_physical_hashes = expert_gate_provenance(
        plan.config_path,
        plan.official_gate,
    )
    if observed_physical_hashes != expected_physical_hashes:
        raise ValueError("promoted expert gate failed production provenance validation")
    require_episode_gate_provenance(plan.episode_paths, plan.new_gate_hash)

    config = load_config(plan.config_path)
    selection = resolve_training_data(
        plan.data_dir,
        split_seed=config.seed,
        include_recovery=config.recovery.enabled,
    )
    recomputed_provenance, provenance_paths = build_training_provenance(
        config,
        selection,
        expert_gate_hash=plan.new_gate_hash,
    )
    if provenance_paths != plan.episode_paths or recomputed_provenance != dict(
        new_training_provenance
    ):
        raise ValueError("post-migration training provenance does not recompute exactly")

    for checkpoint_path in plan.checkpoint_paths:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if any(
            payload.get(name) != value
            for name, value in expected_physical_hashes.items()
        ):
            raise ValueError(
                f"checkpoint failed migrated physical provenance: {checkpoint_path}"
            )
        if dict(payload.get("training_provenance", {})) != recomputed_provenance:
            raise ValueError(
                f"checkpoint failed migrated training provenance: {checkpoint_path}"
            )

    loaded = preload_evaluation_checkpoints(
        config,
        model_seeds=plan.model_seeds,
        device="cpu",
        physical_hashes=expected_physical_hashes,
        expected_training_provenance=recomputed_provenance,
    )
    if len(loaded) != 2 * len(plan.model_seeds):
        raise ValueError("production checkpoint preload returned an incomplete policy set")

    for episode_path in plan.episode_paths:
        relative = episode_path.relative_to(plan.output_dir)
        backup_path = plan.backup_dir / relative
        with np.load(backup_path, allow_pickle=False) as archive:
            before = {name: archive[name].copy() for name in archive.files}
        with np.load(episode_path, allow_pickle=False) as archive:
            after = {name: archive[name].copy() for name in archive.files}
        if set(before) != set(after):
            raise ValueError(f"episode keys differ from backup: {episode_path}")
        for name in before:
            if name != "metadata" and (
                before[name].dtype != after[name].dtype
                or before[name].shape != after[name].shape
                or not _array_equal(before[name], after[name])
            ):
                raise ValueError(f"episode array differs from backup: {episode_path}:{name}")
        old_metadata = json.loads(str(before["metadata"].item()))
        new_metadata = json.loads(str(after["metadata"].item()))
        if new_metadata != {**old_metadata, "expert_gate_hash": plan.new_gate_hash}:
            raise ValueError(f"episode metadata differs outside gate hash: {episode_path}")

    approved_keys = {
        "expert_gate_hash",
        "controller_hash",
        "rollout_integrity_hash",
        "scene_hash",
        "config_hash",
        "training_provenance",
    }
    for checkpoint_path in plan.checkpoint_paths:
        relative = checkpoint_path.relative_to(plan.output_dir)
        backup_payload = torch.load(
            plan.backup_dir / relative,
            map_location="cpu",
            weights_only=False,
        )
        current_payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        _assert_nested_equal(
            {
                key: value
                for key, value in backup_payload.items()
                if key not in approved_keys
            },
            {
                key: value
                for key, value in current_payload.items()
                if key not in approved_keys
            },
            path=f"checkpoint:{relative.as_posix()}",
        )

    backup_manifest = json.loads(
        (plan.backup_dir / "backup_manifest.json").read_text(encoding="utf-8")
    )
    for record in backup_manifest["files"]:
        backup_path = plan.backup_dir / str(record["relative_path"])
        if _sha256(backup_path) != record["sha256"]:
            raise ValueError(f"backup no longer matches its manifest: {backup_path}")
    return {
        "gate_reports_equivalent": True,
        "backup_verified": True,
        "episode_arrays_unchanged": True,
        "checkpoint_payloads_unchanged": True,
        "production_gate_validation": True,
        "production_episode_validation": True,
        "training_provenance_recomputed": True,
        "production_checkpoint_preload": True,
        "rollback_available": True,
    }


def _write_migration_report(
    plan: MigrationPlan,
    new_training_provenance: Mapping[str, object],
    invariant_summary: Mapping[str, bool],
) -> Path:
    report = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(plan.config_path.resolve()),
        "verified_gate": str(plan.verified_gate.resolve()),
        "output_dir": str(plan.output_dir.resolve()),
        "data_dir": str(plan.data_dir.resolve()),
        "backup_dir": str(plan.backup_dir.resolve()),
        "old_gate_hash": plan.old_gate_hash,
        "new_gate_hash": plan.new_gate_hash,
        "old_physical_hashes": dict(plan.old_hashes),
        "new_physical_hashes": dict(plan.new_hashes),
        "migrated_episode_count": len(plan.episode_paths),
        "migrated_checkpoint_count": len(plan.checkpoint_paths),
        "migrated_episodes": [
            path.relative_to(plan.output_dir).as_posix()
            for path in plan.episode_paths
        ],
        "migrated_checkpoints": [
            path.relative_to(plan.output_dir).as_posix()
            for path in plan.checkpoint_paths
        ],
        "old_dataset_content_hash": plan.old_training_provenance[
            "dataset_content_hash"
        ],
        "new_dataset_content_hash": new_training_provenance[
            "dataset_content_hash"
        ],
        "manifest_hash": new_training_provenance["manifest_hash"],
        "recovery_manifest_hash": new_training_provenance.get(
            "recovery_manifest_hash"
        ),
        "invariants": dict(invariant_summary),
    }
    destination = plan.output_dir / "provenance_migration.json"
    with _atomic_path(destination) as temporary:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        reloaded = json.loads(temporary.read_text(encoding="utf-8"))
        if reloaded != report:
            raise RuntimeError("provenance migration report failed round-trip validation")
    return destination


def migrate_from_config(
    config_path: str | Path,
    *,
    verified_gate: str | Path,
    model_seeds: Iterable[int] = (0,),
    backup_dir: str | Path | None = None,
    show_progress: bool = False,
) -> Path:
    plan = prepare_migration(
        config_path,
        verified_gate=verified_gate,
        model_seeds=model_seeds,
        backup_dir=backup_dir,
    )
    _create_backup(plan)
    progress = (
        tqdm(
            total=len(plan.episode_paths) + len(plan.checkpoint_paths) + 2,
            desc="provenance migration",
            unit="artifact",
        )
        if show_progress
        else None
    )
    try:
        for episode_path in plan.episode_paths:
            _rewrite_episode_gate_hash(
                episode_path,
                plan.old_gate_hash,
                plan.new_gate_hash,
            )
            if progress is not None:
                progress.set_postfix(
                    phase="episodes",
                    artifact=episode_path.relative_to(plan.output_dir).as_posix(),
                )
                progress.update(1)
        _promote_verified_gate(plan)
        if progress is not None:
            progress.set_postfix(phase="gate", artifact="expert_gate.json")
            progress.update(1)

        config = load_config(plan.config_path)
        selection = resolve_training_data(
            plan.data_dir,
            split_seed=config.seed,
            include_recovery=config.recovery.enabled,
        )
        new_training_provenance, provenance_paths = build_training_provenance(
            config,
            selection,
            expert_gate_hash=plan.new_gate_hash,
        )
        if provenance_paths != plan.episode_paths:
            raise ValueError("migrated training provenance paths changed")
        new_physical_hashes = {
            "expert_gate_hash": plan.new_gate_hash,
            **plan.new_hashes,
        }
        for checkpoint_path in plan.checkpoint_paths:
            _rewrite_checkpoint(
                checkpoint_path,
                new_physical_hashes=new_physical_hashes,
                new_training_provenance=new_training_provenance,
            )
            if progress is not None:
                progress.set_postfix(
                    phase="checkpoints",
                    artifact=checkpoint_path.relative_to(plan.output_dir).as_posix(),
                )
                progress.update(1)

        invariant_summary = _post_validate(plan, new_training_provenance)
        if progress is not None:
            progress.set_postfix(phase="validation", artifact="production paths")
            progress.update(1)
            completed_progress = progress
            progress = None
            completed_progress.close()
        return _write_migration_report(
            plan,
            new_training_provenance,
            invariant_summary,
        )
    except BaseException as error:
        try:
            _restore_backup(plan)
        except Exception as rollback_error:
            raise RuntimeError(
                "provenance migration failed and rollback verification failed; "
                f"backup: {plan.backup_dir}; original error: {error}"
            ) from rollback_error
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            error.add_note(f"provenance migration rolled back; backup: {plan.backup_dir}")
            raise
        raise RuntimeError(
            "provenance migration failed; rollback completed; "
            f"backup: {plan.backup_dir}"
        ) from error
    finally:
        if progress is not None:
            try:
                progress.close()
            except BaseException:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate verified physical provenance without changing numeric artifacts"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--verified-gate", required=True)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--backup-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = migrate_from_config(
        args.config,
        verified_gate=args.verified_gate,
        model_seeds=args.model_seeds,
        backup_dir=args.backup_dir,
        show_progress=True,
    )
    print(report)


if __name__ == "__main__":
    main()
