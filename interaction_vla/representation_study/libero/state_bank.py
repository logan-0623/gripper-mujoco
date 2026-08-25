from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from ..state_bank.io import write_bytes_atomic, write_json_atomic
from .audit import build_state_bank_audit
from .schema import STATE_BANK_SCHEMA, StateRecord
from .splits import (
    SplitManifest,
    build_episode_group_split,
    build_task_group_split,
    validate_split,
)


MANIFEST_SCHEMA = "libero_interaction_state_bank_manifest_v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _records_bytes(records: Sequence[StateRecord]) -> bytes:
    return b"".join(
        (
            json.dumps(
                record.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )


class EpisodeShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        source_binding_sha256: str,
        compatible_source_bindings: Sequence[str] = (),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.source_binding_sha256 = source_binding_sha256
        self.compatible_source_bindings = frozenset(compatible_source_bindings)
        self.shards = self.output_dir / ".episode_shards"

    def _path(self, episode_key: str) -> Path:
        episode_hash = hashlib.sha256(episode_key.encode("utf-8")).hexdigest()
        return self.shards / f"{episode_hash}.json"

    def load(
        self, episode_key: str
    ) -> tuple[tuple[StateRecord, ...], Mapping[str, object]] | None:
        path = self._path(episode_key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        shard_binding = payload.get("source_binding_sha256")
        if payload.get("episode_key") != episode_key or shard_binding not in {
            self.source_binding_sha256,
            *self.compatible_source_bindings,
        }:
            raise ValueError(f"stale episode shard: {path}")
        records = tuple(StateRecord.from_dict(item) for item in payload.get("records", ()))
        if _sha256_bytes(_records_bytes(records)) != payload.get("record_sha256"):
            raise ValueError(f"corrupt episode shard: {path}")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"episode shard metadata is invalid: {path}")
        return records, metadata

    def write(
        self,
        episode_key: str,
        records: Sequence[StateRecord],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        path = self._path(episode_key)
        payload = {
            "schema_version": "libero_state_bank_episode_shard_v1",
            "episode_key": episode_key,
            "source_binding_sha256": self.source_binding_sha256,
            "record_sha256": _sha256_bytes(_records_bytes(records)),
            "records": [record.to_dict() for record in records],
            "metadata": dict(metadata or {}),
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (
                existing.get("source_binding_sha256") != self.source_binding_sha256
                or existing.get("record_sha256") != payload["record_sha256"]
            ):
                raise ValueError(f"stale episode shard: {path}")
            return path
        write_json_atomic(path, payload)
        return path


def finalize_state_bank(
    records: Sequence[StateRecord],
    *,
    output_dir: str | Path,
    source_binding_sha256: str,
    ontology_sha256: str,
    config_sha256: str | None = None,
    alignment_sha256: str | None = None,
    split_seed: int,
    task_ratios: tuple[float, float, float],
    episode_ratios: tuple[float, float, float],
    replay_statistics: Mapping[str, object],
    minimum_acceptance_rate: float = 0.95,
    l2_p95_tolerance: float = 0.01,
    max_abs_tolerance: float = 0.05,
) -> dict[str, object]:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        _, existing, _, _ = load_state_bank(root)
        if (
            existing.get("source_binding_sha256") != source_binding_sha256
            or existing.get("ontology_sha256") != ontology_sha256
            or (
                config_sha256 is not None
                and existing.get("config_sha256") != config_sha256
            )
            or (
                alignment_sha256 is not None
                and existing.get("alignment_sha256") != alignment_sha256
            )
        ):
            raise FileExistsError(
                f"State Bank output has a different scientific binding: {root}"
            )
        audit_path = root / "audit" / "report.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("passed"):
            raise ValueError("existing State Bank has a failed audit report")
        return audit
    if not records:
        raise ValueError("cannot finalize an empty State Bank")
    ordered = tuple(
        sorted(
            records,
            key=lambda record: (
                record.suite,
                record.task_id,
                record.source_episode_id,
                record.frame_index,
            ),
        )
    )
    state_ids = [record.state_id for record in ordered]
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("State Bank state IDs must be unique")
    task_split = build_task_group_split(ordered, ratios=task_ratios, seed=split_seed)
    episode_split = build_episode_group_split(
        ordered, ratios=episode_ratios, seed=split_seed
    )
    task_validation = validate_split(ordered, task_split)
    episode_validation = validate_split(ordered, episode_split)
    audit = build_state_bank_audit(
        ordered,
        replay_statistics=replay_statistics,
        minimum_acceptance_rate=minimum_acceptance_rate,
        l2_p95_tolerance=l2_p95_tolerance,
        max_abs_tolerance=max_abs_tolerance,
    )
    audit["task_group_split"] = task_validation
    audit["episode_group_split"] = episode_validation
    if not audit["passed"]:
        write_json_atomic(root / "audit" / "report.json", audit)
        raise ValueError(f"State Bank audit gate failed: {audit['gate_reasons']}")

    records_payload = _records_bytes(ordered)
    task_payload = (json.dumps(task_split.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    episode_payload = (
        json.dumps(episode_split.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode()
    audit_payload = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "record_schema_version": STATE_BANK_SCHEMA,
        "source_binding_sha256": source_binding_sha256,
        "ontology_sha256": ontology_sha256,
        "config_sha256": config_sha256,
        "alignment_sha256": alignment_sha256,
        "states": len(ordered),
        "episodes": len(
            {
                (record.suite, record.task_id, record.source_episode_id)
                for record in ordered
            }
        ),
        "tasks": len({(record.suite, record.task_id) for record in ordered}),
        "records_sha256": _sha256_bytes(records_payload),
        "task_group_split_sha256": _sha256_bytes(task_payload),
        "episode_group_split_sha256": _sha256_bytes(episode_payload),
        "audit_sha256": _sha256_bytes(audit_payload),
        "audit_passed": True,
    }
    write_bytes_atomic(root / "records.jsonl", records_payload)
    write_bytes_atomic(root / "splits" / "task_group.json", task_payload)
    write_bytes_atomic(root / "splits" / "episode_group.json", episode_payload)
    write_bytes_atomic(root / "audit" / "report.json", audit_payload)
    write_json_atomic(manifest_path, manifest)
    return audit


def load_state_bank(
    output_dir: str | Path,
) -> tuple[tuple[StateRecord, ...], dict[str, object], SplitManifest, SplitManifest]:
    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("State Bank manifest schema is incompatible")
    records = tuple(
        StateRecord.from_dict(json.loads(line))
        for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    task_split = SplitManifest.from_dict(
        json.loads((root / "splits" / "task_group.json").read_text(encoding="utf-8"))
    )
    episode_split = SplitManifest.from_dict(
        json.loads((root / "splits" / "episode_group.json").read_text(encoding="utf-8"))
    )
    records_payload = _records_bytes(records)
    if _sha256_bytes(records_payload) != manifest.get("records_sha256"):
        raise ValueError("State Bank records hash does not match manifest")
    if len(records) != int(manifest.get("states", -1)):
        raise ValueError("State Bank record count does not match manifest")
    task_path = root / "splits" / "task_group.json"
    episode_path = root / "splits" / "episode_group.json"
    audit_path = root / "audit" / "report.json"
    if _sha256_bytes(task_path.read_bytes()) != manifest.get("task_group_split_sha256"):
        raise ValueError("State Bank task split hash does not match manifest")
    if _sha256_bytes(episode_path.read_bytes()) != manifest.get("episode_group_split_sha256"):
        raise ValueError("State Bank episode split hash does not match manifest")
    if _sha256_bytes(audit_path.read_bytes()) != manifest.get("audit_sha256"):
        raise ValueError("State Bank audit hash does not match manifest")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest.get("audit_passed") is not True or audit.get("passed") is not True:
        raise ValueError("State Bank audit is not passing")
    task_count = len({(record.suite, record.task_id) for record in records})
    episode_count = len(
        {(record.suite, record.task_id, record.source_episode_id) for record in records}
    )
    if task_count != int(manifest.get("tasks", -1)):
        raise ValueError("State Bank task count does not match manifest")
    if episode_count != int(manifest.get("episodes", -1)):
        raise ValueError("State Bank episode count does not match manifest")
    if (
        int(audit.get("states", -1)) != len(records)
        or int(audit.get("tasks", -1)) != task_count
        or int(audit.get("episodes", -1)) != episode_count
    ):
        raise ValueError("State Bank audit counts do not match scientific contents")
    validate_split(records, task_split)
    validate_split(records, episode_split)
    return records, manifest, task_split, episode_split
