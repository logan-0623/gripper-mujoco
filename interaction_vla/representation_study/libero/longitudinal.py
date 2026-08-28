from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..state_bank.io import write_json_atomic
from .config import LiberoStudyConfig
from .latents import (
    _file_sha256,
    _tree_sha256,
    extract_smolvla_latents_from_checkpoint,
    load_latent_cache,
)
from .taps import SEMANTIC_TAPS


@dataclass(frozen=True)
class ConditionSpec:
    condition: str
    stage: str
    data_fraction: float
    step: int


CONDITION_SPECS = (
    ConditionSpec("pretrained", "pretrained", 0.0, 0),
    ConditionSpec("d25_u16070", "sft_25", 0.25, 16070),
    ConditionSpec("d50_u16324", "sft_50", 0.50, 16324),
    ConditionSpec("d100_u16617", "sft_100", 1.00, 16617),
    ConditionSpec("d50_u32650", "sft_50", 0.50, 32650),
    ConditionSpec("d100_u33234", "sft_100", 1.00, 33234),
    ConditionSpec("d100_u49851", "sft_100", 1.00, 49851),
    ConditionSpec("d100_u66470", "sft_100", 1.00, 66470),
)

CONTRASTS = {
    "matched_update_16k": ["d25_u16070", "d50_u16324", "d100_u16617"],
    "matched_update_32k": ["d50_u32650", "d100_u33234"],
    "d100_optimization_trajectory": [
        "pretrained",
        "d100_u16617",
        "d100_u33234",
        "d100_u49851",
        "d100_u66470",
    ],
}


def _checkpoint_path(output_dir: Path, spec: ConditionSpec, manifest: Mapping[str, object]) -> Path:
    if spec.stage == "pretrained":
        return Path(str(manifest["checkpoint"]))
    return (
        output_dir
        / "stages"
        / spec.stage
        / "run"
        / "checkpoints"
        / f"{spec.step:06d}"
        / "pretrained_model"
    )


def _recorded_step(checkpoint: Path) -> int:
    path = checkpoint.parent / "training_state" / "training_step.json"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint training step is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in ("step", "training_step"):
        if key in value:
            return int(value[key])
    raise ValueError(f"checkpoint training step is not recorded: {path}")


def _training_binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint training config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    dataset = value.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError(f"checkpoint training config has no dataset binding: {path}")
    binding = {
        "dataset_repo_id": dataset.get("repo_id"),
        "dataset_revision": dataset.get("revision"),
        "episode_indices": dataset.get("episodes"),
        "seed": value.get("seed"),
        "batch_size": value.get("batch_size"),
        "training_steps": value.get("steps"),
    }
    if (
        not isinstance(binding["dataset_repo_id"], str)
        or not isinstance(binding["dataset_revision"], str)
        or not isinstance(binding["episode_indices"], list)
        or not all(isinstance(item, int | str) for item in binding["episode_indices"])
        or not all(
            isinstance(binding[key], int) and int(binding[key]) > 0
            for key in ("seed", "batch_size", "training_steps")
        )
    ):
        raise ValueError(f"checkpoint training config binding is incomplete: {path}")
    return binding


def _binding_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def plan_longitudinal_conditions(config: LiberoStudyConfig) -> dict[str, object]:
    output_dir = Path(config.output_dir)
    manifests: dict[str, Mapping[str, object]] = {}
    train_bindings: dict[str, dict[str, object]] = {}
    for stage in {spec.stage for spec in CONDITION_SPECS}:
        path = output_dir / "stages" / stage / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"stage manifest is missing: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        final_checkpoint = Path(str(manifest["checkpoint"]))
        if manifest.get("status") != "complete" or not final_checkpoint.is_dir():
            raise ValueError(f"stage is not complete: {stage}")
        if manifest.get("checkpoint_sha256") != _tree_sha256(final_checkpoint):
            raise ValueError(f"stage checkpoint hash is stale: {stage}")
        if stage != "pretrained":
            train_config = final_checkpoint / "train_config.json"
            binding = _training_binding(train_config)
            expected = {
                "dataset_repo_id": manifest.get("dataset_repo_id"),
                "dataset_revision": manifest.get("dataset_revision"),
                "episode_indices": manifest.get("episode_indices"),
                "seed": manifest.get("seed"),
                "training_steps": manifest.get("training_steps"),
            }
            if any(binding[key] != value for key, value in expected.items()):
                raise ValueError(f"stage training config does not match manifest: {stage}")
            train_bindings[stage] = binding
        manifests[stage] = manifest

    conditions = []
    for spec in CONDITION_SPECS:
        checkpoint = _checkpoint_path(output_dir, spec, manifests[spec.stage])
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"planned checkpoint is missing: {checkpoint}")
        if spec.step and _recorded_step(checkpoint) != spec.step:
            raise ValueError(f"checkpoint training step does not match {spec.condition}")
        if spec.stage != "pretrained":
            binding = _training_binding(checkpoint / "train_config.json")
            if binding != train_bindings[spec.stage]:
                raise ValueError(
                    f"checkpoint training config does not match stage: {spec.condition}"
                )
        conditions.append(
            {
                **asdict(spec),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _tree_sha256(checkpoint),
                "training_binding_sha256": (
                    _binding_sha256(train_bindings[spec.stage])
                    if spec.stage != "pretrained"
                    else None
                ),
            }
        )

    report: dict[str, object] = {
        "schema_version": "libero_smolvla_longitudinal_plan_v3",
        "passed": True,
        "conditions": conditions,
        "contrasts": CONTRASTS,
        "note": "matched-update groups are approximate; recorded steps remain explicit",
    }
    path = output_dir / "protocol_v3" / "conditions" / "manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != report:
            raise FileExistsError(f"protocol-v3 condition binding changed: {path}")
    else:
        write_json_atomic(path, report)
    return report


def validate_runtime_bindings(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    keys = (
        "runtime_fingerprint_sha256",
        "state_bank_sha256",
        "implementation_sha256",
    )
    invalid = sum(
        not isinstance(row.get(key), str) or len(str(row.get(key))) != 64
        for row in reports
        for key in keys
    )
    runtime = {
        str(row["runtime_fingerprint_sha256"])
        for row in reports
        if isinstance(row.get("runtime_fingerprint_sha256"), str)
    }
    banks = {
        str(row["state_bank_sha256"])
        for row in reports
        if isinstance(row.get("state_bank_sha256"), str)
    }
    implementations = {
        str(row["implementation_sha256"])
        for row in reports
        if isinstance(row.get("implementation_sha256"), str)
    }
    return {
        "passed": bool(reports)
        and invalid == 0
        and len(runtime) == len(banks) == len(implementations) == 1,
        "reports": len(reports),
        "invalid_bindings": invalid,
        "runtime_fingerprints": len(runtime),
        "state_banks": len(banks),
        "implementations": len(implementations),
    }


def load_longitudinal_plan(config: LiberoStudyConfig) -> dict[str, object]:
    path = Path(config.output_dir) / "protocol_v3" / "conditions" / "manifest.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "libero_smolvla_longitudinal_plan_v3" or not report.get("passed"):
        raise ValueError("protocol-v3 condition plan is incomplete or incompatible")
    identities = [
        {
            key: row.get(key)
            for key in ("condition", "stage", "data_fraction", "step")
        }
        for row in report.get("conditions", [])
    ]
    if identities != [asdict(spec) for spec in CONDITION_SPECS]:
        raise ValueError("protocol-v3 condition grid is incomplete or incompatible")
    if report.get("contrasts") != CONTRASTS:
        raise ValueError("protocol-v3 contrast grid is incomplete or incompatible")
    return report


def condition_from_plan(config: LiberoStudyConfig, condition: str) -> dict[str, object]:
    plan = load_longitudinal_plan(config)
    matches = [row for row in plan["conditions"] if row["condition"] == condition]
    if len(matches) != 1:
        raise ValueError(f"unknown longitudinal condition: {condition}")
    row = dict(matches[0])
    checkpoint = Path(str(row["checkpoint"]))
    if _tree_sha256(checkpoint) != row["checkpoint_sha256"]:
        raise ValueError(f"longitudinal checkpoint hash is stale: {condition}")
    row["plan_sha256"] = _file_sha256(
        Path(config.output_dir) / "protocol_v3" / "conditions" / "manifest.json"
    )
    return row


def extract_longitudinal_condition(
    config: LiberoStudyConfig,
    *,
    condition: str,
    batch_size: int = 8,
) -> dict[str, object]:
    row = condition_from_plan(config, condition)
    return extract_smolvla_latents_from_checkpoint(
        config,
        checkpoint=str(row["checkpoint"]),
        checkpoint_id=f"{condition}:{str(row['checkpoint_sha256'])[:16]}",
        checkpoint_hash=str(row["checkpoint_sha256"]),
        output_dir=Path(config.output_dir) / "protocol_v3" / "latents" / condition,
        label=condition,
        batch_size=batch_size,
        runtime_binding=True,
        report_schema="libero_smolvla_longitudinal_latents_v3",
        report_fields={
            "condition": condition,
            "stage": row["stage"],
            "data_fraction": row["data_fraction"],
            "training_step": row["step"],
            "plan_sha256": row["plan_sha256"],
        },
    )


def inspect_longitudinal_latents(config: LiberoStudyConfig) -> dict[str, object]:
    plan = load_longitudinal_plan(config)
    plan_path = Path(config.output_dir) / "protocol_v3" / "conditions" / "manifest.json"
    plan_hash = _file_sha256(plan_path)
    reports = []
    missing = []
    state_id_reference: tuple[str, ...] | None = None
    tap_metadata_reference: Mapping[str, object] | None = None
    cache_hashes: dict[str, dict[str, str]] = {}
    for condition_row in plan["conditions"]:
        condition = str(condition_row["condition"])
        root = Path(config.output_dir) / "protocol_v3" / "latents" / condition
        report_path = root / "report.json"
        if not report_path.is_file():
            missing.append(condition)
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("schema_version") != "libero_smolvla_longitudinal_latents_v3"
            or not report.get("passed")
            or report.get("condition") != condition
            or report.get("checkpoint_sha256") != condition_row["checkpoint_sha256"]
            or report.get("plan_sha256") != plan_hash
        ):
            raise ValueError(f"protocol-v3 latent report has a stale binding: {condition}")
        cache_hashes[condition] = {}
        tap_metadata = report.get("tap_metadata")
        if not isinstance(tap_metadata, Mapping) or set(tap_metadata) != set(SEMANTIC_TAPS):
            raise ValueError(
                f"protocol-v3 semantic tap metadata is incomplete: {condition}"
            )
        if tap_metadata_reference is None:
            tap_metadata_reference = tap_metadata
        elif tap_metadata != tap_metadata_reference:
            raise ValueError("protocol-v3 semantic tap metadata differs across conditions")
        for tap in SEMANTIC_TAPS:
            state_ids, _, manifest = load_latent_cache(root / tap)
            if manifest != report.get("caches", {}).get(tap):
                raise ValueError(
                    f"protocol-v3 cache does not match extraction report: {condition}/{tap}"
                )
            for key in (
                "checkpoint_sha256",
                "state_bank_sha256",
                "implementation_sha256",
                "runtime_fingerprint_sha256",
            ):
                if manifest.get(key) != report.get(key):
                    raise ValueError(
                        f"protocol-v3 cache/report binding differs: {condition}/{tap}/{key}"
                    )
            if state_id_reference is None:
                state_id_reference = state_ids
            elif state_ids != state_id_reference:
                raise ValueError("protocol-v3 latent caches do not share exact State Bank rows")
            cache_hashes[condition][tap] = str(manifest["values_sha256"])
        reports.append(report)

    runtime_gate = validate_runtime_bindings(reports)
    identical_pairs = []
    conditions = list(cache_hashes)
    for left_index, left in enumerate(conditions):
        for right in conditions[left_index + 1 :]:
            identical_taps = [
                tap for tap in SEMANTIC_TAPS
                if cache_hashes[left][tap] == cache_hashes[right][tap]
            ]
            if identical_taps:
                identical_pairs.append(
                    {"left": left, "right": right, "taps": identical_taps}
                )
    report = {
        "schema_version": "libero_smolvla_longitudinal_latent_gate_v3",
        "passed": not missing and runtime_gate["passed"],
        "planned_conditions": len(plan["conditions"]),
        "completed_conditions": len(reports),
        "missing_conditions": missing,
        "runtime_gate": runtime_gate,
        "identical_cache_pairs": identical_pairs,
        "note": "identical tap caches are reported diagnostically and are not automatically invalid",
    }
    write_json_atomic(
        Path(config.output_dir) / "protocol_v3" / "latent_gate" / "report.json",
        report,
    )
    return report
