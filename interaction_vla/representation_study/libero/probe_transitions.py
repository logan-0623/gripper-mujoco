from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from ..state_bank.io import write_json_atomic
from .config import LiberoStudyConfig
from .probe_runner import (
    _matched_probe_seed,
    _paired_stage_delta,
    _probe_artifact_root,
    inspect_probe_report,
)
from .probes import FACTOR_NAMES, STUDY_STAGES, STUDY_TAPS


ADJACENT_STAGE_PAIRS = tuple(zip(STUDY_STAGES[:-1], STUDY_STAGES[1:], strict=True))
ADJACENT_STAGE_DELTA_SCHEMA = "libero_adjacent_stage_delta_analysis_v1"


def build_adjacent_stage_delta_grid(
    rows: Sequence[Mapping[str, object]],
    *,
    existing_reference_deltas: Sequence[Mapping[str, object]],
    split_name: str,
    config: LiberoStudyConfig,
) -> list[dict[str, object]]:
    """Return every adjacent Stage × Tap × Factor comparison in fixed order."""

    row_lookup = {
        (str(row.get("stage")), str(row.get("tap")), str(row.get("factor"))): row
        for row in rows
    }
    reference_lookup = {
        (
            str(row.get("reference_stage")),
            str(row.get("destination_stage")),
            str(row.get("tap")),
            str(row.get("factor")),
        ): row
        for row in existing_reference_deltas
    }
    result: list[dict[str, object]] = []
    for pair_index, (reference_stage, destination_stage) in enumerate(
        ADJACENT_STAGE_PAIRS
    ):
        for tap in STUDY_TAPS:
            for factor in FACTOR_NAMES:
                identity = {
                    "reference_stage": reference_stage,
                    "destination_stage": destination_stage,
                    "tap": tap,
                    "factor": factor,
                    "split": split_name,
                }
                if reference_stage == "pretrained":
                    existing = reference_lookup.get(
                        (reference_stage, destination_stage, tap, factor)
                    )
                    if existing is None:
                        raise ValueError(
                            "existing Pretrained-to-SFT-25 delta grid is incomplete"
                        )
                    result.append(dict(existing))
                    continue
                reference = row_lookup.get(
                    (reference_stage, tap, factor), {"status": "not_run"}
                )
                destination = row_lookup.get(
                    (destination_stage, tap, factor), {"status": "not_run"}
                )
                delta = _paired_stage_delta(
                    factor=factor,
                    reference=reference,
                    destination=destination,
                    samples=config.probes.bootstrap_samples,
                    confidence=config.probes.confidence_level,
                    minimum_valid_rate=config.probes.minimum_bootstrap_valid_rate,
                    seed=_matched_probe_seed(
                        base_seed=config.seed,
                        tap=tap,
                        factor=factor,
                        split_name=split_name,
                        replicate_offset=30_000 + pair_index,
                    ),
                )
                result.append({**identity, **delta})
    return result


def _load_full_probe_rows(
    artifact_root: Path,
    *,
    report: Mapping[str, object],
    split_name: str,
) -> list[dict[str, object]]:
    split_bindings = report.get("split_manifest_sha256")
    latent_bindings = report.get("latent_cache_manifest_sha256")
    latent_content_bindings = report.get("latent_content_sha256")
    if not isinstance(split_bindings, Mapping) or split_name not in split_bindings:
        raise ValueError("adjacent delta report split binding is missing")
    if not isinstance(latent_bindings, Mapping) or not isinstance(
        latent_content_bindings, Mapping
    ):
        raise ValueError("adjacent delta report latent bindings are missing")
    rows: list[dict[str, object]] = []
    for stage in STUDY_STAGES:
        for tap in STUDY_TAPS:
            latent_key = f"{stage}/{tap}"
            stage_tap_root = artifact_root / ".cells" / stage / tap / split_name
            if latent_key not in latent_bindings:
                if stage_tap_root.is_dir() and any(stage_tap_root.glob("*.json")):
                    raise ValueError(
                        f"orphaned probe cell has no validated latent binding: {latent_key}"
                    )
                rows.extend(
                    {
                        "stage": stage,
                        "tap": tap,
                        "factor": factor,
                        "split": split_name,
                        "status": "not_run",
                    }
                    for factor in FACTOR_NAMES
                )
                continue
            if latent_key not in latent_content_bindings:
                raise ValueError(
                    f"adjacent delta report latent-content binding is missing: {latent_key}"
                )
            for factor in FACTOR_NAMES:
                identity = {
                    "stage": stage,
                    "tap": tap,
                    "factor": factor,
                    "split": split_name,
                }
                path = artifact_root / ".cells" / stage / tap / split_name / f"{factor}.json"
                if not path.is_file():
                    raise ValueError(f"expected bound probe cell is missing: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                row = payload.get("row") if isinstance(payload, Mapping) else None
                binding = payload.get("binding") if isinstance(payload, Mapping) else None
                if not isinstance(row, dict):
                    raise ValueError(f"adjacent delta cell has an invalid row: {path}")
                expected_binding = {
                    **identity,
                    "probe_protocol": report.get("probe_protocol"),
                    "state_bank_manifest_sha256": report.get(
                        "state_bank_manifest_sha256"
                    ),
                    "split_manifest_sha256": split_bindings[split_name],
                    "latent_manifest_sha256": latent_bindings[latent_key],
                    "latent_content_sha256": latent_content_bindings[latent_key],
                    "config_sha256": report.get("config_sha256"),
                    "implementation_sha256": report.get("implementation_sha256"),
                }
                if binding != expected_binding:
                    raise ValueError(f"adjacent delta cell binding is stale: {path}")
                if any(row.get(key) != value for key, value in identity.items()):
                    raise ValueError(f"adjacent delta cell identity is incompatible: {path}")
                rows.append(dict(row))
    return rows


def _analysis_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def enrich_adjacent_stage_deltas(config: LiberoStudyConfig) -> dict[str, object]:
    """Atomically add consecutive-stage paired intervals to a valid Probe v2 report."""

    report = inspect_probe_report(config)
    primary_reference = report.get("stage_deltas")
    secondary_reference = report.get("secondary_stage_deltas")
    if not isinstance(primary_reference, list) or not isinstance(
        secondary_reference, list
    ):
        raise ValueError("probe report reference-stage delta grids are missing")
    artifact_root = _probe_artifact_root(config.output_dir)
    primary_rows = _load_full_probe_rows(
        artifact_root, report=report, split_name="task_group"
    )
    secondary_rows = _load_full_probe_rows(
        artifact_root, report=report, split_name="episode_group"
    )
    enriched = dict(report)
    enriched.update(
        {
            "adjacent_stage_delta_schema_version": ADJACENT_STAGE_DELTA_SCHEMA,
            "adjacent_stage_delta_analysis_sha256": _analysis_sha256(),
            "adjacent_stage_pairs": [
                {
                    "reference_stage": reference_stage,
                    "destination_stage": destination_stage,
                }
                for reference_stage, destination_stage in ADJACENT_STAGE_PAIRS
            ],
            "adjacent_stage_deltas": build_adjacent_stage_delta_grid(
                primary_rows,
                existing_reference_deltas=primary_reference,
                split_name="task_group",
                config=config,
            ),
            "secondary_adjacent_stage_deltas": build_adjacent_stage_delta_grid(
                secondary_rows,
                existing_reference_deltas=secondary_reference,
                split_name="episode_group",
                config=config,
            ),
            "adjacent_stage_delta_interpretation": (
                "destination_minus_reference; improvement flips sign for "
                "lower-is-better metrics; an interval containing zero is inconclusive"
            ),
        }
    )
    write_json_atomic(artifact_root / "report.json", enriched)
    return enriched
