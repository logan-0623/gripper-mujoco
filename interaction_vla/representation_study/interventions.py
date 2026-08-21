from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from interaction_vla.lerobot_bridge.config import load_bridge_config
from interaction_vla.lerobot_bridge.provenance import sha256_file

from .backends import make_backend
from .config import RepresentationStudyConfig
from .extraction import build_stage_manifest
from .evaluation import evaluate_policy_stage, evaluation_cases
from .rl.environment import ResidualMujocoRuntime
from .statistics import (
    benjamini_hochberg,
    clustered_bootstrap_mean,
    paired_sign_flip_pvalue,
)
from .state_bank.io import load_records, load_split, write_json_atomic
from .state_bank.materialize import StateBankMaterializer, collate_observations
from .state_bank.schema import StateBankRecord
from .taps.registry import registered_taps


INTERVENTION_SCHEMA_VERSION = "interaction_latent_intervention_v1"
CLOSED_LOOP_INTERVENTION_SCHEMA_VERSION = "interaction_closed_loop_intervention_v1"


def intervention_probe_binding(
    config: RepresentationStudyConfig, *, backend: str, stage: str
) -> dict[str, str]:
    path = config.probes.output_dir / backend / stage / "linear" / "report.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"linear probe report required before interventions: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("passed") is not True
        or payload.get("schema_version") != "interaction_frozen_probe_v2"
        or payload.get("backend") != backend
        or payload.get("stage") != stage
    ):
        raise ValueError("linear probe report is incompatible with intervention stage")
    return {"uri": path.as_posix(), "sha256": sha256_file(path)}


def paired_intervention_batches(
    records: Sequence[StateBankRecord], *, batch_size: int, max_states: int
) -> tuple[tuple[StateBankRecord, ...], ...]:
    grouped: dict[tuple[str, str, str], list[StateBankRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.domain, record.stratum, record.phase)].append(record)
    result: list[tuple[StateBankRecord, ...]] = []
    retained = 0
    for key in sorted(grouped):
        values = sorted(grouped[key], key=lambda record: record.state_id)
        if len(values) < 2:
            continue
        chunks = [values[index : index + batch_size] for index in range(0, len(values), batch_size)]
        if len(chunks) > 1 and len(chunks[-1]) == 1:
            chunks[-2].extend(chunks.pop())
        for chunk in chunks:
            if retained + len(chunk) > max_states:
                remaining = max_states - retained
                if remaining >= 2:
                    result.append(tuple(chunk[:remaining]))
                return tuple(result)
            result.append(tuple(chunk))
            retained += len(chunk)
    if not result:
        raise ValueError("intervention selection contains no matched State Bank pairs")
    return tuple(result)


def action_change(baseline: np.ndarray, changed: np.ndarray) -> dict[str, float]:
    first = np.asarray(baseline, dtype=np.float64)
    second = np.asarray(changed, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 7:
        raise ValueError("intervention actions must be aligned chunks with width seven")
    difference = second - first
    return {
        "chunk_rms": float(np.sqrt(np.mean(difference**2))),
        "chunk_l2": float(np.linalg.norm(difference)),
        "first_action_l2": float(np.linalg.norm(difference[0])),
        "first_translation_l2": float(np.linalg.norm(difference[0, :3])),
        "first_rotation_l2": float(np.linalg.norm(difference[0, 3:6])),
        "first_gripper_absolute_change": float(abs(difference[0, 6])),
    }


def run_interventions(
    config: RepresentationStudyConfig, *, backend: str, stage: str
) -> dict[str, object]:
    records = load_records(config.state_bank.output_dir / "records.jsonl")
    split = load_split(config.state_bank.output_dir / "split.json")
    selected_ids = set(getattr(split, config.interventions.partition))
    batches = paired_intervention_batches(
        [record for record in records if record.state_id in selected_ids],
        batch_size=config.interventions.batch_size,
        max_states=config.interventions.max_states,
    )
    stage_manifest, hf_revision = build_stage_manifest(config, backend=backend, stage=stage)
    probe_binding = intervention_probe_binding(config, backend=backend, stage=stage)
    destination = config.interventions.output_dir / backend / stage / config.interventions.partition
    report_path = destination / "report.json"
    binding = {
        "schema_version": INTERVENTION_SCHEMA_VERSION,
        "stage_manifest_sha256": hashlib.sha256(stage_manifest.to_json().encode("utf-8")).hexdigest(),
        "state_bank_manifest_sha256": sha256_file(config.state_bank.output_dir / "manifest.json"),
        "modes": list(config.interventions.modes),
        "state_ids": [record.state_id for batch in batches for record in batch],
        "hf_revision": hf_revision,
        "linear_probe": probe_binding,
    }
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if all(report.get(key) == value for key, value in binding.items()):
            return report
        raise ValueError("existing intervention report is stale or incompatible")
    destination.mkdir(parents=True, exist_ok=True)
    runtime = make_backend(backend, device=config.extraction.device)
    runtime.load_checkpoint_for_dataset(
        config.stage_config(backend, stage).checkpoint,
        repo_id=config.dataset.repo_id,
        dataset_root=config.dataset.root,
    )
    materializer = StateBankMaterializer(
        dataset_root=config.dataset.root,
        repo_id=config.dataset.repo_id,
        bridge_config=load_bridge_config(config.dataset.bridge_config),
        replay_position_tolerance=config.state_bank.replay_position_tolerance,
    )
    taps = tuple(tap.tap_id for tap in registered_taps(backend))
    rows: list[dict[str, object]] = []
    progress = tqdm(
        total=len(batches) * (1 + len(taps) * len(config.interventions.modes)),
        desc=f"{backend}/{stage} interventions",
        unit="forward",
        dynamic_ncols=True,
    )
    for batch in batches:
        values = {value.record.state_id: value for value in materializer.iter_records(batch)}
        ordered = [values[record.state_id] for record in batch]
        inputs = collate_observations(ordered)
        seed = int.from_bytes(
            hashlib.sha256("\n".join(record.state_id for record in batch).encode("utf-8")).digest()[:8],
            "big",
        ) % (2**31)
        torch.manual_seed(seed)
        baseline = runtime.act(inputs).detach().to("cpu", torch.float32).numpy()
        progress.update(1)
        for tap in taps:
            for mode in config.interventions.modes:
                torch.manual_seed(seed)
                changed = runtime.intervene_actions(inputs, tap_id=tap, mode=mode).numpy()
                for index, record in enumerate(batch):
                    rows.append(
                        {
                            "state_id": record.state_id,
                            "domain": record.domain,
                            "stratum": record.stratum,
                            "phase": record.phase,
                            "tap": tap,
                            "mode": mode,
                            **action_change(baseline[index], changed[index]),
                        }
                    )
                progress.update(1)
    progress.close()
    aggregate: dict[str, dict[str, float]] = {}
    for tap in taps:
        for mode in config.interventions.modes:
            group = [row for row in rows if row["tap"] == tap and row["mode"] == mode]
            aggregate[f"{tap}/{mode}"] = {
                metric: float(np.mean([float(row[metric]) for row in group]))
                for metric in (
                    "chunk_rms",
                    "chunk_l2",
                    "first_action_l2",
                    "first_translation_l2",
                    "first_rotation_l2",
                    "first_gripper_absolute_change",
                )
            }
    report = {
        **binding,
        "passed": True,
        "backend": backend,
        "stage": stage,
        "partition": config.interventions.partition,
        "records": len({row["state_id"] for row in rows}),
        "rows": rows,
        "aggregate": aggregate,
        "interpretation": "offline functional-use evidence; not a closed-loop causal success estimate",
    }
    write_json_atomic(report_path, report)
    return report


def run_closed_loop_interventions(
    config: RepresentationStudyConfig, *, backend: str, stage: str
) -> dict[str, object]:
    """Measure paired success changes under a zero ablation of each latent tap."""
    baseline = evaluate_policy_stage(config, backend=backend, stage=stage)
    probe_binding = intervention_probe_binding(config, backend=backend, stage=stage)
    stage_manifest, revision = build_stage_manifest(config, backend=backend, stage=stage)
    binding = hashlib.sha256(stage_manifest.to_json().encode("utf-8")).hexdigest()
    baseline_path = (
        config.analysis.output_dir
        / "policy_evaluation"
        / backend
        / stage
        / "report.json"
    )
    baseline_sha256 = sha256_file(baseline_path)
    destination = config.interventions.output_dir / backend / stage / "closed_loop"
    report_path = destination / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("stage_manifest_sha256") == binding
            and report.get("linear_probe") == probe_binding
            and report.get("baseline_report_sha256") == baseline_sha256
        ):
            return report
        raise ValueError("existing closed-loop intervention report is stale")
    runtime = make_backend(backend, device=config.rl.device)
    runtime.load_checkpoint_for_dataset(
        config.stage_config(backend, stage).checkpoint,
        repo_id=config.dataset.repo_id,
        dataset_root=config.dataset.root,
    )
    bridge = load_bridge_config(config.dataset.bridge_config)
    taps = tuple(tap.tap_id for tap in registered_taps(backend))
    cases = evaluation_cases(config)
    baseline_by_case = {
        int(row["case_id"]): float(bool(row["success"]))
        for row in baseline["episode_rows"]
    }
    summaries: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    with ResidualMujocoRuntime(
        bridge=bridge,
        backend=runtime,
        tap_id=taps[-1],
        residual_scale=(0.0,) * 7,
        max_steps=config.rl.max_episode_steps,
        object_counts=config.rl.object_counts,
        layouts=config.rl.layouts,
        seed=config.rl.seed + 50_000,
        reward_mode="sparse",
        progress_reward_scale=0.0,
    ) as environment:
        progress = tqdm(
            total=len(taps) * len(cases),
            desc=f"{backend}/{stage} closed-loop interventions",
            unit="episode",
            dynamic_ncols=True,
        )
        for tap_index, tap in enumerate(taps):
            tap_rows: list[dict[str, object]] = []
            for case in cases:
                environment.reset(
                    case_seed=int(case["environment_seed"]),
                    object_count=int(case["object_count"]),
                    layout=str(case["layout"]),
                )
                torch.manual_seed(int(case["policy_seed"]))
                while True:
                    if environment.current_observation is None:
                        raise RuntimeError("closed-loop intervention lost its observation")
                    actions = runtime.intervene_actions(
                        environment.current_observation,
                        tap_id=tap,
                        mode="zero",
                    )
                    transition = environment.step(
                        base_action=actions[0, 0].numpy(),
                        latent=torch.zeros(1, 1),
                        residual=np.zeros(7, dtype=np.float32),
                    )
                    if transition.done:
                        row = {
                            **case,
                            "tap": tap,
                            "mode": "zero",
                            "success": transition.reason == "success",
                            "termination_reason": transition.reason,
                            "steps": transition.episode_length,
                            "action_clipping_rate": (
                                transition.episode_action_clipping_rate
                            ),
                            "mean_ik_projection_scale": (
                                transition.episode_mean_ik_projection_scale
                            ),
                        }
                        tap_rows.append(row)
                        episode_rows.append(row)
                        progress.update(1)
                        break
            differences = [
                float(row["success"]) - baseline_by_case[int(row["case_id"])]
                for row in tap_rows
            ]
            interval = clustered_bootstrap_mean(
                differences,
                [row["case_id"] for row in tap_rows],
                samples=config.analysis.bootstrap_samples,
                confidence=config.analysis.confidence_level,
                seed=config.analysis.seed + tap_index,
            )
            summaries.append(
                {
                    "tap": tap,
                    "mode": "zero",
                    "success_rate": float(np.mean([float(row["success"]) for row in tap_rows])),
                    "paired_success_delta": interval["estimate"],
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "paired_sign_flip_p": paired_sign_flip_pvalue(
                        differences,
                        samples=config.analysis.bootstrap_samples,
                        seed=config.analysis.seed + 100 + tap_index,
                    ),
                    "episodes": len(tap_rows),
                }
            )
    adjusted = benjamini_hochberg(
        [float(row["paired_sign_flip_p"]) for row in summaries]
    )
    for row, value in zip(summaries, adjusted, strict=True):
        row["bh_adjusted_p"] = value
    report = {
        "schema_version": CLOSED_LOOP_INTERVENTION_SCHEMA_VERSION,
        "passed": True,
        "backend": backend,
        "stage": stage,
        "stage_manifest_sha256": binding,
        "hf_revision": revision,
        "linear_probe": probe_binding,
        "baseline_report": baseline_path.as_posix(),
        "baseline_report_sha256": baseline_sha256,
        "mode": "zero",
        "primary_unit": "paired evaluation case",
        "causal_scope": "checkpoint-specific closed-loop intervention; no claim of learned-feature causality",
        "multiple_testing": "Benjamini-Hochberg across four fixed tap roles",
        "summaries": summaries,
        "episode_rows": episode_rows,
    }
    write_json_atomic(report_path, report)
    return report
