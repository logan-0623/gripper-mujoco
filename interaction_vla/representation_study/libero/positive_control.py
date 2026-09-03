from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from ..state_bank.io import write_json_atomic
from .config import LiberoStudyConfig
from .crossfit_probes import (
    CrossFitManifest,
    _read_gzip_json,
    _write_immutable_gzip_json,
    _write_immutable_json,
    build_crossfit_manifest,
    crossfit_partition_indices,
    run_crossfit_cell,
)
from .feature_binding import LIBERO_SMOLVLA_RENAME_MAP
from .latents import (
    _file_sha256,
    _tree_sha256,
    collate_state_bank_observations,
    extract_smolvla_latents_from_checkpoint,
    load_latent_cache,
)
from .recruitment import (
    _action_effect,
    _atomic_npz,
    _cluster_interval,
    _context_batch_plan,
    _episode_key,
    _factor_data,
    _predict,
    _reconstruct_fold_probes,
    _target_delta,
    consensus_direction,
    phase_stratum,
    same_norm_random_delta,
)
from .schema import StateRecord
from .state_bank import load_state_bank


POSITIVE_CONTROL_SCHEMA = "libero_smolvla_official_positive_control_v1"
PRIMARY_TAP = "action_expert_input"
SUPPORTED_FACTORS = ("stable_grasp", "contact")
PROBE_FACTORS = ("stable_grasp", "contact", "phase", "geometry")
MINIMUM_SUCCESS_RATE = 0.20
EVALUATION_CONTRACT = {
    "source": "lerobot.scripts.lerobot_eval",
    "n_action_steps": 10,
    "num_denoising_steps": 10,
    "empty_cameras": 1,
    "camera_name_mapping": {
        "agentview_image": "camera1",
        "robot0_eye_in_hand_image": "camera2",
    },
    "max_parallel_tasks": 1,
    "recording": False,
}


def positive_control_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / "protocol_v4" / "positive_control"


def _factor_intervention_root(root: Path, factor: str, max_states: int) -> Path:
    if factor not in SUPPORTED_FACTORS or max_states <= 0:
        raise ValueError("positive-control factor profile is invalid")
    return root / "intervention" / factor / f"n_{max_states:04d}"


def _require_contact_route(root: Path, *, factor: str, max_states: int) -> None:
    if factor != "contact":
        return
    stable_report = _factor_intervention_root(
        root, "stable_grasp", max_states
    ) / "report.json"
    if not stable_report.is_file() or json.loads(
        stable_report.read_text(encoding="utf-8")
    ).get("decision") != "replicate_contact_once":
        raise ValueError(
            "Contact requires the immutable StableGrasp decision "
            "replicate_contact_once"
        )


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("latents.py"),
        Path(__file__).with_name("crossfit_probes.py"),
        Path(__file__).with_name("probes.py"),
        Path(__file__).with_name("recruitment.py"),
        Path(__file__).parents[1] / "backends" / "lerobot.py",
    ):
        digest.update(path.name.encode())
        digest.update(_file_sha256(path).encode())
    return digest.hexdigest()


def _runtime_identity() -> dict[str, object]:
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "lerobot": importlib.metadata.version("lerobot"),
    }


def _seal_positive_control_evaluation(
    *,
    checkpoint: str | Path,
    eval_dir: str | Path,
    command: Sequence[str] | None = None,
) -> dict[str, object]:
    checkpoint = Path(checkpoint)
    eval_dir = Path(eval_dir)
    config_path = checkpoint / "config.json"
    eval_path = eval_dir / "eval_info.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"official checkpoint is incomplete: {checkpoint}")
    success_rate, episodes = official_success_rate(eval_path)
    report = {
        "schema_version": "libero_smolvla_positive_control_eval_contract_v1",
        "passed": True,
        "checkpoint_sha256": _tree_sha256(checkpoint),
        "evaluation_report_sha256": _file_sha256(eval_path),
        "success_rate": success_rate,
        "episodes": episodes,
        "runtime_contract": EVALUATION_CONTRACT,
        "provenance": "generated_by_evaluate_positive_control",
        "command": list(command or ()),
    }
    path = eval_dir / "positive_control_evaluation.json"
    _write_immutable_json(path, report)
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluation_command(checkpoint: Path, eval_dir: Path) -> tuple[str, ...]:
    camera_mapping = json.dumps(
        EVALUATION_CONTRACT["camera_name_mapping"], separators=(",", ":")
    )
    return (
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={checkpoint}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.n_action_steps=10",
        "--policy.num_steps=10",
        "--policy.empty_cameras=1",
        "--env.type=libero",
        "--env.task=libero_spatial",
        "--env.task_ids=[0]",
        "--env.obs_type=pixels_agent_pos",
        "--env.init_states=true",
        "--env.fps=30",
        "--env.max_parallel_tasks=1",
        f"--env.camera_name_mapping={camera_mapping}",
        "--eval.n_episodes=10",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        "--eval.recording=false",
        "--seed=2057736129",
        f"--output_dir={eval_dir}",
    )


def evaluate_positive_control(
    *, checkpoint: str | Path, eval_dir: str | Path
) -> dict[str, object]:
    checkpoint = Path(checkpoint)
    eval_dir = Path(eval_dir)
    contract_path = eval_dir / "positive_control_evaluation.json"
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract.get("checkpoint_sha256") != _tree_sha256(checkpoint):
            raise ValueError("existing evaluation belongs to a different checkpoint")
        if contract.get("evaluation_report_sha256") != _file_sha256(
            eval_dir / "eval_info.json"
        ):
            raise ValueError("existing evaluation report changed")
        return contract
    if eval_dir.exists():
        raise FileExistsError(
            f"positive-control evaluation requires a new output directory: {eval_dir}"
        )
    eval_dir.parent.mkdir(parents=True, exist_ok=True)
    command = _evaluation_command(checkpoint, eval_dir)
    subprocess.run(command, check=True)
    report = _seal_positive_control_evaluation(
        checkpoint=checkpoint, eval_dir=eval_dir, command=command
    )
    return report


def factor_specificity_gate(
    *,
    factor: str,
    target_minus_random: Mapping[str, float],
    target_effect: float,
    non_target_effects: Mapping[str, float],
    activation_norm_ratio: float,
    place_target_minus_random: Mapping[str, float] | None,
) -> dict[str, object]:
    if factor not in SUPPORTED_FACTORS:
        raise ValueError(f"unsupported positive-control factor: {factor}")
    failures: list[str] = []
    if float(target_minus_random.get("ci_low", -np.inf)) <= 0:
        failures.append(f"{factor} disruption does not exceed matched random")
    for control in (set(SUPPORTED_FACTORS) - {factor}) | {"phase"}:
        if target_effect <= float(non_target_effects.get(control, np.inf)):
            failures.append(f"{factor} disruption does not exceed {control}")
    if factor == "stable_grasp" and (
        place_target_minus_random is None
        or float(place_target_minus_random.get("ci_low", -np.inf)) <= 0
    ):
        failures.append("place-only StableGrasp specificity is not supported")
    if not 0.8 <= activation_norm_ratio <= 1.2:
        failures.append("intervened activation norm is outside the support band")
    return {"passed": not failures, "failures": sorted(failures)}


def official_success_rate(path: Path) -> tuple[float, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    overall = payload.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError("LeRobot evaluation has no overall metrics")
    percent = float(overall.get("pc_success", float("nan")))
    episodes = int(overall.get("n_episodes", 0))
    if not np.isfinite(percent) or not 0.0 <= percent <= 100.0 or episodes <= 0:
        raise ValueError("LeRobot evaluation metrics must be finite and non-empty")
    return percent / 100.0, episodes


def positive_control_decision(
    *,
    success_rate: float,
    accessible: bool,
    specificity_passed: bool,
    usage_ci: tuple[float, float],
    factor: str,
    formal_profile: bool = True,
) -> dict[str, object]:
    if factor not in SUPPORTED_FACTORS:
        raise ValueError(f"unsupported positive-control factor: {factor}")
    if not np.isfinite((success_rate, *usage_ci)).all():
        raise ValueError("positive-control decision values must be finite")
    if not formal_profile:
        decision = "smoke_only"
    elif success_rate <= MINIMUM_SUCCESS_RATE:
        decision = "failed_policy_floor"
    elif not accessible:
        decision = (
            "replicate_contact_once"
            if factor == "stable_grasp"
            else "pivot_interaction_supervised_sft"
        )
    elif not specificity_passed:
        decision = "failed_specificity"
    elif usage_ci[0] > 0.0:
        decision = "continue_official_longitudinal"
    elif factor == "stable_grasp":
        decision = "replicate_contact_once"
    else:
        decision = "pivot_interaction_supervised_sft"
    return {
        "decision": decision,
        "authorize_longitudinal_training": (
            decision == "continue_official_longitudinal"
        ),
        "hypothesis_passed": decision == "continue_official_longitudinal",
    }


def plan_positive_control(
    config: LiberoStudyConfig,
    *,
    checkpoint: str | Path,
    eval_dir: str | Path,
) -> dict[str, object]:
    checkpoint = Path(checkpoint)
    eval_dir = Path(eval_dir)
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"official checkpoint is incomplete: {checkpoint}")
    eval_report = eval_dir / "eval_info.json"
    if not eval_report.is_file():
        raise FileNotFoundError(f"LeRobot evaluation report is missing: {eval_report}")
    contract_path = eval_dir / "positive_control_evaluation.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"sealed positive-control evaluation is missing: {contract_path}"
        )
    state_bank = Path(config.output_dir) / "state_bank" / "manifest.json"
    if not state_bank.is_file():
        raise FileNotFoundError(f"State Bank manifest is missing: {state_bank}")
    success_rate, episodes = official_success_rate(eval_report)
    if success_rate <= MINIMUM_SUCCESS_RATE:
        raise ValueError(
            "official checkpoint remains at the preregistered closed-loop floor"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checkpoint_sha256 = _tree_sha256(checkpoint)
    if (
        contract.get("schema_version")
        != "libero_smolvla_positive_control_eval_contract_v1"
        or contract.get("passed") is not True
    ):
        raise ValueError("sealed evaluation contract is incomplete")
    if contract.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("sealed evaluation belongs to a different checkpoint")
    if contract.get("evaluation_report_sha256") != _file_sha256(eval_report):
        raise ValueError("sealed evaluation report hash changed")
    if contract.get("runtime_contract") != EVALUATION_CONTRACT:
        raise ValueError("sealed evaluation runtime contract changed")
    command = contract.get("command")
    if (
        contract.get("provenance") != "generated_by_evaluate_positive_control"
        or command != list(_evaluation_command(checkpoint, eval_dir))
    ):
        raise ValueError("sealed evaluation provenance is incomplete")
    report: dict[str, object] = {
        "schema_version": POSITIVE_CONTROL_SCHEMA,
        "passed": True,
        "status": "ready_for_latents",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_dir": str(eval_dir),
        "evaluation_report_sha256": _file_sha256(eval_report),
        "evaluation_contract_sha256": _file_sha256(contract_path),
        "baseline_success_rate": success_rate,
        "baseline_episodes": episodes,
        "state_bank_sha256": _file_sha256(state_bank),
        "config_sha256": _file_sha256(config.source_path),
        "runtime_contract": EVALUATION_CONTRACT,
        "implementation_sha256": _implementation_sha256(),
        "interpretation": "successful-policy floor screen, not a benchmark claim",
    }
    path = positive_control_root(config.output_dir) / "plan.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != report:
            raise FileExistsError(f"positive-control binding changed: {path}")
        return existing
    write_json_atomic(path, report)
    return report


def load_positive_control_plan(config: LiberoStudyConfig) -> dict[str, object]:
    path = positive_control_root(config.output_dir) / "plan.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != POSITIVE_CONTROL_SCHEMA or not report.get(
        "passed"
    ):
        raise ValueError("positive-control plan is incomplete or incompatible")
    checkpoint = Path(str(report["checkpoint"]))
    if _tree_sha256(checkpoint) != report.get("checkpoint_sha256"):
        raise ValueError("positive-control checkpoint hash changed")
    eval_report = Path(str(report["evaluation_dir"])) / "eval_info.json"
    if _file_sha256(eval_report) != report.get("evaluation_report_sha256"):
        raise ValueError("positive-control evaluation report hash changed")
    contract_path = Path(str(report["evaluation_dir"])) / "positive_control_evaluation.json"
    if _file_sha256(contract_path) != report.get("evaluation_contract_sha256"):
        raise ValueError("positive-control evaluation contract changed")
    state_bank = Path(config.output_dir) / "state_bank" / "manifest.json"
    if _file_sha256(state_bank) != report.get("state_bank_sha256"):
        raise ValueError("positive-control State Bank hash changed")
    if _file_sha256(config.source_path) != report.get("config_sha256"):
        raise ValueError("positive-control config hash changed")
    if _implementation_sha256() != report.get("implementation_sha256"):
        raise ValueError("positive-control implementation changed")
    return report


def extract_positive_control(
    config: LiberoStudyConfig, *, batch_size: int = 32
) -> dict[str, object]:
    plan = load_positive_control_plan(config)
    root = positive_control_root(config.output_dir)
    return extract_smolvla_latents_from_checkpoint(
        config,
        checkpoint=str(plan["checkpoint"]),
        checkpoint_id=f"official:{str(plan['checkpoint_sha256'])[:16]}",
        checkpoint_hash=str(plan["checkpoint_sha256"]),
        output_dir=root / "latents",
        label="official_smolvla_libero",
        batch_size=batch_size,
        runtime_binding=True,
        report_schema="libero_smolvla_positive_control_latents_v1",
        report_fields={"plan_sha256": _file_sha256(root / "plan.json")},
        taps=(PRIMARY_TAP,),
    )


def summarize_positive_control_probe_results(
    cells: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if set(cells) != set(PROBE_FACTORS):
        raise ValueError("positive-control probe factors are incomplete")
    status_counts = {
        status: sum(row.get("status") == status for row in cells.values())
        for status in ("complete", "not_estimable", "failed_gate")
    }
    stable = cells["stable_grasp"]
    passed = status_counts["complete"] == len(PROBE_FACTORS)
    return {
        "passed": passed,
        "status_counts": status_counts,
        "stable_grasp_accessible": bool(
            passed and stable.get("accessible") is True
        ),
    }


def _positive_control_probe_binding(
    *,
    plan: Mapping[str, object],
    plan_sha256: str,
    latent_manifest: Mapping[str, object],
    folds_sha256: str,
    factor: str,
) -> dict[str, object]:
    return {
        "schema_version": "libero_smolvla_positive_control_crossfit_cell_v1",
        "condition": "official_smolvla_libero",
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "tap": PRIMARY_TAP,
        "factor": factor,
        "split": "episode_group",
        "plan_sha256": plan_sha256,
        "folds_sha256": folds_sha256,
        "latent_values_sha256": latent_manifest["values_sha256"],
        "state_bank_sha256": plan["state_bank_sha256"],
        "implementation_sha256": plan["implementation_sha256"],
        "runtime": _runtime_identity(),
    }


def run_positive_control_probe(config: LiberoStudyConfig) -> dict[str, object]:
    plan = load_positive_control_plan(config)
    root = positive_control_root(config.output_dir)
    plan_sha256 = _file_sha256(root / "plan.json")
    latent_report_path = root / "latents" / "report.json"
    latent_report = json.loads(latent_report_path.read_text(encoding="utf-8"))
    if (
        latent_report.get("schema_version")
        != "libero_smolvla_positive_control_latents_v1"
        or not latent_report.get("passed")
        or latent_report.get("checkpoint_sha256") != plan["checkpoint_sha256"]
        or latent_report.get("plan_sha256") != plan_sha256
    ):
        raise ValueError("positive-control latent report is stale or incompatible")
    records, _, _, _ = load_state_bank(Path(config.output_dir) / "state_bank")
    state_ids, features, latent_manifest = load_latent_cache(
        root / "latents" / PRIMARY_TAP
    )
    if state_ids != tuple(record.state_id for record in records):
        raise ValueError("positive-control latent rows differ from the State Bank")
    if latent_manifest.get("state_bank_sha256") != plan["state_bank_sha256"]:
        raise ValueError("positive-control latent State Bank binding is stale")

    manifest = build_crossfit_manifest(
        records,
        split_name="episode_group",
        folds=config.probes.crossfit_folds,
        seed=config.seed,
    )
    probe_root = root / "probe"
    folds_path = probe_root / "folds.json"
    _write_immutable_json(
        folds_path,
        {
            "schema_version": "libero_smolvla_positive_control_folds_v1",
            "passed": True,
            "manifest": asdict(manifest),
            "plan_sha256": plan_sha256,
        },
    )
    folds_sha256 = _file_sha256(folds_path)
    cells: dict[str, Mapping[str, object]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for factor in PROBE_FACTORS:
        path = probe_root / "cells" / f"{factor}.json.gz"
        binding = _positive_control_probe_binding(
            plan=plan,
            plan_sha256=plan_sha256,
            latent_manifest=latent_manifest,
            folds_sha256=folds_sha256,
            factor=factor,
        )
        if path.is_file():
            cell = _read_gzip_json(path)
            if cell.get("binding") != binding:
                raise FileExistsError(f"positive-control probe binding changed: {path}")
        else:
            cell = {
                "binding": binding,
                "result": run_crossfit_cell(
                    config=config,
                    records=records,
                    features=features,
                    manifest=manifest,
                    tap=PRIMARY_TAP,
                    factor=factor,
                ),
            }
            _write_immutable_gzip_json(path, cell)
            cell = _read_gzip_json(path)
        result = cell.get("result")
        if not isinstance(result, Mapping):
            raise ValueError(f"positive-control probe cell is malformed: {path}")
        cells[factor] = result
        artifacts[factor] = {"path": str(path), "sha256": _file_sha256(path)}

    summary = summarize_positive_control_probe_results(cells)
    report = {
        "schema_version": "libero_smolvla_positive_control_probe_v1",
        **summary,
        "condition": "official_smolvla_libero",
        "tap": PRIMARY_TAP,
        "split": "episode_group",
        "plan_sha256": plan_sha256,
        "folds_sha256": folds_sha256,
        "artifacts": artifacts,
        "cells": {
            factor: {
                key: cells[factor].get(key)
                for key in (
                    "status",
                    "reason",
                    "accessible",
                    "primary_metric_name",
                    "primary_metric",
                    "accessibility_threshold",
                    "accessibility_utility",
                    "accessibility_utility_ci",
                )
            }
            for factor in PROBE_FACTORS
        },
        "interpretation_boundary": {
            "accessible": "cross-fitted utility exceeds the strongest registered shortcut",
            "functionally_used": "not measured by this report",
        },
    }
    report_path = probe_root / "report.json"
    _write_immutable_json(report_path, report)
    return json.loads(report_path.read_text(encoding="utf-8"))


def _probe_manifest(root: Path) -> CrossFitManifest:
    payload = json.loads((root / "probe" / "folds.json").read_text(encoding="utf-8"))
    row = payload["manifest"]
    return CrossFitManifest(
        split_name=str(row["split_name"]),
        group_unit=str(row["group_unit"]),
        folds=int(row["folds"]),
        seed=int(row["seed"]),
        group_folds={str(key): int(value) for key, value in row["group_folds"].items()},
    )


def _select_factor_records(
    records: Sequence[StateRecord], *, factor: str, max_states: int
) -> tuple[int, ...]:
    if factor not in SUPPORTED_FACTORS or max_states <= 0:
        raise ValueError("positive-control factor and max_states are invalid")
    buckets: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if not (
            record.labels.applicability.stable_grasp
            and getattr(record.labels.applicability, factor)
        ):
            continue
        key = f"{record.suite}:{record.task_id}:{phase_stratum(record.labels.phase)}"
        buckets.setdefault(key, []).append(index)
    for key, rows in buckets.items():
        rows.sort(
            key=lambda index: hashlib.sha256(
                f"{key}:{records[index].state_id}".encode()
            ).digest()
        )
    selected: list[int] = []
    for cursor in range(max(map(len, buckets.values()), default=0)):
        for key in sorted(buckets):
            if cursor < len(buckets[key]):
                selected.append(buckets[key][cursor])
                if len(selected) == max_states:
                    return tuple(selected)
    return tuple(selected)


def _positive_control_specificity(
    config: LiberoStudyConfig, *, factor: str, max_states: int
) -> dict[str, object]:
    if factor not in SUPPORTED_FACTORS:
        raise ValueError(f"unsupported positive-control factor: {factor}")
    root = positive_control_root(config.output_dir)
    _require_contact_route(root, factor=factor, max_states=max_states)
    plan = load_positive_control_plan(config)
    probe_path = root / "probe" / "report.json"
    probe_report = json.loads(probe_path.read_text(encoding="utf-8"))
    if not probe_report.get("passed"):
        raise ValueError("positive-control probes did not pass")
    if probe_report["cells"][factor].get("accessible") is not True:
        return {
            "schema_version": "libero_smolvla_positive_control_specificity_v1",
            "passed": False,
            "status": "blocked_by_accessibility",
            "factor": factor,
        }
    factor_root = _factor_intervention_root(root, factor, max_states)
    report_path = factor_root / "specificity.json"
    binding = {
        "plan_sha256": _file_sha256(root / "plan.json"),
        "probe_sha256": _file_sha256(probe_path),
        "factor": factor,
        "max_states": max_states,
        "implementation_sha256": plan["implementation_sha256"],
        "runtime": _runtime_identity(),
    }
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("binding") != binding:
            raise FileExistsError(f"specificity binding changed: {report_path}")
        artifact = Path(str(report["intervention_artifact"]))
        if _file_sha256(artifact) != report.get("intervention_sha256"):
            raise ValueError("positive-control intervention artifact changed")
        return report

    records, _, _, _ = load_state_bank(Path(config.output_dir) / "state_bank")
    state_ids, features, _ = load_latent_cache(root / "latents" / PRIMARY_TAP)
    if state_ids != tuple(record.state_id for record in records):
        raise ValueError("positive-control latent rows differ from the State Bank")
    selected = _select_factor_records(records, factor=factor, max_states=max_states)
    if not selected:
        raise ValueError(f"no applicable states for {factor}")
    manifest = _probe_manifest(root)
    selected_folds = np.asarray(
        [manifest.group_folds[_episode_key(records[index])] for index in selected]
    )
    cells = {
        name: _read_gzip_json(root / "probe" / "cells" / f"{name}.json.gz")
        for name in PROBE_FACTORS
    }
    probes = {name: {} for name in PROBE_FACTORS}
    for name in PROBE_FACTORS:
        offsets = (
            config.probes.matched_seed_offsets
            if name == factor
            else (config.probes.matched_seed_offsets[0],)
        )
        for fold in range(manifest.folds):
            probes[name][fold] = _reconstruct_fold_probes(
                config=config,
                records=records,
                features=features,
                manifest=manifest,
                condition="official_smolvla_libero",
                factor=name,
                cell=cells[name],
                fold=fold,
                offsets=offsets,
            )

    source = features[np.asarray(selected)]
    target_delta = np.zeros_like(source, dtype=np.float64)
    random_delta = np.zeros_like(source, dtype=np.float64)
    mean_delta = np.zeros_like(source, dtype=np.float64)
    agreements: list[float] = []
    support_caps: list[float] = []
    capped_rates: list[float] = []
    applicable, _, _ = _factor_data(records, factor)
    for fold in range(manifest.folds):
        rows = np.flatnonzero(selected_folds == fold)
        if not len(rows):
            continue
        target_probes = list(probes[factor][fold].values())
        directions = np.stack([probe.direction for probe in target_probes])
        direction = consensus_direction(directions)
        agreements.append(
            float(
                np.min(
                    np.abs(directions @ direction)
                    / np.linalg.norm(directions, axis=1)
                )
            )
        )
        parts = crossfit_partition_indices(
            records, manifest, fold=fold, applicable=applicable
        )
        train = features[np.asarray(parts["train"])]
        cap = float(np.quantile(np.linalg.norm(train - np.roll(train, 1, axis=0), axis=1), 0.95))
        local_target, capped = _target_delta(
            source[rows], target_probes, direction, cap
        )
        target_delta[rows] = local_target
        random_delta[rows] = same_norm_random_delta(
            local_target, target_direction=direction, seed=config.seed + fold
        )
        mean_delta[rows] = (
            ((target_probes[0].train_mean - source[rows]) @ direction)[:, None]
            * direction[None, :]
        )
        support_caps.append(cap)
        capped_rates.append(capped)

    changed = source + target_delta
    random_changed = source + random_delta
    effects: dict[str, np.ndarray] = {}
    random_effects: dict[str, np.ndarray] = {}
    for name in PROBE_FACTORS:
        effects[name] = np.zeros(len(selected))
        random_effects[name] = np.zeros(len(selected))
        for fold in range(manifest.folds):
            rows = np.flatnonzero(selected_folds == fold)
            if not len(rows):
                continue
            fold_probes = list(probes[name][fold].values())
            effects[name][rows] = np.mean(
                [probe.disruption(source[rows], changed[rows]) for probe in fold_probes],
                axis=0,
            )
            random_effects[name][rows] = np.mean(
                [
                    probe.disruption(source[rows], random_changed[rows])
                    for probe in fold_probes
                ],
                axis=0,
            )
    contrasts = {name: effects[name] - random_effects[name] for name in PROBE_FACTORS}
    target_interval = _cluster_interval(
        contrasts[factor], records, selected, config=config
    )
    place_rows = np.asarray(
        [row for row, index in enumerate(selected) if records[index].labels.phase == "place"]
    )
    place_interval = (
        _cluster_interval(
            contrasts[factor][place_rows],
            records,
            [selected[row] for row in place_rows],
            config=config,
        )
        if factor == "stable_grasp" and len(place_rows)
        else None
    )
    activation_ratio = float(
        np.mean(np.linalg.norm(changed, axis=1))
        / max(np.mean(np.linalg.norm(source, axis=1)), 1e-12)
    )
    gate = factor_specificity_gate(
        factor=factor,
        target_minus_random=target_interval,
        target_effect=float(np.mean(contrasts[factor])),
        non_target_effects={
            name: float(np.mean(contrasts[name]))
            for name in PROBE_FACTORS
            if name != factor
        },
        activation_norm_ratio=activation_ratio,
        place_target_minus_random=place_interval,
    )
    artifact = factor_root / "intervention.npz"
    _atomic_npz(
        artifact,
        state_ids=np.asarray([records[index].state_id for index in selected]),
        source=source.astype(np.float32),
        target_delta=target_delta.astype(np.float32),
        random_delta=random_delta.astype(np.float32),
        mean_delta=mean_delta.astype(np.float32),
    )
    report = {
        "schema_version": "libero_smolvla_positive_control_specificity_v1",
        **gate,
        "status": "complete",
        "factor": factor,
        "tap": PRIMARY_TAP,
        "states": len(selected),
        "binding": binding,
        "probe_reconstruction": "categorical_exact_continuous_atol_1e-6",
        "seed_direction_min_cosine": min(agreements),
        "natural_difference_p95_by_fold": support_caps,
        "decision_boundary_cap_rate_by_fold": capped_rates,
        "target_effect": float(np.mean(effects[factor])),
        "matched_random_effect": float(np.mean(random_effects[factor])),
        "target_minus_random_episode_ci": target_interval,
        "target_minus_random_task_ci": _cluster_interval(
            contrasts[factor], records, selected, config=config, task=True
        ),
        "place_target_minus_random_episode_ci": place_interval,
        "non_target_target_minus_random_effects": {
            name: float(np.mean(contrasts[name]))
            for name in PROBE_FACTORS
            if name != factor
        },
        "activation_norm_ratio": activation_ratio,
        "same_norm_max_abs_error": float(
            np.max(
                np.abs(
                    np.linalg.norm(target_delta, axis=1)
                    - np.linalg.norm(random_delta, axis=1)
                )
            )
        ),
        "intervention_artifact": str(artifact),
        "intervention_sha256": _file_sha256(artifact),
        "checkpoint_sha256": plan["checkpoint_sha256"],
    }
    write_json_atomic(report_path, report)
    return report


def _positive_control_actions(
    config: LiberoStudyConfig,
    *,
    factor: str,
    max_states: int,
    batch_size: int,
) -> dict[str, object]:
    root = positive_control_root(config.output_dir)
    plan = load_positive_control_plan(config)
    factor_root = _factor_intervention_root(root, factor, max_states)
    specificity_path = factor_root / "specificity.json"
    specificity = json.loads(specificity_path.read_text(encoding="utf-8"))
    if not specificity.get("passed"):
        return {
            "schema_version": "libero_smolvla_positive_control_actions_v1",
            "passed": False,
            "status": "blocked_by_specificity",
            "factor": factor,
        }
    report_path = factor_root / "action_sensitivity.json"
    latent_report = json.loads(
        (root / "latents" / "report.json").read_text(encoding="utf-8")
    )
    binding = {
        "plan_sha256": _file_sha256(root / "plan.json"),
        "specificity_sha256": _file_sha256(specificity_path),
        "factor": factor,
        "batch_size": batch_size,
        "latent_runtime_fingerprint_sha256": latent_report.get(
            "runtime_fingerprint_sha256"
        ),
        "implementation_sha256": plan["implementation_sha256"],
        "runtime": _runtime_identity(),
    }
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("binding") != binding:
            raise FileExistsError(f"action-sensitivity binding changed: {report_path}")
        if _file_sha256(Path(str(report["actions"]))) != report.get("actions_sha256"):
            raise ValueError("positive-control action artifact changed")
        return report
    latent_batch_size = int(latent_report["runtime"]["batch_size"])
    if batch_size != latent_batch_size:
        raise ValueError(
            f"action batch size must match latent extraction ({latent_batch_size})"
        )
    records, _, _, _ = load_state_bank(Path(config.output_dir) / "state_bank")
    by_id = {record.state_id: record for record in records}
    record_index = {record.state_id: index for index, record in enumerate(records)}
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("LeRobotDataset is required for action sensitivity") from error
    dataset = LeRobotDataset(
        config.sources.lerobot_repo_id,
        root=config.sources.lerobot_root,
        revision=config.sources.lerobot_revision,
        download_videos=True,
    )
    from ..backends.lerobot import SmolVLABackend

    backend = SmolVLABackend(device="auto")
    backend.load_checkpoint_for_dataset(
        str(plan["checkpoint"]),
        repo_id=config.sources.lerobot_repo_id,
        dataset_root=dataset.root,
        rename_map=LIBERO_SMOLVLA_RENAME_MAP,
    )
    policy, preprocessor, postprocessor = backend._loaded()
    with np.load(specificity["intervention_artifact"]) as archive:
        intervention = {name: archive[name].copy() for name in archive.files}
    selected_ids = tuple(str(value) for value in intervention["state_ids"])
    selected = tuple(by_id[state_id] for state_id in selected_ids)
    selected_indices = np.asarray([record_index[state_id] for state_id in selected_ids])
    all_ids, latent_features, _ = load_latent_cache(root / "latents" / PRIMARY_TAP)
    expected_ids = tuple(record.state_id for record in records)
    if all_ids != expected_ids:
        raise ValueError("positive-control action latent rows differ from State Bank")
    if not np.array_equal(intervention["source"], latent_features[selected_indices]):
        raise ValueError("specificity source differs from positive-control latent cache")
    context_plan = _context_batch_plan(
        all_ids, selected_ids, batch_size=batch_size
    )
    modes: dict[str, np.ndarray | None] = {
        name: None for name in ("original", "target", "random", "mean", "zero")
    }
    checkpoint_id = f"official:{str(plan['checkpoint_sha256'])[:16]}"
    for start, stop, context_rows, output_rows in tqdm(
        context_plan,
        desc=f"{factor} actions official",
        unit="context",
    ):
        batch = collate_state_bank_observations(records[start:stop], dataset)
        context_ids = all_ids[start:stop]
        expected = latent_features[start:stop]
        local = np.asarray(context_rows)
        output = np.asarray(output_rows)
        for mode, key, zero in (
            ("original", None, False),
            ("target", "target_delta", False),
            ("random", "random_delta", False),
            ("mean", "mean_delta", False),
            ("zero", None, True),
        ):
            delta = np.zeros_like(expected)
            if key is not None:
                delta[local] = intervention[key][output]
            predicted = _predict(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                backend=backend,
                batch=batch,
                state_ids=context_ids,
                checkpoint_id=checkpoint_id,
                delta=delta,
                expected=expected,
                zero=zero,
            )
            if modes[mode] is None:
                modes[mode] = np.empty(
                    (len(selected), *predicted.shape[1:]), dtype=predicted.dtype
                )
            modes[mode][output] = predicted[local]
    if any(value is None for value in modes.values()):
        raise ValueError("positive-control action batching produced no outputs")
    actions = {name: value for name, value in modes.items() if value is not None}
    effects = {
        mode: _action_effect(actions["original"], actions[mode])
        for mode in ("target", "random", "mean", "zero")
    }
    indices = [record_index[record.state_id] for record in selected]
    metrics: dict[str, object] = {}
    for metric in effects["target"]:
        usage = effects["target"][metric] - effects["random"][metric]
        strata = {}
        for stratum in (
            "pre_contact",
            "contact_grasp",
            "post_grasp",
            "place_release",
        ):
            rows = np.asarray(
                [
                    row
                    for row, record in enumerate(selected)
                    if phase_stratum(record.labels.phase) == stratum
                ]
            )
            if len(rows):
                strata[stratum] = _cluster_interval(
                    usage[rows],
                    records,
                    [indices[row] for row in rows],
                    config=config,
                )
        metrics[metric] = {
            "target_effect": float(np.mean(effects["target"][metric])),
            "matched_random_effect": float(np.mean(effects["random"][metric])),
            "target_minus_random_episode_ci": _cluster_interval(
                usage, records, indices, config=config
            ),
            "target_minus_random_task_ci": _cluster_interval(
                usage, records, indices, config=config, task=True
            ),
            "matched_mean_effect": float(np.mean(effects["mean"][metric])),
            "zero_ood_effect": float(np.mean(effects["zero"][metric])),
            "strata": strata,
        }
    actions_path = factor_root / "actions.npz"
    _atomic_npz(
        actions_path,
        state_ids=np.asarray(selected_ids),
        **{name: value for name, value in actions.items()},
    )
    primary = metrics["first_action_l2"]["target_minus_random_episode_ci"]
    report = {
        "schema_version": "libero_smolvla_positive_control_actions_v1",
        "passed": True,
        "status": "complete",
        "factor": factor,
        "tap": PRIMARY_TAP,
        "states": len(selected),
        "context_batches": len(context_plan),
        "binding": binding,
        "primary_metric": "first_action_l2_target_minus_matched_random",
        "primary_cluster": "episode",
        "functionally_recruited": float(primary["ci_low"]) > 0,
        "metrics": metrics,
        "actions": str(actions_path),
        "actions_sha256": _file_sha256(actions_path),
        "closed_loop_useful": "not_measured",
    }
    write_json_atomic(report_path, report)
    return report


def report_positive_control(
    config: LiberoStudyConfig, *, factor: str, max_states: int = 1600
) -> dict[str, object]:
    if factor not in SUPPORTED_FACTORS:
        raise ValueError(f"unsupported positive-control factor: {factor}")
    root = positive_control_root(config.output_dir)
    _require_contact_route(root, factor=factor, max_states=max_states)
    plan = load_positive_control_plan(config)
    probe_path = root / "probe" / "report.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    accessible = probe.get("cells", {}).get(factor, {}).get("accessible") is True
    factor_root = _factor_intervention_root(root, factor, max_states)
    specificity_path = factor_root / "specificity.json"
    actions_path = factor_root / "action_sensitivity.json"
    if accessible and not specificity_path.is_file():
        raise FileNotFoundError(
            f"positive-control specificity evidence is missing: {specificity_path}"
        )
    specificity = (
        json.loads(specificity_path.read_text(encoding="utf-8"))
        if specificity_path.is_file()
        else {"passed": False, "status": "not_required_inaccessible"}
    )
    if specificity.get("passed") and not actions_path.is_file():
        raise FileNotFoundError(
            f"positive-control action evidence is missing: {actions_path}"
        )
    actions = (
        json.loads(actions_path.read_text(encoding="utf-8"))
        if actions_path.is_file()
        else {"passed": False, "status": "not_required"}
    )
    interval = (
        actions.get("metrics", {})
        .get("first_action_l2", {})
        .get("target_minus_random_episode_ci", {})
    )
    usage_ci = (
        float(interval.get("ci_low", 0.0)),
        float(interval.get("ci_high", 0.0)),
    )
    decision = positive_control_decision(
        success_rate=float(plan["baseline_success_rate"]),
        accessible=accessible,
        specificity_passed=bool(specificity.get("passed")),
        usage_ci=usage_ci,
        factor=factor,
        formal_profile=max_states == 1600,
    )
    report = {
        "schema_version": "libero_smolvla_positive_control_report_v1",
        "passed": bool(decision["hypothesis_passed"]),
        "status": "complete",
        "factor": factor,
        "baseline_success_rate": plan["baseline_success_rate"],
        "accessible": accessible,
        "specificity_passed": bool(specificity.get("passed")),
        "functionally_recruited": bool(actions.get("functionally_recruited")),
        "usage_episode_ci": {"ci_low": usage_ci[0], "ci_high": usage_ci[1]},
        **decision,
        "bindings": {
            "plan_sha256": _file_sha256(root / "plan.json"),
            "probe_sha256": _file_sha256(probe_path),
            "specificity_sha256": (
                _file_sha256(specificity_path) if specificity_path.is_file() else None
            ),
            "actions_sha256": (
                _file_sha256(actions_path) if actions_path.is_file() else None
            ),
        },
        "interpretation_boundary": {
            "closed_loop_useful": "not measured",
            "authorize_new_training": "only continue_official_longitudinal authorizes it",
        },
    }
    output = factor_root / "report.json"
    _write_immutable_json(output, report)
    return json.loads(output.read_text(encoding="utf-8"))


def run_positive_control_intervention(
    config: LiberoStudyConfig,
    *,
    factor: str = "stable_grasp",
    max_states: int = 1600,
    batch_size: int = 32,
) -> dict[str, object]:
    _require_contact_route(
        positive_control_root(config.output_dir),
        factor=factor,
        max_states=max_states,
    )
    specificity = _positive_control_specificity(
        config, factor=factor, max_states=max_states
    )
    if specificity.get("passed"):
        _positive_control_actions(
            config, factor=factor, max_states=max_states, batch_size=batch_size
        )
    return report_positive_control(config, factor=factor, max_states=max_states)
