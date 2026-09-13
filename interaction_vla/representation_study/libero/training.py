from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .config import LiberoStudyConfig
from .feature_binding import LIBERO_SMOLVLA_RENAME_MAP
from .stages import EpisodeInfo
from ..state_bank.io import write_json_atomic


TRAINABLE_STAGES = ("sft_25", "sft_50", "sft_100")


def audit_training_contract(
    config: LiberoStudyConfig,
    episodes: Sequence[EpisodeInfo],
    *,
    training_tasks: Sequence[tuple[str, int]],
    evaluation_tasks: Sequence[tuple[str, int]],
    stages: Sequence[str],
    protocol_path: Path | None = None,
) -> dict[str, object]:
    """Audit an explicitly scoped lineage without treating coverage as readiness."""
    if not training_tasks or not evaluation_tasks or not stages:
        raise ValueError("training tasks, evaluation tasks, and stages must be explicit")
    if any(stage not in TRAINABLE_STAGES for stage in stages):
        raise ValueError(f"stages must be drawn from {TRAINABLE_STAGES}")
    episode_by_id = {row.episode_index: row for row in episodes}
    expected_tasks = sorted(set(training_tasks))
    rows = []
    required_state_files = {
        "optimizer_param_groups.json", "optimizer_state.safetensors",
        "rng_state.safetensors", "scheduler_state.json", "training_step.json",
    }
    for stage in stages:
        path = config.output_dir / "stages" / stage / "manifest.json"
        if not path.is_file():
            rows.append({"stage": stage, "manifest_status": "missing"})
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        selected = [int(value) for value in manifest.get("episode_indices", [])]
        unknown = sorted(set(selected).difference(episode_by_id))
        counts: dict[str, dict[str, int]] = {}
        for episode_id in selected:
            row = episode_by_id.get(episode_id)
            if row is None:
                continue
            key = f"{row.suite}/{row.task_id}"
            cell = counts.setdefault(key, {"episodes": 0, "frames": 0})
            cell["episodes"] += 1
            cell["frames"] += row.frames
        checkpoint_root = config.output_dir / "stages" / stage / "run" / "checkpoints"
        checkpoints = sorted(
            item.name for item in checkpoint_root.glob("[0-9]*") if item.is_dir()
        ) if checkpoint_root.is_dir() else []
        final = Path(str(manifest.get("checkpoint", "")))
        train_config_path = final / "train_config.json"
        train_config = (
            json.loads(train_config_path.read_text(encoding="utf-8"))
            if train_config_path.is_file() else {}
        )
        state_dir = final.parent / "training_state"
        step_path = state_dir / "training_step.json"
        step_state = json.loads(step_path.read_text()) if step_path.is_file() else {}
        planned_batch = manifest.get("batch_size")
        batch_size = step_state.get("batch_size", train_config.get("batch_size"))
        accumulation = train_config.get("gradient_accumulation_steps", 1)
        world_size = step_state.get("num_processes", train_config.get("world_size"))
        covered = set(
            (row.suite, row.task_id) for row in episode_by_id.values()
            if row.episode_index in set(selected)
        )
        missing_tasks = [f"{suite}/{task}" for suite, task in expected_tasks
                         if (suite, task) not in covered]
        model_complete = False
        actual_hash = None
        if manifest.get("status") == "complete" and final.is_dir():
            try:
                actual_hash = _tree_sha256(final)
                model_complete = actual_hash == manifest.get("checkpoint_sha256")
            except (FileNotFoundError, ValueError):
                pass
        state_files = {item.name for item in state_dir.iterdir()} if state_dir.is_dir() else set()
        resume_ready = model_complete and required_state_files.issubset(state_files)
        rows.append({
            "stage": stage,
            "manifest_status": manifest.get("status"),
            "episode_count": len(selected),
            "task_coverage": counts,
            "coverage_source": "stage manifest episode contract",
            "sampler_visit_counts_present": False,
            "training_tasks_missing": missing_tasks,
            "unknown_episode_ids": unknown,
            "planned_optimizer_steps": manifest.get("training_steps"),
            "observed_checkpoint_steps": checkpoints,
            "initialization": {
                "base_model": manifest.get("base_model"),
                "base_revision": manifest.get("base_revision"),
            },
            "batch": {
                "per_device": batch_size,
                "planned_per_device": planned_batch,
                "gradient_accumulation_steps": accumulation,
                "world_size": world_size,
                "effective": (
                    int(batch_size) * int(accumulation) * int(world_size)
                    if world_size is not None else None
                ),
            },
            "train_config_present": bool(train_config),
            "batch_contract_matches": planned_batch is not None and batch_size == planned_batch,
            "checkpoint_hash_verified": model_complete,
            "checkpoint_tree_sha256": actual_hash,
            "training_state_files_missing": sorted(required_state_files.difference(state_files)),
            "artifact_complete": model_complete and bool(train_config),
            "resume_ready": resume_ready,
            "execution_contract_complete": (
                isinstance(planned_batch, int) and planned_batch > 0
                and isinstance(manifest.get("training_steps"), int)
                and int(manifest["training_steps"]) > 0
                and bool(manifest.get("base_model"))
                and len(str(manifest.get("base_revision", ""))) == 40
            ),
        })
    coverage_passed = all(row.get("manifest_status") != "missing" for row in rows) and all(
        not row.get("training_tasks_missing") and not row.get("unknown_episode_ids")
        for row in rows
    )
    artifact_complete = all(bool(row.get("artifact_complete")) for row in rows)
    resume_ready = all(bool(row.get("resume_ready")) for row in rows)
    execution_contract_complete = all(
        bool(row.get("execution_contract_complete")) for row in rows
    )
    protocol = json.loads(protocol_path.read_text()) if protocol_path is not None else {}
    required_protocol = {
        "official_checkpoint", "evaluation_contract", "noninferiority_margin",
        "task_floors", "validation_rule", "compute_ceiling", "stop_rule",
    }
    protocol_missing = sorted(required_protocol.difference(protocol))
    report = {
        "schema_version": "libero_smolvla_training_contract_audit_v1",
        "training_tasks": [f"{suite}/{task}" for suite, task in expected_tasks],
        "evaluation_tasks": [f"{suite}/{task}" for suite, task in sorted(set(evaluation_tasks))],
        "evaluation_roles": {
            f"{suite}/{task}": "in_training_scope" if (suite, task) in expected_tasks else "generalization"
            for suite, task in sorted(set(evaluation_tasks))
        },
        "audited_stages": list(stages),
        "configured_tasks_per_suite": config.coverage.tasks_per_suite,
        "training_coverage_complete": coverage_passed,
        "artifact_complete": artifact_complete,
        "resume_ready": resume_ready,
        "execution_contract_complete": execution_contract_complete,
        "protocol_ready": execution_contract_complete and not protocol_missing,
        "stages": rows,
        "protocol_path": str(protocol_path) if protocol_path is not None else None,
        "protocol_fields_missing": protocol_missing,
        "interpretation": (
            "Coverage and artifact audit only; it does not establish optimization "
            "sufficiency, convergence, or a flow-matching failure mechanism."
        ),
    }
    write_json_atomic(config.output_dir / "stages" / "training_contract_audit.json", report)
    return report


def _tree_sha256(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(path).parts
    )
    if not files:
        raise ValueError(f"checkpoint directory is empty: {path}")
    if not (path / "config.json").is_file():
        raise ValueError(f"policy checkpoint has no config.json: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        file_digest = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(file_digest.hexdigest().encode("ascii"))
    return digest.hexdigest()


def snapshot_pretrained_stage(config: LiberoStudyConfig) -> dict[str, object]:
    manifest_path = config.output_dir / "stages" / "pretrained" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = str(manifest.get("base_revision", ""))
    if len(revision) != 40:
        raise ValueError("pretrained stage manifest is not bound to an immutable model commit")
    checkpoint = Path(str(manifest["checkpoint"]))
    expected_hash = manifest.get("checkpoint_sha256")
    if manifest.get("status") == "complete":
        checkpoint_hash = _tree_sha256(checkpoint)
        if checkpoint_hash != expected_hash:
            raise ValueError("pretrained checkpoint hash is stale")
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=str(manifest["base_model"]),
            revision=revision,
            local_dir=checkpoint,
        )
        checkpoint_hash = _tree_sha256(checkpoint)
    manifest["status"] = "complete"
    manifest["checkpoint_sha256"] = checkpoint_hash
    write_json_atomic(manifest_path, manifest)
    report = {
        "schema_version": "libero_smolvla_pretrained_snapshot_v1",
        "passed": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "base_revision": revision,
    }
    write_json_atomic(
        config.output_dir / "stages" / "pretrained" / "snapshot_report.json",
        report,
    )
    return report


def build_stage_training_command(
    config: LiberoStudyConfig,
    *,
    stage: str,
    resume: bool = False,
) -> tuple[str, ...]:
    if stage not in TRAINABLE_STAGES:
        raise ValueError(f"stage must be one of {TRAINABLE_STAGES}")
    manifest_path = config.output_dir / "stages" / stage / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run libero stages plan first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != stage or not manifest.get("episode_indices"):
        raise ValueError(f"stage manifest is incompatible: {manifest_path}")
    output_dir = config.output_dir / "stages" / stage / "run"
    if resume:
        resume_config = (
            output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        )
        if not resume_config.is_file():
            raise FileNotFoundError(
                f"cannot resume without an upstream LeRobot training checkpoint: {resume_config}"
            )
        return (
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--config_path={resume_config}",
            "--resume=true",
        )
    if not isinstance(manifest.get("batch_size"), int) or int(manifest["batch_size"]) <= 0:
        raise ValueError(f"stage manifest has no valid planned batch size: {manifest_path}")
    pretrained_manifest_path = config.output_dir / "stages" / "pretrained" / "manifest.json"
    pretrained = json.loads(pretrained_manifest_path.read_text(encoding="utf-8"))
    pretrained_checkpoint = Path(str(pretrained["checkpoint"]))
    if pretrained.get("status") != "complete" or not pretrained_checkpoint.is_dir():
        raise ValueError("run libero stages snapshot before SFT training")
    checkpoint_hash = _tree_sha256(pretrained_checkpoint)
    if pretrained.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("pretrained checkpoint hash is stale")
    episodes = "[" + ",".join(str(value) for value in manifest["episode_indices"]) + "]"
    return (
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--policy.path={pretrained_checkpoint}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={manifest['dataset_repo_id']}",
        f"--dataset.revision={manifest['dataset_revision']}",
        f"--dataset.episodes={episodes}",
        "--rename_map="
        + json.dumps(LIBERO_SMOLVLA_RENAME_MAP, sort_keys=True, separators=(",", ":")),
        f"--output_dir={output_dir}",
        f"--job_name=libero-{stage}",
        f"--steps={int(manifest['training_steps'])}",
        f"--batch_size={int(manifest['batch_size'])}",
        f"--num_workers={config.stages.num_workers}",
        f"--seed={config.stages.seed}",
        "--cudnn_deterministic=true",
        "--env_eval_freq=0",
        "--save_checkpoint=true",
        f"--save_freq={max(1, int(manifest['training_steps']) // 4)}",
        "--wandb.enable=false",
    )


def train_stage(
    config: LiberoStudyConfig,
    *,
    stage: str,
    dry_run: bool,
    resume: bool,
) -> dict[str, object]:
    command = build_stage_training_command(config, stage=stage, resume=resume)
    run_dir = config.output_dir / "stages" / stage / "run"
    if run_dir.exists() and any(run_dir.iterdir()) and not dry_run and not resume:
        raise FileExistsError(
            f"training output already exists; use the upstream LeRobot resume command or a new output: {run_dir}"
        )
    report: dict[str, object] = {
        "schema_version": "libero_smolvla_stage_training_v1",
        "stage": stage,
        "status": "implementation_only" if dry_run else "running",
        "executed": not dry_run,
        "resume": resume,
        "command": list(command),
    }
    if dry_run:
        return report
    manifest_path = config.output_dir / "stages" / stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    write_json_atomic(manifest_path, manifest)
    report_path = config.output_dir / "stages" / stage / "training_report.json"
    write_json_atomic(report_path, report)
    try:
        subprocess.run(command, check=True)
    except Exception as error:
        manifest["status"] = "failed"
        report["status"] = "failed"
        report["error"] = type(error).__name__
        report["message"] = str(error)
        write_json_atomic(manifest_path, manifest)
        write_json_atomic(report_path, report)
        raise
    checkpoint = run_dir / "checkpoints" / "last" / "pretrained_model"
    if not checkpoint.is_dir():
        manifest["status"] = "failed"
        report["status"] = "failed"
        report["error"] = "FileNotFoundError"
        report["message"] = f"expected checkpoint is missing: {checkpoint}"
        write_json_atomic(manifest_path, manifest)
        write_json_atomic(report_path, report)
        raise FileNotFoundError(f"LeRobot training did not produce the expected checkpoint: {checkpoint}")
    checkpoint_hash = _tree_sha256(checkpoint)
    manifest["status"] = "complete"
    manifest["checkpoint_sha256"] = checkpoint_hash
    write_json_atomic(manifest_path, manifest)
    report["status"] = "complete"
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_sha256"] = checkpoint_hash
    write_json_atomic(report_path, report)
    return report
