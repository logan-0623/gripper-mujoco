from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .config import LiberoStudyConfig
from ..state_bank.io import write_json_atomic


TRAINABLE_STAGES = ("sft_25", "sft_50", "sft_100")


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
        f"--output_dir={output_dir}",
        f"--job_name=libero-{stage}",
        f"--steps={int(manifest['training_steps'])}",
        f"--batch_size={config.stages.batch_size}",
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
