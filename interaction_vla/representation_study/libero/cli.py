from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Sequence

from ..state_bank.io import write_json_atomic
from .collector import collect_libero_state_bank
from .config import load_libero_study_config
from .sources import load_source_catalog
from .stages import EpisodeInfo, build_stage_manifests, plan_nested_subsets
from .state_bank import load_state_bank
from .visualize import approve_annotation_timelines, render_annotation_timelines


INTERVENTION_CONDITIONS = (
    "pretrained",
    "d25_u16070",
    "d100_u16617",
    "d100_u66470",
)


def add_libero_parser(families: argparse._SubParsersAction) -> None:
    parser = families.add_parser(
        "libero", help="shared LIBERO State Bank and longitudinal SmolVLA study"
    )
    commands = parser.add_subparsers(dest="libero_family", required=True)
    audit = commands.add_parser("audit", help="inspect prerequisites and immutable evidence state")
    audit.add_argument("--config", type=Path, required=True)

    bank = commands.add_parser("state-bank", help="deterministic privileged LIBERO State Bank")
    bank_commands = bank.add_subparsers(dest="libero_command", required=True)
    for name in ("collect", "inspect", "visualize", "approve-timelines"):
        command = bank_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)

    stages = commands.add_parser("stages", help="nested SmolVLA SFT stage manifests")
    stage_commands = stages.add_subparsers(dest="libero_command", required=True)
    stage_plan = stage_commands.add_parser("plan")
    stage_plan.add_argument("--config", type=Path, required=True)
    stage_plan.add_argument("--batch-size", type=int)
    stage_train = stage_commands.add_parser("train")
    stage_train.add_argument("--config", type=Path, required=True)
    stage_train.add_argument(
        "--stage", choices=("sft_25", "sft_50", "sft_100"), required=True
    )
    stage_train.add_argument("--dry-run", action="store_true")
    stage_train.add_argument("--resume", action="store_true")
    stage_snapshot = stage_commands.add_parser("snapshot")
    stage_snapshot.add_argument("--config", type=Path, required=True)

    latents = commands.add_parser("latents", help="semantic SmolVLA latent cache")
    latent_commands = latents.add_subparsers(dest="libero_command", required=True)
    for name in ("extract", "inspect"):
        command = latent_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument(
            "--stage",
            choices=("pretrained", "sft_25", "sft_50", "sft_100"),
            required=True,
        )

    longitudinal = commands.add_parser(
        "longitudinal", help="same-runtime supervision/update protocol v3"
    )
    longitudinal_commands = longitudinal.add_subparsers(
        dest="libero_command", required=True
    )
    for name in ("plan", "inspect", "probes", "probe-report"):
        command = longitudinal_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    longitudinal_extract = longitudinal_commands.add_parser("extract")
    longitudinal_extract.add_argument("--config", type=Path, required=True)
    longitudinal_extract.add_argument("--condition", required=True)
    longitudinal_extract.add_argument("--batch-size", type=int, default=8)

    probes = commands.add_parser("probes", help="Stage × Tap × Factor probes")
    probe_commands = probes.add_subparsers(dest="libero_command", required=True)
    for name in ("run", "report"):
        command = probe_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)

    interventions = commands.add_parser(
        "interventions", help="factor-aligned latent interventions"
    )
    intervention_commands = interventions.add_subparsers(
        dest="libero_command", required=True
    )
    intervention_audit = intervention_commands.add_parser("audit")
    intervention_audit.add_argument("--config", type=Path, required=True)
    intervention_run = intervention_commands.add_parser("run")
    intervention_run.add_argument("--config", type=Path, required=True)
    intervention_run.add_argument("--max-states", type=int, default=1600)
    intervention_run.add_argument("--batch-size", type=int, default=32)
    intervention_run.add_argument("--specificity-only", action="store_true")
    intervention_run.add_argument("--dry-run", action="store_true")

    evaluate = commands.add_parser("evaluate", help="paired LIBERO closed-loop utility")
    evaluate_commands = evaluate.add_subparsers(dest="libero_command", required=True)
    evaluate_paired = evaluate_commands.add_parser("paired")
    evaluate_paired.add_argument("--config", type=Path, required=True)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_intervention_batch_size(protocol_root: Path) -> int:
    values = {}
    for condition in INTERVENTION_CONDITIONS:
        report = json.loads((protocol_root / "latents" / condition / "report.json").read_text())
        runtime = report.get("runtime")
        batch_size = runtime.get("batch_size") if isinstance(runtime, dict) else None
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"latent runtime batch size is invalid: {condition}")
        values[condition] = batch_size
    unique = set(values.values())
    if len(unique) != 1:
        raise ValueError(f"latent runtime batch sizes differ across checkpoints: {values}")
    return unique.pop()


def _hash_python_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.suffix in {".py", ".yaml"})
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(_hash_file(item).encode("ascii"))
    return digest.hexdigest()


def _audit(config_path: Path) -> dict[str, object]:
    config = load_libero_study_config(config_path)
    bank_path = config.output_dir / "state_bank" / "manifest.json"
    bank_validated = False
    if bank_path.is_file():
        try:
            load_state_bank(config.output_dir / "state_bank")
            bank_validated = True
        except (OSError, KeyError, TypeError, ValueError):
            bank_validated = False
    checks: dict[str, object] = {
        "platform_linux": platform.system() == "Linux",
        "raw_hdf5_root_exists": config.sources.raw_hdf5_root.is_dir(),
        "state_bank_exists": bank_path.is_file(),
        "state_bank_validated": bank_validated,
    }
    try:
        import h5py  # noqa: F401
        import libero  # noqa: F401

        checks["optional_dependencies"] = True
    except ImportError:
        checks["optional_dependencies"] = False
    ready_for_collection = bool(
        checks["platform_linux"]
        and checks["raw_hdf5_root_exists"]
        and checks["optional_dependencies"]
    )
    return {
        "schema_version": "libero_representation_prerequisite_audit_v1",
        "passed": ready_for_collection or bank_validated,
        "ready_for_collection": ready_for_collection,
        "checks": checks,
        "formal_suites": list(config.coverage.suites),
        "rl_in_scope": False,
        "interaction_factors": [
            "entity", "geometry", "contact", "stable_grasp", "phase", "next_relation"
        ],
        "experiment_status": {
            "act_graph_v2": "formal_evidence",
            "act_graph_diagnostics": "pilot_complete",
            "reflect_graph_pretraining": "pilot_complete",
            "act_stagewise": "pilot_complete",
            "recovery_rl_v2_calibration": "failed_gate",
            "formal_recovery_rl": "implementation_only",
            "libero_state_bank": "implementation_only" if not bank_validated else "formal_evidence",
            "smolvla_longitudinal": "not_started",
        },
        "output_dir": str(config.output_dir),
    }


def _inspect_bank(config_path: Path) -> dict[str, object]:
    config = load_libero_study_config(config_path)
    records, manifest, task_split, episode_split = load_state_bank(
        config.output_dir / "state_bank"
    )
    audit = json.loads(
        (config.output_dir / "state_bank" / "audit" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "schema_version": "libero_state_bank_inspection_v1",
        "passed": bool(audit.get("passed")),
        "states": len(records),
        "manifest": manifest,
        "task_group": task_split.to_dict(),
        "episode_group": episode_split.to_dict(),
        "audit": audit,
    }


def _visualize(config_path: Path) -> dict[str, object]:
    config = load_libero_study_config(config_path)
    records, manifest, _, _ = load_state_bank(config.output_dir / "state_bank")
    source_revisions = {record.source_revision for record in records}
    if len(source_revisions) != 1:
        raise ValueError("State Bank records do not share one dataset revision")
    try:
        import numpy as np
        import torch
        from PIL import Image
        from lerobot.datasets import LeRobotDataset

        dataset = LeRobotDataset(
            config.sources.lerobot_repo_id,
            root=config.sources.lerobot_root,
            revision=next(iter(source_revisions)),
            download_videos=True,
        )

        def image_loader(record, view):
            key = (
                record.observation.global_rgb_key
                if view == "global"
                else record.observation.wrist_rgb_key
            )
            if key is None:
                return Image.new("RGB", (128, 128), "#dddddd")
            tensor = torch.as_tensor(dataset[record.observation.dataset_index][key]).detach().cpu()
            array = tensor.numpy()
            if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
                array = np.moveaxis(array, 0, -1)
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array, 0.0, 1.0) * 255.0
            return Image.fromarray(array.astype(np.uint8)).convert("RGB")

    except ImportError as error:
        raise RuntimeError(
            "video-backed timeline inspection requires LeRobotDataset, torch, and PIL"
        ) from error
    paths = render_annotation_timelines(
        records,
        output_dir=config.output_dir / "timelines",
        count=config.state_bank.timeline_count,
        seed=config.seed,
        image_loader=image_loader,
    )
    report = {
        "schema_version": "libero_annotation_timeline_report_v1",
        "passed": len(paths) > 0,
        "timelines": [
            {"path": str(path), "sha256": _hash_file(path)} for path in paths
        ],
        "state_bank_manifest_sha256": _hash_file(
            config.output_dir / "state_bank" / "manifest.json"
        ),
        "states": manifest["states"],
        "image_thumbnails": True,
        "manual_review_required": True,
        "manual_review_passed": False,
        "note": "passed means artifacts are complete; a human must still inspect the sampled timelines",
    }
    write_json_atomic(config.output_dir / "timelines" / "report.json", report)
    return report


def _approve_timelines(config_path: Path) -> dict[str, object]:
    config = load_libero_study_config(config_path)
    return approve_annotation_timelines(
        config.output_dir / "timelines" / "report.json",
        state_bank_manifest=config.output_dir / "state_bank" / "manifest.json",
    )


def _plan_stages(config_path: Path, batch_size: int | None) -> dict[str, object]:
    config = load_libero_study_config(config_path)
    catalog = load_source_catalog(config)
    bank_records, _, _, _ = load_state_bank(config.output_dir / "state_bank")
    held_out_episodes = {record.lerobot_episode_index for record in bank_records}
    episodes = tuple(
        EpisodeInfo(
            episode_index=item.episode_index,
            suite=item.descriptor.suite,
            task_id=item.descriptor.task_id,
            frames=len(item.descriptor.actions),
        )
        for item in catalog.lerobot_episodes
        if item.episode_index not in held_out_episodes
    )
    subsets = plan_nested_subsets(
        episodes, fractions=config.stages.fractions, seed=config.stages.seed
    )
    try:
        from huggingface_hub import HfApi

        model_revision = str(
            HfApi().model_info(
                config.stages.base_model, revision=config.stages.base_revision
            ).sha
        )
    except Exception as error:
        raise RuntimeError("could not resolve immutable SmolVLA base-model commit") from error
    if len(model_revision) != 40:
        raise ValueError("Hugging Face did not return an immutable model commit")
    manifests = build_stage_manifests(
        episodes=episodes,
        subsets=subsets,
        output_dir=config.output_dir,
        base_model=config.stages.base_model,
        base_revision=model_revision,
        dataset_repo_id=config.sources.lerobot_repo_id,
        dataset_revision=catalog.dataset_revision,
        seed=config.stages.seed,
        epochs=config.stages.epochs,
        batch_size=config.stages.batch_size if batch_size is None else batch_size,
        code_hash=_hash_python_tree(Path(__file__).parent),
        config_hash=_hash_file(config_path),
    )
    identity_keys = {
        "schema_version", "stage", "base_model", "base_revision", "dataset_repo_id",
        "dataset_revision", "data_fraction", "episode_indices", "subset_sha256", "seed",
        "epochs", "training_steps", "checkpoint", "code_hash", "config_hash",
    }
    for stage, manifest in manifests.items():
        manifest_path = config.output_dir / "stages" / stage / "manifest.json"
        value = manifest.to_dict()
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(existing.get(key) != value.get(key) for key in identity_keys):
                raise FileExistsError(
                    f"stage plan has a different scientific binding; use a new output root: {manifest_path}"
                )
            value = existing
        else:
            write_json_atomic(manifest_path, value)
    return {
        "schema_version": "libero_smolvla_stage_plan_v1",
        "passed": True,
        "nested": all(
            set(subsets[left]).issubset(subsets[right])
            for left, right in zip(
                config.stages.fractions, config.stages.fractions[1:], strict=False
            )
        ),
        "state_bank_episode_overlap": False,
        "held_out_state_bank_episodes": len(held_out_episodes),
        "stages": {name: manifest.to_dict() for name, manifest in manifests.items()},
    }


def dispatch(args: argparse.Namespace) -> dict[str, object]:
    if args.libero_family == "audit":
        return _audit(args.config)
    if args.libero_family == "state-bank":
        if args.libero_command == "collect":
            return collect_libero_state_bank(load_libero_study_config(args.config))
        if args.libero_command == "inspect":
            return _inspect_bank(args.config)
        if args.libero_command == "visualize":
            return _visualize(args.config)
        if args.libero_command == "approve-timelines":
            return _approve_timelines(args.config)
    if args.libero_family == "stages" and args.libero_command == "plan":
        return _plan_stages(args.config, args.batch_size)
    if args.libero_family == "stages" and args.libero_command == "train":
        from .training import train_stage

        return train_stage(
            load_libero_study_config(args.config),
            stage=args.stage,
            dry_run=args.dry_run,
            resume=args.resume,
        )
    if args.libero_family == "stages" and args.libero_command == "snapshot":
        from .training import snapshot_pretrained_stage

        return snapshot_pretrained_stage(load_libero_study_config(args.config))
    if args.libero_family == "latents":
        from .latents import extract_smolvla_latents, inspect_stage_latents

        config = load_libero_study_config(args.config)
        if args.stage is None:
            raise ValueError("--stage is required for LIBERO latent commands")
        if args.libero_command == "extract":
            return extract_smolvla_latents(config, stage=args.stage)
        if args.libero_command == "inspect":
            return inspect_stage_latents(config, stage=args.stage)
    if args.libero_family == "longitudinal":
        from .longitudinal import (
            extract_longitudinal_condition,
            inspect_longitudinal_latents,
            plan_longitudinal_conditions,
        )

        config = load_libero_study_config(args.config)
        if args.libero_command == "plan":
            return plan_longitudinal_conditions(config)
        if args.libero_command == "extract":
            return extract_longitudinal_condition(
                config, condition=args.condition, batch_size=args.batch_size
            )
        if args.libero_command == "inspect":
            return inspect_longitudinal_latents(config)
        if args.libero_command in {"probes", "probe-report"}:
            from .crossfit_probes import (
                inspect_crossfit_probe_report,
                run_crossfit_probe_study,
            )

            if args.libero_command == "probes":
                return run_crossfit_probe_study(config)
            return inspect_crossfit_probe_report(config)
    if args.libero_family == "probes":
        from .probe_runner import run_probe_study
        from .probe_transitions import enrich_adjacent_stage_deltas

        config = load_libero_study_config(args.config)
        if args.libero_command == "run":
            run_probe_study(config)
        if args.libero_command in {"run", "report"}:
            return enrich_adjacent_stage_deltas(config)
    if args.libero_family == "interventions":
        from .recruitment import (
            audit_longitudinal_recruitment,
            run_longitudinal_recruitment,
        )

        config = load_libero_study_config(args.config)
        if args.libero_command == "audit":
            return audit_longitudinal_recruitment(config)
        if args.libero_command == "run":
            if not args.specificity_only and not args.dry_run:
                required = _required_intervention_batch_size(
                    config.output_dir / "protocol_v3"
                )
                if args.batch_size != required:
                    raise ValueError(
                        "action-sensitivity batch size must match the frozen Protocol-v3 "
                        f"latent runtime: expected {required}, got {args.batch_size}"
                    )
            return run_longitudinal_recruitment(
                config,
                max_states=args.max_states,
                batch_size=args.batch_size,
                specificity_only=args.specificity_only,
                dry_run=args.dry_run,
            )
    raise ValueError(
        f"{args.libero_family} {args.libero_command} is implementation-only until its prerequisite gate passes"
    )
