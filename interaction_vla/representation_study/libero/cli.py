from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

from .collector import collect_libero_state_bank
from .config import load_libero_study_config
from .state_bank import load_state_bank
from .visualize import approve_annotation_timelines, render_annotation_timelines


def add_libero_parser(families: argparse._SubParsersAction) -> None:
    parser = families.add_parser(
        "libero", help="shared LIBERO State Bank"
    )
    commands = parser.add_subparsers(dest="libero_family", required=True)
    audit = commands.add_parser("audit", help="inspect prerequisites and immutable evidence state")
    audit.add_argument("--config", type=Path, required=True)

    bank = commands.add_parser("state-bank", help="deterministic privileged LIBERO State Bank")
    bank_commands = bank.add_subparsers(dest="libero_command", required=True)
    for name in ("collect", "inspect", "visualize", "approve-timelines"):
        command = bank_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "interaction_factors": [
            "entity", "geometry", "contact", "stable_grasp", "phase", "next_relation"
        ],
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
    raise ValueError(
        f"unknown LIBERO command: {args.libero_family} {args.libero_command}"
    )
