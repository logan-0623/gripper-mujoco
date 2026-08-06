from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from .franka import FRANKA_MODEL_ROOT


_PHYSICS_CONTROL_MODULES = (
    "config.py",
    "franka.py",
    "franka_controller.py",
    "contact_physics.py",
    "physics_env.py",
    "placement.py",
    "physics_expert.py",
    "physics_recovery.py",
    "physics_data.py",
    "physics_provenance.py",
    "validate_physics_expert.py",
    "graph/builder.py",
    "graph/schema.py",
)

_LEARNED_ROLLOUT_MODULES = (
    "chunked_controller.py",
    "physics_action_safety.py",
    "models/policy.py",
    "models/encoders.py",
    "graph/builder.py",
    "graph/schema.py",
)

_TRAINING_PIPELINE_MODULES = (
    "train.py",
    "sequence_training.py",
    "source_split.py",
    "models/policy.py",
    "models/encoders.py",
)


def _hash_named_files(files: Iterable[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(files):
        encoded_name = name.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def controller_source_hash() -> str:
    module_root = Path(__file__).parent
    return _hash_named_files(
        (name, module_root / name) for name in _PHYSICS_CONTROL_MODULES
    )


def physics_control_module_names() -> tuple[str, ...]:
    return _PHYSICS_CONTROL_MODULES


def learned_rollout_module_names() -> tuple[str, ...]:
    return _LEARNED_ROLLOUT_MODULES


def training_pipeline_module_names() -> tuple[str, ...]:
    return _TRAINING_PIPELINE_MODULES


def learned_rollout_source_hash() -> str:
    module_root = Path(__file__).parent
    return _hash_named_files(
        (name, module_root / name) for name in _LEARNED_ROLLOUT_MODULES
    )


def training_pipeline_source_hash() -> str:
    module_root = Path(__file__).parent
    return _hash_named_files(
        (name, module_root / name) for name in _TRAINING_PIPELINE_MODULES
    )


def scene_asset_hash() -> str:
    files = (
        (path.relative_to(FRANKA_MODEL_ROOT).as_posix(), path)
        for path in FRANKA_MODEL_ROOT.rglob("*")
        if path.is_file()
    )
    return _hash_named_files(files)


def config_file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
