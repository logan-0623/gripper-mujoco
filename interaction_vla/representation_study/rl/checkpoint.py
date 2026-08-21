from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = "interaction_residual_rl_training_v1"


def capture_rng_state() -> dict[str, object]:
    result: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.random.set_rng_state(state["torch"])  # type: ignore[arg-type]
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]


def save_training_checkpoint(
    path: str | Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    policy_state: Mapping[str, torch.Tensor] | None,
    environment_steps: int,
    update: int,
    curve: Sequence[Mapping[str, object]],
    rng_state: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    if environment_steps < 0 or update < 0:
        raise ValueError("RL checkpoint progress must be non-negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "residual_policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "base_policy": None if policy_state is None else dict(policy_state),
        "environment_steps": int(environment_steps),
        "update": int(update),
        "curve": [dict(row) for row in curve],
        "rng_state": dict(rng_state),
        "metadata": dict(metadata),
    }
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_training_checkpoint(
    path: str | Path, *, map_location: str | torch.device
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"residual RL training checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("residual RL checkpoint schema is incompatible")
    required = {
        "residual_policy", "optimizer", "base_policy", "environment_steps", "update",
        "curve", "rng_state", "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError("residual RL checkpoint fields are missing: " + ", ".join(sorted(missing)))
    return payload
