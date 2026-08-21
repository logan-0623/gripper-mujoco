from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import torch

from interaction_vla.lerobot_bridge.provenance import sha256_file

from ..state_bank.io import write_json_atomic


SNAPSHOT_SCHEMA = "recovery_rl_snapshot_v1"
SNAPSHOT_STEPS = (0, 4096, 8192, 12288, 16384, 20480)


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        step: int,
        payload: Mapping[str, object],
        binding: str,
    ) -> Path:
        if step not in SNAPSHOT_STEPS:
            raise ValueError("snapshot step is not registered")
        if not binding.strip():
            raise ValueError("snapshot binding must be non-empty")
        destination = self.root / f"step_{step:06d}"
        if (destination / "COMPLETED").is_file():
            raise FileExistsError(f"snapshot is immutable: {destination}")
        if destination.exists():
            raise FileExistsError(f"incomplete snapshot destination exists: {destination}")
        staging = Path(
            tempfile.mkdtemp(prefix=".snapshot-", dir=self.root)
        )
        try:
            payload_path = staging / "training_state.pt"
            torch.save(dict(payload), payload_path)
            with payload_path.open("rb") as handle:
                os.fsync(handle.fileno())
            write_json_atomic(
                staging / "manifest.json",
                {
                    "schema_version": SNAPSHOT_SCHEMA,
                    "environment_steps": int(step),
                    "binding": binding,
                    "payload_sha256": sha256_file(payload_path),
                },
            )
            (staging / "COMPLETED").write_text("complete\n", encoding="utf-8")
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination

    def load(
        self,
        *,
        step: int,
        expected_binding: str,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, object]:
        if step not in SNAPSHOT_STEPS:
            raise ValueError("snapshot step is not registered")
        source = self.root / f"step_{step:06d}"
        if not (source / "COMPLETED").is_file():
            raise ValueError(f"snapshot is incomplete: {source}")
        manifest = json.loads(
            (source / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
            raise ValueError("snapshot schema is incompatible")
        if int(manifest.get("environment_steps", -1)) != step:
            raise ValueError("snapshot environment step differs")
        if manifest.get("binding") != expected_binding:
            raise ValueError("snapshot binding differs")
        payload_path = source / "training_state.pt"
        if manifest.get("payload_sha256") != sha256_file(payload_path):
            raise ValueError("snapshot payload SHA-256 differs")
        payload = torch.load(
            payload_path,
            map_location=map_location,
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise ValueError("snapshot payload must be a mapping")
        return payload
