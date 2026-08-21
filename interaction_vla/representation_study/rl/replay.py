from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from interaction_vla.lerobot_bridge.provenance import sha256_file


REPLAY_SCHEMA = "recovery_replay_v1"
REPLAY_FAMILIES = {"recovery", "perturbation", "nominal"}


def _image(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 3
        or array.shape[-1] != 3
        or array.dtype != np.uint8
    ):
        raise ValueError(f"{name} must be an HWC uint8 RGB image")
    return array.copy()


def _vector(value: object, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite float32 vector of width {width}")
    return array


def _normalize_transition(value: Mapping[str, object]) -> dict[str, object]:
    required = {
        "transition_id",
        "case_id",
        "family",
        "task",
        "agent_rgb",
        "wrist_rgb",
        "state",
        "next_agent_rgb",
        "next_wrist_rgb",
        "next_state",
        "oracle_state",
        "next_oracle_state",
        "residual",
        "reward",
        "done",
    }
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if unknown:
            details.append("unknown=" + ",".join(sorted(unknown)))
        raise ValueError("replay transition schema differs: " + " ".join(details))
    transition_id = str(value["transition_id"])
    case_id = str(value["case_id"])
    family = str(value["family"])
    task = str(value["task"])
    if not transition_id or not case_id or not task.strip():
        raise ValueError("replay identifiers and task must be non-empty")
    if family not in REPLAY_FAMILIES:
        raise ValueError(f"unknown replay family: {family}")
    reward = float(value["reward"])
    if not np.isfinite(reward):
        raise ValueError("replay reward must be finite")
    residual = _vector(value["residual"], 7, "residual")
    if np.any(np.abs(residual) > 1.0 + 1.0e-6):
        raise ValueError("replay residual must remain within [-1, 1]")
    agent = _image(value["agent_rgb"], "agent_rgb")
    wrist = _image(value["wrist_rgb"], "wrist_rgb")
    next_agent = _image(value["next_agent_rgb"], "next_agent_rgb")
    next_wrist = _image(value["next_wrist_rgb"], "next_wrist_rgb")
    if not (
        agent.shape == wrist.shape == next_agent.shape == next_wrist.shape
    ):
        raise ValueError("replay RGB views must share shape")
    return {
        "transition_id": transition_id,
        "case_id": case_id,
        "family": family,
        "task": task,
        "agent_rgb": agent,
        "wrist_rgb": wrist,
        "state": _vector(value["state"], 10, "state"),
        "next_agent_rgb": next_agent,
        "next_wrist_rgb": next_wrist,
        "next_state": _vector(value["next_state"], 10, "next_state"),
        "oracle_state": _vector(value["oracle_state"], 36, "oracle_state"),
        "next_oracle_state": _vector(
            value["next_oracle_state"], 36, "next_oracle_state"
        ),
        "residual": residual,
        "reward": np.float32(reward),
        "done": np.bool_(value["done"]),
    }


@dataclass(frozen=True)
class ReplayBatch:
    transition_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    families: tuple[str, ...]
    tasks: tuple[str, ...]
    agent_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray
    next_agent_rgb: np.ndarray
    next_wrist_rgb: np.ndarray
    next_state: np.ndarray
    oracle_state: np.ndarray
    next_oracle_state: np.ndarray
    residual: np.ndarray
    reward: np.ndarray
    done: np.ndarray


class RecoveryReplay:
    def __init__(
        self,
        *,
        root: str | Path,
        capacity: int,
        seed: int,
        shard_size: int = 128,
    ) -> None:
        if min(capacity, shard_size) < 1 or seed < 0:
            raise ValueError("replay capacity/shard_size must be positive and seed non-negative")
        self.root = Path(root)
        self.shards_dir = self.root / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.shard_size = int(shard_size)
        self.rng = np.random.default_rng(seed)
        self._entries: list[dict[str, object]] = []
        self._pending: list[dict[str, object]] = []
        self._shard_hashes: dict[str, str] = {}
        self._task_by_case: dict[str, str] = {}
        self._seen_ids: set[str] = set()
        self._next_shard = 0

    def __len__(self) -> int:
        return len(self._entries) + len(self._pending)

    def add(self, transition: Mapping[str, object]) -> None:
        record = _normalize_transition(transition)
        transition_id = str(record["transition_id"])
        case_id = str(record["case_id"])
        task = str(record.pop("task"))
        if transition_id in self._seen_ids:
            raise ValueError(f"duplicate replay transition id: {transition_id}")
        registered = self._task_by_case.setdefault(case_id, task)
        if registered != task:
            raise ValueError(f"replay task changed within case: {case_id}")
        self._seen_ids.add(transition_id)
        self._pending.append(record)
        if len(self._pending) >= self.shard_size:
            self._flush_pending()
        while len(self) > self.capacity:
            if self._entries:
                self._entries.pop(0)
            else:
                self._pending.pop(0)

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        name = f"shard_{self._next_shard:08d}.npz"
        destination = self.shards_dir / name
        if destination.exists():
            raise FileExistsError(f"replay shard already exists: {destination}")
        temporary = destination.with_suffix(".npz.tmp")
        keys = tuple(self._pending[0])
        arrays: dict[str, np.ndarray] = {}
        for key in keys:
            values = [record[key] for record in self._pending]
            if key in {"transition_id", "case_id", "family"}:
                arrays[key] = np.asarray(values, dtype=np.str_)
            else:
                arrays[key] = np.stack(values)
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
        digest = sha256_file(destination)
        self._shard_hashes[name] = digest
        for offset, record in enumerate(self._pending):
            self._entries.append(
                {
                    "shard": name,
                    "offset": offset,
                    "transition_id": str(record["transition_id"]),
                }
            )
        self._pending.clear()
        self._next_shard += 1

    def _load_entry(
        self,
        entry: Mapping[str, object],
        cache: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, object]:
        name = str(entry["shard"])
        offset = int(entry["offset"])
        if name not in cache:
            with np.load(self.shards_dir / name, allow_pickle=False) as archive:
                cache[name] = {key: np.asarray(archive[key]) for key in archive.files}
        arrays = cache[name]
        return {key: values[offset] for key, values in arrays.items()}

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size < 1 or batch_size > len(self):
            raise ValueError("replay batch_size must lie within [1, replay size]")
        positions = np.asarray(
            self.rng.choice(len(self), size=batch_size, replace=False),
            dtype=np.int64,
        )
        loaded: dict[str, dict[str, np.ndarray]] = {}
        records: list[dict[str, object]] = []
        for position in positions:
            index = int(position)
            if index < len(self._entries):
                records.append(self._load_entry(self._entries[index], loaded))
            else:
                records.append(self._pending[index - len(self._entries)])
        case_ids = tuple(str(record["case_id"]) for record in records)
        return ReplayBatch(
            transition_ids=tuple(str(record["transition_id"]) for record in records),
            case_ids=case_ids,
            families=tuple(str(record["family"]) for record in records),
            tasks=tuple(self._task_by_case[case_id] for case_id in case_ids),
            agent_rgb=np.stack([record["agent_rgb"] for record in records]).astype(np.uint8),
            wrist_rgb=np.stack([record["wrist_rgb"] for record in records]).astype(np.uint8),
            state=np.stack([record["state"] for record in records]).astype(np.float32),
            next_agent_rgb=np.stack([record["next_agent_rgb"] for record in records]).astype(np.uint8),
            next_wrist_rgb=np.stack([record["next_wrist_rgb"] for record in records]).astype(np.uint8),
            next_state=np.stack([record["next_state"] for record in records]).astype(np.float32),
            oracle_state=np.stack([record["oracle_state"] for record in records]).astype(np.float32),
            next_oracle_state=np.stack([record["next_oracle_state"] for record in records]).astype(np.float32),
            residual=np.stack([record["residual"] for record in records]).astype(np.float32),
            reward=np.asarray([record["reward"] for record in records], dtype=np.float32),
            done=np.asarray([record["done"] for record in records], dtype=np.bool_),
        )

    def state_dict(self) -> dict[str, object]:
        self._flush_pending()
        return {
            "schema_version": REPLAY_SCHEMA,
            "capacity": self.capacity,
            "shard_size": self.shard_size,
            "next_shard": self._next_shard,
            "entries": [dict(value) for value in self._entries],
            "shard_hashes": dict(sorted(self._shard_hashes.items())),
            "task_by_case": dict(sorted(self._task_by_case.items())),
            "seen_ids": sorted(self._seen_ids),
            "rng_state": dict(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("schema_version") != REPLAY_SCHEMA:
            raise ValueError("replay state schema is incompatible")
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("replay state capacity differs")
        if int(state.get("shard_size", -1)) != self.shard_size:
            raise ValueError("replay state shard_size differs")
        hashes = state.get("shard_hashes")
        if not isinstance(hashes, Mapping):
            raise ValueError("replay shard hashes are missing")
        for name, expected in hashes.items():
            path = self.shards_dir / str(name)
            if not path.is_file() or sha256_file(path) != str(expected):
                raise ValueError(f"replay shard hash differs: {name}")
        entries = state.get("entries")
        tasks = state.get("task_by_case")
        rng_state = state.get("rng_state")
        if not isinstance(entries, list) or not isinstance(tasks, Mapping) or not isinstance(rng_state, Mapping):
            raise ValueError("replay state structure is incompatible")
        self._entries = [dict(value) for value in entries]
        self._pending = []
        self._shard_hashes = {str(key): str(value) for key, value in hashes.items()}
        self._task_by_case = {str(key): str(value) for key, value in tasks.items()}
        self._seen_ids = {str(value) for value in state.get("seen_ids", [])}
        self._next_shard = int(state.get("next_shard", -1))
        if self._next_shard < 0 or len(self._entries) > self.capacity:
            raise ValueError("replay state progress is incompatible")
        self.rng.bit_generator.state = dict(rng_state)
