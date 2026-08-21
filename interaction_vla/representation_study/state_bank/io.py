from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .schema import StateBankManifest, StateBankRecord, StateBankSplit


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: object) -> None:
    write_bytes_atomic(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def encode_records(records: Iterable[StateBankRecord]) -> bytes:
    return b"".join(
        (json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for record in records
    )


def load_records(path: str | Path) -> tuple[StateBankRecord, ...]:
    source = Path(path)
    return tuple(
        StateBankRecord.from_dict(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def load_split(path: str | Path) -> StateBankSplit:
    return StateBankSplit.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_manifest(path: str | Path) -> StateBankManifest:
    return StateBankManifest.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )

