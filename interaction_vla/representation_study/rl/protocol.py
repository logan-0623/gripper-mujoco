from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..state_bank.io import write_json_atomic


GATE_SCHEMA = "recovery_rl_gate_v2"


def write_gate_atomic(path: str | Path, value: Mapping[str, object]) -> Path:
    destination = Path(path)
    encoded = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") == encoded:
            return destination
        raise FileExistsError(f"gate is immutable: {destination}")
    write_json_atomic(destination, dict(value))
    return destination


def require_passing_gate(
    path: str | Path,
    *,
    expected_gate: str | None = None,
    expected_binding: str | None = None,
) -> dict[str, object]:
    source = Path(path)
    label = expected_gate or source.stem
    if not source.is_file():
        raise FileNotFoundError(f"required {label} gate not found: {source}")
    loaded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} gate must be a mapping")
    if loaded.get("schema_version") != GATE_SCHEMA:
        raise ValueError(f"{label} gate schema is incompatible")
    if expected_gate is not None and loaded.get("gate") != expected_gate:
        raise ValueError(f"{label} gate identity is incompatible")
    if loaded.get("passed") is not True:
        reasons = loaded.get("reasons", [])
        raise ValueError(f"{label} gate did not pass: {reasons}")
    if expected_binding is not None:
        inputs = loaded.get("inputs")
        if not isinstance(inputs, Mapping) or inputs.get("binding") != expected_binding:
            raise ValueError(f"{label} gate binding is incompatible")
    return loaded


def _expected_binding(config: Any) -> str | None:
    if not hasattr(config, "config_path"):
        return None
    from .foundation import foundation_binding

    return foundation_binding(config)


def calibrate_distribution(config: Any) -> dict[str, object]:
    from .foundation import calibrate_distribution as implementation

    return implementation(config)


def run_algorithm_screen(config: Any, *, resume: bool) -> dict[str, object]:
    from .foundation import run_algorithm_screen as implementation

    return implementation(config, resume=resume)


def build_oracle_gate(config: Any) -> dict[str, object]:
    from .foundation import build_oracle_gate as implementation

    return implementation(config)


def run_anchor_screen(config: Any, *, resume: bool) -> dict[str, object]:
    from .foundation import run_anchor_screen as implementation

    return implementation(config, resume=resume)


def run_recovery_command(
    config: Any,
    command: str,
    *,
    resume: bool,
) -> dict[str, object]:
    if command == "calibrate":
        return calibrate_distribution(config)
    gates = Path(config.output_dir) / "gates"
    binding = _expected_binding(config)
    require_passing_gate(
        gates / "distribution.json",
        expected_gate="distribution",
        expected_binding=binding,
    )
    if command == "screen":
        return run_algorithm_screen(config, resume=resume)
    require_passing_gate(
        gates / "backend.json",
        expected_gate="backend",
        expected_binding=binding,
    )
    if command == "oracle-gate":
        return build_oracle_gate(config)
    require_passing_gate(
        gates / "oracle.json",
        expected_gate="oracle",
        expected_binding=binding,
    )
    if command == "anchor-screen":
        return run_anchor_screen(config, resume=resume)
    raise ValueError(f"unknown recovery RL command: {command}")
