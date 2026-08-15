from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from interaction_vla.lerobot_bridge.act_diagnostics import (
    evaluate_checkpoint_actions,
)
from interaction_vla.lerobot_bridge.act_recovery import evaluate_recovery
from interaction_vla.lerobot_bridge.act_smoke import (
    ACTION_CODEC_VERSION,
    STATE_CODEC_VERSION,
    check_from_config,
    expected_smoke_report_contract,
    train_from_config,
)
from interaction_vla.lerobot_bridge.collector import collect_from_config
from interaction_vla.lerobot_bridge.config import load_bridge_config
from interaction_vla.lerobot_bridge.provenance import (
    fingerprint_tree,
    source_fingerprint,
)
from interaction_vla.lerobot_bridge.rollout import rollout_from_config
from interaction_vla.lerobot_bridge.teacher_schema import SCHEMA_VERSION
from interaction_vla.lerobot_bridge.validator import validate_from_config


class CLIUsageError(ValueError):
    pass


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIUsageError(message)


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = JSONArgumentParser(
        prog="python -m interaction_vla.lerobot_bridge",
        description="Local LeRobot dual-view dataset and ACT smoke workflow.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="collect a new local dataset")
    _add_config_argument(collect)

    validate = commands.add_parser("validate", help="validate a local dataset")
    _add_config_argument(validate)
    validate.add_argument("--no-replay", action="store_true")

    act_check = commands.add_parser("act-check", help="run one ACT optimizer update")
    _add_config_argument(act_check)
    act_check.add_argument("--output", type=Path)

    act_train = commands.add_parser("act-train", help="train a bounded ACT policy")
    _add_config_argument(act_train)
    act_train.add_argument("--output", type=Path)

    act_diagnose = commands.add_parser(
        "act-diagnose", help="evaluate ACT action errors by interaction phase"
    )
    _add_config_argument(act_diagnose)
    act_diagnose.add_argument("--checkpoint", required=True, type=Path)

    act_recovery = commands.add_parser(
        "act-recovery", help="run the closed-loop ACT recovery gate"
    )
    _add_config_argument(act_recovery)
    act_recovery.add_argument("--checkpoint", required=True, type=Path)

    rollout = commands.add_parser("rollout", help="roll out a local ACT checkpoint")
    _add_config_argument(rollout)
    rollout.add_argument("--checkpoint", required=True, type=Path)
    rollout.add_argument("--seed", type=int)
    rollout.add_argument("--object-count", type=int, default=2)
    rollout.add_argument(
        "--gif",
        dest="gif_path",
        type=Path,
        help="write a 10 FPS side-by-side ACT rollout GIF",
    )

    smoke = commands.add_parser("smoke", help="run the bounded local ACT smoke gate")
    _add_config_argument(smoke)
    return parser


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _all_finite(values: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())
    except (TypeError, ValueError):
        return False


def write_smoke_report(
    config_path: str | Path,
    validation: Mapping[str, object],
    act_check: Any,
    training: Mapping[str, object],
    rollout: Mapping[str, object],
) -> dict[str, object]:
    config = load_bridge_config(config_path)
    checkpoint_value = training.get("checkpoint")
    if checkpoint_value is None:
        raise ValueError("ACT training result is missing the final checkpoint path")
    checkpoint = Path(checkpoint_value)
    losses = training.get("losses", [])
    reload_error = float(_field(act_check, "reload_max_abs_error", math.inf))
    one_batch_update = (
        _all_finite(
            [
                _field(act_check, "loss", math.nan),
                _field(act_check, "gradient_norm", math.nan),
                reload_error,
            ]
        )
        and reload_error <= 1e-5
    )
    expected_steps = config.act.steps
    training_steps = int(training.get("steps", -1))
    training_ok = (
        expected_steps is not None
        and training_steps == expected_steps
        and len(losses) == expected_steps
        and _all_finite(losses)
    )
    dataset_validation = bool(validation.get("passed", False)) and int(
        validation.get("episodes", -1)
    ) == config.dataset.episodes
    finite_rollout = bool(rollout.get("finite_rollout", False))
    report: dict[str, object] = {
        "passed": bool(
            dataset_validation and one_batch_update and training_ok and finite_rollout
        ),
        "dataset_validation": dataset_validation,
        "episodes": int(validation.get("episodes", -1)),
        "one_batch_update": one_batch_update,
        "training_steps": training_steps,
        "checkpoint_reload_max_abs_error": reload_error,
        "finite_rollout": finite_rollout,
        "task_success": bool(rollout.get("task_success", False)),
        "dataset_fingerprint": fingerprint_tree(config.dataset.root),
        "checkpoint_fingerprint": fingerprint_tree(checkpoint),
        "state_codec_version": STATE_CODEC_VERSION,
        "action_codec_version": ACTION_CODEC_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint(),
    }
    report.update(expected_smoke_report_contract())
    _write_json_atomic(config.act.output_dir / "training_summary.json", training)
    _write_json_atomic(config.act.output_dir / "smoke_report.json", report)
    return report


def _dispatch(args: argparse.Namespace) -> Any:
    if args.command == "collect":
        return collect_from_config(args.config)
    if args.command == "validate":
        return validate_from_config(args.config, no_replay=args.no_replay)
    if args.command == "act-check":
        return check_from_config(args.config, output=args.output)
    if args.command == "act-train":
        return train_from_config(args.config, output=args.output)
    if args.command == "act-diagnose":
        return evaluate_checkpoint_actions(args.config, args.checkpoint)
    if args.command == "act-recovery":
        return evaluate_recovery(args.config, args.checkpoint)
    if args.command == "rollout":
        return rollout_from_config(
            args.config,
            args.checkpoint,
            seed=args.seed,
            object_count=args.object_count,
            gif_path=args.gif_path,
        )
    if args.command == "smoke":
        validation = validate_from_config(args.config)
        act_check = check_from_config(args.config)
        training = train_from_config(args.config)
        checkpoint = training.get("checkpoint")
        if checkpoint is None:
            raise ValueError("ACT training result is missing the final checkpoint path")
        rollout = rollout_from_config(args.config, checkpoint)
        return write_smoke_report(
            args.config,
            validation,
            act_check,
            training,
            rollout,
        )
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> None:
    try:
        args = build_parser().parse_args(argv)
        result = _dispatch(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(2 if isinstance(error, CLIUsageError) else 1) from error
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
