from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pipeline import evaluate_from_config, inspect_from_config, train_from_config


class CLIUsageError(ValueError):
    pass


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIUsageError(message)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = JSONArgumentParser(
        prog="python -m interaction_vla.graph_pretrain",
        description="ReflectVLM semantic interaction-graph pretraining.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="validate labels and leakage-safe grouped splits"
    )
    _add_config(inspect)
    train = commands.add_parser("train", help="train the semantic graph estimator")
    _add_config(train)
    evaluate = commands.add_parser(
        "evaluate", help="evaluate a compatible graph checkpoint"
    )
    _add_config(evaluate)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument(
        "--partition", choices=("validation", "test"), default="test"
    )
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


def _dispatch(args: argparse.Namespace) -> Any:
    if args.command == "inspect":
        return inspect_from_config(args.config)
    if args.command == "train":
        return train_from_config(args.config)
    if args.command == "evaluate":
        return evaluate_from_config(
            args.config, args.checkpoint, partition=args.partition
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
